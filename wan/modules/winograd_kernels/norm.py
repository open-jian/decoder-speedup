"""RMS_norm + optional SiLU, preserving the original BF16 rounding steps."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _norm(X,G,Y,C:tl.constexpr,P:tl.constexpr,EPS:tl.constexpr,SCALE:tl.constexpr,
          SILU:tl.constexpr,INPUT_BF16:tl.constexpr,ROUND_SCALE:tl.constexpr,
          BC:tl.constexpr,BP:tl.constexpr):
    p=tl.program_id(0)*BP+tl.arange(0,BP)
    c=tl.arange(0,BC)
    x=tl.load(X+c[:,None]*P+p[None,:],(c[:,None]<C)&(p[None,:]<P),0).to(tl.float32)
    norm=libdevice.sqrt_rn(tl.sum(x*x,axis=0))
    if INPUT_BF16:
        norm=norm.to(tl.bfloat16).to(tl.float32)
    norm=tl.maximum(norm,EPS)
    v=tl.div_rn(x,norm[None,:])
    if INPUT_BF16:
        v=v.to(tl.bfloat16).to(tl.float32)
    scale=tl.full((),SCALE,tl.float32)
    if ROUND_SCALE:
        scale=scale.to(tl.bfloat16).to(tl.float32)
    v=v*scale
    if INPUT_BF16:
        v=v.to(tl.bfloat16).to(tl.float32)
    gamma=tl.load(G+c,c<C,0).to(tl.float32)
    v=v*gamma[:,None]
    if SILU:
        v=tl.div_rn(v,1.+libdevice.exp(-v))
    tl.store(Y+c[:,None]*P+p[None,:],v,(c[:,None]<C)&(p[None,:]<P))


def fused_norm(x,gamma,silu=True,out_bf16=True,round_scale=False,bp=32):
    assert x.shape[0]==1 and x.is_contiguous()
    c=x.shape[1];p=x.numel()//c
    y=torch.empty(x.shape,device=x.device,dtype=torch.bfloat16 if out_bf16 else torch.float32)
    with torch.cuda.device(x.device):
        _norm[(triton.cdiv(p,bp),)](x,gamma,y,c,p,1e-12,c**.5,silu,
                                  x.dtype==torch.bfloat16,round_scale,triton.next_power_of_2(c),bp,
                                  num_warps=8 if c*bp>=16384 else 4,enable_fp_fusion=False)
    return y


class FusedNorm(torch.nn.Module):
    def __init__(self,original,silu):
        super().__init__();self.original=original;self.silu=silu;self.enabled=True
        self._original_forward=original.forward
        self.train(original.training)
    def forward(self,x):
        if self.enabled and (self.original._forward_hooks or self.original._forward_pre_hooks
                             or self.original._backward_hooks):
            raise RuntimeError('Fused normalization does not support modified module hooks; disable the adapter.')
        if (self.enabled and not self.original.training and not torch.is_grad_enabled()
            and torch.is_autocast_enabled('cuda') and torch.get_autocast_dtype('cuda')==torch.bfloat16
            and self.original.channel_first and x.ndim>=3
            and self.original.gamma.numel()==x.shape[1] and self.original.gamma.shape[0]==x.shape[1]
            and self.original.gamma.is_contiguous() and self.original.gamma.device==x.device
            and self.original.scale==x.shape[1]**.5
            and self.original.gamma.dtype==torch.float32 and x.is_cuda and x.shape[0]==1 and x.is_contiguous() and
            x.dtype==torch.bfloat16 and isinstance(self.original.bias,float) and self.original.bias==0.):
            return fused_norm(x,self.original.gamma,silu=self.silu)
        y=self._original_forward(x)
        return torch.nn.functional.silu(y) if self.silu else y

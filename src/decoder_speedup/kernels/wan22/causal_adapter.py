"""Inference adapter fusing causal cache, padding and BF16 casting into input transform."""

import torch
import triton

from .. import winograd_3d_conv as w3
from .causal_kernel import input_causal

from .fused_height import height_gemm, height_output


def make_forward(module, gemm_kind='height', gemm_config=None,
                 workspace_factor=None, input_block=128, input_warps=4,
                 cache_bias=False):
    """Return new CausalConv3d.forward and counters; leave existing methods unchanged until installed.

    Fast path only covers BF16 inference, causal 3x3x3 with one-cell spatial
    and two-cell temporal padding. All other conditions use the existing path.
    """
    if gemm_kind != 'height':
        raise ValueError(f'Unsupported GEMM kind: {gemm_kind}')
    if workspace_factor is not None and workspace_factor<=0:
        raise ValueError('workspace_factor must be positive.')
    old_forward=module.CausalConv3d.forward
    counters={'fast_calls':0,'fallback_calls':0}
    def forward(self,x,cache_x=None):
        if not (self._conv_backend=='winograd_3d' and not self.training
                and x.is_cuda and x.ndim==5 and min(x.shape[2:])>0
                and self.weight.device==x.device and (self.bias is None or self.bias.device==x.device)
                and x.shape[1]==self.weight.shape[1] and self.kernel_size==(3,3,3)
                and self.stride==(1,1,1) and self.dilation==(1,1,1)
                and self.groups==1 and self._padding==(1,1,1,1,2,0)
                and w3._effective_dtype(x,self.weight,self.bias)==torch.bfloat16):
            counters['fallback_calls']+=1
            return old_forward(self,x,cache_x)
        if torch.is_grad_enabled() and any(t.requires_grad for t in (x,self.weight,self.bias) if t is not None):
            return old_forward(self,x,cache_x)
        counters['fast_calls']+=1
        if cache_x is not None:
            cache_x=cache_x.to(x.device)
            assert cache_x.shape[0:2]==x.shape[0:2] and cache_x.shape[3:]==x.shape[3:]
        ct=0 if cache_x is None else cache_x.shape[2]
        cache_tensor=x if cache_x is None else cache_x
        n,c,t,h,w=x.shape
        k=self.weight.shape[0]
        dtype=torch.bfloat16
        with torch.cuda.device(x.device),torch.autocast(device_type='cuda',enabled=False):
            key=w3._cache_key(self.weight,dtype,True)
            key=(*key,'winograd_3d') if key is not None else None
            cached=self._winograd_weight_cache
            if key is not None and cached is not None and cached[0]==key:
                u=cached[1]
            else:
                u=w3._transform_weight_3d(self.weight,dtype,True)
                self._winograd_weight_cache=(key,u) if key is not None else None
            if self.bias is None:
                bias=None
            elif cache_bias:
                bkey=w3._cache_key(self.bias,dtype,True)
                old_bias=getattr(self,'_experimental_bf16_bias_cache',None)
                if bkey is not None and old_bias is not None and old_bias[0]==bkey:
                    bias=old_bias[1]
                else:
                    bias=self.bias.to(dtype)
                    self._experimental_bf16_bias_cache=(bkey,bias) if bkey is not None else None
            else:
                bias=self.bias.to(dtype)
            layout=torch.channels_last_3d if x.is_contiguous(memory_format=torch.channels_last_3d) else torch.contiguous_format
            y=torch.empty((n,k,t,h,w),device=x.device,dtype=dtype,memory_format=layout)
            nt,nh,nw=(t+1)//2,(h+1)//2,(w+1)//2
            total=n*nt*nh*nw
            chunk=max(1,(128*1024*1024)//(64*(c*2+k*4)))
            if chunk>=128: chunk=chunk//128*128
            factor=workspace_factor
            if factor is None: factor=2 if t>1 and c>=1024 else 1
            chunk=max(1,int(chunk*factor))
            config=gemm_config
            if config is None:
                config=(32,64,64,4,2) if t==1 else (64,128,32,8,3)
            bm,bn,bk,warps,stages=config
            for start in range(0,total,chunk):
                p=min(chunk,total-start)
                q=48 if t==1 else 64
                v=torch.empty((q,c,p),device=x.device,dtype=dtype)
                mq=24 if t==1 else 32
                m=torch.empty((mq,k,p),device=x.device,dtype=torch.float32)
                input_causal[(triton.cdiv(c*p,input_block),)](
                    x,cache_tensor,v,c,t,h,w,ct,nt,nh,nw,p,start,*x.stride(),*cache_tensor.stride(),
                    BLOCK=input_block,num_warps=input_warps)
                height_gemm[(triton.cdiv(k,bm),triton.cdiv(p,bn),12 if t==1 else 16)](
                    u,v,m,k,c,p,BM=bm,BN=bn,BK=bk,ONE=t==1,num_warps=warps,num_stages=stages,
                    enable_fp_fusion=False)
                output=height_output
                output_options={'enable_fp_fusion':False}
                output[(triton.cdiv(k*p,128),)](
                    m,bias if bias is not None else y,y,k,p,start,t,h,w,nt,nh,nw,bias is not None,
                    *y.stride(),BLOCK=128,**output_options)
        return y
    return forward,counters

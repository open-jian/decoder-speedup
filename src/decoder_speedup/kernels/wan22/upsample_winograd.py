"""Nearest 2x upsampling + 3x3 convolution, reduced 9-coefficient F(2,3).

Each 4x4 Winograd input tile of a nearest-upsampled image has identical middle
rows and columns. B^T's third row computes their difference and is exactly zero.
Only transform coordinates {0,1,3} x {0,1,3} need GEMM (9 instead of 16).
No enlarged activation is allocated. This changes convolution rounding versus
cuDNN, so whole-decoder quality must be assessed before adopting it.
"""
import torch
from torch import nn
import triton
import triton.language as tl

from .. import winograd_3d_triton as k3
from .. import winograd_3d_conv as w3


@triton.jit
def input_up2(X,V,C:tl.constexpr,H:tl.constexpr,W:tl.constexpr,
              P:tl.constexpr,START,S0:tl.constexpr,S1:tl.constexpr,
              S2:tl.constexpr,S3:tl.constexpr,BLOCK:tl.constexpr):
    z=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    c,p=z//P,z%P
    gp=START+p
    nn=gp//(H*W)
    yy=(gp//W)%H
    xx=gp%W
    valid=z<C*P
    base=nn*S0+c*S1+yy*S2+xx*S3
    d00=tl.load(X+base-S2-S3,valid&(yy>0)&(xx>0),0).to(tl.bfloat16).to(tl.float32)
    d01=tl.load(X+base-S2,valid&(yy>0),0).to(tl.bfloat16).to(tl.float32)
    d02=tl.load(X+base-S2+S3,valid&(yy>0)&(xx+1<W),0).to(tl.bfloat16).to(tl.float32)
    d10=tl.load(X+base-S3,valid&(xx>0),0).to(tl.bfloat16).to(tl.float32)
    d11=tl.load(X+base,valid,0).to(tl.bfloat16).to(tl.float32)
    d12=tl.load(X+base+S3,valid&(xx+1<W),0).to(tl.bfloat16).to(tl.float32)
    d20=tl.load(X+base+S2-S3,valid&(yy+1<H)&(xx>0),0).to(tl.bfloat16).to(tl.float32)
    d21=tl.load(X+base+S2,valid&(yy+1<H),0).to(tl.bfloat16).to(tl.float32)
    d22=tl.load(X+base+S2+S3,valid&(yy+1<H)&(xx+1<W),0).to(tl.bfloat16).to(tl.float32)
    a00,a01,a03=d00-d01,d01+d01,d01-d02
    a10,a11,a13=d10-d11,d11+d11,d11-d12
    a30,a31,a33=d20-d21,d21+d21,d21-d22
    span=C*P
    tl.store(V+z+0*span,a00-a10,valid)
    tl.store(V+z+1*span,a01-a11,valid)
    tl.store(V+z+2*span,a03-a13,valid)
    tl.store(V+z+3*span,a10+a10,valid)
    tl.store(V+z+4*span,a11+a11,valid)
    tl.store(V+z+5*span,a13+a13,valid)
    tl.store(V+z+6*span,a10-a30,valid)
    tl.store(V+z+7*span,a11-a31,valid)
    tl.store(V+z+8*span,a13-a33,valid)


@triton.jit
def output_up2(M,BIAS,Y,K:tl.constexpr,H:tl.constexpr,W:tl.constexpr,
               P:tl.constexpr,START,HAS_BIAS:tl.constexpr,BLOCK:tl.constexpr):
    z=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    k,p=z//P,z%P
    gp=START+p
    nn=gp//(H*W)
    yy=(gp//W)%H
    xx=gp%W
    valid=z<K*P
    span=K*P
    m00=tl.load(M+z+0*span,valid,0)
    m01=tl.load(M+z+1*span,valid,0)
    m03=tl.load(M+z+2*span,valid,0)
    m10=tl.load(M+z+3*span,valid,0)
    m11=tl.load(M+z+4*span,valid,0)
    m13=tl.load(M+z+5*span,valid,0)
    m30=tl.load(M+z+6*span,valid,0)
    m31=tl.load(M+z+7*span,valid,0)
    m33=tl.load(M+z+8*span,valid,0)
    bias=tl.full((BLOCK,),0,tl.float32)
    if HAS_BIAS: bias=tl.load(BIAS+k,valid,0).to(tl.float32)
    a0,a1,a3=m00+m10,m01+m11,m03+m13
    b0,b1,b3=m10-m30,m11-m31,m13-m33
    base=(nn*K+k)*(4*H*W)+2*yy*(2*W)+2*xx
    tl.store(Y+base,a0+a1+bias,valid)
    tl.store(Y+base+1,a1-a3+bias,valid)
    tl.store(Y+base+2*W,b0+b1+bias,valid)
    tl.store(Y+base+2*W+1,b1-b3+bias,valid)


class UpsampleWinograd(nn.Module):
    def __init__(self,original,config=None,workspace_bytes=128*1024*1024,domain_dtype=torch.bfloat16):
        super().__init__()
        self.original=original
        self._original_forward=original.forward
        conv=original[1]
        up=original[0]
        scale=getattr(up,'scale_factor',None)
        scale=(scale,scale) if isinstance(scale,(int,float)) else scale
        self.supported=(isinstance(conv,nn.Conv2d) and conv.kernel_size==(3,3)
                        and conv.stride==(1,1) and conv.padding==(1,1)
                        and conv.dilation==(1,1) and conv.groups==1 and conv.padding_mode=='zeros'
                        and scale is not None and tuple(scale)==(2.,2.)
                        and getattr(up,'mode',None) in ('nearest','nearest-exact'))
        self.config=config
        self.workspace_bytes=workspace_bytes
        assert domain_dtype in (torch.bfloat16,torch.float16)
        self.domain_dtype=domain_dtype
        self._u_cache=None
        self.enabled=True

    def forward(self,x):
        conv=self.original[1]
        hooked=any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks
                   for m in self.original.modules())
        if not(self.enabled and self.supported and not hooked and x.is_cuda and x.ndim==4 and not self.original.training and
               w3._effective_dtype(x,conv.weight,conv.bias)==torch.bfloat16):
            return self._original_forward(x)
        if torch.is_grad_enabled() and any(t.requires_grad for t in (x,conv.weight,conv.bias) if t is not None):
            return self._original_forward(x)
        n,c,h,w=x.shape
        k=conv.out_channels
        with torch.cuda.device(x.device),torch.autocast('cuda',enabled=False):
            key=w3._cache_key(conv.weight,torch.bfloat16,True)
            if key is not None: key=(*key,self.domain_dtype)
            if key is not None and conv.bias is not None:
                if conv.bias.is_inference():key=None
                else:key=(*key,id(conv.bias),conv.bias._version,conv.bias.data_ptr())
            if self._u_cache is not None and self._u_cache[0]==key:
                u,bias=self._u_cache[1:]
            else:
                u=conv.weight.to(torch.bfloat16).float()
                for axis in (3,2):u=w3._g_axis(u,axis)
                ix=torch.tensor([0,1,3],device=x.device)
                u=u.index_select(2,ix).index_select(3,ix).permute(2,3,0,1).reshape(9,k,c).contiguous().to(self.domain_dtype)
                bias=conv.bias.to(torch.bfloat16) if conv.bias is not None else None
                self._u_cache=(key,u,bias) if key is not None else None
            out=torch.empty((n,k,2*h,2*w),device=x.device,dtype=torch.bfloat16)
            total=n*h*w
            chunk=max(128,self.workspace_bytes//(9*(c*2+k*4))//128*128)
            config=self.config
            if config is None:
                config=(64,128,64,4,3) if c>=1024 and total>500 else (128,64,32,4,3)
            bm,bn,bk,warps,stages=config
            for start in range(0,total,chunk):
                p=min(chunk,total-start)
                v=torch.empty((9,c,p),device=x.device,dtype=self.domain_dtype)
                m=torch.empty((9,k,p),device=x.device,dtype=torch.float32)
                input_up2[(triton.cdiv(c*p,128),)](x,v,c,h,w,p,start,*x.stride(),BLOCK=128)
                k3._gemm[(triton.cdiv(k,bm),triton.cdiv(p,bn),9)](
                    u,v,m,k,c,p,BM=bm,BN=bn,BK=bk,num_warps=warps,num_stages=stages)
                output_up2[(triton.cdiv(k*p,128),)](m,bias if bias is not None else out,out,k,h,w,p,start,bias is not None,BLOCK=128)
        return out

"""One-pass DupUp3D shortcut indexing and residual addition."""
import torch
import triton
import triton.language as tl


@triton.jit
def _skip_add(X,MAIN,Y,N:tl.constexpr,K:tl.constexpr,OT:tl.constexpr,
              OH:tl.constexpr,OW:tl.constexpr,
              FT:tl.constexpr,FS:tl.constexpr,REPEATS:tl.constexpr,CROP:tl.constexpr,
              X0:tl.constexpr,X1:tl.constexpr,X2:tl.constexpr,X3:tl.constexpr,X4:tl.constexpr,
              M0:tl.constexpr,M1:tl.constexpr,M2:tl.constexpr,M3:tl.constexpr,M4:tl.constexpr,
              BLOCK:tl.constexpr):
    idx=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=idx<N*K*OT*OH*OW
    ow=idx%OW
    oh=(idx//OW)%OH
    ot=(idx//(OW*OH))%OT
    kk=(idx//(OW*OH*OT))%K
    nn=idx//(OW*OH*OT*K)
    expanded_t=ot+CROP
    rt,rh,rw=expanded_t%FT,oh%FS,ow%FS
    tt,hh,ww=expanded_t//FT,oh//FS,ow//FS
    cc=(kk*(FT*FS*FS)+(rt*FS+rh)*FS+rw)//REPEATS
    a=tl.load(X+nn*X0+cc*X1+tt*X2+hh*X3+ww*X4,valid,0).to(tl.float32)
    b=tl.load(MAIN+nn*M0+kk*M1+ot*M2+oh*M3+ow*M4,valid,0).to(tl.float32)
    tl.store(Y+idx,a+b,valid)


def shortcut_add(x,main,shortcut,first_chunk=False):
    assert x.is_cuda and main.is_cuda
    supported=(torch.float16,torch.bfloat16,torch.float32)
    if x.dtype not in supported or main.dtype not in supported:
        return shortcut(x,first_chunk)+main
    dtype=torch.promote_types(x.dtype,main.dtype)
    y=torch.empty(main.shape,device=main.device,dtype=dtype)
    ft,fs,repeats=shortcut.factor_t,shortcut.factor_s,shortcut.repeats
    crop=ft-1 if first_chunk else 0
    expected=(x.shape[0],shortcut.out_channels,x.shape[2]*ft-crop,x.shape[3]*fs,x.shape[4]*fs)
    assert tuple(main.shape)==expected,(main.shape,expected)
    with torch.cuda.device(x.device):
        _skip_add[(triton.cdiv(y.numel(),256),)](
            x,main,y,*y.shape,ft,fs,repeats,crop,*x.stride(),*main.stride(),BLOCK=256)
    return y

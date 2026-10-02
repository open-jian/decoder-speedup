"""Fuse the original first (height) inverse transform into the GEMM epilogue.
Unlike temporal-first fusion, this preserves inverse addition order exactly.
"""
import torch
import triton
import triton.language as tl
@triton.jit
def height_gemm(U,V,M,K:tl.constexpr,R:tl.constexpr,P:tl.constexpr,
              BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,ONE:tl.constexpr):
    kk=tl.program_id(0)*BM+tl.arange(0,BM)
    pp=tl.program_id(1)*BN+tl.arange(0,BN)
    rr=tl.arange(0,BK)
    qt=tl.program_id(2)//4
    qw=tl.program_id(2)%4
    q=qt*16+qw
    a0=tl.full((BM,BN),0,tl.float32)
    a1=tl.full((BM,BN),0,tl.float32)
    a2=tl.full((BM,BN),0,tl.float32)
    a3=tl.full((BM,BN),0,tl.float32)
    for i in range(tl.cdiv(R,BK)):
        r=i*BK+rr
        up=kk[:,None]*R+r[None,:]
        vp=r[:,None]*P+pp[None,:]
        um=(kk[:,None]<K)&(r[None,:]<R)
        vm=(r[:,None]<R)&(pp[None,:]<P)
        u0=tl.load(U+q*K*R+up,um,0)
        v0=tl.load(V+q*R*P+vp,vm,0)
        a0=tl.dot(u0,v0,a0,input_precision='ieee')
        u1=tl.load(U+(q+4)*K*R+up,um,0)
        v1=tl.load(V+(q+4)*R*P+vp,vm,0)
        a1=tl.dot(u1,v1,a1,input_precision='ieee')
        u2=tl.load(U+(q+8)*K*R+up,um,0)
        v2=tl.load(V+(q+8)*R*P+vp,vm,0)
        a2=tl.dot(u2,v2,a2,input_precision='ieee')
        u3=tl.load(U+(q+12)*K*R+up,um,0)
        v3=tl.load(V+(q+12)*R*P+vp,vm,0)
        a3=tl.dot(u3,v3,a3,input_precision='ieee')
    op=kk[:,None]*P+pp[None,:]
    mask=(kk[:,None]<K)&(pp[None,:]<P)
    mq=qt*8+qw
    tl.store(M+mq*K*P+op,a0+a1+a2,mask)
    tl.store(M+(mq+4)*K*P+op,a1-a2-a3,mask)

@triton.jit
def _width_inverse(M,z,span,valid,Q:tl.constexpr):
    z=z+Q*8*span
    a=tl.load(M+z,valid,0)
    b=tl.load(M+z+span,valid,0)
    c=tl.load(M+z+2*span,valid,0)
    d=tl.load(M+z+3*span,valid,0)
    e=tl.load(M+z+4*span,valid,0)
    f=tl.load(M+z+5*span,valid,0)
    g=tl.load(M+z+6*span,valid,0)
    h=tl.load(M+z+7*span,valid,0)
    return a+b+c,b-c-d,e+f+g,f-g-h

@triton.jit
def height_output(M,BIAS,Y,K:tl.constexpr,P:tl.constexpr,START,
                 OT:tl.constexpr,OH:tl.constexpr,OW:tl.constexpr,
                 NT:tl.constexpr,NH:tl.constexpr,NW:tl.constexpr,
                 HAS_BIAS:tl.constexpr,S0:tl.constexpr,S1:tl.constexpr,
                 S2:tl.constexpr,S3:tl.constexpr,S4:tl.constexpr,BLOCK:tl.constexpr):
    z=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    k,p=z//P,z%P
    valid=z<K*P
    span=K*P
    bias=tl.full((BLOCK,),0,tl.float32)
    if HAS_BIAS: bias=tl.load(BIAS+k,valid,0).to(tl.float32)
    gp=START+p
    nn,tt=gp//(NT*NH*NW),2*((gp//(NH*NW))%NT)
    yy,xx=2*((gp//NW)%NH),2*(gp%NW)
    base=nn*S0+k*S1+tt*S2+yy*S3+xx*S4
    a00,a01,a10,a11=_width_inverse(M,z,span,valid,0)
    b00,b01,b10,b11=_width_inverse(M,z,span,valid,1)
    c00,c01,c10,c11=_width_inverse(M,z,span,valid,2)
    tl.store(Y+base,a00+b00+c00+bias,valid&(tt<OT)&(yy<OH)&(xx<OW))
    tl.store(Y+base+S4,a01+b01+c01+bias,valid&(tt<OT)&(yy<OH)&(xx+1<OW))
    tl.store(Y+base+S3,a10+b10+c10+bias,valid&(tt<OT)&(yy+1<OH)&(xx<OW))
    tl.store(Y+base+S3+S4,a11+b11+c11+bias,valid&(tt<OT)&(yy+1<OH)&(xx+1<OW))
    if OT>1:
        d00,d01,d10,d11=_width_inverse(M,z,span,valid,3)
        tl.store(Y+base+S2,b00-c00-d00+bias,valid&(tt+1<OT)&(yy<OH)&(xx<OW))
        tl.store(Y+base+S2+S4,b01-c01-d01+bias,valid&(tt+1<OT)&(yy<OH)&(xx+1<OW))
        tl.store(Y+base+S2+S3,b10-c10-d10+bias,valid&(tt+1<OT)&(yy+1<OH)&(xx<OW))
        tl.store(Y+base+S2+S3+S4,b11-c11-d11+bias,valid&(tt+1<OT)&(yy+1<OH)&(xx+1<OW))

def make_runner(k3,config=(64,128,32,8,3),workspace_factor=1):
    def run_full_winograd(x,u,bias,out,chunk):
        n,c,t,h,w=x.shape
        _,k,ot,oh,ow=out.shape
        nt,nh,nw=(ot+1)//2,(oh+1)//2,(ow+1)//2
        total=n*nt*nh*nw
        bm,bn,bk,warps,stages=config
        for start in range(0,total,chunk*workspace_factor):
            p=min(chunk*workspace_factor,total-start)
            v=torch.empty((64,c,p),device=x.device,dtype=x.dtype)
            m=torch.empty((24 if ot==1 else 32,k,p),device=x.device,dtype=torch.float32)
            k3._input_transform_3d[(triton.cdiv(c*p,128),)](x,v,c,t,h,w,nt,nh,nw,p,start,*x.stride(),BLOCK=128)
            height_gemm[(triton.cdiv(k,bm),triton.cdiv(p,bn),12 if ot==1 else 16)](u,v,m,k,c,p,BM=bm,BN=bn,BK=bk,ONE=ot==1,num_warps=warps,num_stages=stages,enable_fp_fusion=False)
            height_output[(triton.cdiv(k*p,128),)](m,bias if bias is not None else out,out,k,p,start,ot,oh,ow,nt,nh,nw,bias is not None,*out.stride(),BLOCK=128,enable_fp_fusion=False)
    return run_full_winograd

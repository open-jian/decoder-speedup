"""Mixed F43 GEMM with the temporal inverse transform in its epilogue.

FP16 operands and FP32 accumulation produce 72 spatial coefficients per tile,
or 36 for a single temporal output.
"""
import triton
import triton.language as tl
from .mixed_f43 import spatial_inverse


@triton.jit
def time_gemm(U,V,M,K:tl.constexpr,R:tl.constexpr,P:tl.constexpr,
              BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,ONE:tl.constexpr):
    kk=tl.program_id(0)*BM+tl.arange(0,BM)
    pp=tl.program_id(1)*BN+tl.arange(0,BN)
    rr=tl.arange(0,BK)
    q=tl.program_id(2)
    a0=tl.full((BM,BN),0,tl.float32)
    a1=tl.full((BM,BN),0,tl.float32)
    a2=tl.full((BM,BN),0,tl.float32)
    if not ONE:
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
        u1=tl.load(U+(q+36)*K*R+up,um,0)
        v1=tl.load(V+(q+36)*R*P+vp,vm,0)
        a1=tl.dot(u1,v1,a1,input_precision='ieee')
        u2=tl.load(U+(q+72)*K*R+up,um,0)
        v2=tl.load(V+(q+72)*R*P+vp,vm,0)
        a2=tl.dot(u2,v2,a2,input_precision='ieee')
        if not ONE:
            u3=tl.load(U+(q+108)*K*R+up,um,0)
            v3=tl.load(V+(q+108)*R*P+vp,vm,0)
            a3=tl.dot(u3,v3,a3,input_precision='ieee')
    op=kk[:,None]*P+pp[None,:]
    mask=(kk[:,None]<K)&(pp[None,:]<P)
    tl.store(M+q*K*P+op,a0+a1+a2,mask)
    if not ONE:
        tl.store(M+(q+36)*K*P+op,a1-a2-a3,mask)


@triton.jit
def output_transform(M,BIAS,Y,K:tl.constexpr,P:tl.constexpr,START,
                     OT:tl.constexpr,OH:tl.constexpr,OW:tl.constexpr,
                     NT:tl.constexpr,NH:tl.constexpr,NW:tl.constexpr,
                     HAS_BIAS:tl.constexpr,S0:tl.constexpr,S1:tl.constexpr,
                     S2:tl.constexpr,S3:tl.constexpr,S4:tl.constexpr,BLOCK:tl.constexpr):
    z=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    k,p=z//P,z%P
    valid=z<K*P
    span=K*P
    gp=START+p
    nn,tt=gp//(NT*NH*NW),2*((gp//(NH*NW))%NT)
    yy,xx=4*((gp//NW)%NH),4*(gp%NW)
    base=nn*S0+k*S1+tt*S2+yy*S3+xx*S4
    bias=tl.full((BLOCK,),0,tl.float32)
    if HAS_BIAS:
        bias=tl.load(BIAS+k,valid,0).to(tl.float32)
    for qi in tl.static_range(1 if OT==1 else 2):
        y00,y01,y02,y03,y10,y11,y12,y13,y20,y21,y22,y23,y30,y31,y32,y33=spatial_inverse(M,z,span,valid,qi)
        tl.store(Y+base+qi*S2+0*S3+0*S4,y00+bias,valid & (tt+qi<OT) & (yy+0<OH) & (xx+0<OW))
        tl.store(Y+base+qi*S2+0*S3+1*S4,y01+bias,valid & (tt+qi<OT) & (yy+0<OH) & (xx+1<OW))
        tl.store(Y+base+qi*S2+0*S3+2*S4,y02+bias,valid & (tt+qi<OT) & (yy+0<OH) & (xx+2<OW))
        tl.store(Y+base+qi*S2+0*S3+3*S4,y03+bias,valid & (tt+qi<OT) & (yy+0<OH) & (xx+3<OW))
        tl.store(Y+base+qi*S2+1*S3+0*S4,y10+bias,valid & (tt+qi<OT) & (yy+1<OH) & (xx+0<OW))
        tl.store(Y+base+qi*S2+1*S3+1*S4,y11+bias,valid & (tt+qi<OT) & (yy+1<OH) & (xx+1<OW))
        tl.store(Y+base+qi*S2+1*S3+2*S4,y12+bias,valid & (tt+qi<OT) & (yy+1<OH) & (xx+2<OW))
        tl.store(Y+base+qi*S2+1*S3+3*S4,y13+bias,valid & (tt+qi<OT) & (yy+1<OH) & (xx+3<OW))
        tl.store(Y+base+qi*S2+2*S3+0*S4,y20+bias,valid & (tt+qi<OT) & (yy+2<OH) & (xx+0<OW))
        tl.store(Y+base+qi*S2+2*S3+1*S4,y21+bias,valid & (tt+qi<OT) & (yy+2<OH) & (xx+1<OW))
        tl.store(Y+base+qi*S2+2*S3+2*S4,y22+bias,valid & (tt+qi<OT) & (yy+2<OH) & (xx+2<OW))
        tl.store(Y+base+qi*S2+2*S3+3*S4,y23+bias,valid & (tt+qi<OT) & (yy+2<OH) & (xx+3<OW))
        tl.store(Y+base+qi*S2+3*S3+0*S4,y30+bias,valid & (tt+qi<OT) & (yy+3<OH) & (xx+0<OW))
        tl.store(Y+base+qi*S2+3*S3+1*S4,y31+bias,valid & (tt+qi<OT) & (yy+3<OH) & (xx+1<OW))
        tl.store(Y+base+qi*S2+3*S3+2*S4,y32+bias,valid & (tt+qi<OT) & (yy+3<OH) & (xx+2<OW))
        tl.store(Y+base+qi*S2+3*S3+3*S4,y33+bias,valid & (tt+qi<OT) & (yy+3<OH) & (xx+3<OW))

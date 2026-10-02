"""Mixed F43 convolution with virtual causal padding and history loads.

Inference-only BF16 input/output, FP16 transform coefficients, FP32 transforms
and accumulation. Unsupported calls retain the supplied original forward.
"""
import torch
import triton
import triton.language as tl
from .. import winograd_3d_conv as w3
from .causal_kernel import _point
from .fused_f43 import time_gemm, output_transform
from .mixed_f43 import transform_weight


@triton.jit
def input_causal(X,CACHE,V,C:tl.constexpr,T:tl.constexpr,H:tl.constexpr,W:tl.constexpr,
                    CT:tl.constexpr,NT:tl.constexpr,NH:tl.constexpr,NW:tl.constexpr,P:tl.constexpr,
                    START,S0:tl.constexpr,S1:tl.constexpr,S2:tl.constexpr,
                    S3:tl.constexpr,S4:tl.constexpr,C0:tl.constexpr,C1:tl.constexpr,C2:tl.constexpr,C3:tl.constexpr,C4:tl.constexpr,BLOCK:tl.constexpr):
    z=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    qt=tl.program_id(1)
    c,p=z//P,z%P
    gp=START+p
    nn,tt=gp//(NT*NH*NW),2*((gp//(NH*NW))%NT)-2
    yy,xx=4*((gp//NW)%NH)-1,4*(gp%NW)-1
    ta=tl.where(qt==0,0,tl.where(qt==2,2,1))
    tb=tl.where(qt<2,2,tl.where(qt==2,1,3))
    base=nn*S0+c*S1+tt*S2+yy*S3+xx*S4
    valid=z<C*P
    span=C*P

    da00=_point(X,CACHE,nn,c,tt+ta,yy+0,xx+0,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db00=_point(X,CACHE,nn,c,tt+tb,yy+0,xx+0,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d00=tl.where(qt==1,da00+db00,da00-db00)
    da01=_point(X,CACHE,nn,c,tt+ta,yy+0,xx+1,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db01=_point(X,CACHE,nn,c,tt+tb,yy+0,xx+1,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d01=tl.where(qt==1,da01+db01,da01-db01)
    da02=_point(X,CACHE,nn,c,tt+ta,yy+0,xx+2,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db02=_point(X,CACHE,nn,c,tt+tb,yy+0,xx+2,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d02=tl.where(qt==1,da02+db02,da02-db02)
    da03=_point(X,CACHE,nn,c,tt+ta,yy+0,xx+3,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db03=_point(X,CACHE,nn,c,tt+tb,yy+0,xx+3,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d03=tl.where(qt==1,da03+db03,da03-db03)
    da04=_point(X,CACHE,nn,c,tt+ta,yy+0,xx+4,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db04=_point(X,CACHE,nn,c,tt+tb,yy+0,xx+4,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d04=tl.where(qt==1,da04+db04,da04-db04)
    da05=_point(X,CACHE,nn,c,tt+ta,yy+0,xx+5,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db05=_point(X,CACHE,nn,c,tt+tb,yy+0,xx+5,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d05=tl.where(qt==1,da05+db05,da05-db05)
    da10=_point(X,CACHE,nn,c,tt+ta,yy+1,xx+0,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db10=_point(X,CACHE,nn,c,tt+tb,yy+1,xx+0,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d10=tl.where(qt==1,da10+db10,da10-db10)
    da11=_point(X,CACHE,nn,c,tt+ta,yy+1,xx+1,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db11=_point(X,CACHE,nn,c,tt+tb,yy+1,xx+1,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d11=tl.where(qt==1,da11+db11,da11-db11)
    da12=_point(X,CACHE,nn,c,tt+ta,yy+1,xx+2,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db12=_point(X,CACHE,nn,c,tt+tb,yy+1,xx+2,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d12=tl.where(qt==1,da12+db12,da12-db12)
    da13=_point(X,CACHE,nn,c,tt+ta,yy+1,xx+3,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db13=_point(X,CACHE,nn,c,tt+tb,yy+1,xx+3,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d13=tl.where(qt==1,da13+db13,da13-db13)
    da14=_point(X,CACHE,nn,c,tt+ta,yy+1,xx+4,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db14=_point(X,CACHE,nn,c,tt+tb,yy+1,xx+4,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d14=tl.where(qt==1,da14+db14,da14-db14)
    da15=_point(X,CACHE,nn,c,tt+ta,yy+1,xx+5,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db15=_point(X,CACHE,nn,c,tt+tb,yy+1,xx+5,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d15=tl.where(qt==1,da15+db15,da15-db15)
    da20=_point(X,CACHE,nn,c,tt+ta,yy+2,xx+0,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db20=_point(X,CACHE,nn,c,tt+tb,yy+2,xx+0,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d20=tl.where(qt==1,da20+db20,da20-db20)
    da21=_point(X,CACHE,nn,c,tt+ta,yy+2,xx+1,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db21=_point(X,CACHE,nn,c,tt+tb,yy+2,xx+1,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d21=tl.where(qt==1,da21+db21,da21-db21)
    da22=_point(X,CACHE,nn,c,tt+ta,yy+2,xx+2,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db22=_point(X,CACHE,nn,c,tt+tb,yy+2,xx+2,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d22=tl.where(qt==1,da22+db22,da22-db22)
    da23=_point(X,CACHE,nn,c,tt+ta,yy+2,xx+3,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db23=_point(X,CACHE,nn,c,tt+tb,yy+2,xx+3,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d23=tl.where(qt==1,da23+db23,da23-db23)
    da24=_point(X,CACHE,nn,c,tt+ta,yy+2,xx+4,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db24=_point(X,CACHE,nn,c,tt+tb,yy+2,xx+4,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d24=tl.where(qt==1,da24+db24,da24-db24)
    da25=_point(X,CACHE,nn,c,tt+ta,yy+2,xx+5,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db25=_point(X,CACHE,nn,c,tt+tb,yy+2,xx+5,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d25=tl.where(qt==1,da25+db25,da25-db25)
    da30=_point(X,CACHE,nn,c,tt+ta,yy+3,xx+0,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db30=_point(X,CACHE,nn,c,tt+tb,yy+3,xx+0,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d30=tl.where(qt==1,da30+db30,da30-db30)
    da31=_point(X,CACHE,nn,c,tt+ta,yy+3,xx+1,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db31=_point(X,CACHE,nn,c,tt+tb,yy+3,xx+1,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d31=tl.where(qt==1,da31+db31,da31-db31)
    da32=_point(X,CACHE,nn,c,tt+ta,yy+3,xx+2,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db32=_point(X,CACHE,nn,c,tt+tb,yy+3,xx+2,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d32=tl.where(qt==1,da32+db32,da32-db32)
    da33=_point(X,CACHE,nn,c,tt+ta,yy+3,xx+3,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db33=_point(X,CACHE,nn,c,tt+tb,yy+3,xx+3,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d33=tl.where(qt==1,da33+db33,da33-db33)
    da34=_point(X,CACHE,nn,c,tt+ta,yy+3,xx+4,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db34=_point(X,CACHE,nn,c,tt+tb,yy+3,xx+4,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d34=tl.where(qt==1,da34+db34,da34-db34)
    da35=_point(X,CACHE,nn,c,tt+ta,yy+3,xx+5,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db35=_point(X,CACHE,nn,c,tt+tb,yy+3,xx+5,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d35=tl.where(qt==1,da35+db35,da35-db35)
    da40=_point(X,CACHE,nn,c,tt+ta,yy+4,xx+0,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db40=_point(X,CACHE,nn,c,tt+tb,yy+4,xx+0,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d40=tl.where(qt==1,da40+db40,da40-db40)
    da41=_point(X,CACHE,nn,c,tt+ta,yy+4,xx+1,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db41=_point(X,CACHE,nn,c,tt+tb,yy+4,xx+1,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d41=tl.where(qt==1,da41+db41,da41-db41)
    da42=_point(X,CACHE,nn,c,tt+ta,yy+4,xx+2,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db42=_point(X,CACHE,nn,c,tt+tb,yy+4,xx+2,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d42=tl.where(qt==1,da42+db42,da42-db42)
    da43=_point(X,CACHE,nn,c,tt+ta,yy+4,xx+3,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db43=_point(X,CACHE,nn,c,tt+tb,yy+4,xx+3,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d43=tl.where(qt==1,da43+db43,da43-db43)
    da44=_point(X,CACHE,nn,c,tt+ta,yy+4,xx+4,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db44=_point(X,CACHE,nn,c,tt+tb,yy+4,xx+4,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d44=tl.where(qt==1,da44+db44,da44-db44)
    da45=_point(X,CACHE,nn,c,tt+ta,yy+4,xx+5,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db45=_point(X,CACHE,nn,c,tt+tb,yy+4,xx+5,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d45=tl.where(qt==1,da45+db45,da45-db45)
    da50=_point(X,CACHE,nn,c,tt+ta,yy+5,xx+0,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db50=_point(X,CACHE,nn,c,tt+tb,yy+5,xx+0,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d50=tl.where(qt==1,da50+db50,da50-db50)
    da51=_point(X,CACHE,nn,c,tt+ta,yy+5,xx+1,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db51=_point(X,CACHE,nn,c,tt+tb,yy+5,xx+1,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d51=tl.where(qt==1,da51+db51,da51-db51)
    da52=_point(X,CACHE,nn,c,tt+ta,yy+5,xx+2,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db52=_point(X,CACHE,nn,c,tt+tb,yy+5,xx+2,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d52=tl.where(qt==1,da52+db52,da52-db52)
    da53=_point(X,CACHE,nn,c,tt+ta,yy+5,xx+3,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db53=_point(X,CACHE,nn,c,tt+tb,yy+5,xx+3,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d53=tl.where(qt==1,da53+db53,da53-db53)
    da54=_point(X,CACHE,nn,c,tt+ta,yy+5,xx+4,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db54=_point(X,CACHE,nn,c,tt+tb,yy+5,xx+4,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d54=tl.where(qt==1,da54+db54,da54-db54)
    da55=_point(X,CACHE,nn,c,tt+ta,yy+5,xx+5,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    db55=_point(X,CACHE,nn,c,tt+tb,yy+5,xx+5,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d55=tl.where(qt==1,da55+db55,da55-db55)
    a00=(4.0 * d00) + (-5.0 * d02) + d04
    a01=(-4.0 * d01) + (-4.0 * d02) + d03 + d04
    a02=(4.0 * d01) + (-4.0 * d02) + (-d03) + d04
    a03=(-2.0 * d01) + (-d02) + (2.0 * d03) + d04
    a04=(2.0 * d01) + (-d02) + (-2.0 * d03) + d04
    a05=(4.0 * d01) + (-5.0 * d03) + d05
    a10=(4.0 * d10) + (-5.0 * d12) + d14
    a11=(-4.0 * d11) + (-4.0 * d12) + d13 + d14
    a12=(4.0 * d11) + (-4.0 * d12) + (-d13) + d14
    a13=(-2.0 * d11) + (-d12) + (2.0 * d13) + d14
    a14=(2.0 * d11) + (-d12) + (-2.0 * d13) + d14
    a15=(4.0 * d11) + (-5.0 * d13) + d15
    a20=(4.0 * d20) + (-5.0 * d22) + d24
    a21=(-4.0 * d21) + (-4.0 * d22) + d23 + d24
    a22=(4.0 * d21) + (-4.0 * d22) + (-d23) + d24
    a23=(-2.0 * d21) + (-d22) + (2.0 * d23) + d24
    a24=(2.0 * d21) + (-d22) + (-2.0 * d23) + d24
    a25=(4.0 * d21) + (-5.0 * d23) + d25
    a30=(4.0 * d30) + (-5.0 * d32) + d34
    a31=(-4.0 * d31) + (-4.0 * d32) + d33 + d34
    a32=(4.0 * d31) + (-4.0 * d32) + (-d33) + d34
    a33=(-2.0 * d31) + (-d32) + (2.0 * d33) + d34
    a34=(2.0 * d31) + (-d32) + (-2.0 * d33) + d34
    a35=(4.0 * d31) + (-5.0 * d33) + d35
    a40=(4.0 * d40) + (-5.0 * d42) + d44
    a41=(-4.0 * d41) + (-4.0 * d42) + d43 + d44
    a42=(4.0 * d41) + (-4.0 * d42) + (-d43) + d44
    a43=(-2.0 * d41) + (-d42) + (2.0 * d43) + d44
    a44=(2.0 * d41) + (-d42) + (-2.0 * d43) + d44
    a45=(4.0 * d41) + (-5.0 * d43) + d45
    a50=(4.0 * d50) + (-5.0 * d52) + d54
    a51=(-4.0 * d51) + (-4.0 * d52) + d53 + d54
    a52=(4.0 * d51) + (-4.0 * d52) + (-d53) + d54
    a53=(-2.0 * d51) + (-d52) + (2.0 * d53) + d54
    a54=(2.0 * d51) + (-d52) + (-2.0 * d53) + d54
    a55=(4.0 * d51) + (-5.0 * d53) + d55
    v00=(4.0 * a00) + (-5.0 * a20) + a40
    tl.store(V+z+(qt*36+0)*span,v00,valid)
    v01=(4.0 * a01) + (-5.0 * a21) + a41
    tl.store(V+z+(qt*36+1)*span,v01,valid)
    v02=(4.0 * a02) + (-5.0 * a22) + a42
    tl.store(V+z+(qt*36+2)*span,v02,valid)
    v03=(4.0 * a03) + (-5.0 * a23) + a43
    tl.store(V+z+(qt*36+3)*span,v03,valid)
    v04=(4.0 * a04) + (-5.0 * a24) + a44
    tl.store(V+z+(qt*36+4)*span,v04,valid)
    v05=(4.0 * a05) + (-5.0 * a25) + a45
    tl.store(V+z+(qt*36+5)*span,v05,valid)
    v10=(-4.0 * a10) + (-4.0 * a20) + a30 + a40
    tl.store(V+z+(qt*36+6)*span,v10,valid)
    v11=(-4.0 * a11) + (-4.0 * a21) + a31 + a41
    tl.store(V+z+(qt*36+7)*span,v11,valid)
    v12=(-4.0 * a12) + (-4.0 * a22) + a32 + a42
    tl.store(V+z+(qt*36+8)*span,v12,valid)
    v13=(-4.0 * a13) + (-4.0 * a23) + a33 + a43
    tl.store(V+z+(qt*36+9)*span,v13,valid)
    v14=(-4.0 * a14) + (-4.0 * a24) + a34 + a44
    tl.store(V+z+(qt*36+10)*span,v14,valid)
    v15=(-4.0 * a15) + (-4.0 * a25) + a35 + a45
    tl.store(V+z+(qt*36+11)*span,v15,valid)
    v20=(4.0 * a10) + (-4.0 * a20) + (-a30) + a40
    tl.store(V+z+(qt*36+12)*span,v20,valid)
    v21=(4.0 * a11) + (-4.0 * a21) + (-a31) + a41
    tl.store(V+z+(qt*36+13)*span,v21,valid)
    v22=(4.0 * a12) + (-4.0 * a22) + (-a32) + a42
    tl.store(V+z+(qt*36+14)*span,v22,valid)
    v23=(4.0 * a13) + (-4.0 * a23) + (-a33) + a43
    tl.store(V+z+(qt*36+15)*span,v23,valid)
    v24=(4.0 * a14) + (-4.0 * a24) + (-a34) + a44
    tl.store(V+z+(qt*36+16)*span,v24,valid)
    v25=(4.0 * a15) + (-4.0 * a25) + (-a35) + a45
    tl.store(V+z+(qt*36+17)*span,v25,valid)
    v30=(-2.0 * a10) + (-a20) + (2.0 * a30) + a40
    tl.store(V+z+(qt*36+18)*span,v30,valid)
    v31=(-2.0 * a11) + (-a21) + (2.0 * a31) + a41
    tl.store(V+z+(qt*36+19)*span,v31,valid)
    v32=(-2.0 * a12) + (-a22) + (2.0 * a32) + a42
    tl.store(V+z+(qt*36+20)*span,v32,valid)
    v33=(-2.0 * a13) + (-a23) + (2.0 * a33) + a43
    tl.store(V+z+(qt*36+21)*span,v33,valid)
    v34=(-2.0 * a14) + (-a24) + (2.0 * a34) + a44
    tl.store(V+z+(qt*36+22)*span,v34,valid)
    v35=(-2.0 * a15) + (-a25) + (2.0 * a35) + a45
    tl.store(V+z+(qt*36+23)*span,v35,valid)
    v40=(2.0 * a10) + (-a20) + (-2.0 * a30) + a40
    tl.store(V+z+(qt*36+24)*span,v40,valid)
    v41=(2.0 * a11) + (-a21) + (-2.0 * a31) + a41
    tl.store(V+z+(qt*36+25)*span,v41,valid)
    v42=(2.0 * a12) + (-a22) + (-2.0 * a32) + a42
    tl.store(V+z+(qt*36+26)*span,v42,valid)
    v43=(2.0 * a13) + (-a23) + (-2.0 * a33) + a43
    tl.store(V+z+(qt*36+27)*span,v43,valid)
    v44=(2.0 * a14) + (-a24) + (-2.0 * a34) + a44
    tl.store(V+z+(qt*36+28)*span,v44,valid)
    v45=(2.0 * a15) + (-a25) + (-2.0 * a35) + a45
    tl.store(V+z+(qt*36+29)*span,v45,valid)
    v50=(4.0 * a10) + (-5.0 * a30) + a50
    tl.store(V+z+(qt*36+30)*span,v50,valid)
    v51=(4.0 * a11) + (-5.0 * a31) + a51
    tl.store(V+z+(qt*36+31)*span,v51,valid)
    v52=(4.0 * a12) + (-5.0 * a32) + a52
    tl.store(V+z+(qt*36+32)*span,v52,valid)
    v53=(4.0 * a13) + (-5.0 * a33) + a53
    tl.store(V+z+(qt*36+33)*span,v53,valid)
    v54=(4.0 * a14) + (-5.0 * a34) + a54
    tl.store(V+z+(qt*36+34)*span,v54,valid)
    v55=(4.0 * a15) + (-5.0 * a35) + a55
    tl.store(V+z+(qt*36+35)*span,v55,valid)


def make_forward(module,fallback,workspace_mib=128,config=(64,128,32,8,3),block=128,channel_pairs=None,fp_fusion=False):
    """Use only square 512/256 residual convolutions; preserve fallback elsewhere."""
    pairs={(512,512),(256,256)} if channel_pairs is None else set(channel_pairs)
    def forward(self,x,cache_x=None):
        allowed=(self._conv_backend=='winograd_3d' and not self.training and x.is_cuda
                 and x.ndim==5 and min(x.shape[2:])>0
                 and self.weight.ndim==5 and x.shape[1]==self.weight.shape[1]
                 and x.device==self.weight.device
                 and (self.bias is None or (self.bias.device==x.device and self.bias.shape==(self.weight.shape[0],)))
                 and self.kernel_size==(3,3,3) and self.stride==(1,1,1)
                 and self.dilation==(1,1,1) and self.groups==1
                 and self._padding==(1,1,1,1,2,0)
                 and (self.weight.shape[1],self.weight.shape[0]) in pairs
                 and w3._effective_dtype(x,self.weight,self.bias)==torch.bfloat16)
        if not allowed:return fallback(self,x,cache_x)
        if torch.is_grad_enabled() and any(v.requires_grad for v in (x,self.weight,self.bias) if v is not None):
            return fallback(self,x,cache_x)
        if cache_x is not None:
            cache_x=cache_x.to(x.device)
            if not (cache_x.ndim==5 and cache_x.shape[:2]==x.shape[:2]
                    and cache_x.shape[3:]==x.shape[3:] and cache_x.shape[2]<=2):
                return fallback(self,x,cache_x)
        ct=0 if cache_x is None else cache_x.shape[2]
        cache_tensor=x if cache_x is None else cache_x
        n,c,t,h,w=x.shape
        k=self.weight.shape[0]
        with torch.cuda.device(x.device),torch.autocast('cuda',enabled=False):
            key=w3._cache_key(self.weight,torch.bfloat16,True)
            key=(*key,'mixed_f43_causal',torch.float16) if key is not None else None
            cache=self._winograd_weight_cache
            if key is not None and cache is not None and cache[0]==key:
                u=cache[1]
            else:
                u=transform_weight(self.weight,torch.float16)
                self._winograd_weight_cache=(key,u) if key is not None else None
            bias=self.bias.to(torch.bfloat16) if self.bias is not None else None
            fmt=torch.channels_last_3d if x.is_contiguous(memory_format=torch.channels_last_3d) else torch.contiguous_format
            out=torch.empty((n,k,t,h,w),device=x.device,dtype=torch.bfloat16,memory_format=fmt)
            nt,nh,nw=(t+1)//2,(h+3)//4,(w+3)//4
            total=n*nt*nh*nw
            qi,qo=(3,1) if t==1 else (4,2)
            ws=workspace_mib.get((c,k),workspace_mib.get(k,128)) if isinstance(workspace_mib,dict) else workspace_mib
            chunk=max(1,int(ws*1024*1024)//(36*(qi*2*c+qo*4*k)))
            if chunk>=128:chunk=chunk//128*128
            bm,bn,bk,warps,stages=config
            for start in range(0,total,chunk):
                p=min(chunk,total-start)
                v=torch.empty((qi*36,c,p),device=x.device,dtype=torch.float16)
                m=torch.empty((qo*36,k,p),device=x.device,dtype=torch.float32)
                input_causal[(triton.cdiv(c*p,block),qi)](x,cache_tensor,v,c,t,h,w,ct,nt,nh,nw,p,start,*x.stride(),*cache_tensor.stride(),BLOCK=block,num_warps=4,enable_fp_fusion=fp_fusion)
                time_gemm[(triton.cdiv(k,bm),triton.cdiv(p,bn),36)](u,v,m,k,c,p,BM=bm,BN=bn,BK=bk,ONE=t==1,num_warps=warps,num_stages=stages,enable_fp_fusion=fp_fusion)
                output_transform[(triton.cdiv(k*p,block),)](m,bias if bias is not None else out,out,k,p,start,t,h,w,nt,nh,nw,bias is not None,*out.stride(),BLOCK=block,num_warps=4,enable_fp_fusion=fp_fusion)
            return out
    return forward

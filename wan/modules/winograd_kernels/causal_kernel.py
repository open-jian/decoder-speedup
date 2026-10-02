"""Fuse virtual causal cache, zero padding and BF16 quantization into B^T input transform.

Arithmetic is identical to the ca72457 path: values are first BF16-rounded,
then transformed in FP32. The checkpoint, convolution and causal history remain.
"""
import torch
import triton
import triton.language as tl
from .. import winograd_3d_triton as k3
from .. import winograd_3d_conv as w3
_bt=k3._bt

@triton.jit
def _point(X,CACHE,nn,cc,tt,yy,xx,valid,
           T:tl.constexpr,H:tl.constexpr,W:tl.constexpr,CT:tl.constexpr,
           S0:tl.constexpr,S1:tl.constexpr,S2:tl.constexpr,S3:tl.constexpr,S4:tl.constexpr,
           C0:tl.constexpr,C1:tl.constexpr,C2:tl.constexpr,C3:tl.constexpr,C4:tl.constexpr):
    valid=valid & (yy>=0) & (yy<H) & (xx>=0) & (xx<W)
    d=tl.load(X+nn*S0+cc*S1+tt*S2+yy*S3+xx*S4,
              valid & (tt>=0) & (tt<T),0)
    if CT>0:
        ht=tt+CT
        old=tl.load(CACHE+nn*C0+cc*C1+ht*C2+yy*C3+xx*C4,
                    valid & (tt<0) & (ht>=0),0)
        d=tl.where(tt>=0,d,old)
    return d.to(tl.bfloat16).to(tl.float32)

@triton.jit
def _spatial_causal(X,CACHE,nn,cc,tt,yy,xx,valid,
                    T:tl.constexpr,H:tl.constexpr,W:tl.constexpr,CT:tl.constexpr,
                    S0:tl.constexpr,S1:tl.constexpr,S2:tl.constexpr,S3:tl.constexpr,S4:tl.constexpr,
                    C0:tl.constexpr,C1:tl.constexpr,C2:tl.constexpr,C3:tl.constexpr,C4:tl.constexpr):
    d00=_point(X,CACHE,nn,cc,tt,yy+0,xx+0,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d01=_point(X,CACHE,nn,cc,tt,yy+0,xx+1,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d02=_point(X,CACHE,nn,cc,tt,yy+0,xx+2,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d03=_point(X,CACHE,nn,cc,tt,yy+0,xx+3,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d10=_point(X,CACHE,nn,cc,tt,yy+1,xx+0,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d11=_point(X,CACHE,nn,cc,tt,yy+1,xx+1,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d12=_point(X,CACHE,nn,cc,tt,yy+1,xx+2,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d13=_point(X,CACHE,nn,cc,tt,yy+1,xx+3,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d20=_point(X,CACHE,nn,cc,tt,yy+2,xx+0,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d21=_point(X,CACHE,nn,cc,tt,yy+2,xx+1,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d22=_point(X,CACHE,nn,cc,tt,yy+2,xx+2,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d23=_point(X,CACHE,nn,cc,tt,yy+2,xx+3,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d30=_point(X,CACHE,nn,cc,tt,yy+3,xx+0,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d31=_point(X,CACHE,nn,cc,tt,yy+3,xx+1,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d32=_point(X,CACHE,nn,cc,tt,yy+3,xx+2,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d33=_point(X,CACHE,nn,cc,tt,yy+3,xx+3,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    a00, a01, a02, a03 = _bt(d00, d01, d02, d03)
    a10, a11, a12, a13 = _bt(d10, d11, d12, d13)
    a20, a21, a22, a23 = _bt(d20, d21, d22, d23)
    a30, a31, a32, a33 = _bt(d30, d31, d32, d33)
    v00, v10, v20, v30 = _bt(a00, a10, a20, a30)
    v01, v11, v21, v31 = _bt(a01, a11, a21, a31)
    v02, v12, v22, v32 = _bt(a02, a12, a22, a32)
    v03, v13, v23, v33 = _bt(a03, a13, a23, a33)
    return v00, v01, v02, v03, v10, v11, v12, v13, v20, v21, v22, v23, v30, v31, v32, v33


@triton.jit
def input_causal(X, CACHE, V, C: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
                        W: tl.constexpr, CT: tl.constexpr, NT: tl.constexpr, NH: tl.constexpr,
                        NW: tl.constexpr, P: tl.constexpr, START,
                        S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
                        S3: tl.constexpr, S4: tl.constexpr,
                        C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr,
                        C3: tl.constexpr, C4: tl.constexpr, BLOCK: tl.constexpr):
    z = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    c, p = z//P, z % P
    gp = START+p
    nn, tt = gp//(NT*NH*NW), 2*((gp//(NH*NW)) % NT)-2
    yy, xx = 2*((gp//NW) % NH)-1, 2*(gp % NW)-1
    valid = z<C*P
    span = C*P
    d0_00, d0_01, d0_02, d0_03, d0_10, d0_11, d0_12, d0_13, d0_20, d0_21, d0_22, d0_23, d0_30, d0_31, d0_32, d0_33 = _spatial_causal(
        X,CACHE,nn,c,tt+0,yy,xx,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d1_00, d1_01, d1_02, d1_03, d1_10, d1_11, d1_12, d1_13, d1_20, d1_21, d1_22, d1_23, d1_30, d1_31, d1_32, d1_33 = _spatial_causal(
        X,CACHE,nn,c,tt+1,yy,xx,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d2_00, d2_01, d2_02, d2_03, d2_10, d2_11, d2_12, d2_13, d2_20, d2_21, d2_22, d2_23, d2_30, d2_31, d2_32, d2_33 = _spatial_causal(
        X,CACHE,nn,c,tt+2,yy,xx,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    d3_00, d3_01, d3_02, d3_03, d3_10, d3_11, d3_12, d3_13, d3_20, d3_21, d3_22, d3_23, d3_30, d3_31, d3_32, d3_33 = _spatial_causal(
        X,CACHE,nn,c,tt+3,yy,xx,valid,T,H,W,CT,S0,S1,S2,S3,S4,C0,C1,C2,C3,C4)
    # B^T along time after the spatial transforms; q = t*16 + h*4 + w.
    v0_00, v1_00, v2_00, v3_00 = _bt(d0_00, d1_00, d2_00, d3_00)
    tl.store(V+z+0*span, v0_00, valid)
    tl.store(V+z+16*span, v1_00, valid)
    tl.store(V+z+32*span, v2_00, valid)
    if T>1:
        tl.store(V+z+48*span, v3_00, valid)
    v0_01, v1_01, v2_01, v3_01 = _bt(d0_01, d1_01, d2_01, d3_01)
    tl.store(V+z+1*span, v0_01, valid)
    tl.store(V+z+17*span, v1_01, valid)
    tl.store(V+z+33*span, v2_01, valid)
    if T>1:
        tl.store(V+z+49*span, v3_01, valid)
    v0_02, v1_02, v2_02, v3_02 = _bt(d0_02, d1_02, d2_02, d3_02)
    tl.store(V+z+2*span, v0_02, valid)
    tl.store(V+z+18*span, v1_02, valid)
    tl.store(V+z+34*span, v2_02, valid)
    if T>1:
        tl.store(V+z+50*span, v3_02, valid)
    v0_03, v1_03, v2_03, v3_03 = _bt(d0_03, d1_03, d2_03, d3_03)
    tl.store(V+z+3*span, v0_03, valid)
    tl.store(V+z+19*span, v1_03, valid)
    tl.store(V+z+35*span, v2_03, valid)
    if T>1:
        tl.store(V+z+51*span, v3_03, valid)
    v0_10, v1_10, v2_10, v3_10 = _bt(d0_10, d1_10, d2_10, d3_10)
    tl.store(V+z+4*span, v0_10, valid)
    tl.store(V+z+20*span, v1_10, valid)
    tl.store(V+z+36*span, v2_10, valid)
    if T>1:
        tl.store(V+z+52*span, v3_10, valid)
    v0_11, v1_11, v2_11, v3_11 = _bt(d0_11, d1_11, d2_11, d3_11)
    tl.store(V+z+5*span, v0_11, valid)
    tl.store(V+z+21*span, v1_11, valid)
    tl.store(V+z+37*span, v2_11, valid)
    if T>1:
        tl.store(V+z+53*span, v3_11, valid)
    v0_12, v1_12, v2_12, v3_12 = _bt(d0_12, d1_12, d2_12, d3_12)
    tl.store(V+z+6*span, v0_12, valid)
    tl.store(V+z+22*span, v1_12, valid)
    tl.store(V+z+38*span, v2_12, valid)
    if T>1:
        tl.store(V+z+54*span, v3_12, valid)
    v0_13, v1_13, v2_13, v3_13 = _bt(d0_13, d1_13, d2_13, d3_13)
    tl.store(V+z+7*span, v0_13, valid)
    tl.store(V+z+23*span, v1_13, valid)
    tl.store(V+z+39*span, v2_13, valid)
    if T>1:
        tl.store(V+z+55*span, v3_13, valid)
    v0_20, v1_20, v2_20, v3_20 = _bt(d0_20, d1_20, d2_20, d3_20)
    tl.store(V+z+8*span, v0_20, valid)
    tl.store(V+z+24*span, v1_20, valid)
    tl.store(V+z+40*span, v2_20, valid)
    if T>1:
        tl.store(V+z+56*span, v3_20, valid)
    v0_21, v1_21, v2_21, v3_21 = _bt(d0_21, d1_21, d2_21, d3_21)
    tl.store(V+z+9*span, v0_21, valid)
    tl.store(V+z+25*span, v1_21, valid)
    tl.store(V+z+41*span, v2_21, valid)
    if T>1:
        tl.store(V+z+57*span, v3_21, valid)
    v0_22, v1_22, v2_22, v3_22 = _bt(d0_22, d1_22, d2_22, d3_22)
    tl.store(V+z+10*span, v0_22, valid)
    tl.store(V+z+26*span, v1_22, valid)
    tl.store(V+z+42*span, v2_22, valid)
    if T>1:
        tl.store(V+z+58*span, v3_22, valid)
    v0_23, v1_23, v2_23, v3_23 = _bt(d0_23, d1_23, d2_23, d3_23)
    tl.store(V+z+11*span, v0_23, valid)
    tl.store(V+z+27*span, v1_23, valid)
    tl.store(V+z+43*span, v2_23, valid)
    if T>1:
        tl.store(V+z+59*span, v3_23, valid)
    v0_30, v1_30, v2_30, v3_30 = _bt(d0_30, d1_30, d2_30, d3_30)
    tl.store(V+z+12*span, v0_30, valid)
    tl.store(V+z+28*span, v1_30, valid)
    tl.store(V+z+44*span, v2_30, valid)
    if T>1:
        tl.store(V+z+60*span, v3_30, valid)
    v0_31, v1_31, v2_31, v3_31 = _bt(d0_31, d1_31, d2_31, d3_31)
    tl.store(V+z+13*span, v0_31, valid)
    tl.store(V+z+29*span, v1_31, valid)
    tl.store(V+z+45*span, v2_31, valid)
    if T>1:
        tl.store(V+z+61*span, v3_31, valid)
    v0_32, v1_32, v2_32, v3_32 = _bt(d0_32, d1_32, d2_32, d3_32)
    tl.store(V+z+14*span, v0_32, valid)
    tl.store(V+z+30*span, v1_32, valid)
    tl.store(V+z+46*span, v2_32, valid)
    if T>1:
        tl.store(V+z+62*span, v3_32, valid)
    v0_33, v1_33, v2_33, v3_33 = _bt(d0_33, d1_33, d2_33, d3_33)
    tl.store(V+z+15*span, v0_33, valid)
    tl.store(V+z+31*span, v1_33, valid)
    tl.store(V+z+47*span, v2_33, valid)
    if T>1:
        tl.store(V+z+63*span, v3_33, valid)

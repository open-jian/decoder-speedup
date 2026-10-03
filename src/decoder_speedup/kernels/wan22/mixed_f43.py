"""Mixed F(2 time, 4 space) weight and spatial inverse transforms.

Weights are BF16-rounded, transformed in FP32, then stored in the selected
transform-domain dtype. The production causal path uses FP16 coefficients.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def spatial_inverse(M,z,span,valid,QT:tl.constexpr):
    m00=tl.load(M+z+(QT*36+0)*span,valid,0)
    m01=tl.load(M+z+(QT*36+1)*span,valid,0)
    m02=tl.load(M+z+(QT*36+2)*span,valid,0)
    m03=tl.load(M+z+(QT*36+3)*span,valid,0)
    m04=tl.load(M+z+(QT*36+4)*span,valid,0)
    m05=tl.load(M+z+(QT*36+5)*span,valid,0)
    m10=tl.load(M+z+(QT*36+6)*span,valid,0)
    m11=tl.load(M+z+(QT*36+7)*span,valid,0)
    m12=tl.load(M+z+(QT*36+8)*span,valid,0)
    m13=tl.load(M+z+(QT*36+9)*span,valid,0)
    m14=tl.load(M+z+(QT*36+10)*span,valid,0)
    m15=tl.load(M+z+(QT*36+11)*span,valid,0)
    m20=tl.load(M+z+(QT*36+12)*span,valid,0)
    m21=tl.load(M+z+(QT*36+13)*span,valid,0)
    m22=tl.load(M+z+(QT*36+14)*span,valid,0)
    m23=tl.load(M+z+(QT*36+15)*span,valid,0)
    m24=tl.load(M+z+(QT*36+16)*span,valid,0)
    m25=tl.load(M+z+(QT*36+17)*span,valid,0)
    m30=tl.load(M+z+(QT*36+18)*span,valid,0)
    m31=tl.load(M+z+(QT*36+19)*span,valid,0)
    m32=tl.load(M+z+(QT*36+20)*span,valid,0)
    m33=tl.load(M+z+(QT*36+21)*span,valid,0)
    m34=tl.load(M+z+(QT*36+22)*span,valid,0)
    m35=tl.load(M+z+(QT*36+23)*span,valid,0)
    m40=tl.load(M+z+(QT*36+24)*span,valid,0)
    m41=tl.load(M+z+(QT*36+25)*span,valid,0)
    m42=tl.load(M+z+(QT*36+26)*span,valid,0)
    m43=tl.load(M+z+(QT*36+27)*span,valid,0)
    m44=tl.load(M+z+(QT*36+28)*span,valid,0)
    m45=tl.load(M+z+(QT*36+29)*span,valid,0)
    m50=tl.load(M+z+(QT*36+30)*span,valid,0)
    m51=tl.load(M+z+(QT*36+31)*span,valid,0)
    m52=tl.load(M+z+(QT*36+32)*span,valid,0)
    m53=tl.load(M+z+(QT*36+33)*span,valid,0)
    m54=tl.load(M+z+(QT*36+34)*span,valid,0)
    m55=tl.load(M+z+(QT*36+35)*span,valid,0)
    a00=m00 + m01 + m02 + m03 + m04
    a01=m01 + (-m02) + (2.0 * m03) + (-2.0 * m04)
    a02=m01 + m02 + (4.0 * m03) + (4.0 * m04)
    a03=m01 + (-m02) + (8.0 * m03) + (-8.0 * m04) + m05
    a10=m10 + m11 + m12 + m13 + m14
    a11=m11 + (-m12) + (2.0 * m13) + (-2.0 * m14)
    a12=m11 + m12 + (4.0 * m13) + (4.0 * m14)
    a13=m11 + (-m12) + (8.0 * m13) + (-8.0 * m14) + m15
    a20=m20 + m21 + m22 + m23 + m24
    a21=m21 + (-m22) + (2.0 * m23) + (-2.0 * m24)
    a22=m21 + m22 + (4.0 * m23) + (4.0 * m24)
    a23=m21 + (-m22) + (8.0 * m23) + (-8.0 * m24) + m25
    a30=m30 + m31 + m32 + m33 + m34
    a31=m31 + (-m32) + (2.0 * m33) + (-2.0 * m34)
    a32=m31 + m32 + (4.0 * m33) + (4.0 * m34)
    a33=m31 + (-m32) + (8.0 * m33) + (-8.0 * m34) + m35
    a40=m40 + m41 + m42 + m43 + m44
    a41=m41 + (-m42) + (2.0 * m43) + (-2.0 * m44)
    a42=m41 + m42 + (4.0 * m43) + (4.0 * m44)
    a43=m41 + (-m42) + (8.0 * m43) + (-8.0 * m44) + m45
    a50=m50 + m51 + m52 + m53 + m54
    a51=m51 + (-m52) + (2.0 * m53) + (-2.0 * m54)
    a52=m51 + m52 + (4.0 * m53) + (4.0 * m54)
    a53=m51 + (-m52) + (8.0 * m53) + (-8.0 * m54) + m55
    y00=a00 + a10 + a20 + a30 + a40
    y01=a01 + a11 + a21 + a31 + a41
    y02=a02 + a12 + a22 + a32 + a42
    y03=a03 + a13 + a23 + a33 + a43
    y10=a10 + (-a20) + (2.0 * a30) + (-2.0 * a40)
    y11=a11 + (-a21) + (2.0 * a31) + (-2.0 * a41)
    y12=a12 + (-a22) + (2.0 * a32) + (-2.0 * a42)
    y13=a13 + (-a23) + (2.0 * a33) + (-2.0 * a43)
    y20=a10 + a20 + (4.0 * a30) + (4.0 * a40)
    y21=a11 + a21 + (4.0 * a31) + (4.0 * a41)
    y22=a12 + a22 + (4.0 * a32) + (4.0 * a42)
    y23=a13 + a23 + (4.0 * a33) + (4.0 * a43)
    y30=a10 + (-a20) + (8.0 * a30) + (-8.0 * a40) + a50
    y31=a11 + (-a21) + (8.0 * a31) + (-8.0 * a41) + a51
    y32=a12 + (-a22) + (8.0 * a32) + (-8.0 * a42) + a52
    y33=a13 + (-a23) + (8.0 * a33) + (-8.0 * a43) + a53
    return y00, y01, y02, y03, y10, y11, y12, y13, y20, y21, y22, y23, y30, y31, y32, y33


def transform_weight(weight, transform_dtype):
    g2=torch.tensor([[1,0,0],[.5,.5,.5],[.5,-.5,.5],[0,0,1]],device=weight.device,dtype=torch.float32)
    g4=torch.tensor([[.25,0,0],[-1/6,-1/6,-1/6],[-1/6,1/6,-1/6],[1/24,1/12,1/6],[1/24,-1/12,1/6],[0,0,1]],device=weight.device,dtype=torch.float32)
    w=weight.to(torch.bfloat16).float()
    u=torch.einsum('kcxyz,az->kcxya',w,g4)
    u=torch.einsum('kcxyb,ay->kcxab',u,g4)
    u=torch.einsum('kcxyz,ax->kcayz',u,g2)
    return u.permute(2,3,4,0,1).reshape(144,*weight.shape[:2]).to(transform_dtype).contiguous()

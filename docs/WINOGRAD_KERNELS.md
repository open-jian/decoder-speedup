# Wan2.2 decoder kernels

An opt-in inference implementation for the original Wan2.2 VAE decoder. It keeps
the original parameters, channel counts, layers, latent format, causal schedule,
module hierarchy and checkpoint keys. There is no training or pruning.

## Usage

Load the original checkpoint, then install the adapter. The ordinary decoder
remains the default until `install` is called.

```python
import torch
from wan.modules.vae2_2 import Wan2_2_VAE
from wan.modules.winograd_kernels import install

# Settings used for the reported measurements and transformed-weight precision.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

vae = Wan2_2_VAE(
    vae_pth='/path/to/Wan2.2_VAE.pth',
    device='cuda', dtype=torch.bfloat16,
)
handle = install(vae)

# Normalized latent samples [48, T, H, W], just as with the original wrapper.
with torch.inference_mode():
    videos = vae.decode([latent.cuda()])

handle.disable()  # Restore the exact original forward methods and backends.
handle.enable()   # Re-enable; warm up again before measuring latency.
handle.remove()   # Restore and allow a fresh installation.
```

For a raw, unnormalized latent tensor, use `vae.model.decode(z, [0., 1.])` inside
BF16 autocast and `torch.inference_mode()`. For an encoder-normalized latent,
use the wrapper or `vae.model.decode(z, vae.scale)`. These inputs are different;
mixing them invalidates quality comparisons.

The implementation uses PyTorch and Triton, tested with PyTorch 2.11.0+cu130,
Triton 3.6.0, cuDNN 9.19 and an RTX 6000 Ada. Stored weights remain FP32.
The accelerated path requires CUDA BF16 inference. The existing convolution
backend handles unsupported cases; Winograd remains inference-only. Disable
the adapter before training, changing the architecture, or adding hooks to fused
normalization/activation pairs. Do not toggle the adapter during a decode or
CUDA Graph replay. Save the ordinary `state_dict`, not the runtime handle.

## Implementation

1. Fuse causal history reads, virtual padding and BF16 input conversion into
   the Winograd input kernel; omit unused time-transform coefficients for a
   one-frame convolution output.
2. Fuse the height inverse transform into the F(2,3) GEMM epilogue, halving its
   FP32 intermediate. Preserve the original inverse-transform order.
3. Use F(2,3) in time and F(4,3) in space for the four selected channel pairs
   `(1024,512)`, `(512,512)`, `(512,256)`, `(256,256)`. Use per-pair workspace
   budgets and tuned GEMM tiles. F43 transforms use FP16 coefficients after the
   same BF16 input/weight quantization, with FP32 transforms and accumulation.
4. Fuse RMS normalization, SiLU and the following convolution's BF16 conversion.
5. Fuse nearest-neighbor 2x upsampling into the spatial input transform. Seven
   of sixteen transform coordinates are identically zero, leaving nine GEMMs.
6. Index the upsample shortcut directly during residual addition, and retain
   read-only cache-tail views instead of copying them.

These changes alter floating-point reduction/rounding. They are not bit-exact
relative to native cuDNN or the previous complete decoder. The paired real-video
evaluation is evidence for the tested data, not a proof of lossless behavior for
arbitrary activations. FP16 transform coefficients also have less exponent range
than BF16; extreme out-of-distribution inputs were not certified. The larger
spatial transform can be disabled with `install(vae, spatial_f43=False)`.

## Reproduce latency

Use an idle GPU. The benchmark warms every mode after switching, randomizes mode
blocks, and reports all nine synchronized wall-time samples. It includes the
complete GPU decoder, including `conv2` and causal cache reset. It excludes model
loading, initial JIT compilation, weight-transform warmup, file I/O and CPU/GPU
transfers. No `torch.compile`, CUDA Graph, training or logging service is used.

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_decoder_kernels.py \
  --checkpoint /path/to/Wan2.2_VAE.pth \
  --latents /path/to/unnormalized-9f.npy /path/to/unnormalized-81f.npy \
  --output /path/to/decoder-kernels-results.json
```

The four modes use the same model and inputs: native cuDNN, the existing 28-layer
3D Winograd configuration, its previously fastest selective-cuBLAS variant, and
this adapter. Numerical differences in this latency benchmark measure agreement
with the old decoder; they are not reconstruction-quality metrics. Peak memory
is reported with shared-process caches present, not as a fresh-process minimum.

The local research report records matched latency, six real-video PSNR/SSIM/LPIPS
measurements, independent evaluation split, causal contracts, and source hashes.

## Measured results (2026-10-02)

RTX 6000 Ada, batch 1, 240x320 output, warmed complete GPU decode, nine samples:

| Frames | Native BF16 | Previous Winograd | Previous best selective cuBLAS | New kernels |
|---:|---:|---:|---:|---:|
| 9 | 138.93 ms | 130.04 ms | 125.13 ms | **53.99 ms** |
| 81 | 1252.64 ms | 1116.77 ms | 1082.57 ms | **487.70 ms** |

At 81 frames, new kernels reduce latency by **56.33%** versus previous Winograd,
**54.95%** versus the previous best, and **61.07%** versus native BF16 (2.57x).
A separate 253-frame long-sequence check measures 3488.03 to 1572.95 ms,
**54.90%** lower latency versus previous Winograd, with stable causal cache storage.
The 253-frame measurement uses three samples and another card of the same type.

Six real DROID/RE10K clips (two validation, four held-out evaluation), all 81
frames at 240x320, use identical original-encoder latents across modes:

| Decoder | PSNR (dB) | SSIM | LPIPS (VGG) |
|---|---:|---:|---:|
| Native BF16 | 36.897862 | 0.97146651 | 0.02912155 |
| Previous Winograd / previous best | 36.899677 | 0.97145022 | 0.02912249 |
| New kernels | 36.898512 | 0.97145739 | 0.02914805 |

On the four held-out clips, mean PSNR changes by -0.000101 dB versus native.
This is a small evaluation set, not a claim across resolutions or hardware.
Thirteen streaming/numerical checks and thirty production-adapter checks pass.
Production and the evaluated research implementation match bitwise at both
9 and 81 frames; installation preserves checkpoint keys, tensors and parameters.

The teacher's `open-jian/world-model-uta` project informed transform layout,
shape-aware dispatch and larger spatial tiles. These Triton kernels are our
implementation. Fusing normalization, causal padding and upsampling also
contributes to the complete-decoder speedup; the result is not attributable
solely to replacing a GEMM kernel or to a built-in NVIDIA optimization.

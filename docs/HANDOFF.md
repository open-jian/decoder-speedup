# Wan2.2 compressed decoder — 100k checkpoint user guide

This package replaces the **VAE decoder** in an existing Wan pipeline. It uses
recovered weights for a narrower decoder, optional Winograd/fused kernels, and
optional CUDA Graph replay. Keep the original encoder, diffusion model, and
video-saving code. No training is needed to use this export.

## 1. Check compatibility

| Your pipeline's VAE | This checkpoint |
|---|---|
| `Wan2.2_VAE.pth`, 48 latent channels, spatial stride 16; used by **TI2V-5B** | Supported interface |
| `Wan2.1_VAE.pth`, 16 latent channels, spatial stride 8; used by **T2V-A14B / I2V-A14B** | Incompatible |

The Wan2.2 product name alone does not identify its VAE. This guide targets the
official `WanTI2V` pipeline. Diffusers and ComfyUI need their own integration
adapters. The student file cannot be loaded into the original full-width VAE
class or used by renaming it to `Wan2.2_VAE.pth`; use the loader below.

## 2. Get the code and weights

Code: [decoder-speedup, branch wan22-amd-width-winograd](https://github.com/open-jian/decoder-speedup/tree/wan22-amd-width-winograd).

Use **`student-100000-ema.pt`**, the inference export of EMA weights after
100,000 generator updates. Size: **364,850,355 bytes**, about **365 MB**.
On a machine with the lab NAS mounted, use this path directly:

```text
/data/nas/jian/ckpt/wan22-amd-width-turbo-10k-20261001/student-100000-ema.pt
```

Otherwise, obtain the file from Jian and save it locally. Weights are **not
included in Git or a GitHub release**. The larger `checkpoint-00100000.pt` is
for resuming training; it is not needed for inference.

Run `sha256sum /your/path/student-100000-ema.pt` after copying. Expected SHA256:

```text
f027d0b3ad2d30aa32f5964a53bf78ac5409cb95247d22ecee6e36392aab8a79
```

## 3. Install beside your existing Wan project

Use a project environment with working NVIDIA CUDA PyTorch, BF16 support, and
matching Triton. The kernel validation environment was Linux, PyTorch
**2.11.0+cu130**, Triton **3.6.0**, and RTX 6000 Ada. Other versions and GPUs need
their own execution check.

Clone into new directories without overwriting your existing Wan checkout:

```bash
git clone --branch wan22-amd-width-winograd https://github.com/open-jian/decoder-speedup.git
git clone --branch winograd https://github.com/open-jian/Wan2.2.git Wan2.2-decoder-source
git -C Wan2.2-decoder-source checkout ca724575ae721ac84639c729bc07dbe2428a49de

python -m pip install -e ./decoder-speedup
python -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__); assert torch.cuda.is_available()"
```

The second checkout provides the **exact VAE source required by the export**.
Keep that revision: the loader checks its source-file hashes. It uses a private
import namespace, so your original pipeline can continue importing its own `wan`.
There is no need to install the second checkout's full diffusion dependencies.
Student inference does not need a dataset, W&B, LPIPS, GAN, or teacher weights.
Keep the original VAE weights if your pipeline still uses its encoder.

## 4. Run a small decoding check

Replace the two absolute paths and select your CUDA device. This standalone
check does not load the diffusion model.

```python
import torch
from decoder_speedup.deployment import Wan22CompressedDecoder

fast = Wan22CompressedDecoder(
    checkpoint="/absolute/path/student-100000-ema.pt",
    source="/absolute/path/Wan2.2-decoder-source",
    device="cuda:0",
    use_winograd=True,
    use_cuda_graph=True,
)
try:
    latent = torch.zeros(48, 3, 15, 20, device="cuda:0")
    video = fast.decode([latent])[0]
    assert video.shape == (3, 9, 240, 320)
    assert video.dtype == torch.float32 and torch.isfinite(video).all()
    print(video.shape, video.dtype, video.device)
finally:
    fast.close()
```

First use can compile Triton kernels and capture a graph. A zero latent checks
execution and output shape; it is not a reconstruction-quality test.

## 5. Connect to your existing pipeline

Create a **new decoder instance once** after constructing `wan_ti2v`. The
standalone check above closes its own instance. In official `generate.py`, insert
the attachment after `wan_ti2v = wan.WanTI2V(...)` and before its `generate` call:

```python
from decoder_speedup.deployment import Wan22CompressedDecoder

fast = None
original_decode = None
# Official WanTI2V performs VAE decoding on rank 0.
if wan_ti2v.rank == 0:
    fast = Wan22CompressedDecoder(
        checkpoint="/absolute/path/student-100000-ema.pt",
        source="/absolute/path/Wan2.2-decoder-source",
        device=str(wan_ti2v.device),
        use_winograd=True,
        use_cuda_graph=True,
    )
    original_decode = wan_ti2v.vae.decode
    wan_ti2v.vae.decode = fast.decode
```

All ranks now execute your existing `wan_ti2v.generate(...)` call with its
existing arguments. Keep `fast` alive for all generation calls and save as before.

Place cleanup **after all generation calls**, preferably in the program's
`finally` block. Do not put this cleanup before generation:

```python
if fast is not None:
    wan_ti2v.vae.decode = original_decode
    fast.close()
```

For applications that call the VAE directly, replace `vae.decode(latents)` with
`fast.decode(latents)`. The input is a **Python list** of normalized CUDA tensors,
each `[48, T, H, W]`, on the selected device. Do not add a batch dimension to each
item or apply mean/std again. The loader already includes latent normalization.

The result is a list of FP32 RGB tensors, each
`[3, 1 + 4*(T-1), 16*H, 16*W]`, clipped to `[-1, 1]` on the same GPU.
Your existing video-saving code can use these outputs. The wrapper supplies
inference mode and BF16 autocast; weights stay FP32. Input tensors can be FP32,
BF16, or FP16.

Only `vae.decode` is replaced. `vae.encode` keeps the original encoder, and the
original VAE stays allocated unless your application explicitly manages it.

## 6. Runtime options

| `use_winograd` | `use_cuda_graph` | Execution |
|---|---|---|
| `False` | `False` | Compressed decoder with ordinary execution |
| `True` | `False` | Our Winograd and fused kernels |
| `False` | `True` | Official CUDA Graph replay |
| `True` | `True` | Our kernels plus CUDA Graph replay |

The wrapper retains one graph shape/dtype at a time. Different frame counts,
resolutions, or input dtypes close the old graph and trigger capture again.
Repeatedly alternating shapes pays that cost repeatedly. Set `use_cuda_graph=False`
for ordinary execution when needed; time warmed calls separately from first use.

Do not train, cast, move, or modify the loaded decoder during use. After changing
weights, devices, or precision settings, close it and construct a new instance.
A closed wrapper cannot decode again; intermediate-feature hooks and gradients
are outside this deployment API.

Common fixes:

- **Source mismatch:** use the pinned `ca724575...` checkout, without bypassing hash checks.
- **Expected `[48,T,H,W]`:** supply a list of unbatched latents; 16-channel models need different weights.
- **Device mismatch:** put latents and decoder on the same CUDA device.
- **First call is slow:** it includes compilation and graph capture.

## 7. What has been verified

**100k export:** transfer SHA256 matches erebus; all 114 tensors are finite and
exactly match the checkpoint's EMA state. CPU loading is strict. Structure,
tensor shapes/dtypes, and the frozen latent projection/normalization match the
80k export. **100k GPU speed and reconstruction quality have not been remeasured.**

**Earlier 80k results:** on RTX 6000 Ada, 81 frames at 240×320, warmed core decoding
measured 147.3 ms for the compressed student, 98.1 ms with CUDA Graph, and
56.6 ms with Winograd plus CUDA Graph. With Graph enabled on both sides, the
kernel combination reduced latency by 42.3% (1.73× faster). Winograd alone did not
give a reliable ordinary-execution speedup.

Those are not 100k measurements, 720p/1080p results, or full-generation speedups.
They exclude loading, compilation/capture, and the wrapper's final FP32 conversion
and clamping. The earlier PSNR change of about −0.0015 dB measures the additional
kernel difference against the **80k compressed student**, not compression loss
versus original Wan. See [the original comparison](WIDTH_WINOGRAD.md).

# Use the compressed decoder in an existing Wan pipeline

Give the recipient this branch, the exported student weight file, and this guide.
The exported weight file is a smaller decoder, not a replacement file that the
original full-width `Wan2_2_VAE` class can load directly.

## 1. Check the VAE variant

This artifact supports **Wan2.2_VAE.pth with 48 latent channels and spatial
stride 16**, used by **Wan2.2 TI2V-5B**. The original encoder and diffusion model
remain usable. The student's decoder already applies Wan's latent normalization.

The Wan2.2 T2V-A14B and I2V-A14B pipelines use **Wan2.1_VAE.pth**, with 16 latent
channels and spatial stride 8. They cannot use this artifact. Check the actual
VAE class/latent channels, rather than relying only on the Wan2.2 product name.
Diffusers-specific pipeline integration is not included in the example below.

## 2. Get code and weights

Clone both repositories into new directories next to the existing Wan project:

```bash
git clone --branch wan22-amd-width-winograd https://github.com/open-jian/decoder-speedup.git
git clone --branch winograd https://github.com/open-jian/Wan2.2.git Wan2.2-decoder-source
git -C Wan2.2-decoder-source checkout ca724575ae721ac84639c729bc07dbe2428a49de
```

The second checkout supplies the exact VAE source expected by the weight file.
It can coexist with the recipient's original Wan checkout; keep its recorded
revision. The loader uses a private import namespace, so it does not replace the
original pipeline's `wan` import. Standalone student decoding does not require
installing this second checkout's full diffusion-model dependencies.

Activate a project environment with CUDA PyTorch, then install the framework:

```bash
python -m pip install -e ./decoder-speedup
python -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__); assert torch.cuda.is_available()"
```

Winograd requires NVIDIA CUDA, BF16 support, and Triton. The measured stack was
PyTorch **2.11.0+cu130**, Triton **3.6.0**, on an RTX 6000 Ada. Other versions and
GPUs require their own smoke test and benchmark. Use the Triton version matched
to the CUDA PyTorch installation. Inference does not need W&B, LPIPS, a dataset,
GAN weights, or training optimizer checkpoints.

Receive **`student-80000-ema.pt`** from Jian. It contains the student structure
configuration, latent projection/normalization, and EMA decoder weights from
80,000 recovery updates. It is **364,850,355 bytes** (about 365 MB / 348 MiB).
It is on the lab NAS, not uploaded as a GitHub weight release:

```text
/data/nas/jian/ckpt/wan22-amd-width-turbo-10k-20261001/student-80000-ema.pt
SHA256: f581c5d17f2eec4905d0ef9c05196a654eebc5a9ef92063e55207482587229d8
```

If the recipient can access the same NAS, use that path directly or copy the file.
Otherwise, send the actual `.pt` file through an agreed file-transfer service.
After copying, verify it with `sha256sum student-80000-ema.pt`.
The decoder alone does not need the original VAE weight file; keep the original
weights if the existing pipeline still needs the encoder.

## 3. Replace the decode call

```python
from decoder_speedup.deployment import Wan22CompressedDecoder

# Construct once, on the CUDA device used by the pipeline's VAE.
fast_decoder = Wan22CompressedDecoder(
    checkpoint="/absolute/path/student-80000-ema.pt",
    source="/absolute/path/Wan2.2-decoder-source",
    device="cuda:0",
    use_winograd=True,
    use_cuda_graph=True,
)

# Original:
# videos = pipeline.vae.decode(latents)

# Replacement: same list input and output convention.
videos = fast_decoder.decode(latents)

# At process shutdown, after all decoding is complete:
fast_decoder.close()
```

`latents` is the existing pipeline's **list of normalized CUDA tensors**, each
shaped `[48, latent_frames, latent_height, latent_width]`, on the selected GPU.
Do not normalize or rescale them again. The result is a list of FP32 RGB tensors
in `[-1, 1]`, each `[3, 1 + 4*(latent_frames - 1), 16*latent_height, 16*latent_width]`.
Keep the existing video-saving code.

For a quick interface check before connecting the pipeline, call
`fast_decoder.decode([torch.zeros(48, 3, 15, 20, device="cuda:0")])[0]`
(after `import torch`, using the selected device). Its shape should be
`[3, 9, 240, 320]`. This synthetic input checks execution and shape, not quality.

For the official `WanTI2V` object, `pipeline` above is the `wan_ti2v` instance.
To keep its internal generation code unchanged, attach the compatible decode
method after constructing the pipeline and before calling `generate`:

```python
original_decode = wan_ti2v.vae.decode
wan_ti2v.vae.decode = fast_decoder.decode
# Run wan_ti2v.generate(...) as before. vae.encode still uses the original encoder.

# When finished, restore before closing the replacement:
wan_ti2v.vae.decode = original_decode
fast_decoder.close()
```

For distributed generation, create and attach this decoder only on the rank
that performs VAE decoding (rank 0 in the checked official pipeline), using that
rank's actual CUDA device. This example changes only decode calls; it does not
automatically unload the original VAE from GPU memory.

## 4. Execution modes and first call

| Options | Execution |
|---|---|
| `use_winograd=False, use_cuda_graph=False` | Compressed decoder with ordinary PyTorch execution |
| `use_winograd=True, use_cuda_graph=False` | Compressed decoder with our inference kernels |
| `use_winograd=True, use_cuda_graph=True` | Our kernels plus official CUDA Graph replay |

The wrapper keeps FP32 weights and runs BF16 inference internally. No recovery
training or weight conversion is needed. CUDA Graph is captured on the first
decode and reused for matching latent shape/dtype/device. A different shape or
dtype closes the old graph and captures a new one; alternating shapes repeatedly
will repeatedly pay capture overhead. First use can also compile Triton kernels.
Time warmed calls separately from initialization/capture.

Do not train, move, cast, or modify the loaded decoder while using it. Close and
construct a new wrapper after changing weights/device/precision. Intermediate
feature hooks and gradients are not part of this deployment API. A closed wrapper
cannot decode again.

## 5. What the measurements mean

On the tested RTX 6000 Ada, **81 output frames at 240×320**, complete warmed
decoding measured 147.3 ms for the compressed student, 98.1 ms for the student
with CUDA Graph, and 56.6 ms for Winograd plus CUDA Graph. Winograd alone did not
produce a reliable eager gain. With Graph enabled on both sides, the kernel
combination reduced latency by 42.3% (1.73× faster).

The 56.6 ms result is not a 720p/1080p latency or full video-generation speedup.
It measures the core decoder; the deployment wrapper additionally applies the
official FP32 conversion and output clamping. Time its full call in the recipient's pipeline.
Width compression has its own quality cost versus the original VAE. Enabling
the kernels added about −0.0015 dB PSNR on six tested clips relative to the
already compressed student; this does not claim lossless compression.
See [the detailed comparison](WIDTH_WINOGRAD.md).

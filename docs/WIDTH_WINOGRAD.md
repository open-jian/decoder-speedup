# Width-compressed Wan2.2 with Winograd kernels

Branch `wan22-amd-width-winograd` applies inference kernels from the `winograd`
branch ([`b8d81dc`](https://github.com/open-jian/decoder-speedup/commit/b8d81dc))
to the existing AMD stage-width student. The student still uses stage
widths `[512, 512, 256, 64, 32]`, all 14 residual blocks, and the same latent
projection, normalization, causal schedule, and RGB output convention. Installing
the adapter changes execution methods, not parameters or checkpoint keys.

## Load an existing student

Use the external Wan source recorded in the artifact. The 80,000-update EMA
export requires `open-jian/Wan2.2` commit
`ca724575ae721ac84639c729bc07dbe2428a49de`. This framework packages the new
kernels separately so the external source and its fingerprint can stay fixed.
Do not bypass source checks or overwrite that checkout with another branch.

```python
import torch
from decoder_speedup.export import load_student
from decoder_speedup.winograd import install

student, metadata = load_student(
    "/path/to/student-80000-ema.pt", "/path/to/matching/Wan2.2", "cuda:0"
)
student = student.float().eval()
handle = install(student)  # Width-specific policy; spatial_f43=False by default.

# Normalized Wan latent [B, 48, T, H, W], on the same CUDA device.
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    rgb, features = student(normalized_wan_latent)
    rgb = rgb.float().clamp(-1, 1)

handle.disable()  # Restore the original width-compressed decoder execution.
handle.enable()   # Re-enable and warm up before timing.
handle.remove()   # Restore and release this installation.
```

Keep the handle alive while using the decoder. Installation is per model: it does
not patch the source classes, encoder, or another model sharing the source.
Ordinary `state_dict()` and exported student artifacts retain their original
keys and tensors. Save these artifacts rather than pickling a patched model or
its runtime handle. No new checkpoint conversion, initialization, or recovery
training is needed to use the kernels with an already trained student.

## Configuration and commands

The separate inference preset `configs/wan22/width-winograd.yaml` enables:

```yaml
runtime:
  device: cuda:0
  precision: bf16
  weight_dtype: fp32
  channels_last: false
  compile: false
  winograd: true
```

Set the environment variables referenced by the preset, including
`WAN22_SOURCE`, `WAN22_WEIGHTS`, `VIDGEN_ROOT`, `MANIFEST_DIR`, and `RUN_DIR`.
Then use the existing local-only commands:

```bash
python -m decoder_speedup benchmark configs/wan22/width-winograd.yaml \
  --student /path/to/student-80000-ema.pt --output /path/to/timing.json
python -m decoder_speedup evaluate configs/wan22/width-winograd.yaml \
  --student /path/to/student-80000-ema.pt --output /path/to/quality.json
```

Benchmarking and evaluation do not start W&B unless `--log-wandb` is explicitly
supplied. Model loading, first-time Triton compilation, and transformed-weight
warmup are excluded from warmed decode latency. Quality uses real videos;
random-latent timing is not a reconstruction-quality test.

To measure the additional kernel gain, also run the same student with
`runtime.winograd: false`, holding precision, stored weight dtype, layout,
input dimensions, checkpoint, backend settings, and GPU fixed. Report both
`baseline_ms / accelerated_ms` and `100 * (1 - accelerated_ms / baseline_ms)`.
The original-width Winograd gain must not be multiplied by a width-compression
gain measured separately: narrower layers have different compute and memory
costs, so the combined speedup requires its own measurement.

## Optional CUDA Graph

CUDA Graph is the official PyTorch/NVIDIA facility for replaying captured GPU
work with less CPU launch overhead. It is separate from our Winograd kernels.
The CLI and `runtime.winograd` continue to use eager execution; graph capture is
an explicit Python API:

```python
from decoder_speedup.cuda_graph import capture_decoder

# student is already loaded outside inference_mode, on CUDA, and in eval mode.
handle = install(student)
graph = capture_decoder(student, normalized_wan_latent, kernel_handle=handle)
try:
    with torch.inference_mode():
        rgb, _ = graph(next_normalized_wan_latent)
finally:
    graph.close()
    handle.remove()
```

BF16 autocast is applied internally. Each replay requires the example's exact
latent shape, dtype, and device. It copies the input into graph storage and
returns an independent output clone. Training, gradients, module hooks, and
intermediate-feature requests are unsupported. Keep weights, buffers, precision
settings, and adapters unchanged until `close()`; validity checks reject changed
state, and changing shape requires a fresh capture. Unsafe `.data` writes bypass
PyTorch mutation checks and are unsupported. Captured buffers consume memory.

For a matched baseline, capture the same compressed decoder without installing
Winograd. `kernel_handle` is detected automatically when the runtime installed
it; pass it explicitly when calling `install` directly.

### Measured comparison

Same RTX 6000 Ada, batch 1, 81 output frames at 240×320, FP32 stored weights and
BF16 AMP, TF32 disabled. Values are means of nine synchronized complete-decode
samples in three randomized blocks. Timers include graph validity checks, input
copy, and output clone, but exclude loading, JIT/warmup, graph capture, encoding,
file I/O, and host/device transfers.

| Decoder and execution | Mean latency |
|---|---:|
| Original Wan2.2, eager | 1224.69 ms |
| Width-compressed, eager | 147.31 ms |
| Width-compressed + Winograd, eager | 151.11 ms |
| Width-compressed + CUDA Graph | 98.11 ms |
| Width-compressed + Winograd + CUDA Graph | **56.64 ms** |

Winograd alone gave **no reliable eager benefit**: this run was slightly slower,
while an earlier run changed 141.4 to 139.4 ms. CPU submission overhead limits the
compressed eager decoder. The matched graph-to-graph comparison isolates the
additional kernel benefit: **1.73× faster, 42.27% lower latency**. Kernels plus
graph replay versus compressed eager execution yield **2.60× / 61.55% lower
latency**. Versus original Wan2.2 eager execution, the full combination is
**21.62× / 95.38% lower latency**; that last baseline does not use CUDA Graph.
Attribute these combined results to width compression, our kernels, and the
official graph facility together.

Six real-video clips with the same 80,000-update EMA student measured PSNR
33.651929 → 33.650444 dB and LPIPS 0.05625669 → 0.05627626 when enabling the
kernels. Accelerated graph replay matched accelerated eager output bit-for-bit
on all six quality clips; native graph parity was also checked for the timing input.
These results cover the tested checkpoint, resolution,
GPU, and small quality set, rather than establishing universal speed or quality.

The [recorded measurements and checks](benchmarks/width-winograd-20261002.json)
include individual timing samples, per-clip quality, source hashes, 12 decoder
checks, and 23 graph checks. The full CPU test suite passed 67 tests.

## What is adapted

The framework passes the student's actual source module and decoder to the
adapter, including the width-configured decoder subclass. Layer dispatch uses
the real input/output channel counts rather than assuming original Wan widths.
The policy retains native convolutions where the large-channel Winograd policy
does not apply; it does not force every small convolution onto Winograd.

The opt-in kernels include fused causal input transforms, an F(2,3) inverse
transform partially fused into GEMM, normalization with SiLU, nearest-neighbor
upsampling convolution, upsample residual addition, and read-only history-cache
views. Fused normalization pairs are located by module type and ordering.
Dropout, residual projections, stage boundaries, and latent normalization remain
the compressed model's existing operations.

`install` also accepts `spatial_f43`, `conv_channel_pairs`,
`f43_workspace_mib`, `fuse_norm`, `fuse_upsample`, `structural`, and
`f43_min_area` for controlled ablations. The default convolution policy requires
both input and output widths to be at least 128 channels, leaving residual
convolutions involving 32/64 channels on their native backend. Other fusions can
still apply in those stages. This branch defaults to `spatial_f43=False`: the larger F43 spatial
tile used by the original-width implementation was slower for the compressed
student's tested small feature maps. It remains available as an explicit ablation
for configured channel pairs.

Reusing the kernel implementation does not mean reusing the same layer policy.
Width reduction changes matrix sizes and the balance between GPU computation and
CPU launch overhead. Lower GPU kernel time alone does not establish lower complete
decode latency; comparisons must use synchronized end-to-end timing. These
options select execution policies without changing the student architecture.
Different GPUs, widths, and resolutions can need different policies.

## Inference and training boundaries

The supported accelerated runtime is CUDA BF16 AMP with FP32 stored weights.
`torch.compile` and training are not enabled together with this adapter. CPU
execution and the ordinary width-only training path do not require importing
Triton. Existing `width.yaml` defaults keep `runtime.winograd` disabled.

Disable/remove the adapter before training, changing architecture, moving a
patched model to another device/dtype, or installing hooks on fused
normalization/activation modules. Do not toggle the adapter during an in-flight
decode or CUDA Graph replay. Whole-block feature hooks retain the framework's
feature collection contract, with original structural paths where required.

Continue the existing recovery run from the existing `wan22-amd-width` checkout
and its resolved configuration. This branch does not change a running job,
optimizer, EMA, dataset cursor, or W&B run. Full-state resume still checks recipe,
data, external source, and framework provenance; the fact that student exports
remain compatible does not disable those training checks.

## Numerical and performance limits

The kernels change floating-point operation order and use FP16 transform
coefficients for selected paths after BF16 input/weight quantization. They are
not bit-exact relative to native execution. Validate output finiteness and
PSNR/SSIM/LPIPS for the compressed checkpoint and input distribution being
deployed; original-width quality results are not substitutes for this check.

Compiled kernels, transformed weights, and workspaces consume memory, and first
use includes compilation. Cache-tail views may retain their source tensors
longer than copies. Neither original-width latency numbers nor parameter counts
establish the speed or memory footprint of the combined model.

# decoder-speedup

## Width compression + Winograd branch

This is **`wan22-amd-width-winograd`**. It adds opt-in Winograd and fused inference kernels to the AMD stage-width student from [`wan22-amd-width`](https://github.com/open-jian/decoder-speedup/tree/wan22-amd-width). Existing compressed checkpoints, including `student-80000-ema.pt`, load without weight conversion or retraining. The repository's [main branch](https://github.com/open-jian/decoder-speedup/tree/main) remains the native Wan2.2 baseline; [`winograd`](https://github.com/open-jian/decoder-speedup/tree/winograd) provides the original-width implementation.

```bash
git clone --branch wan22-amd-width-winograd https://github.com/open-jian/decoder-speedup.git
```

For an existing Wan user, start with the [handoff and pipeline integration guide](docs/HANDOFF.md): code setup, the exported weight file, and a `vae.decode`-compatible deployment interface. This artifact targets the 48-channel Wan2.2 VAE used by TI2V-5B; A14B pipelines using the 16-channel Wan2.1 VAE need a different artifact.

Existing compressed exports require the matching external Wan source. For the 80,000-update EMA artifact, use [open-jian/Wan2.2 at ca72457](https://github.com/open-jian/Wan2.2/tree/ca724575ae721ac84639c729bc07dbe2428a49de). Keep that source revision for strict artifact compatibility. The new inference kernels live in this framework and do not require modifying or upgrading the external source.

The new adapter is disabled by default. Enable `runtime.winograd: true` for evaluation/benchmarking, or call `decoder_speedup.winograd.install(student)` after loading a student for deployment. It requires CUDA BF16 AMP, FP32 stored weights, and `compile: false`. The original `width.yaml` recovery-training preset is unchanged; `width-winograd.yaml` is inference-only. See [combined inference and checkpoint compatibility](docs/WIDTH_WINOGRAD.md).

The combined adapter uses a policy for the compressed widths: residual convolutions with both channel counts at least 128 use F(2,3), while residual convolutions involving 32/64 channels retain their native backend. Other fusions can still apply in those stages. Larger spatial F43 tiles are available for ablations but disabled by default. Measure the combined decoder directly; speedups from the two separate branches do not multiply automatically.

Optional `decoder_speedup.cuda_graph.capture_decoder` reduces CPU launch overhead for a fixed latent shape. This uses the official PyTorch/NVIDIA CUDA Graph facility, separately from our Winograd kernels; eager execution remains the default. On one RTX 6000 Ada, 81 frames at 240×320, the compressed decoder measured **147.31 ms eager**, **98.11 ms with CUDA Graph**, and **56.64 ms with Winograd + CUDA Graph**. Winograd alone gave no reliable eager improvement. The matched graph-to-graph gain is **1.73× / 42.27% lower latency**. See [usage, restrictions, and full comparison](docs/WIDTH_WINOGRAD.md#optional-cuda-graph).

An independent framework for decoder acceleration and recovery training. The first version applies **the stage widths from AMD v1/v3 to the original Wan2.2 VAE, changing only channel widths**. The original encoder stays frozen while the student decoder is trained. The 48-channel latent interface and normalization convention remain unchanged.

The framework supports architecture configuration, training, evaluation, and export, with single-GPU and torchrun multi-GPU execution. See [distributed training](docs/DISTRIBUTED.md). The default widths come from AMD's released configurations; recovery quality, training budgets, and speed must be evaluated for this adaptation of the original Wan architecture.

## Project structure

```text
src/decoder_speedup/
  models/wan22/    Wan source integration, student construction, initialization, causal caches
  config.py       Strict configuration and architecture dependency checks
  data.py         Fixed manifests, source-video splits, resumable data cursor
  training/       Reconstruction/perceptual/feature losses, GAN, EMA, full checkpoints
  evaluation.py   PSNR, SSIM, LPIPS, decoding latency, memory usage
  runtime.py      Precision, convolution memory layout, inference compilation
  winograd.py     Reversible inference adapter for exported width-compressed students
  cuda_graph.py   Optional fixed-shape CUDA Graph capture and checked replay
  deployment.py   List-based replacement for the official 48-channel VAE decode API
  kernels/        Vendored Winograd transforms and fused Triton inference kernels
  export.py       Standalone student weights and architecture
configs/wan22/    Original-width, local-width, and AMD stage-width configurations
examples/        Integration checks with official Wan weights
experiments/     Experiment launch scripts, fixed manifests, and log snapshots
```

The Wan VAE architecture source and original weights remain external dependencies. This repository packages its inference kernels separately and does not modify the external checkout. The training loop uses model inputs and feature supervision returned by `teacher.prepare()`. Additional decoders can be integrated through new adapters.

The four-GPU recovery run with 10,000 VidGen training videos is documented in the [experiment archive](experiments/20261001_wan22_width_turbo/README.md).

## Installation

Install in a project-specific virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
# Install PyTorch / torchvision versions appropriate for your machine first.
pip install -e '.[perceptual,tracking,test]'
```

With an existing validated PyTorch environment, you can also set `PYTHONPATH=src` and use `python -m decoder_speedup`. LPIPS uses pretrained VGG weights, which may be downloaded on first use. Set `TORCH_HOME` to choose the cache directory.

## Width configuration

The five entries in `width.stages` specify the entry/middle stage, the first three upsampling stages, and the final stage at RGB resolution, in that order.

```yaml
width:
  stages: [512, 512, 256, 64, 32]
  hidden: {}
```

AMD v1 and v3 both specify `decoder_block_out_channels: [32,64,256,512]`. Their constructor reverses this sequence and uses 512 channels for the entry/middle stage, producing the five-stage configuration above. See [AMD width configuration](docs/AMD_WIDTH.md) for sources and the mapping. This adaptation uses those channel counts in execution order; it does not reproduce AMD's temporal and spatial resolutions at each stage.

The original widths are `[1024, 1024, 1024, 512, 256]`. `hidden` controls the channels between the two large convolutions inside a residual block; unspecified blocks follow their stage widths. The main configuration adds no further internal narrowing. `local-width.yaml` provides a separate width ablation. All 14 residual blocks, middle attention, original operators, and upsampling order are retained. Residual projections, normalization, and adjacent convolution dimensions are adjusted together as required by the widths.

For the first three upsampling stages, Wan's `DupUp3D` requires `output_channels * factor / input_channels` to be an integer, with factors 8, 8, and 4, respectively. Invalid configurations raise an error. When a width change prevents an identity shortcut from matching the residual output, the original Wan class creates a 1x1x1 projection.

Three initialization strategies are available:

- `random`: randomly initialize the student decoder. The latent projection and normalization are copied from the original model and frozen.
- `teacher_prefix`: take the leading teacher channels along each dimension, preserving Q/K/V groups and temporal rearrangement groups. New projections use a rectangular identity initialization. **This is a reproducible warm start, not importance-based pruning or a function-preserving transformation.**
- `checkpoint`: load an exported student with the same architecture, or EMA weights from a training checkpoint, to start a new training plan. Use `--resume` for full training continuation instead.

Inspect the architecture and parameter count without loading weights:

```bash
export WAN22_SOURCE=/absolute/path/to/Wan2.2
python -m decoder_speedup inspect configs/wan22/original.yaml
python -m decoder_speedup inspect configs/wan22/local-width.yaml
```

## Data

Each video JSONL record must contain at least:

```json
{"path":"videos/VidGen_video_174/JynVt9nDUdM-Scene-0044.mp4","source_id":"JynVt9nDUdM"}
```

VidGen filenames allow extraction of the original YouTube video ID. Clips from the same source video belong to only one split. If the download root contains `videos/`, scanning includes only completed files moved into that directory and excludes files still being extracted. No additional quality filtering is applied.

```bash
export VIDGEN_ROOT=/absolute/path/to/vidgen-1m
export MANIFEST_DIR=/absolute/path/to/manifests/experiment-001
python -m decoder_speedup manifest --root "$VIDGEN_ROOT" --output "$MANIFEST_DIR" \
  --limit 10000 --seed 42 --validation-fraction 0.05
```

The 10,000-video limit illustrates how to freeze a subset; it is not a recommendation for the final dataset size. Omit `--limit` to use all videos visible at manifest creation. Later downloads do not change a frozen manifest. For other naming conventions, supply `--input-manifest` with explicit `source_id` values.

Training samples fixed-length clips, resizes while preserving aspect ratio, and applies consistent crops/flips across each clip. Validation uses centered sampling. Corrupt or short videos raise an error naming the file; **they are not silently skipped**. Loading is currently synchronous to preserve reproducible continuation; larger-scale throughput optimization requires measurement.

Some VidGen MP4 headers count frames marked for discard, exceeding the number of decodable frames. After a full decoding check, a manifest can store `decoded_frames` as a verified positive integer. The reader uses this value for sampling to avoid seeking past the actual end. Verified manifests are hashed and remain fixed during training.

## Training, resuming, and the second stage

```bash
export WAN22_WEIGHTS=/absolute/path/to/Wan2.2_VAE.pth
export RUN_DIR=/absolute/path/to/runs/wan22-width-001
python -m decoder_speedup inspect configs/wan22/width.yaml
# Check GPU availability and your configuration before starting training.
CUDA_VISIBLE_DEVICES=0 python -m decoder_speedup train configs/wan22/width.yaml
```

Relative paths are resolved against the configuration file's directory. Referenced environment variables must exist. Unknown fields and duplicate YAML keys raise errors.

The student loss is `L1 + lambda_p * LPIPS + lambda_f * feature_MSE`, supervised by the original video. The frozen teacher provides latents and intermediate features. Trainable 1x1x1 projections align student features with teacher channel counts; these projections are excluded from exports. No ineffective KL gradient is added for the frozen encoder.

`reconstruction_updates` counts first-stage G optimizer updates; `adversarial_updates` counts subsequent GAN-stage G updates. Their sum is the hard stopping budget. `accumulation` specifies the global microbatch count per G update; with four GPUs, each processes one quarter of those microbatches, preserving the effective batch size. During GAN training, each G update is followed by `discriminator_updates` D updates. Each D update averages losses over the N real/generated microbatches. D uses hinge loss; G uses `-D(fake)`. An optional adaptive factor scales the GAN weight using gradient norms at the final RGB convolution.

The main configuration follows [Turbo's public training script](docs/TURBO_RECIPE.md): constant G/D learning rates of 1e-4, batch size 1, accumulation 8, and a 100-epoch reconstruction budget. The fixed manifest determines the actual G update count, replacing the earlier 20,000-update placeholder. GAN is initially disabled.

```bash
# Inspect the budget without loading a model or starting training.
python -m decoder_speedup plan configs/wan22/width.yaml
# Train to an inspection checkpoint while preserving the full budget.
python -m decoder_speedup train configs/wan22/width.yaml --stop-after-updates 1000
# Resume with optimizer, EMA, data cursor, and W&B run state intact.
python -m decoder_speedup train "$RUN_DIR/config.resolved.yaml" --resume "$RUN_DIR/last.pt"
# After reconstruction stabilizes, specify an additional GAN update budget.
python -m decoder_speedup train "$RUN_DIR/config.resolved.yaml" \
  --resume "$RUN_DIR/last.pt" --start-gan-updates "$GAN_UPDATES"
```

`GAN_UPDATES` must be an explicit positive integer; no unvalidated default is supplied. Use `--extend-reconstruction-updates` to extend reconstruction before GAN training begins. After a stage change, resume with the output directory's `config.resolved.yaml`. Ordinary resume strictly checks the recipe. Explicit transition options permit changes only to the corresponding budget fields; model, data, and other training settings remain checked.

Full checkpoints contain the student, feature projections, G/D optimizers, discriminator, EMA, update counters, random states, data cursor, training plan, and monitoring window. The learning rate is constant, with no warmup or decay. See [TURBO_RECIPE.md](docs/TURBO_RECIPE.md) for sources, implementation differences, and stage transitions.

Training outputs include the resolved configuration, source hashes, per-update JSONL logs, checkpoints, per-clip evaluations, and a final EMA student. Recovery depends on initialization, architecture, data, and budget. [Training methods and limitations](docs/TRAINING.md) explains the relationship to Turbo.

## W&B monitoring

The main configuration enables [miaoyin-uta/vae-speedup](https://wandb.ai/miaoyin-uta/vae-speedup). Use an existing login or `WANDB_API_KEY`; keep credentials out of configuration files.

```yaml
wandb:
  enabled: true
  entity: miaoyin-uta
  project: vae-speedup
  mode: online
  group: wan22-width-recovery
  tags: [data:vidgen-1m]
  log_every: 50
  system_sample_seconds: 30
```

- Aggregate loss means/maxima, gradients, learning rates, progress, and update times every 50 G parameter updates. Additional records cover the first update, stage boundaries, and completion. Training curves use optimizer updates rather than elapsed time.
- Validate every 1,000 updates by default, recording teacher/EMA-student PSNR, SSIM, LPIPS, and their differences. SDK system metrics are sampled every 30 seconds.
- Generate tags from the model, width configuration, initialization, stage, and job purpose. Store numerical hyperparameters and dataset hashes in the structured config.
- Organize metrics under `train`, `gan`, `quality`, `optim`, `progress`, `timing`, and `monitor`; standalone benchmarks use `decode`.
- Online resume preserves the run ID and pending aggregation window. Local JSONL retains every update. Videos, weights, source code, and console output are not uploaded by default.
- Standalone `evaluate` and `benchmark` save local JSON by default. Explicit `--log-wandb` creates a separate run after completion, under `wan22-quality-eval` or `wan22-decode-benchmark`, linked by the student SHA256 and source training run. The SDK starts only after benchmarking finishes.
- If a checkpoint predates cloud logs, use `--wandb-log-after-update N` to retain the original run and history, suppress duplicate replayed training/validation points through N, and then resume logging at the fixed update intervals.

See [monitoring conventions](docs/MONITORING.md) for groups, metric definitions, aggregation boundaries, and overhead controls. Set `wandb.enabled: false` to disable monitoring or `mode: offline` to retain local SDK logs only. Check connectivity without loading a model or training:

```bash
decoder-speedup wandb-check --entity miaoyin-uta --project vae-speedup
```

## Evaluation and export

```bash
python -m decoder_speedup export /absolute/path/to/last.pt \
  --source "$WAN22_SOURCE" --output /absolute/path/to/student.pt
python -m decoder_speedup evaluate configs/wan22/width.yaml \
  --student /absolute/path/to/student.pt --output /absolute/path/to/quality.json
python -m decoder_speedup benchmark configs/wan22/width.yaml \
  --student /absolute/path/to/student.pt --output /absolute/path/to/timing.json
```

`evaluate` compares teacher and student PSNR, SSIM, and optional LPIPS on the same validation videos. Training uses unclamped RGB; evaluation clamps to `[-1,1]`. The implementation records the SSIM definition and aggregation method.

`benchmark` measures three cases on the same device and inputs: the original architecture with reference execution settings, the original architecture with the selected execution settings, and the student with those same settings. Timing covers complete VAE decoding, excluding encoding, video I/O, and data transfer. Results include warmup, individual timings, medians, percentiles, and additional peak memory. Speedup factors and latency reductions are reported separately, including the student's gain over the execution-optimized teacher. Timing uses fixed random latents; quality evaluation uses real videos separately.

`runtime.precision` selects FP32 or BF16 AMP; `weight_dtype` selects FP32 or BF16 inference weights. Training retains FP32 master weights. `channels_last` independently controls convolution weight layout. `compile` is an **experimental inference option** using `torch.compile` with graph breaks allowed; compatibility and speedup depend on the Wan source and hardware. Initial training and quality evaluation paths do not compile the model. `runtime.winograd` selects the separate inference adapter documented in [WIDTH_WINOGRAD.md](docs/WIDTH_WINOGRAD.md); it requires BF16 AMP, FP32 stored weights, CUDA, and compilation disabled.

Deployment requires this framework and the matching Wan source version, but no teacher weights:

```python
import torch
from decoder_speedup.export import load_student

student, metadata = load_student("/path/student.pt", "/path/Wan2.2", "cuda:0")
with torch.inference_mode():
    rgb, _ = student(normalized_wan_latent)
    rgb = rgb.float().clamp(-1, 1)
```

The project and command name is `decoder-speedup`; the Python package is `decoder_speedup`. Student exports and training files created before the project rename remain readable. Strict resume continues to check configurations and source fingerprints.

Exports include the student architecture, weights, latent normalization, source fingerprints, and output convention. Loading checks the Wan source file hashes. Each call resets its causal cache; within a call, decoding proceeds one latent frame at a time and preserves gradients across chunks.

## Validation and limitations

```bash
WAN22_SOURCE=/path/to/Wan2.2 CUDA_VISIBLE_DEVICES='' python -m pytest -q
```

Tests cover invalid architectures, original-model equivalence, causality/caches, backward propagation with nonuniform and internal widths, grouped initialization, source-video isolation, video reading, G/D updates, exact training continuation, EMA, the CLI training-to-export flow, student reload, and source checks.

`examples/verify_real_weights.py` checks integration with official weights using small synthetic inputs: original-architecture equivalence, student backpropagation, and export/reload consistency, with zero optimizer updates. Initial validation records are in [docs/VALIDATION.md](docs/VALIDATION.md).

Not yet implemented: depth reduction, automatic width search, AMD's full student architecture, gradient checkpointing, or pre-encoded latent caching. Parameter reductions alone do not establish speedup, and functional integration tests do not establish recovered image quality.

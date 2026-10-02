# Mapping Turbo's public configuration to this framework

The selected recipe follows Turbo-VAED's [train.sh](https://github.com/hustvl/Turbo-VAED/blob/6bd3adf679f140abb7d87fe183ae69e54bf64a7f/train.sh), pinned to revision `6bd3adf679f140abb7d87fe183ae69e54bf64a7f`. It does not mix in different batch sizes reported in the paper or GitHub issues.

## Main configuration

| Setting | Main configuration | Source or explanation |
| --- | --- | --- |
| Student architecture | Wan2.2 with AMD stage widths `[512,512,256,64,32]`, width reduction only | This project's selected architecture, not Turbo's student architecture |
| Initialization | `teacher_prefix` | Retains the selected Wan channel-slicing initialization; Turbo's script initializes its student randomly |
| Training clips | 17 frames, 256x256, stride 1 | Turbo's public script |
| Single-GPU batch / gradient accumulation | **1 / 8**, effective reconstruction batch **8** | Turbo's public single-GPU script; not batch 32 from the paper or 16 from a separate author experiment |
| Learning rate | **1e-4** for G and D, `lr_schedule: constant` | No warmup or decay |
| AdamW | betas=[0.9,0.95], weight_decay=1e-4, eps=1e-15 | Turbo's public script |
| EMA | 0.999 | Turbo's public script |
| Reconstruction budget | **100 epochs** | The public script's upper budget, not a validated convergence point for this student |
| Reconstruction losses | L1 / VGG LPIPS / feature MSE, each with coefficient 1 | Turbo's training approach with the current Wan adapter; features at middle and upsamples.0 |
| GAN | Initially disabled; weight 0.05 times the adaptive factor when enabled | Start with reconstruction and switch explicitly after metrics stabilize; no guessed universal switching update |

AdamW epsilon is adopted as published, with no claim that it outperforms 1e-8. At the initial recipe implementation, real Wan recovery training had not yet run; the subsequent run is documented in the [experiment archive](../experiments/20261001_wan22_width_turbo/README.md).

The framework retains these explicit differences: BF16 AMP with FP32 master weights (the upstream launch script uses FP32), G gradient clipping at 1.0, Wan-specific initialization and projection heads, a small 3D discriminator, deterministic mean latents, G/D update semantics, and monitoring every 50 G updates with validation every 1,000. Current periodic validation uses 16 clips, compared with 490 in the upstream example. These settings do not constitute a full reproduction of the authors' results.

During GAN training, this framework accumulates eight microbatches per G update, then updates D. Upstream interleaves alternating G/D iterations with accumulation counters, changing the effective G sample count. This framework preserves explicit optimizer-update semantics, adopting the hyperparameters and staged approach without claiming identical update sequences.

## Converting 100 epochs to updates

Once the training manifest is fixed:

`reconstruction_G_updates = ceil(training_samples * 100 / (batch_size * global_accumulation))`

The current effective batch size is 8. A manifest containing exactly 10,000 training videos gives **125,000 G updates**, rather than 20,000. Downloading the full dataset does not automatically include it in training; only the selected manifest is counted.

The data stream continues across epoch boundaries, and the budget rounds up to a complete G update. With the current settings, at most seven extra clips may be sampled. Videos are not dropped to make the manifest evenly divisible. Original epochs, sample count, effective batch size, converted budget, and extra sample count are recorded in `training-plan.json`, checkpoints, and W&B config. The resolved configuration contains explicit update counts.

Inspect the budget before loading models:

```bash
python -m decoder_speedup plan configs/wan22/width.yaml
```

This command reads manifests only. It does not decode videos, use a GPU, initialize W&B, or start training. Path environment variables must still be set. Update-based configuration remains available for compatibility with older configurations; the main recipe no longer uses the 20,000-update placeholder.

## Running training stages

The following commands start training. Initial interface validation used CPU regression tests rather than these production commands.

1. Start reconstruction. An independent inspection stop can end a run without changing the 100-epoch budget:

```bash
python -m decoder_speedup train configs/wan22/width.yaml --stop-after-updates 1000
```

The value 1,000 is an example inspection point, not a new convergence budget. At that point, the framework saves full state, evaluates, exports the current EMA, and records `training_complete=false` in W&B. Resume without the stop option, or set a later **absolute** G update count:

```bash
python -m decoder_speedup train "$RUN_DIR/config.resolved.yaml" --resume "$RUN_DIR/last.pt"
```

2. After validation reconstruction stabilizes, explicitly set the G update budget for the GAN stage:

```bash
python -m decoder_speedup train "$RUN_DIR/config.resolved.yaml" \
  --resume "$RUN_DIR/last.pt" --start-gan-updates "$GAN_UPDATES"
```

Set `GAN_UPDATES` to a positive integer. The authors did not publish a universally applicable second-stage budget, so the framework does not invent one. The switch occurs at the checkpoint's completed G update count without waiting for the remaining reconstruction budget.

The transition preserves current student weights, feature projections, G optimizer state, EMA, data position, counters, W&B run, and historical best metrics. On first activation, D is initialized from the checkpoint's RNG state; an existing D state with no updates is retained. Later ordinary resume restores the D optimizer and random state without rebuilding D. Loading an EMA export alone is not full training continuation.

Stage changes are recorded in `training-plan.json` under `changes`, and the output directory's `config.resolved.yaml` is overwritten with the new plan. Use that resolved configuration for later resume. Ordinary strict resume rejects mixing the original epoch recipe with the new stage budget.

3. If reconstruction continues improving at the budget limit, explicitly extend the budget:

```bash
python -m decoder_speedup train "$RUN_DIR/config.resolved.yaml" \
  --resume "$RUN_DIR/last.pt" --extend-reconstruction-updates "$EXTRA_UPDATES"
```

This value is added to the original reconstruction budget, not counted from the checkpoint position. The option accepts only reconstruction plans that have neither begun GAN training nor scheduled a GAN budget. The learning rate stays at 1e-4; optimizers and EMA are not reset.

## Constraints and validation

Ordinary resume requires matching training settings and model/data provenance. Explicit stage transitions and extensions relax only the corresponding budget fields. They cannot change widths, learning rates, losses, initialization, data, or sources. Reconstruction transition options cannot be reused after GAN training has begun.

The switching criterion follows the [Turbo author's guidance](https://github.com/hustvl/Turbo-VAED/issues/7#issuecomment-3305673620): observe reconstruction loss and validation metrics stabilizing before enabling GAN. GAN may reduce PSNR/SSIM while improving LPIPS. Select weights according to the target metrics rather than total loss alone.

CPU tests cover budget conversion/rounding, adoption of the selected upstream recipe, preservation of G state across stage transitions, exact GAN checkpoint continuation, budget extensions, rejection of unrelated recipe changes, inspection stops and W&B state, legacy checkpoint compatibility, and the plan command avoiding model or SDK initialization.

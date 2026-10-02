# Training methods and reproducibility limits

This version applies AMD v1/v3 stage widths `[512,512,256,64,32]` to the original Wan2.2 decoder, with width reduction and recovery training only. See [AMD_WIDTH.md](AMD_WIDTH.md) for the source. It does not implement AMD's full student architecture or claim to reproduce the final AMD/Turbo models.

## Current training design

- Freeze the original Wan encoder and decoder. The encoder produces deterministic mean latents and is not trained.
- Preserve the 48-channel latent interface, normalization, 16x spatial factor, and 4x temporal factor.
- Supervise student RGB against the original video using L1, per-frame VGG LPIPS, and MSE on selected intermediate features.
- Use trainable projections from student to teacher channel counts when feature widths differ. Temporal and spatial dimensions must match.
- Support an optional second GAN stage. G retains reconstruction, perceptual, and feature losses and adds an adversarial term. D is a 3D PatchGAN trained with hinge loss.
- Compute the adaptive adversarial weight as `gan * clamp(||d(L1+LPIPS)/dW_last|| / (||dL_GAN/dW_last|| + 1e-4), 0, 1e4)`. L1 and LPIPS already include their configured coefficients; the feature loss is excluded from the numerator.
- Use BF16 AMP with FP32 student master weights and optimizer state. LPIPS runs internally in FP32. AdamW uses a constant learning rate; EMA defaults to 0.999.

## Relationship to Turbo's public code

The frozen teacher, original-video L1, VGG LPIPS, feature supervision, and later GAN stage follow the approach in [Turbo-VAED](https://github.com/hustvl/Turbo-VAED). The reviewed revision is `6bd3adf679f140abb7d87fe183ae69e54bf64a7f`.

These choices do not constitute AMD's complete training recipe, and Turbo's code does not establish the optimal width, dataset size, update count, or learning rate for this student. This framework makes explicit implementation choices for projection heads, discriminator architecture, data flow, and G/D updates. It does not copy the upstream counting scheme that interleaves microbatch steps, alternating G/D iterations, and gradient accumulation.

The main configuration adopts the public Turbo `train.sh` settings: constant LR 1e-4, batch size 1, accumulation 8, 100 epochs, AdamW epsilon 1e-15, and EMA 0.999. The fixed training manifest determines the update count for 100 epochs; this is a budget, not a validated convergence point. GAN is initially disabled and enabled explicitly based on validation. See [TURBO_RECIPE.md](TURBO_RECIPE.md) for sources and implementation differences.

## Initialization and architecture

`teacher_prefix` takes leading teacher channels while preserving Q/K/V and temporal expansion groups. The student rebuilds RMS normalization with the appropriate channel counts and scaling. This is neither SVD nor channel-importance ranking, and it does not preserve the original function exactly. In particular, width changes alter the rearrangement semantics of upsampling shortcuts, which recovery training must accommodate.

The default model retains all 14 residual blocks, attention, activations, and convolution operators. Width-dependent additions or changes to 1x1 residual projections are recorded in the student architecture and initialization report. The network layout therefore cannot be described as strictly unchanged for arbitrary widths.

## Data and checkpoints

Split training and validation by original-video `source_id`. File manifests and SHA256 hashes are fixed per experiment; later downloads do not change the sample set. No additional VidGen visual-quality filtering is performed. Loading verifies that a video supports the requested clip length and raises an error otherwise.

Each process reads its assigned videos synchronously, without data-worker prefetching. Multiple GPUs share a global sample order and skip entries assigned to other ranks, so the saved cursor identifies the next sample to read. Augmentation is determined by the seed, epoch, and sample index. Checkpoints are written at complete G/D update boundaries and contain all state required for continuation. CPU regression tests compare resumed parameters, optimizers, and EMA with uninterrupted execution. Bitwise equivalence is not promised across GPUs, PyTorch versions, or nondeterministic CUDA kernels.

Ordinary full resume requires the same training plan. Explicit `--start-gan-updates` and `--extend-reconstruction-updates` options allow stage-budget changes while preserving the G optimizer, EMA, data cursor, and run. Stage changes are recorded in the training-plan history. Changes to widths, losses, or other recipe settings require a new experiment and cannot bypass recipe/source checks. Deployment excludes the discriminator and feature projections.

## Online monitoring

The main configuration enables W&B project `miaoyin-uta/vae-speedup`. Full training checkpoints store the run ID, and online resume continues that run. Disabling monitoring or changing its logging interval does not count as a training-recipe change. Changing the W&B project/entity raises an explicit error to prevent resuming into the wrong destination. Other configuration, model, and data-source checks remain in force.

Training means/maxima are logged every 50 G updates, validation defaults to every 1,000 updates, and system metrics are sampled every 30 seconds. Training logs are not triggered by elapsed time. Checkpoints preserve the aggregation window. See [monitoring conventions](MONITORING.md).

See [DISTRIBUTED.md](DISTRIBUTED.md) for multi-GPU execution, global batch counting, and checkpoint migration.

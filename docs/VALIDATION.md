# Initial validation record

Date: 2026-09-30 (America/Chicago). This record covers functional integration; real-data recovery training had not started at that point. Later dated entries describe subsequent checks.

## Automated tests

All 18 initial tests passed:

- ada0: Python 3.12 / PyTorch 2.11.0+cu130, running on CPU.
- erebus: Python 3.10 / PyTorch 2.7.1+cu128, running on CPU.
- Full-checkpoint test: a tiny Wan student trains first with reconstruction and then GAN, accumulating two batches per G update and making two D updates per G update. Resumed student, projection heads, D, G/D optimizers, EMA, and data position match uninterrupted execution individually.
- CLI integration: read temporary synthetic videos, train for two updates, evaluate, save a full checkpoint, export EMA, reload, independently evaluate quality, and benchmark three execution cases.
- Other coverage includes original-architecture equivalence, causality, repeated calls, nonuniform/internal widths, temporal backpropagation, initialization groups, source-video splits, exclusion of unfinished extraction directories, invalid configurations, and export source-fingerprint checks.

Parameter updates in these tests are limited to tiny models and synthetic inputs. They validate framework behavior rather than formal recovery results.

## Integration with official weights

After confirming availability, the checks used erebus GPU 0: RTX PRO 6000 Blackwell Max-Q 96GB, PyTorch 2.7.1+cu128.

- Wan weight SHA256: `20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36`.
- VAE source SHA256: `2234634cfd557e020083b92f36a43c4ed69b95e6dcc7e9cbb11ccb2eaf8dc6b4`.
- Wan fork commit on ada0: `ca724575ae721ac84639c729bc07dbe2428a49de`. Erebus used an existing synchronized working copy whose Git HEAD remained `1ea34ff...` with a dirty tree. Individual relevant source hashes matched ada0. That copy was not reset or modified to conceal the difference.
- With synthetic 9-frame 32x32 input, FP32 weights, and BF16 AMP, the original-width wrapper had maximum absolute output error **0** against the original code.
- A nonuniform student with widths `[512,512,512,256,64]`, internal width 128 at `upsamples.2.upsamples.0`, and internal width 32 at `upsamples.3.upsamples.0` produced `[1,3,9,32,32]` output. Backward gradients were finite and the teacher had no gradients.
- Exporting and reloading that student produced maximum absolute error **0**.
- The real official model received **0 optimizer updates**. A single backward check did not establish training, quality, or speed results.

The actual LPIPS module was also checked on erebus with cached pretrained VGG weights: two synthetic 32x32 frames produced finite, nonzero input gradients, while LPIPS parameters had no gradients. No weights were downloaded again.

## Configuration inspection

At initial commit `3d8a9e2`, `inspect` counted parameters on the meta device: 555,049,228 in the original decoder and 130,297,996 (23.475%) in the then-example `width.yaml`. That custom width configuration was subsequently replaced with the AMD widths below at the user's request. Counts exclude the original encoder, frozen latent projection, and training feature projections. These example counts are neither measured speedups nor an optimal compression configuration.

## Experiments pending at initial validation

- Quality convergence, memory/throughput, and GAN-stage benefits during real VidGen training.
- Selection of stage widths, initialization, and training budgets.
- Decoding timings and quality curves for the original, execution-optimized, and student models on the same GPU and inputs.
- Benefits of full BF16 weights, channels-last, and compilation for specific configurations; compilation remains experimental.
- Multiple GPUs, depth reduction, automatic search, and other backbones.

Raw JSON is stored in the separate research archive under `results/20260930_decoder_compress/`. Large weights, datasets, and machine-private environments are excluded from the framework repository.

## AMD width configuration validation (2026-09-30)

The main configuration changed to `[512,512,256,64,32]` with `hidden: {}`, using the published widths shared by AMD v1/v3. All 19 CPU tests passed on ada0 and erebus, including a new regression for configuration provenance, 14 residual blocks, attention, regular convolutions, and original upsampling.

The exact configuration was checked on available erebus GPU 0 using real Wan weights and synthetic 9-frame 32x32 input: original-width alignment error was 0; student gradients were finite; the teacher had no gradients; export/reload error was 0. The new 1x1x1 shortcut projection at `upsamples.1.upsamples.0` was recorded explicitly in the initialization report. The residual-block count was unchanged. The real model still received zero optimizer updates, with no recovery-quality or speed conclusion. Raw results are in the research archive at `results/20260930_decoder_compress/amd_width_real_weights.json`.

The width-only decoder has 91,198,892 parameters (16.431% of the original). This differs from AMD's complete student because this version retains Wan's convolutions, attention, depth, and upsampling. Parameter counts do not establish AMD-equivalent speed or quality.

## W&B integration validation (2026-09-30)

All 25 CPU tests passed on ada0 and erebus. New coverage includes no SDK initialization when disabled, exclusion of keys/private paths from uploaded config, actual G/D update axes, run-ID checkpoint/resume, failure status, segmented offline resume, and connectivity checks producing no training metrics.

Using erebus's existing credentials and W&B SDK 0.23.1, a [connection-check](https://wandb.ai/miaoyin-uta/vae-speedup/runs/50728c97) run was created and completed in `miaoyin-uta/vae-speedup`. It uploaded connection success and zero optimizer updates only. No model was loaded and no GPU or formal recovery training was run. Monitoring integration did not change the AMD width configuration.

## Fixed-update monitoring validation (2026-09-30)

All 31 CPU tests passed on ada0 and erebus. Coverage includes absolute 50-update logging boundaries independent of update duration, window means/maxima, resuming a checkpoint at update 37 into the window ending at 50, separate reconstruction/GAN windows, exception tails, tags derived from actual configuration, best-metric directions, and source-run association. CLI tests confirmed no W&B initialization during three-case decode timing and no duplicate final validation/evaluation logs.

An offline CPU check on erebus used real W&B SDK 0.23.1 and synthetic scalars for 1,000 simulated updates. Per-update logging made 1,000 calls; logging every 50 updates made 21 calls, including the first-update preview. Cumulative SDK log-call times were 0.2369 and 0.0051 seconds, respectively. These figures exclude initialization, finalization, and background system sampling. They are not production-training or online-network performance results. No model or GPU was used, and formal recovery training had not started. Raw results are in the research archive at `results/20260930_decoder_compress/monitoring_sdk.json`.

## Turbo recipe and stage continuation validation (2026-09-30)

The main configuration adopted Turbo's public `train.sh` settings: constant LR 1e-4, single-GPU batch size 1 with accumulation 8, 100 epochs, and AdamW epsilon 1e-15. All 38 CPU tests passed on both machines. Checks covered conversion/rounding from fixed sample counts to updates, inspection stops preserving full budgets, GAN transitions preserving G parameters/optimizer/EMA/cursor/run, exact GAN checkpoint continuation, explicit budget changes with unrelated LR/loss changes rejected, and compatibility with older configuration fields. CLI integration covered reconstruction checkpoint, GAN activation, inspection stop, and resume with the resolved configuration through budget completion. The plan command was checked to avoid model loading or W&B initialization.

These updates occurred only in CPU regression tests with tiny models and synthetic videos. At that point, formal Wan recovery, memory at real training sizes, and training quality had not been validated. The adopted 100-epoch public budget does not establish sufficient recovery for this student. See TURBO_RECIPE.md for sources and initialization/precision differences.

## Decodable frame-count regression (2026-10-01)

After the user authorized the first formal run with 10,000 VidGen videos, data checks found that some MP4 header counts include discarded frames and exceed the actual decodable count. Manifests can now store verified `decoded_frames`, which the reader uses for training/validation sampling. A new regression simulates a header reporting 2,000 frames for a file with only 20 decodable frames, checks random/centered 17-frame sampling, and rejects invalid counts. All 39 CPU tests passed on both machines. Student architecture, optimizer, and losses were unchanged.

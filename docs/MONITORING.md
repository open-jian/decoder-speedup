# Training monitoring conventions

The project is [miaoyin-uta/vae-speedup](https://wandb.ai/miaoyin-uta/vae-speedup). Monitoring does not change training budgets, optimizer updates, or validation samples. This document specifies logging behavior, not a validated training recipe.

## Experiment organization

| Level | Current convention | Purpose |
| --- | --- | --- |
| Project | `vae-speedup` | All decoder acceleration experiments, including future models |
| Group | `wan22-width-recovery` | Experiments addressing the same research question; use separate groups for depth reduction or operator replacement |
| Job type | `decoder-recovery` / `quality-eval` / `decode-benchmark` / `connection-check` | Training, standalone evaluation, standalone timing, and connectivity checks |
| Run | One training plan | Online resume retains the ID; changed widths/losses require a new run; explicit stage transitions or budget extensions can retain the run and record history |
| Name | `wan22-w512-512-256-64-32-teacher_prefix-rec-s42-<id>` | Identifies model, widths, initialization, stage, and seed; the final ID avoids duplicate names |

Automatic tags: `model:wan22`, `method:width-only`, `width:amd-v1-v3`, `init:teacher_prefix`, `recipe:rec`, and `purpose:decoder-recovery`. The main configuration adds `data:vidgen-1m`.

- `recipe` is derived from the actual budget: `rec` for reconstruction only, `rec-gan` for two stages, and `gan` for GAN only.
- `width:amd-v1-v3` applies only to `[512,512,256,64,32]` without additional internal narrowing. It identifies the source of the width values.
- Original-width controls receive `method:original`. Code owns automatic tag namespaces to prevent manually supplied tags from contradicting the architecture.
- Store learning rates, seeds, batches, accumulation, loss coefficients, dimensions, update counts, precision, and manifest hashes in structured config rather than encoding everything as tags.
- A group does not guarantee direct comparability. Quality comparisons require matching teacher weights, validation manifests, actual clip counts, dimensions/frame counts, and metric definitions. Timing also requires matching hardware, precision, execution options, and warmup/repeat counts.

## Logging frequency

| Content | Default frequency | Meaning |
| --- | --- | --- |
| Training losses, GAN, gradients, LR, progress, update time | **Every 50 G parameter updates** | Absolute updates 50, 100, 150, etc., independent of training speed |
| Teacher/student quality | **Every 1,000 G updates and at completion** | First 16 clips of the fixed validation manifest, using student EMA; reuse results if completion coincides with validation |
| SDK GPU/CPU/memory metrics | **Every 30 seconds** | Displayed by wall time, independently of training updates |
| Decode timing | Explicit `benchmark` command | Local JSON by default; `--log-wandb` uploads to a separate run after timing finishes |
| Weight checkpoints | Every 1,000 updates, reconstruction-stage end, and training end | Saved locally; not uploaded to W&B by default |
| Videos, images, gradient histograms, model graphs | Disabled by default | Start with scalar metrics; add a small fixed sample set at sparse intervals if visual inspection is needed |

One step means **one actual G optimizer update**, not one microbatch or one D update. All training and quality curves use `progress/generator_updates` as the x-axis. W&B's own `_step` is only a log-record index.

Each training window computes **means and maxima** across all updates; total loss also records its final value. Use means for the main curves and maxima to inspect spikes. Local `train.jsonl` retains every update; terminal output is emitted every 50 updates.

The first update and each stage's first update receive an additional preview without being removed from the aggregation window: the record at update 50 still covers updates 1 through 50. Stage endings, training completion, and catchable exceptions flush partial windows with their start, actual update count, and reason. A stage transition clears the window so reconstruction and GAN losses are not averaged together. Regular logging points remain multiples of 50.

A single-stage 120-update run therefore logs at **1, 50, 100, and 120**. The first record is a preview; subsequent windows contain 50, 50, and 20 updates. No elapsed-time trigger is used for training logs.

## Metric namespaces

| Namespace | Contents |
| --- | --- |
| `train/*` | Means/maxima for total loss, RGB L1, LPIPS, and feature MSE; final total loss |
| `gan/*` | Enabled state, G adversarial loss, effective GAN weight, D loss; multiple D updates are averaged |
| `quality/*` | Teacher and EMA-student PSNR, SSIM, LPIPS; student-minus-teacher differences; actual clip count and weight type |
| `optim/*` | G/D learning rates and pre-clipping gradient-norm means/maxima; norms across multiple D updates are averaged first |
| `progress/*` | G/D update counts, batches read, epoch, sample cursor, stage, and budget completion fraction |
| `timing/*` | Training-update mean/maximum duration and validation duration; training timing includes data loading and forward/backward passes, not just decoding |
| `monitor/*` | Window start, window update count, logging reason, SDK log-call count and duration |
| `decode/*` | Decode median/P10/P90, memory, speedup factor, and latency reduction for original, execution-optimized teacher, and student |
| SDK system panels | GPU utilization/memory and CPU/memory metrics for diagnosing loading or resource bottlenecks over time |

Training component metrics are raw losses; coefficients are stored in config. Total loss is weighted. Total losses from different stages are not directly interchangeable quality metrics.

`best/psnr_db` and `best/ssim` track maxima; `best/lpips` tracks the minimum. Each stores its corresponding update count. These values may come from different checkpoints and **do not imply that one set of weights achieves all three best values**. The exported `student.pt` remains the final EMA, not an automatically selected best checkpoint.

Standalone evaluation/benchmarking does not initialize W&B by default, even when using a training configuration with monitoring enabled. Explicit `--log-wandb` uses groups `wan22-quality-eval` and `wan22-decode-benchmark`, respectively. New runs contain the student file SHA256 and source training run ID; they do not resume or write to the training run. Periodic in-training validation still writes to the original training run. Uploaded width tags come from the actual student artifact rather than being inferred from the benchmark configuration. Speedup factors and latency reductions are stored separately: for example, 20x speedup means a 95% latency reduction.

## Controlling overhead

1. Accumulate existing CPU scalars each update and combine them into one `run.log({...})` call every 50 updates. Do not call the SDK separately for each metric.
2. The trainer batches logged loss/gradient scalars into one CPU transfer per update while retaining nonfinite-value checks. The tracker does not call `.item()`, synchronize CUDA, or run additional forward passes.
3. Use the SDK's background uploader rather than an extra unbounded thread queue. Keep cloud queries, login, file uploads, `watch()`, and `finish()` out of the training hot path.
4. Initialize once and finish once. Disable source-code, Git, and console uploads. Create standalone monitoring runs only after timing completes to avoid SDK system sampling during benchmarks.
5. Record main-thread SDK log-call overhead in `monitor/sdk_log_seconds` and `monitor/sdk_log_ms_max`, and write a final summary. Curve fields ending in `*_so_far` cover calls through the previous log. They exclude initialization/finalization and total background network/system-sampling overhead, so they are not the complete monitoring cost.
6. `mode: offline` keeps local SDK logs for later `wandb sync`. Online background transfer does not guarantee zero blocking. If measured calls slow training, increase `log_every` while retaining fixed update intervals.

Logging every 50 updates produces approximately one training record containing tens of scalars per interval. A 20,000-update example yields roughly 400 regular records plus previews, validation, and boundary records. Aggregation keeps the record count independent of model speed.

## Resume and failures

- Full checkpoints store the run ID, pending aggregation window, and historical best quality values. For example, resuming a checkpoint at update 37 still aggregates updates 1 through 50 at the next boundary.
- Online resume reuses the run. Explicit GAN transitions/reconstruction extensions retain it and update the training plan; see TURBO_RECIPE.md. Offline SDK logging cannot resume in place, so it starts a new segment tagged with the original ID. Logging intervals may change while recipe checks remain strict.
- Replaying an older checkpoint does not automatically delete later cloud records. Use student weights to start a new experiment when a clean comparison is needed.
- To retain an online run whose logs extend beyond the checkpoint, first read the maximum `progress/generator_updates=N`, then use `train ... --resume ... --wandb-log-after-update N`. Replayed training/validation points through N are not uploaded. Old windows entirely within that range are cleared; aggregation restarts after N using the original absolute update boundaries. Retained cloud history may contain pre-replay values; GPU replay is not guaranteed bitwise identical. Archive original local logs and record the resume boundary without editing old cloud points. Checkpoints retain this logging boundary; local training logs still include every replayed update.
- Catchable exceptions record the failure type and attempt to flush completed updates. Power loss or SIGKILL cannot guarantee final uploads; local checkpoints and logs support recovery.
- Upload failures are not silently ignored. Local raw metrics are retained. Sensitive files, keys, original video filenames, and model weights are not uploaded.

## Reading results

In W&B, filter by `group=wan22-width-recovery` and `job_type=decoder-recovery`. First inspect the three `quality` metrics and teacher differences, then `train/gan` recovery curves and `optim` stability, followed by `timing/monitor` and system resources. For explicitly uploaded standalone timings, filter by `group=wan22-decode-benchmark` and `job_type=decode-benchmark`, and associate quality results through the source run or student SHA256.

Metric prefixes organize the sections; this document does not claim a custom web dashboard has been created. As experiments accumulate, panels and comparison-table columns can be arranged in this order.

API references: [W&B Run logging and custom axes](https://docs.wandb.ai/ref/python/experiments/run/) and [SDK system-sampling settings](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py). This framework aggregates scalars at fixed update intervals and sets system sampling through `x_stats_sampling_interval`.

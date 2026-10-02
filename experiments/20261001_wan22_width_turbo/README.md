# Experiment archive

This directory contains the preparation/launch scripts, configuration, fixed training and validation manifests, and logs/runtime snapshots captured for this experiment. Scripts retain absolute erebus paths. `prepare.py` additionally requires PyAV (av 16.0.1 for this run).

`train.sh` records the original single-GPU launch; `train-4gpu.sh` records migration to four GPUs and continuation. Do not rerun these launch scripts while an existing training job is active. See [distributed training](../../docs/DISTRIBUTED.md) for ordinary checkpoint resume.

`run-snapshot/` contains captured training output through update 25,175. Snapshot time and file SHA256 hashes are recorded in [publication.json](publication.json). Logs in this archive do not update automatically; see [W&B](https://wandb.ai/miaoyin-uta/vae-speedup/runs/85cbe9c8) for live progress. This README was translated into English on 2026-10-02; its previous and updated file records are retained in the publication manifest's documentation revision history.

## Launch status record

At 00:58 CDT on 2026-10-01, formal recovery training was moved to all four erebus GPUs at the user's request, in tmux session `wan22-width-4gpu`, with [W&B run 85cbe9c8](https://wandb.ai/miaoyin-uta/vae-speedup/runs/85cbe9c8). Each GPU used batch size 1 and local accumulation 2, preserving effective batch size 8, constant LR 1e-4, and the 100-epoch / 125,000-G-update budget. GAN remained disabled.

The single-GPU run had reached update 32 but had saved only update 1. Its original logs were archived, then unsaved updates were recomputed from update 1. Four-GPU execution saved and reloaded `checkpoint-00000040.pt`. At verification, local progress was update 61, cloud progress was update 50 with world_size 4, and all four GPU processes were running. The checkpoint contained four RNG states, 114 optimizer entries, 114 EMA tensors, and global cursor 320.

For matched early updates 5 through 32, median training-update time changed from 4.668 to 1.304 seconds: approximately 3.581x faster, or a 72.07% time reduction. This measures training throughput, not VAE decoding speed. All 43 CPU tests passed on both machines, and framework commit `4f821395` was pushed and verified. The training directory was `/data2/jian/vae-speedup/decoder-speedup/runs/wan22-amd-width-turbo-10k-20261001`, with subsequent checkpoints every 1,000 updates. Experiment scripts, manifests, and original status records are under `experiments/20261001_wan22_width_turbo/`. Do not start a duplicate job.

## Wan2.2 with AMD widths: first formal recovery run

On 2026-10-01, the user authorized training with 10,000 training videos and an independent validation set.

- Execution: erebus GPUs 0 through 3, four RTX PRO 6000 Blackwell Max-Q 96GB devices. The training session survives SSH disconnection.
- Student: stage widths `[512,512,256,64,32]`, with no depth reduction or operator replacement; initialized from leading teacher channels.
- Recipe: Turbo public-script settings, constant LR 1e-4, batch size 1 with global accumulation 8, 100 epochs, and 125,000 G updates. Reconstruction, LPIPS, and feature losses; GAN initially disabled.
- Data: fixed selection of 10,000 training and 256 validation videos from VidGen archives completely downloaded at preparation time, using seed 42. Original source IDs are separated into groups before independent random sampling. Periodic validation uses the configured fixed 16 clips. The full validation manifest is retained separately; 16-clip results do not describe the full dataset.
- Data checks: no aesthetic or visual-quality filtering. Count all decodable frames, require at least 17, and verify OpenCV reads at the beginning, middle, and end. Record failures separately and replace them using a fixed candidate order. Formal training does not silently skip failures.
- Frame-count issue: example `A9UreXYJbT4-Scene-0039.mp4` reports 387 frames in its header, with 156 packets marked discard and 231 actually decodable frames. The reader uses fixed manifest `decoded_frames` values to avoid sampling past the real end.
- Checkpoints: save full state after the first G update and resume automatically, then save every 1,000 updates. `checkpoint-00000001.pt` in the training directory is the first resumable point. `last.pt` is written only when the process exits normally; select the latest checkpoint by its saved update count.
- Logging: W&B project `miaoyin-uta/vae-speedup`, aggregated every 50 G updates; quality every 1,000 G updates. System metrics are sampled independently.

Erebus experiment directory: `/data2/jian/vae-speedup/experiments/20261001_wan22_width_turbo/`.
Training output: `/data2/jian/vae-speedup/decoder-speedup/runs/wan22-amd-width-turbo-10k-20261001/`.

`prepare.py` freezes manifests and configuration. `train.sh` resumes automatically after the first checkpoint. The training directory's `wandb-run.json` records the run link; this experiment directory's `train.exitcode` and `train.log` record exit status and console output.

The frame-count fix is decoder-speedup commit `533c363`, with all 39 CPU tests passing on ada0 and erebus. Use current logs to establish training progress, quality, and resource usage; this preparation record alone is not proof that training has started or converged.

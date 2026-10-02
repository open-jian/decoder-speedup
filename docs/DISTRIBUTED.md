# Multi-GPU recovery training

Use torchrun to launch one process per GPU. Each process holds the frozen teacher, student, and training-loss modules. Video loading and forward/backward passes are distributed across ranks. Gradients are aggregated with torch.distributed/NCCL before each complete G/D optimizer update.

`training.accumulation` always means the **global number of microbatches per update** and must be divisible by world size. With batch size 1 and accumulation 8, one GPU accumulates eight microbatches; four GPUs accumulate two each. Both use effective batch size 8, constant LR 1e-4, and the same 100-epoch / 125,000-G-update budget for the fixed 10,000-video training set. Losses are scaled by global accumulation, gradients are reduced with SUM, and clipping, optimizer updates, and EMA follow on all ranks.

```bash
python -m decoder_speedup plan configs/wan22/width.yaml --world-size 4
python -m torch.distributed.run --standalone --nproc_per_node=4 -m decoder_speedup train configs/wan22/width.yaml
python -m torch.distributed.run --standalone --nproc_per_node=4 -m decoder_speedup train "$RUN_DIR/config.resolved.yaml" --resume "$RUN_DIR/last.pt"
```

Sample order and random augmentation are tied to the global data cursor. Each rank skips other ranks' entries without reading those videos, avoiding duplicate processing of the same microbatch. Only rank 0 logs, validates, exports, and writes checkpoints; all ranks participate in synchronization and random-state collection. G/D counters, epochs, and W&B axes use global update counts.

Checkpoints include the random states of all ranks. Resuming with the same GPU count restores each rank's state. Changing the GPU count preserves weights, optimizers, EMA, sample cursor, global budget, and W&B run, and records the world-size change. Parallel reduction and random computation order may differ, so bitwise equivalence with single-GPU execution is not guaranteed after changing GPU counts.

Model, data, recipe, and framework source fingerprints are checked strictly by default. After reviewing a framework upgrade, use `--allow-framework-change` explicitly to resume with changed framework code. This option permits only framework fingerprint changes and records old/new fingerprints in the training plan. Checks on the original Wan source, weights, data, losses, learning rates, and other settings remain active. Subsequent checkpoints can resume normally.

To request a graceful stop, create `stop.request` in the output directory. All ranks finish the current G/D update, save `last.pt`, and exit without changing the budget. Remove the request file before restarting. After an abrupt termination, resume from the latest complete checkpoint.

CPU regression tests cover two-process reconstruction/GAN gradients against single-process execution, multi-GPU checkpoint resume, single-to-multi-GPU migration, rank-state agreement, four-GPU budgets, and stop requests. Actual GPU throughput gains must be measured from the corresponding training logs.

# 多卡恢复训练

使用torchrun启动每卡一个进程。每个进程持有固定老师、学生及训练损失模块；分摊视频读取和前后向，每次完整G/D更新前通过torch.distributed/NCCL汇总梯度。

`training.accumulation` 始终表示一次更新的**全局microbatch数**，必须可被world size整除。当前配置batch1、accumulation8：单卡累积8次，四卡各累积2次，有效batch均为8，固定LR1e-4、100轮/125000次G更新不变。损失按全局累积数缩放，梯度按SUM规约，随后统一裁剪、更新优化器和EMA。

```bash
python -m decoder_speedup plan configs/wan22/width.yaml --world-size 4
python -m torch.distributed.run --standalone --nproc_per_node=4 -m decoder_speedup train configs/wan22/width.yaml
python -m torch.distributed.run --standalone --nproc_per_node=4 -m decoder_speedup train "$RUN_DIR/config.resolved.yaml" --resume "$RUN_DIR/last.pt"
```

样本顺序和随机增强与全局数据游标绑定，各rank跳过其他rank的条目但不读取视频，因此不会重复训练同一条microbatch。只有rank0记录日志、运行验证、导出与写断点；全部rank参与同步和随机状态收集。G/D计数、轮数、W&B横轴都按全局更新计数。

断点含所有rank的随机状态。同卡数续训恢复各自状态；换卡数保留权重、优化器、EMA、样本游标、全局预算和W&B run，并记录world size变化。并行规约和随机计算次序可能改变，不承诺换卡数后与单卡逐位一致。

默认仍严格核对模型、数据、配方和框架源码指纹。仅经过核对的框架升级续训可以显式加 `--allow-framework-change`，它只允许框架源码指纹变化，并将新旧指纹写入训练计划；原Wan源码、权重、数据、损失、学习率等检查仍保留。后续用新断点常规续训即可。

若需正常结束当前进程，可在输出目录创建 `stop.request`。所有rank完成当前G/D更新后保存last.pt并退出，预算不变。再次启动前删除该请求文件。异常强杀仍需从最近完整断点恢复。

CPU回归包括两进程重建/GAN梯度与单卡比较、多卡断点续训、单卡到多卡迁移、rank间状态一致、四卡预算和停止请求。真实GPU是否提速以对应训练日志为准。

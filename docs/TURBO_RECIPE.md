# Turbo公开配置在本框架中的映射

用户选择采用Turbo-VAED的配置方式。本版明确采用其固定版本 `6bd3adf679f140abb7d87fe183ae69e54bf64a7f` 的 [train.sh](https://github.com/hustvl/Turbo-VAED/blob/6bd3adf679f140abb7d87fe183ae69e54bf64a7f/train.sh)，不混用论文或Issue中的另一套batch数值。

## 已落地的主配置

| 项目 | 主配置 | 来源或说明 |
| --- | --- | --- |
| 学生结构 | Wan2.2，AMD阶段宽度 `[512,512,256,64,32]`，只缩宽 | 沿用本项目已选结构，非Turbo学生架构 |
| 初始化 | `teacher_prefix` | 沿用已选的Wan权重截取初始化；Turbo脚本的学生默认随机初始化，二者不同 |
| 训练视频 | 17帧、256×256、stride1 | Turbo公开脚本 |
| 单卡batch / 梯度累积 | **1 / 8**，重建有效batch **8** | Turbo公开单卡脚本；不是论文32或作者另一实验16 |
| LR | G/D均 **1e-4**，`lr_schedule: constant` | 无warmup、无衰减 |
| AdamW | betas=[0.9,0.95]，weight_decay=1e-4，eps=1e-15 | Turbo公开脚本 |
| EMA | 0.999 | Turbo公开脚本 |
| 重建预算 | **100 epochs** | Turbo公开脚本的上限配置，不代表我们学生已验证的收敛终点 |
| 重建损失 | L1 / VGG LPIPS / 特征MSE，各系数1 | 沿用Turbo训练思路与当前Wan接线；特征位置middle、upsamples.0 |
| GAN | 初始关闭；启用后权重0.05×自适应系数 | 先重建，指标稳定后明确切换；不填写一个猜测的统一GAN切换步数 |

AdamW的epsilon是原样采用的配置值，不宣称相对1e-8更优。真实Wan训练尚未运行。

为适配当前工程，以下差异明确保留：BF16 AMP＋FP32主权重（上游启动脚本为FP32）、G梯度裁剪1.0、当前Wan初始化/投影头/小型3D判别器、确定性均值latent、G/D更新实现，以及每50个G更新记录/每1000个G更新验证的监控频率。验证目前16段，上游示例490段；不能将这份配置称为作者结果的完整复现。

GAN阶段本框架每次G更新累积8个microbatch，再更新D；上游将G/D奇偶交替和累积计数交织，G有效样本数会变化。本版保留明确的优化器更新语义，因此采用的是其超参数和分阶段方法，不宣称逐次更新完全一致。

## 100轮怎么变成更新次数

训练清单固定后：

`重建G更新预算 = ceil(训练条数 × 100 / (单卡batch × 梯度累积))`

当前单卡有效batch8。若训练清单恰好10,000条，则预算为 **125,000次G更新**，而不是2万步。下载全量数据不等于自动把全量数据加入训练；只按指定manifest计数。

数据流跨epoch连续读取，预算向上取整到完整G更新；最多多读取7段采样。不会为了整除而丢掉固定清单中的视频。原始epochs、条数、有效batch、换算结果和多出的采样数写入 `training-plan.json`、checkpoint与W&B config；解析后的配置使用明确更新数。

在加载模型前检查预算：

```bash
python -m decoder_speedup plan configs/wan22/width.yaml
```

该命令只读取清单，不解码视频、不使用GPU、不启动W&B或训练。各路径环境变量仍须先设置。API中保留旧的按updates配置方式以兼容历史配置；正式主配置不再使用2万步占位值。

## 分阶段运行

以下命令会正式训练，当前任务仅实现接口和CPU回归验证，没有执行这些命令。

1. 先按重建配置运行。可设置独立的检查停止点，不改变100轮预算：

```bash
python -m decoder_speedup train configs/wan22/width.yaml --stop-after-updates 1000
```

这里1000是一次运行的检查点示例，不是新收敛预算。到点保存完整状态、评测和当前EMA导出；W&B显示 `training_complete=false`。恢复时去掉停止参数，或将其设为更晚的**绝对**G更新数：

```bash
python -m decoder_speedup train "$RUN_DIR/config.resolved.yaml" --resume "$RUN_DIR/last.pt"
```

2. 验证集重建趋于稳定后，显式指定本次GAN阶段的G更新预算：

```bash
python -m decoder_speedup train "$RUN_DIR/config.resolved.yaml" \
  --resume "$RUN_DIR/last.pt" --start-gan-updates "$GAN_UPDATES"
```

`GAN_UPDATES`必须自行设为正整数；作者没有发布统一适用的阶段二预算，框架不猜测。此时从checkpoint已经完成的G更新处切换，不等待剩余重建预算走完。

保留学生当前权重、特征投影、G优化器状态、EMA、数据位置、计数、W&B run和历史最优值。判别器首次启用时，从断点RNG状态初始化；若已有尚未更新的D状态则继承。之后的普通续训恢复D优化器及随机状态，不重建D。不会通过只加载EMA导出文件来冒充完整续训。

阶段变更写入 `training-plan.json` 的 `changes`，并覆盖输出目录中的 `config.resolved.yaml` 为新计划。后续使用这份解析配置续训；原始epoch配方与新阶段预算不同，普通严格resume会拒绝混用。

3. 如果重建预算用完但仍在改善，显式延长上限：

```bash
python -m decoder_speedup train "$RUN_DIR/config.resolved.yaml" \
  --resume "$RUN_DIR/last.pt" --extend-reconstruction-updates "$EXTRA_UPDATES"
```

该数值加在原重建预算上，不是从checkpoint位置重新计数；仅接受尚未进入GAN、未预排GAN预算的重建计划。固定LR保持1e-4，优化器及EMA不清零。

## 约束与验证

普通resume仍要求训练配置和模型/数据来源一致。只有显式阶段切换或预算延长会放开对应预算字段，不能借此修改宽度、学习率、loss、初始化、数据或来源。GAN已经开始后不允许再次使用重建阶段切换入口。

切换依据参考[Turbo作者说明](https://github.com/hustvl/Turbo-VAED/issues/7#issuecomment-3305673620)：先观察重建损失和验证指标趋于稳定，再启用GAN。GAN可能使PSNR/SSIM下降而LPIPS改善；应按目标选择权重，不能仅按总loss判断。

CPU测试覆盖预算换算/取整、明确采用上游配置、阶段切换保留G状态、GAN断点逐项一致、预算延长、拒绝配方暗改、检查停止点及W&B状态、旧checkpoint字段兼容、plan命令不触发模型或SDK。

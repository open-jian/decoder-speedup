# 本次实验归档

包含实际使用的准备/启动脚本、配置、固定训练和验证清单，以及采集时的日志和运行快照。脚本保留erebus绝对路径；`prepare.py`另依赖PyAV（本次为av 16.0.1）。

`train.sh`记录最初的单卡启动，`train-4gpu.sh`记录单卡到四卡迁移及续训，已有训练不要重复执行这两个启动脚本。常规断点续训见 [多卡训练说明](../../docs/DISTRIBUTED.md)。

`run-snapshot/`为采集时的训练输出，最新记录到第 25175 次更新；归档时间与文件SHA256见 [publication.json](publication.json)。这里的日志不会自动更新，实时进度见 [W&B](https://wandb.ai/miaoyin-uta/vae-speedup/runs/85cbe9c8)。

# 启动时状态记录

2026-10-01 00:58 CDT 已按用户要求将正式恢复训练改为erebus全部4张GPU，tmux会话wan22-width-4gpu，W&B https://wandb.ai/miaoyin-uta/vae-speedup/runs/85cbe9c8。每卡batch1×本地累积2，有效batch8、固定LR1e-4、100轮/125000次G更新不变，GAN关闭。单卡已到32步但仅第1步有断点，原日志完整归档后从第1步重算未保存部分；四卡已保存checkpoint-00000040.pt并重新加载继续，核实时本地61步、云端50步/world_size4，4个GPU进程均在运行。checkpoint含4份RNG、优化器114项、EMA114张量、全局游标320。匹配步骤5—32初始训练更新中位耗时4.668→1.304秒，约3.581倍、耗时降72.07%；这是训练吞吐，不是VAE解码加速结论。两机43项CPU测试通过，框架4f821395已推送核验。训练目录/data2/jian/vae-speedup/decoder-speedup/runs/wan22-amd-width-turbo-10k-20261001；之后每1000步保存。实验/清单/原始状态见experiments/20261001_wan22_width_turbo/，不要重复启动。

# Wan2.2 AMD仅缩宽：首轮正式恢复训练

2026-10-01 用户授权启动训练，并确认1万条训练视频、独立验证集。

- 执行：erebus GPU0—3，共4张RTX PRO 6000 Blackwell Max-Q 96GB。训练会话独立于SSH存活。
- 学生：阶段宽度 `[512,512,256,64,32]`，不减层、不换算子；老师前缀通道初始化。
- 配方：Turbo公开脚本的固定LR1e-4、batch1×累积8、100轮、125000次G更新。重建/LPIPS/特征损失，GAN初始关闭。
- 数据：在准备时已完整下载的VidGen包中按seed42固定抽取10000训练/256验证视频，先按原始来源ID分组隔离，再分别随机抽样。周期验证沿用配置中的固定16条；完整验证清单另存，不把16条结果当成全库结论。
- 数据检查：不做美学/画质筛选；完整解码计数、最少17帧、核实OpenCV首/中/末段可读。失败条目单独记录并按固定候选顺序补足，正式训练中不静默跳过。
- 帧数问题：实查示例A9UreXYJbT4-Scene-0039.mp4头部387帧，其中156个packet标记discard，实际解码231帧。训练读取器使用清单固定的decoded_frames；避免随机抽到实际结尾之外。
- 断点：首次G更新后保存完整状态并自动续训，后续每1000次更新保存。正式训练目录中的checkpoint-00000001.pt是首次可恢复点；last.pt仅在本次进程正常结束时写入，应根据文件步数选择最近断点。
- 日志：W&B miaoyin-uta/vae-speedup，每50次G更新汇总；质量每1000次G更新。系统指标独立采样。

erebus执行目录：`/data2/jian/vae-speedup/experiments/20261001_wan22_width_turbo/`。
训练输出：`/data2/jian/vae-speedup/decoder-speedup/runs/wan22-amd-width-turbo-10k-20261001/`。

prepare.py冻结清单与配置；train.sh完成首次断点后自动续训。实际run链接见训练目录wandb-run.json，退出码见本实验目录train.exitcode，控制台见train.log。

代码修正：decoder-speedup 533c363，ada0/erebus均39项CPU测试通过。训练进度、质量和资源占用以实时日志为准，不以本准备文档作为已启动或已收敛证据。

本项目为独立的 decoder-speedup Git 仓库，父目录是研究归档，两者分别管理。
独立evaluate/benchmark默认仅保存本地JSON，用户明确要求上传时才用--log-wandb并放独立group；训练内周期验证继续写原训练run。断点落后于云端时先核对云端最大G更新数，用--wandb-log-after-update避免补跑重复点，保留原run ID与历史。
先读 README.md 和 docs/TRAINING.md；当前首版只做 Wan2.2 宽度压缩，主配置采用 AMD v1/v3 主宽度 [512,512,256,64,32]，hidden为空。
主配置不得自行换成其他宽度或叠加减层、换算子、移动上采样等改动；宽度所需的通道/归一化/旁路尺寸联动除外。
不要修改外部 Wan 原码、原始权重或共享 Python 环境。GPU 验证前实时检查占用。
2026-10-01 用户已授权开始正式恢复训练，并确认首轮使用1万条训练视频及独立验证集。按已选AMD仅缩宽与Turbo公开脚本配置，在erebus空闲GPU运行；GAN依据验证结果另行切换。
配置示例和功能测试不等于质量或速度结论；报告倍率时同时区分耗时降幅。
源码改变后运行必要的 pytest；外部源码接入测试需要 WAN22_SOURCE。
W&B主配置为miaoyin-uta/vae-speedup；使用现有登录，不将key写入代码或配置。源码同步必须排除.secrets目录。连接检查不等于授权正式训练。
训练指标按绝对G更新次数汇总记录，默认每50次更新；不得用时间触发训练日志导致采样步数漂移。只有SDK系统资源指标按30秒采样。分组、标签和指标口径见docs/MONITORING.md。
用户已选择Turbo公开train.sh（6bd3adf）的训练配置方式：G/D固定1e-4、batch1×累积8、100轮重建上限、eps1e-15；不要混入论文batch32或作者另一次实验batch16。阶段切换以验证结果为依据，用显式完整状态续训入口。详见docs/TURBO_RECIPE.md。
2026-10-01 用户要求使用erebus全部4张空闲GPU。采用torchrun数据并行，accumulation表示全局microbatch数，4卡每卡batch1×本地累积2，保持有效batch8及125000次更新；不可因卡数增加而无意把有效batch变成32。仅rank0写文件/W&B。多卡保存必须由全部rank调用，包含各rank随机状态。

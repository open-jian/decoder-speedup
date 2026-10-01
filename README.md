# decoder-speedup

独立的解码器加速与恢复训练项目。首版采用 **AMD v1/v3 的阶段宽度，接入原 Wan2.2 VAE，仅做缩宽**。原编码器冻结，训练学生解码器；latent 的 48 个通道及归一化约定不变。

目前支持从结构配置到训练、评测、导出的完整流程。首版为**单卡训练**。默认宽度来自 AMD 发布配置；在保留 Wan 原结构的条件下，画质恢复、训练预算和加速效果仍待验证。

## 项目结构

```text
src/decoder_speedup/
  models/wan22/    Wan 原码接入、学生构造、通道权重初始化、因果缓存
  config.py       严格配置与结构依赖检查
  data.py         固定样本清单、按源视频划分、可恢复的数据游标
  training/       重建/特征/感知损失、GAN、EMA、完整断点
  evaluation.py   PSNR、SSIM、LPIPS、解码耗时与显存
  runtime.py      精度、卷积内存布局、推理编译开关
  export.py       单独部署的学生权重与结构
configs/wan22/     原宽度对照、局部缩宽对照、AMD阶段宽度主配置
examples/         真实 Wan 权重的接入验证
```

Wan 源码和原始权重作为外部依赖，不复制进本仓库，也不修改。训练循环只依赖 `teacher.prepare()` 返回的模型输入和特征监督；以后新增解码器时增加对应适配器。

## 安装

在自己的项目虚拟环境中安装，避免改共享环境：

```bash
python -m venv .venv
source .venv/bin/activate
# 根据机器先安装匹配的 PyTorch / torchvision，再安装本项目。
pip install -e '.[perceptual,tracking,test]'
```

已有经过验证的 PyTorch 环境时也可设置 `PYTHONPATH=src` 直接使用 `python -m decoder_speedup`。LPIPS 使用预训练 VGG，首次实例化可能下载官方权重；`TORCH_HOME` 可以指定缓存位置。

## 配置宽度

`width.stages` 的五个数字依次是：入口/中间段、第一次上采样段、第二次上采样段、第三次上采样段、最后高分辨率段。

```yaml
width:
  stages: [512, 512, 256, 64, 32]
  hidden: {}
```

AMD v1、v3 的 `decoder_block_out_channels` 都是 `[32,64,256,512]`，源码逆序执行，并以512作为入口/中间宽度，得到上面的五阶段配置。来源和对应关系见 [AMD宽度配置](docs/AMD_WIDTH.md)。本版只取这组宽度数值；阶段对应按执行顺序，不代表照搬 AMD 各阶段的时空分辨率。

原始宽度为 `[1024, 1024, 1024, 512, 256]`。`hidden` 控制一个残差块中两个大卷积之间的通道数；未指定的块随阶段宽度。主配置不叠加额外块内缩宽，`local-width.yaml` 仅保留为另一个缩宽对照。首版保留全部 14 个残差块、中间注意力、原上采样顺序及算子。宽度变化所需的残差投影、归一化、上下游卷积一起调整。

Wan 的 `DupUp3D` 要求前三次上采样分别满足 `输出通道×8/输入通道`、`×8/`、`×4/` 是整数。非法配置直接报错。阶段宽度改变使原来的恒等残差支路无法相加时，会按原 Wan 类的规则建立 1×1×1 投影；这属于宽度变化的依赖调整。

初始化有三种：

- `random`：学生解码器随机初始化；latent 投影和归一化仍来自原模型并冻结。
- `teacher_prefix`：按维度截取老师前若干通道；Q/K/V、时间重排的分组分别处理。新增投影用矩形单位矩阵初始化。**这是可复现的热启动，不是重要性剪枝，也不保证缩宽前后功能相同。**
- `checkpoint`：加载同结构学生的导出文件，或训练断点中的 EMA；开启新的训练计划。完整续训使用 `--resume`，不能混用。

无需加载权重即可检查结构和参数量：

```bash
export WAN22_SOURCE=/absolute/path/to/Wan2.2
python -m decoder_speedup inspect configs/wan22/original.yaml
python -m decoder_speedup inspect configs/wan22/local-width.yaml
```

## 数据

视频 JSONL 每行至少包含：

```json
{"path":"videos/VidGen_video_174/JynVt9nDUdM-Scene-0044.mp4","source_id":"JynVt9nDUdM"}
```

VidGen 可根据视频文件名提取原 YouTube 视频 ID，同一源视频的不同片段只进入一个划分。下载目录里若有 `videos/`，只扫描已完成并移入该目录的文件，排除正在解压的数据。不做二次质量筛选。

```bash
export VIDGEN_ROOT=/absolute/path/to/vidgen-1m
export MANIFEST_DIR=/absolute/path/to/manifests/experiment-001
python -m decoder_speedup manifest --root "$VIDGEN_ROOT" --output "$MANIFEST_DIR" \
  --limit 10000 --seed 42 --validation-fraction 0.05
```

这里的 1 万仅演示如何固定一份子集，并非建议的最终样本量；去掉 `--limit` 即使用当时所有可见视频。下载继续增加的数据不会自动加入已冻结清单。其他命名的数据集提供带明确 `source_id` 的 `--input-manifest`，不猜测源视频关系。

训练固定长度采样、保持纵横比缩放、整段一致的裁剪/翻转；验证使用中心位置。损坏或过短视频直接报出文件名，**不偷偷跳过**。数据加载目前同步执行，优先保证准确续训；大规模吞吐优化留到实测后做。

部分VidGen MP4的头部帧数包含不可解码的尾部区间。预检后可在清单中写入 `decoded_frames`（完整解码核实的正整数）；读取器优先按此数量抽帧，避免随机位置越过实际结尾。核实后的清单同样固定哈希，不在训练中修改或静默替换样本。

## 训练、续训、第二阶段

```bash
export WAN22_WEIGHTS=/absolute/path/to/Wan2.2_VAE.pth
export RUN_DIR=/absolute/path/to/runs/wan22-width-001
python -m decoder_speedup inspect configs/wan22/width.yaml
# 确认空闲 GPU 和自己的配置后启动；下面命令会正式训练。
CUDA_VISIBLE_DEVICES=0 python -m decoder_speedup train configs/wan22/width.yaml
```

相对路径按配置文件所在目录解析。环境变量必须存在；未知字段和重复 YAML 键会报错。

学生损失为 `L1 + λp·LPIPS + λf·特征MSE`，目标图像为原视频；老师固定，提供 latent 和中间特征。不同宽度的特征通过训练用 1×1×1 投影对齐，投影头不导出。没有给固定编码器添加无效的 KL 梯度。

`reconstruction_updates` 是第一阶段 G 优化器更新数；`adversarial_updates` 是后续 GAN 阶段 G 更新数。两者之和是硬停止预算。`accumulation` 每积累 N 个批次更新一次 G。GAN 阶段每次 G 更新后按 `discriminator_updates` 更新 D，每次 D 更新使用这 N 批真实/生成视频的平均损失。D 用 hinge loss，G 用 `-D(fake)`；可选按最后 RGB 卷积上的梯度范数给 GAN 权重自适应缩放。

主配置采用 [Turbo公开训练配置](docs/TURBO_RECIPE.md)：固定G/D学习率1e-4、batch1、累积8、100轮重建预算。先读取固定训练清单换算实际G更新数，不再用2万步占位；GAN初始关闭。

```bash
# 只检查预算，不加载模型或训练。
python -m decoder_speedup plan configs/wan22/width.yaml
# 以下会正式训练；可在独立检查点结束本次运行，保留完整预算。
python -m decoder_speedup train configs/wan22/width.yaml --stop-after-updates 1000
# 正常续训，保留优化器、EMA、数据位置与W&B run。
python -m decoder_speedup train "$RUN_DIR/config.resolved.yaml" --resume "$RUN_DIR/last.pt"
# 重建指标稳定后，指定额外GAN更新预算，保留G状态进入第二阶段。
python -m decoder_speedup train "$RUN_DIR/config.resolved.yaml" \
  --resume "$RUN_DIR/last.pt" --start-gan-updates "$GAN_UPDATES"
```

`GAN_UPDATES`为显式正整数，不设未经验证的默认值。还可用 `--extend-reconstruction-updates` 延长尚未开始GAN的预算。阶段变更后使用输出目录的 `config.resolved.yaml` 续训。普通resume仍严格核对配方；仅显式变更入口放开对应预算字段，其他模型/数据/训练设置保持核验。

完整断点保存学生、特征投影、G/D优化器、判别器、EMA、更新计数、随机状态、数据游标、训练计划与监控窗口。首版明确采用固定学习率，无warmup和衰减。详细来源、与Turbo的工程差异及阶段操作见 [TURBO_RECIPE.md](docs/TURBO_RECIPE.md)。

训练输出包括解析后的配置、来源哈希、逐步 JSONL、断点、逐片段评测、最终 EMA 学生。恢复训练质量受初始化、结构、数据与预算共同影响；[方法与边界](docs/TRAINING.md) 说明与 Turbo 的关系。

## W&B 训练监控

主配置已启用 [miaoyin-uta/vae-speedup](https://wandb.ai/miaoyin-uta/vae-speedup)，沿用已有登录或 `WANDB_API_KEY`，配置中不放密钥。

```yaml
wandb:
  enabled: true
  entity: miaoyin-uta
  project: vae-speedup
  mode: online
  group: wan22-width-recovery
  tags: [data:vidgen-1m]
  log_every: 50
  system_sample_seconds: 30
```

- 固定每50次G参数更新合并上传损失均值/最大值、梯度、学习率、进度和训练耗时；首步、阶段边界和结束额外记录。训练曲线只按实际更新次数，不由时间触发。
- 画质默认每1000步验证，记录老师/EMA学生PSNR、SSIM、LPIPS及差值；GPU等系统指标每30秒采样。
- 模型、缩宽方案、初始化、训练阶段和任务用途自动生成标签；数值超参数、数据哈希留在config。
- 分为 `train`、`gan`、`quality`、`optim`、`progress`、`timing`、`monitor`；独立测速用 `decode`。
- 在线续训保留run ID和未上报窗口。本地每步JSONL继续保留；默认不上传视频、权重、源码或控制台输出。
- `evaluate`、`benchmark` 完成后分别建立独立评测/测速run，通过学生SHA256和来源训练run关联。SDK在测速结束后才启动。

分组规则、指标含义、窗口边界和性能控制详见 [监控体系](docs/MONITORING.md)。`wandb.enabled: false` 禁用，`mode: offline` 只记录本地SDK日志。可单独检查连接，不加载模型或训练：

```bash
decoder-speedup wandb-check --entity miaoyin-uta --project vae-speedup
```

## 评测和导出

```bash
python -m decoder_speedup export /absolute/path/to/last.pt \
  --source "$WAN22_SOURCE" --output /absolute/path/to/student.pt
python -m decoder_speedup evaluate configs/wan22/width.yaml \
  --student /absolute/path/to/student.pt --output /absolute/path/to/quality.json
python -m decoder_speedup benchmark configs/wan22/width.yaml \
  --student /absolute/path/to/student.pt --output /absolute/path/to/timing.json
```

`evaluate` 同一批验证视频比较老师和学生的 PSNR、SSIM，以及启用时的 LPIPS。训练时不截断 RGB；评测按 `[-1,1]` 截断。SSIM 定义与聚合方式记录在实现中。

`benchmark` 同机同输入测三条：原结构参考、原结构加当前执行设置、学生加相同执行设置。计时只包含完整 VAE 解码，不包含编码器、视频读取和数据传输；记录预热、每次耗时、中位数、分位数及额外峰值显存。分别报告速度倍率、耗时降幅，以及结构压缩相对执行优化老师的额外倍率。测速用固定随机 latent，画质另用真实视频测。

`runtime.precision` 控制 FP32/BF16 AMP；`weight_dtype` 控制推理权重 FP32/BF16，训练必须保留 FP32 主权重。`channels_last` 单独控制卷积权重布局。`compile` 是**推理实验开关**，使用允许图中断的 `torch.compile`，尚未保证所有 Wan 源码/硬件兼容或提速；首版训练和质量评测不编译。

部署只需安装本框架及对应版本的 Wan 源码，不需要老师权重：

```python
import torch
from decoder_speedup.export import load_student

student, metadata = load_student("/path/student.pt", "/path/Wan2.2", "cuda:0")
with torch.inference_mode():
    rgb, _ = student(normalized_wan_latent)
    rgb = rgb.float().clamp(-1, 1)
```

项目名和命令统一为 `decoder-speedup`，Python 包名为 `decoder_speedup`。改名前导出的学生权重和训练文件仍可读取；严格续训仍按原规则核对配置和源码指纹。

导出包含学生结构、权重、latent 归一化、来源指纹和输出约定。加载时检查 Wan 源文件哈希。每次调用独立重置因果缓存；单次调用内部逐 latent 解码并保留跨块梯度。

## 验证与边界

```bash
WAN22_SOURCE=/path/to/Wan2.2 CUDA_VISIBLE_DEVICES='' python -m pytest -q
```

测试覆盖结构非法配置、原结构一致性、因果/缓存、非均匀及内部缩宽反传、分组权重初始化、原视频隔离、视频读取、G/D 更新、完整续训一致性、EMA、CLI 训练到导出、学生回读及来源校验。

真实官方权重的验证程序在 `examples/verify_real_weights.py`：只做合成小输入的原结构对齐、缩宽反传和导出回读，零优化器更新。首次验证记录见 [docs/VALIDATION.md](docs/VALIDATION.md)。

暂未实现：减层、自动宽度搜索、AMD 学生结构、多卡训练、梯度检查点/预编码 latent 缓存。没有把参数减少量当成速度收益，也没有用这次功能验证宣称恢复画质。

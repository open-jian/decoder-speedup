# AMD 宽度配置：本版只取通道数

主配置为 `configs/wan22/width.yaml`，采用 AMD HummingbirdXT **v1和v3共有**的宽度。

发布配置中的 `decoder_block_out_channels` 均为 `[32,64,256,512]`。AMD构造器先逆序为 `[512,256,64,32]`，再以第一个512建立入口和middle，所以主宽度按执行顺序为 `[512,512,256,64,32]`。

| 原 Wan 模块 | 原通道数 | 本版通道数 |
|---|---:|---:|
| 入口 / middle | 1024 | 512 |
| upsamples.0 | 1024 | 512 |
| upsamples.1 | 1024 | 256 |
| upsamples.2 | 512 | 64 |
| upsamples.3 / 输出头输入 | 256 | 32 |

只迁移以上宽度数值，按原 Wan 的阶段顺序应用。AMD的上采样位置不同，因此相同阶段序号不代表相同时间/空间分辨率。这里不能称为AMD完整模型或直接使用AMD学生权重。

保留原Wan的14个残差块、middle注意力、普通3D卷积、RMS归一化、SiLU、上采样顺序和方式、因果缓存、48通道latent投影，以及12通道输出头和unpatchify。`hidden: {}`，不叠加自拟的块内压缩；相邻卷积与归一化通道随主宽度调整。

一个必要依赖：`upsamples.1` 的首个残差块从同宽变成512→256，原恒等旁路变成原Wan类自带的1×1×1通道投影。没有增加完整残差块。三个DupUp3D旁路的通道重复数为8、4、1，均满足原算子的整数约束。

默认channels-last和compile关闭，权重保持FP32；BF16 AMP同时用于老师和学生。训练里的特征投影/GAN属于恢复手段，不改部署学生结构。本次不引入AMD的DW卷积、层数重排、激活替换、去注意力或新输出头。

来源固定为此前核查版本：

- [AMD v1 配置](https://huggingface.co/amd/HummingbirdXT/blob/4d3bd2e3a8c96a189ed158ab74b575ec6a424d2c/vae/wan22_v1_tiling_16_12/config.json)
- [AMD v3 配置](https://huggingface.co/amd/HummingbirdXT/blob/4d3bd2e3a8c96a189ed158ab74b575ec6a424d2c/vae/wan22_v3_tiling_16_12/config.json)
- [AMD构造器：逆序宽度及middle/up blocks](https://github.com/AMD-AGI/HummingbirdXT/blob/929e90a26c2d023c89f4428aa43869c88817a55e/infer/examples/wan2.2/autoencoder_kl_turbo_vaed_ours_wan22.py#L764)

提取字段及本地归档文件哈希保存在 `amd_width_source.json`。初始化与训练参数仍是本框架的选择，不是AMD未公布的训练配方。

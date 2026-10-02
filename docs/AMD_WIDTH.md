# AMD width configuration: channel counts only

The main configuration, `configs/wan22/width.yaml`, uses the widths **shared by AMD HummingbirdXT v1 and v3**.

Both released configurations specify `decoder_block_out_channels: [32,64,256,512]`. AMD's constructor reverses these to `[512,256,64,32]` and uses the first value, 512, for the entry and middle blocks. The resulting widths in execution order are `[512,512,256,64,32]`.

| Original Wan module | Original channels | Student channels |
|---|---:|---:|
| Entry / middle | 1024 | 512 |
| upsamples.0 | 1024 | 512 |
| upsamples.1 | 1024 | 256 |
| upsamples.2 | 512 | 64 |
| upsamples.3 / input to output head | 256 | 32 |

Only these width values are transferred, following the original Wan stage order. AMD places upsampling differently, so matching stage indices do not imply matching temporal/spatial resolutions. This is not AMD's full model and does not directly use AMD student weights.

The original Wan model's 14 residual blocks, middle attention, regular 3D convolutions, RMS normalization, SiLU, upsampling order and method, causal caches, 48-channel latent projection, 12-channel output head, and unpatchify operation are retained. `hidden: {}` adds no custom internal narrowing. Adjacent convolution and normalization dimensions follow the stage widths.

One required dependency change occurs in the first residual block of `upsamples.1`: equal input/output widths become 512 to 256, so the original identity shortcut becomes the 1x1x1 channel projection provided by the Wan class. No residual block is added. Channel repeat counts for the three `DupUp3D` shortcuts are 8, 4, and 1, satisfying the original operator's integer constraints.

Channels-last and compilation are disabled by default. Weights remain FP32, and both teacher and student use BF16 AMP. Training feature projections and GAN losses are recovery mechanisms and do not change the deployed student architecture. This version does not adopt AMD's depthwise convolutions, layer rearrangements, activation replacements, attention removal, or new output head.

Sources are pinned to the revisions previously reviewed:

- [AMD v1 configuration](https://huggingface.co/amd/HummingbirdXT/blob/4d3bd2e3a8c96a189ed158ab74b575ec6a424d2c/vae/wan22_v1_tiling_16_12/config.json)
- [AMD v3 configuration](https://huggingface.co/amd/HummingbirdXT/blob/4d3bd2e3a8c96a189ed158ab74b575ec6a424d2c/vae/wan22_v3_tiling_16_12/config.json)
- [AMD constructor: reversed widths and middle/up blocks](https://github.com/AMD-AGI/HummingbirdXT/blob/929e90a26c2d023c89f4428aa43869c88817a55e/infer/examples/wan2.2/autoencoder_kl_turbo_vaed_ours_wan22.py#L764)

Extracted fields and local archive hashes are recorded in `amd_width_source.json`. Initialization and training parameters remain this framework's choices; they are not AMD's unpublished training recipe.

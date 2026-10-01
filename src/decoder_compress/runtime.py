from contextlib import nullcontext
import torch
from torch import nn


def autocast(config):
    device = torch.device(config.device)
    return torch.autocast(device.type, dtype=torch.bfloat16) if config.precision == "bf16" else nullcontext()


def apply_layout(model, enabled):
    # Reset as well as enable the layout, so repeated runtime comparisons do not
    # inherit a previous channels-last setting. Do not alter unrelated buffers.
    for layer in model.modules():
        if isinstance(layer, nn.Conv3d):
            fmt = torch.channels_last_3d if enabled else torch.contiguous_format
            layer.weight.data = layer.weight.data.contiguous(memory_format=fmt)
        elif isinstance(layer, nn.Conv2d):
            fmt = torch.channels_last if enabled else torch.contiguous_format
            layer.weight.data = layer.weight.data.contiguous(memory_format=fmt)
    return model


def inference_decoder(model, config):
    dtype = torch.float32 if config.weight_dtype == "fp32" else torch.bfloat16
    model = apply_layout(model.to(device=config.device, dtype=dtype).eval(), config.channels_last)
    if config.compile:
        # Upstream cache control flow may graph-break. Report this as default partial
        # compilation, not fullgraph compilation or a guaranteed speed improvement.
        return torch.compile(model, fullgraph=False)
    return model

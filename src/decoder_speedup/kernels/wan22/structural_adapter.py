"""Reversible per-instance inference adapters for cache views and shortcuts."""
import types

import torch

from .cache_views import decoder_forward, residual_forward, resample_forward
from .fused_skip import shortcut_add


def _has_hooks(layer):
    return bool(layer._forward_hooks or layer._forward_pre_hooks or layer._backward_hooks)


def _up_forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
    x_main = x
    for layer in self.upsamples:
        x_main = layer(x_main, feat_cache, feat_idx)
    if self.avg_shortcut is None:
        return x_main
    return shortcut_add(x, x_main, self.avg_shortcut, first_chunk)


class StructuralAdapter:
    """Apply only to a fixed inference decoder; disabling restores exact methods.

    The first argument may be a Wan2_2_VAE wrapper, its model, or its decoder.
    Pass the imported vae2_2 module as the second argument. Runtime hooks,
    gradients, training, and unsupported devices/dtypes retain original paths.
    """

    def __init__(self, model, module):
        model = getattr(model, 'model', model)
        self.decoder = getattr(model, 'decoder', model)
        for name, layer in self.decoder.named_modules():
            if getattr(layer, 'inplace', False):
                raise ValueError(f'Cached views require no inplace decoder modules: {name}')
        self.entries = []
        self.changes = []
        self.enabled = False
        for name, layer in self.decoder.named_modules():
            if isinstance(layer, module.Up_ResidualBlock):
                optimized_fn = _up_forward
                change = 'remove main clone and fuse DupUp3D + addition'
            elif isinstance(layer, module.ResidualBlock):
                optimized_fn = residual_forward
                change = 'cache-tail views'
            elif isinstance(layer, module.Decoder3d):
                optimized_fn = decoder_forward
                change = 'cache-tail views'
            elif isinstance(layer, module.Resample):
                optimized_fn = resample_forward
                change = 'cache-tail views'
            else:
                continue
            original = layer.forward
            optimized = types.MethodType(optimized_fn, layer)
            hook_sources = tuple(layer.modules())

            def guarded(this, x, *args, _old=original, _new=optimized,
                        _sources=hook_sources, **kwargs):
                if (this.training or torch.is_grad_enabled() or not x.is_cuda
                        or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                        or any(_has_hooks(m) for m in _sources)):
                    return _old(x, *args, **kwargs)
                return _new(x, *args, **kwargs)

            self.entries.append((layer, original, types.MethodType(guarded, layer)))
            self.changes.append({'name': name or '<decoder>',
                                 'class': type(layer).__name__, 'change': change})

    def toggle(self, enabled):
        if not isinstance(enabled, bool):
            raise TypeError('enabled must be bool')
        for layer, original, optimized in self.entries:
            layer.forward = optimized if enabled else original
        self.enabled = enabled
        return self

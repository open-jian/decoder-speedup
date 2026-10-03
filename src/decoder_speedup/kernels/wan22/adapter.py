"""Reversible per-instance inference acceleration without checkpoint changes."""
import types

import torch
import torch.nn.functional as F

from .. import winograd_3d_conv as w3
from .causal_adapter import make_forward as make_f23
from .causal_f43 import make_forward as make_f43
from .norm import FusedNorm
from .structural_adapter import StructuralAdapter
from .upsample_winograd import UpsampleWinograd


_DEFAULT_F43_BUDGETS = {(1024, 512): 512, (512, 512): 128,
                        (512, 256): 128, (256, 256): 64}


def _pairs(values):
    result = set()
    for value in values:
        if (len(value) != 2 or any(type(v) is not int or v <= 0 for v in value)):
            raise ValueError('Channel pairs must contain two positive integers.')
        result.add(tuple(value))
    return result


def _native_conv(layer, x, cache_x):
    """The source native forward, bypassing only the runtime backend attribute."""
    padding = list(layer._padding)
    if cache_x is not None and layer._padding[4] > 0:
        cache_x = cache_x.to(x.device)
        x = torch.cat([cache_x, x], dim=2)
        padding[4] -= cache_x.shape[2]
    return torch.nn.Conv3d.forward(layer, F.pad(x, padding))


class DecoderKernels:
    """Own inference methods and caches, never parameters or submodules."""

    def __init__(self, student, *, source, spatial_f43=False,
                 conv_channel_pairs=None, f43_workspace_mib=None,
                 fuse_norm=True, fuse_upsample=True, structural=True, f43_min_area=0):
        self.model = getattr(student, 'model', student)
        self.decoder = self.model.decoder
        if self.model.training or self.decoder.training:
            raise ValueError('Call eval() before installing decoder kernels.')
        if getattr(self.decoder, '_decoder_kernels_installed', False):
            raise ValueError('Decoder kernels are already installed on this model.')
        for name, value in dict(spatial_f43=spatial_f43, fuse_norm=fuse_norm,
                                fuse_upsample=fuse_upsample, structural=structural).items():
            if type(value) is not bool:
                raise TypeError(f'{name} must be bool.')
        if type(f43_min_area) is not int or f43_min_area < 0:
            raise ValueError('f43_min_area must be a nonnegative integer.')
        selected = None if conv_channel_pairs is None else _pairs(conv_channel_pairs)
        budgets = dict(_DEFAULT_F43_BUDGETS if f43_workspace_mib is None else f43_workspace_mib)
        _pairs(budgets)
        if any(type(v) is not int or v <= 0 for v in budgets.values()):
            raise ValueError('F43 workspace budgets must be positive integer MiB.')
        self.source = source
        self.enabled = False
        self.removed = False
        self.entries = []
        self.convs = []
        self.helpers = []
        self.original_decoder_forward = self.decoder.forward
        self.structural = StructuralAdapter(student, source) if structural else None
        causal, self.counters = make_f23(source)
        fast = (make_f43(source, causal, workspace_mib=budgets,
                        channel_pairs=set(budgets), config=(64, 128, 32, 8, 3),
                        min_area=f43_min_area)
                if spatial_f43 else causal)
        selected_names = []
        selected_pairs = set()
        f43_names = []
        for name, layer in self.decoder.named_modules():
            pair = (getattr(layer, 'in_channels', 0), getattr(layer, 'out_channels', 0))
            eligible = (min(pair) >= 128 if selected is None else pair in selected)
            if (isinstance(layer, source.CausalConv3d) and eligible
                    and '.residual.' in name and layer.kernel_size == (3, 3, 3)):
                if not hasattr(layer, '_conv_backend'):
                    raise ValueError('External Wan source lacks the Winograd convolution backend.')
                if layer._conv_backend != 'native':
                    raise ValueError('Install on native convolution backends; remove other adapters first.')
                original_backend = layer._conv_backend
                self.convs.append((layer, original_backend))
                old = layer.forward

                def guarded(this, x, cache_x=None, _old=old, _backend=original_backend):
                    # CPU / FP32 inference retains the original native operation.
                    # CUDA BF16 uses the tested kernels and their shape guards.
                    if (not x.is_cuda
                            or w3._effective_dtype(x, this.weight, this.bias) != torch.bfloat16):
                        return (_native_conv(this, x, cache_x) if _backend == 'native'
                                else _old(x, cache_x))
                    return fast(this, x, cache_x)

                self.entries.append((layer, old, types.MethodType(guarded, layer)))
                selected_names.append(name)
                selected_pairs.add(pair)
                if spatial_f43 and pair in budgets:
                    f43_names.append(name)

        for layer in list(self.decoder.modules()):
            if fuse_norm and isinstance(layer, torch.nn.Sequential):
                for index in range(len(layer) - 1):
                    norm, activation = layer[index], layer[index + 1]
                    if (isinstance(norm, source.RMS_norm)
                            and isinstance(activation, torch.nn.SiLU)):
                        helper = FusedNorm(norm, True)
                        self.helpers.append(helper)
                        self.entries.append((norm, norm.forward, helper.forward))
                        def identity(this, x):
                            if (this._forward_hooks or this._forward_pre_hooks or this._backward_hooks):
                                raise RuntimeError('Disable decoder kernels before adding activation hooks.')
                            return x
                        self.entries.append((activation, activation.forward,
                                             types.MethodType(identity, activation)))
            if fuse_norm and isinstance(layer, source.AttentionBlock):
                norm = layer.norm
                helper = FusedNorm(norm, False)
                self.helpers.append(helper)
                self.entries.append((norm, norm.forward, helper.forward))
            if (fuse_upsample and isinstance(layer, source.Resample)
                    and layer.mode in ('upsample2d', 'upsample3d')):
                seq = layer.resample
                helper = UpsampleWinograd(seq, domain_dtype=torch.float16).eval()
                self.helpers.append(helper)
                self.entries.append((seq, seq.forward, helper.forward))
        self.summary = {
            'conv_names': selected_names,
            'conv_channel_pairs': [list(pair) for pair in sorted(selected_pairs)],
            'conv_count': len(selected_names), 'f43_names': f43_names,
            'f43_count': len(f43_names), 'spatial_f43': spatial_f43,
            'f43_min_area': f43_min_area,
            'fuse_norm': fuse_norm, 'fuse_upsample': fuse_upsample,
            'structural': structural,
            'f43_workspace_mib': {f'{a},{b}': v for (a,b),v in budgets.items()},
        }
        self.conv_layer_names = tuple(selected_names)
        self.configuration = self.summary
        self.decoder._decoder_kernels_installed = True
        self.enable()

    def enable(self):
        if self.removed:
            raise RuntimeError('This decoder-kernel handle has been removed; install a new one.')
        if self.model.training or self.decoder.training:
            raise ValueError('Call eval() before enabling decoder kernels.')
        if self.enabled:
            return self
        if self.structural is not None:
            self.structural.toggle(True)
        for layer, _ in self.convs:
            layer._conv_backend = 'winograd_3d'
            layer._winograd_weight_cache = None
        for layer, _, fast in self.entries:
            layer.forward = fast
        inner = self.decoder.forward
        def inference_only(this, *args, **kwargs):
            if this.training or torch.is_grad_enabled():
                raise RuntimeError('Decoder kernels are inference-only; disable before training and use torch.inference_mode().')
            return inner(*args, **kwargs)
        self.decoder.forward = types.MethodType(inference_only, self.decoder)
        self.enabled = True
        return self

    def disable(self):
        if not self.enabled:
            return self
        if self.structural is not None:
            self.structural.toggle(False)
        self.decoder.forward = self.original_decoder_forward
        for layer, original, _ in self.entries:
            layer.forward = original
        for layer, backend in self.convs:
            layer._conv_backend = backend
            layer._winograd_weight_cache = None
        self.enabled = False
        return self

    def remove(self):
        """Restore original execution and allow a fresh installation."""
        if self.removed:
            return self
        self.disable()
        self.decoder._decoder_kernels_installed = False
        self.removed = True
        return self


def install(student, *, source, **kwargs):
    return DecoderKernels(student, source=source, **kwargs)

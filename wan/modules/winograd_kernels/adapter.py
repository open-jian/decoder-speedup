"""Reversible per-instance inference acceleration without checkpoint changes."""

import importlib
import types

import torch

from .causal_adapter import make_forward as make_f23
from .causal_f43 import make_forward as make_f43
from .norm import FusedNorm
from .structural_adapter import StructuralAdapter
from .upsample_winograd import UpsampleWinograd


class DecoderKernels:
    """Install on an eval-mode Wan2_2_VAE after loading its original checkpoint.

    This handle owns temporary inference caches and bound methods, never model
    parameters. Keep it alive while using the model. ``disable()`` restores the
    original forwards and convolution backends. Do not change modes concurrently
    with an in-flight decode or a CUDA Graph replay.
    """

    def __init__(self, vae, *, spatial_f43=True):
        if vae.model.training:
            raise ValueError('Call eval() before installing decoder kernels.')
        if getattr(vae.model.decoder, '_decoder_kernels_installed', False):
            raise ValueError('Decoder kernels are already installed on this model.')
        self.vae = vae
        self.enabled = False
        self.entries = []
        self.convs = []
        self.helpers = []
        module = importlib.import_module(type(vae.model.decoder).__module__)
        self.structural = StructuralAdapter(vae, module)
        causal, self.counters = make_f23(module)
        budgets = {(1024, 512): 512, (512, 512): 128,
                   (512, 256): 128, (256, 256): 64}
        fast = (make_f43(module, causal, workspace_mib=budgets,
                        channel_pairs=set(budgets), config=(64, 128, 32, 8, 3))
                if spatial_f43 else causal)

        # Select exactly the 28 residual 3x3x3 convolutions. Boundary and RGB
        # convolutions retain their existing backend and weights.
        for name, layer in vae.model.decoder.named_modules():
            if (isinstance(layer, module.CausalConv3d)
                    and '.residual.' in name and layer.kernel_size == (3, 3, 3)):
                self.convs.append((layer, layer._conv_backend))
                self.entries.append((layer, layer.forward,
                                     types.MethodType(fast, layer)))

        for layer in list(vae.model.decoder.modules()):
            if isinstance(layer, torch.nn.Sequential):
                for index in range(len(layer) - 1):
                    norm, activation = layer[index], layer[index + 1]
                    if (isinstance(norm, module.RMS_norm)
                            and isinstance(activation, torch.nn.SiLU)):
                        helper = FusedNorm(norm, True)
                        self.helpers.append(helper)
                        self.entries.append((norm, norm.forward, helper.forward))
                        def identity(this, x):
                            if (this._forward_hooks or this._forward_pre_hooks
                                    or this._backward_hooks):
                                raise RuntimeError('Disable decoder kernels before adding activation hooks.')
                            return x
                        self.entries.append((activation, activation.forward,
                                             types.MethodType(identity, activation)))
            if isinstance(layer, module.AttentionBlock):
                norm = layer.norm
                helper = FusedNorm(norm, False)
                self.helpers.append(helper)
                self.entries.append((norm, norm.forward, helper.forward))
            if (isinstance(layer, module.Resample)
                    and layer.mode in ('upsample2d', 'upsample3d')):
                seq = layer.resample
                helper = UpsampleWinograd(seq, domain_dtype=torch.float16).eval()
                self.helpers.append(helper)
                self.entries.append((seq, seq.forward, helper.forward))

        vae.model.decoder._decoder_kernels_installed = True
        self.enable()

    def enable(self):
        self.structural.toggle(True)
        for layer, _ in self.convs:
            layer._conv_backend = 'winograd_3d'
            layer._winograd_weight_cache = None
        for layer, _, fast in self.entries:
            layer.forward = fast
        self.enabled = True
        return self

    def disable(self):
        self.structural.toggle(False)
        for layer, original, _ in self.entries:
            layer.forward = original
        for layer, backend in self.convs:
            layer._conv_backend = backend
            layer._winograd_weight_cache = None
        self.enabled = False
        return self

    def remove(self):
        """Restore original execution and allow a fresh installation."""
        self.disable()
        self.vae.model.decoder._decoder_kernels_installed = False


def install(vae, *, spatial_f43=True):
    """Enable inference kernels; all model and state_dict keys stay unchanged."""
    return DecoderKernels(vae, spatial_f43=spatial_f43)

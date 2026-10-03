"""Opt-in inference kernels for an already loaded width-compressed Wan decoder.

The external Wan source remains untouched. Its private import namespace provides
its convolution primitives to our kernels, keeping exported-checkpoint source
hashes and the teacher/student class identities intact. Importing this module
alone does not import Triton, create a CUDA context, or modify a model.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys


def _load_kernels(source):
    source_package = getattr(source, '__package__', None)
    if not source_package or sys.modules.get(source.__name__) is not source:
        raise ValueError('Expected the imported Wan vae2_2 source module.')
    directory = Path(__file__).resolve().parent / 'kernels' / 'wan22'
    suffix = hashlib.sha256(str(directory).encode()).hexdigest()[:12]
    name = source_package + '._decoder_speedup_kernels_' + suffix
    if name in sys.modules:
        return sys.modules[name]
    # Relative imports such as ..vae2_2 deliberately resolve against the exact
    # external source used to build this student, not a second Wan checkout.
    spec = importlib.util.spec_from_file_location(
        name, directory / '__init__.py', submodule_search_locations=[str(directory)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        for loaded in tuple(sys.modules):
            if loaded == name or loaded.startswith(name + '.'):
                sys.modules.pop(loaded, None)
        raise
    return module


def install(student, *, spatial_f43=False, conv_channel_pairs=None,
            f43_workspace_mib=None, fuse_norm=True, fuse_upsample=True,
            structural=True, f43_min_area=0):
    """Accelerate an eval-mode compressed decoder without changing its weights.

    Run under ``torch.inference_mode()`` and CUDA BF16 autocast with FP32 weights.
    The handle's ``disable``/``enable`` methods switch execution reversibly;
    ``remove`` restores the original methods and allows a new installation.

    By default only residual 3x3x3 convolutions with both channel counts >=128
    use Winograd; narrow stages keep their existing convolution backend. An
    explicit iterable of ``(input_channels, output_channels)`` overrides this
    selection. The larger F43 spatial tile is disabled by default because its
    original-width tuning did not help this narrower decoder.
    ``f43_workspace_mib`` maps selected channel pairs to positive
    workspace budgets and thereby overrides the larger-spatial-tile selection.
    Flags expose independent inference ablations. They never alter state_dict.
    """
    source = getattr(student, 'source', None)
    if source is None or not hasattr(student, 'decoder'):
        raise TypeError('Expected decoder_speedup.models.wan22.adapter.Decoder.')
    kernels = _load_kernels(source)
    return kernels.install(
        student, source=source, spatial_f43=spatial_f43,
        conv_channel_pairs=conv_channel_pairs,
        f43_workspace_mib=f43_workspace_mib, fuse_norm=fuse_norm,
        fuse_upsample=fuse_upsample, structural=structural, f43_min_area=f43_min_area)


__all__ = ['install']

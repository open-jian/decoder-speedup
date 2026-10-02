"""Opt-in Wan2.2 decoder kernels; importing this package does not patch models."""

from .adapter import DecoderKernels, install

__all__ = ['DecoderKernels', 'install']

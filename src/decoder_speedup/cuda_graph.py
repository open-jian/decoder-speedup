"""Optional fixed-shape CUDA Graph replay for a loaded Wan decoder.

CUDA Graph is a PyTorch/CUDA optimization, separate from the Winograd kernels.
It amortizes Python and kernel-submission overhead; it does not change weights.
The ordinary eager runtime remains the default.
"""
from __future__ import annotations

import threading

import torch


def _forward_identity(layer):
    forward = layer.forward
    return id(getattr(forward, '__func__', forward)), id(getattr(forward, '__self__', None))


def _model_signature(model):
    tensors = []
    for kind, named in (('parameter', model.named_parameters()), ('buffer', model.named_buffers())):
        for name, tensor in named:
            if tensor.is_inference():
                raise ValueError('Load the model outside inference_mode so weight mutation counters are available.')
            tensors.append((kind, name, id(tensor), tensor._version, tensor.data_ptr(),
                            tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device))
    modules = []
    for name, layer in model.named_modules():
        hooks = tuple((key, tuple((i, id(hook)) for i, hook in getattr(layer, key).items()))
                      for key in ('_forward_hooks', '_forward_pre_hooks', '_backward_hooks'))
        modules.append((name, id(layer), type(layer), layer.training,
                        _forward_identity(layer), getattr(layer, '_conv_backend', None), hooks))
    precision = (torch.backends.cuda.matmul.allow_tf32,
                 torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                 torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision())
    return tuple(tensors), tuple(modules), precision


def _training(model):
    return any(layer.training for layer in model.modules())


def _keep_caches(model, kernel_handle):
    """Graphs retain addresses, so keep caches alive after eager stream changes."""
    result = []
    objects = list(model.modules())
    if kernel_handle is not None:
        objects.extend(kernel_handle.helpers)
    for obj in objects:
        for name in ('_winograd_weight_cache', '_experimental_bf16_bias_cache', '_u_cache'):
            cache = getattr(obj, name, None)
            if cache is not None:
                result.append(cache)
    return result


class CapturedDecoder:
    """An inference-only graph for one latent shape/dtype/device.

    Each call copies its input and clones its output. Calls on different CUDA
    streams are ordered by an event so that static graph buffers cannot race.
    Do not mutate model weights or adapters concurrently with a call. Mutation
    through unsafe ``.data`` writes or external GPU pointers bypasses PyTorch's
    version counters and is unsupported. Close and recapture after any changes.
    """

    def __init__(self, model, example_latent, *, kernel_handle=None, warmup=3):
        if type(warmup) is not int or warmup < 1:
            raise ValueError('warmup must be a positive integer.')
        if not isinstance(example_latent, torch.Tensor) or not example_latent.is_cuda:
            raise ValueError('CUDA Graph capture requires a CUDA latent tensor.')
        if example_latent.ndim != 5 or min(example_latent.shape) <= 0:
            raise ValueError('Expected a nonempty normalized B,C,T,H,W latent.')
        if example_latent.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise ValueError('Latent dtype must be FP32, BF16, or FP16.')
        if example_latent.requires_grad:
            raise ValueError('CUDA Graph inference does not accept latents requiring gradients.')
        if _training(model):
            raise ValueError('Call eval() before capturing a decoder.')
        if any(layer._forward_hooks or layer._forward_pre_hooks or layer._backward_hooks
               for layer in model.modules()):
            raise ValueError('Remove module hooks before capture; Python hooks do not run during graph replay.')
        if any(t.device != example_latent.device for t in (*model.parameters(), *model.buffers())):
            raise ValueError('Model parameters, buffers, and latent must be on the same CUDA device.')
        kernel_handle = kernel_handle if kernel_handle is not None else getattr(model, '_winograd_handle', None)
        marker = bool(getattr(getattr(model, 'decoder', None), '_decoder_kernels_installed', False))
        if marker and kernel_handle is None:
            raise ValueError('Pass kernel_handle=handle when kernels were installed directly.')
        if kernel_handle is not None:
            if kernel_handle.model is not model or not kernel_handle.enabled or kernel_handle.removed:
                raise ValueError('kernel_handle must be the enabled adapter installed on this model.')
        self.model = model
        self.kernel_handle = kernel_handle
        self.closed = False
        self._lock = threading.RLock()
        self._shape = tuple(example_latent.shape)
        self._dtype = example_latent.dtype
        self._device = example_latent.device
        self._signature = _model_signature(model)
        self._event_recorded = False
        self._last_event = torch.cuda.Event()
        self._stream = torch.cuda.Stream(device=self._device)
        self._keepalive = []
        with torch.cuda.device(self._device), torch.inference_mode():
            self._input = example_latent.detach().clone().contiguous()
            self._stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._stream):
                for _ in range(warmup):
                    self._decode()
            torch.cuda.current_stream().wait_stream(self._stream)
            self._stream.synchronize()
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph, stream=self._stream):
                self._output = self._decode()
            self._keepalive = _keep_caches(model, kernel_handle)
        self._check_model()

    def _decode(self):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = self.model(self._input)
        if not isinstance(result, tuple) or len(result) != 2 or result[1]:
            raise ValueError('Expected decoder output (rgb, {}) without feature hooks.')
        if not isinstance(result[0], torch.Tensor):
            raise ValueError('Expected tensor RGB output.')
        return result[0]

    def _check_model(self):
        if self.closed:
            raise RuntimeError('This captured decoder is closed; capture a new one.')
        if self.kernel_handle is not None:
            if not self.kernel_handle.enabled or self.kernel_handle.removed:
                raise RuntimeError('Decoder kernel adapter changed; close and recapture.')
        if _training(self.model):
            raise RuntimeError('CUDA Graph decoder is inference-only; call eval() and recapture.')
        if _model_signature(self.model) != self._signature:
            raise RuntimeError('Model weights, buffers, forwards, hooks, or backends changed; close and recapture.')

    def __call__(self, latent, feature_paths=()):
        with self._lock:
            if self.closed:
                raise RuntimeError('This captured decoder is closed; capture a new one.')
            if torch.is_grad_enabled() or latent.requires_grad:
                raise RuntimeError('CUDA Graph decoder is inference-only; use torch.inference_mode().')
            if feature_paths:
                raise ValueError('Captured decoding does not return intermediate features.')
            if (tuple(latent.shape) != self._shape or latent.dtype != self._dtype
                    or latent.device != self._device):
                raise ValueError('Latent shape, dtype, and device must match the captured example; recapture explicitly.')
            self._check_model()
            with torch.cuda.device(self._device), torch.inference_mode():
                stream = torch.cuda.current_stream()
                if self._event_recorded:
                    stream.wait_event(self._last_event)
                self._input.copy_(latent)
                self._graph.replay()
                rgb = self._output.clone()
                self._last_event.record(stream)
                self._event_recorded = True
                return rgb, {}

    def close(self):
        """Wait for pending replay and release graph buffers; safe to call twice."""
        with self._lock:
            if self.closed:
                return
            with torch.cuda.device(self._device):
                if self._event_recorded:
                    self._last_event.synchronize()
                self._stream.synchronize()
            self.closed = True
            self._keepalive.clear()
            self._graph = self._input = self._output = None
            self.model = self.kernel_handle = None


def capture_decoder(model, example_latent, *, kernel_handle=None, warmup=3):
    """Capture fixed-shape BF16 AMP decoding; model must already be CUDA/eval.

    Pass a normalized latent and the result of ``winograd.install`` if installed
    directly. Handles attached by ``inference_decoder`` are discovered. Native
    compressed decoders are also supported for matched graph-to-graph timing.
    Capture/JIT is paid once; invoke the result under ``torch.inference_mode()``.
    """
    return CapturedDecoder(model, example_latent, kernel_handle=kernel_handle, warmup=warmup)


__all__ = ['CapturedDecoder', 'capture_decoder']

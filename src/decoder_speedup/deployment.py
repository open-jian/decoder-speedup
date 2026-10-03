"""List-based deployment API for the 48-channel Wan2.2 VAE student."""
from __future__ import annotations

import threading

import torch

from .export import load_student


class Wan22CompressedDecoder:
    """Replace ``Wan2_2_VAE.decode`` while keeping its encoder unchanged.

    This checkpoint interface is for the 48-channel Wan2.2 VAE used by TI2V-5B,
    not the 16-channel Wan2.1 VAE used by several other Wan2.2 pipelines.
    Inputs are already normalized Wan latents; no extra scaling is required.
    CUDA Graph is optional and retains one shape at a time. Changing shape or
    dtype closes the previous graph and pays capture/warmup again.
    """

    def __init__(self, checkpoint, source, device="cuda:0", *,
                 use_winograd=True, use_cuda_graph=False):
        if type(use_winograd) is not bool or type(use_cuda_graph) is not bool:
            raise ValueError("use_winograd and use_cuda_graph must be booleans")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("Wan22CompressedDecoder requires a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.use_cuda_graph = use_cuda_graph
        self._lock = threading.RLock()
        self._closed = False
        self._graph = self._graph_key = self._handle = None
        # Graph validity checks need normal tensors with mutation counters, even
        # when the caller constructs this object inside inference_mode.
        with torch.inference_mode(False), torch.no_grad():
            self.model, self.metadata = load_student(checkpoint, source, device)
            self.model.float().eval().requires_grad_(False)
        if use_winograd:
            from .winograd import install
            self._handle = install(self.model)

    def _validate(self, zs):
        if not isinstance(zs, list):
            raise TypeError("zs must be a list of normalized [48,T,H,W] latent tensors")
        for z in zs:
            if not isinstance(z, torch.Tensor):
                raise TypeError("Every latent must be a torch.Tensor")
            if z.ndim != 4 or z.shape[0] != 48 or min(z.shape[1:]) <= 0:
                raise ValueError("Expected a nonempty normalized [48,T,H,W] latent")
            if z.dtype not in (torch.float32, torch.bfloat16, torch.float16):
                raise ValueError("Latent dtype must be FP32, BF16, or FP16")
            if z.device != self.device:
                raise ValueError(f"Latent device {z.device} must match decoder device {self.device}")

    @torch.inference_mode()
    def decode(self, zs):
        """Return a list of float32 RGB videos ``[3,F,H,W]`` clipped to [-1,1]."""
        with self._lock:
            if self._closed:
                raise RuntimeError("This compressed decoder is closed")
            self._validate(zs)
            if not zs:
                return []
            videos = []
            with torch.cuda.device(self.device):
                for z in zs:
                    latent = z.detach().unsqueeze(0)
                    if self.use_cuda_graph:
                        key = (tuple(latent.shape), latent.dtype, latent.device)
                        if self._graph is None or self._graph_key != key:
                            if self._graph is not None:
                                self._graph.close()
                                self._graph = self._graph_key = None
                            from .cuda_graph import capture_decoder
                            self._graph = capture_decoder(
                                self.model, latent, kernel_handle=self._handle)
                            self._graph_key = key
                        rgb, _ = self._graph(latent)
                    else:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            rgb, _ = self.model(latent)
                    videos.append(rgb.float().clamp(-1, 1).squeeze(0))
            return videos

    def close(self):
        """Release captured buffers and kernels; safe to call more than once."""
        with self._lock:
            if self._closed:
                return
            if self._graph is not None:
                self._graph.close()
                self._graph = self._graph_key = None
            if self._handle is not None:
                self._handle.remove()
                self._handle = None
            self.model = None
            self._closed = True


__all__ = ["Wan22CompressedDecoder"]

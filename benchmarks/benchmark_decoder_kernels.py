"""Compare native, previous Winograd, and optimized complete decoder latency."""

import argparse
import hashlib
import importlib
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from vae_loader import load_vae_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--latents', type=Path, nargs='+', required=True,
                        help='One or more N,C,T,H,W .npy tensors, already unnormalized.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=9)
    args = parser.parse_args()
    if args.repeats < 2 or args.warmup < 1:
        parser.error('Use at least 2 repeats and 1 warmup.')
    torch.set_num_threads(6)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    root = Path(__file__).resolve().parents[1]
    module = load_vae_module()
    package = importlib.import_module(module.__package__ + '.winograd_kernels')
    k3 = importlib.import_module(module.__package__ + '.winograd_3d_triton')
    original_gemm = k3._gemm

    class SelectiveGemm:
        def __getitem__(self, grid):
            def call(u, v, m, k, c, p, **kwargs):
                if k == c == 1024:
                    return torch.bmm(u, v, out_dtype=torch.float32, out=m)
                return original_gemm[grid](u, v, m, k, c, p, **kwargs)
            return call

    vae = module.Wan2_2_VAE(
        vae_pth=str(args.checkpoint), device='cuda', dtype=torch.bfloat16,
        conv_config=root / 'configs/vae_conv/winograd_3d_all_residuals.json')
    vae.model.encoder.cpu()
    vae.model.conv1.cpu()
    before = {key: (value.data_ptr(), tuple(value.shape), value._version)
              for key, value in vae.model.state_dict().items()}
    handle = package.install(vae)
    convs = [(layer, layer._conv_backend) for layer in vae.model.decoder.modules()
             if isinstance(layer, module.CausalConv3d)]
    report = {
        'gpu': torch.cuda.get_device_name(), 'torch': str(torch.__version__),
        'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'precision': 'FP32 stored weights, BF16 autocast; F43/up2 use FP16 transformed coefficients and FP32 accumulation',
        'measurement': 'Synchronized host wall time, GPU input/output; complete model.decode including conv2/cache reset; excludes loading/JIT/transformed-weight warmup/I/O',
        'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (root / 'wan/modules/winograd_kernels').glob('*.py')},
        'cases': [],
    }
    modes = ['native', 'old_winograd', 'previous_best', 'optimized']

    def select(mode):
        handle.disable()
        for layer, backend in convs:
            layer._conv_backend = 'native' if mode == 'native' else backend
        k3._gemm = SelectiveGemm() if mode == 'previous_best' else original_gemm
        if mode == 'optimized':
            handle.enable()

    @torch.inference_mode()
    def decode(z):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return vae.model.decode(z, [0., 1.])

    rng = random.Random(144)
    for path in args.latents:
        z = torch.from_numpy(np.load(path)).cuda()
        samples = {mode: [] for mode in modes}
        peaks = {mode: [] for mode in modes}
        extra_peaks = {mode: [] for mode in modes}
        outputs = {}
        for start in range(0, args.repeats, 3):
            order = modes.copy()
            rng.shuffle(order)
            for mode in order:
                select(mode)
                for _ in range(args.warmup):
                    decode(z)
                torch.cuda.synchronize()
                for _ in range(min(3, args.repeats - start)):
                    torch.cuda.reset_peak_memory_stats()
                    initial = torch.cuda.memory_allocated()
                    begin = time.perf_counter()
                    value = decode(z)
                    torch.cuda.synchronize()
                    samples[mode].append((time.perf_counter() - begin) * 1000)
                    peaks[mode].append(torch.cuda.max_memory_allocated())
                    extra_peaks[mode].append(torch.cuda.max_memory_allocated() - initial)
                    del value
        for mode in modes:
            select(mode)
            outputs[mode] = decode(z).float().cpu()
        baseline = outputs['old_winograd']
        row = {
            'latent_file': str(path), 'latent_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'latent_shape': list(z.shape), 'output_shape': list(baseline.shape),
            'timing': {mode: {'samples_ms': values, 'mean_ms': statistics.mean(values),
                             'median_ms': statistics.median(values), 'stdev_ms': statistics.stdev(values),
                             'peak_allocated_bytes': max(peaks[mode]),
                             'extra_peak_allocated_bytes': max(extra_peaks[mode])}
                       for mode, values in samples.items()},
            'error_vs_old': {mode: {'relative_l2': float((value-baseline).norm()/baseline.norm()),
                                   'max_abs': float((value-baseline).abs().max()),
                                   'finite': bool(torch.isfinite(value).all())}
                             for mode, value in outputs.items()},
        }
        new = row['timing']['optimized']['mean_ms']
        row['comparisons'] = {mode: {'speedup': row['timing'][mode]['mean_ms'] / new,
                                     'latency_reduction_percent': 100*(1-new/row['timing'][mode]['mean_ms'])}
                              for mode in modes[:-1]}
        report['cases'].append(row)
        print(json.dumps(row), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        del outputs, baseline, z
    handle.remove()
    k3._gemm = original_gemm
    after = {key: (value.data_ptr(), tuple(value.shape), value._version)
             for key, value in vae.model.state_dict().items()}
    report['checkpoint_keys_shapes_storage_versions_unchanged'] = before == after
    assert before == after, 'Inference installation altered checkpoint parameters.'
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

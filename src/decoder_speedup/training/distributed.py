"""Data parallel execution; accumulation always counts GLOBAL microbatches."""
from contextlib import contextmanager
from datetime import timedelta
import os
import torch
import torch.distributed as dist


def active():
    return dist.is_available() and dist.is_initialized()


def rank():
    return dist.get_rank() if active() else 0


def world_size():
    return dist.get_world_size() if active() else 1


def primary():
    return rank() == 0


def barrier():
    if active():
        dist.barrier()


@contextmanager
def execution(runtime):
    created = False
    if int(os.environ.get('WORLD_SIZE', '1')) > 1 and not active():
        if runtime.device.startswith('cuda'):
            device = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(device)
            runtime.device = f'cuda:{device}'
            backend = 'nccl'
        else:
            backend = 'gloo'
        dist.init_process_group(backend, timeout=timedelta(minutes=10))
        created = True
    try:
        yield
    finally:
        if created:
            dist.destroy_process_group()


@torch.no_grad()
def broadcast_module(module):
    if active() and module is not None:
        for value in module.state_dict().values():
            dist.broadcast(value, src=0)


@torch.no_grad()
def sum_gradients(parameters):
    """Local losses are divided by GLOBAL accumulation, so reduce by SUM."""
    parameters = list(parameters)
    if not active() or not parameters:
        return
    used = torch.tensor([p.grad is not None for p in parameters], device=parameters[0].device, dtype=torch.int32)
    dist.all_reduce(used)
    groups = {}
    for p, present in zip(parameters, used.cpu().tolist()):
        if present:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            groups.setdefault((p.device, p.dtype), []).append(p)
    for group in groups.values():
        flat = torch.cat([p.grad.reshape(-1) for p in group])
        dist.all_reduce(flat)
        offset = 0
        for p in group:
            p.grad.copy_(flat[offset:offset+p.numel()].view_as(p))
            offset += p.numel()


def sum_metrics(metrics):
    if active() and metrics:
        values = torch.stack([value.float() for value in metrics.values()])
        dist.all_reduce(values)
        for key, value in zip(metrics, values):
            metrics[key] = value / world_size() if key.endswith('grad_norm') else value


def gather_rng(value):
    if not active():
        return [value]
    values = [None] * world_size()
    dist.all_gather_object(values, value)
    return values


def should_stop(path, device):
    flag = torch.tensor(int(primary() and path.exists()), device=device)
    if active():
        dist.broadcast(flag, src=0)
    return bool(flag.item())

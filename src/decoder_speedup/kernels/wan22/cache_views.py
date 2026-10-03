"""Static inference forwards retaining cache-tail views instead of copying.

Derived from the original Wan2.2 decoder; operation order is unchanged.
No runtime AST transformation or dynamic code execution is used.
"""
import torch
from einops import rearrange
from ..vae2_2 import CausalConv3d, CACHE_T, ResidualBlock

def residual_forward(self, x, feat_cache=None, feat_idx=[0]):
    h = self.shortcut(x)
    for layer in self.residual:
        if isinstance(layer, CausalConv3d) and feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = layer(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = layer(x)
    return x + h

def decoder_forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :]
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        x = self.conv1(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv1(x)
    for layer in self.middle:
        if isinstance(layer, ResidualBlock) and feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)
    for layer in self.upsamples:
        if feat_cache is not None:
            x = layer(x, feat_cache, feat_idx, first_chunk)
        else:
            x = layer(x)
    for layer in self.head:
        if isinstance(layer, CausalConv3d) and feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = layer(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = layer(x)
    return x

def resample_forward(self, x, feat_cache=None, feat_idx=[0]):
    b, c, t, h, w = x.size()
    if self.mode == 'upsample3d':
        if feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = 'Rep'
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -CACHE_T:, :, :]
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None and (feat_cache[idx] != 'Rep'):
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None and (feat_cache[idx] == 'Rep'):
                    cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                if feat_cache[idx] == 'Rep':
                    x = self.time_conv(x)
                else:
                    x = self.time_conv(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                x = x.reshape(b, c, t * 2, h, w)
    t = x.shape[2]
    x = rearrange(x, 'b c t h w -> (b t) c h w')
    x = self.resample(x)
    x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
    if self.mode == 'downsample3d':
        if feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = x.clone()
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -1:, :, :]
                x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
    return x

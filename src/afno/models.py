"""AFNO latent Euler dynamics and a physical autoregressive FNO baseline."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes=4, ndim=1):
        super().__init__()
        if ndim not in (1, 2) or modes < 1:
            raise ValueError("ndim must be 1 or 2 and modes positive")
        self.ndim, self.modes, self.out_channels = ndim, modes, out_channels
        shape = (1 if ndim == 1 else 2, in_channels, out_channels) + (modes,) * ndim
        self.weight = nn.Parameter(torch.randn(*shape, dtype=torch.cfloat) /
                                   (in_channels * out_channels) ** 0.5)

    def forward(self, x):
        dims = tuple(range(-self.ndim, 0))
        xf = torch.fft.rfftn(x, dim=dims)
        out = xf.new_zeros(x.shape[0], self.out_channels, *xf.shape[2:])
        m = self.modes
        if xf.shape[-1] < m or (self.ndim == 2 and x.shape[-2] < 2*m):
            raise ValueError("Grid is too small for nonoverlapping Fourier modes")
        if self.ndim == 1:
            out[..., :m] = torch.einsum("bik,iok->bok", xf[..., :m], self.weight[0])
        else:
            out[..., :m, :m] = torch.einsum("bixy,ioxy->boxy", xf[..., :m, :m], self.weight[0])
            out[..., -m:, :m] = torch.einsum("bixy,ioxy->boxy", xf[..., -m:, :m], self.weight[1])
        return torch.fft.irfftn(out, s=x.shape[-self.ndim:], dim=dims)


class FourierBlock(nn.Module):
    def __init__(self, width, modes, ndim, extra_pointwise=False):
        super().__init__()
        conv = nn.Conv1d if ndim == 1 else nn.Conv2d
        self.spectral = SpectralConv(width, width, modes, ndim)
        self.local = conv(width, width, 1)
        self.extra = conv(width, width, 1) if extra_pointwise else None

    def forward(self, x):
        h = self.spectral(x) + self.local(x)
        if self.extra is not None:
            h = h + self.extra(x)
        return F.gelu(h)


class FNOMap(nn.Module):
    def __init__(self, in_channels, out_channels, width=64, modes=4, depth=4,
                 ndim=1, extra_pointwise=False):
        super().__init__()
        conv = nn.Conv1d if ndim == 1 else nn.Conv2d
        self.lift = conv(in_channels, width, 1)
        self.blocks = nn.Sequential(*[FourierBlock(width, modes, ndim, extra_pointwise)
                                      for _ in range(depth)])
        self.project = conv(width, out_channels, 1)

    def forward(self, x):
        return self.project(self.blocks(self.lift(x)))


def batch_dt(dt, x):
    dt = torch.as_tensor(dt, dtype=x.dtype, device=x.device)
    if dt.ndim == 0:
        dt = dt.expand(x.shape[0])
    if dt.shape != (x.shape[0],) or not torch.all(dt > 0):
        raise ValueError("dt must be positive, scalar or shape (batch,)")
    return dt


class AFNO(nn.Module):
    def __init__(self, channels=1, ndim=1, physical_params=1, latent_channels=16,
                 width=64, modes=4, depth=4, embedding_dim=16, variant="afno"):
        super().__init__()
        if variant != "afno":
            raise ValueError(f"Unknown variant: {variant}")
        self.ndim, self.variant = ndim, variant
        self.physical_params = physical_params
        kw = dict(width=width, modes=modes, depth=depth, ndim=ndim, extra_pointwise=True)
        self.encoder = FNOMap(channels, latent_channels, **kw)
        self.decoder = FNOMap(latent_channels, channels, **kw)
        self.embedding = nn.Sequential(nn.Linear(physical_params+1, embedding_dim),
                                       nn.GELU(), nn.Linear(embedding_dim, embedding_dim))
        self.field = FNOMap(latent_channels+embedding_dim, latent_channels,
                            width, modes, depth, ndim)

    def velocity(self, z, params, dt):
        dt = batch_dt(dt, z)
        if params.shape != (z.shape[0], self.physical_params):
            raise ValueError("Physical parameter shape mismatch")
        b = self.embedding(torch.cat((params, dt[:, None]), dim=1))
        b = b.reshape(*b.shape, *([1]*self.ndim)).expand(-1, -1, *z.shape[2:])
        return self.field(torch.cat((z, b), dim=1))

    def advance(self, z, params, dt):
        v = self.velocity(z, params, dt)
        return z + batch_dt(dt, z).reshape(-1, 1, *([1]*self.ndim)) * v

    def rollout(self, u0, params, dt, steps, return_latent=False):
        if steps < 1:
            raise ValueError("steps must be positive")
        z = self.encoder(u0)
        fields, latents = [], []
        for _ in range(steps):
            z = self.advance(z, params, dt)
            fields.append(self.decoder(z))
            if return_latent:
                latents.append(z)
        out = torch.stack(fields, dim=1)
        return (out, torch.stack(latents, dim=1)) if return_latent else out


class FNO(nn.Module):
    """Physical-space autoregressive control under the same data protocol."""
    def __init__(self, channels=1, ndim=1, physical_params=1, width=64, modes=4,
                 depth=4, **unused):
        super().__init__()
        self.ndim = ndim
        self.operator = FNOMap(channels+physical_params+1, channels, width, modes, depth, ndim)

    def step(self, u, params, dt):
        p = torch.cat((params, batch_dt(dt, u)[:, None]), dim=1)
        p = p.reshape(*p.shape, *([1]*self.ndim)).expand(-1, -1, *u.shape[2:])
        return self.operator(torch.cat((u, p), dim=1))

    def rollout(self, u0, params, dt, steps):
        states = []
        for _ in range(steps):
            u0 = self.step(u0, params, dt)
            states.append(u0)
        return torch.stack(states, dim=1)


def build_model(config):
    config = dict(config)
    name = config.pop("name", "afno")
    if name not in ("afno", "fno"):
        raise ValueError(f"Unknown model {name}")
    return (AFNO if name == "afno" else FNO)(**config)

"""Independent periodic pseudospectral solvers for Appendix A.

Evolution uses float64/complex128. The generator stores float32 snapshots.
Unreported initial conditions, forcing and domains are recorded in each config.
"""
from __future__ import annotations

import numpy as np
from scipy import fft


def wave_numbers(shape, lengths):
    return np.meshgrid(*[2*np.pi*fft.fftfreq(n, d=L/n)
                         for n, L in zip(shape, lengths)], indexing="ij")


def spectral_mask(shape):
    grids = np.meshgrid(*[fft.fftfreq(n)*n for n in shape], indexing="ij")
    mask = np.ones(shape, dtype=bool)
    for k, n in zip(grids, shape):
        mask &= abs(k) < n/3
    return mask


def smooth_random(rng, batch, shape, rms=0.5, bandwidth=5.0):
    axes = tuple(range(-len(shape), 0))
    noise = rng.normal(size=(batch, *shape))
    grids = np.meshgrid(*[fft.fftfreq(n)*n for n in shape], indexing="ij")
    k2 = sum(k*k for k in grids)
    spectrum = fft.fftn(noise, axes=axes) * np.exp(-k2/(2*bandwidth**2))
    spectrum[(slice(None), *([0]*len(shape)))] = 0
    field = fft.ifftn(spectrum, axes=axes).real
    norm = np.sqrt(np.mean(field**2, axis=axes, keepdims=True))
    return rms*field / np.maximum(norm, 1e-15)


def initial_conditions(config, seed, batch):
    rng = np.random.default_rng(seed)
    shape = tuple(config["resolution"])
    name = config["equation"]
    rms = config["initial_rms"]
    args = (rng, batch, shape, rms, config.get("initial_bandwidth", 5.0))
    if name not in ("burgers", "ks") or len(shape) != 1:
        raise ValueError("Only one-dimensional Burgers and KS are included")
    return smooth_random(*args)[:, None]


def ifrk4_step(y, nonlinear, linear, dt):
    """Lawson integrating-factor RK4, no exponentially growing inverse factors."""
    e, e2 = np.exp(dt*linear), np.exp(0.5*dt*linear)
    n1 = nonlinear(y)
    a = e2*(y+0.5*dt*n1)
    n2 = nonlinear(a)
    b = e2*y+0.5*dt*n2
    n3 = nonlinear(b)
    c = e*y+dt*e2*n3
    n4 = nonlinear(c)
    return e*y + dt/6*(e*n1+2*e2*(n2+n3)+n4)


def solve(config, initial):
    name, p = config["equation"], config["params"]
    shape, lengths = tuple(config["resolution"]), config["lengths"]
    ndim = len(shape)
    axes = tuple(range(-ndim, 0))
    k = wave_numbers(shape, lengths)
    k2 = sum(ki**2 for ki in k)
    mask = spectral_mask(shape)
    fwd = lambda u: fft.fftn(u, axes=axes)
    inv = lambda y: fft.ifftn(y, axes=axes)
    invreal = lambda y: inv(y).real
    obs_dt = config["observation_dt"]
    substeps = int(np.ceil(obs_dt/config["internal_dt"]-1e-12))
    dt = obs_dt/substeps
    steps = config["steps"]
    if name not in ("burgers", "ks") or ndim != 1:
        raise ValueError("Only one-dimensional Burgers and KS are included")
    y = fwd(initial[:, 0])
    readout = lambda y: invreal(y)[:, None]
    linear = -p["nu"]*k2 if name == "burgers" else k2-k2**2
    nonlinear = lambda y: -0.5j*k[0]*fwd(invreal(y)**2)*mask
    scheme = "cnab2" if name == "burgers" else "ifrk4"
    y *= mask
    previous_n = None
    snapshots = [readout(y)]
    for frame in range(steps):
        for _ in range(substeps):
            if scheme == "cnab2":
                n = nonlinear(y)
                explicit = n if previous_n is None else 1.5*n-0.5*previous_n
                y = ((1+0.5*dt*linear)*y+dt*explicit)/(1-0.5*dt*linear)
                previous_n = n
            else:
                y = ifrk4_step(y, nonlinear, linear, dt)
            y *= mask
        state = readout(y)
        if not np.isfinite(state).all():
            raise FloatingPointError(f"{name} solver failed at frame {frame+1}")
        snapshots.append(state)
    return np.stack(snapshots, axis=1)

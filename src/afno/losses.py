"""Spatial means approximate normalized domain integrals; time rollout is a sum."""
import torch
from torch.nn import functional as F
from .models import AFNO, batch_dt


def central_gradients(x, lengths):
    for dim, length in zip(range(x.ndim-len(lengths), x.ndim), lengths):
        dx = length / x.shape[dim]
        yield (torch.roll(x, -1, dim) - torch.roll(x, 1, dim)) / (2*dx)


def reconstruction_loss(pred, target, lengths, h1_weight):
    mse = F.mse_loss(pred, target)
    grad = sum(F.mse_loss(p, t) for p, t in zip(
        central_gradients(pred, lengths), central_gradients(target, lengths)))
    return mse + h1_weight*grad


def training_loss(model, states, params, dt, lengths, weights, h1_weight=1e-4,
                  pushforward_steps=0):
    # states: [batch, K+1, channel, *space]; all targets are ground truth.
    if not isinstance(model, AFNO):
        u = states[:, 0]
        with torch.no_grad():
            for _ in range(pushforward_steps):
                u = model.step(u, params, dt)
        pred = model.rollout(u, params, dt, states.shape[1]-1-pushforward_steps)
        roll = (pred-states[:, 1+pushforward_steps:]).square().flatten(2).mean(2).sum(1).mean()
        return roll, {"roll": roll.detach()}
    rec_w, flow_w, roll_w = weights
    zero = states.new_zeros(())
    rec = flow = roll = zero
    z0 = model.encoder(states[:, 0])
    if rec_w:
        rec = reconstruction_loss(model.decoder(z0), states[:, 0], lengths, h1_weight)
    if flow_w:
        z1 = model.encoder(states[:, 1])
        delta = batch_dt(dt, z0).reshape(-1, 1, *([1]*model.ndim))
        flow = F.mse_loss(model.velocity(z0, params, dt), (z1-z0)/delta)
    if roll_w:
        z = z0
        if pushforward_steps:
            with torch.no_grad():
                for _ in range(pushforward_steps):
                    z = model.advance(z, params, dt)
        for step in range(pushforward_steps+1, states.shape[1]):
            z = model.advance(z, params, dt)
            roll = roll + F.mse_loss(model.decoder(z), states[:, step])
    loss = rec_w*rec + flow_w*flow + roll_w*roll
    return loss, {"rec": rec.detach(), "flow": flow.detach(), "roll": roll.detach()}

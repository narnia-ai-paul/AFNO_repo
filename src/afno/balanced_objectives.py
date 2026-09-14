"""Recovery curriculum: explicit, scale-balanced early and full-rollout errors."""
import math

import torch
from torch.utils.checkpoint import checkpoint

from afno.losses import reconstruction_loss
from afno.models import batch_dt


def block_ranges(steps):
    start = 0
    for end in sorted({min(n, steps) for n in (5, 20, 50, 100, steps)}):
        yield start, end
        start = end


def alpha_at(epoch, arm):
    for stage in arm["stages"]:
        if epoch <= stage["through_epoch"]:
            return stage["early_weight"]
    raise ValueError("Epoch exceeds the declared curriculum")


def dual_score(metrics, comparator):
    """Both ratios must be <1. No horizon can hide behind the other average."""
    if metrics["nonfinite_trajectories"]:
        return None
    early = metrics["first5_mse"]/comparator["first5_mse"]
    full = metrics["mean_mse"]/comparator["mean_mse"]
    if not all(math.isfinite(v) and v >= 0 for v in (early, full)):
        return None
    return max(early, full)


def rollout_loss(model, states, params, dt, lengths, scales, alpha, mode="normalized",
                 rec_weight=.4, h1_weight=1e-4, activation_checkpoint=True):
    if mode not in ("normalized", "control") or not 0 <= alpha <= 1:
        raise ValueError("Invalid loss mode or early weight")
    if not all(math.isfinite(scales[k]) and scales[k] > 0 for k in ("early", "full")):
        raise ValueError("Training-only error scales must be finite and positive")
    steps = states.shape[1]-1
    if steps < 5:
        raise ValueError("The early task requires at least five future states")
    z = model.encoder(states[:, 0])
    z1 = model.encoder(states[:, 1])
    delta = batch_dt(dt, z).reshape(-1, 1, *([1]*model.ndim))
    flow = (model.velocity(z, params, dt)-(z1-z)/delta).square().mean()
    rec = states.new_zeros(())
    if rec_weight:
        anchor = states[:, torch.randint(states.shape[1], (), device=states.device)]
        rec = .5*(reconstruction_loss(model.decoder(z), states[:, 0], lengths, h1_weight)
                  + reconstruction_loss(model.decoder(model.encoder(anchor)), anchor, lengths, h1_weight))
    errors = []
    for step in range(1, steps+1):
        def advance_error(latent, physical_params, time_step, target):
            following = model.advance(latent, physical_params, time_step)
            return following, (model.decoder(following)-target).square().mean()
        args = (z, params, dt, states[:, step])
        z, error = checkpoint(advance_error, *args, use_reentrant=False) if activation_checkpoint else advance_error(*args)
        errors.append(error)
    errors = torch.stack(errors)
    early, full = errors[:5].mean(), errors.mean()
    if mode == "control":
        roll = torch.stack([errors[lo:hi].mean() for lo, hi in block_ranges(steps)]).mean()
    else:
        # Fixed scales come only from the common initial model's TRAIN errors.
        # Multiplication by s_full keeps physical loss units while correcting
        # relative task scale. Validation values never enter this loss.
        roll = scales["full"]*(alpha*early/scales["early"]+(1-alpha)*full/scales["full"])
    loss = rec_weight*rec + .5*flow + .5*roll
    return loss, {"rec": rec.detach(), "flow": flow.detach(), "early_mse": early.detach(),
                  "full_mse": full.detach(), "weighted_roll": roll.detach()}

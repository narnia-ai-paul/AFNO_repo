"""Increasing-horizon losses with reconstruction and temporal block balancing."""
import torch
from torch.utils.checkpoint import checkpoint

from afno.losses import reconstruction_loss
from afno.models import batch_dt


def horizon_at(epoch, stages):
    for stage in stages:
        if epoch <= stage["through_epoch"]:
            return stage["horizon"]
    raise ValueError("Epoch outside the declared curriculum")


def rollout_loss(model, states, params, dt, rec_weight, lengths, h1_weight,
                 activation_checkpoint=True, balance_blocks=False):
    """Differentiate through EVERY latent step, including the last loss to E(u0).

    0.5 * mean_t MSE equals the original 0.1 * sum_t MSE when K=5.
    Optional block balancing gives early and late intervals equal total weight.
    Checkpointing recomputes activations; it never detaches the latent trajectory.
    """
    z0 = model.encoder(states[:, 0])
    z1 = model.encoder(states[:, 1])
    delta = batch_dt(dt, z0).reshape(-1, 1, *([1] * model.ndim))
    flow = (model.velocity(z0, params, dt) - (z1-z0)/delta).square().mean()
    rec = states.new_zeros(())
    if rec_weight:
        anchor = states[:, torch.randint(states.shape[1], (), device=states.device)]
        rec = 0.5*(reconstruction_loss(model.decoder(z0), states[:, 0], lengths, h1_weight)
                   + reconstruction_loss(model.decoder(model.encoder(anchor)), anchor, lengths, h1_weight))
    steps = states.shape[1]-1
    time_weights = [1/steps]*steps
    if balance_blocks:
        blocks = list(block_ranges(steps))
        time_weights = [1/(len(blocks)*(hi-lo)) for lo, hi in blocks for _ in range(lo, hi)]
    z, roll = z0, states.new_zeros(())
    for step in range(1, states.shape[1]):
        # Bind the target explicitly: backward recomputation must not capture
        # the loop's final step/target through a mutable closure.
        def advance_and_error(latent, physical_params, time_step, target):
            following = model.advance(latent, physical_params, time_step)
            error = (model.decoder(following)-target).square().mean()
            return following, error
        args = (z, params, dt, states[:, step])
        if activation_checkpoint:
            z, error = checkpoint(advance_and_error, *args, use_reentrant=False)
        else:
            z, error = advance_and_error(*args)
        roll = roll + time_weights[step-1]*error
    loss = rec_weight*rec + 0.5*flow + 0.5*roll
    return loss, {"rec": rec.detach(), "flow": flow.detach(), "roll_mean": roll.detach()}


def block_ranges(steps):
    ends = sorted({min(n, steps) for n in (5, 20, 50, 100, steps)})
    start = 0
    for end in ends:
        yield start, end
        start = end


def eligible_score(metrics, initial, early_tolerance):
    if metrics["nonfinite_trajectories"]:
        return None
    if metrics["first5_mse"] > (1 + early_tolerance)*initial["first5_mse"]:
        return None
    return max(b["mse"] for b in metrics["blocks"])

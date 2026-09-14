# Method and training recipe

## Models

AFNO encodes the initial field once, evolves a latent state using an explicit Euler step, and decodes each future field:

```text
z₀ = Encoder(u₀)
zₖ₊₁ = zₖ + Δt · Field(zₖ, Embedding(parameters, Δt))
ûₖ₊₁ = Decoder(zₖ₊₁)
```

The latent state carries the trajectory forward; decoded predictions are not fed back into the encoder. The Euler step size is the dataset observation interval. The ground-truth PDE solver's internal timestep is a separate quantity.

FNO is the physical-space autoregressive baseline: each predicted field, together with the physical parameters and observation interval, becomes the input for the next prediction. Both models use the same generated data, initial fields, conditioning, spatial resolution, and validation metric.

The AFNO encoder, decoder and latent field use width 64, four Fourier blocks and four retained modes. Encoder and decoder blocks include the additional pointwise branch used by the original implementation. The latent state and parameter embedding each have 16 channels. FNO uses width 64, four blocks and four modes. Model sizes differ and are recorded in `assets/manifest.json`; this is not a parameter-matched comparison.

## Equations and data

| Property | Burgers | KS |
| --- | --- | --- |
| Equation | u_t + u u_x = ν u_xx | u_t + u u_x + u_xx + u_xxxx = 0 |
| Periodic domain | [0, 2π) | [0, 64) |
| Parameter | ν = 0.01 | No varying physical parameter |
| Observation interval | 0.01 | 0.1 |
| Prediction steps | 100 | 100 |
| Internal solver timestep | 0.0005 | 0.025 |
| Solver | Crank–Nicolson / Adams–Bashforth 2 | Integrating-factor RK4 |
| Reference / model grid | 1024 / 256 | 1024 / 256 |
| Training / validation trajectories | 1000 / 100 | 1000 / 100 |

Initial conditions are periodic, zero-mean smoothed Gaussian random fields with RMS 0.5 and spectral bandwidth 5. Each trajectory receives its own child seed from the split seed. The solver applies a two-thirds spectral mask. Evolution uses float64/complex128 and stored fields use float32. Burgers uses an explicit first nonlinear step to initialize AB2. All assumptions are explicit in the JSON recipes; the datasets are independently generated, not downloads of the paper's original data.

## Training

All stages use Adam with a cosine learning-rate schedule and batch size 32. Each stage starts a new optimizer and budget. FNO and AFNO use model initialization seed 0; AFNO fine-tuning seeds are 41011 and 42011.

| Stage | Epochs | Initial learning rate | Rollout horizon |
| --- | ---: | ---: | --- |
| FNO base | 500 | 1e-3 | 5 |
| AFNO base | 500 | 1e-3 | 5 |

Five-step base training samples a ground-truth starting point within each training trajectory and predicts the next five states using its own evolution. FNO minimizes the sum of the five spatial MSEs. AFNO's first 150 base epochs train initial-field reconstruction with an H1 penalty of 1e-4; the remaining 350 epochs use 0.5 times latent flow matching plus 0.1 times the summed rollout MSE. Flow matching compares the learned velocity with `(Encoder(u₁) − Encoder(u₀)) / Δt`.

The warmup uses horizon 5 for epochs 1–10, 20 for 11–25, 50 for 26–40, and 100 for 41–60. It gives equal weight to the available temporal blocks 1–5, 6–20, 21–50, and 51–100. Its loss is 0.4 reconstruction + 0.5 flow matching + 0.5 block-balanced rollout MSE. Reconstruction uses the initial field and a random true field in the training window.

Balanced training always starts at the initial state and predicts the full 100-step trajectory. Before this stage, the final warmup model is evaluated on **all training trajectories** to freeze its first-five error `s_early` and full-100 error `s_full`. With `alpha = 0.5`, the rollout term is:

```text
L_roll = s_full × [alpha × MSE_first5 / s_early
                  + (1 − alpha) × MSE_full100 / s_full]
L = 0.4 L_reconstruction + 0.5 L_flow + 0.5 L_roll
```

This increases the influence of small early errors without removing late errors. Every target, including step 100, contributes gradients through the entire latent trajectory. Activation checkpointing recomputes activations to save memory; it does not detach or truncate the trajectory. The normalizing errors are frozen on training data, not fitted to validation errors.

## Checkpoints and reported comparison

Warmup and balanced stages save both the final checkpoint and a validation-selected checkpoint, including their initial state as a candidate. Warmup selection guards first-five error while minimizing the worst temporal block. Balanced selection minimizes the worse of the first-five and full-horizon MSE ratios to FNO on the 256-point validation grid. Native-grid readout evaluation cannot change this selection.

The bundled AFNO figures and weights specifically use **final balanced epoch 40**, initialized from **final warmup epoch 60**. They do not silently substitute a selected checkpoint. FNO uses final base epoch 500. The `AFNO` label denotes this trained model; the curriculum is part of its training procedure.

At evaluation, initial conditions are Fourier-resampled from 1024 to 256 points. Models evolve for 100 steps at 256 points, and decoded fields are Fourier-resampled back to 1024 points. MSE uses float64 differences, averages over space, then over the indicated steps and trajectories. No future ground truth enters model evolution.

The included figure fields use trajectory 0, fixed by index before KS field export. Error curves use all 100 validation trajectories. Color limits are shared between models within a PDE, cover the complete displayed range, and are not clipped. The PDF supplement shows fixed indices 0–7. These are exploratory, single-seed validation comparisons with unequal model size and training compute; independent test and multi-seed confirmation remain outside this release.

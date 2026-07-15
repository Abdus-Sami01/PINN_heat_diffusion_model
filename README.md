# PINN Heat-Diffusion Model for Neon/LED Tube Thermal Behavior

Physics-informed neural networks for predicting the temperature profile of a
neon/LED tube along its length, and for estimating the convective heat transfer
coefficient from a handful of noisy sensor readings.

## Why

Instrumenting every tube prototype with a full thermal camera rig is expensive.
If you glue 3 thermocouples to a tube you get sparse, noisy point readings.
A PINN can fuse those sparse readings with the known physics (the heat equation)
and reconstruct the full temperature field AND estimate unknown physical
parameters like the convective cooling coefficient. No large dataset needed.

## The physics

1D transient heat equation with convective loss and an internal heat source:

```
dT/dt = alpha * d2T/dx2 - h * (T - T_ambient) + Q(x)
```

- `T(x,t)` temperature along the tube, degrees C
- `alpha = 1e-4` thermal diffusivity
- `h = 0.05` convective heat transfer coefficient (unknown in the inverse problem)
- `Q(x)` Gaussian heat source centred mid-tube (peak 2.0, width 0.15)
- `T_ambient = 20 C`, tube length `L = 1 m`, simulated for 50 s

Boundary conditions: insulated ends (Neumann, dT/dx = 0 at x=0 and x=L).
Initial condition: tube starts at ambient everywhere.

## Ground truth

`fdm_reference.py` solves the PDE with explicit finite differences
(forward Euler + central difference, CFL-safe step). The solver is validated
against the analytical Gaussian-spreading solution for pure diffusion in
`validate_fdm.py` (0.8% L2 error, PASS). Every PINN result below is graded
against this FDM solution, never against the PINN itself.

## Method

### Forward PINN (physics only, zero data)

`train_forward.py`. A small MLP maps (x, t) to T. Two hard constraints are
baked into the architecture instead of being soft loss penalties:

- inputs enter as `cos(k*pi*x/L)` features, so dT/dx is exactly zero at both
  tube ends - the Neumann BC cannot be violated, no BC loss term exists
- the output is multiplied by `(1 - exp(-t/tau))`, so T(x,0) is exactly
  ambient - the IC cannot be violated, no IC loss term exists

That leaves a single loss term (the PDE residual via autograd), which removed
the loss-balancing failure mode entirely. See FAILED_APPROACHES.md for the
soft-constraint versions that plateaued at 29% error and why.

Optimizer: Adam 12k steps with step decay, then L-BFGS fine-tune.

### Inverse PINN (sparse noisy data, unknown h)

`train_inverse.py`. Same architecture, but `h` becomes a learnable parameter
(softplus-wrapped to stay positive) optimized jointly with the network from a
deliberately wrong initial guess (0.15, i.e. 3x the true value). Loss =
PDE residual + 10x data misfit on the sensor readings. Sensor readings are
sampled from the FDM ground truth at 3 x-positions and 8 times, with 1 C
Gaussian noise added to mimic real thermocouples.

## Results

(to be filled from results/ once all runs complete)

## Honest limitations

- 1D along the tube, not 2D radial - no cross-section temperature structure
- ground truth is a numerical solver, not physical sensor data
- radiative losses ignored, convection linearized with a single constant h
- the heat source Q(x) is a synthetic Gaussian, not measured drive power
- the envelope time constant tau is fixed; in the forward model it is set to
  1/h which mildly encodes prior knowledge of the cooling rate (the inverse
  model uses a generic tau=15 so the true h does not leak into it)

## What I'd do differently / extensions

- 2D (x, r) model of the tube cross-section
- ensemble PINNs for uncertainty bands on the recovered h
- validate against a real tube with actual thermocouples
- swap the synthetic Q for drive-current-derived power density

## Repo layout

```
fdm_reference.py        ground truth FDM solver
validate_fdm.py         solver vs analytical solution check
problem.py              shared physical constants + sensor sampling
pinn_model.py           HardPINN (hard-constraint) + plain PINN
losses.py               PDE residual, BC/IC/data losses
train_forward.py        phase 2, forward problem
train_inverse.py        phase 3, inverse h recovery
sensitivity_study.py    h recovery vs sensor count x noise
baselines/
  least_squares_h.py    classical FDM-fit baseline for h
  data_only_nn.py       same net, no physics loss
  lstm_baseline.py      sequence model, no physics
plot_*.py               figure generation
figures/  results/      outputs
FAILED_APPROACHES.md    dead ends, so nobody retries them
```

## Run it

```
pip install -r requirements.txt
python3 validate_fdm.py
python3 train_forward.py
python3 train_inverse.py
python3 sensitivity_study.py
python3 baselines/least_squares_h.py
python3 baselines/data_only_nn.py
python3 baselines/lstm_baseline.py
```

Everything trains on CPU in minutes.

# Executed end-to-end experiment

On 2026-09-16, both `smoke.toml` and `full.toml` were executed from
`experiments/end_to_end_validation/`. The full result directory is
`data/output/end_to_end_validation/full/`. The full optimization was not replaced
by the reduced smoke run.

## Full configuration and optimization

The educated initial train is a smooth 0.25 + 0.35 sin² arch (radians), 64
readouts per direction, reconstructed using the existing cubic spline initializer.
The existing quadratic RF-phase initialization uses fraction 0.18. Three diffusion
directions and 72 retained states are used; TR=15 ms and TE=4 ms. Exact decoded
initial and optimized trains are saved, including the actual RF phases.

SLSQP converged successfully in **48 iterations**. Objective:
**1.0627056911 → 0.6414793525**, a **39.64% reduction**. Minimum constraint
residual was −3.81e−12 (roundoff). Optimization took 1855.66 s including derivative
compilation; the original full run took 1992.82 s on JAX CPU. This preserves the
established T1/T2/internal-equilibrium-M objective, which excludes MD and FA.

## Validation gates

Both sequences passed dictionary self-matching, direct phantom recovery and
noiseless full Cartesian acquisition/reconstruction. T1/T2/MD/FA RMSE was zero.
Relative complex-image error was about 1.22e−5, below the unchanged 1e−4 gate.
Doubling states from 72 to 144 produced identical signals for the two checked
dictionary endpoints in both sequences. Paired additive noise differed by at
most 3.14e−16 between sequences after subtracting their noiseless acquisitions.
No ambiguous dictionary matches were reported in any full-run condition.

## Image-domain results

The 16×16 two-compartment phantom uses physically valid prolate tensors. The
dictionary has 32 independent T1/T2/MD/FA/orientation combinations, with continuous
complex amplitude fitting. Its on-grid, phantom-specific nature is a limitation.
The acquisition uses the independently cross-validated NUFFT conventions,
analytically generated variable-density spirals and golden-angle rotation.
Reconstruction is radial-increment-weighted adjoint with unit impulse gain.

Noiseless global RMSE, **initial → optimized**:

| Interleaves/frame | T1 (ms) | T2 (ms) | MD (mm²/s) | FA |
|---|---:|---:|---:|---:|
| Cartesian reference | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 1 | 0 → 156.25 | 13.5785 → 13.3463 | 1.22952e−4 → 1.67705e−4 | 0.132583 → 0.237993 |
| 4 | 0 → 0 | 6.49519 → 11.7260 | 0 → 9.92157e−5 | 0 → 0.112673 |
| 16 | 0 → 0 | 4.33013 → 5.59017 | 0 → 0 | 0 → 0 |

The same qualitative parameter-specific outcome held with Gaussian k-space noise
SD=0.001 per real/imaginary channel (seed 2026). At one interleaf, only T2 RMSE
improved slightly; T1/MD/FA worsened. At four, T1 tied and T2/MD/FA worsened. At
16, T1/MD/FA tied and T2 worsened. Cartesian recovery remained exact on the grid.
The complete CSV includes both noise levels, density, MAE, RMSE, normalized RMSE,
bias and compartment metrics. Noise was fixed in absolute signal units, not SNR.

The joint sequence therefore **did not show general quantitative undersampling
superiority in this experiment**, despite improving the signal-level objective.
This is not evidence that RF-phase optimization alone causes degradation: both
flip angles and phases changed, and no comparable flip-only optimum was available.

## Information diagnostic

A separate converged central finite-difference diagnostic uses the existing
simulator and valid prolate tensor parameterization. Parameters are T1, T2,
external proton density, MD, FA and constant object phase, with orientation fixed.
This avoids the known-invalid legacy MD/FA derivatives without modifying the
optimizer or MRI equations. At the specified representative tissue, relative SD
bounds decreased for all five quantitative parameters, but those local bounds
did not predict the observed aliasing/matching error ordering. The FIM has rank
six; scaled condition numbers are about 3298 and 3183. Step-halving differences
are below 3.1e−9. Bounds remain conditional on the legacy diffusion signal model.

## Certification and reproducibility

Overall status: **PARTIAL**. The computational workflow completed, but known
tensor attenuation defects, uncalibrated wavevectors, limited orientation/grid,
adjoint reconstruction artifacts and external analytical UEE disagreement remain.
See the full component classification in `report/certification.json` and the
complete scientific report in `report/experiment_report.md`.

The outputs preserve original configuration, fully resolved defaults, source
snapshot, commit/dirty state, package/backend versions, arrays, optimizer history,
raw k-space and reconstructions. No validated MRI physics was changed. The
expanded checked-in full TOML makes previously defaulted choices explicit and
was checked to reproduce the configuration of this run.

Reproduce with the commands in the experiment README. `replot.py` regenerates
analysis from saved arrays; `verify_outputs.py` runs the post-run state/noise
checks without repeating optimization. Result directories are intentionally
git-ignored; keep or archive them separately when sharing the experiment.

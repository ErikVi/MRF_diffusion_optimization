# Quantitative undersampling recovery

Run the bounded experiment with:

```sh
python -m pip install -e ".[experiments]"
mrf-compare-undersampling --config experiments/undersampling/quantitative.toml
```

The forward-only command remains available. The new command reconstructs complex
frames, then fits quantitative maps as a separate operation. It uses the existing
tensor EPG simulator for both the phantom and dictionary, including the known
tensor attenuation defects. It does **not** establish corrected diffusion physics.

## Scope and dictionary budget

The default is a small synthetic unoptimized baseline: eight flip angles with the
existing quadratic RF phase train, repeated in two diffusion direction blocks.
The phantom has two compartments, but the dictionary uses independent Cartesian
products of the parameter axes, not a two-label lookup table.

| Axis | Default samples | Interpretation |
|---|---:|---|
| T1 | 2: 750, 1250 ms | Restricted on-grid validation |
| T2 | 2: 70, 90 ms | Restricted on-grid validation |
| MD | 2: 0.0007, 0.001 mm²/s | Prolate tensor construction |
| FA | 2: 0.2, 0.7 | Prolate tensor construction |
| Principal axis | 2: x, y | Discrete antipodally equivalent orientations |
| Density and constant object phase | No grid | Analytic complex least-squares scalar |
| Signal length | 16 complex samples | Actual frame count, not invented timestamps |

The product is 32 entries. Raw complex128 signals occupy 8,192 bytes; the
conservative host working estimate is 434,176 bytes with 256-voxel batches.
This estimate includes matching temporaries but excludes JAX compiler/device
memory and experiment images. A 30×30×20×20×30 grid would have 10.8 million
entries: 2.76 GB for 16 frames, or 622 GB for 3,600 frames, before workspace.
Such a dictionary is not generated.

The API reports the planned size before simulation and enforces entry and memory
limits. Physical duplicate tensors are removed after budgeting. Do not increase
limits to bypass an impractical grid: first design and validate coarse-to-fine
fitting or a restricted tensor/orientation model. Current sample counts establish
an on-grid integration test; they are **not** adequate resolution for unrestricted
tissue estimation. No off-grid accuracy or general tensor-orientation recovery is
claimed. Two encoding directions do not establish identifiability of six general
tensor components.

## Physical parameterization

`reconstruction/tensor_dictionary.py` calls `simulate_mrf_signal(..., tensor=True)`
at equilibrium M=1 for each T1/T2/tensor candidate. It uses the same
`MRFSequence`, preparation settings, direction order, precision, state count
and sampling options as the phantom. Tensor construction is the validated prolate
family in [forward phantom conventions](forward_phantom.md). MD and FA determine
valid tensor eigenvalues; fitting selects a tensor and then calculates
MD=trace(D)/3 and standard eigenvalue FA using
`diffusion.parameterization.tensor_invariants`. It does not use the incorrect
legacy trace-only MD helper. An unrestricted tensor model would need a different,
validated candidate set and additional directional information.

## Complex fitting

For a voxel time series y and dictionary signal d, minimize over one complex
scalar beta:

```text
beta = (dᴴ y) / (dᴴ d)
residual = ||y - beta*d||₂
selected entry = argmax |dᴴ y|² / (dᴴ d)
relative proton density = |beta|
constant object phase = arg(beta)
```

This follows the complex dictionary-matching approach in
[Ma et al., 2013](https://www.nature.com/articles/nature11971);
the least-squares scalar is explicitly evaluated. This implementation was written
independently. Normalization removes amplitude from entry selection, not temporal
phase. There is one beta over **all** readouts and directions, never independent
per-frame phase correction. RF phase already acts inside EPG and is not reapplied.
Time-varying object phase, motion, B0 drift and receive coils are not modeled.

Density means the external relative proton-density multiplier defined by the
forward model. It is not an estimate of the legacy equilibrium-M parameter.
The scalar can absorb constant receive/object phase. Under aliasing, its phase
and amplitude can be biased; no exact physical object-phase recovery is claimed
for undersampled images.

Matching is batched complex least squares. Spatially correlated reconstruction
noise and undersampling/model errors are not whitened. The result is not claimed
to be a statistically optimal estimator. Outputs include selected tensor, index,
complex scalar, relative residual, validity and normalized-correlation margin.
Zero or nonfinite signals are invalid; tied/near-tied atoms are flagged ambiguous,
not silently presented as uniquely determined tissue. Ambiguous candidates remain
in map error calculations with an explicit count, but block superiority labels.

## Reconstruction and calibration

Three choices are obtained by configuring the comparison:

- `density_compensation="none"`, `unit_impulse_gain=false`: raw adjoint.
- A named radial-increment or Pipe-Menon DCF, gain disabled: weighted adjoint.
- DCF plus `unit_impulse_gain=true`: weighted adjoint with explicit operator gain calibration.

Calibration divides each reconstructed frame by the central value of AᴴWA delta.
It uses the trajectory/operator only, with no phantom truth or dictionary input.
This compensates sample-count/DCF scale when estimating density. It does not undo
blurring or aliasing, and does not make the adjoint an inverse. The applied
divisors are saved. Cartesian reference uses the complete grid, no DCF, and no
gain correction. Forward k-space is never density weighted.

Heesterbeek's analytical UEE computes linearized parameter-error predictions using
temporal PSFs and local derivatives; it is not another nonlinear dictionary
reconstruction. It is deliberately not substituted for this direct experiment.
There is no additional external reconstruction needed to run the direct comparison.

## Controlled sequence comparison

The baseline is explicit in the main TOML. Optional
`comparison.flip_optimized_config` and `joint_optimized_config` name complete
forward TOMLs relative to `input_directory`. Each must select
`sequence.rf_phase_mode="optimized"` and an existing NPZ containing its paired
`flip_angles_rad` and `rf_phases_rad`. Preparation arrays, if present, must
also be paired. The optimized label describes supplied provenance; the loader
does not establish that optimization was performed or invent an optimized train.

For this controlled comparison, timing, gradient settings, diffusion directions,
simulation options and frame/direction ordering must agree with the baseline.
Unequal-duration protocols require a separately designed equal-budget comparison.
Only the train arrays may vary. Sequence/config hashes, full physics/options and
actual train arrays are saved. Phantom, dictionary axes, coordinates, interleaves,
reconstruction and k-space noise realization are shared across sequences.
Identical integer seeds and sample shapes provide paired noise, independently
generated per real/imaginary sample with the specified channel SD. Different
sampling conditions have different shapes and are not claimed to share identical
noise vectors. Noiseless conditions are run only once; noisy seeds remain
separate records. A single seed is a deterministic test, not a statistical study.

The default sweep uses a complete Cartesian reference and spiral interleaf counts
1, 4, 16, with channel SD 0 and 0.001 and seed 2026. The synthetic spiral has 18
samples per arm. Saved H×W/K is a sample-count ratio, **not a calibrated acceleration
factor**: non-Cartesian samples are nonuniform and may repeat. Sixteen interleaves
are a high-sampling condition, not asserted to be fully sampled. Finite readout,
hardware timing and gradient waveform effects are not modeled by this spatial
snapshot acquisition.

## Mandatory validation gate

Before any undersampled condition for any supplied sequence:

1. Every dictionary atom must uniquely match itself.
2. Direct phantom signals must recover on-grid tissue and density.
3. Complete Cartesian noiseless acquisition/reconstruction must reproduce complex
   frames within the configured relative tolerance.
4. Reconstructed tissue maps must recover known values; density and object phase
   allow documented NUFFT error.
5. Zero/invalid/ambiguous reference voxels fail the gate.

Failure writes diagnostics and stops before undersampling interpretation. Off-grid
phantoms deliberately fail exact tissue recovery: enrich the candidate grid or
validate a different estimator; do not simply loosen thresholds to conceal
quantization error. Matching and synthesis from the same model constitute an
on-grid consistency test, not independent validation of the underlying physics.

## Errors and artifacts

Each parameter reports MAE, RMSE, signed bias and NRMSE=RMSE/RMS(truth), globally
and separately per tissue compartment. NRMSE is null for zero RMS truth; absolute
FA error remains meaningful near zero. Units stay separate and no combined score
is produced. Invalid voxel counts accompany finite-voxel metrics. Background is
excluded using the known simulation support; no ground-truth compartment label
is supplied to the matcher. Orientation and density differences define evaluation
compartments alongside T1/T2 and tensor values.

Outputs include:

- `dictionary_budget.json`, `validation_gates.json`, and `status.json`.
- Shared ground-truth tensor/maps/support and per-sequence complex signals/frame labels.
- Per-sequence dictionaries, actual trains, full settings and source hashes.
- Per-condition complex k-space, trajectories, reconstructed frames, tensor and
  scalar estimates, errors, residuals, ambiguity flags and gain divisors.
- `metrics.json` and `metrics.csv`, including region-wise results.
- PNG (300 dpi) and vector PDF truth/recovered/error triptychs for T1/T2/MD/FA/density.
- Per-parameter RMSE-versus-interleaves plots, separating noise levels and showing
  Cartesian references. Multi-seed curves show mean RMSE; individual results are retained.

Per-condition comparisons label lower/higher RMSE versus baseline only when all
voxels are valid and unambiguous. These are descriptive finite-condition results,
not significance tests or evidence of general sequence superiority.

## Provenance and remaining limits

| Component | Relationship to external work |
|---|---|
| Spiral scheduling, PSF/DCF concepts | Methodologically inspired by [D.G.J. Heesterbeek's GPL-3.0 repository](https://github.com/imphys/MRF_undersampling_optimization); independently implemented wrappers/equations, with prior compatible helper comparisons in acquisition documentation |
| SigPy NUFFT/spiral/Pipe-Menon routines | Calls to the installed BSD-licensed SigPy library |
| Complex matching, tensor grid, metrics, comparison runner | Independently implemented here; complex inner-product methodology referenced to Ma et al. |
| External EPG, T1/T2-only fitting, UEE linearization/optimizer | Not copied or used |
| Directly adapted external source in this task | None |

Known D01/D02 and other preserved scientific discrepancies remain documented in
[validation](validation.md). They limit physical interpretation even when fitting
and synthesis agree. Missing optimized archives are recorded explicitly; a
baseline-only run cannot answer whether the optimized joint train is better.
Before substantive conclusions, supply actual optimized inputs, resolve the
tensor-model defects, validate scan-specific state convergence, broaden grids and
phantoms, and repeat noise realizations. No source physics or optimization
behavior was changed in this extension.

# Architecture and migration

## Sequential worked example

`experiments/complete_worked_example/` orchestrates the existing components with
a pre-phantom phase selection gate. `run_experiment.py` handles initialization,
optimization and information; `spatial_validation.py` assembles existing forward,
reconstruction and fitting functions; `analysis.py` regenerates PNG/SVG and
Markdown/HTML reports from saved arrays; `output.py` records source/test provenance.
The example's AUDIT.md was written before implementation.

Reusable `optimization/phase_search.py` preserves angle coefficients while fitting
existing generator phases and searching their fractions. It validates family
names and coefficient dimensions, records every evaluation, and deterministically
selects finite minima. It does not introduce a new objective or repair the legacy
phase-comparison runner's malformed baseline.

`information/tensor_diagnostic.py` contains the previously tested converged
finite-difference diagnostic, with explicit tissue/scaling inputs and raw
covariance/variance outputs. The old experiment adapter remains compatible and
its original test remains unchanged. No EPG or tensor attenuation code moved.

## End-to-end experiment orchestration

`experiments/end_to_end_validation/run_experiment.py` composes the established
optimization, sequence, phantom, simulation, acquisition and reconstruction APIs.
It adds no MRI equations. `information_diagnostic.py` now adapts the shared
`information.tensor_diagnostic` calculation to the original experiment's tissue
and file-output contract; this remains a diagnostic, not a replacement objective.
`analysis.py` creates common-scale figures and reports; `replot.py` regenerates
them from saved data without recomputation. All package physics remains unchanged.
The run stores configuration/source provenance, sequences, optimization, signals,
phantom, acquisition, reconstruction, parameter maps, metrics, figures and report
in separate directories.

This refactor preserves the numerical behavior characterized at commit
`0f76ee49a7ed524431e45abd8fbbbbcbbb698946`. It does not certify or correct the
scientific model. Known discrepancies remain executable strict-xfail tests.

## Spatial encoding extension

The independent `encoding` package accepts complex 2-D images and normalized
`(ky,kx)` coordinates. Trajectory generation, weight estimation, sampling and PSF
diagnostics are separate APIs. `reconstruction.images` composes a weighted
adjoint without implicit normalization. Neither layer imports EPG, MRF simulation
or experiment settings. SigPy is an optional, lazily loaded CPU backend.
See [acquisition conventions and validation](acquisition.md) for array shapes,
units, interpolation tolerances and external provenance. The historical phantom
workflow remains unchanged; no image-domain fitting or sequence optimization is
introduced by this extension.

## Dependency direction

```text
experiments -> optimization -> information -> simulation -> epg
                    |              |              |
                    +---------- sequence ---------+
                                   |
                     diffusion tensor/metric utilities

experiments -> visualization, io
undersampling experiment -> reconstruction, phantoms, optional external tools
reconstruction/images -> encoding -> NumPy, optional SigPy
```

Physics does not import sequences, optimizers, plots, files, or experiments.
Sequence utilities do not import simulation. Information does not import
optimization. The broad `mrf_diffusion.api` module is a convenience export surface,
not a dependency that physics modules should import. `tests/test_architecture.py`
checks these boundaries. Importing an experiment does not load arrays, run an
optimizer, import plotting/NUFFT dependencies, or create an output directory.

## Repository map

| Location | Responsibility and main functions |
|---|---|
| `src/mrf_diffusion/epg/states.py` | `(3,K)` initialization and F+0 extraction |
| `epg/rf.py` | `apply_rf_rotation` |
| `epg/gradients.py` | Positive/negative order shifts and truncation |
| `epg/relaxation.py` | `relax_and_shift`; exponential recovery without diffusion |
| `epg/diffusion.py` | Scalar/tensor combined relaxation, attenuation and shift operators |
| `diffusion/tensor.py` | Historical scaled template and tensor-scale grids |
| `diffusion/metrics.py` | FA, explicitly named legacy MD/metric derivatives |
| `sequence/definition.py` | `MRFSequence`, immutable `SequenceSettings`, `SimulationOptions` |
| `sequence/phase.py` | Analytical phase trains and piecewise quadratic phase |
| `sequence/bspline.py` | Shared Cox-de Boor basis, evaluation and least-squares fit |
| `sequence/parameterization.py` | Knot construction, coefficient initialization and shared decoding |
| `simulation/tissue.py`, `simulation/api.py` | `TissueParameters`, `simulate_mrf_signal` object input boundary |
| `simulation/signal.py` | Differentiable scalar/tensor MRF kernels and magnitude wrappers |
| `information/jacobian.py` | Scalar AD and historical tensor-metric projection |
| `information/fisher.py`, `parameters.py` | Information matrices and named parameter orders |
| `information/crlb.py` | Direct inverse, relative bounds and optional conditioning diagnostics |
| `optimization/objectives.py`, `evaluation.py` | Ensemble objective and reported precision bounds |
| `optimization/constraints.py`, `settings.py`, `solver.py` | Sequence constraints, settings, SLSQP execution with local history |
| `experiments/` inside package | Import-safe optimization, phase comparison, spline benchmark and phantom workflows; settings/TOML loader |
| `reconstruction/` | Dictionary construction, matching and scale lookup |
| `reconstruction/images.py` | Explicit weighted-adjoint image reconstruction |
| `encoding/trajectory.py` | Supplied/generated trajectories and named rotation schedules |
| `encoding/nufft.py`, `density_compensation.py`, `psf.py` | Independent sampling operator, weight estimators and PSF diagnostics |
| `phantoms/checkerboard.py` | Historical tensor phantom generator |
| `visualization/plots.py`, `io/` | Plots, array/metadata output, HDF5 inspection |
| root `experiments/*/default.toml` | Runnable experiment configurations with file-relative paths |
| `tests/` | Analytical, finite-difference, regression and architecture checks |
| `tests/reference/` | Frozen numerical values and provenance, never rewritten by tests |
| `tools/capture_validation_reference.py` | Explicit capture of a candidate to a new path |
| `docs/`, `data/input/`, `data/output/` | Model/conventions/thesis, supplied inputs, generated outputs |

## Migration from original files

| Old file or function | New home/name | Reason |
|---|---|---|
| `EPG_blocks_jaxcode.py` | `epg`, `diffusion`, `sequence`, `simulation`, `information`, `optimization` | Separated physics, trains, signal, information and objectives; no monolithic implementation remains |
| `epg_rf`, `epg_grad`, `epg_mgrad` | `apply_rf_rotation`, `apply_positive_gradient_shift`, `apply_negative_gradient_shift` | Names identify actual operators and shift sign |
| `epg_grelax*` | `relax_diffuse_scalar_and_shift`, `relax_diffuse_tensor_and_shift` | These combine three operations; naming only them relaxation would hide diffusion/shifts |
| `output_generator*` | `simulate_scalar_signal`, `simulate_tensor_signal` | Returns signal, not files |
| `output_plotter*` | `scalar_signal_magnitude`, `tensor_signal_magnitude` | Never plotted; they calculate magnitude |
| `lb_in_param_holistic*` | `*_sequence_precision_bounds` | Returned values are normalized standard-deviation bounds, not covariance matrices |
| `objective_holistic*` | `*_ensemble_objective` | Aggregates tissue costs |
| `MD_from_tensor`, `dMD_dD`, `dFA_dD` | `legacy_mean_diffusivity`, `legacy_mean_diffusivity_gradient`, `legacy_fractional_anisotropy_gradient` | Explicitly flags known incorrect/inconsistent definitions |
| `OPTIMIZATION.py` | `experiments/optimization.py` plus optimization/settings/constraints/solver and visualization/io | Removes global callback state and mixed plotting/optimization responsibilities |
| `BESTMETHOD.py` | `experiments/phase_comparison.py` | A phase-fraction scan, not a generally established best method |
| `BSPLINEPROFILER.py` | `experiments/bspline_benchmark.py` | Benchmark entry point with warm-up and device synchronization |
| `UNDERSAMPLING.py` | `experiments/undersampling.py`, `reconstruction/`, `phantoms/` | Exposes nested operations; missing NUFFT support remains an explicit dependency |
| `FILEREADER.py` | `io/hdf5.py` / `mrf-inspect-hdf5` | Explicit file argument replaces machine-specific filename |
| root thesis PDF | `docs/thesis.pdf` | Scientific reference alongside model documentation |

The removed core `general_3directions` and `test_diffusion_preparation` alternatives
were uncalled/incomplete, not alternate validated simulators. Unused phantom
`Zero_padding`, `Fun_mask`, the one-line `full_run` wrapper, commented alternatives,
debug imports and plotting experiments are retained in Git history, not active APIs.
No compatibility shim for the old monolithic import is shipped; migrate callers
using the table above. Old positional kernel arguments retain their ordering;
keyword names are intentionally clearer and sequence settings are trailing arguments.

## Canonicalization and boundaries retained deliberately

`evaluate_bspline` is the canonical evaluator. The old matrix/vmap names are
aliases, not independent implementations: legacy comparison at degrees 1, 2 and 3,
repeated knots and endpoints differed by at most 1.11e-16. Fitting shares the basis
but retains its distinct fit grid (B01). Objectives and reporting share one phase
decoder; initialization and constraints are independently callable.

Combined diffusion-relaxation operators stay combined to preserve recovery and
attenuation order. Scalar and tensor operators are NOT merged: tests establish
that they currently differ scientifically. Experimental tensor-scale factors are
called scales; they must not be confused with measured MD or FA.

## Configuration and API scope

`TissueParameters` contains physical tissue values; `MRFSequence` contains train
arrays; `SequenceSettings` contains timing/preparation/encoding; `SimulationOptions`
contains sampling/state-count choices. Optimizer/constraint and experiment settings
are separate. The host API validates train shape; it is not a complete physical or
hardware validator. Low-level array kernels remain available for JAX transforms.

TOML can override nested settings. Unknown fields fail; omitted values preserve
historical defaults. Relative input/output paths resolve beside the TOML file.
Without a file, the documented `data/` paths resolve under the caller's directory.
Use a TOML file for portable reproduction. No missing data are fabricated.

The original objective and constraints use cubic splines; optimization/phase
experiments reject other degrees until that API is extended and validated.
The phase benchmark can evaluate other degrees directly. A new dataset or sequence
is not automatically scientifically valid merely because it can be configured.

## Forward phantom extension

See [forward phantom](forward_phantom.md) for geometry, tensor maps, unique-tissue
simulation, frame acquisition and adjoint layers. `mrf-phantom` assembles these
components; historical matching remains under `mrf-legacy-phantom`.

## Quantitative recovery extension

`reconstruction/tensor_dictionary.py` builds bounded tensor-model dictionaries;
the legacy `dictionary.py` remains unchanged. `quantitative.py` fits complex time
series, `metrics.py` evaluates maps, and `calibrated.py` provides optional
operator-only gain calibration around the existing adjoint.
`experiments/quantitative_undersampling.py` controls comparisons and mandatory
reference gates. `visualization/quantitative.py` exports triptychs and curves.
Configuration is `experiments/undersampling/quantitative.toml`.
See [scientific details](quantitative_undersampling.md).

Physics kernels and optimization are unchanged.

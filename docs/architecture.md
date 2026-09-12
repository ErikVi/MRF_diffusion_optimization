# Architecture and migration

This refactor preserves the numerical behavior characterized at commit
`0f76ee49a7ed524431e45abd8fbbbbcbbb698946`. It does not certify or correct the
scientific model. Known discrepancies remain executable strict-xfail tests.

## Dependency direction

```text
experiments -> optimization -> information -> simulation -> epg
                    |              |              |
                    +---------- sequence ---------+
                                   |
                     diffusion tensor/metric utilities

experiments -> visualization, io
undersampling experiment -> reconstruction, phantoms, optional external tools
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

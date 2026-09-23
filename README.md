# Diffusion-Enhanced MRF Sequence Optimization

Research code for the MSc thesis "On the robust optimization of diffusion-enhanced
MRF sequences through flip angle and phase train design with comparative phase
modulation strategies" (Erik Višnar, TU Delft).


The package implements Extended Phase Graph (EPG) simulation, diffusion tensor
modelling, B-spline sequence parameterization, JAX-based Fisher information
analysis, and constrained sequence optimization for Magnetic Resonance
Fingerprinting (MRF). It also provides an executable end-to-end pipeline that
connects sequence design all the way through to image-domain quantitative
recovery.

**Read before using any results:** the codebase preserves nine documented
scientific discrepancies inherited from the original thesis implementation.
Passing the test suite does not certify tensor diffusion physics or
diffusion-parameter CRLBs. See [Known scientific discrepancies](#known-scientific-discrepancies)
below and `docs/validation.md` for full detail.

## Repository layout

| Path | Contents |
|---|---|
| `src/mrf_diffusion/` | The installable science package: EPG, diffusion, sequence, information, optimization, encoding, reconstruction, phantoms, experiments, visualization, io |
| `experiments/` | Runnable experiment configurations (`*.toml`) and workflow scripts, one directory per experiment type |
| `docs/` | Project documentation: physical model, validation status, architecture, conventions, units, and the thesis PDF |
| `tests/` | Analytical, finite-difference, regression, and architecture checks |
| `tools/` | Utility scripts, including capture of new validation reference snapshots |
| `data/` | Input requirements and generated outputs; the completed `complete_worked_example/full` run is versioned, other runs are ignored |

## Install and test

Python 3.12, from the repository root:

```sh
python -m venv .venv
# activate the environment for your shell
python -m pip install -r requirements-forward-validation.txt
python -m pip install -e .
python -m pytest -q
```

Expect **178 passed, 9 expected failures**. The nine expected failures are the
documented scientific discrepancies below; an unexpected pass fails the suite
and must be investigated, not ignored.

For a quicker physics-only loop:

```sh
python -m pytest tests/test_epg.py tests/test_diffusion.py -q
```

## Quick simulation example

```python
import jax.numpy as jnp
from mrf_diffusion.sequence.definition import MRFSequence, SimulationOptions
from mrf_diffusion.simulation.tissue import TissueParameters
from mrf_diffusion.simulation.api import simulate_mrf_signal

sequence = MRFSequence(
    flip_angles=jnp.array([0.2, 0.3, 0.25, 0.4]),  # radians
    rf_phases=jnp.array([0.0, 0.1, 0.2, 0.3]),
)
tissue = TissueParameters(
    t1_ms=1000.0, t2_ms=80.0,
    equilibrium_magnetization=1.0, diffusion=0.001,  # mm^2/s
)
signal = simulate_mrf_signal(tissue, sequence, tensor=False)  # (2, 4) real/imag
```

For tensor simulation, pass a `(3, 3)` tensor with `tensor=True` and a
`SimulationOptions(direction_count=..., state_count=...)`. Tensor diffusion uses
the preserved legacy model; see the limitations below before relying on it.

## The science, by module

All paths are relative to `src/mrf_diffusion/`.

| Responsibility | Module |
|---|---|
| EPG states, RF, gradients, relaxation | `epg/` |
| Diffusion attenuation, tensors, metrics | `epg/diffusion.py`, `diffusion/` |
| Timing, phase trains, B-splines | `sequence/` |
| MRF signal generation | `simulation/` |
| Trajectories, NUFFT, density compensation, PSFs | `encoding/` |
| Weighted-adjoint image reconstruction | `reconstruction/images.py` |
| Jacobians, Fisher information, CRLB | `information/` |
| Objectives, constraints, solver | `optimization/` |
| MSc workflows and settings | `experiments/` |
| Phantom generation and dictionary matching | `phantoms/`, `reconstruction/` |
| Plots and result files | `visualization/`, `io/` |

## Experiment entry points

```sh
mrf-optimize --config experiments/optimization/default.toml --dry-run
mrf-compare-phases --config experiments/phase_comparison/default.toml --dry-run
mrf-benchmark-splines --config experiments/bspline_benchmarks/default.toml --dry-run
mrf-phantom --config experiments/undersampling/default.toml --dry-run
mrf-compare-undersampling --config experiments/undersampling/quantitative.toml --dry-run
mrf-inspect-hdf5 --help
```

`--dry-run` prints the resolved configuration without loading data or running
any computation. Edit the TOML files, or their `[sequence]`, `[simulation]`,
`[solver]` tables, to change a run, then drop `--dry-run` to execute.

Some original MSc experiments require external arrays that are not distributed
with this repository (`fa_array_initial.npy`, `DIFFPREPARATION/*.npy`, the
external `UEEphase_DH` module). Without them, the original thesis experiments
cannot be reproduced from this checkout; see `data/input/README.md`. No
synthetic substitutes are ever presented as a reproduction of thesis results.

## Complete sequential worked example

[`experiments/complete_worked_example/`](experiments/complete_worked_example/README.md)
shows initialization → flip-angle-only optimization → phase-family comparison and
selection → final phase refinement → information → phantom → undersampling → maps.
The [fresh audit](experiments/complete_worked_example/AUDIT.md) records the actual
supported methods and scientific limitations before implementation.

```sh
python -m pytest -q --junitxml=data/output/complete_worked_example/test_suite.xml
python experiments/complete_worked_example/run_experiment.py --config experiments/complete_worked_example/smoke.toml
python experiments/complete_worked_example/run_experiment.py --config experiments/complete_worked_example/full.toml
```

Browse `data/output/complete_worked_example/full/report/REPORT.html` (or `REPORT.md`).
The report, figures and arrays from this completed run are included in Git:
[read the illustrated report](data/output/complete_worked_example/full/report/REPORT.md)
or [browse all saved results](data/output/complete_worked_example/full/).
Figures and arrays are organized chronologically in numbered stage directories.
`summary.csv` gives the sequence/phase progression; `reconstruction_metrics.csv`
gives parameter-specific global and compartment errors. The finite phase search
uses the existing conditional T1/T2/internal-M objective, not a new MD/FA objective.
This is distinct from the earlier joint-optimization experiment below.
The [executed results](docs/complete_worked_example.md) document lower objective
but no general undersampling improvement over the initial sequence.

## The end-to-end experiment

`experiments/end_to_end_validation/` is a single executable pipeline connecting
educated sequence initialization, joint flip-angle/RF-phase optimization,
finite-difference information diagnostics, a tensor phantom, complex NUFFT
acquisition, image reconstruction, and quantitative dictionary matching:

```sh
python -m pytest -q
python experiments/end_to_end_validation/run_experiment.py --config experiments/end_to_end_validation/smoke.toml
python experiments/end_to_end_validation/run_experiment.py --config experiments/end_to_end_validation/full.toml
```

Each run needs a fresh `run_name` and writes arrays, sequences, signals, the
phantom, acquisition and reconstruction data, parameter maps, metrics, figures,
and a generated `report/experiment_report.md` to
`data/output/end_to_end_validation/<run_name>/`. That directory is not tracked
by git. Regenerate the report and figures from saved arrays, without rerunning
the optimizer, with:

```sh
python experiments/end_to_end_validation/replot.py data/output/end_to_end_validation/full
```

The executed full run is summarized in `docs/end_to_end_validation.md`:
optimization converged and both Cartesian recovery gates passed, but the
optimized sequence did not show general undersampling superiority over the
initial one. The optimization objective excludes diffusion parameters (MD, FA);
this is a known limitation, not an oversight. Read the generated experiment
report and component certification before interpreting any run's results, and
read `docs/undersampling_validation.md` alongside it: tensor attenuation is not
certified, the matching dictionary is restricted and on-grid, and spiral
adjoint reconstruction is not a true inverse.

## Known scientific discrepancies

Nine defects from the original thesis implementation are preserved, documented,
and covered by strict `xfail` tests, so an unexpected pass fails the suite:

| ID | Issue |
|---|---|
| G01 | Negative-gradient boundary overwrites arriving F+0 instead of refocusing it |
| D01 | Tensor diffusion assigns identical attenuation to two pathways that should differ |
| D02 | A zero initial wavevector still attenuates F+0, contradicting "gradients off during TE" |
| M01 | Mean diffusivity is computed as trace(D) instead of trace(D)/3 |
| M02 | The stated MD derivative and the automatic-differentiation gradient of MD disagree |
| M03 | The fractional anisotropy derivative has the wrong sign |
| B01 | Spline fitting and experiment reconstruction use inconsistent sample-grid coordinates |
| I01 | The tissue-weighted CRLB aggregation never advances past the first tissue's weight |
| I02 | The "general" bounds function returns three parameters, not the documented five |

Full evidence, required decisions, and literature cross-references for each are
in `docs/validation.md`.

## Documentation index

| Document | Covers |
|---|---|
| `docs/scientific_model.md` | The implemented physics, stage by stage, including retained defects |
| `docs/validation.md` | Full test coverage, the discrepancy table, and regression tolerances |
| `docs/architecture.md` | Module dependency direction and migration from the original scripts |
| `docs/conventions.md` | Sign, unit, and array-shape conventions |
| `docs/units.md` | Units and default values used throughout |
| `docs/acquisition.md` | Spatial encoding API, NUFFT conventions, and its independent validation |
| `docs/forward_phantom.md` | The forward phantom pipeline connecting acquisition to the tensor simulator |
| `docs/quantitative_undersampling.md` | Tensor-dictionary fitting and controlled undersampling comparisons |
| `docs/undersampling_validation.md` | Cross-validation against an external reference implementation |
| `docs/end_to_end_validation.md` | Results of the executed full pipeline run |
| `docs/complete_worked_example.md` | Sequential flip-angle/phase selection example and measured results |
| `docs/thesis.pdf` | The MSc thesis itself |

## Design principle

This package separates reusable science from the original MSc experiments.
Numerical behavior is preserved exactly, including the nine discrepancies
above: refactoring changes structure, not physics. No scientific defect was
silently corrected during the architectural refactor described in
`docs/architecture.md`.

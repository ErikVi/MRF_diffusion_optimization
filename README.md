# Diffusion-enhanced MRF sequence research

EPG simulation, diffusion modeling, B-spline sequence design, JAX derivatives,
Fisher information and constrained optimization for Erik Višnar's [MSc thesis](docs/thesis.pdf).

This package separates reusable science from the original experiments. **Numerical
behavior is preserved, including nine documented scientific discrepancies.** Passing
tests does not certify tensor diffusion or diffusion-parameter CRLBs. Read
[validation](docs/validation.md) before using results for new research.

## Install and test

Python 3.12 was used for validation. From this repository:

```sh
python -m venv .venv
# Activate the environment for your shell.
python -m pip install -r requirements-validation.txt
python -m pip install -e .
python -m pytest -q
```

Pinned requirements reproduce the tested CPU dependency set. Broader package
dependency ranges do not imply every version/platform is tested. Known defects use
strict xfail markers; unexpected passes require review. Tests do not need the
missing experiment arrays and never recapture their numerical references.

## Simulate without external files

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
signal = simulate_mrf_signal(tissue, sequence, tensor=False)  # (2,4), real/imag
```

For tensor simulation supply a `(3,3)` tensor and `tensor=True`, with
`SimulationOptions(direction_count=..., state_count=...)`. This uses the preserved
legacy tensor model, with the limitations documented below. For AD/JIT use array
kernels in `mrf_diffusion.simulation.signal` directly.

## Navigate the science

All paths in this table are relative to `src/mrf_diffusion/`.

| Responsibility | Modules |
|---|---|
| EPG states, RF, gradients, relaxation | `epg/` |
| Diffusion attenuation, tensors and metrics | `epg/diffusion.py`, `diffusion/` |
| Timing, phase trains, B-splines | `sequence/` |
| MRF signals | `simulation/` |
| Jacobians, FIM, CRLB | `information/` |
| Objectives, constraints, solver | `optimization/` |
| MSc workflows and settings | `experiments/` |
| Phantom generation and dictionary matching | `phantoms/`, `reconstruction/` |
| Plots and result files | `visualization/`, `io/` |

Root `tests/` contains analytical, finite-difference, regression and architecture
checks. Root `experiments/*/default.toml` contains runnable configurations.
Documentation: [architecture/migration](docs/architecture.md),
[scientific model](docs/scientific_model.md), [conventions](docs/conventions.md),
[units and defaults](docs/units.md), [validation](docs/validation.md).

## Experiment entry points

```sh
mrf-optimize --config experiments/optimization/default.toml --dry-run
mrf-compare-phases --config experiments/phase_comparison/default.toml --dry-run
mrf-benchmark-splines --config experiments/bspline_benchmarks/default.toml --dry-run
mrf-phantom --config experiments/undersampling/default.toml --dry-run
mrf-inspect-hdf5 --help
```

`--dry-run` prints complete configuration without data loading or computation.
Edit TOML values or nested tables such as `[sequence]`, `[simulation]`, `[solver]`.
Relative paths resolve beside the configuration file. Remove `--dry-run` to execute.
Modules also support `python -m mrf_diffusion.experiments.optimization`, etc.

Install optional packages with `python -m pip install -e ".[experiments]"`.
See [required inputs](data/input/README.md): the original NumPy trains and external
`UEEphase_DH` module are absent, so published experiments cannot be reproduced from
this checkout alone. No fake replacements are provided.

Optimization records coefficients, trains, solver status/history, settings and
package versions, plus diagnostic plots. Phase comparison records its explicitly
legacy baseline and three actual bounds. Benchmark timing excludes compilation
and synchronizes the device; traced memory is Python memory, not device memory.
Phantom reconstruction remains an unvalidated historical workflow.

Former root scripts and monolithic imports are retired. Use the
[migration table](docs/architecture.md#migration-from-original-files) for new names.
No scientific defect was silently corrected during this architectural refactor.

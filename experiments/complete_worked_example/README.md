# Complete worked example

Start with [AUDIT.md](AUDIT.md). This is a new sequential example, not a rerun of
`end_to_end_validation` (which jointly optimized flip angles and phases).

The example performs:

1. Educated smooth flip-angle initialization with zero RF phase and baseline J/F/CRLB.
2. Existing constrained SLSQP flip-angle-only optimization.
3. Existing historical phase-family generation, cubic projection and fraction search,
   at exactly the same optimized flip angles.
4. Objective-based family selection **before phantom simulation**, followed by a
   finer fraction search within that family (not unrestricted joint optimization).
5. Three-sequence information/signal comparison, tensor checkerboard, Cartesian
   recovery gates, golden-angle spiral undersampling, complex matching and metrics.
6. Chronological PNG/SVG figures, Markdown/HTML report and machine-readable results.

## Run

From the repository root in the validated Python environment:

```sh
python -m pip install -e ".[experiments]"
python -m pytest -q --junitxml=data/output/complete_worked_example/test_suite.xml
python experiments/complete_worked_example/run_experiment.py --config experiments/complete_worked_example/smoke.toml
python experiments/complete_worked_example/run_experiment.py --config experiments/complete_worked_example/full.toml
```

On this Windows workspace use `../validation-venv/Scripts/python.exe`. If the
default pytest temporary directory has ownership restrictions, add
`-p no:cacheprovider --basetemp=../pytest-worked-UNIQUE` using a new temporary name.
The runner requires the passing test XML named by `test_report` in the TOML and
copies it into the results. Expected failures remain expected failures, not passes.

The full experiment command above is the single execution entry point. It runs
everything including figures/report. It refuses to overwrite an existing result
directory: change `run_name` for a fresh repeat. Analysis can be repeated cheaply:

```sh
python experiments/complete_worked_example/analysis.py data/output/complete_worked_example/full
```

Expected CPU budget is minutes to tens of minutes depending on JAX compilation
and hardware. This is a practical 40-readout, 48-state, 16×16 example; the previous
64-readout joint optimization took about half an hour. Smoke uses 24 readouts,
32 states, 8×8 images, three SLSQP iterations and coarse phase grids. Its optimizer
is intentionally iteration-limited; smoke is not a scientific optimum.

Executed on 2026-09-23: the full example completed in **544.26 seconds** on CPU.
SLSQP converged in nine iterations. See [the executed results](../../docs/complete_worked_example.md).
Saved evidence can be checked without rerunning simulation or optimization:

```sh
python experiments/complete_worked_example/verify_outputs.py data/output/complete_worked_example/full
```

## Configuration and outputs

`full.toml` and `smoke.toml` are standalone configurations. The full run compares
zero phase plus quadratic/linear/sinusoidal/alternating generator families at 33
fractions, then 129 local refinement points (plus the winning point). It uses
three unit diffusion directions, two tissues, 32 independent dictionary entries,
1/4/16 interleaves and noise SD 0/0.001 per complex channel (seed 2026).
Both configurations save fully resolved package defaults as `data/resolved.json`.
Sequence settings under `optimizer.sequence` are authoritative for every stage;
placeholder forward-runner train settings are not used as experiment sequences.

Outputs: `data/output/complete_worked_example/<run_name>/` with numbered stages
`00_audit` through `11_final_comparison`, plus `data` and `report`.
Open **`report/REPORT.html`** or **`report/REPORT.md`** for the complete illustrated
story. `report/figures.json` indexes all figures. Each figure lives beside its
stage's data. `summary.csv` records the sequence progression and phase candidates;
`reconstruction_metrics.csv` records all map errors globally and per compartment.
Results are git-ignored: archive/share the whole run directory separately.

Dependencies are the existing NumPy/SciPy/JAX/SigPy/Matplotlib stack. No external
trajectory binary, external EPG implementation or documentation framework is needed.
The source snapshot, commit/dirty state, versions, backend, test XML and config
are recorded with the run. PNG and SVG plots can be recreated without optimization.

## Scientific scope

The established objective uses conditional T1/T2/internal-M relative SD bounds;
it **does not optimize MD/FA**. Separate converged finite-difference diagnostics
use physical MD/FA coordinates of valid prolate tensors, external density and
constant object phase. They are not replacement objectives or certified MRI bounds.
The final sequence is flip-angle SLSQP followed by selected-family fraction
refinement; a finite-grid minimum is not an unrestricted phase optimum.

Raw generated phases and applied spline phases are both saved. RF phase enters
EPG once. Object phase and density multiply the resulting complex images once.
Matching uses one complex amplitude over the entire fingerprint. Cartesian gates
must pass for all sequences before spiral acquisition. Noise is identical across
sequences at each sampling condition, but their SNRs need not be equal.

Read `docs/validation.md` and `docs/undersampling_validation.md`: the nine known
discrepancies, especially tensor attenuation and legacy information limitations,
are retained. A completed run does not resolve them or establish clinical validity.
The external Heesterbeek repository remains a methodological acquisition reference;
no external source code is copied by this example.

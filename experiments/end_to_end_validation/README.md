# End-to-end model-consistency experiment

From the repository root, in the validation environment:

```sh
python -m pytest -q
python experiments/end_to_end_validation/run_experiment.py --config experiments/end_to_end_validation/smoke.toml
python experiments/end_to_end_validation/run_experiment.py --config experiments/end_to_end_validation/full.toml
```

Install the project and `requirements-forward-validation.txt` first. The runner
requires an unused run directory and never overwrites a previous run. Change
`run_name` for a repeat. Results are under `data/output/end_to_end_validation/`.
Resolved defaults are saved in `config/resolved.json`; the input TOML controls
the run size, initial arch, iteration budget, sampling/noise conditions and seed.
The full configuration also exposes tissue, tensor, dictionary, trajectory and
optimizer choices. The checked-in expanded defaults match the executed full run.

The smooth sin² starting train is an educated initialization, not a historical
optimized artifact. The established quadratic phase method jointly optimizes
angle spline coefficients and phase curvature. The existing objective only
conditions on fixed diffusion and optimizes T1/T2/internal equilibrium M bounds.
The separate finite-difference MD/FA information diagnostic does not change it.

Reference recovery is a mandatory gate. The report must be read alongside
`docs/undersampling_validation.md`: tensor attenuation remains uncertified,
the dictionary is restricted and on-grid, and spiral adjoint reconstruction is
not an inverse. Passing the pipeline does not certify physical diffusion.

Raw arrays permit replotting without optimization. `analysis.make_report`
accepts the saved root, configuration, truth/support, metrics, gates and optimizer
summary. Matplotlib writes both 300-dpi PNGs and vector PDFs. Error/color scales
are shared across sequence/sampling comparisons for each quantitative parameter.

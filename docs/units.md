# Units and parameter inventory

| Quantity | Units / convention |
|---|---|
| T1, T2, TE, TR, interval duration | milliseconds; relaxation uses ratios in ms |
| D and tensor elements | mm^2/s; diffusion converts duration to seconds |
| Flip angle, RF phase | radians; no degree conversion inside kernels |
| Magnetization, signal, noise SD | normalized signal units |
| Wavevector step | interpreted as rad/mm in analytical tests; calibration unresolved |
| b / b tensor | s/mm^2 under the angular-wavevector interpretation |
| Directions | raw Cartesian coefficients; no automatic normalization |
| MD | physically mm^2/s and trace(D)/3; legacy function returns trace(D) |
| FA | dimensionless, for valid tensors in [0,1] |
| Spline coordinates | dimensionless sample coordinates |
| FIM entries | inverse product of the corresponding parameter units |
| Relative SD bound | dimensionless |

The values called wavevector steps are not hardware gradient amplitudes in mT/m.
No gamma, pulse-width waveform integration or explicit 2*pi factor calibrates them
in this repository. Do not label their magnitudes as scanner gradient strength
without resolving the thesis/experimental conventions. Analytical tests state their
angular-wavevector assumption explicitly.

## Important defaults and where to change them

| Current value | Scientific meaning | Classification | Explicit location |
|---|---|---|---|
| TE=4, TR=15 ms | Evolution timing | Sequence | `SequenceSettings` |
| `[pi/2,pi,-pi/2]`, phase 0 | Diffusion preparation pulses | Sequence | `SequenceSettings.preparation_angles`, `preparation_rf_phase` |
| pi, phase 0 | Optional inversion | Sequence | `SequenceSettings.inversion_*` |
| 612/50 scalar, 300/100 tensor | Preparation/readout wavevector steps | Sequence, calibration uncertain | `SequenceSettings.*wavevector` |
| order step 1; preparation always unit | EPG lattice shift | Sequence/model | `SequenceSettings.order_step`; preparation protocol still fixed-unit |
| 9 raw direction vectors | Cartesian encoding list; diagonals length sqrt(2) | Sequence | `SequenceSettings.directions` |
| initial Z0=1 | Starting magnetization independent of M | Modeling assumption | `SequenceSettings.initial_longitudinal_magnetization` |
| scalar K=150; tensor API K=20 | Retained coherence states | Numerical | `SequenceSettings.scalar_state_count`, `SimulationOptions.state_count` |
| experiment K=5 | Aggressive truncation, convergence unestablished | Experiment | experiment `simulation` section |
| include inversion: optimization true; other workflows false | Prep/readout protocol | Experiment | `SimulationOptions.include_inversion` |
| sample stride32, offset0, disabled | Post-simulation subsampling | Experiment | `SimulationOptions` |
| sigma=10^-1.65 | Gaussian SD per real channel | Noise/model | `information.fisher` sigma argument; historical objectives use default |
| M=.7/.7/.8, T1=1000/1000/1500, T2=70/70/100 | Tissue ensemble | Experiment | `TissueEnsemble.scalar_parameters` |
| three explicit symmetric tensors | Diffusion ensemble | Experiment | `TissueEnsemble.diffusion_tensors` |
| weights all1, tissue weights1/3 | Cost weighting | Optimization/experiment | `TissueEnsemble` |
| N400, knot setting50, cubic | Optimization train parameterization | Experiment | `ExperimentSettings`; objective currently cubic only |
| phase fraction .18; piecewise (.3,.04,-.03) | Starting phase | Experiment | `ExperimentSettings` |
| flip-angle limits pi/18 to pi/3, adjacent difference .02 | Sequence design constraints | Optimization | `ConstraintSettings` |
| coefficient bounds +/-1000; split .2..8, curvature +.01/-.01 | Search domain | Optimization | `ConstraintSettings` |
| SLSQP, maxiter1000; callback delta .01 after >1000 iterations | Solver/legacy stop rule | Optimization | `SolverSettings` |
| phase scan N170, .01..1, 400 candidates, 9 directions | Phase experiment | Experiment | phase-comparison defaults/TOML overrides |
| benchmark N300, knot20, 10000 repeats | Runtime benchmark | Experiment | spline-benchmark defaults/TOML overrides |
| tensor sweep .0005..001, 1000 samples, tissue1500/90/.9 | Optimization diagnostic plot | Experiment | `ExperimentSettings.plot_*` |
| phantom tiles11x11 of11pixels, radius fraction .7 | Spatial phantom | Experiment | `PhantomSettings` |
| phantom T1 750/1250, T2 70/90; dictionary geometric grids30 | Reconstruction phantom/grid | Experiment | `PhantomSettings` |
| tensor scales0..1; yy/zz factor .1+.9*scale | Historical grid parameterization, not MD/FA | Experiment/model | settings ranges; `make_legacy_tensor_grid` equation |
| noise off; SD .005; label SNR200 | Optional spatial complex noise | Experiment | `PhantomSettings`; label does not set SD |
| optional random_seed=None | Original stochastic behavior retained | Experiment | `PhantomSettings`; set an integer to seed NumPy when reproducing |
| input/output directories and filenames | Artifact locations | Experiment | TOML paths, `data/input/README.md` |
| 1/2, 1/3, 1/12, pi, 1000 | RF/diffusion equations and ms-to-s conversion | Mathematical/unit | Remain in equations; not tuning parameters |

Some model choices remain equations, not knobs: tensor cross-term defects, the MD
definition, fixed preparation unit shifts, and spline endpoint semantics require
reviewed scientific/API changes. Configuration is not a mechanism to hide fixes.

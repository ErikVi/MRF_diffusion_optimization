# Fresh audit before implementation — 2026-09-23

Audited working tree: clean `1098950` (local `main`). This document was written
before implementing this example. The preceding joint-optimization experiment is
complete and is neither rerun nor presented as this new sequential experiment.
Sources inspected: AGENTS.md, README, the current docs collection (including the
thesis methods, initialization and validation appendix), current source tree,
experiment configurations and scientific/architecture tests. Historical result
directories are outputs, not source or initialization inputs.

## Current components and exact reuse

| Layer | Current modules/functions | Contract |
|---|---|---|
| Sequence | `sequence.definition`: MRFSequence, SequenceSettings, SimulationOptions | Equal angle/phase arrays in radians; TE/TR in ms; static direction/state/sampling/inversion choices |
| Spline | `sequence.bspline`: fit_bspline_coefficients, evaluate_bspline; `parameterization`: make_legacy_knots, initialize_sequence_parameters, decode_sequence_parameters | Cubic, coefficient count len(knots)-4; historical fit grid linspace(0,N,N), evaluation arange(N) |
| Optimization | `experiments.optimization.build_problem`; `optimization.solver.optimize_sequence`, constraints/settings/objectives/evaluation | SLSQP with JAX derivatives, L1 or sum-of-squares L2, local callback history |
| EPG | `epg.states`, rf, gradients, relaxation, diffusion | (3,K) complex F+,F-,Z; echo is F+0; positive shifts truncate |
| Simulation | `simulation.signal.simulate_tensor_signal`, `api.simulate_mrf_signal` | RF followed by TE evolution, echo, TR-TE evolution; independent prepared direction blocks |
| Information | jacobian, fisher, crlb, parameters | Scalar order T1,T2,D,M; legacy tensor order T1,T2,M,FA,MD |
| Valid tensor family | `diffusion.parameterization.axisymmetric_tensor`, tensor_invariants | Prolate PSD tensors; physical MD=trace/3; eigenvalue FA |
| Spatial model | phantoms geometry/maps; `forward_undersampling.build_phantom`; `simulation.image_series.simulate_phantom_image_series` | Unique T1/T2/D simulation; images (frame,y,x); external density and object phase applied once |
| Trajectory | encoding.trajectory; `forward_undersampling.build_trajectories` | (frame,interleaf,sample,2), (ky,kx) cycles/pixel; Cartesian/supplied/generated spiral |
| Acquisition | encoding.series.acquire_image_series; NufftOperator | Complex unweighted forward NUFFT; Gaussian noise SD per real/imag channel |
| Reconstruction | reconstruct_with_impulse_gain | Explicit DCF weighted adjoint, optionally divided by operator impulse gain; not an inverse |
| Estimation | generate_tensor_dictionary, estimate_dictionary_size, match_complex_dictionary | Bounded independent grid; one fitted complex amplitude per complete fingerprint; MD/FA derived from selected tensor |
| Validation | reference_recovery_gate, truth_and_regions, evaluate_parameter_maps | All-atom/direct/Cartesian recovery before undersampling; global and compartment MAE/RMSE/NRMSE/bias |

Existing `experiments/end_to_end_validation/information_diagnostic.py` supplies a
tested converged central-difference diagnostic on the valid tensor family. Its
reusable calculation will be extracted with the old adapter retained and its
numerical regression test unchanged. No signal equations will be copied.

## Actual scientific pipeline

Spline coefficients -> actual flip/phase trains -> diffusion preparation
(pi/2, pi, -pi/2), with two TE evolution intervals -> direction-specific EPG
RF/TE/echo/TR-TE -> real/imaginary temporal signal -> information or spatial
mapping -> density * exp(i*object_phase_map) -> NUFFT -> weighted adjoint ->
complex dictionary matching -> tensor-derived quantitative maps -> errors.
Default TE=4, TR=15 ms; tensor wavevectors 300 (preparation), 100 (readout),
assumed rad/mm but not scanner-calibrated. Initial Z0=1 regardless of internal M.
No inversion is needed in this example. The first three directions are unit x,y,z.
They support the restricted prolate/orientation-grid experiment, not general DTI.

## Phase methods: two different interfaces

`generate_phase_train` has four historical families, parameterized by fraction:
quadratic cumulative +/-2n degrees, linear cumulative +/-10 degrees, sinusoidal
1000*sin(2*pi*(n/N+fraction)) degrees, alternating cumulative
(-1)^n*1000*fraction degrees. Outputs are radians. The quadratic/linear branches
have a reset gap; do not silently replace them with continuous ideal functions.

The optimization decoder separately supports zero phase, direct free-form spline,
double-cumulative spline (`quadratic`), and a piecewise quadratic with a floored
split index (`quadratic malleable`). The decoder's quadratic is NOT the historical
quadratic generator. Floored split derivatives are not a reliable continuous
optimization coordinate. The `continuing` piecewise mode also resets.

The existing phase-comparison runner fits the four generator trains to cubic
splines and scans fraction, but its zero baseline concatenates an incorrect
number of phase coefficients. It must not be called for this controlled example.
Use the existing generators, fitter and objective with equal-length coefficient
blocks; preserve exactly the optimized angle coefficients, without refitting.
Unknown generator names currently return zeros: validate names at the new boundary.

### Predefined selection and final refinement (before phantom outcomes)

The new experiment will compare zero phase and the four **spline-projected
historical families**, using a configured coarse fraction grid and the unchanged
weighted L1 objective. The minimum objective selects the family; ties retain
configured order. All angle coefficients, tissues, timing and directions are
identical. Then refine fraction in that same family on a denser configured grid,
including the coarse winner so the refinement cannot discard it. This is the
existing finite-grid phase-search methodology, not a new gradient method for a
discontinuous fraction. Save generated AND applied phases and projection errors.
No free-form/joint SLSQP refinement is appended: that would change the selected
family. Decoder methods are documented but not conflated with generator families.
Thus this example is **flip-angle SLSQP then family selection/fraction refinement**,
not a global, unrestricted or jointly optimized phase optimum.

## Objective and precision limitations

The current tensor objective selects F[:3,:3] BEFORE inversion, and aggregates
relative standard-deviation bounds for T1,T2,internal equilibrium M only. It does
not optimize MD/FA despite five weights and the thesis's broader description.
L1 is the weighted sum; L2 is a sum of squared bounds without the thesis's outer
square root. Use L1 and preserve it. Default sigma=10^-1.65 per real channel.
Constraints currently use pi/18 <= angle <= pi/3, adjacent step <=0.02 rad,
coefficient bounds +/-1000, not the thesis's example +/-100 and pi/180 step.
SLSQP uses SciPy's default ftol=1e-6 and no objective reformulation.

Report conditional T1/T2/M bounds separately from finite-difference diagnostic
T1/T2/external density/MD/FA/object-phase bounds. The latter uses valid prolate
tensor perturbations, fixed orientation and step-halving convergence. These are
uncertainty diagnostics of the current signal model, not certified physical CRLBs
and never the phase-selection objective. Save raw J, F, covariance and scaling.
External density scales the whole signal; it is not internal equilibrium M.

## Initialization and thesis check

Thesis section 5.2.2 (PDF p22) used a truncated already-optimized Sommer train;
Figure 5.4 (PDF p23) depicts it. No raw historical array is present under
data/input (only README files). The thesis does not specify a reproducible
three-arch analytical initialization. Do not fabricate historical provenance or
start from the prior optimized outputs. Use the previous example's documented
smooth sin-squared arch idea, with smaller excursion for the shorter readout
budget, fit using the current initializer and check actual decoded constraints.
Start at zero RF phase to isolate flip-angle optimization. This is an educated
example initialization, not reproduction of thesis Figure 5.4.

## Acquisition and fitting status

The encoding namespace, not the suggested acquisition namespace, is established.
SigPy expects centered grid-index coordinates; NufftOperator scales cycles/pixel
by image dimensions. Ideal normalization is 1/sqrt(HW). Golden rotation is
-pi*(3-sqrt(5)) per acquired arm. DCF radial_increment is explicitly
r*abs(delta r), first sample zero per arm; it is not universal density compensation.
RF phase enters EPG once; signal phase, static object phase and trajectory angle
are distinct. All matching retains complex temporal phase with one complex scale.

Cross-validation in docs/undersampling_validation.md establishes acquisition
coordinate/weight/NUFFT agreement and a constrained no-diffusion signal common
case. Analytical UEE does not reproduce full image reconstruction error and is
not a certification of diffusion/FA/phase physics. Generated spirals have no
certified scanner-readout-time connection. Retain the existing operator.
External Heesterbeek GPL-3.0 code is not copied in this task. Existing provenance
and frozen comparisons remain authoritative for what was adapted/reimplemented.

## Tests and risks

Previous recorded full suite: 171 passing, 9 strict expected failures; rerun before
this example. EPG/RF/relaxation, tensor invariants, JAX/finite differences, splines,
frozen regressions, Fourier dense oracles, phantom caching/phase, matching/gates,
architecture and external common-case checks are present. Relevant files are
tests/test_epg.py, test_diffusion.py, test_bspline.py, test_information.py,
test_initialization.py, test_regression.py, test_architecture*.py,
test_acquisition.py, test_forward_phantom.py, test_quantitative_recovery.py,
test_undersampling_crossvalidation.py and test_end_to_end_diagnostic.py.

| ID | Preserved discrepancy / status |
|---|---|
| B01 | Spline fitting and experiment evaluation grids differ |
| D01 | Tensor pathway cross terms cancel; nonzero-order isotropic scalar equivalence fails |
| D02 | Tensor gradient-off zero-order attenuation persists |
| M01/M02/M03 | Legacy MD trace and metric derivatives inconsistent/invalid |
| G01 | Negative gradient overwrites arriving F+0 |
| I01 | Scalar tissue weighting uses first tissue weight |
| I02 | Tensor precision objective excludes MD/FA |

These are not new regressions and will not be repaired to improve this example.
Finite-state checks must use the actual new sequences. Phantom reference recovery
is an on-grid consistency test with 32 independent dictionary combinations, not
off-grid/clinical accuracy. High spiral sampling remains a weighted adjoint,
not guaranteed artifact-free. Local minima, finite fraction-grid resolution,
spline projection, fixed orientation diagnostic and correlated parameters remain.
Overall physical certification must remain PARTIAL unless independent evidence
resolves those limitations; execution alone cannot establish it.

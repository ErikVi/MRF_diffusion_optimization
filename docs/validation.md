# Scientific validation foundation

## Scope and execution

This suite tests the unchanged `EPG_blocks_jaxcode.py` at baseline commit
`0f76ee49a7ed524431e45abd8fbbbbcbbb698946`. It does not import the experiment
scripts, require the missing experiment arrays, or change optimization behavior.
No existing scientific source file was modified.

Use Python 3.12 in a fresh environment, then from the repository root:

```sh
python -m pip install -r requirements-validation.txt
python -m pytest -q
```

The pinned versions describe the validation environment, not the environment used
for the thesis. Tests select CPU and x64. Missing JAX is an installation failure,
not an automatic skip. Only the core scientific dependency set is installed.

Validated result: **63 passed, 9 expected failures, no unexpected failures or
skips**, 102.28 seconds including compilation, on Windows / Python 3.12.14 /
JAX 0.11.1 CPU. References were captured and verified in separate processes.

For a quicker physics-only loop:

```sh
python -m pytest tests/test_epg.py tests/test_diffusion.py -q
```

`pytest.ini` uses `xfail_strict=true`. Nine known-discrepancy tests execute their
assertions but are expected to fail. An unexpected pass FAILS the suite and must
be reviewed. Import errors and unrelated exceptions are not hidden by their
`raises=AssertionError` markers. To expose their full failing assertions:

```sh
python -m pytest -m discrepancy --runxfail -q
```

Passing the suite means the established valid checks and legacy characterizations
are preserved; it does NOT mean the entire scientific model is correct.

## Scientific contract and references

- EPG array `(3,K)`: rows F+, F-, Z; column is nonnegative order. F+0 is Mx+iMy.
- RF flip angles/phases are radians. A positive x-axis rotation sends +z to -y.
- Time arguments are milliseconds; diffusivity/tensor elements are mm^2/s.
- Analytical b tests interpret k as angular wavevector in rad/mm. This is a
  stated assumption, not a calibration of the experiment's physical gradients.
- FIM tests use independent real/imaginary Gaussian channels with SD sigma.
- Scalar Jacobian ordering is [T1,T2,D,M]; generalized ordering is nominally
  [T1,T2,M,FA,MD]. Only the first three generalized columns are physically
  validated here; the FA/MD coordinate conversion is not accepted as valid.
- "Relative standard-deviation bound" denotes sqrt(CRLB)/abs(parameter).

Sources checked against the preceding audit and the local thesis:

1. [Hargreaves, RAD229 B2](https://web.stanford.edu/class/rad229/Notes/B2-ExtendedPhaseGraphs.pdf),
   slides 17 and 20 (RF matrix), 39 (diffusion and physical gradient twist).
2. [Weigel et al., 2010](https://doi.org/10.1016/j.jmr.2010.05.011),
   *Extended phase graphs with anisotropic diffusion*. Pathway-dependent
   diffusion is the relevant model; the counterexamples below follow by directly
   integrating a linear wavevector trajectory, not by assuming a different RF sign.
3. Repository thesis: section 4.1.1, printed page 11 / PDF page 12 (gradients off
   during TE); equations 4.1 and 4.2, printed page 14 / PDF page 15 (information
   objective and tissue weights); Appendix A.1.2, printed page 37 / PDF page 38
   (MD=trace(D)/3 and trace-based FA).
4. [SciPy BSpline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html)
   is an independent numerical reference for the basis and interpolation domain.

## Coverage and independent invariants

Test names in the following tables are the exact function names (parameterized RF
and spline tests expand to multiple cases). Fixtures are in `tests/cases.py`.

### EPG: tests/test_epg.py

| Test | Invariant or interpretation |
|---|---|
| test_simulator_initializes_only_unit_z0 | Spy observes the actual simulator input to its first RF operation: only Z0=1, shape (3,150). No duplicate initializer is tested. |
| test_first_echo_after_ideal_preparation | Existing preparation returns -Z0 in the zero-diffusion/infinite-relaxation limit; first F+0 has positive imaginary sign. |
| test_zero_flip_angle | RF identity for all retained components. |
| test_rf_against_rodrigues_rotation | Independent Cartesian Bloch/Rodrigues rotation at four angles and three phases. |
| test_positive_gradient_and_refocusing | F+0 advances to order one; 180-degree RF plus positive gradient refocuses with conjugacy. |
| test_positive_gradient_truncates_without_wrap | Highest order is discarded or explicitly retained with one added state. |
| test_negative_gradient_refocuses_positive_order | Negative unit shift must return arriving F+1 to F+0 (G01). |
| test_zero_gradient_step_is_identity | Compiled and eager zero-step operation preserve the input. |
| test_relaxation_analytical_no_gradient | Exponential T1/T2 factors and recovery only into Z0. |
| test_long_t1_preserves_longitudinal_states | Infinite T1 removes longitudinal decay/recovery. |
| test_long_t2_preserves_transverse_states | Infinite T2 removes transverse decay. |
| test_equilibrium_stationary_under_relaxation | Equilibrium is fixed within the legacy float32 precision budget. |
| test_zero_diffusion_matches_relaxation | Scalar D=0 reduces to relaxation. |
| test_jit_rf_and_relaxation_match_eager | Compilation preserves these elementary operators. |

### Diffusion: tests/test_diffusion.py

| Test | Invariant or interpretation |
|---|---|
| test_zero_tensor_equals_zero_scalar_diffusion | Zero tensor and zero scalar coefficient agree with gradients on and off. |
| test_scalar_zero_gradient_attenuates_existing_order_only | Stationary spatial modulation decays by exp(-D*k^2*T); Z0 does not. |
| test_scalar_gradient_analytical_pathway | Starting at zero order, b=k^2*T/3. |
| test_tensor_isotropic_limit_matches_scalar_at_nonzero_order | D=dI must reproduce scalar diffusion (D01). |
| test_tensor_gradient_off_zero_order_has_no_diffusion_decay | A zero wavevector with gradients off has b=0 (D02). |
| test_isotropic_diffusion_is_direction_independent | Unit x and normalized oblique direction agree for D=dI. |
| test_anisotropic_zero_order_attenuation_uses_directional_diffusivity | Oblique g^T D g includes off-diagonals; higher directional diffusivity attenuates more. |
| test_tensor_and_direction_rotate_together | Simultaneous coordinate rotation leaves the response unchanged. |
| test_diffusion_direction_sign_symmetry | Reversing g leaves quadratic diffusion weighting unchanged. |
| test_generated_tensor_is_symmetric_positive_definite | Existing template generator gives a symmetric SPD tensor at positive scale. This is not a claim that all inputs/phantoms are validated. |
| test_mean_diffusivity_analytical | MD=trace(D)/3 (M01). |
| test_mean_diffusivity_derivative_matches_implementation | Declared MD derivative equals AD of the implemented metric (M02). |
| test_fractional_anisotropy_eigenvalue_definition | Isotropic, rank-one and anisotropic tensors match the eigenvalue definition. |
| test_fractional_anisotropy_scale_invariance | Value is scale invariant and AD derivative along D vanishes. |
| test_fractional_anisotropy_hand_derivative | Manual derivative agrees with AD away from isotropy (M03). |
| test_scalar_diffusion_derivative_is_minus_b_times_signal | Analytical derivative d exp(-bD)/dD. |
| test_tensor_diffusion_derivative_in_symmetric_direction | AD matches a centered tensor perturbation. |
| test_scalar_and_tensor_diffusion_jit_equal_eager | Compilation consistency for each existing implementation. |

### Splines: tests/test_bspline.py

| Test | Invariant or interpretation |
|---|---|
| test_basis_partition_and_scipy_consistency | One-hot coefficients expose basis; nonnegative partition of unity and SciPy agreement for all three evaluators. |
| test_fit_coefficient_count_and_reconstruction_on_actual_fit_grid | n_coeff=len(knots)-degree-1; recovery of known coefficients and sampled curve. |
| test_fit_reconstructs_on_experiment_sample_grid | Fit should reconstruct data on the experiment grid (B01). |
| test_left_boundary_and_right_hand_limit | Clamped left endpoint and approach to right endpoint agree with SciPy. |
| test_legacy_exact_right_boundary_is_excluded | Explicit characterization of half-open basis: exact final knot returns zero. |
| test_spline_coefficient_derivative_and_jit | Coefficient Jacobian equals independent basis matrix; eager/JIT agree. |

### Differentiation/information: tests/test_information.py

| Test | Invariant or interpretation |
|---|---|
| test_rf_derivatives_against_finite_differences | Angle and phase derivatives match centered differences. |
| test_fa_autodiff_against_symmetric_tensor_perturbation | Actual FA function remains differentiable away from isotropy. |
| test_scalar_jacobian_dimensions_and_parameter_order | Shape (4,2,N); each [T1,T2,D,M] column matches independently perturbed input. |
| test_tensor_signal_jacobian_scalar_columns_and_jit | T1/T2/M tensor-signal derivatives match differences; eager/JIT agree. |
| test_general_jacobian_shape_and_scalar_parameter_order | Current generalized shape is (5,2,N); first three columns match differences. |
| test_fim_matches_finite_difference_information_and_noise_scaling | FIM from finite-difference Jacobian; symmetry, scaled positive definiteness/conditioning, inverse-square noise scaling. |
| test_crlb_matches_svd_inverse_for_one_tissue | Actual bound function matches independent SVD inversion in normalized parameter coordinates. |
| test_single_complex_sample_cannot_identify_four_parameters | Rank cannot exceed number of real observations; identifies a singular design. |
| test_crlb_analytical_diagonal_information | Inject known diagonal FIM at the information boundary; real CRLB routine returns [1,1/2,1/3,1/4]. |
| test_crlb_exact_singular_information_is_not_finite_precision | Exact singular FIM produces nonfinite legacy bounds, not false finite certainty. |
| test_crlb_nonuniform_tissue_weights | Ensemble result equals separately evaluated weighted tissues (I01). |
| test_general_bounds_include_all_five_parameters | Published five-parameter interface returns five bounds (I02). |
| test_scalar_signal_jit_equals_eager_and_magnitude | Short signal compilation consistency within float32 budget; magnitude is abs(real+i*imag). |

## Discrepancy classification (no scientific fixes applied)

| ID | Classification | Evidence and required decision |
|---|---|---|
| G01 | Actual operator bug | epg_mgrad overwrites arriving F+0 with wrapped F-0. An order-one impulse fails the negative-refocusing invariant. Fix boundary conjugacy before using negative gradients. |
| D01 | Actual diffusion bug | Tensor cross1 and cross2 are identical; same attenuation is assigned to F+ and F-. For a linear q path, integral(q q^T dt) includes symmetric cross terms. The isotropic limit fails at nonzero order. |
| D02 | Actual diffusion bug | Gon=0 does not set dk=0; F+0 attenuates even with zero initial wavevector. Thesis explicitly states gradients off during TE. |
| M01 | Actual metric bug | mean(trace(D)) is trace(D), not trace(D)/3; contradicts Appendix A.1.2. |
| M02 | Internal inconsistency | dMD_dD returns I/3 while AD of current MD returns I. Correct the value and derivative together, with separate reviewed scientific change. |
| M03 | Actual derivative bug | For q=trace(R^2), dFA/dq=1/(4*FA*q^2); current negative expression and subsequent matrix treatment fail AD comparison and scale invariance. |
| B01 | Coordinate/API discrepancy | Fit uses linspace(0,N,N), experiment reconstruction uses arange(N). Fitting is correct on its own grid. Decide the shared coordinate convention, then update callers separately. |
| I01 | Actual aggregation bug | Scalar bound scan never increments carry; every tissue uses weight[0]. Unequal weights expose it; equal weights hide it. |
| I02 | Thesis/interface mismatch | General bound function returns three values and removes FA/MD from the FIM before inversion. This may be an intentional preparation-optimization mode, but is not the documented five-parameter problem. Resolve estimated/fixed parameters explicitly. |

### Test assumptions corrected during implementation

- **Zero shift:** The audit suspected deltak=0 slicing was invalid. Runtime
  validation shows identity on the tested physical state in JAX 0.11.1, both
  eager and compiled. This is a passing regression, not a confirmed bug.
- **Preparation sign:** 90x/180x/-90x with ideal refocusing leaves -Z0 under
  the existing convention. The original test assumed +Z0 and was wrong. We
  corrected the test after following the RF/gradient pathway, not the code.
- **Zero readout angles:** They do not guarantee zero total signal/Jacobian:
  the preparation has pulses and relaxation-driven pathways. The singular-design
  test instead uses one complex observation for four real parameters, a rank
  bound independent of preparation details.
- **JIT equality:** Scalar simulation explicitly casts RF inputs to float32.
  Fused and eager trigonometric evaluation differ at sub-1e-8 magnetization scale
  for this fixture. The test uses rtol=3e-7, atol=1e-9 rather than asserting
  float64 equality. The operator tests retain tighter tolerances where appropriate.
- **Spline endpoint:** Exact right endpoint is excluded by the legacy half-open
  basis. This is characterized explicitly; a future closed-end API must be an
  intentional contract change, not silently accepted as a refactor.

## Regression cases and tolerances

`tests/reference/legacy_cpu_x64.json` records nine cases: simple EPG, relaxation,
scalar diffusion, tensor diffusion, short scalar MRF, short tensor MRF, scalar
Jacobian, scalar FIM, and relative standard-deviation bounds. The snapshot records
source commit/hash, Python/package versions, platform/backend, and core inputs;
complete input definitions are in `tests/cases.py` and `tests/reference_cases.py`.

Most frozen comparisons use rtol=2e-8 and atol=1e-11; inverse-information bounds
use rtol=2e-6 because inversion amplifies roundoff. Analytical scalar diffusion
uses rtol=1e-7 to accommodate the existing float32 order/b-value arithmetic.
Legacy epg_relax permits 3e-8 absolute error due to its float32 decay matrix.
Finite differences use physically scaled steps (T1=.1 ms, T2=.01 ms,
D=1e-7 mm^2/s, M=1e-4); metric perturbations are taken away from isotropy.
Condition numbers are checked after parameter scaling to avoid conflating units
with identifiability. Threshold 1e9 is a fixture guard, not a universal scientific
criterion for acceptable estimation.

Capture a candidate only through the explicit tool:

```sh
python tools/capture_validation_reference.py candidate-reference.json
```

It refuses to overwrite existing files. Review each difference before replacing a
baseline. Tests never regenerate expected values. Tensor legacy snapshots preserve
known defects and are not independent diffusion validation.

## Before restructuring

First correct/validate the tensor diffusion operator, negative-gradient boundary,
MD/FA derivatives and tissue-weight aggregation in separate scientific changes.
Resolve the FA/MD coordinate model: a matrix pseudoinverse of the metric gradient
does not define a unique inverse parameterization of a symmetric diffusion tensor.
Decide whether diffusion is fixed or estimated before interpreting generalized
CRLBs. Resolve M0 initial-state scaling and spline sample coordinates.

Not yet covered: all generalized diffusion-parameter derivatives, optimized-train
state-count convergence, negative multi-step gradients, actual gradient calibration,
FA derivative at isotropy, invalid-tensor rejection, hardware/sequence validation,
phantom reconstruction, optimizer convergence, and reproduction of thesis figures.
The current direct FIM inverse still has no explicit rank/conditioning guard.
The suite detects illustrative singular designs; it does not add such a guard.

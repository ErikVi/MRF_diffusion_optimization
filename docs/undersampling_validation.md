# Cross-validation of undersampling against Heesterbeek

## Subsequent end-to-end experiment

The full educated-initialization → joint optimization → tensor phantom → NUFFT →
quantitative matching experiment was subsequently executed on 2026-09-16.
See [executed results](end_to_end_validation.md). Both Cartesian recovery gates
passed and optimization converged, but the optimized sequence did not show
general undersampling superiority. This adds model-consistency evidence; it
does **not** resolve the diffusion-physics or UEE limitations documented below.
The complete pipeline remains **PARTIAL**, not physically certified.

Validation date: 2026-09-16. Reference repository:
[D.G.J. Heesterbeek, MRF undersampling optimization](https://github.com/imphys/MRF_undersampling_optimization),
pinned to commit `4a7c0eddb8be2a6ce8a36c97a01fafcf13e7665e`.

**Result: the restricted acquisition common case passes 68 required checks.**
This is not a blanket equivalence claim. Three exploratory precision targets
failed and remain recorded; the external direct and FFT PSF helpers use inconsistent
array-axis conventions; rotating-case analytical UEE maps do not agree with
nonlinear dictionary maps. **The full diffusion/FA/optimized-phase experiment is
not certified by this comparison.** No package MRI physics or acquisition
implementation was changed.

## Reproducible configuration

| Setting | Common value |
|---|---|
| Image | 9×9 pixels; asymmetric complex object and separate two-compartment tissue case |
| PSF displacement support | 17×17 = (2H−1)×(2W−1), sufficient for all pixel displacements |
| Spatial origin | index minus floor(size/2) |
| Trajectory | Actual external `Data structures/Single spiral.npz`, `Coords` float64 (2,1802), normalized cycles/pixel, original component order xy |
| Rotation | Golden angle −π(3−√5), and native discrete 32-arm schedule; offset 0.17 rad |
| Geometry checks | 12 frames, 1/3/4 interleaves; all samples inside clipping bounds |
| Temporal acquisition checks | Complete Cartesian grid, rotating spiral with 1 or 4 interleaves |
| DCF | Native radial-increment rule r·abs(diff(r)), first difference zero; separately per arm |
| Signal | 12 RF pulses; alpha[n]=0.15+0.45 sin²(linspace(0,π,12)[n]) rad |
| RF phase | Zero throughout, independently of object phase |
| TR / TE | 15 ms / 0 ms |
| Preparation | Zero preparation angles; no inversion |
| Diffusion | Zero tensor; zero scalar coefficient in the separate scalar comparison |
| EPG states | 32 retained in both; external starts with two columns and grows, ours has fixed storage |
| Single tissue | T1=1000 ms, T2=80 ms, initial Z0=1, equilibrium M0=1 |
| Spatial tissues | (T1,T2)=(900,70) and (1100,90) ms; dictionary is all four independent combinations |
| Density | External relative multiplier 1 or 0.8; constant in time per voxel |
| Object phase | External quadratic phase field, order 0.1; our edge phase 2π×0.1 rad |
| Noise | None |
| Precision | Complex128; complex64 and scalar float32 are explicit diagnostic comparisons |
| NUFFT | SigPy defaults 1.25/4 versus tighter oversampling=2, kernel width=6 |

The odd image shape avoids conflating this comparison with the external
`phase_field` helper's even-size behavior. The asymmetric spatial object exposes
axis swaps and phase loss. No external optimized sequence is substituted for this
project's missing optimized inputs.

David's `Optimisation.Cost.signal` samples immediately after RF and has no explicit
TE evolution. Our normal sequence includes preparation and TE=4 ms. TE=0 and
zero preparation angles are supplied through existing sequence settings for the
common case; no EPG equations are modified. Relaxation and positive unit dephasing
between RF pulses then agree at zero diffusion. The external and internal state
counts are ample for this short sequence.

Software used: NumPy 2.5.3, SciPy 1.18.1, JAX/JAXlib 0.11.1, SigPy 0.1.27,
Numba 0.67.0, Python 3.12. The external `environment.yml` does not pin versions;
this is validation under the recorded environment, not reproduction of the
authors' original historical binary environment.

## What is actually compared

The external project provides PSF and analytical error-prediction functions,
not a standalone image-to-k-space-to-nonlinear-dictionary pipeline.

1. Original `Spiral_coord`, `Spiral_dcf`, `Coord`, `P_single`,
   `P_single_fft`, `P_all`, `phase_field` and `Zero_padding` are executed
   from hash-verified source without editing their methods.
2. Forward sampling is compared with a SigPy linear operator constructed as in
   the external FFT helper, **and** an independently evaluated exact Fourier
   matrix using the external image coordinates. Shared SigPy agreement alone
   would not provide independent numerical validation.
3. Adjoint frames are compared with that exact matrix's Hermitian transpose and
   convolution with the external direct PSFs.
4. Temporal signals come from the actual external `Optimisation.Cost.signal`
   and our existing tensor/scalar signal APIs.
5. The same existing complex matcher is applied to both reconstructed branches
   to isolate acquisition differences. This is not an independent validation of
   our matching algorithm or a claim that David supplies that nonlinear matcher.
6. The original external `Evaluation_data`, `S_matrices`,
   `Error_matrices` and `Theta_1_star` are also executed. Their analytical
   parameter prediction is assessed separately from nonlinear matching.

## Numerical agreement

Relative differences below are ||actual−reference||₂/||reference||₂.
Maximum errors are absolute in the array's declared units. For “unit gain” frame
or PSF comparisons, both arrays are divided by sum(weights)/(H×W) solely to make
the interpolation error interpretable; raw arrays are retained.

| Quantity | Maximum difference | Relative L2 difference | Assessment |
|---|---:|---:|---|
| Loaded measured spiral | 0 | 0 | Exact |
| Golden/discrete rotated coordinates, all tested interleaves | ≤1.67e−16 | ≤1.12e−16 | Roundoff |
| DCF in cycles/pixel | 0 | 0 | Exact |
| DCF after radians-to-cycles unit conversion | 6.82e−17 | 2.13e−13 | Roundoff in radial differences |
| Complex object-phase field | 1.57e−16 | 4.93e−17 | Roundoff |
| Forward: our default NUFFT vs external-style SigPy, axes converted | 4.55e−16 | 1.98e−16 | Agreement of wrappers |
| Forward: tight NUFFT vs independent exact DFT | 6.48e−6 | 3.01e−6 | Interpolation error |
| Adjoint: our default vs external-style SigPy, axes converted | 1.75e−18 | 3.05e−16 | Agreement of wrappers |
| Adjoint: tight NUFFT vs exact DFT, unit gain | 2.03e−6 | 1.63e−6 | Interpolation error |
| Extended direct PSF vs external `P_single` | 3.47e−18 | 1.45e−15 | Agreement |
| Actual tight AᴴWA impulse response vs direct Fourier PSF, unit gain | 6.46e−7 | 2.53e−6 | Interpolation error |
| External `P_all` direct branch after its normalization | 1.88e−15 | 5.51e−15 | Agreement |
| Compatible complex tensor signal | 2.78e−17 | 4.99e−17 | Agreement |
| External T1 derivative vs our finite difference | 4.19e−14 | 3.33e−9 | Agreement within finite-difference precision |
| External T2 derivative vs our finite difference | 2.50e−13 | 3.31e−10 | Agreement within finite-difference precision |

The measured-coordinate gates are 2e−15, exact/direct algebra gates are generally
1e−12 to 1e−14, and tight NUFFT maximum-error gates are 5e−5 (single image) or
8e−5 (gain-scaled temporal frames). These numerical tolerances are not claims of
clinical accuracy or acceptable quantitative-map bias.

### Complete complex frame series

| Sampling | K-space maximum error vs exact DFT | Gain-scaled frame maximum error | Frame relative L2 error | Matched-index disagreements |
|---|---:|---:|---:|---:|
| Cartesian | 1.21e−5 | 1.88e−5 | 1.72e−5 | 0/81 |
| Spiral, 1 interleaf | 1.36e−5 | 1.04e−5 | 5.77e−6 | 0/81 |
| Spiral, 4 interleaves | 1.71e−5 | 1.87e−6 | 1.78e−6 | 0/81 |

For all 24 rotating-frame cases, exact forward/adjoint sampling also agrees with
linear convolution using David's direct PSF to better than 1e−13 maximum error.
This tests displacement support, centering and cropping rather than relying only
on the central PSF pixel.

The fully sampled matched T1/T2 maps recover truth exactly on the four-entry grid.
Both spiral branches choose the same dictionary entries as the independent DFT
branch. That means they agree with each other, not that the undersampled estimates
are accurate: T1 RMSE is 136.99 ms for one interleaf and 153.96 ms for four; T2
grid RMSE is zero in these particular cases. Different interleaf counts use
different, non-nested golden schedules.

Raw-adjoint density NRMSE versus truth is 1.71e−5 for Cartesian, 0.9980 for one
spiral and 0.9922 for four spirals. These large spiral values expose the uncorrected
DCF/operator gain. They are not presented as calibrated density estimates.
David's global PSF normalization and our optional impulse-gain calibration are
different operations and must not be silently substituted for one another.

## Differences and their causes

### Coordinate convention: verified reference-helper inconsistency

The external direct PSF treats the first coordinate as x (image columns). SigPy
treats coordinate component zero as array axis zero (rows). David's FFT helper
passes xy directly to SigPy. Thus its output must be transposed for comparison
with its own direct helper's yx images.

Unconverted forward coordinates produce relative error 0.8761 for the asymmetric
image. Native `P_all` FFT versus direct PSFs have relative error 1.1553; transposing
the FFT output reduces it to 0.003934. The remaining error is interpolation.
This inconsistency is verified between the reference helpers; it is not evidence
that our validated yx convention should be changed. No external source was fixed.

### Normalization and DCF units

Our ideal forward operator is
A[k,r]=exp(−2πi k·r)/sqrt(V), V=H×W. A direct normal-operator PSF is
sum(w exp(+2πi k·r))/V.

The external `P_single_fft` computes A_extᴴw, not AᴴWA delta. Its raw output
therefore needs the xy/yx transpose and multiplication by sqrt(V_ext)/V to compare
with a direct PSF normalized on the original image support. Here that factor is
17/81. It is not appropriate to compare the raw arrays without this conversion.

The external DCF is evaluated in the units supplied to it. Scaling coordinates
by c multiplies its radial-increment weights by c². Its radians branch therefore
has a (2π)² factor; its extended-grid branch has a 17² factor relative to normalized
cycles/pixel. `P_all` subsequently divides by abs(sum(mean PSF)), cancelling these
global constants. The harness checks primitive weights and final normalization
separately.

### Precision: three exploratory targets did not pass

The initial run applied some overly strict targets to the external/default
precision paths. Their failures are **retained**, not hidden by relaxed thresholds:

| Exploratory comparison | Observed maximum error | Original target | Classification |
|---|---:|---:|---|
| Default NUFFT vs exact DFT | 5.89e−3 | 5e−3 | Default interpolation accuracy |
| Scalar signal after matching only float32-rounded input angles | 1.05e−8 | 1e−12 | Remaining float32 RF arithmetic |
| Native FFT PSF after transpose vs direct | 1.17e−3 | 1e−3 | Default interpolation accuracy |

They are diagnostic comparisons, not required passes for the selected complex128,
tight-NUFFT common configuration. The final report explicitly lists them in
`exploratory_precision_failures`.

Increasing oversampling/width from 1.25/4 to 2/6 lowers forward relative DFT error
from 0.002318 to 3.01e−6. A **validation-only backend factory injection** into the
isolated external helper namespace similarly lowers transposed `P_all` relative
error from 0.003934 to 2.66e−6 (maximum 9.93e−7). External method source stays
unchanged, but that injected run is explicitly not the native default run.
Both convergence ratios exceed 100× and are required checks.

Our scalar simulator intentionally casts readout RF inputs to float32. Its
signal difference is 2.43e−8 before matching rounded input angles and 1.05e−8
afterward. Rounding input values alone does not reproduce float32 trigonometry.
The isolated RF matrix difference is 2.06e−8 at float32 and 1.25e−16 at float64.
Complex64 spatial encoding differs from complex128 by 5.36e−7 maximum. These
measurements support a precision explanation without changing physics.

### EPG and sequence conventions

Under the selected zero-diffusion, TE=0, no-preparation settings, the complex
tensor signal agrees to roundoff. With our **normal** prepared TE=4 ms sequence
instead, maximum difference is 0.7224, relative difference 1.7359. This is an
expected sequence mismatch; no global phase correction is applied to force it
away. Initial equilibrium M0=1 is common. Tests of other M0 values explicitly
scale initial Z0 consistently; the forward phantom's density remains its separate
external multiplier.

### FFT centering and phase: negative controls

Deliberately shifting the image origin with `ifftshift` gives relative k-space
error 1.0532. Discarding complex image phase gives error 0.2353. These are negative
controls, not mistakes present in our production pipeline. Correctly centered
odd-to-extended zero padding agrees exactly with the external helper.

## Final external UEE behavior

For homogeneous T1=1000 ms/T2=80 ms, stationary sampling and spatially varying
complex density, the temporal PSF residual vanishes. The actual external UEE
stages and our complex matcher then agree:

- T1 maximum difference: 2.27e−13 ms.
- T2 maximum difference: 2.84e−14 ms.
- Reconstructed density magnitude: 6.66e−16.

This is a valid common limiting case; density is the blurred, normalized density,
not necessarily the original phantom density.
For this limiting check the matcher receives the exact PSF-convolved series;
finite-NUFFT differences are measured separately in the temporal acquisition cases.

For rotating sampling and two tissue compartments, the external predictor and
our discrete nonlinear matcher differ strongly: maximum T1 difference is
2.01e5 ms and T2 difference is 110.64 ms. Their algorithms solve different
problems. Additional diagnostics give a real normal-matrix condition number of
6.23e4, maximum input log contrast 0.1335, but predicted log correction 5.953.
Minimum blurred density magnitude is 0.740, so a near-zero density denominator
does not explain this case. Derivatives independently agree with our finite
differences. The predicted correction is far outside a small local perturbation.

The evidence supports an invalid approximation regime for this short sequence,
rather than an acquisition discrepancy; it does not prove the external predictor
is bug-free in general. These maps are not interchangeable, and this experiment
does not validate applying UEE to diffusion/FA or optimized RF phases.

## Provenance and reproducibility

| External file | Git blob at pinned revision | Use |
|---|---|---|
| `UEE_phase.py` | `d28215569e2cbe74b6a1247fdc896edf87fb9bc2` | Original geometry, DCF, PSF and UEE methods |
| `Optimisation.py` | `7e262600ef732f584982f61d1ea03029a6e994a2` | Original EPG signal and derivatives |
| `Data structures/Single spiral.npz` | `2f500c53153b7d46d6b438219235b6f1dfd22715` | Actual measured trajectory |
| `LICENSE` | `f288702d2fa16d3cdf0035b15a9fcbc552cd88e7` | GPL-3.0 retained with external inspection files |

The script verifies these Git blobs before execution. SHA-256 hashes are also
saved in the report. No external source or binary trajectory is vendored into
package source. The external GPL license is retained with the separately obtained
reference files. The validation harness executes selected original class methods
without importing the external experiment's module-level setup.

Obtain that revision separately and, as the external README requires, place
`Single spiral.npz` in its working root alongside `UEE_phase.py`,
`Optimisation.py` and `LICENSE`. Then run from our repository:

```sh
python tools/validate_undersampling_common_case.py /path/to/reference \
  --output-directory data/output/common_case_validation
```

The output directory must be new. Outputs are `report.json`,
`intermediates.npz` and `signal_fixture_candidate.json`. The latter was reviewed
and preserved as `tests/reference/undersampling_common_signal.json`; tests never
recapture it automatically. Intermediate arrays include reference and internal
signals, measured coordinates, DCF, k-space, adjoints, native PSFs and UEE maps.
The initial failed exploratory run is retained separately in the workspace.

`tests/test_undersampling_crossvalidation.py` adds six offline cases: tensor
agreement, bounded scalar-precision difference, two consistently initialized M0
scales, explicit normal-sequence mismatch and rejection of modified external
source before execution.

Final full-suite result: **170 passed, 9 expected failures**, 197.72 s. The nine
expected failures are the existing documented scientific discrepancies. The live
external comparison is run by the separate harness; offline tests do not require
downloading reference source or data.

## Remaining uncertainties and validation boundary

No new bug in our acquisition implementation was identified in these common
cases. This finding does not cover anisotropic diffusion, FA estimation,
nonzero optimized RF phase trains, coils, noise cross-validation, general
rectangular/even external phantom conventions, trajectory clipping or variable-
density generator equivalence. Scanner waveform/dwell-time calibration is absent
from the measured NPZ metadata.

This comparison is a necessary acquisition validation step, not sufficient
evidence to treat the complete diffusion-MRF experiment as scientifically validated.
Known D01/D02 and other legacy discrepancies remain; actual optimized sequence
inputs and broader sequence-specific checks are still needed. See
[project validation](validation.md) and [quantitative limitations](quantitative_undersampling.md).

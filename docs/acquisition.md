# Spatial acquisition and undersampling

The encoding layer maps complex images to non-Cartesian k-space and back.
It does not simulate magnetization, fit tissue maps or optimize sequences.
The historical undersampling experiment has not been connected to this API.

## Install and validate

Install the optional backend with:
~~~sh
python -m pip install -e ".[acquisition]"
~~~
For the complete tested environment:
~~~sh
python -m pip install -r requirements-acquisition-validation.txt
python -m pytest tests/test_acquisition.py -q
python -m pytest -q
~~~
Missing SigPy fails its tests rather than silently skipping them. SigPy 0.1.27
is the tested minimum acquisition version. Exact requirements record the tested
environment, not a claim that newer versions are incompatible. SigPy loads
lazily only for NUFFT, spiral design and Pipe-Menon; geometry/direct PSFs use
NumPy. This CPU host API is not JAX differentiable. EPG has no SigPy dependency.

## Coordinate and array contract

- Images: (height,width), centered at index floor(size/2), including odd, even
  and rectangular shapes.
- Coordinates: (samples,2), **(ky,kx)** in **cycles/pixel**. Multiply by
  (height,width) for SigPy's array-axis grid coordinates. Nyquist ±0.5 maps
  to ±size/2. No coordinates are silently clipped.
- Beyond-Nyquist sampling aliases. Apply any sampling mask to weights/data too.
- Forward output / adjoint input: (samples,), not (samples,1). This initial API
  deliberately excludes implicit batching/broadcasting.
- Positive rotation sends +kx toward +ky: counterclockwise with Cartesian y-up,
  clockwise when displayed image rows increase downward.
- Rotate physical coordinates BEFORE per-axis normalization for unequal pixel
  sizes. Euclidean rotation of cycles/pixel assumes equal pixel spacing.
- Imaging cycles/pixel are distinct from EPG angular wavevectors in rad/mm.
  No RF phase, object phase, gradient calibration or unit conversion is inferred.

The ideal operator is A[k,r]=exp(-2πi k·r)/sqrt(H*W), with pixel position
r=index-floor(shape/2). Its adjoint uses the conjugate exponent and identical
normalization. SigPy approximates this with Kaiser-Bessel interpolation:
defaults oversampling=1.25, kernel_width=4; tighter tested settings 2 and 6.

Complex64/128 retain complex phase and precision. Float32 becomes complex64;
float64 and integer inputs become complex128. Inputs must be finite.
No magnitude conversion, automatic DCF, fitted gain or phase correction occurs.
An adjoint is not generally an inverse.

## APIs

| Module | Function/class | Responsibility |
|---|---|---|
| encoding/trajectory.py | rotate_trajectory | Rotation retaining supplied units |
| same | generate_rotated_spiral_trajectory | (frames,interleaves,samples,2) coordinates |
| same | load_spiral_coordinates | Explicit NPZ layout/component conversion; no download/unit inference |
| same | generate_variable_density_spiral | One square/isotropic SigPy-designed arm in cycles/pixel |
| same | cartesian_trajectory, to_sigpy_coordinates | Centered full grid and backend conversion |
| encoding/nufft.py | NufftOperator.forward, .adjoint | Unweighted A and Aᴴ |
| encoding/density_compensation.py | radial_increment_density_compensation | Explicit reference heuristic |
| same | pipe_menon_density_compensation | Separate iterative gridding estimator |
| reconstruction/images.py | reconstruct_weighted_adjoint | AᴴW y, weights applied exactly once |
| encoding/psf.py | calculate_point_spread_function | Actual AᴴWA response to a unit central pixel |
| same | direct_sampling_psf | Exact Fourier sum; optional extended displacement grid |

## Trajectories and weights

Schedules are explicitly distinct:
- uniform: offset + frame*increment + arm*2π/L.
- golden: offset - π(3-sqrt(5))*(frame*L+arm), David's negative 137.507764°
  increment, not the radial 111.246° convention.
- reference_arms: offset + 2π/A*(frame+floor(A/L)*arm), A=32 by default.
  It preserves unequal arm gaps when L does not divide A; uniform instead
  gives equal angular separation.

Angles are radians. Only uniform uses angular_increment_rad; only
reference_arms uses reference_arm_count. Retain separate interleaves for radial
DCF; flatten arm/sample axes only when creating a frame's operator.

The reference heuristic is w[k]=r[k]*abs(r[k]-r[k-1]), first weight zero
for EACH arm. Its input can have leading frame/interleaf axes. It is not
universal quadrature. Scaling coordinates by c scales weights by c²:
weights evaluated in square H×H SigPy coordinates are H² times those in
cycles/pixel. This gain is deliberately not hidden. Pipe-Menon has its own
iterative method, iteration count and raw scale; it is not identified with
radial DCF. Neither method is automatically normalized or guaranteed to improve
every object's reconstruction.

Variable-density design takes explicit FOV (m), square matrix size, frequency
sampling factor, acceleration, density exponent, gradient limit (T/m) and slew
limit (T/m/s). SigPy's radial scale N/(2*FOV) is interpreted as cycles/m and
multiplied by FOV/N, yielding maximum radius 0.5 in the tested case. We do not
repeat the external wrapper's unexplained division by 256. This is a coordinate
design, not scanner-certified gradient waveforms or dwell-time metadata.
Intra-readout signal evolution remains outside this snapshot-image model.

## Small composition example

~~~python
import numpy as np
from mrf_diffusion.encoding import (
    generate_rotated_spiral_trajectory, radial_increment_density_compensation,
    NufftOperator, calculate_point_spread_function,
)
from mrf_diffusion.reconstruction.images import reconstruct_weighted_adjoint

arm = np.array([[0, 0], [0.02, 0.08], [0.12, 0.15], [0.24, -0.10]])
trajectories = generate_rotated_spiral_trajectory(
    arm, frame_count=2, interleaves_per_frame=3, schedule="golden",
)
frame = trajectories[0]
weights = radial_increment_density_compensation(frame).reshape(-1)
operator = NufftOperator((8, 10), frame.reshape(-1, 2))
image = np.zeros((8, 10), dtype=np.complex128)
image[2:6, 3:7] = np.exp(0.4j)
samples = operator.forward(image)
reconstructed = reconstruct_weighted_adjoint(operator, samples, weights)
psf = calculate_point_spread_function(operator, weights)
~~~

This heavily undersampled example demonstrates composition, not exact recovery.

## PSF and padding

The direct PSF is sum_k w[k] exp(+2πi k·r)/(H*W). The actual NUFFT PSF
includes approximation in A and Aᴴ. Aᴴw alone is not AᴴWAδ: with ideal
unitary scaling it needs a factor 1/sqrt(H*W).

displacement_shape=(2H-1,2W-1) evaluates all displacements for linear convolution.
It changes neither pixel spacing nor image normalization. This is not image
upsampling or internal NUFFT oversampling. Tests compare this convolution with
an independent dense Fourier normal operator, including even/rectangular images.
No PSF peak/sum normalization hides amplitude errors.

## Validation and direct external comparison

The tests cover all twelve requested basics, plus dense DFT agreement, complex
adjoint identity, off-center phase ramps, dtype preservation, absolute gain,
per-arm DCF, PSF convolution, invalid input, backend isolation, variable-density
scale and default-versus-tighter interpolation. Tight NUFFT tests allow measured
interpolation error, usually 2e-5 or 3e-5 absolute; direct geometry/Fourier tests
use near-roundoff budgets. Existing physical reference files are unchanged.

The comparison tool runs unmodified named helpers from
[UEE_phase.py, revision 4a7c0ed](https://github.com/imphys/MRF_undersampling_optimization/blob/4a7c0eddb8be2a6ce8a36c97a01fafcf13e7665e/UEE_phase.py):
Spiral_coord, Spiral_dcf, Coord, P_single and P_single_fft. It verifies the
Git blob hash and does not import external EPG/script initialization.
A synthetic six-point supplied arm makes the comparison independent of binary data.

| Quantity | Maximum absolute difference |
|---|---:|
| Golden / reference-arm rotation | 5.55e-17 |
| Radial DCF | 0 |
| Direct PSF | 1.65e-18 |
| External default NUFFT versus direct PSF, adjusted axes/scale | 1.31e-5 |
| Our default AᴴWA PSF versus direct | 1.27e-5 |
| Our oversampling=2, width=6 PSF versus direct | 1.04e-8 |

The external NUFFT passes xy directly to SigPy's array-axis interface. For this
square comparison its result is transposed to yx and Aᴴw is divided by sqrt(V).
Our AᴴWA PSF also uses the numerical Aδ rather than assuming it exactly constant.
These are explicit convention/interpolation differences; no EPG science changes.

The audit's suspected (K,1) weight incompatibility was NOT reproduced: SigPy
0.1.27 accepts the external column weights, giving exactly the same PSF as vector
weights. Our public vector contract still avoids ambiguous broadcasting.
No full published phantom or sequence optimization was reproduced.

To repeat with a separately obtained pinned reference checkout:
~~~sh
python tools/compare_acquisition_reference.py /path/to/reference --output candidate.json
~~~
Optional --fixture captures a NEW candidate only; neither tests nor the tool
overwrite tests/reference/acquisition_external.json.

## Provenance and limits

The external repository by D.G.J. Heesterbeek is GPL-3.0. Production modules
implement the described geometry/Fourier equations and call SigPy; no external
source file or extracted method is distributed in our package. The comparison
tool evaluates user-supplied, hash-verified reference methods. Frozen numerical
outputs record source identity. Future source copying/adaptation requires its
own license compatibility review; this work assigns no project license.

SigPy 0.1.27 is BSD licensed. It supplies NUFFT, Pipe-Menon DCF and spiral design.
Upstream definitions:
[Fourier](https://sigpy.readthedocs.io/en/latest/_modules/sigpy/fourier.html),
[spiral](https://sigpy.readthedocs.io/en/latest/_modules/sigpy/mri/samp.html),
[DCF](https://sigpy.readthedocs.io/en/latest/generated/sigpy.mri.pipe_menon_dcf.html).
Scientific context: [Heesterbeek et al.](https://doi.org/10.1002/mrm.29554).

Single spiral.npz is optional, unbundled external input; see
[data provenance](../data/input/trajectories/README.md). The [forward phantom extension](forward_phantom.md) now supplies complex MRF
frames, optional seeded receiver noise and per-frame adjoints. Coils, nonlinear
inverse reconstruction, analytical UEE and unrestricted tensor fitting remain future work.
A bounded [complex quantitative matcher](quantitative_undersampling.md) now operates
on reconstructed frames, outside the acquisition layer.

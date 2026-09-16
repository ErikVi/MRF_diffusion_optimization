# Forward diffusion-MRF phantom validation

`mrf-phantom --config experiments/undersampling/default.toml` runs a small,
synthetic 16-by-16 checkerboard with eight readouts in each of two diffusion
directions. Install `.[experiments]` first. This is a validation fixture, not a
reproduction of a published optimized sequence. No quantitative matching is run.

## Scientific boundaries

`phantoms/geometry.py` defines labels, support and optional object phase;
`phantoms/maps.py` assigns tissue tables to geometry. The historical
`UNDERSAMPLING.py` checkerboard is a conceptual reference. Its tensor scale maps
were not measured MD/FA, and its spatial images did not use the displayed tensor
map consistently. That workflow remains available as `mrf-legacy-phantom` with
`experiments/undersampling/legacy.toml`; it is not used by the new forward path.

`diffusion/parameterization.py` constructs tensors independently of the preserved
legacy tensor metric and attenuation functions. For requested MD > 0, FA in [0,1]
and a normalized principal axis n, choose a **prolate axisymmetric** tensor:

```
a = FA / sqrt(3 - 2 FA**2)
lambda_parallel = MD * (1 + 2*a)
lambda_perpendicular = MD * (1 - a)
D = lambda_perpendicular * I + 3*MD*a * outer(n, n)
```

The eigenvalue mean is MD, and the standard eigenvalue definition gives
`FA**2 = 3*a**2 / (1 + 2*a**2)`. Eigenvalues are nonnegative, including the
rank-one FA=1 boundary. This is one explicit tensor family; MD and FA alone do not
uniquely specify a general tensor. Axes use physical `(x,y,z)` components, with
antipodal axes equivalent. Isotropic tensors have no identifiable principal axis.
Returned MD/FA maps are calculated from the tensors, not independent display
scales. Definitions follow the eigenvalue formulation described in
[this diffusion tensor study](https://pmc.ncbi.nlm.nih.gov/articles/PMC4252798/).

T1/T2 use ms; tensor entries and MD use the existing model's mm²/s convention;
FA and relative proton density are dimensionless. Existing gradient and diffusion
unit concerns in `units.md` still apply. Image rows/columns are `(y,x)` and are
separate from tensor component order. Background support is zero signal.

## Signal and phase

`simulation/image_series.py` groups exact `(T1,T2,D)` combinations and calls
`simulation.api.simulate_mrf_signal(..., tensor=True)` once per unique combination.
The existing EPG implementation remains authoritative. Its two real/imaginary
channels become a complex128 image series of shape `(frames,y,x)`.

The simulator receives the actual `MRFSequence.rf_phases`. In
`simulation/signal.py`, the tensor scan passes each phase to `apply_rf_rotation`.
Fixed preparation and inversion pulses retain their existing sequence settings;
explicit preparation trains are passed through when configured. No RF phase is
multiplied onto the signal afterward.

Modes are `none`, `reference` (existing phase generator), `explicit`, and
`optimized`. Optimized mode requires an NPZ containing the paired, actual
`flip_angles_rad` and `rf_phases_rad` arrays. Optional preparation arrays are
`preparation_flip_angles_rad` and `preparation_rf_phases_rad`, supplied together.
Timing and diffusion directions come from `[sequence.physics]`; they must match
the sequence being evaluated. Loading does not generate or re-optimize a train.
The saved run includes actual arrays and the archive SHA-256. Tests use a
deterministic archive to check this path, not an independently verified optimum.

Only after simulation, pixels receive
`proton_density * exp(1j*object_phase_map)`.
Proton density is an external relative signal multiplier with simulator equilibrium
M fixed to one. It is not the legacy equilibrium-M parameter (whose initialization
and recovery do not implement a simple amplitude scale), nor an M0 estimate.
Object phase is optional, in radians, and distinct from RF phase, evolved signal
phase and trajectory rotation angle. The supplied quadratic field is a spatial
test pattern with configurable amplitude, not a B0 evolution model.

Frame metadata records direction, readout index, inversion/readout block and
source sample index, including optional sampling stride. Direction blocks are
concatenated according to the current simulator; frame indices are not asserted
to be physical timestamps. Preparation/inversion behavior is preserved.

## Acquisition and outputs

`encoding/series.py` applies the existing NUFFT to each complex frame. Coordinates
have shape `(frames,interleaves,samples,2)` in normalized `(ky,kx)` cycles/pixel;
the NUFFT wrapper converts to SigPy grid coordinates. K-space has shape
`(frames,interleaves,samples)`. Trajectory generation, rotation and interleaf
selection stay in `encoding/trajectory.py`. Generated designs require the square
design matrix to match the image grid. Supplied trajectories retain explicit
layout and component-order settings. See `acquisition.md` for scaling details.

Optional seeded independent complex Gaussian noise is added in k-space, with
the configured standard deviation for each real and imaginary channel. DCF is
never applied to forward samples. `reconstruction/series.py` applies the selected
none/radial-increment/Pipe-Menon weights before the adjoint. An adjoint image is
not an inverse reconstruction; spiral intensity and aliasing depend on sampling
and weights. No inferred quantitative maps are generated.

The TOML separates phantom, diffusion, sequence, simulation, trajectory/interleaves,
object phase, noise, density compensation and reconstruction. Run artifacts contain
configuration, software metadata, tensor-derived truth maps, actual trains,
complex images, trajectories, complex samples, frame labels and optional adjoints.
Plots show T1/T2/MD/FA/proton-density truth, object phase, representative image
magnitude/phase, k-space magnitude with sampling locations, and adjoint frames.

## Validation and limits

`tests/test_forward_phantom.py` has 32 deterministic cases covering MD/FA tensor
construction, homogeneous and single-parameter contrasts, combined checkerboards,
unique-entry reuse, four RF phase modes, independent object phase/proton density,
inversion frame ordering, complex Cartesian recovery, spiral interleaves, seeded
noise, state-count comparison and saved experiment artifacts. Every contrasting
tissue is compared directly with the existing tensor simulator. FA and MD affect
the tensor passed to that simulator, never an artificial image operation.

The existing nine expected scientific failures remain, including tensor diffusion
D01/D02. Passing these tests validates assembly and numerical preservation, not
the physical correctness of those known defects. Resolve and independently
validate the documented tensor model issues before quantitative interpretation
or claims about optimized-sequence superiority. This command remains forward-only.

## Subsequent quantitative extension

The separately validated [quantitative comparison command](quantitative_undersampling.md)
adds reconstruction-to-dictionary recovery without changing this forward simulator.

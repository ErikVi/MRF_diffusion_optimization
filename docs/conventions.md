# Numerical and MRI conventions

## EPG and RF

State array has shape `(3,K)`, complex entries, rows `[F+,F-,Z]` and columns
nonnegative coherence order. The acquired transverse signal is F+0=Mx+iMy.
Physical zero order satisfies F-0=conj(F+0); this is not an assertion that all
positive-order columns are conjugates of one another.

For c=cos(alpha/2), s=sin(alpha/2), RF phase phi, the matrix is

```text
[ c^2                    exp(2i phi)*s^2     -i exp(i phi)*sin(alpha) ]
[ exp(-2i phi)*s^2        c^2                 +i exp(-i phi)*sin(alpha)]
[ -i/2 exp(-i phi)*sin(a) +i/2 exp(i phi)*sin(a) cos(alpha)            ]
```

Angles/phases are radians. A positive pi/2 pulse at phase zero sends +Z to -i F+.
The RF function's default phase is -pi/2; sequence routines explicitly supply
phase. Preparation's ideal zero-diffusion/infinite-relaxation final state is -Z0,
so the following first positive-angle readout signal has positive imaginary sign.
This sign is covered by tests and is not changed to match another convention.

Positive unit gradients move F+ toward higher order, F- toward lower, then assign
F+0=conj(F-0). Z orders do not move. Negative-gradient boundary handling is wrong
(G01). With truncation disabled, exactly one column is added even for larger
requested shifts. Zero-step identity is validated for the tested physical states.
Only positive unit shifts are relied upon by default simulations.

## Trains and splines

Phase-train entries are supplied RF phases, not implicitly accumulated increments
by the simulator. Phase generators may construct them by accumulation beforehand.
Piecewise quadratic phase chooses a discrete split index from a fraction; AD through
that index is not a smooth optimization of split location. Keep this modeling
choice visible when interpreting optimizer gradients.

Cox-de Boor basis intervals are half-open. Exact final knot evaluates to zero;
tests compare SciPy on the interior and the right-hand limit, not at its differently
defined endpoint. Fit grid is linspace(0,N,N); experiment evaluation is arange(N).
The discrepancy is preserved. `knot_setting` is neither knot-vector length nor
coefficient count: the vector has knot_setting+1 entries and coefficients number
len(knots)-degree-1. Experiments use an endpoint of N+.01.

## Precision, JAX and differentiability

Importing the package enables JAX x64, preserving the original process-level
configuration. Select the backend before initializing JAX if another backend is
required. Validation explicitly uses CPU. The code intentionally retains mixed
precision: scalar RF trains and diffusion state-order arrays use float32;
`relax_and_shift` uses a float32 decay matrix; state initialization is complex128;
the inversion train is cast to complex128. Tensor train inputs otherwise retain
their dtype. FIM inputs are explicitly converted to float64. No blanket precision
conversion was performed during refactoring.

JIT static controls include gradient shift/truncation flags, wavevector steps,
sampling flags/offset/rate, direction count, state count, inversion flag, phase
method/aggregation, spline degree, noise sigma and immutable sequence settings.
Changing them or array shapes/dtypes may compile a new executable. Array-valued
tissues/train coefficients remain differentiable inputs. JAX `jacobian`, `vmap`,
`lax.scan` and `lax.map` remain in numerical paths. SciPy SLSQP and file/plotting IO
are host operations; spline least-squares initialization also uses Python loops.

Numerically fused and eager scalar operations need float32-scale tolerances.
AD/finite differences are checked with physically scaled perturbations. FA at
isotropy, zero tensors, rank-changing matrix inverses and discrete split indices
are not covered by a blanket differentiability claim. The optimization module
uses `jax.jacobian` for scalar objectives; use of `jax.grad` is not required for
equivalence. No NumPy/SciPy operation was inserted into compiled physics kernels.

## Information ordering

Use constants in `information.parameters`: scalar `[T1,T2,D,M]`, historical tensor
`[T1,T2,M,FA,MD]`, actual tensor objective `[T1,T2,M]`. Jacobians store parameter
first, real/imaginary channel second, sample last. FIM flattening combines channels
and samples without complex conjugation because this Jacobian is real-valued.
Relative standard-deviation bounds are sqrt(diag(inv(F)))/abs(parameter), with
legacy weight placement preserved. Zero parameters and singular F require
scientific decisions before meaningful normalization.

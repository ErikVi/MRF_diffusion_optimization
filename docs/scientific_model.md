# Implemented scientific model

This document describes the actual kernels, including retained defects. See
[validation](validation.md) for classifications and literature checked during the
audit. The local [thesis](thesis.pdf) is the project-specific reference.

## Pipeline

| Stage | Function/module | Inputs -> outputs | Interpretation |
|---|---|---|---|
| Parameterization | `sequence.parameterization.initialize_sequence_parameters` | Initial radians, knots, method -> coefficients, phase, affine terms | Historical least-squares initialization |
| Train decoding | `decode_sequence_parameters` | Coefficients, sample coordinates, knots -> flip angles and phases `(N,)` | Cubic angle spline; free phase spline, twice-integrated curvature spline, zero phase or piecewise quadratic phase |
| Initial state | `epg.states.initial_epg_state` | State count, initial Z -> complex `(3,K)` | Only Z0 nonzero; default 1 independently of equilibrium M |
| RF | `epg.rf.apply_rf_rotation` | States, angle, phase -> states | Instantaneous rotation at every retained order |
| Relaxation | `epg.relaxation.relax_and_shift` | States, T1/T2, time, M -> states | Exponential transverse/longitudinal relaxation and Z0 recovery |
| Scalar diffusion | `epg.diffusion.relax_diffuse_scalar_and_shift` | States, scalar D, wavevector increment, timing -> states | Pathway-specific exp(-bD), recovery, then shift |
| Tensor diffusion | `relax_diffuse_tensor_and_shift` | States, D `(3,3)`, direction, wavevector increment -> states | Historical tensor contraction with known errors D01/D02 |
| Gradient | `epg.gradients.apply_*_gradient_shift` | States, order step, truncation -> states | Opposite transverse order shifts; longitudinal orders remain |
| Signal | `simulation.signal.simulate_*_signal` | Tissue and trains -> `(2,samples)` | Real/imaginary F+0 at TE |
| Derivatives | `information.jacobian` | Simulation arguments -> parameter/channel/sample Jacobian | AD for scalar parameters; invalid legacy metric projection retained |
| Information | `information.fisher` | Jacobian and noise SD -> FIM | Independent Gaussian real/imaginary channels |
| Bounds | `information.crlb`, `optimization.evaluation` | FIM, tissue scales, weights -> inverse/relative SD | Direct inversion, no regularization; tensor objectives condition on diffusion |
| Optimization | `optimization.objectives`, `constraints`, `solver` | Train coefficients, tissue ensemble, bounds -> SciPy result/history | Historical SLSQP with JAX derivatives |

## Relaxation and scalar diffusion

For duration t in ms, E1=exp(-t/T1), E2=exp(-t/T2). Recovery adds
M(1-E1) only to Z0. In combined operators recovery precedes longitudinal diffusion
attenuation, whose order-zero factor is one for the implemented bZ.

Let n be retained nonnegative order, k the wavevector-step magnitude, and
tau=t/1000 seconds. Scalar code calculates

```text
bZ = (n*k)^2*tau
b+ = ((n + Gon/2)*k)^2*tau + Gon*k^2*tau/12
b- = ((-n + Gon/2)*k)^2*tau + Gon*k^2*tau/12
F+ <- F+ * E2 * exp(-b+*D)
F- <- F- * E2 * exp(-b-*D)
Z  <- (Z * E1 + recovery) * exp(-bZ*D)
```

For Gon=1 and a positive unit shift this is the integrated linear-wavevector
result. Gon=0 removes the ramp but existing spatial modulation still diffuses.
The attenuation formula does not incorporate arbitrary signed `order_step`;
negative/multiple-step equivalence is not established. Gon is a binary control,
not a continuously varying gradient amplitude. The shift runs only when Gon==1.

## Tensor diffusion: actual equations, not a corrected derivation

The code constructs k1=g*n*k and k2=g*(n+order_step)*k. It then evaluates

```text
bL = tau * outer(k1,k1)
cross1 = outer(k1,k2)
cross2 = outer(k1,k2)                  # identical in current implementation
bT = bL + tau*(cross1-cross2)/2 + tau*outer(k2-k1,k2-k1)/3
transverse attenuation = exp(-sum(bT*D))  # used for BOTH F+ and F-
longitudinal attenuation = exp(-sum(bL*D))
```

Thus the cross term vanishes and setting Gon=0 does not remove the ramp term.
These contradict scalar nonzero-order and gradient-off limits, rather than
representing an RF sign convention. They remain unchanged in this refactor.
Directions are used exactly as supplied: oblique defaults `(1,1,0)`, `(1,0,1)`,
`(0,1,1)` are not normalized. There is no runtime tensor symmetry/PSD check.

## Sequence timing and ordering

Both kernels default to TE=4 ms, TR=15 ms. Diffusion preparation uses three
instantaneous pulses `[pi/2,pi,-pi/2]` with two TE-long gradient-on intervals.
Scalar preparation/readout wavevector steps are 612/50; tensor values are 300/100.
Readout: RF -> evolve TE with Gon=0 -> extract F+0 -> evolve TR-TE with Gon=1.
The tensor Gon=0 defect still affects readout TE despite the intended gradient-off
choice. Fixed state truncation is part of the numerical model.

Tensor directions are evaluated as independent blocks from a shared starting
state; they are not sequentially carried across a complete nine-direction scan.
With `include_inversion=True`, the first direction is removed from the ordinary
blocks and used for an inversion-plus-train block. Its final state initializes
each remaining block. The default reuses the readout train here; supplied
preparation arrays are ignored unless this choice is disabled. Sampling is applied
after block concatenation. This is not an arbitrary waveform simulator.

## Metrics and information

FA uses R=D/trace(D), FA=sqrt((3-1/trace(R@R))/2), equivalent to the eigenvalue
definition for symmetric nonzero tensors. Zero tensors are undefined. The function
named `legacy_mean_diffusivity` returns trace(D); physical MD is trace(D)/3.
The manual FA derivative and the declared MD derivative are retained inconsistencies.

Scalar Jacobian is `(4,2,N)` in `[T1,T2,D,M]` order. Tensor Jacobian is nominally
`(5,2,samples)` in `[T1,T2,M,FA,MD]` order. Only its first three columns are
validated as physical parameter derivatives. A matrix pseudoinverse of each metric
gradient does not define a unique inverse mapping from FA/MD to six tensor entries.

Flatten real/imaginary observations: F=J@J.T/sigma^2, default sigma=10^-1.65 per
real channel. Parameter-dependent covariance is not modeled. Bounds are calculated
using direct inversion; singular F may produce nonfinite values. Optional
`information_diagnostics` reports scaled singular values/conditioning but does not
alter objectives or automatically regularize them.

Tensor objectives select F[:3,:3] BEFORE inversion and normalize by `[T1,T2,M]`.
They are not marginalized five-parameter bounds. L1 sums weighted relative standard
deviations; L2 sums their squares without a final square root. Scalar objective
instead sums tissue-weighted square roots of weighted normalized variances.
Scalar reporting has the separate tissue-weight bug I01.

## Experiment limitations

Phase comparison preserves the malformed historical zero-phase baseline: its
coefficient vector concatenates angle coefficients with sample-count zeros and is
then split in half. Three actual returned bounds are now plotted and labeled as
T1/T2/M; no fictional FA/MD columns are generated to satisfy old five-column plots.

The phantom workflow still simulates fixed white/gray tensors instead of using its
spatial tensor map. Dictionary keys are `[T1,T2,diffusion_scale,shape_scale]`, while
historical reconstruction assigns their last entries to swapped map names. These
scale factors are not themselves MD/FA. Some generated phantom tensors are not PSD.
Optional noise is spatial and broadcast over time; its SD is fixed independently
of the historical SNR label. No phantom accuracy claim is made. External
`UEEphase_DH` and original input arrays are required and unavailable here.

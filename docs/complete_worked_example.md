# Executed sequential worked example — 2026-09-23

The fresh [audit](../experiments/complete_worked_example/AUDIT.md) preceded code.
Both standalone configurations in `experiments/complete_worked_example/` were
executed. Full results are `data/output/complete_worked_example/full/`; open
`report/REPORT.html` or `REPORT.md`. The run contains 27 PNG/SVG figures, raw
arrays, source/configuration snapshots, metrics and component certification.

## Execution evidence

- Complete suite before execution: **178 passed, 9 strict expected failures,
  0 unexpected failures**, 233.09 s.
- Smoke: the entire scientific chain completed; SLSQP deliberately stopped at
  three iterations. Reporting initially used the wrong global-region key, which
  was corrected from `global` to the existing `all`. Reports were regenerated
  from saved data without rerunning simulations. This failure and correction are
  retained in the smoke output; no MRI model, metric or tolerance changed.
- Full: 40 readouts per x/y/z direction, 48 retained states, seven cubic angle
  coefficients, 16×16 two-compartment phantom. Total original runtime **544.26 s**
  on JAX CPU/x64, including reporting. Flip-angle SLSQP took **96.32 s**, converged
  in **9 iterations / 10 evaluations**, minimum constraint residual −9.06e−12.
- All three sequences passed all-atom, direct and Cartesian recovery gates.
  Complex image relative error was approximately **1.22e−5** (gate 1e−4).
  T1/T2/MD/FA reference RMSE was zero; density RMSE was 1.1604e−5.
- Doubling states from 48 to 96 gave identical endpoint signals for all three
  sequences. Maximum paired-noise difference was 3.14e−16. No ambiguous matches
  were observed. Finite-difference step-halving differences were at most 4.18e−9.
- Saved-output verification independently recomputed map metrics and error arrays,
  checked J/F/covariance consistency, selection, identical angle trains, noise and
  state evidence, configuration hash and report/figure links. It passed.

## Sequence progression and phase selection

The missing historical Sommer array was not fabricated. Initialization used
0.25 + 0.15 sin²(pi*n/39) radians, projected by the existing initializer, with
zero RF phase. Flip-angle optimization retained zero phase. The established
conditional T1/T2/internal-M L1 objective and existing constraints were preserved.

| Stage / candidate | Objective |
|---|---:|
| Initial | 1.298615904 |
| Flip-angle only / zero-phase candidate | 1.129605185 |
| Quadratic generator candidate | 1.089022054 |
| Linear generator candidate | 1.120142540 |
| Sinusoidal generator candidate | 1.058669161 |
| Alternating generator candidate | 1.101389905 |
| Final refined sinusoidal generator | 1.058386616 |

Each generator was projected into the existing cubic phase spline and evaluated
on 33 fractions. Selection used the objective **before phantom generation**.
The sinusoidal fraction moved from 0.7834375 to **0.77376953125** in the local
129-point refinement (including the coarse winner). Final versus initial objective
reduction was **18.50%**. This is sequential SLSQP plus finite phase-family search,
not joint/free-form/global phase optimization. The final applied phase differs
from the raw generator by 1.0818 rad RMS, explicitly saved and plotted.

The objective excludes MD/FA. Separate diagnostic relative SD bounds at tissue 0
use valid tensor perturbations, fixed orientation, external density and object
phase as a nuisance parameter:

| Parameter | Initial | Flip-only | Final |
|---|---:|---:|---:|
| T1 | 0.239654 | 0.192889 | 0.226226 |
| T2 | 2.683161 | 1.727997 | 1.599106 |
| Density | 0.186380 | 0.161246 | 0.184721 |
| MD | 1.236406 | 0.820358 | 0.921493 |
| FA | 1.303519 | 0.911571 | 0.982596 |

These are derivatives of the current signal model, not certified physical bounds
or the optimized objective. Raw conditional bounds, diagnostic covariance, J/F,
scales and both tissues' results are retained separately.

## Image-domain results

Identical phantom, actual paired angle/phase trains, 32 independent dictionary
combinations, trajectory, weighted-adjoint reconstruction, complex matching and
noise realization were used. The Cartesian reference is the recovery gate;
16 spiral interleaves is a high-sampling comparison, not an exact inverse.

Noiseless global RMSE, **initial → final**:

| Interleaves/frame | T1 (ms) | T2 (ms) | MD (mm²/s) | FA |
|---|---:|---:|---:|---:|
| Cartesian reference | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 1 | 0 → 0 | 13.5785 → 14.1421 | 1.20059e−4 → 1.45237e−4 | 0.146575 → 0.259582 |
| 4 | 0 → 0 | 5.72822 → 10.8253 | 3.24760e−5 → 1.10926e−4 | 0 → 0.0441942 |
| 16 | 0 → 0 | 4.33013 → 5.15388 | 0 → 0 | 0 → 0 |

The same improved/tied/worsened ordering relative to initialization held with
noise SD=0.001 per real/imaginary channel, seed 2026. This fixed absolute noise is
not identical SNR for every sequence. All MAE/RMSE/NRMSE/bias and compartment
metrics, including density, are in `reconstruction_metrics.csv`.

Compared with **flip-angle only**, adding the selected phase reduced noiseless
MD and FA errors at one interleaf while worsening T2 slightly; at four interleaves
T2 and MD improved, with T1/FA tied; at 16, T2 improved and T1/MD/FA tied.
Those limited phase-specific gains did not outperform the educated initial
sequence. This distinction is visible in the three-sequence maps and tables.

## Interpretation and certification

The experiment demonstrates a reproducible computational chain and exact
on-grid reference recovery. It does **not** demonstrate general undersampling
superiority of the final sequence. A better conditional signal-level objective
did not guarantee smaller image-domain matching errors.

Overall status remains **PARTIAL**: the nine documented discrepancies, tensor
attenuation defects, uncalibrated wavevectors, restricted prolate/orientation grid,
spline projection, finite/local phase search, weighted-adjoint artifacts and
analytical UEE disagreement are unresolved. No new external code was copied;
Heesterbeek's repository remains a methodological reference for acquisition.
See the full report's component table and `docs/undersampling_validation.md`.

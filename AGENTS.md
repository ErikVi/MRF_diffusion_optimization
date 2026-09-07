# AGENTS.md

## Project

This repository contains the research code for the MSc thesis:

"On the optimization of diffusion-enhanced Magnetic Resonance Fingerprinting sequences through flip angle and phase train design with comparative phase modulation strategies."

The project implements MRI simulations and sequence optimization using:

* Extended Phase Graph (EPG) simulation
* RF pulse and phase evolution
* T1/T2 relaxation
* Gradient dephasing
* Diffusion attenuation
* Diffusion tensor modelling
* Magnetic Resonance Fingerprinting (MRF)
* B-spline sequence parameterization
* JAX automatic differentiation
* Fisher Information Matrices
* Cramér-Rao Lower Bounds
* Constrained optimization
* Diffusion tensor metrics
* Phase modulation strategies
* Simulated phantom and undersampling experiments

This is scientific research software.

Scientific correctness, reproducibility, and clarity of the physical model are more important than preserving the current implementation or file structure.

---

# Core objective

Transform this repository into a scientifically rigorous, well-structured, readable, maintainable, reproducible research codebase.

The existing repository should be treated as legacy research code.

Do not assume that its current structure is good.

You are explicitly allowed to:

* Create files.
* Create directories.
* Move files.
* Rename files.
* Rename functions.
* Rename classes.
* Rename variables.
* Split large modules.
* Combine genuinely related modules.
* Remove dead code.
* Replace duplicated implementations.
* Improve APIs.
* Add tests.
* Add documentation.
* Create configuration files.
* Create package structure.

However, every substantial scientific change must be validated.

---

# Scientific priority

Use this priority order:

1. Scientific correctness.
2. Numerical correctness and stability.
3. Reproducibility.
4. Scientific clarity.
5. Testability.
6. Architecture.
7. Maintainability.
8. Performance.
9. Style.

Never sacrifice scientific correctness for cleaner code.

Never change a physical convention simply because another convention is more convenient.

Never silently change scientific behaviour.

---

# Repository organization

The final repository should have a clear scientific architecture.

A possible target structure is:

```text
project/
│
├── src/
│   └── mrf_diffusion/
│       ├── epg/
│       │   ├── states.py
│       │   ├── rf.py
│       │   ├── gradients.py
│       │   ├── relaxation.py
│       │   └── diffusion.py
│       │
│       ├── sequence/
│       │   ├── definition.py
│       │   ├── phase_modulation.py
│       │   └── bspline.py
│       │
│       ├── information/
│       │   ├── jacobian.py
│       │   ├── fisher.py
│       │   └── crlb.py
│       │
│       ├── diffusion/
│       │   └── tensor.py
│       │
│       ├── optimization/
│       │   ├── objectives.py
│       │   ├── constraints.py
│       │   └── optimizer.py
│       │
│       └── simulation/
│           └── signal.py
│
├── experiments/
│   ├── optimization/
│   ├── phase_comparison/
│   ├── bspline_benchmarks/
│   └── undersampling/
│
├── tests/
│   ├── epg/
│   ├── diffusion/
│   ├── sequence/
│   ├── information/
│   └── integration/
│
├── docs/
│   ├── scientific_model.md
│   ├── conventions.md
│   ├── units.md
│   ├── architecture.md
│   ├── validation.md
│   └── literature.md
│
├── data/
│   ├── input/
│   └── output/
│
├── README.md
├── pyproject.toml
└── AGENTS.md
```

This is an example target, not a requirement to reproduce this exact structure.

Choose the structure that best represents the actual scientific dependencies discovered in the repository.

Do not reorganize files merely to match this example.

---

# Scientific architecture

Separate the repository into distinct conceptual layers.

## Physics

Contains implementations of physical models.

Examples:

* EPG state evolution.
* RF rotations.
* Gradient operators.
* Relaxation.
* Diffusion.
* Diffusion tensors.

Physics modules must not depend on experiment-specific scripts.

## Sequence definition

Contains:

* Flip angle trains.
* Phase trains.
* TR.
* TE.
* Gradient waveforms.
* Diffusion encoding.
* B-spline parameterizations.
* Phase modulation strategies.

Sequence definitions should be explicit data or objects.

They should not be hidden inside simulation functions.

## Simulation

Contains the actual signal simulation.

A simulator should receive a clearly defined physical and sequence configuration.

It should not contain experiment-specific optimization logic.

## Information theory

Contains:

* Jacobians.
* Fisher Information Matrices.
* CRLB calculations.
* Noise models.
* Parameter ordering.

## Optimization

Contains:

* Objectives.
* Constraints.
* Parameterizations.
* Optimizer configuration.
* Optimization execution.

Optimization code should call reusable simulation and information-theory code.

The physics layer must not know that optimization exists.

## Experiments

Contains scripts that reproduce specific MSc experiments.

Experiment scripts may combine the reusable modules.

They should contain experiment-specific choices.

Do not put experiment-specific assumptions into reusable physics modules.

---

# Naming

Names must make scientific meaning obvious.

Prefer:

```python
simulate_epg()
apply_rf_rotation()
apply_gradient_shift()
apply_relaxation()
apply_diffusion()
simulate_mrf_signal()
calculate_fisher_information()
calculate_crlb()
generate_phase_train()
generate_flip_angle_train()
```

over vague names such as:

```python
epg()
rf()
grad()
relax()
diff()
calc()
run()
```

Use established MRI terminology where appropriate.

Do not replace standard terminology merely to make names longer.

Prefer explicit names when abbreviations create ambiguity.

Examples:

```python
flip_angle_train
phase_train
diffusion_tensor
repetition_time
echo_time
t1
t2
magnetization
gradient_direction
diffusion_coefficient
epg_states
```

Avoid ambiguous names such as:

```python
fa
ph
D
dt
x
arr
tmp
res
```

unless their meaning is genuinely obvious from the immediate context.

Do not use `FA` for both flip angle and fractional anisotropy.

This project contains both concepts.

Use names such as:

```python
flip_angle
flip_angle_train
fractional_anisotropy
```

---

# File naming

File names should describe the scientific responsibility of the module.

Prefer:

```text
rf.py
relaxation.py
diffusion.py
gradients.py
phase_modulation.py
bspline.py
fisher.py
crlb.py
```

Avoid files whose names describe implementation history rather than purpose.

For example, names such as:

```text
BESTMETHOD.py
BSPLINEPROFILER.py
EPG_blocks_jaxcode.py
```

should be replaced when their responsibilities become clear.

Experiment names can remain descriptive where appropriate.

---

# Function design

Each function should have one clear scientific or computational responsibility.

A function should ideally answer:

"What physical or mathematical operation does this function perform?"

Avoid functions that simultaneously:

* Generate sequences.
* Simulate EPG.
* Calculate CRLBs.
* Optimize parameters.
* Save files.
* Generate plots.

Split such workflows into separate functions.

Functions implementing scientific equations should be easy to inspect.

Avoid unnecessary abstraction that hides the underlying physics.

---

# Documentation

Important scientific functions must document:

* Purpose.
* Inputs.
* Outputs.
* Units.
* Array shapes.
* Important conventions.
* Physical assumptions.
* Relevant literature when appropriate.

For example:

```python
def apply_relaxation(
    epg_states,
    duration_ms,
    t1_ms,
    t2_ms,
):
    """Apply T1/T2 relaxation to EPG states.

    Parameters
    ----------
    epg_states:
        EPG state array with shape (...).
    duration_ms:
        Evolution time in milliseconds.
    t1_ms:
        Longitudinal relaxation time in milliseconds.
    t2_ms:
        Transverse relaxation time in milliseconds.
    """
```

Scientific documentation should explain the model, not merely restate the code.

---

# Units

Make physical units explicit.

Pay particular attention to:

* T1.
* T2.
* TE.
* TR.
* Diffusion coefficient.
* Diffusion tensor.
* Gradient strength.
* Gradient duration.
* Gradient separation.
* b-value.
* Flip angle.
* RF phase.

Do not rely on undocumented assumptions such as "this function expects milliseconds".

Where useful, include units in variable names:

```python
echo_time_ms
repetition_time_ms
diffusion_coefficient_mm2_per_s
```

Do not blindly add units to every variable if the surrounding API already makes them unambiguous.

---

# MRI conventions

Explicitly document and preserve:

* EPG state ordering.
* F+ convention.
* F- convention.
* Z convention.
* RF rotation convention.
* RF phase convention.
* Gradient direction convention.
* Complex signal convention.
* Flip angle convention.
* Phase units.
* Diffusion convention.

Never change these silently.

If a convention is ambiguous, investigate the existing implementation, thesis, and relevant literature before modifying it.

---

# EPG

The EPG implementation is scientifically critical.

Verify:

* State representation.
* State ordering.
* F+.
* F-.
* Z.
* Zero-order behaviour.
* RF rotations.
* Gradient shifts.
* Relaxation.
* Diffusion.
* Initial conditions.
* Signal extraction.
* Number of retained EPG states.

Do not refactor the EPG implementation substantially before creating characterization tests.

---

# Diffusion

Verify the complete derivation and implementation of:

* Scalar diffusion.
* Tensor diffusion.
* Gradient-direction dependence.
* Diffusion attenuation.
* EPG-order dependence.
* Diffusion tensor symmetry.
* Tensor positive semidefiniteness.
* MD.
* FA.

Check all units.

Compare implementation against the MSc thesis and relevant literature.

Do not "correct" the implementation based solely on generic MRI knowledge.

---

# JAX

JAX is a core dependency.

Preserve:

* JIT compilation.
* Automatic differentiation.
* Vectorization.
* JAX-compatible numerical operations.
* Required static arguments.
* Correct dtypes.

Before changing a function, determine whether it is used inside:

```python
jax.jit
jax.jacobian
jax.grad
jax.vmap
```

Do not introduce ordinary NumPy/SciPy operations into differentiable JAX paths without verifying compatibility.

Test gradients after refactoring.

---

# Numerical precision

The repository uses JAX x64.

Investigate every explicit dtype conversion.

In particular, check for accidental float32 or complex64 conversion.

Do not change precision merely for consistency.

Determine whether precision affects:

* EPG simulation.
* Jacobians.
* FIM conditioning.
* CRLB values.
* Optimization convergence.

---

# Hard-coded values

Systematically identify:

* Physical constants.
* Tissue parameters.
* Sequence parameters.
* Optimization parameters.
* Plotting parameters.
* File paths.
* Array dimensions.
* Number of EPG states.
* Sequence lengths.
* B-spline parameters.
* Phase modulation parameters.

Move experiment-specific parameters into explicit configuration.

Do not turn every constant into configuration.

Fundamental mathematical constants can remain constants.

---

# Reproducibility

Experiments should be reproducible from explicit configuration.

Avoid hidden global state.

Avoid current-working-directory assumptions.

Avoid machine-specific paths.

Record important:

* Sequence parameters.
* Tissue parameters.
* Optimization parameters.
* Random seeds.
* Numerical tolerances.
* Software versions.

Do not introduce complex experiment tracking unless necessary.

---

# Tests

Build tests around scientific invariants.

Prioritize:

1. EPG initialization.
2. RF rotation.
3. Gradient shifts.
4. Relaxation.
5. Zero-diffusion behaviour.
6. Scalar diffusion.
7. Tensor diffusion.
8. MD.
9. FA.
10. B-spline evaluation.
11. Jacobians.
12. Fisher Information.
13. CRLB.
14. Complete small MRF simulation.
15. Small optimization regression case.

Tests should verify scientific behaviour, not merely whether functions execute.

---

# Regression testing

Before substantial changes:

1. Establish representative reference cases.
2. Record outputs.
3. Make one conceptual change.
4. Re-run the reference cases.
5. Compare results.
6. Explain any difference.

Never silently accept changed scientific results.

A changed result must be classified as:

* Intended scientific correction.
* Numerical difference.
* Refactoring error.
* Previously hidden bug.
* Unknown.

If unknown, investigate before proceeding.

---

# Refactoring

The existing repository structure is not sacred.

You are encouraged to restructure it.

However:

* Refactor in coherent stages.
* Keep scientific behaviour stable during pure refactors.
* Update imports.
* Update documentation.
* Update tests.
* Remove obsolete files only after confirming they are obsolete.
* Do not combine unrelated scientific changes with large structural changes.

Prefer several understandable changes over one enormous rewrite.

---

# Dead and duplicate code

Look for:

* Duplicate functions.
* Commented-out implementations.
* Legacy implementations.
* Duplicate parameter definitions.
* Unused imports.
* Unused variables.
* Unreachable code.

Do not delete scientifically meaningful legacy implementations until you understand why they exist.

If useful, preserve them in documentation or version history rather than active production code.

---

# Performance

Performance matters because optimization repeatedly evaluates simulations.

Investigate:

* Repeated calculations.
* Unnecessary allocations.
* Python loops.
* JAX recompilation.
* Repeated B-spline evaluation.
* Repeated tensor calculations.
* Repeated sequence construction.

Benchmark before making significant performance changes.

Separate JAX compilation time from execution time.

Do not sacrifice scientific clarity for small performance improvements.

---

# Missing dependencies

Identify all missing:

* Python modules.
* Data files.
* NumPy arrays.
* External packages.
* Experimental dependencies.

Do not create fake replacements.

Do not silently remove missing dependencies.

Document what is required to reproduce each experiment.

---

# Git

Before substantial work:

* Inspect git status.
* Do not overwrite unrelated user changes.
* Do not reset or discard work without explicit instruction.
* Keep changes logically separated.
* Show a clear summary of modifications.

Do not modify existing commits unless explicitly instructed.

---

# Workflow

When asked to improve the repository:

## Step 1: Understand

Inspect the relevant repository structure and code.

## Step 2: Map

Create or update documentation describing:

* Architecture.
* Scientific model.
* Important conventions.
* Dependencies.

## Step 3: Validate

Establish tests or reference outputs before changing critical scientific code.

## Step 4: Audit

Look for:

* Scientific errors.
* Numerical problems.
* Hard-coded assumptions.
* Poor naming.
* Poor architecture.
* Duplication.
* Missing tests.
* Reproducibility problems.

## Step 5: Plan

Prioritize findings.

Do not make large speculative changes.

## Step 6: Implement

Make coherent changes.

You are allowed to reorganize the repository substantially.

## Step 7: Validate

Run appropriate tests and representative simulations.

## Step 8: Document

Update documentation when architecture, scientific conventions, or usage changes.

## Step 9: Report

Explain:

* What changed.
* Why it changed.
* Scientific impact.
* Tests performed.
* Numerical differences.
* Remaining uncertainties.

---

# Final standard

The final repository should feel like a professional scientific software project rather than a collection of MSc experiment scripts.

A new researcher should be able to understand:

* Where the MRI physics lives.
* Where the EPG implementation lives.
* Where sequences are defined.
* Where diffusion is implemented.
* Where optimization happens.
* Where information metrics are calculated.
* Where experiments are run.
* Where tests are located.
* What physical conventions the code uses.
* How to reproduce the published results.

The code should be understandable without having to reverse-engineer the author's original MSc workflow.

Clarity is a scientific requirement.

The repository should make the physics easier to inspect, validate, reproduce, and extend.

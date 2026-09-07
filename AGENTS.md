# AGENTS.md

## Project

This repository contains the code for my MSc research project:

"On the robust optimization of diffusion-enhanced MRF sequences through flip angle and phase train design with comparative phase modulation strategies."

The project concerns Magnetic Resonance Fingerprinting (MRF), diffusion-weighted MRI, Extended Phase Graph (EPG) simulations, flip-angle and phase-train optimization, and comparison of phase-modulation strategies.

This is scientific research code. Treat physical and mathematical correctness as more important than superficial code cleanliness.

The repository is based on published MRI literature and physical principles. When modifying scientific code, preserve the intended physical model unless there is strong evidence that the existing implementation is incorrect.

---

## Primary objective

Continuously improve this repository.

When asked to improve, refactor, review, or optimize the code:

1. First understand the entire repository.
2. Identify the scientific purpose of each major component.
3. Trace the flow of data through the simulation.
4. Identify assumptions, approximations, hard-coded values, duplicated logic, fragile code, and unclear interfaces.
5. Check the implementation against the relevant physical and mathematical principles.
6. Improve the code incrementally.
7. Add or improve tests where possible.
8. Verify that changes do not unintentionally change the scientific results.
9. Explain important scientific or numerical changes.

Do not perform large refactors simply because they look aesthetically better.

---

## Scientific correctness comes first

The code implements MRI physics and numerical simulations.

Pay particular attention to:

* Bloch-equation consistency.
* Extended Phase Graph (EPG) formalism.
* EPG state transitions.
* Longitudinal and transverse magnetization.
* RF rotations.
* Flip-angle conventions.
* RF phase conventions.
* Phase cycling and phase modulation.
* Gradient-induced dephasing.
* Diffusion attenuation.
* T1 and T2 relaxation.
* Echo formation.
* Spoiling assumptions.
* Signal evolution.
* MRF dictionary generation.
* Fingerprint construction.
* Optimization objectives.
* Numerical precision and stability.

Never change a physical convention merely to make the implementation more convenient.

If a convention is ambiguous, determine how the existing implementation uses it and document it before changing it.

---

## Literature and physical principles

Treat published literature as a source of truth for the intended physical model.

When a scientific implementation appears questionable:

1. Identify the relevant equation or physical principle.
2. Locate the corresponding implementation.
3. Determine whether the code actually implements the equation correctly.
4. Check units, signs, indexing, rotations, state ordering, and conventions.
5. Only then modify the implementation.

Do not "correct" something based solely on intuition.

If external literature is needed, search for the original paper or another authoritative source.

Prefer primary research papers, textbooks, and established MRI references over blogs or informal explanations.

When a change depends on a paper, record the relevant citation and explain what part of the implementation it supports.

---

## EPG implementation

Treat the EPG implementation as a core scientific component.

Explicitly verify:

* State representation.
* F+, F-, and Z states.
* RF rotation matrices.
* RF phase handling.
* Gradient dephasing.
* State shifting.
* Conjugate relationships.
* Relaxation.
* Diffusion attenuation.
* Initial conditions.
* Echo/readout state.
* Number and indexing of EPG states.
* Treatment of higher-order states.
* Numerical truncation.

Check whether the implementation preserves expected physical symmetries and limiting cases.

Where practical, create tests for analytically known or physically obvious cases.

Examples include:

* Zero flip angle.
* Zero diffusion.
* Zero gradient.
* No relaxation.
* Very long T1/T2.
* Single RF pulse.
* Constant flip-angle train.
* Known spin-echo behaviour.
* Known spoiled-gradient behaviour.
* Comparison against an independent implementation where possible.

---

## MRI conventions

Make conventions explicit.

Do not leave important scientific conventions hidden in arbitrary code.

Examples include:

* Degrees vs radians.
* Seconds vs milliseconds.
* Hz vs rad/s.
* Tesla.
* Diffusion coefficient units.
* Gradient units.
* Gyromagnetic ratio.
* RF phase convention.
* Flip-angle convention.
* Complex signal convention.
* EPG state ordering.
* Echo-time definition.
* Repetition-time definition.

If a quantity has a unit, make that unit clear in its variable name, documentation, validation, or API.

Prefer names such as:

`T1_ms`

over ambiguous names such as:

`T1`

when the distinction matters.

---

## Remove hard-coded assumptions

One of the main goals of this project is to make the implementation as general and configurable as reasonably possible.

Search systematically for:

* Hard-coded physical constants.
* Hard-coded tissue parameters.
* Hard-coded sequence parameters.
* Hard-coded flip-angle trains.
* Hard-coded phase trains.
* Hard-coded diffusion coefficients.
* Hard-coded T1/T2 values.
* Hard-coded number of TRs.
* Hard-coded EPG state counts.
* Hard-coded optimization bounds.
* Hard-coded sampling intervals.
* Magic numbers.
* Hard-coded file paths.
* Hard-coded array dimensions.
* Hard-coded optimization settings.
* Duplicated parameter definitions.

Replace hard-coded values with explicit parameters or configuration objects when appropriate.

Do not blindly turn every number into a parameter.

A constant that is genuinely fundamental to the algorithm may remain internal if making it configurable would make the code worse.

The goal is meaningful configurability, not configuration for its own sake.

---

## Separate physics from experiment configuration

Where appropriate, separate:

1. Physical models.
2. Sequence definitions.
3. Simulation parameters.
4. Optimization parameters.
5. Data processing.
6. Visualization.
7. Experiment-specific configuration.

For example, an EPG simulator should ideally not need to know which particular optimization experiment is being performed.

A sequence definition should ideally be capable of being passed into the simulator rather than being embedded inside the simulator.

Optimization code should operate on clearly defined parameters rather than reaching into simulation internals.

---

## Reproducibility

Scientific results must be reproducible.

Make sure simulations and optimization runs have clearly defined:

* Input parameters.
* Random seeds where randomness exists.
* Optimization settings.
* Initial conditions.
* Numerical tolerances.
* Parameter ranges.
* Output locations.
* Software dependencies.

Avoid hidden global state.

Avoid dependence on the current working directory where practical.

Avoid implicit configuration.

If a result depends on a parameter, make that dependency explicit.

---

## Code architecture

Prefer small functions with one clear responsibility.

Prefer explicit data flow over global variables.

Prefer meaningful names over abbreviated names.

Prefer reusable scientific functions over duplicated implementations.

Avoid unnecessary abstraction.

Do not introduce elaborate class hierarchies unless they clearly improve the scientific model or maintainability.

Keep numerical kernels easy to inspect.

A researcher should be able to open a function and understand which physical operation it performs.

---

## Numerical correctness

When modifying numerical code, check:

* Array shapes.
* Broadcasting.
* Complex-valued calculations.
* Floating-point precision.
* Numerical stability.
* Boundary conditions.
* Indexing.
* Vectorization.
* Memory usage.
* Convergence.
* Optimization tolerances.

Do not replace a clear implementation with a faster implementation unless the new implementation can be verified to produce equivalent results within an appropriate numerical tolerance.

For performance improvements, benchmark before and after when practical.

Scientific correctness takes priority over speed.

---

## Testing philosophy

Tests should verify scientific behaviour, not only whether functions execute.

Prefer tests that answer questions such as:

* Does the simulator obey the expected physical behaviour?
* Does changing diffusion coefficient affect signal evolution correctly?
* Does changing T1/T2 produce the expected relaxation behaviour?
* Does RF phase produce the expected EPG state evolution?
* Does the implementation converge appropriately with increasing EPG state count?
* Does a simplified case reproduce a known analytical or literature result?
* Does refactoring preserve previous results?

When possible, use independent calculations or analytical limiting cases as references.

For optimization code, test the objective function independently from the optimizer.

---

## Regression protection

Before making substantial changes, identify important existing outputs or reference results.

When possible:

1. Run the existing implementation.
2. Save representative outputs.
3. Make the change.
4. Run the same case again.
5. Compare the results.

If results change, determine whether the change is:

* An intended scientific correction.
* A numerical difference.
* A refactoring error.
* A previously hidden bug.

Never silently accept changed scientific results.

---

## Refactoring policy

Refactor in small, understandable steps.

Before a major refactor:

* Understand the existing behaviour.
* Identify dependencies.
* Identify important outputs.
* Add tests where necessary.
* Make one conceptual change at a time.

Do not combine a major architectural refactor with an unrelated scientific modification unless necessary.

Preserve behaviour first. Improve architecture second.

---

## Performance

Performance matters because MRI simulations and optimization can involve many repeated simulations.

Look for:

* Unnecessary Python loops.
* Repeated calculations.
* Repeated allocation of large arrays.
* Unnecessary copies.
* Inefficient optimization objectives.
* Redundant simulation work.
* Opportunities for vectorization.
* Opportunities to cache calculations.

However:

Do not optimize prematurely.

Do not sacrifice readability or physical transparency for small performance gains.

Measure performance before making major performance-driven changes.

---

## Scientific documentation

Important scientific functions should explain:

* What physical process they represent.
* What the inputs mean.
* Their units.
* What convention they use.
* What they return.
* Important assumptions.
* Relevant literature where appropriate.

Use equations in documentation when they clarify the implementation.

For example, if a function implements an EPG transition, document the mathematical operation rather than merely saying "updates the EPG state."

---

## Visualization

Plots are part of scientific validation.

When improving plotting code:

* Label physical quantities clearly.
* Include units.
* Use scientifically meaningful axes.
* Avoid misleading normalization.
* Make assumptions explicit.
* Keep plotting separate from simulation logic.
* Make plots reproducible from stored results where practical.

Do not alter scientific data merely to make a plot look better.

---

## File and data handling

Avoid hard-coded absolute paths.

Use repository-relative paths or explicit configuration.

Do not commit:

* Large generated datasets unless intentionally part of the project.
* Temporary files.
* Machine-specific paths.
* Credentials.
* API keys.
* Personal information.

Do not delete existing research data unless explicitly instructed.

---

## Git discipline

Before modifying substantial code:

* Inspect the current Git state.
* Understand recent changes when relevant.
* Avoid overwriting unrelated user work.

Keep changes logically separated.

Do not create commits unless explicitly asked.

Do not reset, checkout, or discard user changes unless explicitly instructed.

---

## How to work on requests

For a request such as "improve the code":

### Phase 1: Understand

Inspect the repository before changing anything.

Identify:

* Main entry points.
* Simulation code.
* EPG implementation.
* Optimization code.
* Data processing.
* Plotting.
* Configuration.
* Tests.
* Documentation.
* Dependencies.

Create a concise internal map of the architecture.

### Phase 2: Audit

Look for:

* Scientific errors.
* Incorrect assumptions.
* Hard-coded parameters.
* Duplicated logic.
* Poor naming.
* Hidden state.
* Fragile indexing.
* Missing validation.
* Missing tests.
* Numerical inefficiencies.
* Reproducibility problems.
* Poor separation of concerns.

Rank findings by importance.

Prioritize scientific correctness over style.

### Phase 3: Plan

Before making a large change, describe:

1. What is wrong.
2. Why it matters.
3. What you will change.
4. How you will validate it.

Do not make large speculative changes.

### Phase 4: Implement

Make the smallest coherent set of changes.

Keep scientific behaviour unchanged unless the purpose of the change is explicitly to correct scientific behaviour.

### Phase 5: Validate

Run all relevant tests and checks.

For scientific changes, run representative simulations.

Compare outputs against the previous implementation or known reference cases.

Report:

* Tests performed.
* Results.
* Numerical differences.
* Any remaining uncertainty.

---

## Important behaviour

Do not assume that existing code is correct.

Do not assume that existing code is wrong.

Treat every important scientific operation as something that can be investigated.

Do not rewrite the repository from scratch.

Do not make broad changes merely because another coding style is more fashionable.

Do not remove seemingly redundant code until you understand why it exists.

Do not change scientific conventions silently.

Do not invent scientific justification.

When uncertain, investigate first.

The ideal result is code that is:

* Scientifically trustworthy.
* Reproducible.
* General.
* Modular.
* Testable.
* Understandable to an MRI physicist.
* Efficient enough for optimization.
* Easy to extend to new MRF sequence designs.

The final code should make the underlying MRI physics easier to inspect, not harder.

# External experiment inputs

The repository does not contain the measured/optimized arrays required by the MSc scripts.
Supply `fa_array_initial.npy` (1-D radians), `DIFFPREPARATION/fa_train_diffprep.npy`
and `DIFFPREPARATION/pm_train_diffprep.npy` (preparation trains in radians).
Do not substitute synthetic data when claiming reproduction of thesis experiments.
The `--dry-run` entry points and scientific tests require none of these arrays.
Phantom reconstruction also requires the original external `UEEphase_DH` module.
Its source/distribution is not supplied. The unused `BlochSimulation_DH` import
was removed, not replaced; it was never called by the original workflow.

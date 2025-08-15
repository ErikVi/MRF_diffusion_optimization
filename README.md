# MRF_diffusion_optimization
Codes used for the purpose of the generation of results for the MSc project titled "On the robust optimization of diffusion-enhanced MRF sequences through flip angle and phase train design with comparative phase modulation strategies"

- BESTMETHOD.py: This script evaluates and compares different phase modulation strategies for Magnetic Resonance Fingerprinting (MRF) flip angle trains using an Extended Phase Graph (EPG) simulation in JAX. It loads a predefined flip angle (FA) sequence, represents it with B-spline coefficients, and systematically applies various phase modulation methods (quadratic, linear, sinusoidal, alternating) at different modulation fractions. For each configuration, it computes normalized Cramér–Rao Lower Bounds (nCRLBs) for T1, T2, M0, fractional anisotropy (FA), and mean diffusivity (MD). The script identifies the phase modulation method and fraction that minimize the total nCRLB, and generates summary plots of performance metrics.
- BSPLINEPROFILER.py: Benchmarks and profiles multiple B-spline evaluation implementations (bspline, bspline_vectorized, bspline_vmap) in terms of execution time and memory usage when reconstructing FA sequences from coefficients. This is intended to guide performance optimization in high-iteration MRI simulations.
- EPG_blocks_jaxcode.py: Core EPG simulation library implementing RF pulses, gradient dephasing, relaxation, diffusion (scalar and tensor models), and phase modulation schemes. Provides B-spline utilities for representing and fitting flip angle and phase trains, as well as functions for computing Jacobians, Fisher Information Matrices (FIM), and Cramér–Rao Lower Bounds (CRLBs) for multiple tissue parameters, including T1, T2, M0, fractional anisotropy (FA), and mean diffusivity (MD). Supports both standard and generalized multi-directional diffusion encoding.
- FILEREADER.py: This utility script inspects and prints the contents of an HDF5 file (iterations.h5) using the h5py library. It recursively traverses all groups and datasets in the file, listing each dataset’s path and displaying its contents (scalars or arrays). The script will raise a FileNotFoundError if the file is missing from the current working directory.
- OPTIMIZATION.py: This script performs flip angle (FA) and phase modulation (PM) sequence optimization for Magnetic Resonance Fingerprinting (MRF) using a JAX-based Extended Phase Graph (EPG) simulation framework. It supports multiple phase modulation strategies (free form, no phase modulation, quadratic, and quadratic malleable), applying them to B-spline–parameterized FA trains under physical and smoothness constraints.

For each configuration (knots, sequence length, EPG state count, gradient count, and objective type), the script:

  - Initializes FA and PM coefficients and constraint functions.
  - Runs constrained nonlinear optimization (SLSQP) to minimize a holistic objective function based on weighted Cramér–Rao Lower Bounds (CRLBs) for multiple tissue parameters (T1, T2, M0, fractional anisotropy, mean diffusivity).
  - Saves optimized FA and PM sequences, their plots, and simulated signal outputs across diffusion values.
  - Produces comparative visualizations of initial vs. optimized nCRLBs both per-parameter and averaged across parameters.

Results are organized into directories for reproducibility and downstream analysis, with all figures and NumPy arrays saved for later inspection.

- UNDERSAMPLING.py: This code generates simulated diffusion MRI phantom images with varying T1, T2, fractional anisotropy (FA), and mean diffusivity (MD) values. It constructs a dictionary of simulated signals for combinations of relaxation and diffusion parameters, and then generates checkerboard-style phantoms. The code also applies phase modulation, undersampling in k-space (spiral trajectories), and optional noise to the simulated signals.
Output includes:

  - FA and MD maps of the phantom
  - Relaxation parameter images
  - Undersampled k-space data

The workflow uses JAX for fast numerical computation, SigPy for NUFFT operations, and custom modules for Bloch/EPG simulations and phase modulation.

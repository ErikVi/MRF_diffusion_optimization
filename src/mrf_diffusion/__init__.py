"""Diffusion-enhanced MRF research models; see docs/validation.md for known defects."""

import jax

# Preserve the original numerical policy before any package arrays are created.
jax.config.update("jax_enable_x64", True)

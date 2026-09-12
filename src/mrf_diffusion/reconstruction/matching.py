"""Magnitude-dictionary matching and legacy scale-grid lookup."""

import numpy as np
import jax.numpy as jnp


def match_magnitude_dictionary(normalized_dictionary, signal, keys, signal_norm):
    signal_norm_value = np.linalg.norm(signal)
    if signal_norm_value == 0:
        raise ValueError("Zero sgnal")
    signal_normalized = signal / signal_norm_value
    inner_products = jnp.dot(normalized_dictionary, signal_normalized.flatten())
    idx_best = jnp.argmax(jnp.abs(inner_products))
    best_params = [keys[idx_best]]
    M0 = signal_norm_value / signal_norm[idx_best]
    inprod_val = inner_products[idx_best]
    return (best_params, M0, inprod_val)


def closest_scale_pair(param_pairs, target):
    diffs = param_pairs - jnp.array(target)
    distances = jnp.linalg.norm(diffs, axis=2)
    index = jnp.unravel_index(jnp.argmin(distances), distances.shape)
    return index


def stack_scale_maps(anisotropy_map, diffusivity_map):
    return jnp.stack([diffusivity_map, anisotropy_map], axis=-1)

"""Spatial tissue assignment with tensors as the source of diffusion truth."""

from dataclasses import dataclass
import numpy as np
from mrf_diffusion.diffusion.parameterization import (
    axisymmetric_tensor,
    tensor_invariants,
)


@dataclass(frozen=True)
class PhantomMaps:
    """Maps (H,W), tensor (H,W,3,3) in xyz, object_phase_map in radians.

    proton_density is an EXTERNAL multiplier of a simulation at equilibrium M=1,
    not a fitted physical equilibrium-M parameter. T1/T2 are ms. Outside support
    all maps are zeroed. MD/FA/orientation are derived properties, never stored
    independently of D. Arrays are copied and read-only.
    """

    support: np.ndarray
    t1_ms: np.ndarray
    t2_ms: np.ndarray
    proton_density: np.ndarray
    diffusion_tensor: np.ndarray
    object_phase_map: np.ndarray

    def __post_init__(self):
        support = np.asarray(self.support)
        if support.ndim != 2 or support.dtype != bool or not np.any(support):
            raise ValueError("support must be a nonempty boolean (H,W) mask")
        support = support.copy()
        support.setflags(write=False)
        object.__setattr__(self, "support", support)
        for name in (
            "t1_ms",
            "t2_ms",
            "proton_density",
            "object_phase_map",
            "diffusion_tensor",
        ):
            values = np.array(getattr(self, name), dtype=float, copy=True)
            expected = support.shape + ((3, 3) if name == "diffusion_tensor" else ())
            if values.shape != expected or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite with shape {expected}")
            if name in ("t1_ms", "t2_ms") and np.any(values[support] <= 0):
                raise ValueError("Active tissue relaxation times must be positive")
            if name == "proton_density" and np.any(values < 0):
                raise ValueError("proton_density cannot be negative")
            values[~support] = 0
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        md, _, _ = tensor_invariants(self.diffusion_tensor)
        if np.any(md[support] <= 0):
            raise ValueError("Active tensors must have positive trace")

    @property
    def mean_diffusivity_map(self):
        return tensor_invariants(self.diffusion_tensor)[0]

    @property
    def fractional_anisotropy_map(self):
        return tensor_invariants(self.diffusion_tensor)[1]

    @property
    def principal_direction_map(self):
        return tensor_invariants(self.diffusion_tensor)[2]


def assign_tissue_maps(
    labels,
    support,
    *,
    t1_ms,
    t2_ms,
    proton_density,
    mean_diffusivity,
    fractional_anisotropy,
    principal_directions,
    object_phase_map=None,
):
    """Assign label-indexed tissue arrays to a geometry, with valid prolate tensors."""
    labels = np.asarray(labels)
    support = np.asarray(support)
    if labels.shape != support.shape or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be an integer map matching support")
    count = len(t1_ms)
    if any(
        np.asarray(v).shape != (count,)
        for v in (t1_ms, t2_ms, proton_density, mean_diffusivity, fractional_anisotropy)
    ):
        raise ValueError("Tissue parameter tables must have equal lengths")
    if np.asarray(principal_directions).shape != (count, 3):
        raise ValueError("One xyz principal direction is required per tissue")
    if np.any(labels < 0) or np.any(labels >= count):
        raise ValueError("Tissue label is outside the parameter table")
    tensors = axisymmetric_tensor(
        mean_diffusivity, fractional_anisotropy, principal_directions
    )
    return PhantomMaps(
        support,
        np.asarray(t1_ms)[labels],
        np.asarray(t2_ms)[labels],
        np.asarray(proton_density)[labels],
        tensors[labels],
        np.zeros(labels.shape) if object_phase_map is None else object_phase_map,
    )

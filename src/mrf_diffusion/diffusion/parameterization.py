"""Validated host tensor construction for phantoms; no legacy metrics are changed."""

import numpy as np


def axisymmetric_tensor(mean_diffusivity, fractional_anisotropy, principal_direction):
    """Return (...,3,3) prolate tensors in mm²/s, in the simulator's xyz frame.

    MD>0, 0<=FA<=1; direction (...,3) is normalized explicitly.
    a=FA/sqrt(3-2 FA²); lambda_parallel=MD*(1+2a),
    lambda_perpendicular=MD*(1-a). D=lambda_perp*I+3*MD*a*n*n.T.
    This chosen two-eigenvalue family is NOT a unique general MD/FA inversion.
    FA=1 is a valid rank-one limit; direction sign does not affect D.
    """
    md, fa = np.broadcast_arrays(
        np.asarray(mean_diffusivity, float), np.asarray(fractional_anisotropy, float)
    )
    direction = np.asarray(principal_direction, float)
    if direction.shape[-1:] != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("principal_direction must be finite (...,3) xyz vectors")
    if not np.all(np.isfinite(md)) or not np.all(np.isfinite(fa)):
        raise ValueError("MD and FA must be finite")
    if np.any(md <= 0) or np.any((fa < 0) | (fa > 1)):
        raise ValueError("Require MD>0 and 0<=FA<=1")
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("principal direction cannot be zero")
    unit = direction / norm
    a = fa / np.sqrt(3 - 2 * fa**2)
    return (md * (1 - a))[..., None, None] * np.eye(3) + (3 * md * a)[
        ..., None, None
    ] * unit[..., :, None] * unit[..., None, :]


def tensor_invariants(diffusion_tensor):
    """Return physical MD (mm²/s), FA and principal xyz axis from symmetric PSD D.

    D has shape (...,3,3). Background D=0 has MD=0, FA=0 by explicit mask
    convention; no physical anisotropy is claimed there. Isotropic tensors have
    axis=0 because orientation is undefined. Eigenvector signs are arbitrary.
    """
    tensor = np.asarray(diffusion_tensor, float)
    if tensor.shape[-2:] != (3, 3) or not np.all(np.isfinite(tensor)):
        raise ValueError("tensor must be finite (...,3,3)")
    if not np.allclose(tensor, tensor.swapaxes(-1, -2), atol=1e-14, rtol=1e-12):
        raise ValueError("tensor must be symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    tolerance = 1e-12 * np.maximum(np.max(np.abs(eigenvalues), axis=-1), 1e-15)
    if np.any(eigenvalues[..., 0] < -tolerance):
        raise ValueError("tensor must be positive semidefinite")
    md = eigenvalues.mean(axis=-1)
    denominator = np.sum(eigenvalues**2, axis=-1)
    fa = np.sqrt(
        np.divide(
            1.5 * np.sum((eigenvalues - md[..., None]) ** 2, axis=-1),
            denominator,
            out=np.zeros_like(md),
            where=denominator > 0,
        )
    )
    axis = eigenvectors[..., :, -1].copy()
    axis = np.where((fa > 1e-10)[..., None], axis, 0)
    return md, fa, axis

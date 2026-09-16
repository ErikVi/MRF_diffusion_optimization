"""Complex least-squares matching, separate from image reconstruction.

Fits one complex scalar per voxel across ALL frames/directions. Relative temporal
phase is retained. This is not magnitude matching or independent per-frame phase.
"""

from dataclasses import dataclass
import numpy as np
from mrf_diffusion.diffusion.parameterization import tensor_invariants


@dataclass(frozen=True)
class QuantitativeMaps:
    """Maps (H,W), tensor (H,W,3,3). Invalid voxels have NaN estimates/index -1."""

    maps: dict
    tensors: np.ndarray
    complex_scale: np.ndarray
    dictionary_index: np.ndarray
    valid: np.ndarray
    ambiguous: np.ndarray
    correlation_margin: np.ndarray
    relative_residual: np.ndarray


def match_complex_dictionary(
    images, dictionary, support, *, batch_size=256, ambiguity_tolerance=1e-10
):
    """Select argmax |d^H y|²/(d^H d); fit beta=d^H y/(d^H d).

    PD=abs(beta), object phase=arg(beta). MD/FA come from the selected tensor.
    Constant phase is a nuisance; time-dependent RF phase remains in each atom.
    Finite nonzero masked signals are fitted; invalids remain explicit. Ties
    return a candidate AND ambiguous=True and must not be interpreted as certainty.
    """
    images, support = np.asarray(images), np.asarray(support)
    signals = np.asarray(dictionary.signals)
    if support.dtype != bool or images.ndim != 3 or support.shape != images.shape[1:]:
        raise ValueError("Expected images (F,H,W) and boolean support (H,W)")
    if signals.ndim != 2 or signals.shape[1] != len(images) or len(signals) == 0:
        raise ValueError("Dictionary frames must match images")
    if not np.all(np.isfinite(signals)) or np.any(np.linalg.norm(signals, axis=1) == 0):
        raise ValueError("Dictionary atoms must be finite and nonzero")
    if (
        not isinstance(batch_size, int)
        or batch_size < 1
        or not np.isfinite(ambiguity_tolerance)
        or ambiguity_tolerance < 0
    ):
        raise ValueError("Invalid matching batch or ambiguity tolerance")
    flat = images.reshape(len(images), -1).T
    norm_y = np.linalg.norm(flat, axis=1)
    valid = support.ravel() & np.all(np.isfinite(flat), axis=1) & (norm_y > 0)
    indices = np.full(len(flat), -1, int)
    scale = np.full(len(flat), np.nan + 1j * np.nan, complex)
    margin = np.full(len(flat), np.nan)
    residual = np.full(len(flat), np.nan)
    norm_d = np.linalg.norm(signals, axis=1)
    normalized = signals / norm_d[:, None]
    pixels = np.flatnonzero(valid)
    for start in range(0, len(pixels), batch_size):
        loc = pixels[start : start + batch_size]
        corr = normalized.conj() @ (flat[loc] / norm_y[loc, None]).T
        score = np.abs(corr) ** 2
        best = np.argmax(score, axis=0)
        column = np.arange(len(loc))
        winner = score[best, column]
        runner_up = (
            np.partition(score, -2, axis=0)[-2]
            if len(signals) > 1
            else np.zeros(len(loc))
        )
        indices[loc] = best
        scale[loc] = corr[best, column] * norm_y[loc] / norm_d[best]
        margin[loc] = np.maximum(winner - runner_up, 0)
        # Direct residual avoids cancellation for exact matches.
        residual[loc] = (
            np.linalg.norm(flat[loc] - scale[loc, None] * signals[best], axis=1)
            / norm_y[loc]
        )
    tensor = np.full((len(flat), 3, 3), np.nan)
    tensor[valid] = dictionary.tensors[indices[valid]]
    maps = {
        name: np.full(len(flat), np.nan)
        for name in (
            "t1_ms",
            "t2_ms",
            "md_mm2_per_s",
            "fa",
            "proton_density",
            "object_phase_rad",
        )
    }
    maps["t1_ms"][valid] = dictionary.t1_ms[indices[valid]]
    maps["t2_ms"][valid] = dictionary.t2_ms[indices[valid]]
    if np.any(valid):
        md, fa, _ = tensor_invariants(tensor[valid])
        maps["md_mm2_per_s"][valid], maps["fa"][valid] = md, fa
    maps["proton_density"][valid] = np.abs(scale[valid])
    maps["object_phase_rad"][valid] = np.angle(scale[valid])
    shape = support.shape
    return QuantitativeMaps(
        {k: v.reshape(shape) for k, v in maps.items()},
        tensor.reshape(*shape, 3, 3),
        scale.reshape(shape),
        indices.reshape(shape),
        valid.reshape(shape),
        (valid & (margin <= ambiguity_tolerance)).reshape(shape),
        margin.reshape(shape),
        residual.reshape(shape),
    )

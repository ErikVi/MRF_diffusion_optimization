"""Masked parameter errors in native units; no cross-parameter aggregate."""

import numpy as np


def parameter_errors(truth, recovered, mask):
    """NRMSE=RMSE/RMS(truth); undefined for zero RMS truth (reported as None).

    Report exclusions explicitly. Nonfinite recovered values never silently count
    as correct; callers must inspect valid/invalid counts alongside finite errors.
    """
    truth, recovered, mask = np.asarray(truth), np.asarray(recovered), np.asarray(mask)
    if (
        truth.shape != recovered.shape
        or mask.shape != truth.shape
        or mask.dtype != bool
    ):
        raise ValueError("Truth, estimate and boolean mask must share shape")
    selected = mask & np.isfinite(truth) & np.isfinite(recovered)
    count = int(selected.sum())
    result = dict(
        requested_voxels=int(mask.sum()),
        valid_voxels=count,
        invalid_voxels=int(mask.sum()) - count,
    )
    if not count:
        return {**result, "mae": None, "rmse": None, "nrmse": None, "bias": None}
    error = recovered[selected] - truth[selected]
    rmse = float(np.sqrt(np.mean(error**2)))
    denominator = float(np.sqrt(np.mean(truth[selected] ** 2)))
    return {
        **result,
        "mae": float(np.mean(np.abs(error))),
        "rmse": rmse,
        "nrmse": rmse / denominator if denominator > 0 else None,
        "bias": float(np.mean(error)),
    }


def evaluate_parameter_maps(truth_maps, recovered_maps, support, regions):
    """Global and per-compartment errors. Region labels are evaluation-only."""
    output = {}
    masks = {"all": support}
    masks.update(
        {
            f"region_{label}": support & (regions == label)
            for label in np.unique(regions[support])
        }
    )
    for region, mask in masks.items():
        output[region] = {
            name: parameter_errors(truth, recovered_maps[name], mask)
            for name, truth in truth_maps.items()
        }
    return output

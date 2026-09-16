"""Load paired, evaluated sequence trains without inventing optimized phases."""

from pathlib import Path
import numpy as np
from .definition import MRFSequence


def load_sequence_archive(path, settings):
    """Load explicit radian trains from NPZ; no truncation or phase regeneration.

    Required keys: flip_angles_rad, rf_phases_rad. Optional paired keys:
    preparation_flip_angles_rad, preparation_rf_phases_rad. Caller provides the
    exact static physics settings separately. Archive provenance must be retained
    by the experiment; 'optimized' labels do not imply optimization was rerun.
    """
    with np.load(Path(path), allow_pickle=False) as archive:
        angles = np.asarray(archive["flip_angles_rad"], float)
        phases = np.asarray(archive["rf_phases_rad"], float)
        prep_keys = ("preparation_flip_angles_rad", "preparation_rf_phases_rad")
        if (prep_keys[0] in archive) != (prep_keys[1] in archive):
            raise ValueError("Preparation angles and RF phases must be paired")
        prep = (
            [np.asarray(archive[k], float) for k in prep_keys]
            if prep_keys[0] in archive
            else [None, None]
        )
    for first, second in ((angles, phases),):
        if (
            first.ndim != 1
            or first.size == 0
            or first.shape != second.shape
            or not (np.all(np.isfinite(first)) and np.all(np.isfinite(second)))
        ):
            raise ValueError(
                "Sequence archive requires paired finite 1-D radian trains"
            )
    if prep[0] is not None and (
        prep[0].ndim != 1
        or prep[0].size == 0
        or prep[0].shape != prep[1].shape
        or not all(np.all(np.isfinite(x)) for x in prep)
    ):
        raise ValueError("Invalid preparation trains")
    return MRFSequence(angles, phases, settings, *prep)

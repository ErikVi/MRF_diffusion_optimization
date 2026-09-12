"""Display historical phantom outputs; labels distinguish legacy trace from MD."""

from pathlib import Path
import numpy as np


def plot_phantom_map(
    values, directory, filename, title, label, *, cmap=None, limits=None
):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    image = axis.imshow(np.asarray(values), cmap=cmap)
    if limits is not None:
        image.set_clim(*limits)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label=label)
    figure.tight_layout()
    figure.savefig(Path(directory) / filename)
    plt.close(figure)


def plot_reconstruction_maps(
    directory,
    t1_map,
    t2_map,
    anisotropy_map,
    legacy_diffusivity_map,
    phase_map,
    relative_t1_error,
    relative_t2_error,
    rms_t1,
    rms_t2,
):
    for name, values, rms in [
        ("T1", relative_t1_error, rms_t1),
        ("T2", relative_t2_error, rms_t2),
    ]:
        plot_phantom_map(
            values,
            directory,
            f"Sim Error approximation {name}.png",
            f"Simulated MRF error in {name}. RMS = {rms:.1f}",
            "Relative error",
            cmap="RdBu_r",
            limits=(-0.4, 0.4),
        )
    for filename, values, title, label in [
        ("FA_image.png", anisotropy_map, "Legacy reconstructed FA", "FA"),
        (
            "MD_image.png",
            legacy_diffusivity_map,
            "Legacy reconstructed tensor trace",
            "trace(D) [mm^2/s]",
        ),
        ("T1_image.png", t1_map, "T1 Map", "T1 [ms]"),
        ("T2_image.png", t2_map, "T2 Map", "T2 [ms]"),
    ]:
        plot_phantom_map(values, directory, filename, title, label, cmap="magma")
    plot_phantom_map(
        phase_map,
        directory,
        "Phase.png",
        "Phase",
        "Phase [rad]",
        cmap="twilight_shifted",
        limits=(-np.pi, np.pi),
    )

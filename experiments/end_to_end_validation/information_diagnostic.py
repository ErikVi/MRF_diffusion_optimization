"""Backward-compatible adapter for the original experiment diagnostic."""

import numpy as np
from mrf_diffusion.information.tensor_diagnostic import evaluate_tensor_information


def evaluate(sequence, options, directory, name, sigma=10 ** (-1.65)):
    arrays, summary = evaluate_tensor_information(
        sequence,
        options,
        [750.0, 70.0, 1.0, 0.0007, 0.2, 0.0],
        (1.0, 0.0, 0.0),
        [750.0, 70.0, 1.0, 0.0007, 0.2, 1.0],
        sigma=sigma,
    )
    np.savez(directory / (name + "_physical_parameter_information.npz"), **arrays)
    return summary

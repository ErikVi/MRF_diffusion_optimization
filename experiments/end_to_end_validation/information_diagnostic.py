"""Finite-difference audit of the existing simulator on a valid tensor family.

This diagnostic is not an optimization objective or replacement MRI model.
Orientation is fixed; a constant object phase is a fitted nuisance parameter.
"""

import numpy as np
from mrf_diffusion.diffusion.parameterization import axisymmetric_tensor
from mrf_diffusion.simulation.api import simulate_mrf_signal
from mrf_diffusion.simulation.tissue import TissueParameters


def evaluate(sequence, options, directory, name, sigma=10 ** (-1.65)):
    parameters = np.array([750.0, 70.0, 1.0, 0.0007, 0.2, 0.0])
    scales = np.array([750.0, 70.0, 1.0, 0.0007, 0.2, 1.0])

    def signal(values):
        t1, t2, density, md, fa, object_phase = values
        tensor = axisymmetric_tensor(md, fa, (1.0, 0.0, 0.0))
        components = np.asarray(
            simulate_mrf_signal(
                TissueParameters(t1, t2, 1.0, tensor), sequence, options
            )
        )
        complex_signal = (
            density * np.exp(1j * object_phase) * (components[0] + 1j * components[1])
        )
        return np.concatenate((complex_signal.real, complex_signal.imag))

    def jacobian(step):
        columns = []
        for i in range(len(parameters)):
            delta = np.zeros(6)
            delta[i] = step * scales[i]
            columns.append(
                (signal(parameters + delta) - signal(parameters - delta))
                / (2 * delta[i])
            )
        return np.array(columns).T

    coarse = jacobian(1e-4)
    jac = jacobian(5e-5)
    relative = float(
        np.linalg.norm((jac - coarse) * scales) / np.linalg.norm(jac * scales)
    )
    if relative > 1e-5:
        raise ValueError("Finite-difference information diagnostic did not converge")
    scaled = jac * scales / sigma
    fim = jac.T @ jac / sigma**2
    scaled_fim = scaled.T @ scaled
    rank = int(np.linalg.matrix_rank(scaled_fim))
    covariance = np.linalg.inv(scaled_fim) if rank == 6 else np.full((6, 6), np.nan)
    relative_sd = np.sqrt(np.diag(covariance))
    np.savez(
        directory / (name + "_physical_parameter_information.npz"),
        parameters=parameters,
        scales=scales,
        jacobian=jac,
        fim=fim,
        scaled_fim=scaled_fim,
        relative_sd=relative_sd,
    )
    return dict(
        parameter_order=["T1", "T2", "proton_density", "MD", "FA", "object_phase_rad"],
        relative_sd=relative_sd.tolist() if rank == 6 else None,
        rank=rank,
        scaled_condition=float(np.linalg.cond(scaled_fim)),
        derivative_step_convergence=relative,
        sigma_per_channel=sigma,
        scope="Finite-difference diagnostic, fixed orientation, current legacy tensor signal; NOT certified physical CRLB and NOT optimized objective",
    )

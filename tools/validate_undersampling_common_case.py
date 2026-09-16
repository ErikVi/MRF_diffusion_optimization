"""Validation only: run pinned external helpers and compare common MRI cases.

No external source is vendored or modified. Obtain the GPL reference separately.
No package physics or scientific capability is implemented by this harness.
"""

import argparse
import ast
from contextlib import contextmanager
from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import scipy as sc
import scipy.signal
import sigpy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mrf_diffusion.encoding import (
    NufftOperator,
    calculate_point_spread_function,
    direct_sampling_psf,
)
from mrf_diffusion.encoding.trajectory import (
    load_spiral_coordinates,
    generate_rotated_spiral_trajectory,
    cartesian_trajectory,
)
from mrf_diffusion.encoding.density_compensation import (
    radial_increment_density_compensation,
)
from mrf_diffusion.encoding.series import acquire_image_series
from mrf_diffusion.reconstruction.series import reconstruct_acquisition_frames
from mrf_diffusion.reconstruction.quantitative import match_complex_dictionary
from mrf_diffusion.reconstruction.tensor_dictionary import TensorDictionary
from mrf_diffusion.phantoms.geometry import quadratic_object_phase_map
from mrf_diffusion.sequence.definition import (
    MRFSequence,
    SequenceSettings,
    SimulationOptions,
)
from mrf_diffusion.simulation.api import simulate_mrf_signal
from mrf_diffusion.simulation.tissue import TissueParameters

REVISION = "4a7c0eddb8be2a6ce8a36c97a01fafcf13e7665e"
BLOBS = {
    "UEE_phase.py": "d28215569e2cbe74b6a1247fdc896edf87fb9bc2",
    "Optimisation.py": "7e262600ef732f584982f61d1ea03029a6e994a2",
    "Single spiral.npz": "2f500c53153b7d46d6b438219235b6f1dfd22715",
    "LICENSE": "f288702d2fa16d3cdf0035b15a9fcbc552cd88e7",
}


def load_reference(root):
    provenance = {}
    for name, expected in BLOBS.items():
        source = (root / name).read_bytes()
        blob = hashlib.sha1(
            b"blob " + str(len(source)).encode() + b"\0" + source
        ).hexdigest()
        if blob != expected:
            raise ValueError(f"Pinned reference mismatch: {name}: {blob}")
        provenance[name] = {
            "git_blob": blob,
            "sha256": hashlib.sha256(source).hexdigest(),
        }

    def cost_class(name, methods, namespace):
        tree = ast.parse((root / name).read_bytes())
        node = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Cost"
        )
        node.body = [
            n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods
        ]
        if {n.name for n in node.body} != set(methods):
            raise ValueError("Missing external method")
        module = ast.Module(body=[node], type_ignores=[])
        exec(
            compile(ast.fix_missing_locations(module), str(root / name), "exec"),
            namespace,
        )
        return namespace["Cost"]

    signal = cost_class(
        "Optimisation.py",
        (
            "__init__",
            "Grad",
            "RF_matrix",
            "Relax_matrix",
            "signal_step",
            "signal",
            "dm_dT",
        ),
        {"np": np},
    )
    helpers = cost_class(
        "UEE_phase.py",
        (
            "Spiral_coord",
            "Spiral_dcf",
            "Coord",
            "P_single",
            "P_single_fft",
            "P_all",
            "phase_field",
            "Zero_padding",
            "Log_scaling",
            "Evaluation_data",
            "S_matrices",
            "Error_matrices",
            "Theta_1_star",
        ),
        {
            "np": np,
            "sp": sp,
            "sc": sc,
            "Optimisation": SimpleNamespace(Cost=signal),
            "tqdm": lambda x, **kwargs: x,
        },
    )
    return signal, helpers, provenance


@contextmanager
def reference_cwd(root):
    before = Path.cwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(before)


def difference(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    delta = a - b
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "relative_l2": float(
            np.linalg.norm(delta) / max(np.linalg.norm(b), np.finfo(float).tiny)
        ),
    }


def ours_signal(angles, t1, t2, *, scalar=False, settings=None):
    settings = settings or replace(
        SequenceSettings(), echo_time_ms=0.0, preparation_angles=(0.0, 0.0, 0.0)
    )
    sequence = MRFSequence(np.asarray(angles), np.zeros(len(angles)), settings)
    tissue = TissueParameters(t1, t2, 1.0, 0.0 if scalar else np.zeros((3, 3)))
    channels = np.asarray(
        simulate_mrf_signal(
            tissue,
            sequence,
            SimulationOptions(direction_count=1, state_count=32),
            tensor=not scalar,
        )
    )
    return channels[0] + 1j * channels[1]


def external_model(signal_class, angles, t1=1000.0, t2=80.0):
    state = np.zeros((3, 2), complex)
    state[2, 0] = 1
    return signal_class(
        state,
        1.0,
        len(angles),
        np.zeros(len(angles)),
        t1,
        t2,
        1.0,
        False,
        32,
        np.full(len(angles) - 1, 15.0),
        "without_TR",
        "rCRB",
    )


def dft_matrix(cost, shape, coordinates_yx):
    # Independent exact Fourier operator using EXTERNAL centered image coordinates.
    xy = cost.Coord(shape).reshape(2, -1).T
    return np.exp(-2j * np.pi * (coordinates_yx[:, ::-1] @ xy.T)) / np.sqrt(
        np.prod(shape)
    )


def uee_prediction(cost, signal_class, angles, psfs, rho, t1_map, t2_map, shape):
    """Execute the original analytical predictor stages, not a new package method."""
    instance = cost.__new__(cost)
    instance.shape = shape
    instance.shape_extended = psfs.shape[1:]
    instance.N = len(angles)
    instance.theta_0 = np.array([np.log(1000.0), np.log(80.0), 0.0, 0.0])
    instance.theta_1 = np.stack(
        (
            np.log(t1_map / 1000.0),
            np.log(t2_map / 80.0),
            np.zeros(shape),
            np.zeros(shape),
        )
    )
    instance.S = np.zeros((3, 2), complex)
    instance.S[2, 0] = 1
    instance.M0 = 1.0
    instance.sigma = 1.0
    instance.phi = np.zeros(len(angles))
    instance.clip_state = 32
    instance.TR = np.full(len(angles) - 1, 15.0)
    instance.Opt_type = "without_TR"
    instance.weighting = "rCRB"
    instance.W_T1 = instance.W_T2 = instance.W_M0 = 0.0
    instance.P_j_tot = psfs.mean(axis=0)
    instance.P_j_resid = psfs - instance.P_j_tot
    instance.rho = rho
    instance.rho_0_star = sc.signal.convolve(rho, instance.P_j_tot, mode="same")
    instance.PSF_err = True
    signal, derivatives = instance.Evaluation_data(angles)
    normal, s10, s11 = instance.S_matrices(signal, derivatives)
    e1, e2 = instance.Error_matrices(s10, s11)
    theta, _, _, _ = instance.Theta_1_star(e1, e2, normal)
    return {
        "t1_ms": np.exp(instance.theta_0[0] + theta[0]),
        "t2_ms": np.exp(instance.theta_0[1] + theta[1]),
        "density": np.abs(instance.rho_0_star * (1 + theta[2] + 1j * theta[3])),
    }, {
        "real_normal_condition": float(np.linalg.cond(normal.real)),
        "maximum_input_log_contrast": float(np.max(np.abs(instance.theta_1[:2]))),
        "maximum_predicted_log_correction": float(np.max(np.abs(theta[:2]))),
        "minimum_blurred_density_magnitude": float(np.min(np.abs(instance.rho_0_star))),
    }


def compare(root, output):
    signal_class, cost, provenance = load_reference(root)
    output.mkdir(parents=True, exist_ok=False)
    shape = (9, 9)
    extended = (17, 17)
    frames = 12
    offset = 0.17
    angles = 0.15 + 0.45 * np.sin(np.linspace(0, np.pi, frames)) ** 2
    base = load_spiral_coordinates(root / "Single spiral.npz")
    raw = np.load(root / "Single spiral.npz")["Coords"]
    results = {}
    checks = {}
    artifacts = {}

    def record(
        name,
        actual,
        expected,
        tolerance=None,
        classification="numerical agreement",
        *,
        required=True,
    ):
        measured = difference(actual, expected)
        measured["classification"] = classification
        if tolerance is not None:
            measured["max_abs_tolerance"] = tolerance
            measured["passed"] = measured["max_abs"] <= tolerance
            measured["required_for_common_case"] = required
            if required:
                checks[name] = measured["passed"]
        results[name] = measured

    record("loaded_spiral", base, raw.T[:, ::-1], 0.0)
    for schedule in ("golden", "reference_arms"):
        for arms in (1, 3, 4):
            own = generate_rotated_spiral_trajectory(
                base,
                frames,
                interleaves_per_frame=arms,
                schedule=schedule,
                initial_angle_rad=offset,
            )
            external = []
            with reference_cwd(root):
                for frame in range(frames):
                    xy, mask, length = cost.Spiral_coord(
                        frame,
                        "2-pi",
                        shape,
                        schedule == "golden",
                        "Philips_spiral",
                        offset,
                        arms,
                    )
                    if not np.all(mask):
                        raise ValueError("Common case cannot drop clipped samples")
                    external.append(
                        (xy[:, ::-1] / (2 * np.pi)).reshape(arms, length, 2)
                    )
            record(f"rotation_{schedule}_L{arms}", own, external, 2e-15)
    weights = radial_increment_density_compensation(base)
    record("dcf_cycles_per_pixel", weights, cost.Spiral_dcf(raw.T, True)[:, 0], 1e-15)
    record(
        "dcf_radian_conversion",
        weights,
        cost.Spiral_dcf(raw.T * 2 * np.pi, True)[:, 0] / (2 * np.pi) ** 2,
        1e-15,
        "coordinate-unit scaling: DCF is quadratic in coordinate units",
    )
    object_phase = quadratic_object_phase_map(shape, edge_phase_rad=2 * np.pi * 0.1)
    record(
        "object_phase", np.exp(1j * object_phase), cost.phase_field(shape, 0.1), 1e-15
    )
    # Asymmetric complex image exposes transposes/phase loss.
    image = np.zeros(shape, complex)
    image[1:4, 3:7] = 0.7
    image[6, 2] = 1.2 - 0.4j
    image *= np.exp(1j * object_phase)
    coordinates = generate_rotated_spiral_trajectory(
        base, 1, interleaves_per_frame=3, schedule="golden", initial_angle_rad=offset
    )[0]
    flat = coordinates.reshape(-1, 2)
    weight = radial_increment_density_compensation(coordinates).reshape(-1)
    matrix = dft_matrix(cost, shape, flat)
    exact_kspace = matrix @ image.ravel()
    reference_op = sp.linop.NUFFT(shape, flat[:, ::-1] * shape[0])
    # Reference's xy coordinates address SigPy array axes in xy order.
    reference_kspace = reference_op * image.T
    default = NufftOperator(shape, flat)
    tight = NufftOperator(shape, flat, 2.0, 6.0)
    record(
        "forward_same_sigpy_converted_axes",
        default.forward(image),
        reference_kspace,
        1e-12,
        "xy/yx input transpose is required",
    )
    record(
        "forward_without_axis_conversion",
        default.forward(image),
        reference_op * image,
        None,
        "coordinate convention: unconverted xy/yx",
    )
    record(
        "forward_default_vs_exact_DFT",
        default.forward(image),
        exact_kspace,
        5e-3,
        "NUFFT interpolation",
        required=False,
    )
    record(
        "forward_tight_vs_exact_DFT",
        tight.forward(image),
        exact_kspace,
        5e-5,
        "NUFFT interpolation",
    )
    # Divide by operator's ideal unit-impulse diagonal only for error reporting.
    gain = weight.sum() / np.prod(shape)
    exact_adjoint = (matrix.conj().T @ (weight * exact_kspace)).reshape(shape)
    reference_adjoint = (reference_op.H * (weight * exact_kspace)).T
    record(
        "adjoint_same_sigpy_converted_axes",
        default.adjoint(weight * exact_kspace),
        reference_adjoint,
        1e-14,
    )
    record(
        "adjoint_tight_vs_exact_DFT_unit_gain",
        tight.adjoint(weight * exact_kspace) / gain,
        exact_adjoint / gain,
        5e-5,
        "NUFFT interpolation",
    )
    record(
        "complex64_vs_complex128_kspace",
        tight.forward(image.astype(np.complex64)),
        tight.forward(image),
        2e-6,
        "precision",
    )
    psf = direct_sampling_psf(flat, shape, weight, displacement_shape=extended)
    external_direct = cost.P_single(
        cost.Coord(extended), shape, flat[:, ::-1] * 2 * np.pi, weight[:, None]
    )
    record("psf_direct_extended", psf, external_direct, 1e-14)
    # P_single_fft returns A_ext^H w; ideal PSF uses 1/V, not 1/sqrt(V_ext).
    raw_fft = cost.P_single_fft(extended, flat[:, ::-1] * extended[0], weight)
    converted_fft = raw_fft.T * np.sqrt(np.prod(extended)) / np.prod(shape)
    record(
        "psf_fft_reference_vs_direct",
        converted_fft / gain,
        psf / gain,
        8e-3,
        "NUFFT interpolation after transpose and sqrt(V_ext)/V scaling",
    )
    record(
        "psf_fft_raw_vs_direct",
        raw_fft / gain,
        psf / gain,
        None,
        "different normalization and xy/yx convention",
    )
    record(
        "psf_normal_operator_tight",
        calculate_point_spread_function(tight, weight) / gain,
        direct_sampling_psf(flat, shape, weight) / gain,
        5e-5,
        "actual A^HWA impulse versus exact Fourier kernel",
    )
    ref = external_model(signal_class, angles).signal(angles)
    own = ours_signal(angles, 1000.0, 80.0)
    record("signal_tensor_zero_diffusion", own, ref, 1e-12)
    record(
        "signal_scalar_float32",
        ours_signal(angles, 1000.0, 80.0, scalar=True),
        ref,
        1e-6,
        "existing scalar RF float32 cast",
    )
    rounded = angles.astype(np.float32).astype(np.float64)
    record(
        "signal_scalar_matched_float32_angles",
        ours_signal(angles, 1000.0, 80.0, scalar=True),
        external_model(signal_class, rounded).signal(rounded),
        1e-12,
        "input rounding alone does not align float32 RF trigonometry with float64",
        required=False,
    )
    record(
        "signal_normal_sequence_vs_external",
        ours_signal(angles, 1000.0, 80.0, settings=SequenceSettings()),
        ref,
        None,
        "expected sequence difference: preparation and TE=4 ms",
    )
    record("signal_proton_density_scaling", 0.8 * own, 0.8 * ref, 1e-12)
    # Separate RF arithmetic from input quantization, using existing RF operators.
    import jax.numpy as jnp
    from mrf_diffusion.epg.rf import apply_rf_rotation

    model = external_model(signal_class, angles)
    rf_reference = model.RF_matrix(float(rounded[4]), 0.0)
    record(
        "RF_float64",
        apply_rf_rotation(
            jnp.eye(3, dtype=jnp.complex128), jnp.float64(rounded[4]), jnp.float64(0.0)
        ),
        rf_reference,
        1e-12,
    )
    record(
        "RF_float32",
        apply_rf_rotation(
            jnp.eye(3, dtype=jnp.complex128), jnp.float32(rounded[4]), jnp.float32(0.0)
        ),
        rf_reference,
        1e-6,
        "float32 RF arithmetic",
    )
    d1, d2, _ = model.dm_dT(angles)
    for label, h, lo, hi, derivative in (
        (
            "T1",
            0.001,
            ours_signal(angles, 999.999, 80.0),
            ours_signal(angles, 1000.001, 80.0),
            d1,
        ),
        (
            "T2",
            0.0001,
            ours_signal(angles, 1000.0, 79.9999),
            ours_signal(angles, 1000.0, 80.0001),
            d2,
        ),
    ):
        record(
            "external_derivative_vs_our_FD_" + label,
            derivative[0] + 1j * derivative[1],
            (hi - lo) / (2 * h),
            1e-9,
            "hand derivatives checked against our finite differences",
        )
    artifacts.update(
        simple_image=image,
        reference_signal=ref,
        our_signal=own,
        angles_rad=angles,
        base_yx=base,
        weights=weights,
        exact_kspace=exact_kspace,
        our_kspace=tight.forward(image),
        exact_adjoint=exact_adjoint,
        our_adjoint=tight.adjoint(weight * exact_kspace),
    )
    # Four independent T1/T2 entries; no tensor/FA recovery is claimed.
    parameters = np.array(
        [[900.0, 70.0], [900.0, 90.0], [1100.0, 70.0], [1100.0, 90.0]]
    )
    ours = np.stack([ours_signal(angles, *p) for p in parameters])
    theirs = np.stack(
        [external_model(signal_class, angles, *p).signal(angles) for p in parameters]
    )
    record("common_tissue_dictionary", ours, theirs, 1e-12)
    dictionary = TensorDictionary(
        ours, parameters[:, 0], parameters[:, 1], np.zeros((4, 3, 3))
    )
    rows, cols = np.indices(shape)
    labels = (rows // 3 + cols // 2) % 2
    atom_index = np.where(labels == 0, 0, 3)
    density = np.where(labels == 0, 1.0, 0.8)
    images = (
        ours[atom_index].transpose(2, 0, 1)
        * density[None]
        * np.exp(1j * object_phase)[None]
    )
    external_images = (
        theirs[atom_index].transpose(2, 0, 1)
        * density[None]
        * np.exp(1j * object_phase)[None]
    )
    record("complex_phantom_series", images, external_images, 1e-12)
    support = np.ones(shape, bool)
    summaries = []
    raw_psfs = None
    for sampling, arms in (("cartesian", 1), ("spiral", 1), ("spiral", 4)):
        if sampling == "cartesian":
            coords = np.broadcast_to(
                cartesian_trajectory(shape), (frames, 1, np.prod(shape), 2)
            ).copy()
        else:
            coords = generate_rotated_spiral_trajectory(
                base,
                frames,
                interleaves_per_frame=arms,
                schedule="golden",
                initial_angle_rad=offset,
            )
        acquired = acquire_image_series(
            images, coords, oversampling=2.0, kernel_width=6.0
        )
        actual = reconstruct_acquisition_frames(
            acquired,
            density_compensation=(
                "none" if sampling == "cartesian" else "radial_increment"
            ),
        )
        oracle_frames = []
        kernels = []
        oracle_k = []
        for frame in range(frames):
            k = coords[frame].reshape(-1, 2)
            w = (
                np.ones(len(k))
                if sampling == "cartesian"
                else radial_increment_density_compensation(coords[frame]).reshape(-1)
            )
            mat = dft_matrix(cost, shape, k)
            values = mat @ external_images[frame].ravel()
            oracle_k.append(values.reshape(arms, -1))
            oracle_frames.append((mat.conj().T @ (w * values)).reshape(shape))
            if sampling == "spiral":
                kernel = cost.P_single(
                    cost.Coord(extended), shape, k[:, ::-1] * 2 * np.pi, w[:, None]
                )
                convolved = sc.signal.convolve(
                    external_images[frame], kernel, mode="same"
                )
                record(
                    f"linear_convolution_{arms}_{frame}",
                    convolved,
                    oracle_frames[-1],
                    1e-13,
                )
                kernels.append(kernel)
        expected = np.asarray(oracle_frames)
        divisor = (
            1.0
            if sampling == "cartesian"
            else radial_increment_density_compensation(coords[0]).sum() / np.prod(shape)
        )
        record(
            f"series_{sampling}_L{arms}_kspace",
            acquired.kspace,
            oracle_k,
            5e-5,
            "independent DFT and external compatible signals",
        )
        record(
            f"series_{sampling}_L{arms}_frames_unit_gain",
            actual / divisor,
            expected / divisor,
            8e-5,
            "NUFFT interpolation versus external direct-PSF convolution",
        )
        fitted = match_complex_dictionary(actual, dictionary, support)
        expected_fit = match_complex_dictionary(expected, dictionary, support)
        mismatches = int(
            np.count_nonzero(fitted.dictionary_index != expected_fit.dictionary_index)
        )
        checks[f"matching_indices_{sampling}_{arms}"] = mismatches == 0
        summaries.append(
            {
                "sampling": sampling,
                "interleaves": arms,
                "index_disagreements": mismatches,
                "t1_rmse_ms": float(
                    np.sqrt(
                        np.mean((fitted.maps["t1_ms"] - parameters[atom_index, 0]) ** 2)
                    )
                ),
                "t2_rmse_ms": float(
                    np.sqrt(
                        np.mean((fitted.maps["t2_ms"] - parameters[atom_index, 1]) ** 2)
                    )
                ),
                "density_nrmse_raw_adjoint": float(
                    np.linalg.norm(fitted.maps["proton_density"] - density)
                    / np.linalg.norm(density)
                ),
            }
        )
        artifacts[f"{sampling}_{arms}_our_frames"] = actual
        artifacts[f"{sampling}_{arms}_reference_frames"] = expected
        if sampling == "spiral" and arms == 1:
            raw_psfs = np.asarray(kernels)
    # Native external P_all branches include their own global PSF sum normalization.
    with reference_cwd(root):
        native_explicit, _ = cost.P_all(
            cost.Coord(extended),
            frames,
            shape,
            extended,
            True,
            True,
            "Philips_spiral",
            offset,
            1,
            "Explicit",
            True,
            "unused",
            False,
            False,
        )
        native_fft, _ = cost.P_all(
            cost.Coord(extended),
            frames,
            shape,
            extended,
            True,
            True,
            "Philips_spiral",
            offset,
            1,
            "FFT",
            True,
            "unused",
            False,
            False,
        )
    normalized = raw_psfs / np.abs(raw_psfs.mean(axis=0).sum())
    record(
        "native_P_all_explicit_normalized",
        normalized,
        native_explicit,
        1e-12,
        "unit factors cancel under David's global normalization",
    )
    record(
        "native_P_all_fft_unconverted",
        native_fft,
        native_explicit,
        None,
        "external xy/yx convention inconsistency",
    )
    record(
        "native_P_all_fft_transposed",
        native_fft.transpose(0, 2, 1),
        native_explicit,
        1e-3,
        "remaining default NUFFT interpolation; no source change",
        required=False,
    )
    # Keep external methods unchanged; explicitly inject a tighter SigPy factory
    # only in their isolated validation namespace to diagnose interpolation error.
    namespace = cost.P_single_fft.__globals__
    original_backend = namespace["sp"]
    namespace["sp"] = SimpleNamespace(
        linop=SimpleNamespace(
            NUFFT=lambda shape, coords: sp.linop.NUFFT(
                shape, coords, oversamp=2.0, width=6.0
            )
        )
    )
    try:
        with reference_cwd(root):
            tighter_fft, _ = cost.P_all(
                cost.Coord(extended),
                frames,
                shape,
                extended,
                True,
                True,
                "Philips_spiral",
                offset,
                1,
                "FFT",
                True,
                "unused",
                False,
                False,
            )
    finally:
        namespace["sp"] = original_backend
    record(
        "reference_P_all_tighter_backend_transposed",
        tighter_fft.transpose(0, 2, 1),
        native_explicit,
        1e-5,
        "validation-only SigPy 2/6 backend injection; external source unchanged",
    )
    checks["NUFFT_precision_convergence"] = (
        results["forward_tight_vs_exact_DFT"]["relative_l2"]
        < results["forward_default_vs_exact_DFT"]["relative_l2"] / 100
    )
    checks["reference_PSF_precision_convergence"] = (
        results["reference_P_all_tighter_backend_transposed"]["relative_l2"]
        < results["native_P_all_fft_transposed"]["relative_l2"] / 100
    )
    artifacts.update(
        raw_psfs=raw_psfs,
        external_native_explicit_psfs=native_explicit,
        external_native_fft_psfs=native_fft,
    )
    # Exact common limit of the external final analytical predictor: identical
    # trajectories and homogeneous tissue => only density/object phase is blurred.
    stationary = np.repeat(normalized[:1], frames, axis=0)
    # Normalize this stationary PSF separately, as P_all would for constant sampling.
    stationary = stationary / np.abs(stationary[0].sum())
    rho = density * np.exp(1j * object_phase)
    predicted, conditioning = uee_prediction(
        cost,
        signal_class,
        angles,
        stationary,
        rho,
        np.full(shape, 1000.0),
        np.full(shape, 80.0),
        shape,
    )
    blurred = sc.signal.convolve(rho, stationary[0], mode="same")
    homogeneous_images = own[:, None, None] * blurred[None]
    hom_dictionary = TensorDictionary(
        own[None], np.array([1000.0]), np.array([80.0]), np.zeros((1, 3, 3))
    )
    matched = match_complex_dictionary(homogeneous_images, hom_dictionary, support)
    record(
        "UEE_stationary_homogeneous_T1", predicted["t1_ms"], matched.maps["t1_ms"], 1e-8
    )
    record(
        "UEE_stationary_homogeneous_T2", predicted["t2_ms"], matched.maps["t2_ms"], 1e-8
    )
    record(
        "UEE_stationary_homogeneous_density",
        predicted["density"],
        matched.maps["proton_density"],
        1e-10,
    )
    # Nonlinear dictionary maps and UEE are not identical estimators. Measure,
    # but do NOT use agreement as a pass criterion for this approximation case.
    predicted_rot, cond_rot = uee_prediction(
        cost,
        signal_class,
        angles,
        normalized,
        rho,
        parameters[atom_index, 0],
        parameters[atom_index, 1],
        shape,
    )
    normalized_images = artifacts["spiral_1_reference_frames"] / np.abs(
        raw_psfs.mean(axis=0).sum()
    )
    nonlinear = match_complex_dictionary(normalized_images, dictionary, support)
    record(
        "UEE_rotating_vs_dictionary_T1",
        predicted_rot["t1_ms"],
        nonlinear.maps["t1_ms"],
        None,
        "expected implementation difference: local analytical approximation versus discrete nonlinear matching",
    )
    record(
        "UEE_rotating_vs_dictionary_T2",
        predicted_rot["t2_ms"],
        nonlinear.maps["t2_ms"],
        None,
        "expected implementation difference: local analytical approximation versus discrete nonlinear matching",
    )
    for key, value in predicted_rot.items():
        artifacts["UEE_rotating_" + key] = value
    record(
        "padding_odd_to_extended",
        cost.Zero_padding(image, extended),
        np.pad(image, ((4, 4), (4, 4))),
        0.0,
    )
    # Negative controls expose centering and magnitude loss rather than concealing them.
    record(
        "forward_uncentered_image",
        tight.forward(np.fft.ifftshift(image)),
        exact_kspace,
        None,
        "FFT centering",
    )
    record(
        "forward_magnitude_only_image",
        tight.forward(np.abs(image)),
        exact_kspace,
        None,
        "complex phase discarded",
    )
    report = {
        "reference_repository": "https://github.com/imphys/MRF_undersampling_optimization",
        "revision": REVISION,
        "license": "GPL-3.0",
        "external_files": provenance,
        "versions": {
            p: version(p) for p in ("numpy", "scipy", "jax", "jaxlib", "sigpy", "numba")
        },
        "configuration": {
            "image_shape": shape,
            "displacement_shape": extended,
            "frames": frames,
            "flip_angles_rad": angles.tolist(),
            "rf_phase_rad": 0.0,
            "TR_ms": 15.0,
            "TE_ms": 0.0,
            "preparation_angles_rad": [0.0, 0.0, 0.0],
            "diffusion_tensor": np.zeros((3, 3)).tolist(),
            "state_count": 32,
            "object_phase_edge_rad": float(2 * np.pi * 0.1),
            "noise": 0.0,
            "offset_rad": offset,
            "trajectory_samples_per_arm": len(base),
            "tight_nufft": {"oversampling": 2.0, "width": 6.0},
            "default_nufft": {"oversampling": 1.25, "width": 4.0},
        },
        "comparisons": results,
        "checks": checks,
        "common_case_passed": all(checks.values()),
        "final_direct_comparisons": summaries,
        "external_UEE_stationary_diagnostics": conditioning,
        "external_UEE_rotating_diagnostics": cond_rot,
        "exploratory_precision_failures": [
            name
            for name, value in results.items()
            if value.get("passed") is False
            and not value.get("required_for_common_case", True)
        ],
        "scope": "Acquisition and zero-diffusion TE=0 common-case validation only; not full diffusion/FA/optimized-phase validation.",
    }
    np.savez(output / "intermediates.npz", **artifacts)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    fixture = {
        "revision": REVISION,
        "external_files": provenance,
        "angles_rad": angles.tolist(),
        "external_signal_real": ref.real.tolist(),
        "external_signal_imag": ref.imag.tolist(),
        "t1_ms": 1000.0,
        "t2_ms": 80.0,
        "TR_ms": 15.0,
        "TE_ms": 0.0,
    }
    (output / "signal_fixture_candidate.json").write_text(
        json.dumps(fixture, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": all(checks.values()),
                "failed": [k for k, v in checks.items() if not v],
                "comparisons": results,
                "final_direct_comparisons": summaries,
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_directory", type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()
    report = compare(
        args.reference_directory.resolve(), args.output_directory.resolve()
    )
    if not report["common_case_passed"]:
        raise SystemExit(1)

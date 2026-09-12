from mrf_diffusion.optimization import evaluation as evaluation_module

"""Tests use real/imaginary iid noise and scalar ordering [T1,T2,D,M]."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
import mrf_diffusion.api as epg
from tests.cases import (
    ANGLES,
    PHASES,
    PARAMETERS,
    TENSOR,
    PREP,
    POINTS,
    KNOTS,
    COEFFICIENTS,
    central_difference,
    equilibrium,
)


def test_rf_derivatives_against_finite_differences():

    def response(p):
        value = epg.apply_rf_rotation(equilibrium(), p[0], p[1])[0, 0]
        return jnp.array([value.real, value.imag])

    p = jnp.array([0.47, -0.23])
    actual = jax.jacobian(response)(p).T
    expected = central_difference(response, p, [1e-05, 1e-05])
    assert np.isfinite(actual).all()
    assert_allclose(actual, expected, rtol=1e-08, atol=1e-10)


def test_fa_autodiff_against_symmetric_tensor_perturbation():
    direction = jnp.array([[0.2, 0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.0, 0.3]])
    step = 1e-08
    expected = (
        epg.fractional_anisotropy(TENSOR + step * direction)
        - epg.fractional_anisotropy(TENSOR - step * direction)
    ) / (2 * step)
    actual = jnp.sum(jax.grad(epg.fractional_anisotropy)(TENSOR) * direction)
    assert_allclose(actual, expected, rtol=1e-07, atol=1e-07)


def test_scalar_jacobian_dimensions_and_parameter_order():
    actual = epg.scalar_signal_jacobian(ANGLES, PHASES, *PARAMETERS)
    assert actual.shape == (4, 2, len(ANGLES))
    expected = central_difference(
        lambda p: epg.simulate_scalar_signal(ANGLES, PHASES, *p),
        PARAMETERS,
        [0.1, 0.01, 1e-07, 0.0001],
    )
    assert np.isfinite(actual).all()
    assert_allclose(actual, expected, rtol=2e-05, atol=2e-08)


def test_tensor_signal_jacobian_scalar_columns_and_jit():

    def response(p):
        return epg.simulate_tensor_signal(
            ANGLES[:3],
            PHASES[:3],
            *p,
            TENSOR,
            PREP,
            PREP,
            direction_count=1,
            state_count=5,
        )

    p = jnp.array([1000.0, 80.0, 0.9])
    actual = jax.jacobian(response)(p).transpose(2, 0, 1)
    expected = central_difference(response, p, [0.1, 0.01, 0.0001])
    assert_allclose(actual, expected, rtol=2e-05, atol=2e-08)
    compiled = response(p)
    with jax.disable_jit():
        assert_allclose(response(p), compiled, atol=1e-12)


def test_general_jacobian_shape_and_scalar_parameter_order():
    actual = epg.legacy_tensor_metric_signal_jacobian(
        ANGLES[:3],
        PHASES[:3],
        1000.0,
        80.0,
        0.9,
        TENSOR,
        PREP,
        PREP,
        direction_count=1,
        state_count=5,
    )
    assert actual.shape == (5, 2, 3)
    expected = central_difference(
        lambda p: epg.simulate_tensor_signal(
            ANGLES[:3],
            PHASES[:3],
            *p,
            TENSOR,
            PREP,
            PREP,
            direction_count=1,
            state_count=5,
        ),
        [1000.0, 80.0, 0.9],
        [0.1, 0.01, 0.0001],
    )
    assert_allclose(actual[:3], expected, rtol=2e-05, atol=2e-08)


def test_fim_matches_finite_difference_information_and_noise_scaling():
    sigma = 0.03
    actual = epg.scalar_fisher_information(ANGLES, PHASES, *PARAMETERS, sigma=sigma)
    derivative = central_difference(
        lambda p: epg.simulate_scalar_signal(ANGLES, PHASES, *p),
        PARAMETERS,
        [0.1, 0.01, 1e-07, 0.0001],
    ).reshape(4, -1)
    expected = derivative @ derivative.T / sigma**2
    assert_allclose(actual, expected, rtol=3e-05, atol=1e-08)
    assert_allclose(actual, actual.T, atol=1e-10)
    scaled = np.asarray(actual) * np.outer(PARAMETERS, PARAMETERS)
    assert np.linalg.eigvalsh(scaled).min() > 0
    assert np.linalg.cond(scaled) < 1000000000.0
    assert_allclose(
        epg.scalar_fisher_information(ANGLES, PHASES, *PARAMETERS, sigma=2 * sigma),
        actual / 4,
        rtol=1e-12,
    )


def test_crlb_matches_svd_inverse_for_one_tissue():
    angles = epg.evaluate_bspline(POINTS, KNOTS, COEFFICIENTS)
    fim = np.asarray(epg.scalar_fisher_information(angles, PHASES, *PARAMETERS))
    scale = np.diag(np.asarray(PARAMETERS))
    normalized_fim = scale @ fim @ scale
    u, singular, vh = np.linalg.svd(normalized_fim)
    assert singular[-1] > singular[0] * 1e-09
    covariance = vh.T / singular @ u.T
    expected = np.sqrt(np.diag(covariance))
    actual = epg.scalar_sequence_precision_bounds(
        COEFFICIENTS,
        PHASES,
        POINTS,
        KNOTS,
        PARAMETERS[None, :],
        jnp.ones(4),
        jnp.ones(1),
    )
    assert_allclose(actual, expected, rtol=2e-06, atol=1e-08)


def test_single_complex_sample_cannot_identify_four_parameters():
    fim = np.asarray(epg.scalar_fisher_information(ANGLES[:1], PHASES[:1], *PARAMETERS))
    normalized = fim * np.outer(PARAMETERS, PARAMETERS)
    singular = np.linalg.svd(normalized, compute_uv=False)
    assert np.linalg.matrix_rank(normalized, tol=singular[0] * 1e-12) <= 2
    assert singular[-1] < singular[0] * 1e-12


def test_crlb_analytical_diagonal_information(monkeypatch):
    fim = jnp.diag(jnp.array([1.0, 4.0, 9.0, 16.0]) / PARAMETERS**2)
    monkeypatch.setattr(
        evaluation_module, "scalar_fisher_information", lambda *args, **kwargs: fim
    )
    with jax.disable_jit():
        actual = epg.scalar_sequence_precision_bounds(
            COEFFICIENTS,
            PHASES,
            POINTS,
            KNOTS,
            PARAMETERS[None],
            jnp.ones(4),
            jnp.ones(1),
        )
    assert_allclose(actual, [1.0, 0.5, 1 / 3, 0.25], atol=1e-14)


def test_crlb_exact_singular_information_is_not_finite_precision(monkeypatch):
    fim = jnp.diag(jnp.array([1.0, 4.0, 9.0, 0.0]))
    monkeypatch.setattr(
        evaluation_module, "scalar_fisher_information", lambda *args, **kwargs: fim
    )
    with jax.disable_jit():
        actual = epg.scalar_sequence_precision_bounds(
            COEFFICIENTS,
            PHASES,
            POINTS,
            KNOTS,
            PARAMETERS[None],
            jnp.ones(4),
            jnp.ones(1),
        )
    assert not np.isfinite(actual).all()


@pytest.mark.discrepancy
@pytest.mark.xfail(
    reason="I01: scalar bound scan applies first tissue weight to every tissue",
    raises=AssertionError,
)
def test_crlb_nonuniform_tissue_weights():
    tissues = jnp.stack([PARAMETERS, PARAMETERS.at[0].set(1400.0)])

    def bounds(p, weights):
        return epg.scalar_sequence_precision_bounds(
            COEFFICIENTS, PHASES, POINTS, KNOTS, p, jnp.ones(4), weights
        )

    expected = 0.2 * bounds(tissues[:1], jnp.ones(1)) + 0.8 * bounds(
        tissues[1:], jnp.ones(1)
    )
    assert_allclose(bounds(tissues, jnp.array([0.2, 0.8])), expected, rtol=1e-10)


@pytest.mark.discrepancy
@pytest.mark.xfail(
    reason="I02: generalized bounds omit FA and MD", raises=AssertionError
)
def test_general_bounds_include_all_five_parameters():
    actual = epg.tensor_sequence_precision_bounds(
        COEFFICIENTS,
        POINTS,
        KNOTS,
        jnp.array([[1000.0, 80.0, 0.9]]),
        TENSOR[None],
        jnp.ones(5),
        jnp.ones(1),
        0.0,
        0.0,
        PREP,
        PREP,
        direction_count=1,
        state_count=5,
        method="no phase modulation",
    )
    assert actual.shape == (5,)


def test_scalar_signal_jit_equals_eager_and_magnitude():
    compiled = epg.simulate_scalar_signal(ANGLES, PHASES, *PARAMETERS)
    with jax.disable_jit():
        eager = epg.simulate_scalar_signal(ANGLES, PHASES, *PARAMETERS)
    assert_allclose(compiled, eager, rtol=3e-07, atol=1e-09)
    assert_allclose(
        epg.scalar_signal_magnitude(ANGLES, PHASES, *PARAMETERS),
        np.abs(np.asarray(compiled)[0] + 1j * np.asarray(compiled)[1]),
        atol=1e-14,
    )

"""Tests use real/imaginary iid noise and scalar ordering [T1,T2,D,M]."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
import EPG_blocks_jaxcode as epg
from tests.cases import (ANGLES, PHASES, PARAMETERS, TENSOR, PREP, POINTS,
                         KNOTS, COEFFICIENTS, central_difference, equilibrium)


def test_rf_derivatives_against_finite_differences():
    def response(p):
        value = epg.epg_rf(equilibrium(), p[0], p[1])[0, 0]
        return jnp.array([value.real, value.imag])
    p = jnp.array([0.47, -0.23])
    actual = jax.jacobian(response)(p).T
    expected = central_difference(response, p, [1e-5, 1e-5])
    assert np.isfinite(actual).all()
    assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)


def test_fa_autodiff_against_symmetric_tensor_perturbation():
    direction = jnp.array([[0.2, 0.1, 0.], [0.1, -0.1, 0.], [0., 0., 0.3]])
    step = 1e-8
    expected = (epg.fractional_anisotropy(TENSOR+step*direction)-
                epg.fractional_anisotropy(TENSOR-step*direction))/(2*step)
    actual = jnp.sum(jax.grad(epg.fractional_anisotropy)(TENSOR)*direction)
    assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)


def test_scalar_jacobian_dimensions_and_parameter_order():
    actual = epg.jacobian_calculator(ANGLES, PHASES, *PARAMETERS)
    assert actual.shape == (4, 2, len(ANGLES))
    expected = central_difference(lambda p: epg.output_generator(ANGLES, PHASES, *p),
                                  PARAMETERS, [0.1, 0.01, 1e-7, 1e-4])
    assert np.isfinite(actual).all()
    assert_allclose(actual, expected, rtol=2e-5, atol=2e-8)


def test_tensor_signal_jacobian_scalar_columns_and_jit():
    def response(p):
        return epg.output_generator_general(ANGLES[:3], PHASES[:3], *p, TENSOR,
                                             PREP, PREP, gradient_number=1, n_states=5)
    p = jnp.array([1000., 80., 0.9])
    actual = jax.jacobian(response)(p).transpose(2, 0, 1)
    expected = central_difference(response, p, [0.1, 0.01, 1e-4])
    assert_allclose(actual, expected, rtol=2e-5, atol=2e-8)
    compiled = response(p)
    with jax.disable_jit():
        assert_allclose(response(p), compiled, atol=1e-12)


def test_general_jacobian_shape_and_scalar_parameter_order():
    actual = epg.jacobian_calculator_general(ANGLES[:3], PHASES[:3], 1000., 80., .9,
                                            TENSOR, PREP, PREP, gradient_number=1, n_states=5)
    assert actual.shape == (5, 2, 3)
    expected = central_difference(
        lambda p: epg.output_generator_general(ANGLES[:3], PHASES[:3], *p, TENSOR,
                                                PREP, PREP, gradient_number=1, n_states=5),
        [1000., 80., .9], [.1, .01, 1e-4])
    assert_allclose(actual[:3], expected, rtol=2e-5, atol=2e-8)
    # FA/MD columns are legacy coordinates, not validated physical derivatives.


def test_fim_matches_finite_difference_information_and_noise_scaling():
    sigma = .03
    actual = epg.fisher_information(ANGLES, PHASES, *PARAMETERS, sigma=sigma)
    derivative = central_difference(lambda p: epg.output_generator(ANGLES, PHASES, *p),
                                     PARAMETERS, [.1, .01, 1e-7, 1e-4]).reshape(4, -1)
    expected = derivative@derivative.T/sigma**2
    assert_allclose(actual, expected, rtol=3e-5, atol=1e-8)
    assert_allclose(actual, actual.T, atol=1e-10)
    # Normalize coordinates before inspecting conditioning across disparate units.
    scaled = np.asarray(actual)*np.outer(PARAMETERS, PARAMETERS)
    assert np.linalg.eigvalsh(scaled).min() > 0
    assert np.linalg.cond(scaled) < 1e9
    assert_allclose(epg.fisher_information(ANGLES, PHASES, *PARAMETERS, sigma=2*sigma), actual/4, rtol=1e-12)


def test_crlb_matches_svd_inverse_for_one_tissue():
    angles = epg.bspline(POINTS, KNOTS, COEFFICIENTS)
    fim = np.asarray(epg.fisher_information(angles, PHASES, *PARAMETERS))
    scale = np.diag(np.asarray(PARAMETERS))
    normalized_fim = scale@fim@scale
    u, singular, vh = np.linalg.svd(normalized_fim)
    assert singular[-1] > singular[0]*1e-9
    covariance = (vh.T/singular)@u.T
    expected = np.sqrt(np.diag(covariance))
    actual = epg.lb_in_param_holistic(COEFFICIENTS, PHASES, POINTS, KNOTS,
                                     PARAMETERS[None, :], jnp.ones(4), jnp.ones(1))
    assert_allclose(actual, expected, rtol=2e-6, atol=1e-8)


def test_single_complex_sample_cannot_identify_four_parameters():
    # Two real observations imply rank <=2 regardless of preparation transients.
    fim = np.asarray(epg.fisher_information(ANGLES[:1], PHASES[:1], *PARAMETERS))
    normalized = fim*np.outer(PARAMETERS, PARAMETERS)
    singular = np.linalg.svd(normalized, compute_uv=False)
    assert np.linalg.matrix_rank(normalized, tol=singular[0]*1e-12) <= 2
    assert singular[-1] < singular[0]*1e-12
    # A finite result from inv() would not make this an identifiable experiment.


def test_crlb_analytical_diagonal_information(monkeypatch):
    # Inject only the information boundary to isolate the real repository CRLB
    # calculation from signal physics; no simulation implementation is replaced.
    fim = jnp.diag(jnp.array([1., 4., 9., 16.])/PARAMETERS**2)
    monkeypatch.setattr(epg, "fisher_information", lambda *args, **kwargs: fim)
    with jax.disable_jit():
        actual = epg.lb_in_param_holistic(COEFFICIENTS, PHASES, POINTS, KNOTS,
                                         PARAMETERS[None], jnp.ones(4), jnp.ones(1))
    assert_allclose(actual, [1., .5, 1/3, .25], atol=1e-14)


def test_crlb_exact_singular_information_is_not_finite_precision(monkeypatch):
    fim = jnp.diag(jnp.array([1., 4., 9., 0.]))
    monkeypatch.setattr(epg, "fisher_information", lambda *args, **kwargs: fim)
    with jax.disable_jit():
        actual = epg.lb_in_param_holistic(COEFFICIENTS, PHASES, POINTS, KNOTS,
                                         PARAMETERS[None], jnp.ones(4), jnp.ones(1))
    assert not np.isfinite(actual).all()


@pytest.mark.discrepancy
@pytest.mark.xfail(reason="I01: scalar bound scan applies first tissue weight to every tissue", raises=AssertionError)
def test_crlb_nonuniform_tissue_weights():
    tissues = jnp.stack([PARAMETERS, PARAMETERS.at[0].set(1400.)])
    def bounds(p, weights):
        return epg.lb_in_param_holistic(COEFFICIENTS, PHASES, POINTS, KNOTS, p, jnp.ones(4), weights)
    expected = .2*bounds(tissues[:1], jnp.ones(1))+.8*bounds(tissues[1:], jnp.ones(1))
    assert_allclose(bounds(tissues, jnp.array([.2, .8])), expected, rtol=1e-10)


@pytest.mark.discrepancy
@pytest.mark.xfail(reason="I02: generalized bounds omit FA and MD", raises=AssertionError)
def test_general_bounds_include_all_five_parameters():
    actual = epg.lb_in_param_holistic_general(
        COEFFICIENTS, POINTS, KNOTS, jnp.array([[1000., 80., .9]]), TENSOR[None],
        jnp.ones(5), jnp.ones(1), 0., 0., PREP, PREP, gradient_number=1,
        n_states=5, method="no phase modulation")
    assert actual.shape == (5,)


def test_scalar_signal_jit_equals_eager_and_magnitude():
    compiled = epg.output_generator(ANGLES, PHASES, *PARAMETERS)
    with jax.disable_jit():
        eager = epg.output_generator(ANGLES, PHASES, *PARAMETERS)
    # Legacy simulator casts RF inputs to float32; fused/non-fused trig evaluation
    # therefore need not agree at float64 precision. Absolute budget is <1e-8 M0.
    assert_allclose(compiled, eager, rtol=3e-7, atol=1e-9)
    assert_allclose(epg.output_plotter(ANGLES, PHASES, *PARAMETERS),
                    np.abs(np.asarray(compiled)[0]+1j*np.asarray(compiled)[1]), atol=1e-14)

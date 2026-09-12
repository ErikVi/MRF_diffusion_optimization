import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
import EPG_blocks_jaxcode as epg
from tests.cases import equilibrium, populated_states, TENSOR


def tensor_step(state, tensor, direction, **kwargs):
    return epg.epg_grelax_generalized(state, 1000., 80., 4., tensor, 1.,
                                      jnp.asarray(direction), kstrength=100, **kwargs)


@pytest.mark.parametrize("gradient", [0, 1])
def test_zero_tensor_equals_zero_scalar_diffusion(gradient):
    state = populated_states()
    actual = tensor_step(state, jnp.zeros((3, 3)), [1., 0., 0.], Gon=gradient)
    expected = epg.epg_grelax(state, 1000., 80., 4., 0., 1., kstrength=100, Gon=gradient)
    assert_allclose(actual, expected, atol=2e-14)


def test_scalar_zero_gradient_attenuates_existing_order_only():
    state = equilibrium().at[0, 1].set(1.).at[1, 1].set(1.)
    actual = epg.epg_grelax(state, jnp.inf, jnp.inf, 4., 0.001, 1., kstrength=100, Gon=0)
    assert_allclose(actual[:2, 1], np.exp(-100**2*0.004*0.001), rtol=1e-7)
    assert_allclose(actual[2, 0], 1.)


def test_scalar_gradient_analytical_pathway():
    # Starting F+0 -> F+1: integral_0^tau (k*t/tau)^2 dt = k^2*tau/3.
    state = equilibrium().at[0, 0].set(1.).at[1, 0].set(1.)
    actual = epg.epg_grelax(state, jnp.inf, jnp.inf, 4., 0.001, 1., kstrength=100)
    assert_allclose(actual[0, 1], np.exp(-100**2*0.004*0.001/3), rtol=1e-7)


@pytest.mark.discrepancy
@pytest.mark.xfail(reason="D01: tensor cross-term and F-/F+ attenuation fail isotropic limit", raises=AssertionError)
def test_tensor_isotropic_limit_matches_scalar_at_nonzero_order():
    state = populated_states()
    actual = tensor_step(state, jnp.eye(3)*0.001, [1., 0., 0.])
    expected = epg.epg_grelax(state, 1000., 80., 4., 0.001, 1., kstrength=100)
    assert_allclose(actual, expected, rtol=2e-7, atol=1e-10)


@pytest.mark.discrepancy
@pytest.mark.xfail(reason="D02: Gon=0 leaves a spurious dk contribution", raises=AssertionError)
def test_tensor_gradient_off_zero_order_has_no_diffusion_decay():
    state = equilibrium().at[0, 0].set(1.).at[1, 0].set(1.)
    actual = tensor_step(state, TENSOR, [1., 0., 0.], Gon=0)
    assert_allclose(actual[0, 0], np.exp(-4/80), rtol=1e-12)


def test_isotropic_diffusion_is_direction_independent():
    state = populated_states()
    a = tensor_step(state, jnp.eye(3)*0.001, [1., 0., 0.])
    b = tensor_step(state, jnp.eye(3)*0.001, [1/np.sqrt(2), 1/np.sqrt(2), 0.])
    assert_allclose(a, b, atol=1e-14)


def test_anisotropic_zero_order_attenuation_uses_directional_diffusivity():
    state = equilibrium().at[0, 0].set(1.).at[1, 0].set(1.)
    direction = np.array([1., 1., 0.])/np.sqrt(2)
    actual = tensor_step(state, TENSOR, direction)
    diffusivity = direction @ np.asarray(TENSOR) @ direction
    assert_allclose(actual[0, 1], np.exp(-4/80-100**2*0.004*diffusivity/3), rtol=1e-12)
    x = tensor_step(state, TENSOR, [1., 0., 0.])[0, 1]
    z = tensor_step(state, TENSOR, [0., 0., 1.])[0, 1]
    assert abs(x) < abs(z)


def test_tensor_and_direction_rotate_together():
    rotation = jnp.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    direction = jnp.array([1., 0., 0.])
    assert_allclose(tensor_step(populated_states(), TENSOR, direction),
                    tensor_step(populated_states(), rotation@TENSOR@rotation.T, rotation@direction), atol=1e-14)


def test_diffusion_direction_sign_symmetry():
    assert_allclose(tensor_step(populated_states(), TENSOR, [1., 0., 0.]),
                    tensor_step(populated_states(), TENSOR, [-1., 0., 0.]), atol=1e-14)


def test_generated_tensor_is_symmetric_positive_definite():
    tensor = np.asarray(epg.generate_diffusion_tensor(0.001))
    assert_allclose(tensor, tensor.T, atol=0)
    assert np.linalg.eigvalsh(tensor).min() > 0


@pytest.mark.discrepancy
@pytest.mark.xfail(reason="M01: MD_from_tensor returns trace instead of trace/3", raises=AssertionError)
def test_mean_diffusivity_analytical():
    assert_allclose(epg.MD_from_tensor(TENSOR), np.trace(TENSOR)/3, rtol=1e-14)


@pytest.mark.discrepancy
@pytest.mark.xfail(reason="M02: dMD_dD disagrees with derivative of current MD function", raises=AssertionError)
def test_mean_diffusivity_derivative_matches_implementation():
    assert_allclose(epg.dMD_dD(), jax.grad(epg.MD_from_tensor)(TENSOR), atol=1e-14)


@pytest.mark.parametrize("eigenvalues", [[1., 1., 1.], [1., 0., 0.], [1.5, 0.7, 0.3]])
def test_fractional_anisotropy_eigenvalue_definition(eigenvalues):
    values = np.asarray(eigenvalues)*0.001
    expected = np.sqrt(1.5*np.sum((values-values.mean())**2)/np.sum(values**2))
    assert_allclose(epg.fractional_anisotropy(jnp.diag(values)), expected, atol=3e-8)


def test_fractional_anisotropy_scale_invariance():
    assert_allclose(epg.fractional_anisotropy(TENSOR*3.7), epg.fractional_anisotropy(TENSOR), atol=1e-14)
    assert_allclose(jnp.sum(jax.grad(epg.fractional_anisotropy)(TENSOR)*TENSOR), 0., atol=1e-13)


@pytest.mark.discrepancy
@pytest.mark.xfail(reason="M03: hand-coded FA derivative violates analytic/AD derivative", raises=AssertionError)
def test_fractional_anisotropy_hand_derivative():
    assert_allclose(epg.dFA_dD(TENSOR), jax.grad(epg.fractional_anisotropy)(TENSOR), rtol=1e-10, atol=1e-10)


def test_scalar_diffusion_derivative_is_minus_b_times_signal():
    state = equilibrium().at[0, 0].set(1.).at[1, 0].set(1.)
    def response(d):
        return epg.epg_grelax(state, jnp.inf, jnp.inf, 4., d, 1., kstrength=100)[0, 1].real
    b = 100**2*.004/3
    assert_allclose(jax.grad(response)(.001), -b*np.exp(-b*.001), rtol=1e-7)


def test_tensor_diffusion_derivative_in_symmetric_direction():
    state = populated_states()
    perturbation = jnp.array([[.2, .1, 0.], [.1, -.1, 0.], [0., 0., .3]])
    def response(t):
        return tensor_step(state, TENSOR+t*perturbation, [1., 0., 0.])[0, 1].real
    step = 1e-8
    expected = (response(step)-response(-step))/(2*step)
    assert_allclose(jax.grad(response)(0.), expected, rtol=1e-7, atol=1e-8)


def test_scalar_and_tensor_diffusion_jit_equal_eager():
    state = populated_states()
    scalar = epg.epg_grelax(state, 1000., 80., 4., .001, 1., kstrength=100)
    tensor = tensor_step(state, TENSOR, [1., 0., 0.])
    with jax.disable_jit():
        assert_allclose(epg.epg_grelax(state, 1000., 80., 4., .001, 1., kstrength=100), scalar, atol=1e-12)
        assert_allclose(tensor_step(state, TENSOR, [1., 0., 0.]), tensor, atol=1e-12)

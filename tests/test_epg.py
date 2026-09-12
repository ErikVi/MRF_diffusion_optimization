from mrf_diffusion.simulation import signal as signal_module

"""RF convention: F+ = Mx+iMy; positive x rotation sends +z to -y.

Reference: Hargreaves RAD229 B2, slides 17,20,39. Zero-gradient
relaxation tolerance accounts for the legacy float32 relaxation matrix.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
import mrf_diffusion.api as epg
from tests.cases import equilibrium, populated_states


def test_simulator_initializes_only_unit_z0(monkeypatch):
    observed = []
    original = epg.apply_rf_rotation

    def record_input(state, *args, **kwargs):
        observed.append(np.asarray(state))
        return original(state, *args, **kwargs)

    monkeypatch.setattr(signal_module, "apply_rf_rotation", record_input)
    with jax.disable_jit():
        epg.simulate_scalar_signal(
            jnp.array([0.4]), jnp.array([0.0]), 1000.0, 80.0, 0.0, 1.0
        )
    expected = np.zeros((3, 150), dtype=complex)
    expected[2, 0] = 1.0
    assert_allclose(observed[0], expected, atol=0)


def test_first_echo_after_ideal_preparation():
    angle = 0.4
    signal = epg.simulate_scalar_signal(
        jnp.array([angle]), jnp.array([0.0]), jnp.inf, jnp.inf, 0.0, 1.0
    )
    expected = np.sin(np.float32(angle))
    assert_allclose(signal[:, 0], [0.0, expected], atol=3e-08)


def test_zero_flip_angle():
    state = populated_states()
    assert_allclose(epg.apply_rf_rotation(state, 0.0, 0.73), state, atol=1e-14)


@pytest.mark.parametrize("angle", [np.pi / 2, -np.pi / 2, np.pi, 0.37])
@pytest.mark.parametrize("phase", [0.0, np.pi / 2, -0.43])
def test_rf_against_rodrigues_rotation(angle, phase):
    magnetization = np.array([0.2, -0.3, 0.7])
    axis = np.array([np.cos(phase), np.sin(phase), 0.0])
    expected = (
        magnetization * np.cos(angle)
        + np.cross(axis, magnetization) * np.sin(angle)
        + axis * np.dot(axis, magnetization) * (1 - np.cos(angle))
    )
    state = (
        equilibrium()
        .at[:, 0]
        .set(
            jnp.array(
                [
                    magnetization[0] + 1j * magnetization[1],
                    magnetization[0] - 1j * magnetization[1],
                    magnetization[2],
                ]
            )
        )
    )
    actual = np.asarray(epg.apply_rf_rotation(state, angle, phase))[:, 0]
    assert_allclose(
        actual,
        [expected[0] + 1j * expected[1], expected[0] - 1j * expected[1], expected[2]],
        atol=2e-14,
    )


def test_positive_gradient_and_refocusing():
    state = equilibrium().at[0, 0].set(1j).at[1, 0].set(-1j)
    shifted = epg.apply_positive_gradient_shift(state)
    assert_allclose(shifted[0], [0, 1j, 0, 0, 0], atol=1e-14)
    assert_allclose(shifted[1], 0, atol=1e-14)
    assert_allclose(shifted[2], state[2])
    rephased = epg.apply_positive_gradient_shift(
        epg.apply_rf_rotation(shifted, np.pi, 0.0)
    )
    assert_allclose(rephased[:2, 0], [-1j, 1j], atol=1e-14)


def test_positive_gradient_truncates_without_wrap():
    state = equilibrium().at[0, -1].set(2j)
    assert_allclose(epg.apply_positive_gradient_shift(state)[0], 0.0)
    grown = epg.apply_positive_gradient_shift(state, truncate=0)
    assert grown.shape == (3, 6)
    assert_allclose(grown[0, -1], 2j)


@pytest.mark.discrepancy
@pytest.mark.xfail(
    reason="G01: negative shift overwrites arriving F+0", raises=AssertionError
)
def test_negative_gradient_refocuses_positive_order():
    state = equilibrium().at[0, 1].set(0.3 + 0.4j)
    actual = epg.apply_negative_gradient_shift(state)
    assert_allclose(actual[:2, 0], [0.3 + 0.4j, 0.3 - 0.4j])


def test_zero_gradient_step_is_identity():
    state = populated_states()
    assert_allclose(epg.apply_positive_gradient_shift(state, order_step=0), state)
    with jax.disable_jit():
        assert_allclose(epg.apply_positive_gradient_shift(state, order_step=0), state)


def test_relaxation_analytical_no_gradient():
    state = populated_states()
    expected = (
        np.asarray(state)
        * np.array([np.exp(-12 / 80), np.exp(-12 / 80), np.exp(-12 / 1000)])[:, None]
    )
    expected[2, 0] += 1 - np.exp(-12 / 1000)
    assert_allclose(
        epg.relax_and_shift(state, 1000.0, 80.0, 12.0, gradient_enabled=0),
        expected,
        rtol=6e-08,
        atol=3e-08,
    )


def test_long_t1_preserves_longitudinal_states():
    state = populated_states()
    actual = epg.relax_and_shift(state, jnp.inf, 80.0, 12.0, gradient_enabled=0)
    assert_allclose(actual[2], state[2], atol=1e-14)


def test_long_t2_preserves_transverse_states():
    state = populated_states()
    actual = epg.relax_and_shift(state, 1000.0, jnp.inf, 12.0, gradient_enabled=0)
    assert_allclose(actual[:2], state[:2], atol=1e-14)


def test_equilibrium_stationary_under_relaxation():
    assert_allclose(
        epg.relax_and_shift(equilibrium(), 1000.0, 80.0, 12.0, gradient_enabled=0),
        equilibrium(),
        atol=3e-08,
    )


def test_zero_diffusion_matches_relaxation():
    state = populated_states()
    assert_allclose(
        epg.relax_diffuse_scalar_and_shift(
            state, 1000.0, 80.0, 12.0, 0.0, 1.0, gradient_enabled=0
        ),
        epg.relax_and_shift(state, 1000.0, 80.0, 12.0, gradient_enabled=0),
        atol=3e-08,
    )


def test_jit_rf_and_relaxation_match_eager():
    state = populated_states()
    compiled_rf = epg.apply_rf_rotation(state, 0.47, -0.21)
    compiled_relax = epg.relax_and_shift(state, 1000.0, 80.0, 12.0, gradient_enabled=0)
    with jax.disable_jit():
        assert_allclose(
            epg.apply_rf_rotation(state, 0.47, -0.21), compiled_rf, atol=1e-14
        )
        assert_allclose(
            epg.relax_and_shift(state, 1000.0, 80.0, 12.0, gradient_enabled=0),
            compiled_relax,
            atol=1e-14,
        )

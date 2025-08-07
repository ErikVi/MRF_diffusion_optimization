import jax.numpy as jnp
import numpy as np
from jax import lax, jit
import jax
from functools import partial
import jax
import jax.numpy as jnp
from jax import grad
import inspect
from jax import jacobian, jacfwd, jacrev
from scipy.optimize  import minimize
jax.config.update("jax_enable_x64", True)
from functools import partial
from scipy.interpolate import BSpline


@partial(jax.jit, static_argnums = (1, 2))
def epg_mgrad(FpFmZ, deltak=1, noadd=1):
    """
    Applies a negative gradient shift to the EPG state matrix FpFmZ.

    Simulates the effect of a negative gradient by circularly shifting the F+ and F- states
    in opposite directions. Optionally appends a zero state before shifting to increase
    the maximum EPG order.

    Parameters
    ----------
    FpFmZ : jax.Array, shape (3, N)
        EPG state matrix containing F+, F-, and Z components along the columns.
    deltak : int, default=1
        Number of gradient steps to shift the EPG states.
    noadd : int or float, default=0
        If 0, appends a zero-order state before shifting; if nonzero, no state is added.

    Returns
    -------
    jax.Array, shape (3, N + deltak) or (3, N)
        Updated EPG state matrix after applying the negative gradient shift.
    """
    if noadd == 0.:
        FpFmZ = jnp.concatenate((FpFmZ, jnp.array([[0.], [0.], [0.]])), axis=1)

    N = FpFmZ.shape[1]

    FpFmZTop = jnp.concatenate([FpFmZ[:, deltak:], FpFmZ[:, :deltak]], axis=1)

    FpFmZBottom = jnp.concatenate([FpFmZ[:, -deltak:], FpFmZ[:, :-deltak]], axis=1)

    FpFmZ = FpFmZ.at[0:2, :].set(jnp.stack([FpFmZTop[0, :], FpFmZBottom[1, :]]))

    zero_block = jnp.zeros((1, deltak), dtype=FpFmZ.dtype)
    FpFmZ = lax.dynamic_update_slice(FpFmZ, zero_block, (0, N - deltak))

    FpFmZ = FpFmZ.at[0, 0].set(jnp.conj(FpFmZ[1, 0]))

    return FpFmZ

@partial(jax.jit, static_argnums = (1, 2))
def epg_grad(FpFmZ, deltak=1, noadd=1):
    """
    Applies a positive gradient shift to the EPG state matrix FpFmZ.

    Simulates the effect of a positive gradient by circularly shifting the F+ and F− states
    in opposite directions. Optionally appends a zero state before shifting to allow for 
    higher EPG order tracking.

    Parameters
    ----------
    FpFmZ : jax.Array, shape (3, N)
        EPG state matrix containing F+, F−, and Z components along the columns.
    deltak : int, default=1
        Number of gradient steps to shift the EPG states.
    noadd : int or float, default=0
        If 0, appends a zero-order state before shifting; if nonzero, no state is added.

    Returns
    -------
    jax.Array, shape (3, N + deltak) or (3, N)
        Updated EPG state matrix after applying the positive gradient shift.
    """
    if noadd == 0.:
        FpFmZ = jnp.concatenate((FpFmZ, jnp.array([[0.], [0.], [0.]])), axis=1)

    N = FpFmZ.shape[1]

    FpFmZTop = jnp.concatenate([FpFmZ[:, -deltak:], FpFmZ[:, :-deltak]], axis=1)

    FpFmZBottom = jnp.concatenate([FpFmZ[:, deltak:], FpFmZ[:, :deltak]], axis=1)

    FpFmZ = FpFmZ.at[0:2, :].set(jnp.stack([FpFmZTop[0, :], FpFmZBottom[1, :]]))

    zero_block = jnp.zeros((1, deltak), dtype=FpFmZ.dtype)
    FpFmZ = lax.dynamic_update_slice(FpFmZ, zero_block, (1, N - deltak))

    FpFmZ = FpFmZ.at[0, 0].set(jnp.conj(FpFmZ[1, 0]))

    return FpFmZ

@partial(jax.jit, static_argnums = (4, 5, 6))
def epg_relax(FpFmZ, T1, T2, T, deltak = 1, Gon = 1., noadd = 1):
    """
    Applies relaxation and optional gradient dephasing to the EPG state matrix FpFmZ.

    Simulates T1 and T2 relaxation over a time interval T by scaling the EPG states with
    exponential decay factors. The longitudinal component (Z state) is additionally driven 
    toward equilibrium. Optionally applies a positive gradient shift to simulate dephasing.

    Parameters
    ----------
    FpFmZ : jax.Array, shape (3, N)
        EPG state matrix containing F+, F−, and Z components along the columns.
    T1 : float
        Longitudinal relaxation time constant (ms).
    T2 : float
        Transverse relaxation time constant (ms).
    T : float
        Time interval over which relaxation occurs (ms).
    deltak : int, default=1
        Number of gradient steps to shift if Gon is enabled.
    Gon : float, default=1.0
        If 1.0, applies a gradient dephasing step after relaxation.
    noadd : int or float, default=0
        If 0, appends a zero-order state before gradient shifting; otherwise, skips it.

    Returns
    -------
    jax.Array, shape (3, N + deltak) or (3, N)
        Updated EPG state matrix after relaxation and optional gradient dephasing.
    """
    E2 = jnp.exp(-T / T2)
    E1 = jnp.exp(-T / T1)
    EE = jnp.diag(jnp.array([E2, E2, E1], dtype=jnp.float32))

    FpFmZ = jnp.matmul(EE, FpFmZ)
    FpFmZ = FpFmZ.at[2, 0].add(1.0 - E1)
    if Gon == 1.0:
        FpFmZ = epg_grad(FpFmZ, deltak, noadd)
    
    return FpFmZ

@partial(jax.jit, static_argnums = (6, 7, 8, 9))
def epg_grelax(FpFmZ, T1, T2, T, D, M, kstrength = 10, deltak = 1, Gon = 1, noadd = 1):    
    """
    Applies relaxation and diffusion attenuation to the EPG state matrix with optional gradient dephasing.

    This function models T1 and T2 relaxation using exponential decay, includes diffusion effects
    through component-specific b-values scaled by a scalar diffusivity D, and optionally applies
    gradient dephasing by shifting the EPG states.

    Parameters
    ----------
    FpFmZ : jax.Array, shape (3, N)
        EPG state matrix containing F+, F−, and Z components along the columns.
    T1 : float
        Longitudinal relaxation time constant (ms).
    T2 : float
        Transverse relaxation time constant (ms).
    T : float
        Time interval over which relaxation and diffusion occur (ms).
    D : float
        Scalar diffusivity (mm²/s).
    M : float
        Equilibrium magnetization (used for recovery term in Z state).
    gradient_dir : jax.Array, shape (3,)
        Gradient direction vector (unused in this simplified model).
    kstrength : float, default=10
        Gradient strength in units of 1/mm.
    deltak : int, default=1
        Number of gradient steps for the EPG shift.
    Gon : float, default=1.0
        If 1.0, applies a gradient dephasing step after relaxation and diffusion.
    noadd : int or float, default=0
        If 0, appends a zero-order state before gradient shifting; otherwise, skips it.

    Returns
    -------
    jax.Array, shape (3, N + deltak) or (3, N)
        Updated EPG state matrix after applying relaxation, diffusion attenuation,
        and optional gradient dephasing.
    """  
    E1 = jnp.exp(-T / T1)
    E2 = jnp.exp(-T / T2)
    EE = jnp.diag(jnp.array([E2, E2, E1]))

    #FpFmZ = jnp.matmul(EE, FpFmZ)
    #FpFmZ = FpFmZ.at[2, 0].add(1.0 - E1)

    E1a = jnp.zeros(FpFmZ.shape[1]).at[0].set(M * (1.0 - E1))

    Findex = jnp.arange(FpFmZ.shape[1], dtype = jnp.float32)
    bvalZ = (Findex * kstrength) ** 2 * T / 1000
    bvalp = ((Findex + 0.5 * Gon) * kstrength) ** 2 * T / 1000 + Gon * (kstrength ** 2) / 12 * T / 1000
    bvalm = ((-Findex + 0.5 * Gon) * kstrength) ** 2 * T / 1000 + Gon * (kstrength ** 2) / 12 * T / 1000
    FpFmZ = jnp.stack([FpFmZ[0, :] * E2 * jnp.exp(-bvalp * D),
                                            FpFmZ[1, :] * E2 * jnp.exp(-bvalm * D),
                                            (FpFmZ[2, :] * E1 + E1a) * jnp.exp(-bvalZ * D)])

    if Gon == 1.0:
        FpFmZ = lax.cond(deltak >= 0.0, lambda x: epg_grad(x, deltak, noadd), lambda x: epg_mgrad(x, deltak, noadd), FpFmZ)
    
    return FpFmZ



'''
@partial(jax.jit, static_argnums = (7, 8, 9, 10))
def epg_grelax_generalized(FpFmZ, T1, T2, T, D_tensor, M, gradient_dir, kstrength = 10, deltak = 1, Gon = 1, noadd = 0):
    """
    Applies generalized relaxation, diffusion attenuation, and optional gradient dephasing to the EPG state matrix.

    This function simulates T1 and T2 relaxation, diffusion effects through a diffusion tensor, and optionally applies 
    gradient dephasing by shifting the EPG states. The diffusion is modeled using a tensor, and the gradient is applied based 
    on the supplied direction and strength.

    Parameters
    ----------
    FpFmZ : jax.Array, shape (3, N)
        EPG state matrix containing F+, F−, and Z components along the columns.
    T1 : float
        Longitudinal relaxation time constant (ms).
    T2 : float
        Transverse relaxation time constant (ms).
    T : float
        Time interval over which relaxation and diffusion occur (ms).
    D_tensor : jax.Array, shape (3, 3, N)
        Diffusion tensor, representing the diffusion properties along different directions.
    M : float
        Equilibrium magnetization (used for recovery term in Z state).
    gradient_dir : jax.Array, shape (3,)
        Gradient direction vector, used for calculating the gradient's effect.
    kstrength : float, default=10
        Gradient strength in units of 1/mm.
    deltak : int, default=1
        Number of gradient steps for the EPG shift.
    Gon : float, default=1.0
        If 1.0, applies a gradient dephasing step after relaxation and diffusion.
    noadd : int or float, default=0
        If 0, appends a zero-order state before gradient shifting; otherwise, skips it.

    Returns
    -------
    jax.Array, shape (3, N + deltak) or (3, N)
        Updated EPG state matrix after applying generalized relaxation, diffusion attenuation,
        and optional gradient dephasing.
    """    
    E1 = jnp.exp(-T / T1)
    E2 = jnp.exp(-T / T2)
    EE = jnp.diag(jnp.array([E2, E2, E1]))
    
    E1a = jnp.zeros(FpFmZ.shape[1]).at[0].set(M * (1.0 - E1))

    Findex = jnp.arange(FpFmZ.shape[1], dtype = jnp.float32)
    k_vals = jnp.outer(gradient_dir, Findex * kstrength)  
    b_s_T = jnp.zeros((3, 3, FpFmZ.shape[1]))
    b_s_L = jnp.zeros((3, 3, FpFmZ.shape[1]))

    k1 = k_vals[:, None, :]  
    k2 = k_vals[None, :, :] 

    k1k2 = k1 * k2  

    b_s_T = k1k2 * (T + 0.5 * T + (1.0 / 3.0) * T)  
    b_s_L = k1k2 * T  
    
    diffusion_decay_T = jnp.exp(-jnp.einsum('mnr,mn->r', b_s_T, D_tensor))
    diffusion_decay_L = jnp.exp(-jnp.einsum('mnr,mn->r', b_s_L, D_tensor))
    
    FpFmZ = jnp.stack([
        FpFmZ[0, :] * E2 * diffusion_decay_T,
        FpFmZ[1, :] * E2 * diffusion_decay_T,
        (FpFmZ[2, :] * E1 + E1a) * diffusion_decay_L
    ])

    if Gon == 1.0:
        FpFmZ = lax.cond(deltak >= 0.0, lambda x: epg_grad(x, deltak, noadd), lambda x: epg_mgrad(x, deltak, noadd), FpFmZ)

    return FpFmZ
'''




@partial(jax.jit, static_argnums = (7, 8, 9, 10))
def epg_grelax_generalized(FpFmZ, T1, T2, T, D_tensor, M, gradient_dir, kstrength=10, deltak=1, Gon=1, noadd=1):
    
    E1 = jnp.exp(-T / T1)
    E2 = jnp.exp(-T / T2)
    EE = jnp.diag(jnp.array([E2, E2, E1]))
    T = T/1000
    
    E1a = jnp.zeros(FpFmZ.shape[1]).at[0].set(M * (1.0 - E1))

    
    Findex = jnp.arange(FpFmZ.shape[1], dtype=jnp.float32)

    # k1 = initial k value, k2 = final k value
    k1 = jnp.outer(gradient_dir, Findex * kstrength)                 
    k2 = jnp.outer(gradient_dir, (Findex + deltak) * kstrength)     

    # Compute longitudinal b-tensor: b_L[m,n] = k1_m * k1_n * T
    b_s_L = jnp.einsum('in,jn->ijn', k1, k1) * T                   

    # Cross terms: (1/2) * (k1_m * k2_n - k1_n * k2_m) * T
    cross1 = jnp.einsum('in,jn->ijn', k1, k2)
    cross2 = jnp.einsum('in,jn->ijn', k1, k2)
    cross_term = 0.5 * (cross1 - cross2) * T                         

    # Pure diffusion term: (1/3) * (k2_m - k1_m)(k2_n - k1_n) * T
    dk = k2 - k1                                                    
    pure_term = jnp.einsum('in,jn->ijn', dk, dk) * (T / 3.0)

    # Total transvers btensor = b_L + cross_term + pure_term
    b_s_T = b_s_L + cross_term + pure_term

    # Compute decay factors for each state (I indexed over N)
    diffusion_decay_T = jnp.exp(-jnp.einsum('mnr,mn->r', b_s_T, D_tensor))
    diffusion_decay_L = jnp.exp(-jnp.einsum('mnr,mn->r', b_s_L, D_tensor))

    
    FpFmZ = jnp.stack([
        FpFmZ[0, :] * E2 * diffusion_decay_T,
        FpFmZ[1, :] * E2 * diffusion_decay_T,
        (FpFmZ[2, :] * E1 + E1a) * diffusion_decay_L
    ])

    if Gon == 1.0:
        FpFmZ = lax.cond(deltak >= 0,
                         lambda x: epg_grad(x, deltak, noadd),
                         lambda x: epg_mgrad(x, deltak, noadd),
                         FpFmZ)

    return FpFmZ







@jax.jit
def epg_rf(FpFmZ, alpha, phi = (-jnp.pi / 2)):
    """
    Applies a radiofrequency (RF) pulse to the EPG state matrix.

    This function simulates the effect of an RF pulse with a specified flip angle (alpha) and phase (phi) on the EPG state 
    matrix. The RF pulse is modeled as a rotation matrix applied to the F+, F−, and Z components of the matrix.

    Parameters
    ----------
    FpFmZ : jax.Array, shape (3, N)
        EPG state matrix containing F+, F−, and Z components along the columns.
    alpha : float
        Flip angle (in radians) of the RF pulse.
    phi : float, optional, default=(-jnp.pi / 2)
        Phase of the RF pulse (in radians). The default is a phase of -π/2, commonly used in spin-lock or refocusing pulses.

    Returns
    -------
    jax.Array, shape (3, N)
        Updated EPG state matrix after applying the RF pulse with the specified flip angle and phase.
    """
    exp_2j_phi = jnp.exp(2j * phi)
    exp_j_phi = jnp.exp(1j * phi)

    row_0 = jnp.array([
        jnp.cos(alpha / 2) ** 2,
        exp_2j_phi * (jnp.sin(alpha / 2) ** 2),
        -1j * exp_j_phi * jnp.sin(alpha)
    ])

    row_1 = jnp.array([
        jnp.conj(exp_2j_phi) * (jnp.sin(alpha / 2) ** 2),
        jnp.cos(alpha / 2) ** 2,
        1j * jnp.conj(exp_j_phi) * jnp.sin(alpha)
    ])

    row_2 = jnp.array([
        (-1j / 2) * jnp.conj(exp_j_phi) * jnp.sin(alpha),
        (1j / 2) * exp_j_phi * jnp.sin(alpha),
        jnp.cos(alpha)
    ])

    RR = jnp.stack([row_0, row_1, row_2], axis=0)
    FpFmZ = jnp.matmul(RR, FpFmZ)
    return FpFmZ


def phase_modulation_generator(FA_array, fraction, method):
    """
    Generates a phase modulation array based on the specified method.

    This function creates a phase modulation array that varies according to different methods such as quadratic, linear, 
    sinusoidal, or alternating. The phase values are then converted from degrees to radians and returned.

    Parameters
    ----------
    FA_array : array-like, shape (N,)
        An array representing the flip angle values. Its length determines the length of the phase modulation array.
    fraction : float
        A fraction (between 0 and 1) used to control the transition point for the phase modulation in methods that require it.
    method : str
        The phase modulation method to apply. Can be one of:
            - 'quadratic': Applies a quadratic phase modulation.
            - 'linear': Applies a linear phase modulation.
            - 'sinusoidal': Applies a sinusoidal phase modulation.
            - 'alternating': Applies an alternating phase modulation with a sign change.

    Returns
    -------
    jax.Array, shape (N,)
        The phase modulation array in radians, based on the specified method.
    """
    N = len(FA_array)
    phase_modulation = np.zeros(N)

    if method == 'quadratic':
        for n in range(1, N):
            if n < N * fraction:
                phase_modulation[n] = phase_modulation[n - 1] + 2 * n
            elif n > N * fraction + 1:
                phase_modulation[n] = phase_modulation[n - 1] - 2 * n

    elif method == 'linear':
        for n in range(1, N):
            if n < N * fraction:
                phase_modulation[n] = phase_modulation[n - 1] + 10
            elif n > N * fraction + 1:
                phase_modulation[n] = phase_modulation[n - 1] - 10

    elif method == 'sinusoidal':
        phase_modulation = 1000 * np.sin(2 * np.pi * (np.arange(N) / N + fraction))

    elif method == 'alternating':
        for n in range(1, N):
            phase_modulation[n] = phase_modulation[n - 1] + ((-1) ** n) * 1000 * fraction

    phase_modulation = np.deg2rad(phase_modulation)
    return jnp.array(phase_modulation)

@jax.jit
def output_generator(FA, phase_modulation, T1, T2, D, M):
    TE, TR, TI, deltak = 4, 15, 15, 1
    n_states = 150
    FpFmZ = jnp.zeros((3, n_states), dtype = jnp.complex128).at[2, 0].set(1.0)

    FpFmZ = epg_rf(FpFmZ, alpha = jnp.pi / 2, phi = 0)
    FpFmZ = epg_grelax(FpFmZ, T1, T2, TE, D, M, kstrength = 612, deltak = 1, Gon = 1, noadd = 1)
    FpFmZ = epg_rf(FpFmZ, alpha = jnp.pi, phi = 0)
    FpFmZ = epg_grelax(FpFmZ, T1, T2, TE, D, M, kstrength = 612, deltak = 1, Gon = 1, noadd = 1)
    FpFmZ = epg_rf(FpFmZ, alpha = - jnp.pi / 2, phi = 0)
    '''#............INVERSION PULSE REMOVE AFTERWARDS............
    FpFmZ = epg_rf(FpFmZ, alpha = jnp.pi, phi=0)
    FpFmZ = epg_relax(FpFmZ, T1, T2, jnp.array(TI, dtype=jnp.float32), deltak = 50)  ''' 
    

    def scan_fn(FpFmZ, inputs):
        alpha, phi = inputs
        FpFmZ = epg_rf(FpFmZ, alpha = alpha, phi = phi)
        FpFmZ = epg_grelax(FpFmZ, T1, T2, TE, D, M, kstrength = 50, deltak = deltak, Gon = 0, noadd=1)

        real_part = jnp.real(FpFmZ[0, 0])
        imag_part = jnp.imag(FpFmZ[0, 0])
        
        FpFmZ = epg_grelax(FpFmZ, T1, T2, TR - TE, D, M, kstrength = 50, deltak = deltak, Gon = 1, noadd=1)
        return FpFmZ, (real_part, imag_part)

    inputs = (jnp.array(FA, dtype = jnp.float32), jnp.array(phase_modulation, dtype = jnp.float32))
    _, (transversal_data_real, transversal_data_imag) = jax.lax.scan(scan_fn, FpFmZ, inputs)

    return jnp.array([transversal_data_real, transversal_data_imag])

@jax.jit
def output_plotter(FA, phase_modulation, T1, T2, D, M):
    raw_output = output_generator(FA, phase_modulation, T1, T2, D, M)
    return jnp.sqrt(raw_output[0] ** 2 + raw_output[1] ** 2)


@partial(jax.jit, static_argnames = ['degree'])
def bspline(t_values,knots, coefficients, degree = 3):
    """
    Evaluates a B-spline curve at given parameter values using specified knots, coefficients, and spline degree.

    Parameters
    ----------
    t_values : array_like
        Array of parameter values at which to evaluate the B-spline.
    knots : array_like
        Non-decreasing sequence of knot positions defining the B-spline basis.
    coefficients : array_like
        Control point coefficients for the B-spline curve.
    degree : int, optional
        Degree of the B-spline basis functions (default is 3 for cubic splines).

    Returns
    -------
    curve_points : jax.numpy.ndarray
        Evaluated points of the B-spline curve at the specified `t_values`.

    Notes
    -----
    Implements the Cox–de Boor recursion formula to construct B-spline basis functions.
    """
    def basis_function(i, k, t, knots):
        if k == 0:
            return jnp.where((knots[i] <= t) & (t < knots[i + 1]), 1.0, 0.0)
        else:
            denom1 = knots[i + k] - knots[i]
            denom2 = knots[i + k + 1] - knots[i + 1]

            term1 = jnp.where(denom1 != 0, (t - knots[i]) / denom1 * basis_function(i, k - 1, t, knots), 0.0)
            term2 = jnp.where(denom2 != 0, (knots[i + k + 1] - t) / denom2 * basis_function(i + 1, k - 1, t, knots), 0.0)

            return term1 + term2 
 
    def scan_fn(curve_points, i):
        v = basis_function(i, degree, t_values, knots)
        return curve_points + v * coefficients[i], None

    curve_points = jnp.zeros(len(t_values))
    curve_points, _ = lax.scan(scan_fn, curve_points, jnp.arange(len(coefficients)))
    return curve_points




@partial(jax.jit, static_argnames=['degree'])
def bspline_vectorized(t_values, knots, coefficients, degree=3):
    
    n_coeffs = len(coefficients)

    def basis_fn_matrix(t):
        N = jnp.zeros(n_coeffs)

        def cox_de_boor(j, d):
            if d == 0:
                return jnp.where((knots[j] <= t) & (t < knots[j + 1]), 1.0, 0.0)
            else:
                denom1 = knots[j + d] - knots[j]
                denom2 = knots[j + d + 1] - knots[j + 1]

                term1 = jnp.where(denom1 > 0, (t - knots[j]) / denom1 * cox_de_boor(j, d - 1), 0.0)
                term2 = jnp.where(denom2 > 0, (knots[j + d + 1] - t) / denom2 * cox_de_boor(j + 1, d - 1), 0.0)
                return term1 + term2

        for i in range(n_coeffs):
            N = N.at[i].set(cox_de_boor(i, degree))

        return N

    B_matrix = jax.vmap(basis_fn_matrix)(t_values) 
    return B_matrix @ coefficients




@partial(jax.jit, static_argnames = ['degree'])
def bspline_vmap(t_values, knots, coefficients, degree=3):
    

    def basis_function(i, k, t):
        if k == 0:
            return jnp.where((knots[i] <= t) & (t < knots[i + 1]), 1.0, 0.0)
        else:
            denom1 = knots[i + k] - knots[i]
            denom2 = knots[i + k + 1] - knots[i + 1]

            term1 = jnp.where(denom1 != 0,
                              (t - knots[i]) / denom1 * basis_function(i, k - 1, t),
                              0.0)
            term2 = jnp.where(denom2 != 0,
                              (knots[i + k + 1] - t) / denom2 * basis_function(i + 1, k - 1, t),
                              0.0)
            return term1 + term2

    basis_function_vmap = jax.vmap(lambda t: jnp.array([basis_function(i, degree, t) for i in range(len(coefficients))]))

    basis_vals = basis_function_vmap(t_values)

    curve_points = jnp.dot(basis_vals, coefficients)

    return curve_points




def bspline_coefficients(y_values, knots, degree):
    """
    Computes B-spline coefficients that best fit the given data using least squares approximation.

    Parameters
    ----------
    y_values : array_like
    Data values to be approximated by the B-spline curve.
    knots : array_like
    Non-decreasing sequence of knot positions defining the B-spline basis.
    degree : int
    Degree of the B-spline basis functions.

    Returns
    -------
    coefficients : jax.numpy.ndarray
    Optimal B-spline coefficients that minimize the least squares error to `y_values`.

    Notes
    -----
    Constructs the B-spline basis matrix using the Cox–de Boor recursion formula and solves the
    linear system using least squares to obtain the best-fitting coefficients.
    """    
    num_coeffs = len(knots) - degree - 1  
    num_points = len(y_values)    
    t_values = jnp.linspace(0, num_points, num_points)   
    def basis_function(i, k, t, knots):
        if k == 0:
            return jnp.where((knots[i] <= t) & (t < knots[i + 1]), 1.0, 0.0)
        else:
            denom1 = knots[i + k] - knots[i]
            denom2 = knots[i + k + 1] - knots[i + 1]
            term1 = (t - knots[i]) / denom1 * basis_function(i, k - 1, t, knots) if denom1 != 0 else 0
            term2 = (knots[i + k + 1] - t) / denom2 * basis_function(i + 1, k - 1, t, knots) if denom2 != 0 else 0
            return term1 + term2  
    B = jnp.zeros((num_points, num_coeffs))
    for i in range(num_coeffs):
        B = B.at[:, i].set(basis_function(i, degree, t_values, knots))
    coefficients, _, _, _ = jax.numpy.linalg.lstsq(B, y_values, rcond=None)    
    return coefficients




@jax.jit
def jacobian_calculator(FA, phase_modulation, T1, T2, D, M):
    return jnp.array(jacobian(output_generator, argnums = [2, 3, 4, 5])(FA, phase_modulation, T1, T2, D, M))
    
@partial(jax.jit, static_argnums = 6)
def fisher_information(FA, phase_modulation, T1, T2, D, M, sigma = 10**(-1.65)):
    """
    Computes the Fisher Information Matrix (FIM) for a given signal model and noise level.

    Parameters
    ----------
    FA : array_like
        Flip angle train (in radians) used in the acquisition.
    phase_modulation : array_like
        Phase increments applied during the acquisition.
    T1 : array_like
        Longitudinal relaxation times (in milliseconds).
    T2 : array_like
        Transverse relaxation times (in milliseconds).
    D : array_like
        Diffusion coefficients or diffusion-related parameters.
    M : array_like
        Simulated magnetization or signal evolution.
    sigma : float, optional
        Standard deviation of Gaussian noise in the signal (default is 10^(-1.65)).

    Returns
    -------
    FIM : jax.numpy.ndarray
        Fisher Information Matrix corresponding to the parameters of interest.

    Notes
    -----
    The Fisher Information Matrix is computed as:
        FIM = (1 / sigma^2) * J @ J.T
    where `J` is the Jacobian matrix of the signal with respect to the model parameters.
    """

    T1 = jnp.asarray(T1, dtype = jnp.float64)
    T2 = jnp.asarray(T2, dtype = jnp.float64)
    D  = jnp.asarray(D,  dtype = jnp.float64)
    M  = jnp.asarray(M,  dtype = jnp.float64)
       
    J = jacobian_calculator(FA, phase_modulation, T1, T2, D, M)
    J = J.reshape(J.shape[0], -1)
    FIM = J @ J.T 
    

    return 1/(sigma**2) * FIM

@jax.jit
def objective_holistic(coefficients, phase_modulation, eval_points, knots, params_list, W_mtrx, W_mtrx2):
    FA = bspline(eval_points, knots, coefficients)
    
    def compute_crlb(params, weight):
        T1, T2, PD, M = params
        FIM = fisher_information(FA, phase_modulation, T1, T2, PD, M)
        crlb = jnp.sqrt(jnp.sum(W_mtrx * jnp.diag(jnp.linalg.inv(FIM)) / (params ** 2)))
        return crlb * weight

    total_crlb = jax.vmap(compute_crlb)(params_list, W_mtrx2)
    
    return jnp.sum(total_crlb)



@jax.jit
def lb_in_param_holistic(coefficients, phase_modulation, eval_points, knots, params_list, W_mtrx, W_mtrx2):
    FA = bspline(eval_points, knots, coefficients)
    crlb_array = jnp.zeros(len(params_list[0]))      
    def scan_fn(carry, params):
        T1, T2, PD, M = params        
        FIM = fisher_information(FA, phase_modulation, T1, T2, PD, M)        
        crlb = W_mtrx * jnp.sqrt(jnp.diag(jnp.linalg.inv(FIM) / (params ** 2)))
        return carry, crlb * W_mtrx2[carry]
    
    _, crlb_contributions = jax.lax.scan(scan_fn, 0, params_list)
    return jnp.sum(crlb_contributions, axis = 0)




#       ███████  █████         ███    ███ ███████  █████  ███    ██     ██████  ██ ███████ ███████ 
#       ██      ██   ██        ████  ████ ██      ██   ██ ████   ██     ██   ██ ██ ██      ██      
#       █████   ███████        ██ ████ ██ █████   ███████ ██ ██  ██     ██   ██ ██ █████   █████   
#       ██      ██   ██        ██  ██  ██ ██      ██   ██ ██  ██ ██     ██   ██ ██ ██      ██      
#       ██      ██   ██ ▄█     ██      ██ ███████ ██   ██ ██   ████     ██████  ██ ██      ██      

@jax.jit                                                                                          
def generate_diffusion_tensor(scale):
    return jnp.array([
        [scale * 1.8, scale * 0.1, scale * 0.1],  
        [scale * 0.1, scale * 1., scale * 0.1], 
        [scale * 0.1, scale * 0.1, scale * 0.2]  
    ])

@jax.jit
def fractional_anisotropy(D):
    """
    Computes the fractional anisotropy (FA) from a diffusion tensor.

    Parameters
    ----------
    D : array_like
        3×3 diffusion tensor represented as a symmetric positive-definite matrix.

    Returns
    -------
    FA : float
        Fractional anisotropy value, a scalar between 0 (isotropic) and 1 (maximally anisotropic).

    Notes
    -----
    FA is computed using a trace-normalized diffusion tensor R = D / trace(D),
    and the relation:
        FA = sqrt(0.5 * (3 - 1 / trace(R^2)))
    This formulation ensures scale invariance with respect to the overall diffusivity.
    """

    trace_D = jnp.trace(D)
    R = D / trace_D
    R_squared = jnp.dot(R, R)
    trace_R_squared = jnp.trace(R_squared)
    FA = jnp.sqrt(0.5 * (3 - 1 / trace_R_squared))    
    return FA

@jax.jit
def dFA_dD(D):
    """
    Compute the derivative of the fractional anisotropy (FA) with respect to the diffusion tensor D.

    Parameters:
        D (jnp.ndarray): A 3×3 symmetric diffusion tensor.

    Returns:
        jnp.ndarray: A 3×3 matrix representing the gradient of the fractional anisotropy with respect to D.
    
    Notes:
        - The FA is expressed as a function of the normalized diffusion tensor R = D / trace(D).
        - The derivative is computed using the chain rule:
            dFA/dD = (dFA/dR) · (dR/dD),
          where dFA/dR is based on the trace of R², and dR/dD accounts for the trace normalization.
        - Assumes `fractional_anisotropy(D)` is defined and returns a scalar FA value.
    """
    trace_D = jnp.trace(D)
    R = D / trace_D
    R_squared = jnp.dot(R, R)
    trace_R_squared = jnp.trace(R_squared)
    FA = fractional_anisotropy(D)
    
    dFA_dTr_R2 = -FA / (2 * trace_R_squared**2)
    dTr_R2_dR = 2 * R
    dFA_dR = dFA_dTr_R2 * dTr_R2_dR

    I = jnp.eye(3)
    dR_dD = (I / trace_D) - (D / trace_D**2)

    dFA_dD = jnp.dot(dFA_dR, dR_dD)
    return dFA_dD

@jax.jit
def MD_from_tensor(D):
    return jnp.mean(jnp.trace(D))

@jax.jit
def dMD_dD():
    return jnp.diag(jnp.full(3, 1/3))

@jax.jit
def inverse_matrix(matrix):
    return jnp.linalg.pinv(matrix)


@partial(jax.jit, static_argnums = (8,9,10,11,12,13))
def output_generator_general(
    FA, phase_modulation, T1, T2, M, D, prep_FA_opt, prep_PM_opt,
    sampling=False, sampling_idx=0, sampling_rate=32,
    gradient_number=3, n_states=20,
    prepend_inversion_and_30deg_train=False):
    TE, TR, TI, deltak = 4, 15, 15, 1
    FpFmZ = jnp.zeros((3, n_states), dtype=jnp.complex128).at[2, 0].set(1.0)
    gradients = jnp.array([
        [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
        [-1., 0., 0.], [0., -1., 0.], [0., 0., -1.],
        [1., 1., 0.], [1., 0., 1.], [0., 1., 1.]
    ])[:gradient_number]
    FA_type = FA.dtype

    def scan_fun(carry, inputs):
        FpFmZ, g = carry
        alpha, phi = inputs

        FpFmZ = epg_rf(FpFmZ, alpha=alpha, phi=phi)
        FpFmZ = epg_grelax_generalized(FpFmZ, T1, T2, TE, D_tensor=D, M=M, gradient_dir=g, kstrength=100, deltak=deltak, Gon=0, noadd=1)
        transversal_real = jnp.real(FpFmZ[0, 0])
        transversal_imag = jnp.imag(FpFmZ[0, 0])
        FpFmZ = epg_grelax_generalized(FpFmZ, T1, T2, TR - TE, D_tensor=D, M=M, gradient_dir=g, kstrength=100, deltak=deltak, Gon=1, noadd=1)
        return (FpFmZ, g), (transversal_real, transversal_imag)
    
    
    optimizing_prep_module = True
    if optimizing_prep_module:
        prep_module_FA = FA
        prep_module_PM = phase_modulation
    else:
        prep_module_FA = prep_FA_opt
        prep_module_PM = prep_PM_opt
        
    
    
    if prepend_inversion_and_30deg_train:
        inversion_FA = jnp.array([jnp.pi])
        inversion_phase = jnp.array([0.0])
        full_prep_FA = jnp.array(jnp.concatenate([inversion_FA, prep_module_FA]), dtype=jnp.complex128)
        full_prep_phase = jnp.array(jnp.concatenate([inversion_phase, prep_module_PM]), dtype=jnp.complex128)
        prep_inputs = (full_prep_FA, full_prep_phase)

        g_prep = gradients[0]
        gradients = gradients[1:]

        (FpFmZ, _), (prep_real, prep_imag) = lax.scan(scan_fun, (FpFmZ, g_prep), prep_inputs)
        prep = jnp.stack([prep_real, prep_imag], axis=0)

    def process_gradient(g):
        FpFmZ_local = FpFmZ
        FpFmZ_local = epg_rf(FpFmZ_local, alpha=jnp.pi / 2, phi=0)
        FpFmZ_local = epg_grelax_generalized(FpFmZ_local, T1, T2, TE, D_tensor=D, M=M, gradient_dir=g, kstrength=300, deltak=1, Gon=1, noadd=1)
        FpFmZ_local = epg_rf(FpFmZ_local, alpha=jnp.pi, phi=0)
        FpFmZ_local = epg_grelax_generalized(FpFmZ_local, T1, T2, TE, D_tensor=D, M=M, gradient_dir=g, kstrength=300, deltak=1, Gon=1, noadd=1)
        FpFmZ_local = epg_rf(FpFmZ_local, alpha=-jnp.pi / 2, phi=0)

        inputs = (FA, phase_modulation)
        (FpFmZ_out, _), (real_main, imag_main) = lax.scan(scan_fun, (FpFmZ_local, g), inputs)

        return jnp.array([real_main, imag_main])

    results = lax.map(process_gradient, gradients)
    results = results.transpose(1, 0, 2).reshape(2, -1) 
    
    if prepend_inversion_and_30deg_train:
        results_processed = jnp.concatenate([prep, results], axis=1)
    else:
        results_processed = results
        
    if sampling:
        processed_signal = results_processed[:, sampling_idx::sampling_rate]
    else:
        processed_signal = results_processed
        
    return processed_signal




@partial(jax.jit, static_argnums = (8,9,10,11,12,13))
def output_plotter_general(FA, phase_modulation, T1, T2, M, D, prep_FA_opt, prep_PM_opt, sampling = False, sampling_idx = 0, sampling_rate = 32, gradient_number = 3, n_states = 20, prepend_inversion_and_30deg_train=False):
    raw_output = output_generator_general(FA, phase_modulation, T1, T2, M, D, prep_FA_opt, prep_PM_opt, sampling, sampling_idx, sampling_rate, gradient_number, n_states, prepend_inversion_and_30deg_train)
    
    signal = jnp.sqrt(raw_output[0] ** 2 + raw_output[1] ** 2)    
    return signal

@partial(jax.jit, static_argnums = (8,9,10,11,12,13))
def jacobian_calculator_general(FlipAngles, phase_modulation, T1, T2, M, D, prep_FA_opt, prep_PM_opt, sampling = False, sampling_idx = 0, sampling_rate = 32, gradient_number = 3, n_states = 20, prepend_inversion_and_30deg_train=False):
    """
    Compute the Jacobian matrix of the MR signal with respect to tissue parameters, including T1, T2, M0, FA, and MD.

    Parameters:
        FlipAngles (jnp.ndarray): Array of flip angles used in the sequence.
        phase_modulation (jnp.ndarray): Phase modulation applied across the sequence.
        T1 (float): Longitudinal relaxation time.
        T2 (float): Transverse relaxation time.
        M (float): Proton density or equilibrium magnetization.
        D (jnp.ndarray): 3×3 diffusion tensor.
        sampling (bool, optional): If True, output is subsampled using `sampling_idx`. Default is False.
        sampling_idx (int or array-like, optional): Indices used for subsampling if `sampling` is True. Default is 0.

    Returns:
        jnp.ndarray: Combined Jacobian vector concatenating partial derivatives of the output signal 
                     with respect to [T1, T2, M0, FA, MD].

    Notes:
        - Uses JAX's `jacobian` to compute derivatives of the MR signal model `output_generator_general`.
        - Derivatives with respect to FA and MD are computed via the chain rule using `dFA_dD` and `dMD_dD`.
        - The function assumes `inverse_matrix`, `dFA_dD`, and `dMD_dD` are defined and return appropriate tensors.
    """
    jacobian_scalars = jnp.array(jacobian(output_generator_general, argnums = [2, 3, 4])(FlipAngles, phase_modulation, T1, T2, M, D, prep_FA_opt, prep_PM_opt, sampling, sampling_idx, sampling_rate, gradient_number, n_states, prepend_inversion_and_30deg_train))
    jacobian_diffusion = jnp.array(jacobian(output_generator_general, argnums = [5])(FlipAngles, phase_modulation, T1, T2, M, D, prep_FA_opt, prep_PM_opt, sampling, sampling_idx, sampling_rate, gradient_number, n_states, prepend_inversion_and_30deg_train))

    jacobian_FA = jnp.sum(jacobian_diffusion * inverse_matrix(dFA_dD(D)), axis = (-2, -1))
    jacobian_MD = jnp.sum(jacobian_diffusion * inverse_matrix(dMD_dD()), axis = (-2, -1))

    jacobian_combined = jnp.concatenate([jacobian_scalars, jacobian_FA, jacobian_MD], axis = 0)
    return jacobian_combined

@partial(jax.jit, static_argnums = (8,9,10,11,12,13,14))
def fisher_information_general(FlipAngle, phase_modulation, T1, T2, M, D, prep_FA_opt, prep_PM_opt, sampling = False, sampling_idx= 0, sampling_rate = 32, gradient_number = 3, n_states = 20, prepend_inversion_and_30deg_train=False, sigma = 10**(-1.65)):
    T1 = jnp.asarray(T1, dtype=jnp.float64)
    T2 = jnp.asarray(T2, dtype=jnp.float64)
    M = jnp.asarray(M, dtype=jnp.float64)
    D = jnp.asarray(D, dtype=jnp.float64)    

    J = jacobian_calculator_general(FlipAngle, phase_modulation, T1, T2, M, D, prep_FA_opt, prep_PM_opt, sampling, sampling_idx, sampling_rate, gradient_number, n_states, prepend_inversion_and_30deg_train)
    J = J.reshape(J.shape[0], -1)
    FIM = J @ J.T

    return 1 / (sigma**2) * FIM

import numpy as np


@partial(jax.jit, static_argnums = 4)
def quadratic_PM_function_malleable(FA_array, fraction, slope1, slope2, discontinuity_type='nulling'):
    """
    Generate a malleable quadratic phase modulation (PM) function based on a flip angle (FA) array.

    Parameters:
        FA_array (array-like): Input array of flip angles, used to determine the length of the output array.
        fraction (float): Fractional index (0–1) at which the modulation changes behavior.
        slope1 (float): Quadratic slope applied before the split index.
        slope2 (float): Quadratic slope applied after the split index.
        discontinuity_type (str, optional): Type of transition at the split index; 
            'nulling' introduces a zero discontinuity point, 
            'continuing' ensures a continuous transition. Default is 'nulling'.

    Returns:
        jnp.ndarray: Phase modulation array of the same length as FA_array, constructed from two quadratic segments.
    """
    N = len(FA_array)
    phase_modulation = jnp.zeros(N)
    split_index = jnp.floor(N * fraction).astype(int)

    if discontinuity_type == 'nulling':
        phase_modulation = jnp.where(
            jnp.arange(N) < split_index,
            slope1 * jnp.square(jnp.arange(N)),
            jnp.where(
                jnp.arange(N) > split_index,
                slope2 * jnp.square(jnp.arange(N) - split_index),
                0.0
            )
        )
    elif discontinuity_type == 'continuing':
        phase_modulation = jnp.where(
            jnp.arange(N) < split_index,
            slope1 * jnp.square(jnp.arange(N)),
            slope2 * jnp.square(jnp.arange(N) - split_index)
        )

    return phase_modulation



@partial(jax.jit, static_argnums = (11,12,13,14,15,16,17))
def lb_in_param_holistic_general(coefficients, eval_points, knots, params_scalars_list, params_difs_list, W_mtrx, W_mtrx2, C0, C1, prep_FA_opt, prep_PM_opt, sampling = False, sampling_idx = 0, sampling_rate = 32, gradient_number = 3, n_states = 20, prepend_inversion_and_30deg_train=False,  method = 'quadratic'):
    if method == 'quadratic': 
        second_derivative = bspline(eval_points, knots, coefficients[len(coefficients)//2:])
        second_integral = jnp.cumsum(jnp.cumsum(second_derivative))
        phase_modulation = second_integral + C1 * eval_points + C0
    elif method == 'free form':
        phase_modulation = bspline(eval_points, knots, coefficients[len(coefficients)//2:]) 
    elif method == 'no phase modulation':
        phase_modulation = jnp.zeros(len(eval_points))
    elif method == 'quadratic malleable':
        f, s1, s2 = coefficients[-3:]
        phase_modulation = quadratic_PM_function_malleable(eval_points, f, s1, s2, discontinuity_type = 'nulling')
    else:
        raise ValueError("Enter a valid method of phase modulation optimization: 'quadratic', 'free form', 'no phase modulation', or 'quadratic malleable'")


    if method == 'quadratic malleable':
         FA = bspline(eval_points, knots, coefficients[:-3])
    elif method == 'no phase modulation':
        FA = bspline(eval_points, knots, coefficients)
    else:    
        FA = bspline(eval_points, knots, coefficients[:len(coefficients)//2])
    
    params_FA = jnp.array([fractional_anisotropy(x) for x in params_difs_list])
    params_MD = jnp.array([MD_from_tensor(x) for x in params_difs_list])
    params_combined = jnp.concatenate([params_scalars_list, params_FA[:, None], params_MD[:, None]], axis = 1)

    def compute_crlb(params, params_scalars, params_difs, weight):
        T1, T2, M = params_scalars
        D = params_difs

        FIM = fisher_information_general(FA, phase_modulation, T1, T2, M, D, prep_FA_opt, prep_PM_opt, sampling, sampling_idx, sampling_rate, gradient_number, n_states, prepend_inversion_and_30deg_train)
        crlb = W_mtrx[:3] * jnp.sqrt(jnp.diag(jnp.linalg.inv(FIM[:-2, :-2])) / (params[:3] ** 2))
        return crlb * weight
    
    crlb_array = jax.vmap(compute_crlb)(params_combined, params_scalars_list, params_difs_list, W_mtrx2)
    return jnp.sum(crlb_array, axis = 0)

@partial(jax.jit, static_argnums = (11,12,13,14,15,16,17,18))
def objective_holistic_general(coefficients, eval_points, knots, params_scalars_list, params_difs_list, W_mtrx, W_mtrx2, C0, C1, prep_FA_opt, prep_PM_opt, sampling = False, sampling_idx = 0, sampling_rate = 32, gradient_number = 3, n_states = 20, prepend_inversion_and_30deg_train=False, objective_method = 'L2', method = 'quadratic'):   
    if method == 'quadratic':        
        second_derivative = bspline(eval_points, knots, coefficients[len(coefficients)//2:])
        second_integral = jnp.cumsum(jnp.cumsum(second_derivative))
        phase_modulation = second_integral + C1 * eval_points + C0
    elif method == 'free form':
        phase_modulation = bspline(eval_points, knots, coefficients[len(coefficients)//2:]) 
    elif method == 'no phase modulation':
        phase_modulation = jnp.zeros(len(eval_points))
    elif method == 'quadratic malleable':
        f, s1, s2 = coefficients[-3:]
        phase_modulation = quadratic_PM_function_malleable(eval_points, f, s1, s2, discontinuity_type = 'nulling')
    else:
        raise ValueError("EPG.OBJECTIVE_HOLISTIC_GENERAL ERROR: Enter a valid method of phase modulation optimization: 'quadratic', 'free form', 'no phase modulation', or 'quadratic malleable'")

    if method == 'quadratic malleable':
         FA = bspline(eval_points, knots, coefficients[:-3])
    elif method == 'no phase modulation':
        FA = bspline(eval_points, knots, coefficients)
    else:    
        FA = bspline(eval_points, knots, coefficients[:len(coefficients)//2])  
    
    
    
    if objective_method == 'L2':
        second_order = 2
    elif objective_method == 'L1':
        second_order = 1
    else:
        raise ValueError("EPG.OBJECTIVE_HOLISTIC_GENERAL ERROR: Enter a valid method for the CRLB summing: 'L2' or 'L1' ")
    params_FA = jnp.array([fractional_anisotropy(x) for x in params_difs_list])
    params_MD = jnp.array([MD_from_tensor(x) for x in params_difs_list])

    params_FA_expanded = params_FA[:, None]
    params_MD_expanded = params_MD[:, None]
    params_combined = jnp.concatenate([params_scalars_list, params_FA_expanded, params_MD_expanded], axis=1)

    params_scalars_list = jnp.array(params_scalars_list)  
    params_difs_list = jnp.array(params_difs_list)

    def scan_body(total_crlb_accum, i):
        i = i.astype(int)
        params = params_combined[i]
        T1, T2, M = params_scalars_list[i] 
        D = params_difs_list[i]  
        FIM = fisher_information_general(FA, phase_modulation, T1, T2, M, D, prep_FA_opt, prep_PM_opt, sampling, sampling_idx, sampling_rate, gradient_number, n_states, prepend_inversion_and_30deg_train)
        
        crlb = jnp.sum(W_mtrx[:3] * (jnp.sqrt(jnp.diag(jnp.linalg.inv(FIM[:-2, :-2])) / (params[:3] ** 2)))**second_order)
        weighted_crlb = crlb * W_mtrx2[i]
        
        return total_crlb_accum + weighted_crlb, None 

    total_crlb, _ = lax.scan(scan_body, 0.0, jnp.arange(len(params_combined)))  
    
    return total_crlb









  



#        ██████      ██████  ██ ███████ ███████ ██    ██ ███████ ██  ██████  ███    ██     ██████  ██ ██████  ███████  ██████ ████████ ██  ██████  ███    ██  ██████ ███████ 
#       ██           ██   ██ ██ ██      ██      ██    ██ ██      ██ ██    ██ ████   ██     ██   ██ ██ ██   ██ ██      ██         ██    ██ ██    ██ ████   ██ ██      ██      
#       ███████      ██   ██ ██ █████   █████   ██    ██ ███████ ██ ██    ██ ██ ██  ██     ██   ██ ██ ██████  █████   ██         ██    ██ ██    ██ ██ ██  ██ ██      ███████ 
#       ██    ██     ██   ██ ██ ██      ██      ██    ██      ██ ██ ██    ██ ██  ██ ██     ██   ██ ██ ██   ██ ██      ██         ██    ██ ██    ██ ██  ██ ██ ██           ██ 
#        ██████      ██████  ██ ██      ██       ██████  ███████ ██  ██████  ██   ████     ██████  ██ ██   ██ ███████  ██████    ██    ██  ██████  ██   ████  ██████ ███████ 
                                                                                                                                                                     





def jacobian_calculator_general_3directions(FA, phase_modulation, T1, T2, M, D):
    jacobian_scalars = jnp.array(jacobian(output_generator_general, argnums=[2, 3, 4])(FA, phase_modulation, T1, T2, M, D))
    jacobian_diffusion = jnp.array(jacobian(output_generator_general, argnums=[5])(FA, phase_modulation, T1, T2, M, D))

    jacobian_diffusion = jacobian_diffusion.squeeze(0).transpose(2, 3, 0, 1).reshape(9, 2, -1)
    selected_indices = jnp.array([0, 4, 8])
    jacobian_diffusion = jacobian_diffusion[selected_indices]

    jacobian_combined = jnp.concatenate([jacobian_scalars, jacobian_diffusion], axis=0)
    print(jacobian_combined.shape)
    return jacobian_combined


def fisher_information_general_3directions(FA, phase_modulation, T1, T2, M, D, sigma=10**(-1.65)):
    T1 = jnp.asarray(T1, dtype=jnp.float64)
    T2 = jnp.asarray(T2, dtype=jnp.float64)
    M = jnp.asarray(M, dtype=jnp.float64)
    D = jnp.asarray(D, dtype=jnp.float64)    

    J = jacobian_calculator_general(FA, phase_modulation, T1, T2, M, D)
    
    J = J.reshape(J.shape[0], -1)
    FIM = J @ J.T

    return 1 / (sigma**2) * FIM


def test_diffusion_preparation(N_states, x, T1=1000, T2=100, TE=10):
    """
    Simule une préparation de diffusion et affiche les normes des états EPG.

    Paramètres :
    ------------
    N_states : int
        Nombre d'états EPG simulés (ordre max + 1)
    T1, T2 : float
        Constantes de relaxation (secondes)
    TE : float
        Temps effectif d'écho entre gradients (secondes)
    """
    print(f"\n=== Test with {N_states} EPG states ===")
    
    FpFmZ = jnp.zeros((3, N_states), dtype=jnp.complex64)
    FpFmZ = FpFmZ.at[2, 0].set(1.0) 

    
    D = jnp.diag(jnp.array([0.001, 0.0005, 0.0002]))  
    g = jnp.array([0, 1, 0.0])                  
    M = 1                                   

    
    FpFmZ = epg.epg_rf(FpFmZ, alpha=jnp.pi / 2, phi=0.0)
    norms = jnp.linalg.norm(FpFmZ, axis=0)
    print("Normes par ordre intermediate 1 before diffusion :", norms[:5])
    FpFmZ = epg.epg_grelax_generalized(FpFmZ, T1, T2, TE, D_tensor=D, M=M,
                                   gradient_dir = g, kstrength = x, deltak=1, Gon=1, noadd=1)
    norms = jnp.linalg.norm(FpFmZ, axis=0)
    print("Normes par ordre intermediate :", norms[:5])
    FpFmZ = epg.epg_rf(FpFmZ, alpha=jnp.pi, phi=0.0)
    FpFmZ = epg.epg_grelax_generalized(FpFmZ, T1, T2, TE, D_tensor=D, M=M,
                                   gradient_dir = g, kstrength = x, deltak=1, Gon=1, noadd=1)

    FpFmZ = epg.epg_rf(FpFmZ, alpha=-jnp.pi / 2, phi=0.0)

    energy_out = jnp.sum(jnp.abs(FpFmZ)**2).item()
    norms = jnp.linalg.norm(FpFmZ, axis=0)

    
    print(f"Énergie finale : {energy_out:.4f}")
    print("Normes par ordre :", norms[:5])
    
    
'''    
    
    def compute_scalar_bvalue(T_ms, gradient_dir, kstrength=450, deltak=1):
    """
    Computes the scalar b-value from the EPG gradient simulation by evaluating 
    the trace of the total b-tensor for F-state 0.

    Parameters:
        T_ms : float
            Time interval in milliseconds.
        gradient_dir : array_like, shape (3,)
            Unit gradient direction vector.
        kstrength : float
            Strength of gradient in cycles/mm.
        deltak : int
            Step size for gradient increment (usually 1).

    Returns:
        b_scalar : float
            Scalar b-value in s/mm².
    """
    # Convert T from ms to seconds
    T = T_ms / 1000.0

    # Use number of F-states = 100 (sufficient for gradient spectrum)
    N = 150
    Findex = jnp.arange(N, dtype=jnp.float32)

    # Generate k-space vectors
    k1 = jnp.outer(gradient_dir, Findex * kstrength)              # shape (3, N)
    k2 = jnp.outer(gradient_dir, (Findex + deltak) * kstrength)   # shape (3, N)

    # Compute b-tensor components
    b_s_L = jnp.einsum('in,jn->ijn', k1, k1) * T
    cross1 = jnp.einsum('in,jn->ijn', k1, k2)
    cross2 = jnp.einsum('in,jn->ijn', k1, k2)
    cross_term = 0.5 * (cross1 - cross2) * T
    dk = k2 - k1
    pure_term = jnp.einsum('in,jn->ijn', dk, dk) * (T / 3.0)

    # Total b-tensor for each F-state
    b_s_T = b_s_L + cross_term + pure_term  # shape (3, 3, N)

    # Extract scalar b-value for F-state 0 (order 0)
    b_scalar = jnp.trace(b_s_T[:, :, 0])    # scalar in s/mm²

    return b_scalar


# === Example usage ===
T1 = 1000.0  # ms (not needed here)
T2 = 80.0    # ms (not needed here)
T = 15.0     # ms
M = 1.0      # (not needed here)

# Diffusion tensor (not needed to compute b-value, only for signal decay)
D = 1e-3 * jnp.array([[0.89, 0.1, 0.1],
                      [0.1, 0.2, 0.1],
                      [0.1, 0.1, 0.2]])

# Gradient direction (example)
gradient_dir = jnp.array([1.0, 0.0, 0.0])  # Must be unit vector

# Compute scalar b-value
b_val = compute_scalar_bvalue(T_ms=T, gradient_dir=gradient_dir)
print("Scalar b-value [s/mm²]:", b_val)

'''



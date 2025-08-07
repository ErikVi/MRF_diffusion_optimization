import timeit
import importlib
import itertools
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad
import inspect
from jax import jacobian, jacfwd, jacrev
from scipy.optimize  import minimize
jax.config.update("jax_enable_x64", True)
from functools import partial
from scipy.interpolate import BSpline
from functools import partial
import EPG_blocks_jaxcode as epg 
importlib.reload(epg)
import matplotlib.pyplot as plt




print("Loading FA array...")
FA_array = jnp.array(np.load("fa_array_initial.npy"))[:170]
print("FA array loaded.")

num_points = len(FA_array)
num_knots = 50
degree = 3
knots = jnp.concatenate((jnp.zeros(degree),
                         jnp.linspace(0.0, num_points + 0.01, num_knots + 1 - 2 * degree),
                         jnp.ones(degree) * (num_points + 0.01)))
eval_points=jnp.linspace(0,num_points-1,num_points)


params_scalars = jnp.array([[1000, 70,  0.7],
                            [1000, 70,  0.7],
                            [1500, 100,  0.8]])

params_difs = 1e-3 * jnp.array(
    [[[0.89, 0.1, 0.1],  
    [0.1, 0.2, 0.1], 
    [0.1, 0.1, 0.2]],

    [[0.7, 0.1, 0.1],  
    [0.1, 0.3, 0.1], 
    [0.1, 0.1, 0.3]],

    [[0.75, 0.1, 0.1],  
    [0.1, 0.4, 0.1], 
    [0.1, 0.1, 0.4]]])


initial_coefficients_FA = epg.bspline_coefficients(FA_array, knots, degree)


jax.config.update("jax_enable_x64", True)

crlb_results = {}


j_values = jnp.linspace(0.01, 1.0, 400)
phase_methods = ['quadratic', 'linear', 'sinusoidal', 'alternating']

print("UNOPTIMIZED CRLBS, no phase modulation:")
no_phase = epg.lb_in_param_holistic_general(
    jnp.concatenate((initial_coefficients_FA, jnp.zeros(len(FA_array)))),
    eval_points,
    knots,
    params_scalars,
    params_difs,
    W_mtrx = jnp.array([1, 1, 1, 1, 1]),
    W_mtrx2 = jnp.ones(len(params_scalars)) / len(params_scalars),
    C0 = 0,
    C1 = 0,
    prep_FA_opt=jnp.zeros(3), 
    prep_PM_opt=jnp.zeros(3),
    sampling = False,
    sampling_idx = 0,
    sampling_rate = 32,
    gradient_number = 9,
    n_states = 5,
    prepend_inversion_and_30deg_train=False,
    method = 'free form'
)


print(f"[T1, T2, M, FA, MD] = {no_phase}\n")

for method in phase_methods:
    crlb_T1, crlb_T2, crlb_M, crlb_FA, crlb_MD = [], [], [], [], []
    sum_crlb = []

    for j in j_values:
        phase_at_j = epg.phase_modulation_generator(FA_array, fraction = j, method=method)
        phase_coeffs = epg.bspline_coefficients(phase_at_j, knots, degree)
        coefficients = jnp.concatenate((initial_coefficients_FA, phase_coeffs))

        crlb = epg.lb_in_param_holistic_general(
            coefficients = coefficients,
            eval_points = eval_points,
            knots = knots,
            params_scalars_list=params_scalars,
            params_difs_list=params_difs,
            W_mtrx=jnp.array([1, 1, 1, 1, 1]),
            W_mtrx2=jnp.ones(len(params_scalars)) / len(params_scalars),
            C0 = 0,
            C1 = 0,
            prep_FA_opt=jnp.zeros(3), 
            prep_PM_opt=jnp.zeros(3),          
            sampling = False,
            sampling_idx = 0,
            sampling_rate = 32,
            gradient_number = 9,
            n_states = 5,
            prepend_inversion_and_30deg_train=False,
            method = 'free form'
        )

                      
        crlb_T1.append(crlb[0])
        crlb_T2.append(crlb[1])
        crlb_M.append(crlb[2])
        crlb_FA.append(crlb[3])
        crlb_MD.append(crlb[4])
        sum_crlb.append(jnp.sum(crlb))

    crlb_results[method] = {
        "T1": crlb_T1,
        "T2": crlb_T2,
        "M": crlb_M,
        "FA": crlb_FA,
        "MD": crlb_MD,
        "sum": sum_crlb,
        "fractions": j_values
    }

colors = np.array(["red","orange", "green", "blue"])


best_val = float("inf")
best_method = None
best_idx = -1

for method, res in crlb_results.items():
    crlb_sum = jnp.array(res["sum"])
    idx = int(jnp.argmin(crlb_sum))
    val = crlb_sum[idx]
    if val < best_val:
        best_val = val
        best_method = method
        best_idx = idx

best_j = crlb_results[best_method]["fractions"][best_idx]

labels = ['T1', 'T2', 'M', 'FA', 'MD']
num_metrics = len(labels)
fig, axes = plt.subplots(3, 2, figsize=(22, 16))

for i, label in enumerate(labels):
    ax = axes.flat[i]
    for m_idx, (method, res) in enumerate(crlb_results.items()):
        y = res[label]
        j_vals = res["fractions"]
        color = colors[m_idx]
        ax.plot(j_vals, y, label=method, color=color, linewidth=4)

    if no_phase is not None:
        ax.axhline(no_phase[i], color='purple', linestyle='--', linewidth=4, label='No phase')

    ax.set_title(f'nCRLB for {label}')
    ax.set_xlabel("Fraction")
    ax.set_ylabel("CRLB")
    ax.grid(True)
    ax.legend()

ax_total = axes[2, 1]
for m_idx, (method, res) in enumerate(crlb_results.items()):
    y = res["sum"]
    j_vals = res["fractions"]
    color = colors[m_idx]
    ax_total.plot(j_vals, y, label=method, color=color, linewidth=4)
    
ax_total.axhline(jnp.sum(no_phase), color='purple', linestyle='--', linewidth=4, label='No phase')
ax_total.scatter(best_j, best_val, color='black', zorder=5, label=f"Min nCRLB ({best_method}, j={best_j:.2f})")

ax_total.set_title("Sum of All nCRLBs")
ax_total.set_xlabel("Fraction")

ax_total.set_ylabel("CRLB Sum")
ax_total.grid(True)
ax_total.legend()

plt.tight_layout()
plt.savefig(f"crlb_summary.png", dpi=300)
plt.close()
for method in phase_methods:
    plt.figure(figsize=(11, 8))

    best_phase = None
    best_label = None
    best_color = None
    
    for j_idx, j in enumerate(j_values):
        grey_value = 0.8 * (j_idx / (len(j_values) - 1))
        clr = 'red' if method == best_method and j == best_j else (grey_value, grey_value, grey_value)

        phase = epg.phase_modulation_generator(FA_array, fraction=j, method = method)
        label = f"{'Best: ' if method == best_method and j == best_j else ''}{method}, j={j:.2f}" if j_idx % 10 == 0 else None
        
        if method == best_method and j == best_j:
            best_phase = phase
            best_label = label
            best_color = clr
        else:
            plt.plot(phase, label=label, color=clr)

    if best_phase is not None:
        plt.plot(best_phase, label = best_label, color = best_color, linewidth=2)  

    plt.title(f'Phase Modulation Schemes - {method}')
    plt.xlabel('Time Index')
    plt.ylabel('Phase (radians)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"crlb_summary_{method}.png", dpi=300)
    plt.close()
    
    
    print('Best method:', best_method)
    print('Best j:', best_j)

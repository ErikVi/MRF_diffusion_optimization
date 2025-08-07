import time
import os
import h5py
import numpy as np
import pandas as pd
import jax
from jax import lax
import jax.numpy as jnp
import matplotlib.cm as cm
import importlib
from jax import grad, jacobian
from scipy.optimize import minimize
from functools import partial
from scipy.interpolate import BSpline
import socket
import importlib
import EPG_blocks_jaxcode as epg 
importlib.reload(epg)
import matplotlib.pyplot as plt
import jax.profiler
import tracemalloc





length = 300              
start_time = time()
method = 'quadratic_malleable' 

print("Loading FA array...")
FA_array = jnp.array(np.load("fa_array_initial.npy"))[:length]
print("FA array loaded.")


num_points = len(FA_array)
num_knots = 20
degree = 3
knots = jnp.concatenate((jnp.zeros(degree),
                        jnp.linspace(0.0, num_points + 0.01, num_knots + 1 - 2 * degree),
                        jnp.ones(degree) * (num_points + 0.01)))
eval_points=jnp.linspace(0,num_points-1,num_points)

params_difs = 1e-3 * jnp.array(
    [[[0.89, 0.1, 0.1],  
    [0.1, 0.2, 0.1], 
    [0.1, 0.1, 0.2]],
    [[0.70, 0.1, 0.1],  
    [0.1, 0.3, 0.1], 
    [0.1, 0.1, 0.3]],
    [[0.75, 0.1, 0.1],  
    [0.1, 0.4, 0.1], 
    [0.1, 0.1, 0.4]]])

initial_coefficients_FA = epg.bspline_coefficients(FA_array, knots, degree)



D_tensor = 1e-3 * jnp.array(
[[0.89, 0.1, 0.1],
[0.1, 0.2, 0.1], 
[0.1, 0.1, 0.2]])

_ = epg.bspline(eval_points, knots, initial_coefficients_FA)
_ = epg.bspline_vectorized(eval_points, knots, initial_coefficients_FA)
_ = epg.bspline_vmap(eval_points, knots, initial_coefficients_FA)

def benchmark(func, name):
    
    start = time.time()
    N = 10000
    for _ in range(N):
        result = func(eval_points, knots, initial_coefficients_FA)
        jax.block_until_ready(result)
    duration = time.time() - start

    tracemalloc.start()
    result = func(eval_points, knots, initial_coefficients_FA)
    jax.block_until_ready(result)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"{name}:")
    print(f"  Temps moyen par appel : {duration / N * 1e6:.10f} µs")
    print(f"  Mémoire actuelle : {current / 1e6:.10f} MB, Mémoire pic : {peak / 1e6:.10f} MB\n")
    return result

fa_orig = benchmark(epg.bspline, "bspline (original)")
fa_vec = benchmark(epg.bspline_vectorized, "bspline_vectorized")
fa_vmap = benchmark(epg.bspline_vmap, "bspline_vmap")

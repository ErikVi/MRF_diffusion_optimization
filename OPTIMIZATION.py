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
from time import time
import socket
import importlib
import EPG_blocks_jaxcode as epg 
importlib.reload(epg)
import matplotlib.pyplot as plt
import jax.profiler

import matplotlib.patches as mpatches


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


'''
modules = [
    'os',
    'h5py',
    'numpy',
    'pandas',
    'jax',
    'matplotlib',
    'scipy',
    'functools',  
    'time',       
    'socket'      
]

for mod in modules:
    try:
        module = importlib.import_module(mod)
        version = getattr(module, '__version__', 'builtin or no __version__ attribute')
        print(f"{mod}: {version}")
    except ImportError:
        print(f"{mod}: not installed")
'''



jax.config.update("jax_enable_x64", True)




start_time = time()

summary_ncrlb_initial = {}
summary_ncrlb_optimized = {}
summary_colors = ['red', 'orange', 'green', 'blue', 'cyan', 'pink', 'purple']
summary_labels = []





for prep_type in jnp.array([[1,1]]):
  for method in ['free form', 'no phase modulation', 'quadratic', 'quadratic malleable']:  
    optimized_fa_sequences = [] 
    optimized_pm_sequences = []
    knots_list = []    
    for num_knots in jnp.array([50]):
      for length in jnp.array([400]):
          for objective_type in ["L1"]:          
              for n_st in jnp.array([5]):
                  length = int(length)
                  prep = bool(prep_type[1])
                  gr_num = int(prep_type[0])
                  n_st = int(n_st)
                  num_knots = int(num_knots)    
                  method = str(method)
                  objective_type = str(objective_type)
                   
                  print("////////////////////////////")
      
                  FA_array = jnp.array(np.load("fa_array_initial.npy"))[:length]
      
      
      
                  num_points = len(FA_array)
    
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
                      [[0.70, 0.1, 0.1],  
                      [0.1, 0.3, 0.1], 
                      [0.1, 0.1, 0.3]],
                      [[0.75, 0.1, 0.1],  
                      [0.1, 0.4, 0.1], 
                      [0.1, 0.1, 0.4]]])
                  initial_coefficients_FA = epg.bspline_coefficients(FA_array, knots, degree)
                  best_j = 0.18
                  best_method = 'quadratic'
      
                  minimum_angle = jnp.pi/18
      
                  opt_min = 1
                  if minimum_angle == 0:
                      opt_min = 0
                  else:
                      opt_min = 10
      
      
                  print('Using minimum angle:', minimum_angle)
      
                  if method == 'no phase modulation':
                      initial_phase = jnp.zeros(len(eval_points))
                      initial_coefficients =  initial_coefficients_FA
                      C1, C0 = 0, 0
                      
                      constr_fun = lambda x:  epg.bspline(eval_points, knots, x) - minimum_angle
                      constr_fun_bis = lambda x: jnp.pi/3 -  epg.bspline(eval_points, knots, x)
                      constr_fun_bis_bis = lambda x: 0.02 - jnp.abs(jnp.diff(epg.bspline(eval_points, knots, x)))
                      
                          
                  elif method == 'free form':
                      initial_phase =  epg.phase_modulation_generator(FA_array, best_j, best_method)
                      initial_coefficients_PM =  epg.bspline_coefficients(initial_phase, knots, degree)
                      initial_coefficients = jnp.concatenate((initial_coefficients_FA, initial_coefficients_PM))
                      C1, C0 = 0, 0
      
                      constr_fun = lambda x:  epg.bspline(eval_points, knots, x[:len(x)//2]) - minimum_angle
                      constr_fun_bis = lambda x: jnp.pi/3 -  epg.bspline(eval_points, knots, x[:len(x)//2])
                      constr_fun_bis_bis = lambda x: 0.02 - jnp.abs(jnp.diff(epg.bspline(eval_points, knots, x[:len(x)//2])))
      
                  elif method == 'quadratic':
                      initial_phase =  epg.phase_modulation_generator(FA_array, best_j, best_method)
                      second_der = jnp.diff(jnp.diff(initial_phase))
                      initial_coefficients_PM =  epg.bspline_coefficients(second_der, knots, degree)
                      initial_coefficients = jnp.concatenate((initial_coefficients_FA, initial_coefficients_PM))
                      approx_second_der =  epg.bspline(eval_points, knots, initial_coefficients_PM)        
                      first_integral = jnp.cumsum(approx_second_der)
                      second_integral = jnp.cumsum(first_integral)
                      n = jnp.arange(len(second_integral))
                      A = jnp.stack([n, jnp.ones_like(n)], axis=1)
                      residual = initial_phase - second_integral
                      C1, C0 = jnp.linalg.lstsq(A, residual, rcond=None)[0]
      
                      constr_fun = lambda x:  epg.bspline(eval_points, knots, x[:len(x)//2]) - minimum_angle
                      constr_fun_bis = lambda x: jnp.pi/3 -  epg.bspline(eval_points, knots, x[:len(x)//2])
                      constr_fun_bis_bis = lambda x: 0.02 - jnp.abs(jnp.diff(epg.bspline(eval_points, knots, x[:len(x)//2])))
      
                  elif method == 'quadratic malleable':    
                      initial_coefficients_PM = jnp.array([0.3, 0.04, -0.030])
                      initial_phase = epg.quadratic_PM_function_malleable(eval_points, initial_coefficients_PM[0], initial_coefficients_PM[1], initial_coefficients_PM[2], discontinuity_type = 'nulling')
                      initial_coefficients = jnp.concatenate((initial_coefficients_FA, initial_coefficients_PM))
                      C1, C0 = 0, 0
                      constr_fun = lambda x: epg.bspline(eval_points, knots, x[:-3]) - minimum_angle
                      constr_fun_bis = lambda x: jnp.pi/3 - epg.bspline(eval_points, knots, x[:-3])
                      constr_fun_bis_bis = lambda x: 0.02 - jnp.abs(jnp.diff(epg.bspline(eval_points, knots, x[:-3])))
                      constr_fun_bis_bis_bis = lambda x: x[-3] - 0.2
                      constr_fun_bis_bis_bis_bis = lambda x: 0.8 - x[-3]
                      constr_fun_bis_bis_bis_bis_bis = lambda x: x[-2] - 0.01
                      constr_fun_bis_bis_bis_bis_bis_bis = lambda x: -1 * x[-1] - 0.01
                  else:
                      raise ValueError(f"Unknown method: {method}")  
      
      
                  if method == 'quadratic malleable':
                      constr = [
                      {'type': 'ineq', 'fun': constr_fun, 'jac': jacobian(constr_fun)},
                      {'type': 'ineq', 'fun': constr_fun_bis},
                      {'type': 'ineq', 'fun': constr_fun_bis_bis},
                      {'type': 'ineq', 'fun': constr_fun_bis_bis_bis},
                      {'type': 'ineq', 'fun': constr_fun_bis_bis_bis_bis},
                      {'type': 'ineq', 'fun': constr_fun_bis_bis_bis_bis_bis},
                      {'type': 'ineq', 'fun': constr_fun_bis_bis_bis_bis_bis_bis},]
                  else:
                      constr = [
                      {'type': 'ineq', 'fun': constr_fun, 'jac': jacobian(constr_fun)},
                      {'type': 'ineq', 'fun': constr_fun_bis},
                      {'type': 'ineq', 'fun': constr_fun_bis_bis},]
      
      
      
      
      
                  bounds = [(-1000, 1000)] * len(initial_coefficients)
      
                  iteration_count = 0
                  obj_values = []
                  lb_values = []
                  prev_obj_value = 0
                  W = jnp.array([1, 1, 1, 1, 1])  
                  W_2 = jnp.array([1,1,1])/3
                  tol = 1e-2
                  min_iterations = 1000
                  jac2 = jacobian(epg.objective_holistic_general, 0,)
                  optimization_method = 'SLSQP'
                  sampling_idx = 0
                  sampling = False
                  sampling_rate = 32
                  prep_PM = np.load('DIFFPREPARATION/pm_train_diffprep.npy')
                  prep_FA = np.load('DIFFPREPARATION/fa_train_diffprep.npy')
      
      
                  print("Optimization settings:")
                  print("Number of gradients:", gr_num)
                  print("Number of EPG states:", n_st)
                  print("FA array length:", len(FA_array))
                  print("Optimization method:", optimization_method)
                  print("Number of knots:", num_knots)
                  print("Degree:", degree)
                  print("Method:", method)
                  print("Number of points:", num_points)
                  print("FA array length:", len(FA_array))
                  print("Initial coefficients length:", len(initial_coefficients))
                  print("Objective function type:",  objective_type)
                  print("Inversion preparation:", prep)
      
      
                  if method == 'no_phase_modulation':
                      print('Phase left on 0')
                  else:
                      print("Initial phase non-zero according to:", method)
      
                  
                  
                  
                  if __name__ == "__main__":
                      
      
                      
              
                      
                      
                  
      
                      def callback_fn(xk):
                          global iteration_count 
                          global prev_obj_value    
                          iteration_count += 1
                          
                          obj_value = epg.objective_holistic_general(xk, eval_points = eval_points, knots = knots, params_scalars_list = params_scalars, params_difs_list = params_difs, W_mtrx = W, W_mtrx2 = W_2, C0=C0, C1=C1, prep_FA_opt=prep_FA, prep_PM_opt=prep_PM, sampling=sampling, sampling_idx=sampling_idx, sampling_rate = sampling_rate, gradient_number = gr_num, n_states = n_st, prepend_inversion_and_30deg_train=prep, objective_method = objective_type, method = method)  
                          
                          lb_value = epg.lb_in_param_holistic_general(xk, eval_points = eval_points, knots = knots, params_scalars_list = params_scalars, params_difs_list = params_difs, W_mtrx = W, W_mtrx2 = W_2, C0=C0, C1=C1, prep_FA_opt=prep_FA, prep_PM_opt=prep_PM, sampling=sampling, sampling_idx=sampling_idx, sampling_rate = sampling_rate, gradient_number = gr_num, n_states = n_st, prepend_inversion_and_30deg_train=prep, method = method)
                          
                          rel_dif = jnp.abs(obj_value - prev_obj_value)
                          if iteration_count > min_iterations and rel_dif < tol:
                              raise StopIteration   
                          prev_obj_value = obj_value
                          '''
                          with h5py.File("latest_iteration_data.h5", "a") as f:
                              if "iteration" in f:
                                  del f["iteration"]
                              if "cost_function_value" in f:
                                  del f["cost_function_value"]
                              if "lower_bound_value" in f:
                                  del f["lower_bound_value"]
                              f.create_dataset("iteration", data = iteration_count)
                              f.create_dataset("cost_function_value", data = obj_value)
                              f.create_dataset("lower_bound_value", data = lb_value)
                          '''
      
                      optimized_results = []
                      print("Callback function initialized.")
                      
                      iteration_count = 0  
                      obj_values = []
                      lb_values = []
      
                      result = minimize(epg.objective_holistic_general, initial_coefficients,
                                          args = (eval_points, knots, params_scalars, params_difs, W, W_2, C0, C1, prep_FA, prep_PM, sampling, sampling_idx, sampling_rate, gr_num, n_st, prep, objective_type, method),
                                          jac = jac2,
                                          method = optimization_method, bounds = bounds,
                                          constraints = constr, options = {'maxiter': min_iterations},
                                          callback = callback_fn)
      
      
                      optimized_coefficients = result.x
                      key = f"{objective_type} + {'InvPrep' if prep else 'NoPrep'} + {method}"

                      initial_ncrlb = epg.lb_in_param_holistic_general(initial_coefficients, eval_points, knots, params_scalars, params_difs,
                                                                       W, W_2, C0, C1, prep_FA, prep_PM, sampling, sampling_idx, sampling_rate,
                                                                       gr_num, n_st, prep, method)
                      
                      optimized_ncrlb = epg.lb_in_param_holistic_general(optimized_coefficients, eval_points, knots, params_scalars, params_difs,
                                                                         W, W_2, C0, C1, prep_FA, prep_PM, sampling, sampling_idx, sampling_rate,
                                                                         gr_num, n_st, prep, method)
                      
                      summary_ncrlb_initial[key] = np.array(initial_ncrlb)
                      summary_ncrlb_optimized[key] = np.array(optimized_ncrlb)
                      summary_labels.append(key)
                      print("Initial nCRLBS:",
                          initial_ncrlb)
                      print("Optimized nCRLBS:",
                          optimized_ncrlb)
      
      
      
      
                      print("Python test finished (running time: {0:.1f}s)".format(time() - start_time))
                      print("Optimization converged after {} iterations.".format(iteration_count))
                      print("Optimization settings:", "optimization method:", optimization_method, "number of knots:", num_knots, "degree:", degree, "method:", method)
                      print("Final objective function value:", result.fun)
                      
                  
                  jax.profiler.save_device_memory_profile("fjkegfgzeuifzgeifgzuiefgze.prof")
      
      
      
      
      
      
                  optimized_fa_train = epg.bspline(eval_points, knots, optimized_coefficients[:len(initial_coefficients_FA)])            
                  
                  
                  
                  
                  if method == 'quadratic': 
                      second_derivative = epg.bspline(eval_points, knots, optimized_coefficients[len(initial_coefficients_FA):])
                      second_integral = jnp.cumsum(jnp.cumsum(second_derivative))
                      optimized_phase_modulation = second_integral + C1 * eval_points + C0
                  elif method == 'free form':
                      optimized_phase_modulation = epg.bspline(eval_points, knots, optimized_coefficients[len(initial_coefficients_FA):]) 
                  elif method == 'no phase modulation':
                      optimized_phase_modulation = jnp.zeros(len(eval_points))
                  elif method == 'quadratic malleable':
                      f, s1, s2 = optimized_coefficients[len(initial_coefficients_FA):]
                      optimized_phase_modulation = epg.quadratic_PM_function_malleable(eval_points, f, s1, s2, discontinuity_type = 'nulling')
                  
                  
                  optimized_fa_sequences.append(optimized_fa_train)
                  optimized_pm_sequences.append(optimized_phase_modulation)
                  knots_list.append(num_knots)
                  
                  
                  filename_FA = f"FA_train_sequence_{method}_{objective_type}_prep{prep}_knots{num_knots}_deg{degree}_opt{optimization_method}_length{length}_minangle{opt_min}_ngrad{gr_num}_nstates{n_st}.png"
                  filename_PM = f"PM_train_sequence_{method}_{objective_type}_prep{prep}_knots{num_knots}_deg{degree}_opt{optimization_method}_length{length}_minangle{opt_min}_ngrad{gr_num}_nstates{n_st}.png"
                  filename_output = f"OUTPUT_sequence_{method}_{objective_type}_prep{prep}_knots{num_knots}_deg{degree}_opt{optimization_method}_length{length}_minangle{opt_min}_ngrad{gr_num}_nstates{n_st}.png"
                  filename_init_PM = f"init_PM_sequence_{method}_{objective_type}_prep{prep}_knots{num_knots}_deg{degree}_opt{optimization_method}_length{length}_minangle{opt_min}_ngrad{gr_num}_nstates{n_st}.png"
                  filename_init_FA = f"init_FA_sequence_{method}_{objective_type}_prep{prep}_knots{num_knots}_deg{degree}_opt{optimization_method}_length{length}_minangle{opt_min}_ngrad{gr_num}_nstates{n_st}.png"
                  
      
                  sequence_dir = os.path.join(os.getcwd(), "SEQUENCES", filename_FA.replace(".png", ""))
                  os.makedirs(sequence_dir, exist_ok=True)
      
                  save_path_FA = os.path.join(sequence_dir, filename_FA)
                  save_path_PM = os.path.join(sequence_dir, filename_PM)
                  save_path_output = os.path.join(sequence_dir, filename_output)
                  save_path_init_PM = os.path.join(sequence_dir, filename_init_PM)
                  save_path_init_FA = os.path.join(sequence_dir, filename_init_FA)
      
                  np.save(os.path.join(sequence_dir, "optimized_fa_train.npy"), optimized_fa_train)
                  np.save(os.path.join(sequence_dir, "optimized_phase_modulation.npy"), optimized_phase_modulation)
      
                  plt.figure(figsize=(10, 4))
                  plt.plot(eval_points, optimized_fa_train)
                  plt.title(f"Optimized Flip Angle Sequence\nMethod: {method}, Knots: {num_knots}, Degree: {degree}")
                  plt.xlabel("TR index")
                  plt.ylabel("Flip Angle (rad)")
                  plt.grid(True)
                  plt.tight_layout()
                  plt.savefig(save_path_FA, dpi=300)
                  plt.close()
                  
                  
                  
                  
                  plt.figure(figsize=(10, 4))
                  plt.plot(eval_points, optimized_phase_modulation)
                  plt.title(f"Optimized phase modulation\nMethod: {method}, Knots: {num_knots}, Degree: {degree}")
                  plt.xlabel("TR index")
                  plt.ylabel("Angle (rad)")
                  plt.grid(True)
                  plt.tight_layout()
                  plt.savefig(save_path_PM, dpi=300)
                  plt.close()
                  
                  
                  
                  fig, ax = plt.subplots(figsize=(16, 6))              
                  
                  D_min, D_max = 5e-4, 1e-3  
                  D_values = jnp.logspace(np.log10(D_min), np.log10(D_max), 1000)
                  colors = cm.rainbow(np.linspace(0, 1, len(D_values)))

                  for i, (D_scale, color) in enumerate(zip(D_values, colors)):
                      D_tensor = epg.generate_diffusion_tensor(D_scale)
                      result = epg.output_plotter_general(optimized_fa_train, optimized_phase_modulation, T1 = 1500, T2 = 90, M = 0.9, D = D_tensor, prep_FA_opt=prep_FA, prep_PM_opt=prep_PM, sampling = sampling, sampling_idx = sampling_idx, sampling_rate = sampling_rate, gradient_number = gr_num, n_states = n_st, prepend_inversion_and_30deg_train=prep)
                      if i % 100 == 0:  
                          mantissa, exponent = f"{D_scale:.1e}".split('e')
                          mantissa = float(mantissa)
                          exponent = int(exponent)
                          label = fr"${mantissa} \times 10^{{{exponent}}} \, \mathrm{{mm/s^2}}$"
                          ax.plot(np.array(result), label=label, color=color)
                      else:
                          ax.plot(np.array(result), color=color)
                  
                  for start in range(0, length * gr_num, length):
                      print(start)
                      ax.axvspan(start , start + 3, color='gray', alpha=0.9)
                  
                  legend1 = ax.legend(title="Diffusion Scale", loc="lower right", fontsize=9)
                  ax.add_artist(legend1)  
                  
                  zone_patch = mpatches.Patch(color='gray', alpha=0.9, label='Diffusion preparation')
                  ax.legend(handles=[zone_patch], title="Sequence elements", loc="upper right", fontsize=9)
                  
                  
                  ax.set_xlabel("Flip Angle Index")
                  ax.set_ylabel("Output Signal")
                  ax.set_title("Signal strength for optimized sequence")
                  ax.grid(True)
                  fig.tight_layout()
                  fig.savefig(save_path_output, dpi=300)
                  plt.close(fig)
                  
                  
                  
                  plt.figure(figsize=(10, 4))
                  plt.plot(initial_phase)
                  plt.title(f"Initial phase modulation\nMethod: {method}, Knots: {num_knots}, Degree: {degree}")
                  plt.xlabel("TR index")
                  plt.ylabel("Angle (rad)")
                  plt.grid(True)
                  plt.tight_layout()
                  plt.savefig(save_path_init_PM, dpi=300)
                  plt.close()
                  
                  
                  
                  plt.figure(figsize=(10, 4))
                  plt.plot(FA_array)
                  plt.title(f"Initial flip angle array \nMethod: {method}, Knots: {num_knots}, Degree: {degree}")
                  plt.xlabel("TR index")
                  plt.ylabel("Angle (rad)")
                  plt.grid(True)
                  plt.tight_layout()
                  plt.savefig(save_path_init_FA, dpi=300)
                  plt.close()
                  
    filename_knots_PM = f"PM_multipleknots_{method}_{objective_type}_prep{prep}_minknots{np.min(knots_list)}_maxknots{np.max(knots_list)}_length{length}_minangle{opt_min}_ngrad{gr_num}_nstates{n_st}.png"
    filename_knots_FA = f"FA_multipleknots_{method}_{objective_type}_prep{prep}_minknots{np.min(knots_list)}_maxknots{np.max(knots_list)}_length{length}_minangle{opt_min}_ngrad{gr_num}_nstates{n_st}.png"
                    
        
    useful_dir = os.path.join(os.getcwd(), "USEFUL", filename_knots_PM.replace(".png", ""))
    os.makedirs(useful_dir, exist_ok=True)
    
    save_path_knots_FA = os.path.join(useful_dir, filename_knots_FA)
    save_path_knots_PM = os.path.join(useful_dir, filename_knots_PM)
    
    optimized_fa_sequences = [np.array(seq) for seq in optimized_fa_sequences]
    optimized_pm_sequences = [np.array(seq) for seq in optimized_pm_sequences]
    
    unique_knots = sorted(set(knots_list))
    colors = cm.rainbow(np.linspace(0, 1, len(unique_knots)))
    knots_to_color = {k: colors[i] for i, k in enumerate(unique_knots)}
    
    plt.figure(figsize=(10, 6))
    for seq, k in zip(optimized_fa_sequences, knots_list):
        plt.plot(seq, color=knots_to_color[k], label=f"{k} knots")
    plt.title("Optimized FA Sequences")
    plt.xlabel("Timepoint")
    plt.ylabel("Flip Angle [rad]")
    plt.legend(loc = 'best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path_knots_FA, dpi=300)
    plt.close()
      
    plt.figure(figsize=(10, 6))
    for seq, k in zip(optimized_pm_sequences, knots_list):
        plt.plot(seq, color=knots_to_color[k], label=f"{k} knots")
    plt.title("Optimized PM Sequences")
    plt.xlabel("Timepoint")
    plt.ylabel("Phase Modulation [rad]")
    plt.legend(loc = 'best')
    plt.grid(True)
    plt.tight_layout()  
    plt.savefig(save_path_knots_PM, dpi=300)  
    plt.close()
















param_labels = ['T1', 'T2', 'M0']
x_spacing = 0.18

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [1, 1]})

for i, label in enumerate(summary_labels):
    color = summary_colors[i % len(summary_colors)]
    
    for j, param in enumerate(param_labels):
        x_pos = j + i * x_spacing
        y_init = summary_ncrlb_initial[label][j]
        y_opt = summary_ncrlb_optimized[label][j]
        
        ax1.annotate(
            '', xy=(x_pos, y_opt), xytext=(x_pos, y_init),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            annotation_clip=False
        )
        ax1.scatter(x_pos, y_init, color=color, s=30, label=f"{label} Initial" if j == 0 else "", marker='o', edgecolor='black', alpha=0.5, zorder=3)
        ax1.scatter(x_pos, y_opt, color=color, s=30, label=f"{label} Optimized" if j == 0 else "", marker='o', edgecolor='black', zorder=3)

group_centers = [j + (len(summary_labels)-1)*x_spacing/2 for j in range(len(param_labels))]
ax1.set_xticks(group_centers)
ax1.set_xticklabels(param_labels, fontsize=12)
ax1.set_ylabel("Normalized CRLB", fontsize=14)
ax1.set_title("Initial and Optimized nCRLBs per Parameter", fontsize=16)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

for xi in np.arange(len(param_labels) - 1) + x_spacing*(len(summary_labels)-1)/2 + 0.5:
    ax1.axvline(x=xi, color='gray', linestyle='dotted', alpha=1)

handles, labels_ = ax1.get_legend_handles_labels()
by_label = dict(zip(labels_, handles))
ax1.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10, title="Configuration")


shift = 0.15  

for i, label in enumerate(summary_labels):
    color = summary_colors[i % len(summary_colors)]
    
    initial_vals = np.array(summary_ncrlb_initial[label])
    optimized_vals = np.array(summary_ncrlb_optimized[label])
    
    y_init_mean = np.mean(initial_vals)
    y_opt_mean = np.mean(optimized_vals)
    
    y_init_std = np.std(initial_vals)
    y_opt_std = np.std(optimized_vals)
    
    x_init = i - shift
    x_opt = i + shift
    x_center = i
    
    ax2.errorbar(x_init, y_init_mean, yerr=y_init_std, fmt='o', color=color, markersize=7, markeredgecolor='black', markeredgewidth=1.3,
                 label=f"{label} Initial", ecolor='gray', elinewidth=3, capsize=5,
                 alpha=0.5, zorder=3)
    
    ax2.plot([x_init, x_center], [y_init_mean, y_init_mean], linestyle='dotted',
             color=color, lw=2)
    
    
    ax2.errorbar(x_opt, y_opt_mean, yerr=y_opt_std, fmt='o', color=color, markersize=7, markeredgecolor='black', markeredgewidth=1.3,
                 label=f"{label} Optimized", ecolor='gray', elinewidth=3, capsize=5,
                 zorder=3)
    
    ax2.plot([x_opt, x_center], [y_opt_mean, y_opt_mean], linestyle='dotted',
             color=color, lw=2)
    
    
    ax2.annotate(
        '', xy=(x_center, y_opt_mean), xytext=(x_center, y_init_mean),
        arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
        annotation_clip=False
    )

ax2.set_xticks(range(len(summary_labels)))
ax2.set_xticklabels(summary_labels, rotation=0, ha='center', fontsize=14)
ax2.set_ylabel("Mean Normalized CRLBs", fontsize=14)
ax2.set_title("Initial and Optimized nCRLB means per Configuration", fontsize=16)
ax2.grid(axis='y', linestyle='--', alpha=0.5)



plt.tight_layout()

summary_dir = os.path.join(os.getcwd(), "Optimization summaries")
os.makedirs(summary_dir, exist_ok=True)
summary_path = os.path.join(summary_dir, f"BAHAHAHHAHAHAHAHAHHAHAHAH_ncrlb_summary_comparison_allmethods_70knots_free_form.png")
plt.savefig(summary_path, dpi=300)
plt.close()
                
           
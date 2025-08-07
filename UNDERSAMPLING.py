import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import cmath
import pdb
from tqdm import tqdm
import jax
import jax.numpy as jnp
import sigpy as sp
# import nibabel as nib
import BlochSimulation_DH
import UEEphase_DH as uee
import EPG_blocks_jaxcode as epg
import pandas as pd
jax.config.update("jax_enable_x64", True)
import os


output_dir = "phantom images"
os.makedirs(output_dir, exist_ok=True)


prep_PM = np.load('DIFFPREPARATION/pm_train_diffprep.npy')
prep_FA = np.load('DIFFPREPARATION/fa_train_diffprep.npy')



def main():    

    def matching(dict_matrix_norm, signal, keys, signal_norm):
        signal_norm_value = np.linalg.norm(signal)
        if signal_norm_value == 0:
            raise ValueError('Zero sgnal')
        signal_normalized = signal / signal_norm_value

        inner_products = jnp.dot(dict_matrix_norm, signal_normalized.flatten())

        idx_best = jnp.argmax(jnp.abs(inner_products))

        best_params = [keys[idx_best]]
        M0 = signal_norm_value / signal_norm[idx_best]

        inprod_val = inner_products[idx_best]

        return best_params, M0, inprod_val

    def Zero_padding(image, shape):
        """
        This funtion only works when the image had uneven dimensions and the resulting shape has uneven dimensions
        :param image: Image to be zero-padded
        :param shape: Desired shape
        :return: Image with desired shape
        """
        shape_image = np.array(image.shape)
        shape = np.array(shape)
        add_block = np.floor((shape - shape_image)/2)
        add_block = add_block.astype(int)
        image_new = np.zeros(shape, dtype=type(image[0, 0]))
        image_new[add_block[0]:(add_block[0]+shape_image[0]), add_block[1]:(add_block[1]+shape_image[1])] = image

        return image_new

    def Fun_mask(Image, threshold):
        mean = np.mean(np.abs(Image), axis=0)
        mask = mean > threshold
        return mask


    #%% Settings
    Stochastic_noise = False
    plot = True
    SNR = 200

    dcf_bool = True
    golden_angle = False
    spiral = 'Philips_spiral'
    off_set = 0
    slice = 18



    length_module = 170
    FA_array = jnp.array(np.load("fa_array_initial.npy"))[:length_module]
    phase_modulation = epg.quadratic_PM_function_malleable(FA_array, 0.5, 0.03, -0.03)
    plt.plot(FA_array, label='FA')
    plt.savefig(os.path.join(output_dir, 'FA_array.png'))

    T1 = np.geomspace(150, 3000, 30)
    T2 = np.geomspace(30, 1000, 30)
    FA_scales = np.linspace(0, 1, 30)
    MD_scales = np.linspace(0, 1, 30) 

    D0 = 1e-3 * jnp.array([[1, 0.1, 0.1],
                                [0.1, 1, 0.1],
                                [0.1, 0.1, 1]])


    def generate_diffusion_tensor_grid(D0, MD_sc, FA_sc):
        D_matrices = []
        param_pairs = []

        for md in MD_sc:
            row_D = []
            row_params = []
            for fa in FA_sc:
                scale_yyzz = 0.1 + 0.9 * fa
                D_mod = D0.at[1, 1].set(D0[1, 1] * scale_yyzz)
                D_mod = D_mod.at[2, 2].set(D0[2, 2] * scale_yyzz)
                row_D.append(md * D_mod)
                row_params.append(jnp.array([md, fa]))
            D_matrices.append(jnp.stack(row_D))
            param_pairs.append(jnp.stack(row_params))

        return jnp.stack(D_matrices), jnp.stack(param_pairs)



    def create_dict(alpha, T1, T2, MD, FA): 
        D_mtrx = generate_diffusion_tensor_grid(D0, MD, FA)[0]
        
        #use itertool instead
        dict = {}
        for i in tqdm(range(T1.shape[0]), desc='Create dictionary'):
            for ii in range(T2.shape[0]):
                if T2[ii] < T1[i]:
                    for iii in range(MD.shape[0]):
                        for iiii in range(FA.shape[0]):   
                            dict[(T1[i], T2[ii], MD[iii], FA[iiii])] = epg.output_plotter_general(alpha, phase_modulation = phase_modulation, T1 = T1[i], T2 = T2[ii], M = 1, D = D_mtrx[iii][iiii], prep_FA_opt=prep_FA, prep_PM_opt=prep_PM, sampling = False, sampling_idx = 0, sampling_rate = 32, gradient_number = 9, n_states = 5, prepend_inversion_and_30deg_train=False)
        return dict




    def find_closest_pair(param_pairs, target):
        diffs = param_pairs - jnp.array(target)
        distances = jnp.linalg.norm(diffs, axis=2)
        index = jnp.unravel_index(jnp.argmin(distances), distances.shape)
        return index

    def combine_maps(FA_map, MD_map):
        return jnp.stack([MD_map, FA_map], axis=-1)

    def full_run(alpha, T1, T2, MD_scales, FA_scales):   

        dict = create_dict(alpha = alpha, T1 = T1, T2 = T2, MD = MD_scales, FA = FA_scales)

        return dict

    def generate_checkerboard_D(n_tiles = 11, radius = 0.7/2 * 121):
    
        D0 = 1e-3 * jnp.array([[1, 0.1, 0.1],
                            [0.1, 1, 0.1],
                            [0.1, 0.1, 1]])

        tile_size = 11
        height, width = n_tiles * tile_size, n_tiles * tile_size
        D_map = np.zeros((height, width, 3, 3))

        for i in range(n_tiles):
            md_scale = 0.5 + i / (n_tiles - 1)  

            for j in range(n_tiles):
                scale_yyzz = 1.0 -  j / (n_tiles - 1)  
                D = D0.at[1, 1].set(D0[1, 1] * scale_yyzz)
                D = D.at[2, 2].set(D0[2, 2] * scale_yyzz)
                D_scaled = md_scale * D

                y_start, y_end = i * tile_size, (i + 1) * tile_size
                x_start, x_end = j * tile_size, (j + 1) * tile_size
                D_map[y_start:y_end, x_start:x_end, :, :] = np.array(D_scaled)

        y_grid, x_grid = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        dist_from_center = np.sqrt((y_grid - center_y) ** 2 + (x_grid - center_x) ** 2)
        mask = dist_from_center <= radius

        D_map[~mask] = jnp.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]])

        return D_map


    print('Create dictionary...')
    idx = length_module
    N = idx * 9
    alpha = FA_array[:idx]
    dict = full_run(alpha, T1, T2, MD_scales, FA_scales)

    dict_matrix = np.array([dict[i] for i in dict.keys()])
    signal_norm = np.linalg.norm(dict_matrix, 2, axis=1)
    dict_matrix_norm = dict_matrix / signal_norm[:, None]
    dict_matrix

    print('Create dictionary done!')

    base = (11, 11)                     
    block_size = 11                     
    rad_frac = 0.7
    shape = (block_size*base[0], block_size*base[1])
    checkerboard = np.indices(base).sum(axis=0) % 2
    field = np.repeat(np.repeat(checkerboard, block_size, axis=0), block_size, axis=1)
    print(field)
    T1 = 750 + 500 * field
    T2 = 70 + 20 * field

    rho = uee.Cost.Create_mask(shape[0], rad_frac) * np.ones((shape[0], shape[1]))
    Mask = rho > 0.5

    figure = plt.figure(1)
    plt.title('Relaxation parameters component of the phantom')
    fig = plt.imshow(rho*T2)
    fig_bar = figure.colorbar(fig)
    plt.clim(60, 100)
    plt.savefig(os.path.join(output_dir, 'Sim Original image.png'))

    D_map = generate_checkerboard_D()

    ny, nx, _, _ = D_map.shape
    FA_map = np.zeros((ny, nx))
    MD_map = np.zeros((ny, nx))
    for iy in range(ny):
        for ix in range(nx):
            D_entry = D_map[iy, ix]
            FA_map[iy, ix] = epg.fractional_anisotropy(D_entry)
            MD_map[iy, ix] = epg.MD_from_tensor(D_entry)

    plt.figure()
    plt.title("Fractional Anisotropy (FA) component of the phantom")
    FA_map_cleaned = np.nan_to_num(FA_map, nan=0.0)
    plt.imshow(FA_map_cleaned, cmap='magma')
    plt.clim(0, 1) 
    plt.colorbar(label='[-]')
    plt.savefig(os.path.join(output_dir, 'FA_map.png'))

    plt.figure()
    plt.title("Mean Diffusivity (MD) component of the phantom")
    plt.imshow(MD_map , cmap='magma')
    plt.colorbar(label='MD [mm²/s]')
    plt.savefig(os.path.join(output_dir, 'MD_map.png'))

    error_T1 = np.zeros([1, shape[0], shape[1]])
    error_T2 = np.zeros([1, shape[0], shape[1]])
    error_FA = np.zeros([1, shape[0], shape[1]])
    error_MD = np.zeros([1, shape[0], shape[1]])

    D_tensor = 1e-3 * jnp.array([[1, 0.1, 0.1],
                                 [0.1, 1, 0.1], 
                                 [0.1, 0.1 ,1]])
    
    M_white = epg.output_plotter_general(alpha, phase_modulation=phase_modulation, T1=750, T2=70, M=1, D=D_tensor, prep_FA_opt=prep_FA, prep_PM_opt=prep_PM, sampling = False, sampling_idx = 0, sampling_rate = 32, gradient_number = 9, n_states = 5, prepend_inversion_and_30deg_train=False)
    M_gray = epg.output_plotter_general(alpha, phase_modulation=phase_modulation, T1=1250, T2=90, M=1, D=D_tensor, prep_FA_opt=prep_FA, prep_PM_opt=prep_PM, sampling = False, sampling_idx = 0, sampling_rate = 32, gradient_number = 9, n_states = 5, prepend_inversion_and_30deg_train=False)

    phase = uee.Cost.phase_field(shape=shape, order=2)
    M_j = np.zeros([N, shape[0], shape[1]], dtype=complex)
    M_j = np.tile(field[None, :, :], [N, 1, 1]) * np.tile(M_gray[:, None, None], [1, shape[0], shape[1]])
    M_j += np.tile(np.abs(field[None, :, :] - 1), [N, 1, 1]) * np.tile(M_white[:, None, None], [1, shape[0], shape[1]])
    M_j = rho * M_j * phase
    if Stochastic_noise:
        Signal_2 = np.mean(np.imag(M_j)**2, axis=0)
        # std = np.sqrt(Signal_2 / SNR)
        std = 0.005
        print('Standard deviation is: ', std)
        Error_field = rho * np.random.normal(0, std, shape) + rho * np.random.normal(0, std, shape) * 1j
        M_j += Error_field


    #%% Undersampling
    k_space_arr = []
    imag_arr = np.zeros([N, shape[0], shape[1]], dtype=complex)

    for ii in tqdm(range(N), desc='Undersampling the images'):
        if ii == 0:
            Bool = True
        else: Bool = False

        spiral_coord, bool_mask, len_one_spiral= uee.Cost.Spiral_coord(ii, bounds='full', shape=shape, golden_angle=golden_angle, spiral=spiral, off_set=off_set, interleaf=1)
        dcf = uee.Cost.Spiral_dcf(spiral_coord, dcf_bool)

        spiral_coord = spiral_coord[bool_mask]
        dcf = dcf[bool_mask]

        nufftlinop = sp.linop.NUFFT(shape, spiral_coord)

        k_space = nufftlinop * M_j[ii, :, :]
        k_space_arr.append(k_space)

        image = nufftlinop.H * (k_space[:, None] * dcf)
        imag_arr[ii, :, :] = image


    #%% Matching
    keys = list(dict.keys())
    T1_map = np.zeros([shape[0], shape[1]])
    T2_map = np.zeros([shape[0], shape[1]])
    M0_map = np.zeros([shape[0], shape[1]])
    Phase_map = np.zeros([shape[0], shape[1]])
    FA_map_raw = np.zeros([shape[0], shape[1]])
    MD_map_raw = np.zeros([shape[0], shape[1]])

    print('Start Matching...')
    for ii in tqdm(range(shape[0]), desc='Matching the signals to the dictionary'):
        for iii in range(shape[1]):
            if Mask[ii, iii]:
                res, M0, inprod_val = matching(dict_matrix_norm, imag_arr[:, ii, iii][:, None], keys, signal_norm)
                print(res[0])
                T1_map[ii, iii] = res[0][0]
                T2_map[ii, iii] = res[0][1]
                FA_map_raw[ii, iii] = res[0][2]
                MD_map_raw[ii, iii] = res[0][3]
                M0_map[ii, iii] = M0
                Phase_map[ii, iii] = -np.angle(inprod_val)



    combined_map = combine_maps(FA_map_raw, MD_map_raw)
    FA_map = np.zeros([shape[0], shape[1]])
    MD_map = np.zeros([shape[0], shape[1]])

    tensor_grid, param_pairs = generate_diffusion_tensor_grid(D0, MD_scales, FA_scales)

    for i in range(combined_map.shape[0]):
        for j in range(combined_map.shape[1]):
            (index1, index2) = find_closest_pair(param_pairs, combined_map[i, j])
            Diffusion_at_point = tensor_grid[index1, index2]
            FA_map[i, j] = epg.fractional_anisotropy(Diffusion_at_point)
            MD_map[i, j] = epg.MD_from_tensor(Diffusion_at_point)
    
    
    
    T = np.sum(rho)
    T1_div = np.copy(T1)
    T1_div[T1_div==0] = 1
    frac_error_T1 = rho * (T1_map - T1)/T1_div
    squared_sum_rel_err = np.sum((100 * frac_error_T1)**2)
    error_T1[0, :, :] = rho * (T1_map - T1)
    RMS_rel_T1 = np.sqrt(1 / T * squared_sum_rel_err)
    print('RMS relative error for T1: ', RMS_rel_T1)
    
    T2_div = np.copy(T2)
    T2_div[T2_div==0] = 1
    frac_error_T2 = rho * (T2_map - T2)/T2_div
    squared_sum_rel_err = np.sum((100 * frac_error_T2)**2)
    error_T2[0, :, :] = rho * (T2_map - T2)
    RMS_rel_T2 = np.sqrt(1 / T * squared_sum_rel_err)
    print('RMS relative error for T2: ', RMS_rel_T2)
    
    plt.figure()
    plt.title('Simulated MRF error in T1. RMS = ' + str(np.round(RMS_rel_T1, 1)))
    im = plt.imshow(frac_error_T1, cmap='RdBu_r')
    fig_bar = plt.colorbar(im)   # <-- use plt.colorbar
    plt.clim(-0.4, 0.4)
    fig_bar.set_label('Error')
    plt.savefig('Sim Error approximation T1.png')

    
    plt.figure()
    plt.title('Simulated MRF error in T2. RMS = ' + str(np.round(RMS_rel_T2, 1)))
    im = plt.imshow(frac_error_T2, cmap='RdBu_r')
    fig_bar = plt.colorbar(im)  # use plt.colorbar, not fig.colorbar
    plt.clim(-0.4, 0.4)
    fig_bar.set_label('Error')
    plt.savefig('Sim Error approximation T2.png')

    
    
    
    
    
    
    plt.figure()
    plt.imshow(FA_map, cmap='magma')
    plt.colorbar(label='FA')
    plt.title('FA Map')
    plt.savefig(os.path.join(output_dir, "FA_image.png"))
    
    plt.figure()
    plt.imshow(MD_map, cmap='magma')
    plt.colorbar(label='MD')
    plt.title('MD Map')
    plt.savefig(os.path.join(output_dir, 'MD_image.png'))
    
    plt.figure()
    plt.imshow(T1_map, cmap='magma')
    plt.colorbar(label='T1 [ms]')
    plt.title('T1 Map')
    plt.savefig("T1_image.png")
    
    
    plt.figure()
    plt.figure()
    plt.imshow(T2_map, cmap='magma')
    plt.colorbar(label='T2 [ms]')
    plt.title('T2 Map')
    plt.savefig("T2_image.png")

    plt.figure()
    plt.title('Phase')
    plt.imshow(Phase_map, cmap='twilight_shifted', vmin=-np.pi, vmax=np.pi)
    plt.colorbar(label = 'Phase [rad]')
    plt.savefig('Phase')

if __name__ == "__main__":
    main()
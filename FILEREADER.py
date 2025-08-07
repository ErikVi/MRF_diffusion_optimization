import h5py
import os

file_path = "iterations.h5"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"{file_path} not found in {os.getcwd()}")

with h5py.File(file_path, 'r') as hdf:
    def list_datasets(hdf_group, path=''):
        datasets = []
        for key in hdf_group.keys():
            item = hdf_group[key]
            if isinstance(item, h5py.Dataset):
                datasets.append(path + '/' + key)
            elif isinstance(item, h5py.Group):
                datasets.extend(list_datasets(item, path + '/' + key))
        return datasets

    datasets = list_datasets(hdf)
    print("Datasets in the file:")
    for dataset_path in datasets:
        dataset = hdf[dataset_path]
        print(f"{dataset_path}:")
        if dataset.shape == ():
            print(dataset[()])
        else:
            print(dataset[:])
        print()

"""Inspect an explicitly selected HDF5 file; h5py is an optional dependency."""

import argparse


def inspect_hdf5(path):
    import h5py

    with h5py.File(path, "r") as file:

        def show(name, item):
            if isinstance(item, h5py.Dataset):
                print(name, item[()], sep="\n")

        file.visititems(show)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    args = parser.parse_args(argv)
    inspect_hdf5(args.path)


if __name__ == "__main__":
    main()

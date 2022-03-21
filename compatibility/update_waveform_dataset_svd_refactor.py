import argparse

import h5py


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()

    f = h5py.File(args.dataset, "r+")
    try:
        grp = f.create_group("svd")
        grp["V"] = f["svd_V"]
        del f["svd_V"]
    except ValueError:
        print("Dataset is already in correct format.")

    f.close()


if __name__ == "__main__":
    main()

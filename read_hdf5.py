# read the hdf5 file
import h5py
import argparse
import glob
import os
import numpy as np
import imageio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    # read the hdf5 file
    all_hdf5_files = glob.glob(os.path.join(args.input_dir, "*.hdf5"))
    os.makedirs(args.output_dir, exist_ok=True)
    for hdf5_file in all_hdf5_files:
        with h5py.File(hdf5_file, 'r') as f:
            for key in f.keys():
                if key == "colors":
                    # convert the colors to numpy array and save it as png
                    colors = f[key][:]
                    colors = colors.astype(np.uint8)
                    colors = colors.reshape(colors.shape[0], colors.shape[1], 3)
                    colors = colors.transpose(1, 0, 2)
                    colors = colors.reshape(colors.shape[0], colors.shape[1], 3)
                    # save the colors by imageio
                    imageio.imwrite(os.path.join(args.output_dir, os.path.basename(hdf5_file).replace(".hdf5", ".png")), colors)
                    

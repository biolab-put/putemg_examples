#!/usr/bin/env python3

import os
import sys

import pandas as pd

from putemg_features import biolab_utilities


def usage():
    print()
    print("Applies denoising filter")
    print()
    print("Usage: {:s} <input_hdf5> <output_hdf5>".format(os.path.basename(__file__)))
    print("     <input_hdf5>:  putEMG HDF5 file containing raw experiment data")
    print("     <output_hdf5>: output HDF5 file containing filtered data")
    print()
    print("Example:")
    print("{:s} ../putEMG/Data-HDF5/emg_gestures-14-sequential-2018-04-06-10-30-11-595.hdf5 "
          "filtered-14-sequential-1.hdf5".
          format(os.path.basename(__file__)))
    exit(1)


if __name__ == '__main__':
    if '-h' in sys.argv or '--help' in sys.argv:
        usage()

    if len(sys.argv) != 3:
        print("Invalid parameter count")
        usage()

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    df: pd.DataFrame = pd.read_hdf(input_file)
    biolab_utilities.apply_filter(df)
    df.to_hdf(output_file, 'data', format='table', mode='w', complevel=5)

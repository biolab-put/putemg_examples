#!/usr/bin/env python3

import os
import sys
import pandas as pd
from putemg_features import biolab_utilities


def usage():
    print()
    print("Usage: {:s} command [parameters]".format(os.path.basename(__file__)))
    print("Commands:")
    print("    filter <input_hdf5> <output_hdf5>")
    print("        Applies denoising filter")
    print()
    print("        <input_hdf5>: putEMG HDF5 file containing raw experiment data")
    print("        <output_hdf5>: output HDF5 file")
    print()
    print("    calculate_features <feature_config_xml> <input_hdf5> <output_hdf5>")
    print("        Calculates EMG signal features based on given XML configuration")
    print()
    print("        <feature_config_xml>: XML fil,e containing feature descriptors")
    print("                               (see putemg_processing/all_features.xml for example)")
    print("        <input_hdf5>: putEMG raw or filtered HDF5 file containing experiment data")
    print()
    print("Examples:")
    print("{:s} all_features.xml ./putEMG/Data-HDF5/emg_gestures-14-sequential-2018-04-06-10-30-11-595.hdf5".
          format(os.path.basename(__file__)))
    exit(1)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Too few parameters")
        usage()

    command = sys.argv[1]
    if command not in ["filter", "calculate_features"]:
        print("Unknown command \"{:s}\"".format(command))
        usage()

    if command == "filter":
        if len(sys.argv) != 4:
            print("Invalid parameter count")
            usage()

        input_file = sys.argv[2]
        output_file = sys.argv[3]

        df: pd.DataFrame = pd.read_hdf(input_file)
        biolab_utilities.apply_filter(df)
        df.to_hdf(output_file, 'data', format='table', mode='w', complevel=5)

#!/usr/bin/env python3

import os
import sys

import putemg_features


def usage():
    print()
    print('Usage: {:s} <feature_config_xml> <input_hdf5> <output_hdf5>'.format(os.path.basename(__file__)))
    print()
    print('Arguments:')
    print('    <feature_config_xml>     XML file containing feature descriptors')
    print('    <input_hdf5>             putEMG HDF5 file containing experiment data')
    print("    <output_hdf5>            output HDF5 file containing calculated feature data")
    print()
    print('Examples:')
    print('{:s} putemg_features/all_features.xml '
          '../putEMG/Data-HDF5/emg_gestures-14-sequential-2018-04-06-10-30-11-595.hdf5 '
          'features-14-sequential-1.hdf5'.
          format(os.path.basename(__file__)))
    exit(1)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Illegal number of parameters')
        usage()

    xml_file_url = sys.argv[1]
    if not os.path.isfile(xml_file_url):
        print('XML file with feature descriptors does not exist - {:s}'.format(xml_file_url))
        usage()

    hdf5_file_url = sys.argv[2]
    if not os.path.isfile(hdf5_file_url):
        print('putEMG HDF5 file does not exist - {:s}'.format(hdf5_file_url))
        usage()

    print('Calculating features for {:s} file'.format(hdf5_file_url))
    ft = putemg_features.features_from_xml(xml_file_url, hdf5_file_url)

    output_hfd5_filename = sys.argv[3]
    print('Saving result to {:s} file'.format(output_hfd5_filename))
    ft.to_hdf(output_hfd5_filename, 'data', format='table', mode='w', complevel=5)

#!/usr/bin/env python3

import os
import sys
import glob
import pickle
import re
from typing import List, Dict

import pandas as pd
from sklearn.metrics import confusion_matrix

import putemg_features
from putemg_features import biolab_utilities


def usage():
    print()
    print('Usage: {:s} <putEMG_HDF5_folder> <output_folder> [options]'.format(os.path.basename(__file__)))
    print()
    print('Arguments:')
    print('    <putEMG_HDF5_folder>     URL to a folder containing raw HDF5 putEMG dataset files')
    print('    <output_folder>          URL to a output folder - results and intermediate files will be written here')
    print()
    print('Options:')
    print('    -nf      Skip filtering phase, use only if filtering was already applied, and filtered filed do exist')
    print('    -nc      Skip feature calculation phase, use only if features were already calculated and files do exist')
    print()
    print('Example:')
    print('{:s} ../putEMG/Data-HDF5/ ./shallow_learn_results/'.format(os.path.basename(__file__)))
    exit(1)


if __name__ == '__main__':
    if '-h' in sys.argv or '--help' in sys.argv:
        usage()

    if len(sys.argv) < 3:
        print('Illegal number of parameters')
        usage()

    putemg_folder = os.path.abspath(sys.argv[1])
    result_folder = os.path.abspath(sys.argv[2])

    if not os.path.isdir(putemg_folder):
        print('{:s} is not a valid folder'.format(putemg_folder))
        exit(1)

    if not os.path.isdir(result_folder):
        print('{:s} is not a valid folder'.format(result_folder))
        exit(1)

    filtered_data_folder = os.path.join(result_folder, 'filtered_data')
    calculated_features_folder = os.path.join(result_folder, 'calculated_features')

    # list all hdf5 files in given input folder
    all_files = [f for f in sorted(glob.glob(os.path.join(putemg_folder, "*.hdf5")))]

    # if not skipped filter the input data and save to consequent output files
    if '-nf' not in sys.argv:
        # create folder for filtered data
        if not os.path.exists(filtered_data_folder):
            os.makedirs(filtered_data_folder)

        # by each filename in download folder
        for file in all_files:
            basename = os.path.basename(file)
            print('Denoising file: {:s}'.format(basename))

            # read raw putEMG data file and run filter
            df: pd.DataFrame = pd.read_hdf(file)
            biolab_utilities.apply_filter(df)

            # save filtered data to designated folder with prefix filtered_
            output_file = 'filtered_' + basename
            print('Saving to file: {:s}'.format(output_file))
            df.to_hdf(os.path.join(filtered_data_folder, output_file),
                      'data', format='table', mode='w', complevel=5)
    else:
        print('Denoising skipped!')
        print()

    # if not skipped calculate features from filtered files
    if '-nc' not in sys.argv:
        # create folder for calculated features
        if not os.path.exists(calculated_features_folder):
            os.makedirs(calculated_features_folder)

        # by each filename in download folder
        for file in all_files:
            filtered_file_name = 'filtered_' + os.path.basename(file)
            print('Calculating features for {:s} file'.format(filtered_file_name))

            # for filtered data file run feature extraction, use xml with limited feature set
            ft: pd.DataFrame = putemg_features.features_from_xml('./features_shallow_learn.xml',
                                                                 os.path.join(filtered_data_folder, filtered_file_name))

            # save extracted features file to designated folder with features_filtered_ prefix
            output_file = 'features_' + filtered_file_name
            print('Saving result to {:s} file'.format(output_file))
            ft.to_hdf(os.path.join(calculated_features_folder, output_file),
                      'data', format='table', mode='w', complevel=5)
    else:
        print('Feature extraction skipped!')
        print()

    # start shallow learn process

    # create list of records
    all_feature_records = [biolab_utilities.Record(os.path.basename(f)) for f in all_files]

    # data can be additionally filtered based on subject id
    records_filtered_by_subject = biolab_utilities.record_filter(all_feature_records)
    # records_filtered_by_subject = record_filter(all_feature_records,
    #                                             whitelists={"id": ["01", "02", "03", "04", "07"]})
    # records_filtered_by_subject = pu.record_filter(all_feature_records, whitelists={"id": ["01"]})

    # load feature data to memory
    dfs: Dict[biolab_utilities.Record, pd.DataFrame] = {}
    for r in records_filtered_by_subject:
        print("Reading features for input file: ", r)
        dfs[r] = pd.DataFrame(pd.read_hdf(os.path.join(calculated_features_folder, 'features_filtered_' + r.path)))

    # create k-fold validation set, with 3 splits - for each experiment day 3 combination are generated
    # this results in 6 data combination for each subject
    splits_all = biolab_utilities.data_per_id_and_date(records_filtered_by_subject, n_splits=3)

    # defines feature sets to be used in shallow learn
    feature_sets = {
        "RMS": ["RMS"],
        "Hudgins": ["MAV", "WL", "ZC", "SSC"],
        "Du": ["IAV", "VAR", "WL", "ZC", "SSC", "WAMP"]
    }

    # defines gestures to be used in shallow learn
    gestures = {
        0: "Idle",
        1: "Fist",
        2: "Flexion",
        3: "Extension",
        6: "Pinch index",
        7: "Pinch middle",
        8: "Pinch ring",
        9: "Pinch small"
    }

    # defines classifiers and its options to be used in shallow learn
    classifiers = {
        "LDA": {"solver": "svd", "shrinkage": None, "priors": None, "n_components": None,
                "store_covariance": False, "tol": 0.0001},
        "QDA": {"priors": None, "reg_param": 0.3, "store_covariance": False, "tol": 0.0001, "store_covariances": None},
        "kNN": {"n_neighbors": 5, "weights": "uniform", "algorithm": "auto", "leaf_size": 30, "p": 2,
                "metric": "minkowski", "metric_params": None, "n_jobs": None},
        "SVM": {"C": 1.0, "kernel": "rbf", "degree": 3, "gamma": "auto_deprecated", "coef0": 0.0, "shrinking": True,
                "probability": False, "tol": 0.001, "cache_size": 200, "class_weight": None, "verbose": False,
                "max_iter": -1, "decision_function_shape": "ovr", "random_state": None}
    }

    # defines channels configurations for which classification will be run
    channel_range = {
        "24chn": {"begin": 1, "end": 24},
        # "8chn_1band": {"begin": 1, "end": 8},
        "8chn_2band": {"begin": 9, "end": 16},
        # "8chn_3band": {"begin": 17, "end": 24}
    }

    print()
    print('Starting to shallow learn')

    # for each channel configuration
    for ch_range_name, ch_range in channel_range.items():
        output: Dict[str, any] = dict()

        output["gestures"] = gestures
        output["classifiers"] = classifiers
        output["feature_sets"] = feature_sets
        output["results"]: List[Dict[str, any]] = list()

        print()
        print('Channel configuration: {:s}'.format(ch_range_name), flush=True)

        # for each experiment (single subject, single day)
        for id_, id_splits in splits_all.items():
            print('\tTrial ID: {:s}'.format(id_), flush=True)

            # for split in k-fold validation of each day of each subject
            for i_s, s in enumerate(id_splits):

                # for each feature set
                for feature_set_name, features in feature_sets.items():
                    print('\t\tFeature set: {:s} -'.format(feature_set_name), end='', flush=True)

                    # prepare training and testing set based on combination of k-fold split, feature set and gesture set
                    data = biolab_utilities.prepare_data(dfs, s, features, list(gestures.keys()))

                    # list columns containing only feature data
                    regex = re.compile(r'input_[0-9]+_[A-Z]+_[0-9]+')
                    cols = list(filter(regex.search, list(data["train"].columns.values)))

                    # strip columns to include only selected channels, eg. only one band
                    cols = [c for c in cols if (ch_range["begin"] <= int(c[c.rindex('_')+1:]) <= ch_range["end"])]

                    # extract limited training x and y, only with chosen channel configuration
                    train_x = data["train"][cols]
                    train_y = data["train"]["output_0"]

                    # extract limited testing x and y, only with chosen channel configuration
                    test_x = data["test"][cols]
                    test_y_true = data["test"]["output_0"]

                    # for each defined classifier
                    for clf, clf_args in classifiers.items():
                        print(' {:s}'.format(clf), end='', flush=True)

                        # prepare classifier pipeline
                        # fit the classifier to train data
                        pipeline = biolab_utilities.prepare_pipeline(train_x, train_y, predictor=clf,
                                                                     norm_per_feature=False, **clf_args)

                        # run prediction on test data
                        test_y_pred = pipeline.predict(test_x)

                        # calculate confusion matrix
                        cm = confusion_matrix(test_y_true.values.astype(int), test_y_pred, list(gestures.keys()))

                        # save classification results to output structure
                        output["results"].append({"id": id_, "split": i_s, "clf": clf, "feature_set": feature_set_name,
                                                  "cm": cm, "y_true": test_y_true.values.astype(int),
                                                  "y_pred": test_y_pred})
                    print()

        # for each channel configuration dump classification results to file
        pickle.dump(output, open(os.path.join(result_folder, "classification_result_" + ch_range_name + ".bin"), "wb"))

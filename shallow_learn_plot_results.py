#!/usr/bin/env python3

import sys
import os
import pickle
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def usage():
    print()
    print('Usage: {:s} <result_folder>'.format(os.path.basename(__file__)))
    print()
    print('Arguments:')
    print('    <result_folder>          URL to a folder containing shallow learn classification results')
    print()
    print('Example:')
    print('{:s} ./shallow_learn_results/'.format(os.path.basename(__file__)))
    exit(1)


if __name__ == '__main__':
    if '-h' in sys.argv or '--help' in sys.argv:
        usage()

    if len(sys.argv) != 2:
        print('Illegal number of parameters')
        usage()

    working_directory = sys.argv[1]

    # Plot precision, recall (presented as ROC), f1, accuracy by classifier and feature set
    if 1:
        output: Dict[str, any] = dict()
        output = pickle.load(open(os.path.join(working_directory, "classification_result_24chn.bin"), "rb"))

        fig_precision, ax_precision = plt.subplots(num="precision", figsize=(1800/96, 450/96), dpi=96)
        fig_recall, ax_recall = plt.subplots(num="recall", figsize=(1800/96, 450/96), dpi=96)
        fig_f1, ax_f1 = plt.subplots(num="f1 score", figsize=(1800/96, 450/96), dpi=96)
        fig_accuracy, ax_accuracy = plt.subplots(num="accuracy", figsize=(1800/96, 450/96), dpi=96)

        figs = [fig_precision, fig_recall, fig_f1, fig_accuracy]
        axes = [ax_precision, ax_recall, ax_f1, ax_accuracy]

        fig_roc, ax_roc = plt.subplots(num="precision_vs_recall", figsize=(1800/96, 800/96), dpi=96)

        roc_cls_mark_types = {'LDA': 'o', 'QDA': '^', 'SVM': 'D', 'kNN': 's'}
        roc_fset_mark_colors = {'RMS': '#21557A', 'Hudgins': '#FF7F0E', 'Du': '#2CA02A'}
        roc_legend_str = ()

        index = np.arange(len(output["classifiers"]))
        bar_width = 0.15
        bar_spacer = 0.025

        for i, set_ in enumerate(output["feature_sets"].keys()):

            precision_mean = []
            precision_std = []
            precision_median = []
            precision_25percentile = []
            precision_75percentile = []

            recall_mean = []
            recall_std = []
            recall_median = []
            recall_25percentile = []
            recall_75percentile = []

            f1_mean = []
            f1_std = []
            f1_median = []
            f1_25percentile = []
            f1_75percentile = []

            accuracy_mean = []
            accuracy_std = []
            accuracy_median = []
            accuracy_25percentile = []
            accuracy_75percentile = []

            for clf in output["classifiers"].keys():
                data = list(filter(lambda r: r["clf"] == clf and r["feature_set"] == set_, output["results"]))

                y_true = [r["y_true"] for r in data]
                y_pred = [r["y_pred"] for r in data]

                precision = [precision_score(t, p, average="macro", labels=np.unique(p))
                             for t, p in zip(y_pred, y_true)]

                recall = [recall_score(t, p, average="macro", labels=np.unique(t))
                          for t, p in zip(y_pred, y_true)]

                f1 = [f1_score(t, p, average="macro", labels=np.unique(np.concatenate((t, p))))
                      for t, p in zip(y_pred, y_true)]

                accuracy = [accuracy_score(t, p, normalize=True) for t, p in zip(y_pred, y_true)]

                cm_sum = sum([r["cm"] for r in data])

                precision_std.append(np.std(precision))
                recall_std.append(np.std(recall))
                f1_std.append(np.std(f1))
                accuracy_std.append(np.std(accuracy))

                precision_mean.append(np.mean(precision))
                recall_mean.append(np.mean(recall))
                f1_mean.append(np.mean(f1))
                accuracy_mean.append(np.mean(accuracy))

                precision_median.append(np.median(precision))
                recall_median.append(np.median(recall))
                f1_median.append(np.median(f1))
                accuracy_median.append(np.median(accuracy))

                precision_25percentile.append(np.percentile(precision, 25))
                recall_25percentile.append(np.percentile(recall, 25))
                f1_25percentile.append(np.percentile(f1, 25))
                accuracy_25percentile.append(np.percentile(accuracy, 25))

                precision_75percentile.append(np.percentile(precision, 75))
                recall_75percentile.append(np.percentile(recall, 75))
                f1_75percentile.append(np.percentile(f1, 75))
                accuracy_75percentile.append(np.percentile(accuracy, 75))

                ax_roc.scatter(recall_median[-1], precision_median[-1],
                               c=roc_fset_mark_colors[set_], marker=roc_cls_mark_types[clf],
                               s=[300], zorder=3)
                ax_roc.errorbar(recall_median[-1], precision_median[-1],
                                fmt='none', ecolor=roc_fset_mark_colors[set_], lw=2, capsize=10, capthick=2,
                                yerr=[[precision_median[-1] - precision_25percentile[-1]],
                                      [precision_75percentile[-1] - precision_median[-1]]],
                                xerr=[[recall_median[-1] - recall_25percentile[-1]],
                                      [recall_75percentile[-1] - recall_median[-1]]],
                                zorder=2)
                roc_legend_str = roc_legend_str + (clf + ' - ' + set_,)

            ax_precision.bar(index + (bar_width + bar_spacer) * i, precision_mean, bar_width, label=set_)
            ax_precision.errorbar(index + (bar_width + bar_spacer) * i, precision_median,
                                  fmt='ko', ecolor='k', lw=2, capsize=10,
                                  yerr=[np.array(precision_median) - np.array(precision_25percentile),
                                        np.array(precision_75percentile) - np.array(precision_median)])

            ax_recall.bar(index + (bar_width + bar_spacer) * i, recall_mean, bar_width, label=set_)
            ax_recall.errorbar(index + (bar_width + bar_spacer) * i, recall_median,
                               fmt='ko', ecolor='k', lw=2, capsize=10,
                               yerr=[np.array(recall_median) - np.array(recall_25percentile),
                                     np.array(recall_75percentile) - np.array(recall_median)])

            ax_f1.bar(index + (bar_width + bar_spacer) * i, f1_mean, bar_width, label=set_)
            ax_f1.errorbar(index + (bar_width + bar_spacer) * i, f1_median,
                           fmt='ko', ecolor='k', lw=2, capsize=10,
                           yerr=[np.array(f1_median) - np.array(f1_25percentile),
                                 np.array(f1_75percentile) - np.array(f1_median)])

            ax_accuracy.bar(index + (bar_width + bar_spacer) * i, accuracy_mean, bar_width, label=set_)
            ax_accuracy.errorbar(index + (bar_width + bar_spacer) * i, accuracy_median,
                                 fmt='ko', ecolor='k', lw=2, capsize=10,
                                 yerr=[np.array(accuracy_median) - np.array(accuracy_25percentile),
                                       np.array(accuracy_75percentile) - np.array(accuracy_median)])

        ax_precision.set_ylabel('Precision')
        ax_recall.set_ylabel('Recall')
        ax_f1.set_ylabel('F1 score')
        ax_accuracy.set_ylabel('Accuracy')

        maj_ticks = np.arange(0, 1100, step=100)
        maj_ticks = maj_ticks / 1000

        for f, a in zip(figs, axes):
            a.yaxis.label.set_size(25)

            a.set_ylim([0, 1.01])
            a.set_yticks(maj_ticks)
            a.set_xticks(index + ((bar_width + bar_spacer) * (len(output["feature_sets"]) - 1)) / 2)
            a.set_xticklabels(output["classifiers"].keys())
            a.tick_params(axis='x', which='major', labelsize=25)
            a.tick_params(axis='y', which='major', labelsize=15)

            a.yaxis.grid(b=True, which='major', linestyle='-')
            a.set_axisbelow(True)

            a.legend(fontsize=25, loc=[0.18, 0.11])

            f.subplots_adjust(left=0.05, right=0.99, top=0.99, bottom=0.09)

        ax_roc.legend(roc_legend_str, ncol=3, fontsize=25, loc='upper left')

        maj_ticks = np.arange(400, 1000, step=100)
        min_ticks = np.arange(375, 950, step=25)
        min_ticks = np.setdiff1d(min_ticks, maj_ticks)
        maj_ticks = maj_ticks / 1000
        min_ticks = min_ticks / 1000

        ax_roc.set_xlim([0.35, 0.95])
        ax_roc.set_xticks(min_ticks, minor=True)
        ax_roc.set_xticks(maj_ticks)
        ax_roc.set_ylim([0.35, 0.95])
        ax_roc.set_yticks(min_ticks, minor=True)
        ax_roc.set_yticks(maj_ticks)

        ax_roc.tick_params(axis='both', which='major', labelsize=15)

        ax_roc.yaxis.grid(b=True, which='minor', color='lightgray', linestyle='--')
        ax_roc.yaxis.grid(b=True, which='major', linestyle='-')
        ax_roc.xaxis.grid(b=True, which='minor', color='lightgray', linestyle='--')
        ax_roc.xaxis.grid(b=True, which='major', linestyle='-')

        ax_roc.set_axisbelow(True)

        ax_roc.set_ylabel('Precision')
        ax_roc.set_xlabel('Recall')
        ax_roc.yaxis.label.set_size(25)
        ax_roc.xaxis.label.set_size(25)

        fig_roc.subplots_adjust(left=0.05, right=0.99, top=0.985, bottom=0.09)

    # precision of separate gesture classification using a given classifier and set
    if 1:
        clf = "LDA"
        set_ = "Du"

        channel_set = {
            "24chn": "24 channels",
            "8chn_2band": "8 channels - middle band",
        }

        for ch_set in channel_set.keys():

            output: Dict[str, any] = dict()
            output = pickle.load(
                open(os.path.join(working_directory, "classification_result_" + ch_set + ".bin"), "rb"))

            precision_by_gesture: Dict[int, List] = dict()
            acc_by_gesture: Dict[int, List] = dict()

            data = list(filter(lambda r: r["clf"] == clf and r["feature_set"] == set_, output["results"]))

            y_true = [r["y_true"] for r in data]
            y_pred = [r["y_pred"] for r in data]

            for t, p in zip(y_true, y_pred):
                for gesture in output["gestures"].keys():
                    t_bin = (t == gesture)
                    p_bin = (p == gesture)

                    prec = precision_score(t_bin, p_bin)
                    if gesture in precision_by_gesture:
                        precision_by_gesture[gesture].append(prec)
                    else:
                        precision_by_gesture[gesture] = [prec]

            precision_mean = []
            precision_std = []
            precision_median = []
            precision_25percentile = []
            precision_75percentile = []

            for gesture in output["gestures"].keys():
                precision_mean.append(np.mean(precision_by_gesture[gesture]))
                precision_std.append(np.std(precision_by_gesture[gesture]))
                precision_median.append(np.median(precision_by_gesture[gesture]))
                precision_25percentile.append(np.percentile(precision_by_gesture[gesture], 25))
                precision_75percentile.append(np.percentile(precision_by_gesture[gesture], 75))

            index = np.arange(len(output["gestures"]))
            bar_width = 0.40

            fig, ax = plt.subplots(num="precision_" + ch_set, figsize=(800 / 96, 500 / 96), dpi=96)

            ax.title.set_fontsize(25)

            ax.bar(index, precision_mean, bar_width, color='#2CA02A')
            ax.errorbar(index, precision_median, fmt='ko', ecolor='k', lw=2, capsize=10,
                        yerr=[np.array(precision_median) - np.array(precision_25percentile),
                              np.array(precision_75percentile) - np.array(precision_median)])

            ax.set_ylabel('Precision')
            ax.yaxis.label.set_size(25)

            ax.set_xticklabels(output["gestures"].values(), rotation=45)
            ax.set_xticks(index)

            maj_ticks = np.arange(0, 1100, step=100)
            maj_ticks = maj_ticks / 1000

            ax.set_ylim([0, 1.01])
            ax.set_yticks(maj_ticks)
            ax.yaxis.grid(b=True, which='major', linestyle='-')
            ax.set_axisbelow(True)

            ax.tick_params(axis='both', which='major', labelsize=15)

            fig.subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.22)

    plt.show()

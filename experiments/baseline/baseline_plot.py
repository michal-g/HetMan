
"""
Creates plots for baseline testing.
"""

import numpy as np
import pandas as pd

import pickle
import re
from os import listdir
from os.path import isfile, join
from math import log

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


base_dir = '/home/users/grzadkow/compbio/scripts/HetMan/experiments/baseline'
marker_map = {'All': "*",
              'Neigh': "s",
              'Down': ">",
              'expr': '+',
              'Up': '<'}
alg_order = ('NaiveBayes', 'Lasso', 'SVCrbf', 'rForest')


def load_output():
    output_dir = base_dir + '/output/'
    file_list = [fl for fl in listdir(output_dir)
                 if isfile(join(output_dir, fl))
                 and re.search('base__run[0-9]+\\.p$', fl)]
    return [pickle.load(open(join(output_dir, fl), 'rb')) for fl in file_list]


def plot_performance():
    out_data = load_output()
    mut_genes = np.unique([x[1] for x in out_data[0]['AUC'].keys()])
    auc_data = [x['AUC'] for x in out_data]
    auc_min = min([min(x.values()) for x in auc_data]) * 0.9
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(7,11))

    for i, gene in enumerate(mut_genes):
        perf_data = pd.DataFrame(
            [{k[0]:v for k,v in x.items() if k[1] == gene}
             for x in auc_data])
        alg_indx = [alg_order.index(x) for x in perf_data.columns]
        perf_data = perf_data.ix[:, alg_indx]
        gene_lbl = '{1} ({0})'.format(*gene.split('_'))

        axes[i // 3, i % 3].set_title(gene_lbl, fontsize=15)
        axes[i // 3, i % 3].boxplot(
            x=np.array(perf_data),
            boxprops={'linewidth': 1.5},
            medianprops={'linewidth': 3, 'color': '#960c20'},
            flierprops={'markersize': 2}
            )

        if (i // 3) == 1:
            axes[i // 3, i % 3].set_xticklabels(
                perf_data.columns,
                fontsize=12, rotation=45, ha='right')
        else:
            axes[i // 3, i % 3].set_xticklabels(
                np.repeat('', len(alg_indx)))

        if (i % 3) == 0:
            axes[i // 3, i % 3].set_ylabel('AUC', fontsize=19)
        else:
            axes[i // 3, i % 3].set_yticklabels([])

        axes[i // 3, i % 3].set_title(gene_lbl, fontsize=16)
        axes[i // 3, i % 3].plot(
            list(range(len(alg_indx)+2)), np.repeat(0.5, len(alg_indx)+2),
            c="black", lw=0.8, ls='--', alpha=0.8)
        axes[i // 3, i % 3].set_ylim(auc_min, 1.0)

    plt.tight_layout(w_pad=-1.2, h_pad=1.5)
    plt.savefig(base_dir + '/plots/performance.png', dpi=700)


def plot_base(label):
    in_data = load_files(in_dir, 'base_' + label + '_')
    mut_genes = list(list(in_data[0]['AUC'].values())[0].keys())
    clf_list = list(in_data[0]['AUC'].keys())
    neighb_list = list(list(list(
        in_data[0]['AUC'].values())[0].values())[0].keys())
    scores = {gn:None for gn in mut_genes}
    times = {gn:None for gn in mut_genes}

    for mut_gn in mut_genes:
        auc_lower_q = pd.DataFrame(index=clf_list, columns=neighb_list)
        auc_med_q = pd.DataFrame(index=clf_list, columns=neighb_list)
        auc_upper_q = pd.DataFrame(index=clf_list, columns=neighb_list)
        tm_lower_q = pd.DataFrame(index=clf_list, columns=neighb_list)
        tm_med_q = pd.DataFrame(index=clf_list, columns=neighb_list)
        tm_upper_q = pd.DataFrame(index=clf_list, columns=neighb_list)
        scores[mut_gn] = auc_med_q
        times[mut_gn] = tm_med_q

        for neighb in neighb_list:
            for clf in clf_list:
                auc_list = [dt['AUC'][clf][mut_gn][neighb] for dt in in_data]
                tm_list = [dt['time'][clf][mut_gn][neighb] for dt in in_data]
                auc_lower_q.loc[clf, neighb] = np.percentile(auc_list, 25)
                auc_med_q.loc[clf, neighb] = np.percentile(auc_list, 50)
                auc_upper_q.loc[clf, neighb] = np.percentile(auc_list, 75)
                tm_lower_q.loc[clf, neighb] = log(np.percentile(tm_list, 25),
                                                  10)
                tm_med_q.loc[clf, neighb] = log(np.percentile(tm_list, 50),
                                                10)
                tm_upper_q.loc[clf, neighb] = log(np.percentile(tm_list, 75),
                                                  10)

            plt.scatter(
                x=tm_med_q.loc[:, neighb], y=auc_med_q.loc[:, neighb],
                c=tuple(range(len(clf_list))), marker=marker_map[neighb],
                s=55, alpha=0.7)
            plt.legend()

        #new_patch = mpatches.Patch(color=tuple(range(len(clf_list))),
        #                           label=clf_list)
        #plt.legend(handles=[new_patch])
        plt.savefig(out_dir + 'base_' + label + '_' + mut_gn + '.png',
                    bbox_inches='tight')
        plt.clf()

    return scores, times



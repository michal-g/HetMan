
import numpy as np
import pandas as pd

import pickle
from os import listdir
from os.path import isfile, join
from math import log

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


base_dir = '/home/users/grzadkow/compbio/scripts/HetMan/experiments/'
in_dir = base_dir + 'output/'
out_dir = base_dir + 'output/plots/'

marker_map = {'All': "*",
              'Neigh': "s",
              'Down': ">",
              'expr': '+',
              'Up': '<'}


def load_files(path, pattern):
    file_list = [f for f in listdir(path)
                 if isfile(join(path, f)) and pattern in f]
    return [pickle.load(open(path + "/" + fl, 'rb')) for fl in file_list]


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



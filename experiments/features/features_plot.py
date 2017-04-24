
"""
Creates plots for feature selection testing.
"""

base_dir = '/home/users/grzadkow/compbio/scripts/HetMan/experiments/features'
from .config import *
from ..utils import load_output, get_set_plotlbl

import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


# defines colours and symbols to use for plotting various classification
# algorithms and feature selection strategies
marker_map = {'All': "o",
              'expr': "D",
              'Neigh': "s",
              'Down': "<",
              'Up': ">"}
alg_map = (('NaiveBayes', 'black'),
           ('Lasso', '#805415'),
           ('SVCrbf', '#567714'),
           ('rForest', '#133353'),)


def plot_performance(clf_set='default', mtype_set='default'):
    """Plots classifier performance vs classification time."""
    out_data = load_output('features', clf_set, mtype_set)
    base_data = load_output('baseline', clf_set, mtype_set)

    # parse test results into time and performance datasets
    auc_data = [x['AUC'] for x in out_data]
    time_data = [x['time'] for x in out_data]
    base_auc = [x['AUC'] for x in base_data]
    base_time = [x['time'] for x in base_data]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16,6))
    for i, gene in enumerate(mtype_list[mtype_set]):
        gene_lbl = '{} - {}'.format(gene[0], str(gene[1]))
        axes[i].set_title(gene_lbl, fontsize=14)

        # get quantiles of feature selection results
        auc_quants = pd.DataFrame(
            [{tuple(k[0].split('_')):v for k,v in x.items() if k[1] == gene and
             isinstance(k[0], str)}
             for x in auc_data]).quantile((0.25,0.5,0.75))
        time_meds = pd.DataFrame(
            [{tuple(k[0].split('_')):v for k,v in x.items() if k[1] == gene and
             isinstance(k[0], str)}
             for x in time_data]).quantile(0.5)

        # add quantiles of baseline results to data
        auc_quants = auc_quants.join(pd.DataFrame(
            [{tuple(k[0].split('_')):v for k,v in x.items() if k[1] == gene}
             for x in base_auc]).quantile((0.25,0.5,0.75)))
        time_meds = time_meds.append(pd.DataFrame(
            [{tuple(k[0].split('_')):v for k,v in x.items() if k[1] == gene}
             for x in base_time]).quantile(0.5))

        # get labels for feature selection methods to use in plotting
        auc_quants.columns = [(x[0], [k for k,v in key_list.items()
                                      if str(v) == x[1]][0])
                              for x in auc_quants.columns]
        time_meds.index = [(x[0], [k for k,v in key_list.items()
                                   if str(v) == x[1]][0])
                           for x in time_meds.index]
        auc_quants = auc_quants.loc[:, time_meds.index]

        for run_key in time_meds.index:
            # figure out if the given algorithm + feature selection method
            # is dominated by any other (has better time and performance)
            dom_status = any((time_meds < time_meds[run_key])
                             & (auc_quants.ix[0.50,:]
                                > auc_quants.ix[0.50, run_key]))

            # choose plotting symbol parameters according to dominated status
            if dom_status:
                plot_alpha = 0.20
                plot_size = 4
            else:
                plot_alpha = 0.65
                plot_size = 10

            # plot errorbars of time vs. performance quantiles
            axes[i].errorbar(
                x=time_meds[run_key], y=auc_quants.ix[0.50, run_key],
                yerr=np.array(
                    [[auc_quants.ix[0.50, run_key]
                      - auc_quants.ix[0.25, run_key]],
                     [auc_quants.ix[0.75, run_key]
                      - auc_quants.ix[0.50, run_key]]]),
                fmt=marker_map[run_key[1]], c=dict(alg_map)[run_key[0]],
                ms=plot_size, alpha=plot_alpha, lw=0.8
                )

        # create the axis labels
        axes[i].set_xlabel('Time (s)', fontsize=20)
        if i == 0:
            axes[i].set_ylabel('AUC', fontsize=20)
        else:
            axes[i].set_yticklabels([])

        # configure the axis scales and limits
        axes[i].set_xscale('log', basex=2, subsx=[])
        axes[i].plot(axes[i].get_xlim(), [0.5,0.5],
                     c="black", lw=0.8, ls='--', alpha=0.8)
        axes[i].set_xticklabels([])
        axes[i].set_ylim(0.47, 1.0)

    # set up legend symbols
    alg_patch = [mpatches.Patch(color=c, label=alg) for alg,c in alg_map]
    key_patch = [mlines.Line2D([], [], color='black', label=key, marker=s)
                 for key,s in marker_map.items()]

    # add the legends to the plot
    legend2 = plt.legend(
        handles=key_patch, loc=8, ncol=len(key_patch),
        bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(0,-0.05,1,1))
    plt.legend(handles=alg_patch, loc=8, ncol=len(alg_patch),
               bbox_transform=plt.gcf().transFigure,
               bbox_to_anchor=(0,-0.1,1,1))
    plt.gca().add_artist(legend2)

    # adjust spacing between subplots and save the final plot to file
    plt.tight_layout(w_pad = 1.0)
    plt.savefig(base_dir + '/plots/'
                + get_set_plotlbl(clf_set) + '_' + get_set_plotlbl(mtype_set)
                + '__performance.png',
                bbox_inches='tight', dpi=700)



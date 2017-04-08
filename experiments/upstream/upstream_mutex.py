
import pickle
import time
import sys
import os
sys.path += ['/home/users/grzadkow/compbio/scripts/HetMan']

import synapseclient
import numpy as np
import data
from mutation import *
import classif


def main(argv):
    start_time = time.time()

    # define genes whose mutations we want to relate to one another
    up_gene = str(argv[1])
    down_gene = str(argv[2])
    up_mtype = MuType({('Gene', up_gene):None})
    down_mtype = MuType({('Gene', down_gene):None})

    # load in expression and mutation datasets
    while True:
        try:
            syn = synapseclient.Synapse()
            syn.login('grzadkow', 'W0w6g1i8A')
            cdata = data.MutExpr(
                syn, cohort=argv[0], mut_genes=[up_gene,down_gene],
                cv_info={'Prop':2/3.0, 'Seed':int(argv[-1])+1})
            break
        except:
            print("Synapse login failed, trying again...")

    # define the classifier to be used as well as the interval
    # cross-validation training samples for testing and tuning
    clf_lbl = str(argv[3])
    if clf_lbl in dir(classif):
        clf = eval('classif.' + clf_lbl)
    else:
        raise InputError("Unknown classifier specified!")
    score_indx = list(range(8))
    pred_indx = list(range(24,48))
    tune_indx = list(range(56,58))
    expr_genes = cdata.train_expr_.columns

    # test genes' mutations individually
    print('Starting classification...')
    up_test = cdata.score_clf(
        clf=clf(mut_genes=[up_gene], expr_genes=expr_genes),
        mtype=up_mtype, score_indx=score_indx, tune_indx=tune_indx)
    print(str(up_test))
    down_test = cdata.score_clf(
        clf=clf(mut_genes=[down_gene], expr_genes=expr_genes),
        mtype=down_mtype, score_indx=score_indx, tune_indx=tune_indx)
    print('Using the ' + clf_lbl + ' classifier, we get AUCs of '
          + str(round(up_test, 4)) + ' for the upstream gene ' + up_gene
          + ' and ' + str(round(down_test, 4)) + ' for the downstream gene '
          + down_gene + ' when considered separately.')

    # test genes' mutations with the other gene's mutations excluded
    up_test = cdata.score_clf(
        clf=clf(mut_genes=[up_gene], expr_genes=expr_genes),
        mtype=up_mtype, score_indx=score_indx, tune_indx=tune_indx,
        exclude_samps=down_mtype.get_samples(cdata.train_mut_))
    down_test = cdata.score_clf(
        clf=clf(mut_genes=[down_gene], expr_genes=expr_genes),
        mtype=down_mtype, score_indx=score_indx, tune_indx=tune_indx,
        exclude_samps=up_mtype.get_samples(cdata.train_mut_))
    print('We get AUCs of ' + str(round(up_test, 4)) + ' for the upgene and '
          + str(round(down_test, 4)) + ' for the downgene '
          + 'when considered exclusively of one another.')

    up_subtypes = cdata.train_mut_.subsets(
        mtype=up_mtype, levels=['Gene','Form'])
    down_subtypes = cdata.train_mut_.subsets(
        mtype=down_mtype, levels=['Gene','Form'])
    print('Subtype classification:')
    for mtype in up_subtypes:
        if len(mtype.get_samples(cdata.train_mut_)) >= 10:
            print(mtype)
            print(str(round(cdata.score_clf(
                clf=clf(mut_genes=[up_gene], expr_genes=expr_genes),
                mtype=mtype, score_indx=score_indx, tune_indx=tune_indx), 4)))
    for mtype in down_subtypes:
        if len(mtype.get_samples(cdata.train_mut_)) >= 10:
            print(mtype)
            print(str(round(cdata.score_clf(
                clf=clf(mut_genes=[down_gene], expr_genes=expr_genes),
                mtype=mtype, score_indx=score_indx, tune_indx=tune_indx), 4)))

    down_samps = down_mtype.get_samples(cdata.train_mut_)
    up_samps = up_mtype.get_samples(cdata.train_mut_)
    both_samps = list(set(down_samps) & set(up_samps))
    null_samps = list(
        set(cdata.train_expr_.index) - set(down_samps) - set(up_samps))
    up_pred = cdata.predict_clf(
        clf=clf(mut_genes=[up_gene], expr_genes=expr_genes),
        mtype=up_mtype, pred_indx=pred_indx, tune_indx=tune_indx,
        exclude_samps=down_samps)
    down_pred = cdata.predict_clf(
        clf=clf(mut_genes=[down_gene], expr_genes=expr_genes),
        mtype=down_mtype, pred_indx=pred_indx, tune_indx=tune_indx,
        exclude_samps=up_samps)
    print('Intersection: '
          + str(round(np.mean(up_pred[both_samps].tolist()), 3))
          + '\n' + 'Upgene: '
          + str(round(np.mean(up_pred[up_samps].tolist()), 3))
          + '\n' + 'Downgene: '
          + str(round(np.mean(up_pred[down_samps].tolist()), 3))
          + '\n' + 'Neither: '
          + str(round(np.mean(up_pred[null_samps].tolist()), 3)))
    print('Intersection: '
          + str(round(np.mean(down_pred[both_samps].tolist()), 3))
          + '\n' + 'Upgene: '
          + str(round(np.mean(down_pred[up_samps].tolist()), 3))
          + '\n' + 'Downgene: '
          + str(round(np.mean(down_pred[down_samps].tolist()), 3))
          + '\n' + 'Neither: '
          + str(round(np.mean(down_pred[null_samps].tolist()), 3)))

    total_time = time.time() - start_time
    print('Run time: ' + str(round(total_time, 1)) + ' seconds.')


if __name__ == "__main__":
    main(sys.argv[1:])


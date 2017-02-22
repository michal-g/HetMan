
import pickle
import time
import sys
import os
sys.path += ['/home/users/grzadkow/compbio/scripts/HetMan']

import synapseclient
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
    syn = synapseclient.Synapse()
    syn.login('grzadkow', 'W0w6g1i8A')
    cdata = data.MutExpr(
        syn, cohort=argv[0], mut_genes=[up_gene,down_gene],
        cv_info={'Prop':2/3.0, 'Seed':int(argv[-1])+1})

    # define the classifier to be used as well as the interval
    # cross-validation training samples for testing and tuning
    clf_lbl = str(argv[3])
    if clf_lbl in dir(classif):
        clf = eval('classif.' + clf_lbl)
    else:
        raise InputError("Unknown classifier specified!")
    score_indx = list(range(32))
    pred_indx = list(range(32))
    tune_indx = list(range(55,60))

    # test genes' mutations individually
    up_test = cdata.score_clf(
        clf=clf([up_gene]), mtype=up_mtype,
        score_indx=score_indx, tune_indx=tune_indx)
    down_test = cdata.score_clf(
        clf=clf([down_gene]), mtype=down_mtype,
        score_indx=score_indx, tune_indx=tune_indx)
    print('Using the ' + clf_lbl + ' classifier, we get AUCs of '
          + str(round(up_test, 4)) + ' for the upstream gene ' + up_gene
          + ' and ' + str(round(down_test, 4)) + ' for the downstream gene '
          + down_gene + ' when considered separately.')

    # test genes' mutations with the other gene's mutations excluded
    up_test = cdata.score_clf(
        clf=clf([up_gene]), mtype=up_mtype,
        score_indx=score_indx, tune_indx=tune_indx,
        exclude_samps=down_mtype.get_samples(cdata.train_mut_))
    down_test = cdata.score_clf(
        clf=clf([down_gene]), mtype=down_mtype,
        score_indx=score_indx, tune_indx=tune_indx,
        exclude_samps=up_mtype.get_samples(cdata.train_mut_))
    print('We get AUCs of ' + str(round(up_test, 4)) + ' for the upgene and '
          + str(round(down_test, 4)) + ' for the downgene '
          + 'when considered exclusively of one another.')

    up_pred = cdata.predict_clf(
        clf=clf([up_gene]), mtype=up_mtype,
        pred_indx=pred_indx, tune_indx=tune_indx,
        exclude_samps=down_mtype.get_samples(cdata.train_mut_))
    down_pred = cdata.predict_clf(
        clf=clf([down_gene]), mtype=down_mtype,
        pred_indx=pred_indx, tune_indx=tune_indx,
        exclude_samps=up_mtype.get_samples(cdata.train_mut_))

    total_time = time.time() - start_time
    print('Run time: ' + str(round(total_time, 1)) + ' seconds.')

if __name__ == "__main__":
    main(sys.argv[1:])


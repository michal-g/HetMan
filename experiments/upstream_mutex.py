
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

    # define interval cross-validation training samples for testing and
    # tuning, test genes' mutations individually
    test_indx = list(range(32))
    tune_indx = list(range(55,60))
    up_test = cdata.test_classif_cv(
        classif=classif.Lasso(), mtype=up_mtype,
        test_indx=test_indx, tune_indx=tune_indx)
    down_test = cdata.test_classif_cv(
        classif=classif.Lasso(), mtype=down_mtype,
        test_indx=test_indx, tune_indx=tune_indx)

    # output classification statistics
    total_time = time.time() - start_time
    print('Classified individually, we get AUCs of '
          + str(round(up_test, 4)) + ' for the upstream gene ' + up_gene
          + ' and ' + str(round(down_test, 4)) + ' for the downstream gene '
          + down_gene)
    print('Run time: ' + str(round(total_time, 1)) + ' seconds.')

if __name__ == "__main__":
    main(sys.argv[1:])


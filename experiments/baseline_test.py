
import pickle
import time
import sys
import os
import synapseclient
import pandas as pd

sys.path += ['/home/users/grzadkow/compbio/scripts/HetMan']
from cohorts import Cohort
from mutation import MuType
import classif


def main(argv):

    # define which mutations we want to consider in our test
    mtypes = {
        'TP53': MuType(
            {('Gene', 'TP53'):{('Form', 'Missense_Mutation'):None}}),
        'PIK3CA': MuType(
            {('Gene', 'PIK3CA'):{('Protein', 'p.H1047R'):None}}),
        'CDH1': MuType(
            {('Gene', 'CDH1'):{('Form', 'Frame_Shift'):None}})
        }
    key_list = {'All': None, 'Down': ((['Down'], ()), )}

    # load in expression and mutation datasets
    syn = synapseclient.Synapse()
    syn.login('grzadkow', 'W0w6g1i8A')
    cdata = Cohort(
        syn, cohort=argv[0], mut_genes=['TP53', 'PIK3CA', 'CDH1'],
        mut_levels=['Gene','Form','Protein'],
        cv_info={'Prop':2/3.0, 'Seed':int(argv[-1])+1})

    # define the classifiers to be used as well as the interval
    # cross-validation training samples for testing and tuning
    clf_list = [classif.NaiveBayes, classif.Lasso,
                classif.SVCrbf, classif.rForest]

    scores = {}
    times = {}

    for clf in clf_list:
        clf_lbl = clf.__name__
        scores[clf_lbl] = {}
        times[clf_lbl] = {}

        for mut_lbl, mtype in mtypes.items():
            scores[clf_lbl][mut_lbl] = {}
            times[clf_lbl][mut_lbl] = {}

            for key_lbl, k in key_list.items():
                start_time = time.time()
                scores[clf_lbl][mut_lbl][key_lbl] = round(cdata.score_clf(
                    clf(mut_gene=mut_lbl, path_keys=k),
                    score_splits=32, tune_splits=4, mtype=mtype), 4)
                times[clf_lbl][mut_lbl][key_lbl] = round(
                    time.time() - start_time, 1)

    print(pd.DataFrame(scores))
    print(pd.DataFrame(times))


if __name__ == "__main__":
    main(sys.argv[1:])



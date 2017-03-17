
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
    mtypes = [
        MuType({('Gene', 'TP53'):{('Form', 'Missense_Mutation'):None}}),
        MuType({('Gene', 'PIK3CA'):{('Protein', 'p.H1047R'):None}})
        ]
    mut_genes = [list(list(mtype.child.keys())[0])[0] for mtype in mtypes]
    key_list = {'All': None,
                'Up': ((['Up'], ()), ),
                'Neigh': ((['Up', 'Down'], ()), ),
                'expr': (((), ['controls-expression-of']), ),
                'Down': ((['Down'], ()), )
               }

    # load in expression and mutation datasets
    syn = synapseclient.Synapse()
    syn.login('grzadkow', 'W0w6g1i8A')
    cdata = Cohort(
        syn, cohort=argv[0], mut_genes=mut_genes,
        mut_levels=['Gene','Form','Protein'],
        cv_info={'Prop':2/3.0, 'Seed':int(argv[-1])+1})

    # define the classifiers to be used as well as the interval
    # cross-validation training samples for testing and tuning
    clf_list = [classif.NaiveBayes, classif.Lasso,
                classif.SVCrbf, classif.rForest, classif.PCpipe]

    scores = {}
    times = {}

    for clf in clf_list:
        clf_lbl = clf.__name__
        scores[clf_lbl] = {}
        times[clf_lbl] = {}

        for mut_gene, mtype in zip(mut_genes, mtypes):
            scores[clf_lbl][mut_gene] = {}
            times[clf_lbl][mut_gene] = {}

            for key_lbl, k in key_list.items():
                clf_obj = clf(path_keys=k)
                start_time = time.time()
                cdata.tune_clf(clf_obj, mtype=mtype,
                               tune_splits=4, test_count=32)
                print(clf_obj.named_steps['fit'])

                cdata.fit_clf(clf_obj, mtype=mtype)
                scores[clf_lbl][mut_gene][key_lbl] = cdata.eval_clf(
                    clf_obj, mtype=mtype)
                times[clf_lbl][mut_gene][key_lbl] = time.time() - start_time

    out_file = ('/home/users/grzadkow/compbio/scripts/HetMan/'
                + 'experiments/output/base_'
                + argv[-2] + '_' + argv[-1] + '_data.p')
    out_data = {'AUC': scores, 'time': times}
    pickle.dump(out_data, open(out_file, 'wb'))


if __name__ == "__main__":
    main(sys.argv[1:])



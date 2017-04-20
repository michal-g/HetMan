
"""
HetMan (Heterogeneity Manifold)
Classification of mutation sub-types using expression data.
This file contains a series of baseline tests of classifier performance.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

import sys
sys.path += ['/home/users/grzadkow/compbio/scripts/']
from HetMan.cohorts import Cohort
from HetMan.experiments.baseline.config import mtype_list, clf_list

import pickle
import time
import synapseclient


def main(argv):
    """Runs the experiment."""
    syn = synapseclient.Synapse()
    syn.login('grzadkow', 'W0w6g1i8A')

    clfs = clf_list[argv[0]]
    mtypes = mtype_list[argv[1]]
    scores = {}
    times = {}

    for coh in set([coh for coh, _ in mtypes]):
        coh_mtypes = [mtype for ch, mtype in mtypes if ch == coh]
        coh_genes = [tuple(tuple(mtype.child.keys())[0])[0]
                     for mtype in coh_mtypes]

        # load in expression and mutation datasets
        cdata = Cohort(
            syn, cohort=coh, mut_genes=coh_genes,
            mut_levels=['Gene', 'Form', 'Protein'],
            cv_info={'Prop':2/3.0, 'Seed':int(argv[-1])+1})


        for clf in clfs:
            for mut_gene, mtype in zip(coh_genes, coh_mtypes):

                # tune the hyper-parameters of the classifier using the
                # training samples
                clf_obj = clf(path_keys=None)
                start_time = time.time()
                cdata.tune_clf(clf_obj, mtype=mtype,
                               tune_splits=4, test_count=32)
                print(clf_obj)

                # fit the tuned classifier and score using the testing samples
                cdata.fit_clf(clf_obj, mtype=mtype)
                scores[(clf_obj, (coh,mtype))] = cdata.eval_clf(
                    clf_obj, mtype=mtype)
                times[(clf_obj, (coh,mtype))] = time.time() - start_time

    # saves classifier results to file
    out_file = ('/home/users/grzadkow/compbio/scripts/HetMan/experiments/'
                'baseline/output/' + argv[0] + '_' + argv[1]
                + '__run' + argv[-1] + '.p')
    out_data = {'AUC': scores, 'time': times}
    pickle.dump(out_data, open(out_file, 'wb'))


if __name__ == "__main__":
    main(sys.argv[1:])



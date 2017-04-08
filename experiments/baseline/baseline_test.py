
"""
HetMan (Heterogeneity Manifold)
Classification of mutation sub-types using expression data.
This file contains a series of baseline tests of classifier performance.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

import sys
sys.path += ['/home/users/grzadkow/compbio/scripts']

from HetMan.cohorts import Cohort
from HetMan.mutation import MuType
import HetMan.classifiers as classif

import pickle
import time
import synapseclient


def main(argv):
    """Runs the experiment."""
    syn = synapseclient.Synapse()
    syn.login('grzadkow', 'W0w6g1i8A')

    # which mutations we want to consider in our test
    mtypes = (
        ('BRCA', MuType({('Gene', 'TP53'):
                         {('Form', 'Missense_Mutation'): None}})),
        ('BRCA', MuType({('Gene', 'PIK3CA'):
                         {('Protein', 'p.H1047R'): None}})),
        ('BRCA', MuType({('Gene', 'CDH1'):
                         {('Form', ('Frame_Shift_Ins', 'Frame_Shift_Del')):
                          None}})),
        ('SKCM', MuType({('Gene', 'BRAF'):
                         {('Protein', 'p.V600E'): None}})),
        ('COAD', MuType({('Gene', 'TTN'):
                         {('Form', 'Intron'): None}})),
        ('UCEC', MuType({('Gene', 'PTEN'):
                         {('Form', ('Frame_Shift_Del', 'Nonsense_Mutation')):
                          None}})),
        )

    # which classifiers we want to consider in our test
    clf_list = [classif.NaiveBayes, classif.Lasso,
                classif.SVCrbf, classif.rForest]
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


        for clf in clf_list:
            clf_lbl = clf.__name__
            scores[clf_lbl] = {}
            times[clf_lbl] = {}

            for mut_gene, mtype in zip(coh_genes, coh_mtypes):
                mut_lbl = coh + '_' + mut_gene
                scores[clf_lbl][mut_lbl] = {}
                times[clf_lbl][mut_lbl] = {}

                # tune the hyper-parameters of the classifier using the
                # training samples
                clf_obj = clf(path_keys=None)
                start_time = time.time()
                cdata.tune_clf(clf_obj, mtype=mtype,
                               tune_splits=4, test_count=32)
                print(clf_obj)

                # fit the tuned classifier and score using the testing samples
                cdata.fit_clf(clf_obj, mtype=mtype)
                scores[clf_lbl][mut_lbl] = cdata.eval_clf(
                    clf_obj, mtype=mtype)
                times[clf_lbl][mut_lbl] = time.time() - start_time

    out_file = ('/home/users/grzadkow/compbio/scripts/HetMan/experiments/'
                'baseline/output/base__run' + argv[-1] + '.p')
    out_data = {'AUC': scores, 'time': times}
    pickle.dump(out_data, open(out_file, 'wb'))


if __name__ == "__main__":
    main(sys.argv[1:])



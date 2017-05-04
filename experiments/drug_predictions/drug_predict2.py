
import sys
import pickle
sys.path += ['/home/users/grzadkow/compbio/scripts']

from HetMan.cohorts import Cohort
from HetMan.drugs import *
from HetMan.mutation import MuType

import numpy as np
import pandas as pd

from math import log10
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score
from copy import deepcopy

base_dir = ('/home/users/grzadkow/compbio/scripts/HetMan/'
            'experiments/drug_predictions')


def main(argv):
    """Runs the experiment."""
    syn = synapseclient.Synapse()
    syn.login('grzadkow', 'W0w6g1i8A')

    # load drug-mutation association data,
    # filter for pan-cancer associations
    drug_ints = pd.read_csv(base_dir + '/input/drug_data.txt',
                            sep='\t', comment='#')
    drug_ints = drug_ints.ix[drug_ints['PANCAN'] != 0, :]

    # categorize associations by mutation type
    pnt_indx = drug_ints['FEAT'].str.contains('_mut$')
    cnv_indx = drug_ints['FEAT'].str.contains('^(?:loss|gain):')
    fus_indx = drug_ints['FEAT'].str.contains('_fusion$')

    # get list of genes affected by point mutations, load TCGA cohort
    # with corresponding set of mutations
    pnt_genes = list(set(
        x[0] for x in drug_ints['FEAT'][pnt_indx].str.split('_')))
    cdata = Cohort(syn, cohort=argv[0], mut_genes=pnt_genes,
                   mut_levels=['Gene', 'Form', 'Protein'],
                   cv_info={'Prop':1, 'Seed':int(argv[-1])+1})
    cdata.train_expr_ = exp_norm(cdata.train_expr_)
    cdata_expr = deepcopy(cdata.train_expr_)

    # get list of point mutation types and drugs associated with at least one
    pnt_mtypes = [
        MuType({('Gene', gn):
                {('Form', ('Nonsense_Mutation', 'Missense_Mutation')): None}}
              ) for gn in pnt_genes]
    pnt_muts = {(gn + '_mut'):mtype for gn,mtype
                in zip(pnt_genes, pnt_mtypes)
                if len(mtype.get_samples(cdata.train_mut_)) >= 3}
    pnt_drugs = list(set(
        drug_ints['DRUG'][pnt_indx][drug_ints['FEAT'][pnt_indx].
                                    isin(pnt_muts.keys())]))

    # array that stores classifier performance on held-out cell lines ...
    response_vec = pd.Series(float('nan'), index=pnt_drugs)

    # ... stores predicted drug response for organoid sample ...
    predict_vec = pd.Series(float('nan'), index=pnt_drugs)

    # ... stores trained classifier instances ...
    clf_vec = {}

    # ... stores t-test p-values for mutation state vs predicted
    # drug responses in TCGA cohort ...
    resp_mat = pd.DataFrame(float('nan'),
                            index=pnt_drugs, columns=pnt_muts.keys())

    # ... stores AUC scores for mutation vs drug response in TCGA ...
    auc_mat = pd.DataFrame(float('nan'),
                           index=pnt_drugs, columns=pnt_muts.keys())

    # loads organoid RNAseq data
    pred_data = pd.read_csv(
        '/home/users/grzadkow/compbio/input-data/rnaseq_4315.csv',
        header=0)

    for drug in pnt_drugs:
        print("Testing drug " + drug + " ....")
        drug_clf = eval(argv[1])()

        # loads cell line drug response and array expression data
        coh = DrugCohort(drug, source='ioria', random_state=int(argv[-1]))
        cdata.train_expr_ = deepcopy(cdata_expr)
        use_genes = (set(cdata.train_expr_.columns)
                     & set(coh.drug_expr.columns))

        # processes organoid RNAseq data to match TCGA and drug response data
        pred_samp = pred_data.ix[pred_data['Symbol'].isin(use_genes), :]
        pred_samp.loc[:, 'RPKM'] = (
            pred_samp.loc[:, 'RPKM']
            + min(pred_samp.loc[:, 'RPKM'][pred_samp.loc[:,'RPKM'] > 0]) / 2)
        pred_samp = pred_samp.groupby(['Symbol'])['RPKM'].mean()
        pred_samp = exp_norm(pd.DataFrame(pred_samp).transpose())

        use_genes &= set(pred_samp.columns)
        cdata.train_expr_ = cdata.train_expr_.loc[:, use_genes]
        coh.drug_expr = coh.drug_expr.loc[:, use_genes]
        pred_samp = pred_samp.loc[:, use_genes]

        # tunes and fits the classifier on the CCLE data, and evaluates its
        # performance on the held-out samples
        coh.tune_clf(drug_clf)
        coh.fit_clf(drug_clf)
        response_vec[drug] = coh.eval_clf(drug_clf)

        # predicts drug response for the organoid, stores classifier
        # for later use
        predict_vec[drug] = drug_clf.predict(pred_samp)[0]
        clf_vec[drug] = drug_clf

        for gn, mtype in pnt_muts.items():
            # for each mutated gene, get the vector of mutation status
            # for the TCGA samples
            mut_stat = np.array(
                cdata.train_mut_.status(cdata.train_expr_.index,
                                        mtype=mtype)
                )

            # gets the classifier's predictions of drug response for the
            # TCGA cohort, and evaluate its concordance with mutation status
            tcga_pred = drug_clf.predict(cdata.train_expr_)
            resp_mat.loc[drug, gn] = -log10(
                ttest_ind(tcga_pred[mut_stat], tcga_pred[~mut_stat],
                          equal_var=False)[1]
                )
            auc_mat.loc[drug, gn] = roc_auc_score(mut_stat, tcga_pred)

    # save everything to file
    out_data = {'Response': response_vec, 'Pred': predict_vec, 'Clf': clf_vec,
                'T-Test': resp_mat, 'AUC': auc_mat}
    out_file = ('/home/users/grzadkow/compbio/scripts/HetMan/experiments/'
                'drug_predictions/output/mat_' + argv[0] + '_' + argv[1]
                + '__run' + argv[-1] + '.p')
    pickle.dump(out_data, open(out_file, 'wb'))


if __name__ == "__main__":
        main(sys.argv[1:])



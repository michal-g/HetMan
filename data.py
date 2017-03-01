
"""
Hetman (Heterogeneity Manifold)
Classification of mutation sub-types using expression data.
This file contains functions for reading in transcriptomic and genomic data
downloaded from sources such as TCGA, ICGC, and Firehose.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

import numpy as np
import pandas as pd
import random
import defunct
import re
import sys

from sklearn import model_selection
from scipy.stats import fisher_exact
from itertools import groupby
from functools import reduce
from mutation import MutLevel, MuTree
from bioservices import PathwayCommons 


# .. directories containing raw -omics data and cross-validation samples ..
_base_dir = '/home/users/grzadkow/compbio/'
_icgc_dir = _base_dir + 'input-data/ICGC/raw/'
_cv_dir = _base_dir + 'auxiliary/HetMan/cv-samples/'
_firehose_dir = _base_dir + 'input-data/analyses__2016_01_28/'


# .. helper functions for parsing -omic datasets ..
def parse_tcga_barcodes(barcodes):
    """Extracts the sample labels from TCGA barcodes."""
    return [reduce(lambda x,y: x + '-' + y,
                   s.split('-', 4)[:4])
            for s in barcodes]


def log_norm_expr(expr):
    """Log-normalizes expression data."""
    log_add = np.amin(expr[expr > 0]) * 0.5
    return np.log2(expr + log_add)


def get_annot(version='v19'):
    """Gets annotation data for protein-coding genes on non-sex
       chromosomes from a Gencode file.

    Parameters
    ----------
    version : str, 'v*', optional (default='v19')
        Which version of the Gencode data to use, must correspond
        to a file in input-data/.
        Default is to use v19, which matches what is used for the data
        available at the ICGC data portal (11/09/2016).

    Returns
    -------
    annot : dict
        Dictionary with keys corresponding to Ensembl gene IDs and values
        consisting of dicts with annotation fields.
    """
    annot = pd.read_csv(
        _base_dir + 'input-data/gencode.' + version + '.annotation.gtf.gz',
        usecols=[0,2,3,4,8], names=['Chr', 'Type', 'Start', 'End', 'Info'],
        sep = '\t', header=None, comment='#'
        )

    # filter out annotation records that aren't
    # protein-coding genes on non-sex chromosomes
    chroms_use = ['chr' + str(i+1) for i in range(22)]
    annot = annot.loc[annot['Type'] == 'gene', ]
    chr_indx = np.array([chrom in chroms_use for chrom in annot['Chr']])
    annot = annot.loc[chr_indx, ]

    # parse the info field to get each gene's annotation data
    gn_annot = {
        re.sub('\.[0-9]+', '', z['gene_id']).replace('"', ''):z
        for z in [dict([['chr', an[0]]] +
                       [['Start', an[2]]] +
                       [['End', an[3]]] +
                       [y for y in [x.split(' ')
                                    for x in an[4].split('; ')]
                        if len(y) == 2])
                  for an in annot.values]
        if z['gene_type'] == '"protein_coding"'
        }
    for g in gn_annot:
        gn_annot[g]['gene_name'] = gn_annot[g]['gene_name'].replace('"', '')

    return gn_annot


# .. functions for reading in mRNA expression datasets ..
def get_expr_firehose(cohort):
    """Gets expression data as a matrix from a Firehose GDAC fzile."""
    expr_file = (
        _firehose_dir + cohort
        + '/gdac.broadinstitute.org_BRCA.Merge_rnaseqv2__illuminahiseq_'
        + 'rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.'
        + 'Level_3.2016012800.0.0.tar.gz'
        )
    raw_data = pd.read_csv(expr_file, sep='\t', dtype=object)

    # parses gene and sample names to get expression matrix axis labels
    gene_names = [re.sub('\|.*$', '', str(x)) for x in raw_data.ix[:,0]]
    gene_indx = [x not in ['gene_id','nan','?'] for x in gene_names]
    expr_data = raw_data.ix[gene_indx, 1:]
    expr_data.index = pd.Series(gene_names).loc[gene_indx]
    expr_data.columns = parse_tcga_barcodes(expr_data.columns)

    return expr_data.T.astype('float')


def get_expr_icgc(expr_file):
    """Gets expression data as a matrix from an ICGC tsv.gz file.

    Parameters
    ----------
    expr_file : str
        A file containing expression data.

    Returns
    -------
    expr : ndarray, shape (n_samples, n_features)
        An expression matrix with genes as features, in the case of
        duplicate gene names, values are averaged.
    """
    expr = pd.read_csv(
        expr_file, usecols=(4,7,8), header=0, sep = '\t',
        names=('Sample', 'Gene', 'FPKM')
        )

    # gets patient IDs from TCGA barcodes
    expr['Sample'] = [reduce(lambda x,y: x + '-' + y,
                             s.split('-', 3)[:3])
                      for s in expr['Sample']]
    
    # transforms raw long-format expression data into wide-format
    return expr.pivot_table(index='Sample', columns='Gene',
                            values='FPKM', aggfunc=np.mean)


# .. functions for reading in mutation datasets ..
def get_mut_mc3(syn, mut_levels=('Gene', 'Form', 'Exon')):
    """Reads ICGC mutation data from the MC3 synapse file.

    Parameters
    ----------
    syn : object
        An instance of Synapse that has already been logged into.

    mut_levels : tuple of strs
        A list of mutation levels, must be in the keys of the _mc3_levels
        dict defined above.

    Returns
    -------
    muts : ndarray, shape (n_mutations, mut_levels+1)
        A mutation array, with a row for each mutation appearing in an
        individual sample.
    """
    # columns in the MC3 file containing each level in the mutation hierarchy
    mut_cols = {
        MutLevel.Gene: 0,
        MutLevel.Form: 8,
        MutLevel.PolyPhen: 72,
        MutLevel.Exon: 38,
        MutLevel.Protein: 36
        }

    # gets data from Synapse, figures out which columns to use
    mc3 = syn.get('syn7824274')
    data_names = ['Sample'] + list(mut_levels)
    use_cols = [15]
    for lvl in mut_levels:
        use_cols.append(mut_cols[MutLevel[lvl]])
    data_names = np.array(data_names)[np.array(use_cols).argsort()]

    # imports data into a DataFrame, parses TCGA sample barcodes
    # and PolyPhen scores
    muts = pd.read_csv(
        mc3.path, usecols=use_cols, sep='\t', header=None,
        names=data_names, comment='#', skiprows=1)
    muts['Sample'] = parse_tcga_barcodes(muts['Sample'])
    if 'PolyPhen' in mut_levels:
        muts['PolyPhen'] = [re.sub('\)$', '', re.sub('^.*\(', '', x))
                            if x != '.' else 0
                            for x in muts['PolyPhen']]

    return muts


def get_mut_icgc(mut_file):
    """Gets mutation data as an numpy array from an ICGC tsv.gz file.
       Deprecated in favour of using MC3 mutation calls.

    Parameters
    ----------
    mut_file : str
        A file containing mutation data.

    Returns
    -------
    mut : ndarray, shape (n_mutations,)
        A 1-D array with each entry corresponding to a single mutation
        affecting a single transcript in a single sample.
    """
    mut_dt = np.dtype(
        [('Name', np.str_, 16), ('Sample', np.str_, 16),
         ('Position', np.int), ('Consequence', np.str_, 46),
         ('Protein', np.str_, 16), ('Gene', np.str_, 16),
         ('Transcript', np.str_, 16)]
        )
    return np.loadtxt(
        fname=mut_file,
        dtype=mut_dt, skiprows=1,
        delimiter='\t',
        usecols=(0,1,9,25,26,28,29)
        )


def get_cnv_icgc(cnv_file):
    """Gets copy number variation as an numpy array from an ICGC tsv.gz file.

    Parameters
    ----------
    cnv_file : str
        A file containing copy number variation data.

    Returns
    -------
    cnv_data : ndarray, shape (n_cnvs,)
        A 1-D array with each entry corresponding to a single copy number
        variation affecting a single sample.
    """
    cnv_dt = np.dtype([('Sample',np.str_,64), ('Mean',np.float),
                       ('Chr',np.str_,64), ('Start',np.int), ('End',np.int)])
    return np.loadtxt(fname=cnv_file, dtype=cnv_dt, skiprows=1,
                      delimiter='\t', usecols=(0,9,11,12,13))


def get_cnv_gdac(cohort):
    """Gets CNV data as a matrix from a Firehose GDAC file."""
    cnv_file = (_firehose_dir + cohort
                + '/GDAC_Gistic2Report_22529547_broad_data_by_genes.txt')
    raw_data = pd.read_csv(cnv_file, sep='\t')

    # parses gene and sample names into CNV matrix axis labels
    gene_names = [re.sub('\|.*$', '', str(x))
                  for x in pd.Series(raw_data.ix[1:, 0])]
    cnv_data = raw_data.ix[1:, 3:]
    cnv_data.index = gene_names
    cnv_data.columns = parse_tcga_barcodes(cnv_data.columns)

    return cnv_data.T


# .. functions for reading in pathway data ..
def get_pc2_neighb(gene):
    """Gets the neighbourhood of a gene as defined by Pathway Commons."""
    pc2 = PathwayCommons(verbose=False)
    pc2.settings.TIMEOUT = 1000
    neighb = {}

    # sets the parameters of the PC2 query
    url = 'graph'
    gene_id = pc2.idmapping(gene)[gene]
    params = {'format': 'BINARY_SIF',
              'kind': 'neighborhood',
              'limit': 1,
              'source': 'http://identifiers.org/uniprot/' + gene_id}

    # runs the PC2 query, makes sure the output has the correct format
    raw_data = 0
    while isinstance(raw_data, int):
        print("Reading in Pathway Commons data for gene " + gene + "...")
        raw_data = pc2.http_get(url, frmt=None, params=params)
        print(type(raw_data))

    print("Moving on...")
    # parses interaction data according to direction
    sif_data = raw_data.splitlines()
    sif_data = [x.split() for x in sif_data]
    up_neighbs = sorted([(x[1], x[0]) for x in sif_data if x[2] == gene],
                        key=lambda x: x[0])
    down_neighbs = sorted([(x[1], x[2]) for x in sif_data if x[0] == gene],
                          key=lambda x: x[0])
    other_neighbs = sorted([x for x in sif_data
                            if x[0] != gene and x[2] != gene],
                           key=lambda x: x[1])

    # parses according to interaction type
    neighb['Up'] = {k:[x[1] for x in v] for k,v in
                    groupby(up_neighbs, lambda x: x[0])}
    neighb['Down'] = {k:[x[1] for x in v] for k,v in
                      groupby(down_neighbs, lambda x: x[0])}
    neighb['Other'] = {k:[(x[0],x[2]) for x in v] for k,v in
                       groupby(other_neighbs, lambda x: x[1])}

    return neighb


def read_sif(mut_genes,
             sif_file='input-data/babur-mutex/data-tcga/Network.sif'):
    """Gets the edges containing at least one of given genes from a SIF
       pathway file and arranges them according to the direction of the
       edge and the type of interaction it represents.

    Parameters
    ----------
    mut_genes : array-like, shape (n_genes,)
        A list of genes whose interactions are to be retrieved.

    sif_file : str, optional
        A file in SIF format describing gene interactions.
        The default is the set of interactions used in the MutEx paper.

    Returns
    -------
    link_data : dict
        A list of the interactions that involve one of the given genes.
    """
    if isinstance(mut_genes, str):
        mut_genes = [mut_genes]
    sif_dt = np.dtype(
        [('Gene1', np.str_, 16),
         ('Type', np.str_, 32),
         ('Gene2', np.str_, 16)])
    sif_data = np.loadtxt(
        fname = _base_dir + sif_file, dtype = sif_dt, delimiter = '\t')
    link_data = {g:{'in':None, 'out':None} for g in mut_genes}

    for gene in mut_genes:
        in_data = np.sort(sif_data[sif_data['Gene2'] == gene],
                          order='Type')
        out_data = np.sort(sif_data[sif_data['Gene1'] == gene],
                           order='Type')
        link_data[gene]['in'] = {k:[x['Gene1'] for x in v] for k,v in
                                 groupby(in_data, lambda x: x['Type'])}
        link_data[gene]['out'] = {k:[x['Gene2'] for x in v] for k,v in
                                  groupby(out_data, lambda x: x['Type'])}
    return link_data


# .. classes for combining different datasets ..
class MutExpr(object):
    """A class corresponding to expression and mutation data
       for a given TCGA cohort.

    Parameters
    ----------
    syn : synapseclient object
        A logged-into instance of the synapseclient.Synapse() class.

    cohort : str
        An ICGC/TCGA cohort, i.e. 'BRCA' or 'UCEC' available for download
        in Broad Firehose.

    mut_genes : list of strs
        A list of genes whose mutations we want to consider,
        i.e. ['TP53','KRAS'].

    mut_levels : tuple, optional
        A list of mutation levels we want to consider, see
        MuTree and MuType above.

    cv_info : {'Prop': float in (0,1), 'Seed': int}
        A dictionary giving the proportion of samples to use for training
        in cross-validation, and the seed to use for the random choice
        of training samples.

    Attributes
    ----------
    intern_cv_ : int
        Which seed to use for internal cross-validation sampling of the
        training set.

    train_expr_ : array-like, shape=(n_samples,n_tr_features)
        The subset of expression data used for training of classifiers.

    test_expr_ : array-like, shape=(n_samples,n_tst_features)
        The subset of expression data used for testing of classifiers.

    train_mut_ : MuTree
        Hierarchy of mutations present in the training samples.

    test_mut_ : MuTree
        Hierarchy of mutations present in the testing samples.
    """

    def __init__(self,
                 syn, cohort, mut_genes,
                 mut_levels=('Gene', 'Form', 'Protein'), cv_info=None):
        self.cohort_ = cohort
        if cv_info is None:
            cv_info = {'Prop': 2.0/3, 'Seed':1}
        self.intern_cv_ = cv_info['Seed'] ** 2
        self.mut_genes = mut_genes

        # loads gene expression and mutation data
        annot = get_annot()
        expr = get_expr_firehose(cohort)
        muts = get_mut_mc3(syn, list(set(mut_levels) - set(['GISTIC'])))
        cnvs = get_cnv_gdac(cohort)

        # filters out genes that are not expressed in any samples, don't have
        # any variation across the samples, are not included in the
        # annotation data, or are not in the mutation datasets
        expr = expr.loc[:, expr.apply(
            lambda x: np.mean(x) > 0 and np.var(x) > 0, axis=0)]
        annot = {g:a for g,a in annot.items()
                 if a['gene_name'] in expr.columns}
        annot_genes = [a['gene_name'] for g,a in annot.items()]
        expr = expr.loc[:, annot_genes]
        expr = expr.loc[:,~expr.columns.duplicated()]
        muts = muts.loc[muts['Sample'].isin(expr.index), :]
        muts = muts.loc[muts['Sample'].isin(cnvs.index), :]

        # gets set of samples shared across expression and mutation datasets,
        # subsets these datasets to use only these samples
        self.samples = set(muts['Sample']) & set(expr.index) & set(cnvs.index)
        expr = expr.loc[self.samples, :]
        cnvs = cnvs.loc[self.samples, mut_genes]

        # merges simple somatic mutations with CNV calls
        cnvs['Sample'] = cnvs.index
        cnvs = pd.melt(cnvs, id_vars=['Sample'],
                       value_name='GISTIC', var_name='Gene')
        cnvs = cnvs.loc[cnvs['GISTIC'] != 0, :]
        cnvs['Form'] = ['Gain' if x > 0 else 'Loss' for x in cnvs['GISTIC']]
        muts = pd.concat(objs=(muts, cnvs), axis=0,
                         join='outer', ignore_index=True)

        # gets annotation data for the genes whose mutations
        # are under consideration
        annot_data = {mut_g:{'ID':g, 'Chr':a['chr'],
                             'Start':a['Start'], 'End':a['End']}
                      for g,a in annot.items() for mut_g in mut_genes
                      if a['gene_name'] == mut_g}
        annot_ids = {k:v['ID'] for k,v in annot_data.items()}
        self.annot = annot

        # gets subset of samples to use for training
        random.seed(a=cv_info['Seed'])
        self.cv_seed = random.getstate()
        self.train_samps_ = frozenset(
            random.sample(population=self.samples,
                          k=int(round(len(self.samples) * cv_info['Prop'])))
            )

        # creates training and testing expression and mutation datasets
        self.train_expr_ = log_norm_expr(
            expr.loc[self.train_samps_, :])
        self.test_expr_ = log_norm_expr(
            expr.loc[self.samples - self.train_samps_, :])
        self.train_mut_ = MuTree(
            muts=muts, samples=self.train_samps_,
            genes=mut_genes, levels=mut_levels
            )
        self.test_mut_ = MuTree(
            muts=muts, samples=(self.samples - self.train_samps_),
            genes=mut_genes, levels=mut_levels
            )

    def mutex_test(self, mtype1, mtype2):
        """Checks the mutual exclusivity of two mutation types in the
           training data using a one-sided Fisher's exact test.

        Parameters
        ----------
        mtype1,mtype2 : MuTypes
            The mutation types to be compared.

        Returns
        -------
        pval : float
            The p-value given by the test.
        """
        samps1 = mtype1.get_samples(self.train_mut_)
        samps2 = mtype2.get_samples(self.train_mut_)
        if not samps1 or not samps2:
            raise HetmanDataError("Both sets must be non-empty!")
        all_samps = set(self.train_expr_.index)
        both_samps = samps1 & samps2
        _,pval = fisher_exact(
            [[len(all_samps - (samps1 | samps2)),
              len(samps1 - both_samps)],
             [len(samps2 - both_samps),
              len(both_samps)]],
            alternative='less')
        return pval

    def tune_clf(self,
                 clf, tune_indx=(50,51), mtype=None,
                 gene_list=None, exclude_samps=None, verbose=False):
        """Tunes a classifier using cross-validation within the training
           samples of this dataset.

        Parameters
        ----------
        clf : UniClassifier
            An instance of the classifier to test.

        tune_indx : list of ints, optional
            Which of the internal cross-validation samples to use for tuning
            the hyper-parameters of the given classifier.

        mtype : MuType, optional
            The mutation sub-type to test the classifier on.
            Default is to use all of the mutations available.

        verbose : boolean
            Whether or not the classifier should print information about the
            optimal hyper-parameters found during tuning.
        """
        if gene_list is None:
            gene_list = self.train_expr_.columns
        tune_samps = set(self.train_expr_.index)
        if exclude_samps is not None:
            tune_samps -= set(exclude_samps)
        tune_muts = self.train_mut_.status(tune_samps, mtype)
        tune_cvs = np.array([
            (x,y) for x,y in model_selection.StratifiedShuffleSplit(
                n_splits=max(tune_indx)+1, test_size=0.2,
                random_state=self.intern_cv_).split(
                    self.train_expr_.loc[tune_samps, gene_list], tune_muts)
            ])[tune_indx, :]

        return clf.tune(expr=self.train_expr_.loc[tune_samps, gene_list],
                        mut=tune_muts, cv_samples=tune_cvs, verbose=verbose)

    def score_clf(self,
                  clf, score_indx=tuple(range(16)), tune_indx=None, mtype=None,
                  gene_list=None, exclude_samps=None, final_fit=False,
                  verbose=False):
        """Test a classifier using tuning and cross-validation
           within the training samples of this dataset.

        Parameters
        ----------
        clf : UniClassifier
            An instance of the classifier to test.

        mtype : MuType, optional
            The mutation sub-type to test the classifier on.
            Default is to use all of the mutations available.

        test_indx : list of ints, optional
            Which of the internal cross-validation samples to use for testing
            classifier performance.

        tune_indx : list of ints, optional
            Which of the internal cross-validation samples to use for tuning
            the hyper-parameters of the given classifier.
            Default is to not do any tuning and thus use the default
            hyper-parameter settings.

        final_fit : boolean
            Whether or not to fit the given classifier to all of the training
            data after tuning and testing is complete. Useful if, for
            instance, we want to learn about the coefficients of this
            classifier when predicting the given set of mutations.

        verbose : boolean
            Whether or not the classifier should print information about the
            optimal hyper-parameters found during tuning.

        Returns
        -------
        P : float
            The 1st quartile of tuned classifier performance across the
            cross-validation samples. Used instead of the mean of performance
            to take into account performance variation for "hard" samples.

            Performance is measured using the area under the receiver operator
            curve metric.
        """
        if gene_list is None:
            gene_list = self.train_expr_.columns
        score_samps = set(self.train_expr_.index)
        if exclude_samps is not None:
            score_samps -= set(exclude_samps)
        if tune_indx is not None:
            clf = self.tune_clf(clf, tune_indx, mtype,
                                gene_list, exclude_samps, verbose)
            if verbose:
                print((clf.named_steps['fit']))
        score_muts = self.train_mut_.status(score_samps, mtype)
        score_cvs = np.array([
            (x,y) for x,y in model_selection.StratifiedShuffleSplit(
                n_splits=max(score_indx)+1, test_size=0.2,
                random_state=self.intern_cv_).split(
                    self.train_expr_.loc[score_samps, gene_list], score_muts)
            ])[score_indx, :]

        return np.percentile(model_selection.cross_val_score(
            estimator=clf,
            X=self.train_expr_.loc[score_samps, gene_list], y=score_muts,
            scoring=clf.score_auc, cv=score_cvs, n_jobs=16
            ), 25)

    def predict_clf(self,
                    clf, mtype=None, gene_list=None, exclude_samps=None,
                    pred_indx=tuple(range(16)), tune_indx=None,
                    final_fit=False, verbose=False):
        """Test a classifier using tuning and cross-validation
           within the training samples of this dataset.

        Parameters
        ----------
        clf : UniClassifier
            An instnce of the classifier to test.

        mtype : MuType, optional
            The mutation sub-type to test the classifier on.
            Default is to use all of the mutations available.

        test_indx : list of ints, optional
            Which of the internal cross-validation samples to use for testing
            classifier performance.

        tune_indx : list of ints, optional
            Which of the internal cross-validation samples to use for tuning
            the hyper-parameters of the given classifier.
            Default is to not do any tuning and thus use the default
            hyper-parameter settings.

        final_fit : boolean
            Whether or not to fit the given classifier to all of the training
            data after tuning and testing is complete. Useful if, for
            instance, we want to learn about the coefficients of this
            classifier when predicting the given set of mutations.

        verbose : boolean
            Whether or not the classifier should print information about the
            optimal hyper-parameters found during tuning.

        Returns
        -------
        P : float
            The 1st quartile of tuned classifier performance across the
            cross-validation samples. Used instead of the mean of performance
            to take into account performance variation for "hard" samples.

            Performance is measured using the area under the receiver operator
            curve metric.
        """
        if gene_list is None:
            gene_list = self.train_expr_.columns
        if exclude_samps is not None:
            ex_indx = np.array(range(self.train_expr_.shape[0]))
            ex_indx = set(ex_indx[self.train_expr_.index.isin(exclude_samps)])
        else:
            ex_indx = set()

        if tune_indx is not None:
            clf = self.tune_clf(clf, tune_indx, mtype,
                                gene_list, exclude_samps, verbose)
            if verbose:
                print(clf.named_steps['fit'])

        pred_muts = self.train_mut_.status(self.train_expr_.index, mtype)
        pred_scores = np.zeros((self.train_expr_.shape[0], 1))
        for pred_i in pred_indx:
            pred_seed = (self.intern_cv_ ** pred_i) % 4294967293
            pred_cvs = [
                (list(set(tr) - ex_indx), tst)
                for tr,tst in model_selection.StratifiedKFold(
                    n_splits=5, shuffle=True,
                    random_state=pred_seed).split(
                        self.train_expr_.loc[:, gene_list], pred_muts)
                ]
            pred_scores += model_selection.cross_val_predict(
                estimator=clf,
                X=self.train_expr_.loc[:, gene_list], y=pred_muts,
                method='prob_mut', cv=pred_cvs, n_jobs=16
                ) / len(pred_indx)

        pred_scores = pd.Series(pred_scores.tolist(), dtype=np.float)
        pred_scores.index = self.train_expr_.index
        return pred_scores

    def test_clf(self,
                 clf, mtype=None, tune_indx=None,
                 gene_list=None, verbose=False):
        """Test a classifier using by tuning within the training samples,
           training on all of them, and then testing on the testing samples.

        Parameters
        ----------
        classif : MutClassifier
            The classifier to test.

        mtype : MuType, optional
            The mutation sub-type to test the classifier on.
            Default is to use all of the mutations available.

        tune_indx : list of ints, optional
            Which of the internal cross-validation samples to use for tuning
            the hyper-parameters of the given classifier.

        Returns
        -------
        P : float
            Performance of the classifier on the testing samples as measured
            using the AUC ROC metric.
        """
        train_expr,train_mut,train_cv = self.training(mtype)
        test_expr,test_mut = self.testing(mtype)
        if tune_indx is not None:
            tune_cvs = [x for i,x in enumerate(train_cv)
                        if i in tune_indx]
            classif.tune(train_expr, train_mut, tune_cvs)
        clf.fit(train_expr, train_mut)
        return clf.score_auc(clf, test_expr, test_mut)



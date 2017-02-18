
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

from sklearn import model_selection
from scipy.stats import fisher_exact
from itertools import groupby
from functools import reduce
from mutation import MutLevel, MuTree


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
    log_add = np.min(np.min(expr[expr > 0])) * 0.5
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
    """Gets expression data as a matrix from a Firehose GDAC file."""
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


def get_graph(gene):
    """Downloads interaction data from Pathway Commons using their API.
    """
    pc = PC()
    id_url = "idmapping?id=" + gene
    gene_id = str(list(pc.http_get(id_url, frmt="json").values())[0])
    graph_url = "graph"
    params = {'source':gene_id,
              'kind':'neighborhood',
              'limit':1, 'format':'BINARY_SIF'}
    res = pc.http_get(graph_url, frmt=None, params=params)
    return res


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
                 mut_levels=('Gene', 'Form', 'Protein'),
                 cv_info={'Prop': 2.0/3, 'Seed':1}):
        self.cohort_ = cohort
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

    def training(self, mtype=None, gene_list=None):
        """Gets the expression data and the mutation status corresponding
           to a given mutation sub-type for the training samples in this
           dataset.

        Parameters
        ----------
        mtype : MuType, optional
            A mutation sub-type(s).
            The default is to use all available mutations.

        Returns
        -------
        expr : array-like, shape(n_tr_samples,n_features)
            The expression data for training samples.

        mut : list of bools, shape(n_tr_samples,)
            Mutation status for the training samples, True iff a sample has a
            mutation in the given set.

        cv : tuple
            A list of internal cross-validation splits to be used for
            classifier tuning, model selection, etc.
        """
        if gene_list is None:
            gene_list = self.train_expr_.columns
        mut_status = self.train_mut_.status(self.train_expr_.index, mtype)
        return (self.train_expr_.loc[:,gene_list],
                mut_status,
                [(x,y) for x,y
                 in model_selection.StratifiedShuffleSplit(
                     n_splits = 100, test_size = 0.2,
                     random_state=self.intern_cv_
                 ).split(self.train_expr_, mut_status)])

    def testing(self, mtype=None, gene_list=None):
        """Gets the expression data and the mutation status corresponding
           to a given mutation sub-type for the testing samples in this
           dataset.

        Parameters
        ----------
        mtype : MuType, optional
            A mutation sub-type(s).
            The default is to use all available mutations.

        Returns
        -------
        expr : array-like, shape(n_tst_samples,n_features)
            The expression data for testing samples.

        mut : list of bools, shape(n_tst_samples,)
            Mutation status for the testing samples, True iff a sample has a
            mutation in the given set.
        """
        if self.train_samps_ is None:
            raise HetmanError("No testing set defined!")
        if gene_list is None:
            gene_list = self.test_expr_.columns
        mut_status = self.test_mut_.status(self.test_expr_.index, mtype)
        return (self.test_expr_.loc[:,gene_list],
                mut_status)

    def test_classif_cv(self,
                        clx, mtype=None,
                        gene_list=None, exclude_samps=None,
                        test_indx=list(range(20)), tune_indx=None,
                        final_fit=False, verbose=False):
        """Test a classifier using tuning and cross-validation
           within the training samples of this dataset.

        Parameters
        ----------
        clx : UniClassifier
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
        train_expr,train_mut,train_cv = self.training(mtype, gene_list)
        if exclude_samps is not None:
            use_samps = [s for s,m in zip(train_expr.index,train_mut)
                         if s not in exclude_samps or m]
            use_indx = set([i for i,s in enumerate(train_expr.index)
                        if s in use_samps])
            train_cv = [(np.array(list(set(tr) & use_indx)),
                         np.array(list(set(tst) & use_indx)))
                         for tr,tst in train_cv]

        test_cvs = [x for i,x in enumerate(train_cv)
                    if i in test_indx]
        if tune_indx is not None:
            tune_cvs = [x for i,x in enumerate(train_cv)
                        if i in tune_indx]
            clx.tune(expr=train_expr, mut=train_mut,
                         cv_samples=tune_cvs, verbose=verbose)

        print((clx.named_steps['fit']))
        perf = np.percentile(model_selection.cross_val_score(
            estimator=clx, X=train_expr, y=train_mut,
            scoring=clx.score_auc, cv=test_cvs, n_jobs=16
            ), 25)
        if final_fit:
            if exclude_samps is not None:
                train_mut = [m for s,m in zip(train_expr.index,train_mut)
                             if s in use_samps]
                train_expr = train_expr.loc[use_samps,:]
            clx.fit(X=train_expr, y=train_mut)
        return perf

    def test_classif_full(self,
                          clx, mtype=None, tune_indx=list(range(5))):
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
        clx.fit(train_expr, train_mut)
        return clx.score_auc(clx, test_expr, test_mut)



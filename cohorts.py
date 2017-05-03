
"""
HetMan (Heterogeneity Manifold)
Classification of mutation sub-types using expression data.
This file contains classes that consolidate -omics datasets for use in
testing classifiers.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

from .data import *
from .mutation import MuTree

import numpy as np
import pandas as pd

from functools import reduce
from scipy.stats import fisher_exact
import random

from sklearn.base import is_classifier, clone
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils import indexable, check_random_state, safe_indexing
from sklearn.utils.validation import check_array, _num_samples
from sklearn.utils.fixes import bincount

from sklearn.model_selection import (
    StratifiedShuffleSplit, StratifiedKFold, cross_val_score)
from sklearn.model_selection._split import (
    _validate_shuffle_split, _approximate_mode)
from sklearn.model_selection._validation import _fit_and_predict
from sklearn.metrics import roc_auc_score


# .. helper functions ..
def _safe_load_path(genes):
    """Loads in pathway data from API or local source as available."""
    try:
        pathways = {gn:get_pc2_neighb(gn) for gn in genes}

    except HetManDataError:
        print("Defaulting to pathway data stored on file...")
        pathways = parse_sif(genes, get_sif_neighb(
            '/home/users/grzadkow/compbio/input-data/'
            'babur-mutex/data-tcga/Network.sif'))

    return pathways


class HetManCohortError(Exception):
    pass


class Cohort(object):
    """A class corresponding to -omics data for a single cohort.

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

    def _validate_dims(self,
                       mtype=None, include_samps=None, exclude_samps=None,
                       gene_list=None, use_test=False):
        if include_samps is not None and exclude_samps is not None:
            raise HetManMutError("Cannot specify samples to be included and"
                                  "samples to be excluded at the same time!")

        # get samples and genes from the specified cohort as specified
        if use_test:
            samps = self.test_samps_
            genes = set(self.test_expr_.columns)
        else:
            samps = self.train_samps_
            genes = set(self.train_expr_.columns)

        # remove samples and/or genes as necessary
        if include_samps is not None:
            samps &= set(include_samps)
        elif exclude_samps is not None:
            samps -= set(exclude_samps)
        if gene_list is not None:
            genes &= set(gene_list)

        # if a mutation type is specified include samples with that mutation
        if mtype is not None:
            samps |= mtype.get_samples(self.train_mut_)

        return samps, genes

    def __init__(self,
                 syn, cohort, mut_genes, mut_levels=('Gene', 'Form'),
                 load_path=True, cv_info=None):
        self.cohort_ = cohort
        if cv_info is None:
            cv_info = {'Prop': 2.0/3, 'Seed':1}
        self.intern_cv_ = cv_info['Seed'] ** 2
        self.mut_genes = mut_genes

        # loads gene expression and mutation data, as well as pathway
        # neighbourhood for mutated genes
        annot = get_annot()
        expr = get_expr_firehose(cohort)
        muts = get_mut_mc3(syn)
        cnvs = get_cnv_firehose(cohort)
        if load_path:
            self.path_ = _safe_load_path(mut_genes)

        # filters out genes that are not expressed in any samples, don't have
        # any variation across the samples, are not included in the
        # annotation data, or are not in the mutation datasets
        expr = expr.loc[:, expr.apply(
            lambda x: np.mean(x) > 0.005 and np.var(x) > 0.005)].dropna()
        annot = {g:a for g,a in annot.items()
                 if a['gene_name'] in expr.columns}
        annot_genes = [a['gene_name'] for g,a in annot.items()]
        expr = expr.loc[:, annot_genes]
        expr = expr.loc[:, ~expr.columns.duplicated()]
        muts = muts.loc[muts['Sample'].isin(expr.index), :]
        muts = muts.loc[muts['Sample'].isin(cnvs.index), :]

        # gets set of samples shared across expression and mutation datasets,
        # subsets these datasets to use only these samples
        self.samples = frozenset(
            set(muts['Sample']) & set(expr.index) & set(cnvs.index))
        expr = expr.loc[self.samples, :]
        cnvs = cnvs.loc[self.samples, mut_genes]
        muts = muts.loc[muts['Gene'].isin(mut_genes), :]

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
        self.annot = annot
        self.mut_annot = annot_data

        # gets subset of samples to use for training
        random.seed(a=cv_info['Seed'])
        self.cv_seed = random.getstate()
        if cv_info['Prop'] < 1.0 and cv_info['Prop'] > 0.0:
            self.train_samps_ = frozenset(
                random.sample(
                    population=self.samples,
                    k=int(round(len(self.samples) * cv_info['Prop'])))
                )
            self.test_samps_ = self.samples - self.train_samps_
            self.test_expr_ = log_norm_expr(expr.loc[self.test_samps_, :])
            self.test_mut_ = MuTree(
                muts=muts.loc[muts['Sample'].isin(self.test_samps_), :],
                levels=mut_levels)

        elif cv_info['Prop'] == 1:
            self.train_samps_ = self.samples
            self.test_samps_ = None

        else:
            raise ValueError("Improper cross-validation ratio that is"
                             "not > 0 and <= 1.0")

        # creates training and testing expression and mutation datasets
        self.train_expr_ = log_norm_expr(expr.loc[self.train_samps_, :])
        self.train_mut_ = MuTree(
            muts=muts.loc[muts['Sample'].isin(self.train_samps_), :],
            levels=mut_levels)

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
            raise HetManMutError("Both sets must be non-empty!")
        all_samps = set(self.train_expr_.index)
        both_samps = samps1 & samps2
        _,pval = fisher_exact(
            [[len(all_samps - (samps1 | samps2)),
              len(samps1 - both_samps)],
             [len(samps2 - both_samps),
              len(both_samps)]],
            alternative='less')

        return pval

    def fit_clf(self, clf, mtype=None, gene_list=None, exclude_samps=None):
        """Fits a classifier."""
        samps, genes = self._validate_dims(exclude_samps=exclude_samps,
                                           gene_list=gene_list)
        fit_muts = self.train_mut_.mut_vec(clf, samps, mtype)
        return clf.fit(X=self.train_expr_.loc[samps, genes], y=fit_muts,
                       feat__mut_genes=list(
                           reduce(lambda x,y: x|y, mtype.child.keys())),
                       feat__path_obj=self.path_)

    def predict_clf(self,
                    clf, use_test=False, gene_list=None, exclude_samps=None):
        """Predicts mutation status using a classifier."""
        samps, genes = self._validate_dims(exclude_samps=exclude_samps,
                                           gene_list=gene_list,
                                           use_test=use_test)
        if use_test:
            muts = clf.predict_mut(expr=self.test_expr_.loc[samps, genes])
        else:
            muts = clf.predict_mut(expr=self.train_expr_.loc[samps, genes])

        return muts

    def tune_clf(self,
                 clf, mtype=None, tune_splits=2, test_count=16,
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

        # gets samples, genes, and mutation vector to use for tuning
        samps, genes = self._validate_dims(exclude_samps=exclude_samps,
                                           gene_list=gene_list)
        tune_muts = self.train_mut_.mut_vec(clf, samps, mtype)

        # get internal cross-validation splits in the training set and use
        # them to tune the classifier
        tune_cvs = MutShuffleSplit(
            n_splits=tune_splits, test_size=0.2,
            random_state=(self.intern_cv_ ** 2) % 42949672)
        return clf.tune(expr=self.train_expr_.loc[samps, genes],
                        mut=tune_muts, path_obj=self.path_,
                        mut_genes=list(
                            reduce(lambda x,y: x|y, mtype.child.keys())),
                        test_count=test_count, cv_samples=tune_cvs,
                        verbose=verbose)

    def score_clf(self,
                  clf, mtype=None, score_splits=16, gene_list=None,
                  exclude_samps=None, verbose=False):
        """Test a classifier using tuning and cross-validation
           within the training samples of this dataset.

        Parameters
        ----------
        clf : UniClassifier
            An instance of the classifier to test.

        mtype : MuType, optional
            The mutation sub-type to test the classifier on.
            Default is to use all of the mutations available.

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
        samps, genes = self._validate_dims(mtype=mtype,
                                           exclude_samps=exclude_samps,
                                           gene_list=gene_list)
        score_muts = self.train_mut_.mut_vec(clf, samps, mtype)
        score_cvs = MutShuffleSplit(
            n_splits=score_splits, test_size=0.2,
            random_state=self.intern_cv_)

        cv_score = np.percentile(cross_val_score(
            estimator=clf, X=self.train_expr_.loc[samps, genes], y=score_muts,
            fit_params={'feat__mut_genes': list(
                reduce(lambda x,y: x|y, mtype.child.keys())),
                        'feat__path_obj': self.path_},
            scoring=clf.score_mut, cv=score_cvs, n_jobs=-1
            ), 25)

        return cv_score

    def infer_clf(self,
                  clf, mtype, infer_splits=16, gene_list=None,
                  exclude_samps=None, verbose=False):
        samps, genes = self._validate_dims(gene_list=gene_list)
        infer_muts = self.train_mut_.mut_vec(clf, samps, mtype)

        infer_scores = cross_val_predict_mut(
            estimator=clf, X=self.train_expr_.loc[:, genes], y=infer_muts,
            exclude_samps=exclude_samps, cv_fold=4, cv_count=infer_splits,
            fit_params={'feat__mut_genes': list(
                reduce(lambda x,y: x|y, mtype.child.keys())),
                        'feat__path_obj': self.path_},
            random_state=int(self.intern_cv_ ** 1.5) % 42949672, n_jobs=-1
            )

        return infer_scores

    def eval_clf(self, clf, mtype=None, gene_list=None, exclude_samps=None):
        """Evaluate the performance of a classifier."""
        samps, genes = self._validate_dims(exclude_samps=exclude_samps,
                                           gene_list=gene_list, use_test=True)
        eval_muts = self.test_mut_.mut_vec(clf, samps, mtype)
        return clf.score_mut(clf,
                             self.test_expr_.loc[samps, genes], eval_muts)


class MultiCohort(object):
    """A class for storing -omics data from multiple cohorts.

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

    def _validate_dims(self, 
                       include_samps=None, exclude_samps=None,
                       gene_list=None):
        if include_samps is not None and exclude_samps is not None:
            raise HetManMutError("Cannot specify samples to be included and"
                                  "samples to be excluded at the same time!")
        elif include_samps is not None:
            samps = [coh.train_samps_ & set(inc_samps)
                     for inc_samps, (_,coh) in zip(include_samps, self)]
        elif exclude_samps is not None:
            samps = [coh.train_samps_ - set(exc_samps)
                     for exc_samps, (_,coh) in zip(exclude_samps, self)]
        else:
            samps = [coh.train_samps_ for _,coh in self]

        genes = [set(coh.train_expr_.columns) for _,coh in self]
        if gene_list is not None:
            genes = [gns & set(gene_list) for gns in genes]

        return samps, genes

    def __init__(self,
                 syn, cohorts, mut_genes, mut_levels=('Gene', 'Form'),
                 load_path=True, cv_info=None):
        if cv_info is None:
            cv_info = {'Prop': 2.0/3, 'Seed':1}
        self.intern_cv_ = cv_info['Seed'] ** 2
        if load_path:
            self.path_ = _safe_load_path(mut_genes)
        self.cohorts_ = dict(
            (cohort, Cohort(syn, cohort, mut_genes, mut_levels,
                            cv_info=cv_info, load_path=False))
            for cohort in cohorts)

    def __iter__(self):
        """Allows for iteration over the component cohorts."""
        return iter(self.cohorts_.items())

    def __getitem__(self, key):
        """Gets a particular cohort."""
        return self.cohorts_[key]

    def tune_multiclf(self,
                      multiclf, tune_splits=2, mtype=None,
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
        samps, genes = self._validate_dims(gene_list=gene_list,
                                           exclude_samps=exclude_samps)
        tune_muts = [coh.train_mut_.status(smps, mtype)
                     for (_,coh), smps in zip(self, samps)]
        tune_cvs = MutShuffleSplit(
            n_splits=tune_splits, test_size=0.2,
            random_state=(self.intern_cv_ ** 2) % 42949672)

        return multiclf.tune(
            expr_list=[coh.train_expr_.loc[smps, gns]
                       for (_,coh), smps, gns in zip(self, samps, genes)],
            mut_list=tune_muts, cv_samples=tune_cvs, path_obj=self.path_,
            mut_genes=list(reduce(lambda x,y: x|y, mtype.child.keys())),
            verbose=verbose
            )

    def score_multiclf(self,
                       multiclf, score_splits=8, tune_splits=None,
                       mtype=None, gene_list=None, exclude_samps=None,
                       final_fit=False, verbose=False):
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
        samps, genes = self._validate_dims(gene_list=gene_list,
                                           exclude_samps=exclude_samps)
        if tune_splits is not None and tune_splits > 0:
            multiclf = self.tune_multiclf(multiclf, tune_splits, mtype,
                                          gene_list, exclude_samps, verbose)
            if verbose:
                print("Classifier has been tuned to:\n"
                      + multiclf.named_steps['fit'])

        score_muts = [coh.train_mut_.status(smps, mtype)
                      for (_,coh), smps in zip(self, samps)]
        score_cvs = MutShuffleSplit(
            n_splits=score_splits, test_size=0.2,
            random_state=self.intern_cv_ % 42949672)

        return np.percentile(cross_val_score(
            estimator=multiclf,
            X=[coh.train_expr_.loc[smps, gns]
               for (_,coh), smps, gns in zip(self, samps, genes)],
            y=score_muts, fit_params={
                'feat__mut_genes': list(reduce(lambda x,y: x|y,
                                               mtype.child.keys())),
                'feat__path_obj': self.path_},
            scoring=multiclf.score_auc, cv=score_cvs, n_jobs=-1
            ), 25)

    def predict_multiclf(self,
                         multiclf, mtype=None, gene_list=None,
                         exclude_samps=None, pred_indx=tuple(range(16)),
                         tune_indx=None, final_fit=False, verbose=False):
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


class MutShuffleSplit(StratifiedShuffleSplit):
    """Generates splits of single or multiple cohorts into training and
       testing sets that are stratified according to the mutation vectors.
    """

    def __init__(self,
                 n_splits=10, test_size=0.1, train_size=None,
                 random_state=None):
        super(MutShuffleSplit, self).__init__(
            n_splits, test_size, train_size, random_state)

    def _iter_indices(self, expr, mut=None, groups=None):
        """Generates indices of training/testing splits for use in
           stratified shuffle splitting of cohort data.
        """

        # with one cohort, proceed with stratified sampling, binning mutation
        # values if they are continuous
        if hasattr(expr, 'shape'):
            if len(np.unique(mut)) > 2:
                mut = mut > np.percentile(mut, 50)
            for train, test in super(MutShuffleSplit, self)._iter_indices(
                X=expr, y=mut, groups=groups):
                yield train, test

        # otherwise, perform stratified sampling on each cohort separately
        else:

            # gets info about input
            n_samples = [_num_samples(X) for X in expr]
            mut = [check_array(y, ensure_2d=False, dtype=None)
                      for y in mut]
            n_train_test = [
                _validate_shuffle_split(n_samps,
                                        self.test_size, self.train_size)
                for n_samps in n_samples]
            class_info = [np.unique(y, return_inverse=True) for y in mut]
            n_classes = [classes.shape[0] for classes,_ in class_info]
            classes_counts = [bincount(y_indices)
                              for _,y_indices in class_info]

            # makes sure we have enough samples in each class for stratification
            for i, (n_train, n_test) in enumerate(n_train_test):
                if np.min(classes_counts[i]) < 2:
                    raise ValueError("The least populated class in y has only 1 "
                                     "member, which is too few. The minimum "
                                     "number of groups for any class cannot "
                                     "be less than 2.")

                if n_train < n_classes[i]:
                    raise ValueError('The train_size = %d should be greater or '
                                     'equal to the number of classes = %d'
                                     % (n_train, n_classes[i]))
                
                if n_test < n_classes[i]:
                    raise ValueError('The test_size = %d should be greater or '
                                     'equal to the number of classes = %d'
                                     % (n_test, n_classes[i]))

            # generates random training and testing cohorts
            rng = check_random_state(self.random_state)
            for _ in range(self.n_splits):
                n_is = [_approximate_mode(class_counts, n_train, rng)
                        for class_counts, (n_train, _)
                        in zip(classes_counts, n_train_test)]
                classes_counts_remaining = [class_counts - n_i
                                           for class_counts, n_i
                                            in zip(classes_counts, n_is)]
                t_is = [_approximate_mode(class_counts_remaining, n_test, rng)
                        for class_counts_remaining, (_, n_test)
                        in zip(classes_counts_remaining, n_train_test)]

                train = [[] for _ in expr]
                test = [[] for _ in expr]

                for i, (classes, _) in enumerate(class_info):
                    for j, class_j in enumerate(classes):
                        permutation = rng.permutation(classes_counts[i][j])
                        perm_indices_class_j = np.where(
                            (mut[i] == class_j))[0][permutation]
                        train[i].extend(perm_indices_class_j[:n_is[i][j]])
                        test[i].extend(
                            perm_indices_class_j[n_is[i][j]:n_is[i][j]
                                                 + t_is[i][j]])
                    train[i] = rng.permutation(train[i])
                    test[i] = rng.permutation(test[i])

                yield train, test

    def split(self, expr, mut=None, groups=None):
        if not hasattr(expr, 'shape'):
            mut = [check_array(y, ensure_2d=False, dtype=None)
                      for y in mut]
        else:
            mut = check_array(mut, ensure_2d=False, dtype=None)

        expr, mut, groups = indexable(expr, mut, groups)
        return self._iter_indices(expr, mut, groups)


def cross_val_predict_mut(estimator, X, y=None, groups=None,
                          exclude_samps=None, cv_fold=4, cv_count=16,
                          n_jobs=1, verbose=0, fit_params=None,
                          pre_dispatch='2*n_jobs', random_state=None):
    """Generates predicted mutation states for samples using internal
       cross-validation via repeated stratified K-fold sampling.
    """

    # gets the number of K-fold repeats
    if (cv_count % cv_fold) != 0:
        raise ValueError("The number of folds should evenly divide the total"
                         "number of cross-validation splits.")
    cv_rep = int(cv_count / cv_fold)

    # checks that the given estimator can predict continuous mutation states
    if not callable(getattr(estimator, 'predict_proba')):
        raise AttributeError('predict_proba not implemented in estimator')

    # gets absolute indices for samples to train and test over
    X, y, groups = indexable(X, y, groups)
    if exclude_samps is None:
        exclude_samps = []
    else:
        exclude_samps = list(set(exclude_samps) - set(X.index[y]))
    use_samps = list(set(X.index) - set(exclude_samps))
    use_samps_indx = X.index.get_indexer_for(use_samps)
    ex_samps_indx = X.index.get_indexer_for(exclude_samps)

    # generates the training/prediction splits
    cv_iter = []
    for i in range(cv_rep):
        cv = StratifiedKFold(n_splits=cv_fold, shuffle=True,
                             random_state=(random_state * i) % 12949671)
        cv_iter += [
            (use_samps_indx[train],
             np.append(use_samps_indx[test], ex_samps_indx))
            for train, test in cv.split(X.ix[use_samps_indx, :],
                                        np.array(y)[use_samps_indx],
                                        groups)
            ]

    # for each split, fit on the training set and get predictions for
    # remaining cohort
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    prediction_blocks = parallel(delayed(_fit_and_predict)(
        clone(estimator), X, y,
        train, test, verbose, fit_params, 'predict_proba')
        for train, test in cv_iter)

    # consolidates the predictions into an array
    pred_mat = [[] for _ in range(X.shape[0])]
    for i in range(cv_rep):
        predictions = np.concatenate(
            [pred_block_i for pred_block_i, _
             in prediction_blocks[(i*cv_fold):((i+1)*cv_fold)]])
        test_indices = np.concatenate(
            [indices_i for _, indices_i
            in prediction_blocks[(i*cv_fold):((i+1)*cv_fold)]]
            )

        for j in range(X.shape[0]):
            pred_mat[j] += list(predictions[test_indices == j, 1])
        
    return pred_mat



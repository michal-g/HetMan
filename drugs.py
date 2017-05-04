
"""
Finds differential drug profiles for novel samples.
"""

import sys
sys.path += ['/home/users/grzadkow/compbio/software/ophion/client/python/']

import ophion
oph = ophion.Ophion("http://bmeg.compbio.ohsu.edu")
import synapseclient
import json

from HetMan.data import get_expr_firehose
from HetMan.cohorts import Cohort
from HetMan.mutation import MuType

import numpy as np
import pandas as pd

from scipy import stats
from math import exp, log10
from functools import reduce
from operator import mul
import random

from sklearn.base import is_classifier, clone
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils import indexable, check_random_state, safe_indexing
from sklearn.utils.validation import check_array, _num_samples
from sklearn.utils.fixes import bincount

from sklearn.model_selection import (
    StratifiedShuffleSplit, StratifiedKFold, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection._split import (
    _validate_shuffle_split, _approximate_mode)
from sklearn.model_selection._validation import _fit_and_predict
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import ElasticNet as ENet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import pickle
from fuzzywuzzy import process


def exp_norm(expr):
    out_expr = expr.apply(
        lambda x: stats.expon.ppf((x.rank()-1) / len(x)), 1)
    return out_expr.fillna(0.0)


class DrugCohort(object):

    def __init__(self, drug, source='CCLE', random_state=None):
        exp_data = {}
        drug_data = {}

        if source == 'CCLE':
            for i in oph.query().has(
                "gid", "cohort:CCLE").outgoing(
                    "hasSample").incoming("expressionForSample").execute():

                if ('properties' in i
                    and 'serializedExpressions' in i['properties']):
                    drug_querr = oph.query().has("gid", i['gid']).outgoing(
                        "expressionForSample").outEdge(
                            "responseToCompound").values(
                                ['gid', 'responseSummary']).execute()
                    drug_data[i['gid']] = {}

                    for dg in drug_querr:
                        if dg:
                            dg_parse = dg.split(':')
                            if dg_parse[0] == 'responseCurve':
                                cur_drug = dg_parse[-1]

                            elif cur_drug == drug:
                                drug_data[i['gid']] = [
                                    x['value'] for x in json.loads(dg)
                                    if x['type'] == 'AUC'][0]

                s = json.loads(i['properties']['serializedExpressions'])
                exp_data[i['gid']] = s

            drug_resp = pd.Series({k:v for k,v in drug_data.items() if v})
            drug_expr = exp_norm(pd.DataFrame(exp_data).transpose())
            drug_expr = drug_expr.loc[drug_resp.index, :]

        elif source == 'ioria':
            cell_expr = pd.read_csv(
                '/home/users/grzadkow/compbio/input-data/ioria-landscape/'
                'cell-line/Cell_line_RMA_proc_basalExp.txt',
                sep='\t', comment='#')
            cell_expr = cell_expr.ix[~pd.isnull(cell_expr['GENE_SYMBOLS']), :]
            drug_annot = pd.read_csv('/home/users/grzadkow/compbio/'
                                     'scripts/HetMan/experiments/'
                                     'drug_predictions/input/drug_annot.txt',
                                     sep='\t', comment='#')

            cell_expr.index = cell_expr['GENE_SYMBOLS']
            cell_expr = cell_expr.ix[:, 2:]

            drug_resp = pd.read_csv('/home/users/grzadkow/compbio/input-data'
                                    '/ioria-landscape/cell-line/drug-auc.txt',
                                    sep='\t', comment='#')
            drug_match = process.extractOne(drug, drug_annot['Name'])
            drug_lbl = 'X' + str(drug_annot['Identifier'][drug_match[-1]])
            drug_resp = drug_resp.loc[:, drug_lbl]
            drug_resp = drug_resp[~pd.isnull(drug_resp)]
            drug_expr = exp_norm(
                cell_expr.loc[:, drug_resp.index].transpose().dropna(
                    axis=0, how='all').dropna(axis=1, how='any'))

        random.seed(a=random_state)
        cv_seed = random.getstate()
        self.train_samps_ = frozenset(
            random.sample(population=list(drug_expr.index),
                          k=int(round(drug_expr.shape[0] * 0.8)))
            )
        self.test_samps_ = frozenset(
            set(drug_expr.index) - self.train_samps_)

        self.random_state = random_state
        self.drug_resp = drug_resp
        self.drug_expr = drug_expr

    def fit_clf(self, clf):
        clf.fit(X=self.drug_expr.loc[list(self.train_samps_), :],
                y=self.drug_resp[list(self.train_samps_)])
        return clf

    def tune_clf(self, clf, tune_splits=4, test_count=16):
        tune_cvs = DrugShuffleSplit(
            n_splits=tune_splits, test_size=0.2,
            random_state=(self.random_state ** 2) % 42949672)
        return clf.tune(expr=self.drug_expr.loc[list(self.train_samps_), :],
                        drug=self.drug_resp[list(self.train_samps_)],
                        cv_samples=tune_cvs, test_count=test_count)

    def eval_clf(self, clf):
        return clf.score(X=self.drug_expr.loc[list(self.test_samps_), :],
                         y=self.drug_resp[list(self.test_samps_)])


class DrugPipe(Pipeline):
    """A class corresponding to pipelines for predicting gene drugations
       using expression data.
    """

    # the parameters that are to be tuned, with either statistical
    # distributions or iterables to be sampled from as values
    tune_priors = ()

    def __init__(self, steps):
        super(DrugPipe, self).__init__(steps)
        self.cur_tuning = dict(self.tune_priors)

    def tune(self,
             expr, drug, cv_samples, test_count=16):
        """Tunes the pipeline by sampling over the tuning parameters."""

        # checks if the classifier has parameters to be tuned
        if self.tune_priors:
            prior_counts = [len(x) if hasattr(x, '__len__') else float('Inf')
                            for x in self.cur_tuning.values()]
            max_tests = reduce(mul, prior_counts, 1)
            test_count = min(test_count, max_tests)

            # samples parameter combinations and tests each one
            grid_test = RandomizedSearchCV(
                estimator=self, param_distributions=self.cur_tuning,
                n_iter=test_count, cv=cv_samples, n_jobs=-1, refit=False
                )
            grid_test.fit(expr, drug)

            # finds the best parameter combination and updates the classifier
            tune_scores = (grid_test.cv_results_['mean_test_score']
                           - grid_test.cv_results_['std_test_score'])
            self.set_params(
                **grid_test.cv_results_['params'][tune_scores.argmax()])

        return self


class DrugShuffleSplit(StratifiedShuffleSplit):
    """Generates splits of single or multiple cohorts into training and
       testing sets that are stratified according to the mutation vectors.
    """

    def __init__(self,
                 n_splits=10, test_size=0.1, train_size=None,
                 random_state=None):
        super(DrugShuffleSplit, self).__init__(
            n_splits, test_size, train_size, random_state)

    def _iter_indices(self, expr, drug=None, groups=None):
        """Generates indices of training/testing splits for use in
           stratified shuffle splitting of drug cohort data.
        """

        # with one cohort, proceed with stratified sampling, binning mutation
        # values if they are continuous
        if hasattr(expr, 'shape'):
            if len(np.unique(drug)) > 2:
                drug = drug > np.percentile(drug, 50)
            for train, test in super(DrugShuffleSplit, self)._iter_indices(
                X=expr, y=drug, groups=groups):
                yield train, test

        # otherwise, perform stratified sampling on each cohort separately
        else:

            # gets info about input
            n_samples = [_num_samples(X) for X in expr]
            drug = [check_array(y, ensure_2d=False, dtype=None)
                      for y in drug]
            n_train_test = [
                _validate_shuffle_split(n_samps,
                                        self.test_size, self.train_size)
                for n_samps in n_samples]
            class_info = [np.unique(y, return_inverse=True) for y in drug]
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
                            (drug[i] == class_j))[0][permutation]
                        train[i].extend(perm_indices_class_j[:n_is[i][j]])
                        test[i].extend(
                            perm_indices_class_j[n_is[i][j]:n_is[i][j]
                                                 + t_is[i][j]])
                    train[i] = rng.permutation(train[i])
                    test[i] = rng.permutation(test[i])

                yield train, test

    def split(self, expr, drug=None, groups=None):
        if not hasattr(expr, 'shape'):
            drug = [check_array(y, ensure_2d=False, dtype=None)
                      for y in drug]
        else:
            drug = check_array(drug, ensure_2d=False, dtype=None)

        expr, drug, groups = indexable(expr, drug, groups)
        return self._iter_indices(expr, drug, groups)


def cross_val_predict_drug(estimator, X, y=None, groups=None,
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


class ElasticNet(DrugPipe):
    """A class corresponding to elastic net regression
       of gene gain/loss status.
    """

    tune_priors = (
        ('fit__alpha', stats.lognorm(scale=exp(1), s=exp(1))),
        ('fit__l1_ratio', (0.05,0.25,0.5,0.75,0.95))
        )

    def __init__(self):
        norm_step = StandardScaler()
        fit_step = ENet(normalize=False, max_iter=5000)
        DrugPipe.__init__(self, [('norm', norm_step), ('fit', fit_step)])


class SVRrbf(DrugPipe):
    """A class corresponding to Support Vector Machine regression
       of gene gain/loss status using a radial basis kernel.
    """

    tune_priors = (
            ('fit__C', stats.lognorm(scale=exp(-1), s=exp(1))),
            ('fit__gamma', stats.lognorm(scale=1e-5, s=exp(2)))
        )

    def __init__(self):
        norm_step = StandardScaler()
        fit_step = SVR(kernel='rbf', cache_size=500)
        DrugPipe.__init__(self, [('norm', norm_step), ('fit', fit_step)])


class rForest(DrugPipe):
    """A class corresponding to Random Forest regression
       of gene gain/loss status.
    """

    tune_priors = (
        ('fit__min_samples_leaf', (0.001,0.005,0.01,0.05)),
        ('fit__max_features', (5, 'sqrt', 'log2', 0.02))
        )

    def __init__(self):
        norm_step = StandardScaler()
        fit_step = RandomForestRegressor(n_estimators=1000)
        DrugPipe.__init__(self, [('norm', norm_step), ('fit', fit_step)])



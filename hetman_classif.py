
"""
Hetman (Heterogeneity Manifold)
Classification of mutation sub-types using expression data.
This file contains the algorithms used as building blocks for the
classification ensembles of mutation sub-types.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

from sklearn import (neighbors, linear_model, feature_selection, svm,
                     gaussian_process, pipeline, model_selection,
                     preprocessing, ensemble, neural_network, naive_bayes,
                     decomposition, mixture)
from sklearn.gaussian_process.kernels import *
from sklearn.metrics import roc_auc_score
import numpy as np
import re
from itertools import groupby
from math import log,exp
from scipy import stats
import dill as pickle

_base_dir = '/home/users/grzadkow/compbio/'


# .. helper functions for classification and feature selection classes ..
def _score_auc(estimator, expr, mut):
    """Computes the AUC score for a mutation classifier on a given
       expression matrix and a mutation state vector.

    Parameters
    ----------
    estimator : MutClassifier
        A mutation classification algorithm as defined below.

    expr : array-like, shape (n_samples,n_features)
        An expression dataset.

    mut : array-like, shape (n_samples,)
        A boolean vector corresponding to the presence of a particular type of
        mutation in the same set of samples as the given expression dataset.

    Returns
    -------
    S : float
        The AUC score corresponding to mutation classification accuracy.
    """
    mut_scores = estimator.prob_mut(expr)
    return roc_auc_score(mut, mut_scores)


def _read_sif(mut_genes, sif_file='input-data/babur-mutex/data-tcga/Network.sif'):
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


def _mut_ttest(expr_vec, mut):
    """Performs the Student's t-test on the hypothesis that the given
       expression values differ according to the given mutation status.

    Parameters
    ----------
    expr_vec : array-like, shape (n_samples,)
        Expression values for a set of samples.

    mut : array-like, shape (n_samples,)
        Boolean mutation status for the same set of samples.

    Returns
    -------
    score : float
        A negative log-transform of the t-test p-value, so that higher
        values -> higher significance.
    """
    score = log(stats.ttest_ind(
        expr_vec[mut],
        expr_vec[np.invert(mut)])[1] + 10 ** -323,
        10) * -1
    if np.isnan(score):
        score = 0
    return score


def _update_params(param_list):
    """Returns an updated list of hyper-parameters for the log-normal
       distribution based on the given list of parameter,performance pairs.

    Parameters
    ----------
    param_list : tuple

    Returns
    -------
    new_mean : float

    new_sd : float
    """
    perf_list = [perf for param,perf in param_list]
    perf_list = (perf_list - np.mean(perf_list)) / (np.var(perf_list) ** 0.5)
    new_perf = [param * (exp(perf))
                for param,perf in
                zip([param for param,perf in param_list],
                    perf_list)]
    new_mean = reduce(lambda x,y: x*y, new_perf) ** (1.0/len(param_list))
    new_sd = np.mean([(log(x) - log(new_mean)) ** 2 for x in new_perf]) ** 0.5
    return new_mean,new_sd


class GeneSelect(feature_selection.GenericUnivariateSelect):
    """A class for pathway Commons-based feature selection for use in
       in classification of mutation sub-types.
    """

    def __init__(self, path_prior, path_dir,
                 mut_genes, expr_genes, link_data):
        self.mut_genes = mut_genes
        self.expr_genes = expr_genes
        self.link_data = link_data
        link_genes = set()
        for gene,directs in self.link_data.items():
            for direction,int_types in directs.items():
                for int_type,genes in int_types.items():
                    link_genes |= set(genes)
        self.link_genes = tuple(link_genes & set(self.expr_genes))
        self.link_indx = {g:self.expr_genes.index(g)
                          for g in self.link_genes}
        self.params = {'path_prior':path_prior, 'path_dir':path_dir}
        feature_selection.GenericUnivariateSelect.__init__(self,
            score_func=self._score_genes,
            mode='k_best',
            param=12
            )

    def _score_genes(self, X, y):
        """Scores genes according to a univariate test for association with
        mutation status in conjunction with their interactions with the
        mutated genes.
        """
        gene_scores = np.apply_along_axis(
            func1d=lambda x: _mut_ttest(x, y),
            axis=0, arr=X[:,self.link_indx.values()]
            )
        for gene,directs in self.link_data.items():
            for direction,int_types in directs.items():
                for int_type,genes in int_types.items():
                    for g in list(set(genes) & set(self.link_genes)):
                        g_indx = self.link_genes.index(g)
                        gene_scores[g_indx] *= self.params['path_prior']
                        if direction == 'in':
                            gene_scores[g_indx] *= self.params['path_dir']
                        elif direction == 'out':
                            gene_scores[g_indx] *= (self.params['path_dir'] **
                                                    -1)

        return [gene_scores[self.link_genes.index(g)]
                if g in self.link_genes
                else 0
                for g in self.expr_genes], [0 for g in self.expr_genes]


class MutClassifier(pipeline.Pipeline):
    """A class corresponding to expression-based"""
    """classifiers of mutation status."""

    def __init__(self, steps):
        pipeline.Pipeline.__init__(self, steps)

    def __str__(self):
        param_list = self.get_params()
        return reduce(
            lambda x,y: x + ', ' + y,
            [k + ': ' + str(param_list[k])
             for k in self._tune_params.keys()]
            )

    def prob_mut(self, expr):
        mut_scores = self.predict_proba(expr)
        if hasattr(self, 'classes_'):
            true_indx = [i for i,x in enumerate(self.classes_) if x]
        else:
            wghts = tuple(self.named_steps['fit'].weights_)
            true_indx = wghts.index(min(wghts))
        return [m[true_indx] for m in mut_scores]

    def tune(self, expr, mut, cv_samples, test_count=8, verbose=False):
        if test_count == 'auto':
            test_count = int(16 ** (len(self._tune_params) ** -1.0))
        if self._tune_params:
            new_grid = {param:distr.rvs(test_count)
                        for param,distr in self._param_priors.items()}
            grid_test = model_selection.GridSearchCV(
                estimator=self, param_grid=new_grid,
                scoring=_score_auc, cv=cv_samples,
                n_jobs=-1
                )
            grid_test.fit(expr, mut)
            self.set_params(**grid_test.best_params_)
            for param in self._tune_params.keys():
                new_mean,new_sd = _update_params(
                    [(x[param],y)
                     for x,y in zip(grid_test.cv_results_['params'],
                                    grid_test.cv_results_['mean_test_score'])
                    ])
                self._param_priors[param] = stats.lognorm(
                    scale=new_mean, s=new_sd)
            if verbose:
                print self


class naiveBayes(MutClassifier):
    """A class corresponding to gaussian kernal support vector"""
    """classification of mutation status."""

    _tune_params = {}

    def __init__(self):
        norm_step = preprocessing.StandardScaler()
        fit_step = naive_bayes.GaussianNB()
        MutClassifier.__init__(self,
            [('norm', norm_step), ('fit', fit_step)])


class Lasso(MutClassifier):
    """A class corresponding to logistic regression classification"""
    """of mutation status with the lasso regularization penalty."""

    def __init__(self, mut_genes=None, expr_genes=None):
        self._tune_params = {'fit__C':1.0}
        self._param_priors = {'fit__C':stats.lognorm(scale=exp(0), s=2)}
        norm_step = preprocessing.StandardScaler()
        fit_step = linear_model.LogisticRegression(
                    penalty='l1', C=self._tune_params['fit__C'],
                    class_weight={False:1, True:1}
                    )
        MutClassifier.__init__(self,
            [('norm', norm_step), ('fit', fit_step)])
        self.set_params(**self._tune_params)


class LassoPCA(MutClassifier):
    """A class corresponding to logistic regression classification"""
    """of mutation status with the lasso regularization penalty."""

    _tune_params = {'fit__C':[2 ** x for x in range(-12,4)]}

    def __init__(self, C=1.0):
        self._tune_params = {'fit__C':1.0}
        self._param_priors = {'fit__C':stats.lognorm(scale=exp(0), s=2)}
        norm_step = preprocessing.StandardScaler()
        pca_step = decomposition.PCA()
        fit_step = linear_model.LogisticRegression(
                    penalty='l1', C=self._tune_params['fit__C'],
                    class_weight={False:1, True:1}
                    )
        MutClassifier.__init__(self,
            [('norm', norm_step), ('pca', pca_step), ('fit', fit_step)])
        self.set_params(**self._tune_params)


class enSGD(MutClassifier):
    """A class corresponding to elastic net logreg"""
    """classification of mutation status."""

    _tune_params = {'fit__l1_ratio':(0.1,0.2,0.4,0.8),
                    'fit__alpha':[10 ** x for x in range(-5,-2)]}

    def __init__(self, l1_ratio=0.2, alpha=0.001):
        self.params = {'l1_ratio':l1_ratio, 'alpha':alpha}
        norm_step = preprocessing.StandardScaler()
        fit_step = linear_model.SGDClassifier(
                    loss='log', penalty='elasticnet',
                    class_weight={False:1, True:1},
                    l1_ratio=l1_ratio, alpha=alpha
                    )
        MutClassifier.__init__(self,
            [('norm', norm_step), ('fit', fit_step)])
        self.set_params(**self.params)


class Mixture(MutClassifier):
    """A class corresponding to gaussian mixture model"""
    """classification of mutation status."""

    def __init__(self, mut_genes, expr_genes):
        self.mut_genes = mut_genes
        self.link_data = _read_sif(mut_genes)
        self.expr_genes = expr_genes
        self._tune_params = {'feat__path_prior':10**6, 'feat__path_dir':1}
        self._param_priors = {
            'feat__path_prior':stats.lognorm(scale=exp(6), s=3),
            'feat__path_dir':stats.lognorm(scale=exp(0), s=1)
            }
        norm_step = preprocessing.StandardScaler()
        feat_step = GeneSelect(
            self._tune_params['feat__path_prior'],
            self._tune_params['feat__path_dir'],
            self.mut_genes, self.expr_genes, self.link_data
            )
        fit_step = mixture.GaussianMixture(
            n_components=2, covariance_type='spherical',
            tol=5e-3, max_iter=50, 
            n_init=3, init_params='kmeans', weights_init=[0.8,0.2]
            )
        MutClassifier.__init__(self,
            [('norm', norm_step),
             ('feat', feat_step),
             ('fit', fit_step)])
        self.set_params(**self._tune_params)


class PCfeatures(feature_selection.GenericUnivariateSelect):
    """A class corresponding to Pathway Commons based feature
       selection for use in classification of mutation status.
    """

    def __init__(self,
                 mut_genes, expr_genes, link_data,
                 path_prior=100, path_dir=1):
        self.link_genes = list(set(
            [item for sublist
             in [item for sublist
                 in [val.values() for sublist
                     in link_data.values() for val
                     in sublist.values()] for item
                 in sublist] for item
             in sublist]
            ))
        self.link_params = {
            g:{'in':{k:(path_prior*path_dir)
                     for k in link_data[g]['in'].keys()},
               'out':{k:(path_prior/path_dir)
                      for k in link_data[g]['out'].keys()}}
            for g in mut_genes
            }
        feature_selection.GenericUnivariateSelect.__init__(self,
            lambda expr,mut: _score_genes(
                expr, mut, expr_genes, self.link_genes,
                link_data, _mut_ttest, self.link_params
                ),
            mode='k_best',
            param=15
            )


class rbfSVM(MutClassifier):
    """A class corresponding to gaussian kernal support vector"""
    """classification of mutation status."""

    _tune_params = {'fit__gamma':[10 ** x for x in range(-10,1,2)],
                    'fit__C':[10 ** x for x in range(-2,5,2)]}

    def __init__(self, gamma=10 ** -8, C=0.001):
        self.params = {'gamma':gamma, 'C':C}
        norm_step = preprocessing.StandardScaler()
        feat_step = feature_selection.SelectFromModel(
            linear_model.LogisticRegression(penalty='l1', C=0.1),
            threshold="2*mean"
            )
        fit_step = svm.SVC(
                    kernel='rbf', C=C, gamma=gamma,
                    class_weight={False:1, True:1},
                    probability=True
                    )
        MutClassifier.__init__(self,
            [('norm', norm_step),
             ('feat', feat_step),
             ('fit', fit_step)])
        self.set_params(**self.params)


class rbfSVMpc(MutClassifier):
    """A class corresponding to gaussian kernal support vector"""
    """classification of mutation status."""

    _tune_params = {'fit__gamma':[10 ** x for x in range(-12,-3,2)],
                    'fit__C':[10 ** x for x in range(-2,5,2)]}


    def __init__(self, mut_genes=None, gamma=0.0001, C=1):
        if mut_genes is None:
            mut_genes = self.mut_genes
        else:
            self.mut_genes = mut_genes
        self.link_data = _read_sif(mut_genes)
        self.link_params = {
            g:{'in':{k:10 for k in self.link_data[g]['in'].keys()},
               'out':{k:100000 for k in self.link_data[g]['out'].keys()}}
            for g in mut_genes
            }
        self.params = {'gamma':gamma, 'C':C}
        norm_step = preprocessing.StandardScaler()
        feat_step = feature_selection.GenericUnivariateSelect(
            lambda expr,mut: _score_genes(
                expr, mut, self.gene_list, _mut_ttest, self.link_data,
                self.link_params),
            mode='percentile',
            param=99
            )
        fit_step = svm.SVC(
            kernel='rbf',
            C=C, gamma=gamma,
            class_weight={False:1, True:1},
            probability=True
            )
        MutClassifier.__init__(self,
            [('norm', norm_step),
             ('feat', feat_step),
             ('fit', fit_step)])
        self.set_params(**self.params)


class KNeigh(MutClassifier):
    """A class corresponding to k-nearest neighbours voting
       classification of mutation status."""
       
    _tune_params = {'fit__n_neighbors':[40,80,120,160,200,240,300]}

    def __init__(self, n_neighbors=100):
        self.params = {'n_neighbors':n_neighbors}
        norm_step = preprocessing.StandardScaler()
        fit_step = neighbors.KNeighborsClassifier(
            n_neighbors=n_neighbors, weights='distance',
            algorithm='ball_tree', leaf_size=20
            )
        MutClassifier.__init__(self,
            [('norm', norm_step), ('fit', fit_step)])
        self.set_params(**self.params)


class RBFgbc(MutClassifier):
    """A class corresponding to gaussian process classifier"""
    """of mutation status."""

    _tune_params = {'kernel':[RBF(10**x) for x in range(-2,5)]}

    def __init__(self, kernel=RBF(0.001)):
        self.params = {'kernel':kernel}
        norm_step = preprocessing.StandardScaler()
        fit_step = gaussian_process.GaussianProcessClassifier(
                        kernel=kernel)
        MutClassifier.__init__(self,
            [('norm', norm_step), ('fit', fit_step)])
        self.set_params(**self.params)


class rForest(MutClassifier):
    """A class corresponding to random forest classification"""
    """of mutation status."""

    _tune_params = {'fit__max_features':[0.01,0.02,0.05,0.1,0.2],
                    'fit__min_samples_leaf':[0.0001,0.02,0.04,0.06]}

    def __init__(self, max_features=0.02, min_samples_leaf=0.0001):
        self.params = {'max_features':max_features,
                       'min_samples_leaf':min_samples_leaf}
        norm_step = preprocessing.StandardScaler()
        fit_step = ensemble.RandomForestClassifier(
                    max_features=max_features,
                    min_samples_leaf=min_samples_leaf,
                    n_estimators=1000, class_weight={False:1, True:1}
                    )
        MutClassifier.__init__(self,
            [('norm', norm_step), ('fit', fit_step)])
        self.set_params(**self.params)




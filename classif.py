
"""
Hetman (Heterogeneity Manifold)
Classification of mutation sub-types using expression data.
This file contains the algorithms used as building blocks for the
classification ensembles of mutation sub-types.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

from data import get_pc2_neighb
from itertools import chain
import re
import dill as pickle
from functools import reduce

from math import log, exp
import numpy as np
from scipy import stats
import dirichlet

from sklearn.feature_selection.base import SelectorMixin
from sklearn.svm.base import BaseSVC
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, PolynomialFeatures, RobustScaler)
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedShuffleSplit)
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import (
    rbf_kernel, check_pairwise_arrays, pairwise_distances)
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.cluster import KMeans


# .. helper functions for classification and feature selection classes ..
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


def _gene_mean(X, y):
    gene_scores = np.apply_along_axis(
        func1d=lambda x: np.mean(x),
        axis=0, arr=X
        )
    return gene_scores, [0 for g in gene_scores]


def _gene_sd(X, y):
    gene_scores = np.apply_along_axis(
        func1d=lambda x: np.var(x) ** 0.5,
        axis=0, arr=X
        )
    return gene_scores, [0 for g in gene_scores]


def _gene_cv(X, y):
    gene_scores = np.apply_along_axis(
        func1d=lambda x: np.mean(x) / (np.var(x) ** 0.5),
        axis=0, arr=X
        )
    return gene_scores, [0 for g in gene_scores]


def _gene_meancv(X, y):
    gene_scores = np.apply_along_axis(
        func1d=lambda x: np.mean(x) * (np.var(x) ** 0.5),
        axis=0, arr=X
        )
    return gene_scores, [0 for g in gene_scores]


class GeneSelect(GenericUnivariateSelect):
    """A class for pathway Commons-based feature selection for use in
       in classification of mutation sub-types.
    """

    def __init__(self,
                 path_prior, path_dir,
                 mut_genes, expr_genes):
        self.mut_genes = mut_genes
        self.expr_genes = expr_genes
        self.link_data = read_sif(mut_genes)
        link_genes = set()
        for gene,directs in list(self.link_data.items()):
            for direction,int_types in list(directs.items()):
                for int_type,genes in list(int_types.items()):
                    link_genes |= set(genes)
        self.link_genes = tuple(link_genes & set(self.expr_genes))
        self.link_indx = {g:self.expr_genes.index(g)
                          for g in self.link_genes}
        self._tune_params = {'path_prior':path_prior, 'path_dir':path_dir}
        GenericUnivariateSelect.__init__(self,
            score_func=self._score_genes, mode='k_best', param=30
            )

    def _score_genes(self, X, y):
        """Scores genes according to a univariate test for association with
        mutation status in conjunction with their interactions with the
        mutated genes.
        """
        gene_scores = np.apply_along_axis(
            func1d=lambda x: _mut_ttest(x, y),
            axis=0, arr=X[:,list(self.link_indx.values())]
            )
        for gene,directs in list(self.link_data.items()):
            for direction,int_types in list(directs.items()):
                for int_type,genes in list(int_types.items()):
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


class PathwaySelect(SelectorMixin):
    """Chooses gene features based on their presence
       in Pathway Commons pathways.
    """

    def __init__(self, path_obj):
        self.path_obj = path_obj
        SelectorMixin.__init__(self)

    def fit(self, X, y, **fit_params):
        self.expr_genes = X.columns
        self.path_genes = list(set(
            chain(*chain(
                *[list(x.values()) for x in list(self.path_obj.values())]
            ))))
        return self

    def _get_support_mask(self):
        return np.array([g in self.path_genes for g in self.expr_genes])

    def get_params(self, deep=True):
        return {'path_obj':self.path_obj}


class UniPipe(Pipeline):
    """A class corresponding to expression-based classifiers of mutation
       status that use a single expr-mut dataset.
    """

    def __init__(self, steps):
        Pipeline.__init__(self, steps)

    def __str__(self):
        param_list = self.get_params()
        return reduce(
            lambda x,y: x + ', ' + y,
            [k + ': ' + str(param_list[k])
             for k in list(self._tune_priors.keys())]
            )

    def prob_mut(self, expr):
        """Returns the probability of mutation presence calculated by the
           classifier based on the given expression matrix.
        """
        mut_scores = self.predict_proba(expr)
        if hasattr(self, 'classes_'):
            true_indx = [i for i,x in enumerate(self.classes_) if x]
        else:
            wghts = tuple(self.named_steps['fit'].weights_)
            true_indx = wghts.index(min(wghts))
        return [m[true_indx] for m in mut_scores]

    @classmethod
    def score_auc(cls, estimator, expr, mut):
        """Computes the AUC score using the classifier on a expr-mut pair.
           Used for compatibility with tuning, scoring methods.

        Parameters
        ----------
        expr : array-like, shape (n_samples,n_features)
            An expression dataset.

        mut : array-like, shape (n_samples,)
            A boolean vector corresponding to the presence of a particular
            type of mutation in the same set of samples as the given
            expression dataset.

        Returns
        -------
        S : float
            The AUC score corresponding to mutation classification accuracy.
        """
        mut_scores = estimator.prob_mut(expr)
        return roc_auc_score(mut, mut_scores)

    def tune(self, expr, mut, cv_samples, test_count=16,
             update_priors=False, verbose=False):
        """Tunes the classifier by sampling over the parameter space and
           choosing the parameters with the best 25th percentile.
        """
        if self._tune_priors:
            grid_test = RandomizedSearchCV(
                estimator=self, param_distributions=self._tune_priors,
                n_iter=test_count, scoring=self.score_auc, cv=cv_samples,
                n_jobs=-1, refit=False
                )
            grid_test.fit(expr, mut)

            tune_scores = (grid_test.cv_results_['mean_test_score']
                           - grid_test.cv_results_['std_test_score'])
            self.set_params(
                **grid_test.cv_results_['params'][tune_scores.argmax()])
            if update_priors:
                self.update_priors(grid_test.cv_results_)
            if verbose:
                print(self)

        return self

    def update_priors(self, cv_results):
        """Experimental method for updating priors of tuning parameter space
           based on the outcome of cross-validation testing.
        """
        for param in self._tune_priors.keys():
            new_perf = [param * (exp(perf))
                        for param,perf in
                zip([param for param,perf in param_list],
                    perf_list)]
            new_mean = reduce(lambda x,y: x*y, new_perf)
            new_mean = new_mean ** (1.0/len(param_list))
            new_sd = np.mean([(log(x) - log(new_mean)) ** 2
                              for x in new_perf]) ** 0.5

            new_mean,new_sd = _update_params(
                [(x[param],y)
                 for x,y in zip(
                     grid_test.cv_results_['params'],
                     grid_test.cv_results_['mean_test_score'])
                    ])
            self._param_priors[param] = stats.lognorm(
                scale=new_mean, s=new_sd)
        return self

    def get_coef(self):
        """Gets the coefficients of the classifier."""
        return self.named_steps['fit'].coefs_


# .. classifiers that don't use any prior information ..
class NaiveBayes(UniPipe):
    """A class corresponding to Gaussian Naive Bayesian classification
       of mutation status.
    """

    def __init__(self, mut_genes=None, expr_genes=None):
        self._tune_priors = {}
        norm_step = StandardScaler()
        fit_step = GaussianNB()
        UniPipe.__init__(self,
            [('norm', norm_step), ('fit', fit_step)])


class Lasso(UniPipe):
    """A class corresponding to logistic regression classification
       of mutation status with the lasso regularization penalty.
    """

    def __init__(self, mut_genes=None, expr_genes=None):
        self._tune_priors = {
            'fit__C':stats.lognorm(scale=exp(-1), s=exp(1))}
        norm_step = StandardScaler()
        fit_step = LogisticRegression(
            penalty='l1', tol=1e-2, class_weight='balanced')
        UniPipe.__init__(self,
            [('norm', norm_step), ('fit', fit_step)])


class LogReg(UniPipe):
    """A class corresponding to logistic regression classification
       of mutation status with the elastic net regularization penalty.
    """

    def __init__(self, mut_genes=None, expr_genes=None):
        self._tune_priors = {
            'fit__alpha':stats.lognorm(scale=exp(1), s=exp(1)),
            'fit__l1_ratio':[0,0.25,0.5,0.75,1.0]}
        norm_step = StandardScaler()
        fit_step = SGDClassifier(
            loss='log', penalty='elasticnet',
            n_iter=100, class_weight='balanced')
        UniPipe.__init__(self,
            [('norm', norm_step), ('fit', fit_step)])


class Ridge(UniPipe):
    """A class corresponding to logistic regression classification
       of mutation status with the lasso regularization penalty.
    """

    def __init__(self, mut_genes=None, expr_genes=None):
        self._tune_priors = {
            'fit__C':stats.lognorm(scale=exp(-1), s=exp(1))}
        norm_step = RobustScaler()
        fit_step = LogisticRegression(
            penalty='l2', tol=1e-2, class_weight='balanced')
        UniPipe.__init__(self,
            [('norm', norm_step), ('fit', fit_step)])


class SVCrbf(UniPipe):
    """A class corresponding to C-support vector classification
       of mutation status with a radial basis kernel.
    """
   
    def __init__(self, mut_genes=None, expr_genes=None):
        self._tune_priors = {
            'fit__C':stats.lognorm(scale=exp(-1), s=exp(1)),
            'fit__gamma':stats.lognorm(scale=1e-5, s=exp(2))}
        norm_step = StandardScaler()
        fit_step = SVC(
            kernel='rbf', probability=True, class_weight='balanced')
        UniPipe.__init__(self,
            [('norm', norm_step), ('fit', fit_step)])


class rForest(UniPipe):
    """A class corresponding to random forest classification
       of mutation status.
    """

    def __init__(self, mut_genes=None, expr_genes=None):
        self._tune_priors = {
            'fit__max_features':[0.01,0.02,0.05,0.1,0.2],
            'fit__min_samples_leaf':[0.0001,0.02,0.04,0.06]}
        norm_step = StandardScaler()
        fit_step = RandomForestClassifier(
                    n_estimators=1000, class_weight='balanced')
        UniPipe.__init__(self,
            [('norm', norm_step), ('fit', fit_step)])


class KNeigh(UniPipe):
    """A class corresponding to k-nearest neighbours voting classification
       of mutation status.
    """

    def __init__(self, mut_genes=None, expr_genes=None):
        self._tune_priors = {
            'fit__n_neighbors':[40,80,120,160,200,240,300],
            'fit__leaf_size':[5,10,15,20,30,50]}
        norm_step = StandardScaler()
        fit_step = KNeighborsClassifier(
            weights='distance', algorithm='ball_tree')
        UniPipe.__init__(self,
            [('norm', norm_step), ('fit', fit_step)])


class GBCrbf(UniPipe):
    """A class corresponding to gaussian process classification
       of mutation status with a radial basis kernel.
    """

    def __init__(self, mut_genes=None, expr_genes=None):
        self._tune_priors = {}
        norm_step = StandardScaler()
        fit_step = GaussianProcessClassifier()
        UniPipe.__init__(self,
            [('norm', norm_step), ('fit', fit_step)])


class NewTest(UniPipe):
    """A class for testing miscellaneous new classification pipelines."""

    def __init__(self, mut_genes=None, expr_genes=None):
        self._tune_priors = {
            'proj__eps':[0.2,0.5,0.8],
            'fit__C':stats.lognorm(scale=exp(-1), s=exp(2))}
        proj_step = GaussianRandomProjection()
        feat_step = PolynomialFeatures(2)
        fit_step = LogisticRegression(
            penalty='l2', tol=1e-2, class_weight='balanced')
        UniPipe.__init__(self,
            [('proj', proj_step), ('feat', feat_step), ('fit', fit_step)])


# .. classifiers that use Pathway Commons as prior information ..
class SVCpath(SVC):
    """A Support Vector Machine classifier with a radial basis kernel
       based on Pathway Commons neighbours.
    """

    def pc_kernel(self, X, Y=None):
        X,Y = check_pairwise_arrays(X,Y)
        out_kern = np.zeros((X.shape[0], Y.shape[0]))
        for dr,dmix in self.dir_mix.items():
            for tp,tmix in self.type_mix.items():
                gn_indx = [i for i,x in enumerate(self.expr_genes)
                           if x in self.path_obj[dr][tp]]
                out_kern += rbf_kernel(X[:, gn_indx], Y[:, gn_indx],
                                       gamma=self.gamma) * dmix * tmix
        return out_kern

    def __init__(self,
                 mut_gene, expr_genes,
                 dir_mix={'Up': 1.0/3, 'Down':2.0/3}, type_mix=None,
                 C=1.0, gamma=1.0):
        self.path_obj = get_pc2_neighb(mut_gene)
        self.expr_genes = expr_genes

        if not dir_mix or dir_mix is None:
            dir_mix = {k:1.0 for k in self.path_obj.keys()}
        self.dir_mix = {k:(v/sum(list(dir_mix.values())))
                   for k,v in dir_mix.items()}
        if not type_mix or type_mix is None:
            type_mix = {
                k:1.0 for k in reduce(
                    lambda x,y: x|y,
                    [set(v.keys()) for v in self.path_obj.values()])
                }
        self.type_mix = {k:(v/sum(list(type_mix.values())))
                         for k,v in type_mix.items()}
        SVC.__init__(self,
                     kernel=self.pc_kernel, C=C, gamma=gamma,
                     probability=True)
        self.set_params(mut_gene=mut_gene)


class SVCpcRS(UniPipe):
    """A class corresponding to Pathway Commons - based classification
       of mutation status using robust standardization.
    """

    def __init__(self, mut_genes, expr_genes):
        self._tune_priors = {
            'fit__C':stats.lognorm(scale=exp(-1), s=exp(1)),
            'fit__gamma':stats.lognorm(scale=1e-5, s=exp(2))}
        norm_step = StandardScaler()
        fit_step = SVCpath(
            mut_gene=mut_genes[0], expr_genes=expr_genes,
            dir_mix={'Down':1.0}, type_mix={'controls-state-change-of':1.0})
        UniPipe.__init__(self,
                               [('norm', norm_step), ('fit', fit_step)])
        self.set_params(mut_genes=mut_genes, expr_genes=expr_genes)


class PCdir(BaseSVC):
   
    def _weight_eucl(self, x, y, weights=None):
        if weights is None:
            weights = [1.0 for i in x]
        k = [w*((i-j)**2) for i,j,w in zip(x,y,weights)]
        return sum(k)

    def _dir_kern(self, X, Y=None, weights=None, gamma=None):
        X,Y = check_pairwise_arrays(X,Y)
        if gamma is None:
            gamma = 2.0
        K = pairwise_distances(
            X, Y, metric=lambda x,y: self._weight_eucl(x,y,weights))
        K *= -gamma
        np.exp(K, K)
        return K

    def _lin_kern(self, X, Y=None, weights=None):
        X,Y = check_pairwise_arrays(X,Y)
        if weights is None:
            weights = [1.0 for i in X.shape[1]]
        return np.dot(weights * X, Y.T)

    def _poly_kern(self, X, Y=None, weights=None):
        X,Y = check_pairwise_arrays(X,Y)
        if weights is None:
            weights = [1.0 for i in X.shape[1]]
        return (np.dot(weights * X, Y.T) + 1.0) ** 2.0

    def _sigmoid_kern(self, X, Y=None, weights=None):
        X,Y = check_pairwise_arrays(X,Y)
        if weights is None:
            weights = [1.0 for i in X.shape[1]]
        return np.dot(X,Y.T)

    def _cor_kern(self, X, Y=None, weights=None):
        X,Y = check_pairwise_arrays(X,Y)
        if weights is None:
            weights = [1.0 for i in X.shape[1]]
        M_x = np.dot(X, weights.T)
        M_y = np.dot(Y, weights.T)
        diff_x = (X.T - M_x).T
        diff_y = (Y.T - M_y).T
        S_x = np.dot(diff_x**2.0, weights.T)
        S_y = np.dot(diff_y**2.0, weights.T)
        S_xy = np.dot(weights * diff_x, diff_y.T)
        return abs(S_xy / (np.outer(S_x,S_y)**0.5))

    def _diss_kern(self, X, Y=None, weights=None):
        X,Y = check_pairwise_arrays(X,Y)
        if weights is None:
            weights = [1.0 for i in X.shape[1]]
        M_x = np.dot(X, weights.T)
        M_y = np.dot(Y, weights.T)
        diff_x = (X.T - M_x).T
        diff_y = (Y.T - M_y).T
        S_x = np.dot(diff_x**2.0, weights.T)
        S_y = np.dot(diff_y**2.0, weights.T)
        S_xy = np.dot(weights * diff_x, diff_y.T)
        return (1.0 + (S_xy / (np.outer(S_x,S_y)**0.5))) / 2

    def test_kern(self, X, y):
        self.clf.fit(X, y)
        pred_labs = np.sign(y - 0.5)
        pred_distance = self.clf.decision_function(X)
        self.dist += [np.mean(np.sign(pred_distance))]
        return np.mean([dis*labs for dis,labs in zip(pred_distance,pred_labs)])

    def dis_regr(self, X, y):
        mut_vec = y - np.mean(y)
        H_mat = np.outer(mut_vec * np.inner(mut_vec,mut_vec) ** -1.0,
                         mut_vec)
        norm_mat = np.eye(X.shape[0]) - np.ones([X.shape[0]]*2) / X.shape[0]
        G_mat = np.dot(np.dot(norm_mat, X), norm_mat)
        nH_mat = np.eye(X.shape[0]) - H_mat
        return (np.trace(np.dot(np.dot(H_mat, G_mat), H_mat))
                / np.trace(np.dot(np.dot(nH_mat, G_mat), nH_mat)))

    def dist_metr(self, X, y):
        same_dist = (np.mean(np.triu(X[np.ix_(y, y)], k=1))
                     + np.mean(np.triu(X[np.ix_(~y, ~y)], k=1))) / 2.0
        diff_dist = np.mean(X[np.ix_(y, ~y)])
        return same_dist - diff_dist

    def __init__(self, n_wghts=250):
        self.clf = SVC(C=1.0, kernel='precomputed', probability=False,
                       class_weight='balanced')
        self.clf_prob = SVC(C=1.0, kernel='precomputed', probability=True,
                           class_weight='balanced')
        self.n_wghts = n_wghts
        self.dist = []

    def fit(self, X, y, **fit_params):
        X = np.array(X)
        y = np.array(y)
        self.n_samp = int(round(X.shape[0] ** 0.5))
        dir_params = [1.0 for x in range(X.shape[1])]
        convergence = False
        iter_count = 0
        cur_score = float('-inf')
        while not convergence and iter_count < 100:
            new_wghts = np.random.dirichlet(
                dir_params, size=self.n_wghts)
            new_samps = [
                x[0] for x in StratifiedShuffleSplit(
                    n_splits=self.n_samp, train_size=X.shape[0]**-exp(-1)
                    ).split(X,y)
                ]
            test_mat = np.zeros([len(new_wghts), len(new_samps)])
            for s in range(len(new_samps)):
                for w in range(len(new_wghts)):
                    new_kern = self._lin_kern(
                        X[new_samps[s],], weights=new_wghts[w])
                    test_mat[w,s] = self.test_kern(
                        new_kern, y[new_samps[s].tolist()])
            prm_likeli = np.apply_along_axis(np.percentile, 1, test_mat, 25)
            new_score = np.mean(prm_likeli)
            if new_score < cur_score and iter_count > 30:
                convergence = True
            else:
                cur_score = new_score
                if (iter_count % 10) == 0:
                    print(('Iter ' + str(iter_count) + ', perf: '
                           + str(round(new_score, 4))))
                prm_likeli = ((prm_likeli - min(prm_likeli))
                          / (max(prm_likeli) - min(prm_likeli))) ** 2.0
                prior_lglike = np.array([dirichlet.loglikelihood(
                    np.array([new_wghts[i,:]]),
                    np.array(dir_params))
                    for i in range(new_wghts.shape[0])])
                prior_lglike = ((prior_lglike - max(prior_lglike)) /
                                (min(prior_lglike) - max(prior_lglike)))
                if not any(np.isnan(prior_lglike)):
                    prm_likeli = (prm_likeli ** 0.67) * (prior_lglike ** 0.33)
                prm_likeli = prm_likeli / sum(prm_likeli)
                prm_draw = np.random.choice(
                    range(new_wghts.shape[0]),
                    size=int(round(self.n_wghts**0.5)),
                    replace=True, p=prm_likeli)
                if (np.mean(np.apply_along_axis(np.var, 1, new_wghts[prm_draw,:]))
                    < 1e-6):
                    convergence = True
                else:
                    try:
                        new_params = dirichlet.mle(
                            new_wghts[prm_draw,:], tol=1e-3, maxiter=int(1e5))
                        if (np.mean(abs(new_params - dir_params)) < 0.1
                            or np.mean(new_params) > 20):
                            convergence = True
                        else:
                            iter_count += 1
                            dir_params = new_params
                            print(np.round(np.mean(new_params), 3))
                    except:
                        convergence = True
        self.feat_wghts = np.random.dirichlet(dir_params)
        self.old_kern = self._lin_kern(X, weights=self.feat_wghts)
        self.old_X = X
        self.clf_prob.fit(self.old_kern, y)
        self.classes_ = self.clf_prob.classes_
        return self

    def predict(self, X, **fit_params):
        new_kern = self._lin_kern(
            X=self.old_X, Y=X, weights=self.feat_wghts).T
        return self.clf_prob.predict(new_kern)

    def predict_proba(self, X, **fit_params):
        new_kern = self._lin_kern(
            X=self.old_X, Y=X, weights=self.feat_wghts).T
        return self.clf_prob.predict_proba(new_kern)


class PathwayCluster(Pipeline):

    def __init__(self, path_key, expr_genes):
        feat_step = PathwaySelect(path_key, expr_genes)
        norm_step = StandardScaler()
        fit_step = KMeans(n_clusters=2, n_init=20, init='random')
        Pipeline.__init__(self,
            [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)])


class PCsvm(UniPipe):
    """A class corresponding to SVM classification of mutation status using
       a convex combination of kernels based on Pathway Commons.
    """

    def __init__(self, mut_genes):
        self._tune_priors = {
            'fit__PCdir':stats.beta(a=1, b=1),
            'fit__PCtype':stats.beta(a=1, b=1),
            'fit__C':stats.lognorm(scale=exp(-1), s=exp(1)),
            'fit__gamma':stats.lognorm(scale=exp(1), s=exp(1))
            }
        self.path_obj = list(read_sif(mut_genes).values())[0]
        self.mut_genes = mut_genes
        feat_step = PathwaySelect(self.path_obj)
        norm_step = StandardScaler()
        fit_step = PCsvc(self.path_obj)
        UniPipe.__init__(self,
            [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)])

    def fit(self, X, y, **fit_params):
        self.expr_genes = X.columns
        new_X = self.named_steps['feat'].fit(X, y, **fit_params).transform(X)
        mask = self.named_steps['feat']._get_support_mask()
        old_cols = X.loc[:,mask].columns
        self.named_steps['fit'].expr_out_indx = [
            old_cols.get_loc(g)
            for g in self.path_obj['out']['controls-expression-of']
            if g in old_cols]
        self.named_steps['fit'].expr_in_indx = [
            old_cols.get_loc(g)
            for g in self.path_obj['in']['controls-expression-of']
            if g in old_cols]
        self.named_steps['fit'].state_out_indx = [
            old_cols.get_loc(g)
            for g in self.path_obj['out']['controls-state-change-of']
            if g in old_cols]
        self.named_steps['fit'].state_in_indx = [
            old_cols.get_loc(g)
            for g in self.path_obj['in']['controls-state-change-of']
            if g in old_cols]
        new_X = self.named_steps['norm'].fit_transform(new_X, y, **fit_params)
        self.named_steps['fit'].fit(new_X, y, **fit_params)
        return self


class PCpipe(UniPipe):
    """A class corresponding to
    """

    def __init__(self, mut_genes):
        self._tune_priors = {}
        path_obj = list(read_sif(mut_genes).values())[0]
        self.path_obj = {k:v for k,v in list(path_obj.items()) if k =='out'}
        self.mut_genes = mut_genes
        feat_step = PathwaySelect(self.path_obj)
        norm_step = RobustScaler()
        fit_step = PCdir()
        UniPipe.__init__(self,
            [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)])


class PCgbc(UniPipe):
    """A class corresponding to
    """

    def _pc_kernel(X, Y):
        state_kern = metrics.pairwise.rbf_kernel(
            X.ix[:,self._kern_genes[0]], Y.ix[:,self._kern_genes[0]],
            gamma=1000)
        expr_kern = metrics.pairwise.rbf_kernel(
            X.ix[:,self._kern_genes[1]], Y.ix[:,self._kern_genes[1]],
            gamma=1000)
        return (state_kern * self.get_params()['I']
                + expr_kern * self.get_params()['I'] ** -1)

    def __init__(self, mut_genes=None, expr_genes=None):
        self._tune_params = {'I':1.0}
        self._param_priors = {'I':stats.lognorm(scale=1e-3, s=3)}
        self._expr_genes = expr_genes
        kern_genes = {k:list(set(v) & set(expr_genes)) for k,v in
                      list(read_sif(mut_genes)[mut_genes[0]]['out'].items())}
        self._kern_genes = {k:[[i for i,g
                                in enumerate(expr_genes) if g == x][0]
                               for x in v]
                            for k,v in list(kern_genes.items())}
        norm_step = StandardScaler()
        fit_step = gaussian_process.GaussianProcessClassifier(
            kernel=self._pc_kernel)
        UniPipe.__init__(self,
            [('norm', norm_step), ('fit', fit_step)])
        self.set_params(**self._tune_params)


class PolyLasso(UniPipe):
    """A class corresponding to logistic regression classification"""
    """of mutation status with the lasso regularization penalty."""

    def __init__(self, mut_genes=None, expr_genes=None):
        self._tune_priors = {'fit__C':stats.lognorm(scale=1.0, s=2)}
        feat_step = GenericUnivariateSelect(
            score_func=_gene_sd,
            mode='k_best',
            param = 200
            )
        norm_step = StandardScaler()
        poly_step = PolynomialFeatures(2)
        fit_step = LogisticRegression(penalty='l1', tol=5e-3)
        UniPipe.__init__(self,
            [('feat', feat_step), ('poly', poly_step),
             ('norm', norm_step), ('fit', fit_step)])


class Mixture(UniPipe):
    """A class corresponding to gaussian mixture model
       classification of mutation status.
    """
    def __init__(self, mut_genes, expr_genes):
        self.mut_genes = mut_genes
        self.expr_genes = expr_genes
        self._tune_params = {'feat__path_prior':10**6, 'feat__path_dir':1}
        self._param_priors = {
            'feat__path_prior':stats.lognorm(scale=exp(6), s=3),
            'feat__path_dir':stats.lognorm(scale=exp(0), s=1)
            }
        norm_step = StandardScaler()
        feat_step = GeneSelect(
            self._tune_params['feat__path_prior'],
            self._tune_params['feat__path_dir'],
            self.mut_genes, self.expr_genes
            )
        fit_step = mixture.GaussianMixture(
            n_components=2, covariance_type='spherical',
            tol=5e-3, max_iter=50, 
            n_init=3, init_params='kmeans', weights_init=[0.8,0.2]
            )
        UniPipe.__init__(self,
            [('norm', norm_step),
             ('feat', feat_step),
             ('fit', fit_step)])
        self.set_params(**self._tune_params)


class PolyrbfSVM(UniPipe):
    """A class corresponding to gaussian kernal support vector"""
    """classification of mutation status."""

    _tune_params = {'fit__gamma':[10 ** x for x in range(-10,1,2)],
                    'fit__C':[10 ** x for x in range(-2,5,2)]}

    def __init__(self, mut_genes=None, expr_genes=None):
        self.mut_genes = mut_genes
        self.expr_genes = expr_genes
        self._tune_params = {'fit__gamma':exp(-10), 'fit__C':exp(-5)}
        self._param_priors = {'fit__gamma':stats.lognorm(scale=exp(-10), s=4),
                              'fit__C':stats.lognorm(scale=exp(-5), s=2)}
        norm_step = StandardScaler()
        feat_step = GeneSelect(
            10**8, 1, self.mut_genes, self.expr_genes)
        poly_step = PolynomialFeatures(2)
        fit_step = SVC(
            kernel='rbf',
            C=self._tune_params['fit__C'],
            gamma=self._tune_params['fit__gamma'],
            class_weight={False:1, True:1}, probability=True
            )
        UniPipe.__init__(self,
            [('norm', norm_step),
             ('feat', feat_step),
             ('poly', poly_step),
             ('fit', fit_step)])
        self.set_params(**self._tune_params)


class rbfSVMpc(UniPipe):
    """A class corresponding toz gaussian kernal support vector"""
    """classification of mutation status."""

    _tune_params = {'fit__gamma':[10 ** x for x in range(-12,-3,2)],
                    'fit__C':[10 ** x for x in range(-2,5,2)]}


    def __init__(self, mut_genes=None, expr_genes=None):
        self._tune_params = {'fizt__gamma':-6.0}
        self._param_priors = {'fit__gamma':stats.lognorm(scale=exp(-6), s=3)}
        self._tune_params = {'fit__C':1.0}
        self._param_priors = {'fit__C':stats.lognorm(scale=exp(1.0), s=1)}
        norm_step = StandardScaler()
        self.link_data = read_sif(mut_genes)
        self.link_params = {
            g:{'in':{k:10 for k in list(self.link_data[g]['in'].keys())},
               'out':{k:100000 for k in list(self.link_data[g]['out'].keys())}}
            for g in mut_genes
            }
        self.params = {'gamma':gamma, 'C':C}
        feat_step = GenericUnivariateSelect(
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
        UniPipe.__init__(self,
            [('norm', norm_step),
             ('feat', feat_step),
             ('fit', fit_step)])
        self.set_params(**self.params)




"""
Hetman (Heterogeneity Manifold)
Prediction of mutation sub-types using expression data.
This file contains the algorithms used to predict discrete mutation states.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

from .pipelines import ClassPipe
from .selection import PathwaySelect

from itertools import chain
import re
import dill as pickle
from functools import reduce

from math import log, exp
import numpy as np
from scipy import stats
import dirichlet

from sklearn.svm.base import BaseSVC
from sklearn.metrics.pairwise import (
    rbf_kernel, check_pairwise_arrays, pairwise_distances)

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.cluster import KMeans


# .. classifiers that don't use any prior information ..
class NaiveBayes(ClassPipe):
    """A class corresponding to Gaussian Naive Bayesian classification
       of mutation status.
    """

    def __init__(self, path_keys=None):
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = StandardScaler()
        fit_step = GaussianNB()
        ClassPipe.__init__(self,
            [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
            path_keys)


class RobustNB(ClassPipe):
    """A class corresponding to Gaussian Naive Bayesian classification
       of mutation status with robust feature scaling.
    """

    def __init__(self, path_keys=None):
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = RobustScaler()
        fit_step = GaussianNB()
        ClassPipe.__init__(self,
            [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
            path_keys)


class Lasso(ClassPipe):
    """A class corresponding to logistic regression classification
       of mutation status with the lasso regularization penalty.
    """

    tune_priors = (
        ('fit__C', stats.lognorm(scale=exp(-1), s=exp(1))),
    )

    def __init__(self, path_keys=None):
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = StandardScaler()
        fit_step = LogisticRegression(
            penalty='l1', tol=1e-2, class_weight='balanced')
        ClassPipe.__init__(self,
            [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
            path_keys)


class LogReg(ClassPipe):
    """A class corresponding to logistic regression classification
       of mutation status with the elastic net regularization penalty.
    """

    tune_priors = (
        ('fit__alpha', stats.lognorm(scale=exp(1), s=exp(1))),
        ('fit__l1_ratio', (0.05,0.25,0.5,0.75,0.95))
    )

    def __init__(self, path_keys=None):
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = StandardScaler()
        fit_step = SGDClassifier(
            loss='log', penalty='elasticnet',
            n_iter=100, class_weight='balanced')
        ClassPipe.__init__(self,
            [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
            path_keys)


class Ridge(ClassPipe):
    """A class corresponding to logistic regression classification
       of mutation status with the ridge regularization penalty.
    """

    tune_priors = (
        ('fit__C', stats.lognorm(scale=exp(-1), s=exp(1))),
    )

    def __init__(self, path_keys=None):
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = StandardScaler()
        fit_step = LogisticRegression(
            penalty='l1', tol=1e-2, class_weight='balanced')
        ClassPipe.__init__(self,
            [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
            path_keys)


class SVCpoly(ClassPipe):
    """A class corresponding to C-support vector classification
       of mutation status with a radial basis kernel.
    """
   
    tune_priors = (
        ('fit__C', stats.lognorm(scale=exp(-1), s=exp(1))),
        ('fit__coef0', [-2., -1., -0.5, 0., 0.5, 1., 2.]),
    )

    def __init__(self, path_keys=None):
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = StandardScaler()
        fit_step = SVC(
            kernel='poly', probability=True, degree=2,
            cache_size=500, class_weight='balanced')
        ClassPipe.__init__(self,
            [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
            path_keys)


class SVCrbf(ClassPipe):
    """A class corresponding to C-support vector classification
       of mutation status with a radial basis kernel.
    """
   
    tune_priors = (
        ('fit__C', stats.lognorm(scale=exp(-1), s=exp(2))),
        ('fit__gamma', stats.lognorm(scale=1e-4, s=exp(2)))
    )

    def __init__(self, path_keys=None):
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = StandardScaler()
        fit_step = SVC(
            kernel='rbf', probability=True,
            cache_size=500, class_weight='balanced')
        ClassPipe.__init__(self,
            [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
            path_keys)


class rForest(ClassPipe):
    """A class corresponding to random forest classification
       of mutation status.
    """

    tune_priors = (
        ('fit__max_features', (0.005,0.01,0.02,0.04,0.08,0.15)),
        ('fit__min_samples_leaf', (0.0001,0.01,0.05))
    )

    def __init__(self, path_keys=None):
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = StandardScaler()
        fit_step = RandomForestClassifier(
                    n_estimators=500, class_weight='balanced')
        ClassPipe.__init__(self,
            [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
            path_keys)


class KNeigh(ClassPipe):
    """A class corresponding to k-nearest neighbours voting classification
       of mutation status.
    """
    
    tune_priors = (
        ('fit__n_neighbors', (4,8,16,25,40)),
    )

    def __init__(self, path_keys=None):
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = StandardScaler()
        fit_step = KNeighborsClassifier(
            weights='distance', algorithm='ball_tree')
        ClassPipe.__init__(self,
            [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
            path_keys)


class GBCrbf(ClassPipe):
    """A class corresponding to gaussian process classification
       of mutation status with a radial basis kernel.
    """

    def __init__(self, mut_genes=None, expr_genes=None):
        self._tune_priors = {}
        norm_step = StandardScaler()
        fit_step = GaussianProcessClassifier()
        ClassPipe.__init__(self,
            [('norm', norm_step), ('fit', fit_step)])


class NewTest(ClassPipe):
    """A class for testing miscellaneous new classification pipelines."""

    def __init__(self, mut_genes=None, expr_genes=None):
        self._tune_priors = {
            'proj__eps':[0.2,0.5,0.8],
            'fit__C':stats.lognorm(scale=exp(-1), s=exp(2))}
        proj_step = GaussianRandomProjection()
        feat_step = PolynomialFeatures(2)
        fit_step = LogisticRegression(
            penalty='l2', tol=1e-2, class_weight='balanced')
        ClassPipe.__init__(self,
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


class SVCpcRS(ClassPipe):
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
        ClassPipe.__init__(self,
                               [('norm', norm_step), ('fit', fit_step)])
        self.set_params(mut_genes=mut_genes, expr_genes=expr_genes)


class PCdir(BaseSVC):
    """A classifier that forms a kernel based on gene pathway presence.
    """
   
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


class PathwayCluster(ClassPipe):

    def __init__(self, path_key, expr_genes):
        feat_step = PathwaySelect(path_key, expr_genes)
        norm_step = StandardScaler()
        fit_step = KMeans(n_clusters=2, n_init=20, init='random')
        Pipeline.__init__(self,
            [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)])


class PCsvm(ClassPipe):
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
        ClassPipe.__init__(self,
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


class PCpipe(ClassPipe):
    """A class corresponding to
    """

    def __init__(self, path_keys=None):
        self._tune_priors = {}
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = StandardScaler()
        fit_step = PCdir()
        ClassPipe.__init__(self,
            [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)])
        self.set_params(path_keys=path_keys)


class PCgbc(ClassPipe):
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
        ClassPipe.__init__(self,
            [('norm', norm_step), ('fit', fit_step)])
        self.set_params(**self._tune_params)


class PolyLasso(ClassPipe):
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
        ClassPipe.__init__(self,
            [('feat', feat_step), ('poly', poly_step),
             ('norm', norm_step), ('fit', fit_step)])


class Mixture(ClassPipe):
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
        ClassPipe.__init__(self,
            [('norm', norm_step),
             ('feat', feat_step),
             ('fit', fit_step)])
        self.set_params(**self._tune_params)


class PolyrbfSVM(ClassPipe):
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
        ClassPipe.__init__(self,
            [('norm', norm_step),
             ('feat', feat_step),
             ('poly', poly_step),
             ('fit', fit_step)])
        self.set_params(**self._tune_params)


class rbfSVMpc(ClassPipe):
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
        ClassPipe.__init__(self,
            [('norm', norm_step),
             ('feat', feat_step),
             ('fit', fit_step)])
        self.set_params(**self.params)



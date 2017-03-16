
"""
KBTL (Kernelized Bayesian Transfer Learning)
Sharing knowledge between related but distinct tasks to improve
classification using an efficient variational approximation techniques.

Ported into python from the original Matlab code written by Mehmet Gonen and
available at https://github.com/mehmetgonen/kbtl and described in further
detail in http://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8132.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

from classif import PathwaySelect

from random import gauss as rnorm
from math import log, exp
import numpy as np
from scipy import stats

import collections
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, PolynomialFeatures, RobustScaler)
import sklearn.model_selection._validation
from sklearn.utils import safe_indexing


# .. helper functions for use in transfer learning ..
def bhatta_dist(dist1, dist2):
    """Calculates Bhattacharyya distance between two normal distributions.
       See https://en.wikipedia.org/wiki/Bhattacharyya_distance for details.

    Parameters
    ----------
    dist1, dist2 : dicts
        A pair of normal distributions each characterized by a dictionary
        with 'mu' and 'sigma' fields.

    Returns
    -------
    D : The distance between the two distributions. Useful for measuring how
        close the prior distributions of transfer learning parameters are to
        convergence.
    """

    return (0.25 * log(0.25 * ((dist1['sigma'] / dist2['sigma'])
                               + (dist2['sigma'] / dist1['sigma'])
                               + 2.0))
            + 0.25 * (((dist1['mu'] - dist2['mu']) ** 2.0)
                      / (dist1['sigma'] + dist2['sigma']))
           )


def _safe_split(estimator, X_list, y_list, indx_list, train_indx_list=None):
    """Overrides sklearn function for getting the training split of an input
       dataset to accomodate transfer learning datasets with multiple tasks.
    """

    # makes sure kernel matrix is precomputed
    from sklearn.gaussian_process.kernels import Kernel as GPKernel
    if (hasattr(estimator, 'kernel') and callable(estimator.kernel)
        and not isinstance(estimator.kernel, GPKernel)):
            raise ValueError("Cannot use a custom kernel function, "
                             "precompute the kernel matrix instead.")
    
    X_subset = [[] for _ in X_list]
    y_subset = [[] for _ in y_list]

    # for each transfer task, checks format of task kernel and forms
    # a training split accordingly
    for i, (X,indices) in enumerate(zip(X_list, indx_list)):
        if not hasattr(X, "shape"):
            if getattr(estimator, "_pairwise", False):
                raise ValueError(
                    "Precomputed kernels or affinity matrices have "
                    "to be passed as arrays or sparse matrices.")
            X_subset[i] = [X[index] for index in indices]

        else:
            if getattr(estimator, "_pairwise", False):
                if X.shape[0] != X.shape[1]:
                    raise ValueError("X should be a square kernel matrix.")
                if train_indx_list is None:
                    X_subset[i] = X[np.ix_(indices, indices)]
                else:
                    X_subset[i] = X[np.ix_(indices, train_indx_list[i])]
            else:
                X_subset[i] = safe_indexing(X, indices)

        if y_list[i] is not None:
            y_subset[i] = safe_indexing(y_list[i], indices)
        else:
            y_subset[i] = None

    return X_subset, y_subset


# .. classes for integrating transfer learning into the sklearn framework ..
class MultiPipe(Pipeline):
    """A class corresponding to expression-based classifiers of mutation
       status that use multiple expr-mut datasets ("tasks").
    """

    def __init__(self, steps):
        super(MultiPipe, self).__init__(steps)

    def _fit(self, X_list, y_list=None, **fit_params):
        """Fit the model - fit all of the transforms in succession and
           transform the data, then fit the transformed data using the
           final estimator.
        """

        # parse the passed parameters for each step of the pipeline
        self._validate_steps()
        fit_params_steps = dict((name, {}) for name, step in self.steps
                                if step is not None)
        for pname, pval in fit_params.items():
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        Xt_list = X_list
        if y_list is None:
            y_list = [None for _ in X_list]

        # for each transform in the pipeline, split it into a separate
        # copy for each of the tasks
        for i in range(len(self.steps[:-1])):
            if self.steps[i][1] is not None:
                self.steps[i] = (self.steps[i][0],
                                 tuple(type(self.steps[i][1])()
                                       for _ in X_list))

        # apply each transform to each task
        for name, transform in self.steps[:-1]:
            if transform is None:
                pass
            elif hasattr(transform, "fit_transform"):
                Xt_list = [transform[i].fit_transform(
                    Xt, y, **fit_params_steps[name])
                    for i, (Xt,y) in enumerate(zip(Xt_list, y_list))]
            else:
                Xt_list = [transform[i].fit(
                    Xt, y, **fit_params_steps[name]).transform(Xt)
                    for i, (Xt,y) in enumerate(zip(Xt_list, y_list))]

        # if we have a final estimator, return its parameters as well
        if self._final_estimator is None:
            return Xt_list, {}
        else:
            return Xt_list, fit_params_steps[self.steps[-1][0]]

    def _pretransform(self, X_list):
        """Apply all the transformers to each of the tasks."""
        Xt_list = X_list
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt_list = [trx.transform(Xt)
                           for trx, Xt in zip(transform, Xt_list)]
        return Xt_list

    def predict(self, X_list):
        """Apply transforms to the data, and predict
           with the final estimator.
        """
        Xt_list = self._pretransform(X_list)
        return self.steps[-1][-1].predict(Xt_list)

    def predict_proba(self, X_list):
        """Apply transforms to the data, and predict probabilities
           with the final estimator.
        """
        Xt_list = self._pretransform(X_list)
        return self.steps[-1][-1].predict_proba(Xt_list)

    @classmethod
    def score_auc(cls, estimator, expr_list, mut_list):
        """Scores the predictions made by the classifier
           across different tasks.
        """
        mut_scores = estimator.predict_proba(expr_list)
        return np.mean([metrics.roc_auc_score(mut, mut_sc)
                        for mut,mut_sc in zip(mut_list, mut_scores)])

    def tune(self, expr_list, mut_list, mut_genes, path_obj,
             cv_samples, test_count=8, update_priors=False, verbose=False):
        """Tune the hyperparameters of the classifier using internal
           cross-validation.
        """

        if self._tune_priors:
            # get internal splits of the tasks using random sampling
            sklearn.model_selection._validation._safe_split = _safe_split
            grid_test = sklearn.model_selection._search.RandomizedSearchCV(
                estimator=self, param_distributions=self._tune_priors,
                fit_params={'feat__mut_genes': mut_genes,
                            'feat__path_obj': path_obj},
                n_iter=test_count, scoring=self.score_auc, cv=cv_samples)

            # score the classifier on each combination of split and parameter
            # value and update the classifier with the best values
            grid_test.fit(expr_list, mut_list)
            tune_scores = (grid_test.cv_results_['mean_test_score']
                           - grid_test.cv_results_['std_test_score'])
            self.set_params(
                **grid_test.cv_results_['params'][tune_scores.argmax()])

        return self


class MultiClf(MultiPipe):
    """A class corresponding to expression-based classifiers of mutation
       status that use multiple expr-mut datasets.
    """

    def __init__(self, path_keys=None):
        self._tune_priors = {
            'fit__sigma_h': stats.lognorm(scale=exp(-1), s=exp(1))}
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = StandardScaler()
        fit_step = KBTL()
        MultiPipe.__init__(self,
            [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)])
        self.set_params(path_keys=path_keys)


# .. transfer learning classifiers ..
class KBTL(object):
    """Kernel based transfer learning classifier adapted for use in HetMan
       optimizers. 

    Parameters
    ----------
    kernel : str, default 'rbf'
        Which function to use for kernel-based dimensionality reduction.
        Default is to use the Gaussian kernel with width equal to the mean
        of pairwise distances between the training data points, calculated
        separately for each domain.

    R : int, default 20
        How many features to use in the shared classification subspace.
        Must be a positive integer.


    """

    def __init__(self,
                 kernel='rbf', R=20, sigma_h=0.1,
                 lambda_par=(1.0, 1.0), gamma_par=(1.0, 1.0),
                 eta_par=(1.0, 1.0), margin=1.0,
                 max_iter=50, stopping_tol=1e-3):
        self.kernel = kernel
        self.R = R
        self.sigma_h = sigma_h
        self.lambda_par = {'a':lambda_par[0], 'b':lambda_par[1]}
        self.gamma_par = {'a':gamma_par[0], 'b':gamma_par[1]}
        self.eta_par = {'a':eta_par[0], 'b':eta_par[1]}
        self.margin = margin
        self.max_iter = max_iter
        self.stopping_tol = stopping_tol

    def compute_kernel(self, X_list):
        """Gets the kernel matrices from a list of feature matrices."""

        if isinstance(self.kernel, collections.Callable):
            kernel_list = [self.kernel(X, X) for X in X_list]
        elif self.kernel == 'rbf':
            X_gamma = [np.mean(metrics.pairwise.pairwise_distances(x))
                       for x in X_list]
            kernel_list = [metrics.pairwise.rbf_kernel(x,gamma=gam**-2.0)
                           for x,gam in zip(X_list, X_gamma)]
        elif self.kernel == 'linear':
            pass
        else:
            raise InputError("Unknown kernel "
                             + str(self.kernel) + " specified!")

        return kernel_list 

    def fit(self, X_list, y_list, verbose=False):
        """Fits the classifier."""
        kernel_list = self.compute_kernel(X_list)
        data_count = [x.shape[0] for x in X_list]
        y_list = [[1.0 if x else -1.0 for x in y] for y in y_list]

        # initializes matrix of priors for task-specific projection matrices
        lambdas = [{'alpha': np.matrix([[self.lambda_par['a'] + 0.5
                                         for i in range(self.R)]
                                        for j in range(d_count)]),
                    'beta': np.matrix([[self.lambda_par['b']
                                        for i in range(self.R)]
                                       for j in range(d_count)])}
                   for d_count in data_count]

        # initializes task-specific projection matrices
        A_list = [{'mu': np.matrix([[rnorm(0,1)
                                     for i in range(self.R)]
                                    for j in range(d_count)]),
                   'sigma': np.array([np.diag([1.0 for i in range(d_count)])
                                      for r in range(self.R)])}
                  for d_count in data_count]

        # initializes task-specific representations in shared sub-space
        H_list = [{'mu': np.matrix([[rnorm(0,1)
                                     for i in range(d_count)]
                                    for j in range(self.R)]),
                   'sigma': np.matrix(
                       np.diag([1.0 for i in range(self.R)]))}
                  for d_count in data_count]

        # initializes hyper-parameters
        gamma_alpha = self.gamma_par['a'] + 0.5
        gamma_beta = self.gamma_par['b']
        eta_alpha = [self.eta_par['a'] + 0.5 for i in range(self.R)]
        eta_beta = [self.eta_par['b'] for i in range(self.R)]
        bw_mu = [0]
        bw_mu.extend([rnorm(0,1) for i in range(self.R)])
        bw_sigma = np.matrix(np.diag([1 for i in range(self.R+1)]))

        # initializes predicted outputs
        f_list = [{'mu': [(abs(rnorm(0,1)) + self.margin)
                          * np.sign(lbl[i]) for i in range(d_count)],
                   'sigma': [1 for i in range(d_count)]}
                  for d_count,lbl in zip(data_count, y_list)]

        # precomputes kernel crossproducts, initializes lower-upper matrix
        kkt_list = [np.dot(x,x.transpose()) for x in kernel_list]
        lu_list = [{'lower': [-1e40 if y_list[j][i] <= 0 else self.margin
                              for i in range(d_count)],
                    'upper': [1e40 if y_list[j][i] >= 0 else -self.margin
                              for i in range(d_count)]}
                   for j,d_count in enumerate(data_count)]

        # does inference using variational Bayes for the given number
        # of iterations
        cur_iter = 1
        f_dist = 1e10
        while(cur_iter <= self.max_iter and f_dist > self.stopping_tol):
            if verbose and (cur_iter % 10) == 0:
                print(("On iteration " + str(cur_iter) + "...."))
                print(("gamma_beta: " + str(round(gamma_beta, 4))))
            for i in range(len(X_list)):

                # updates posterior distributions of projection priors
                for j in range(self.R):
                    lambdas[i]['beta'][:,j] = np.power(
                        (self.lambda_par['b'] ** -1
                         + 0.5 * (
                             np.power(A_list[i]['mu'][:,j].transpose(), 2)
                             + np.diag(A_list[i]['sigma'][j,:,:])
                         )), -1).transpose()

                # updates posterior distributions of projection matrices
                for j in range(self.R):
                    A_list[i]['sigma'][j,:,:] = np.linalg.inv(
                        np.diag(np.multiply(
                            lambdas[i]['alpha'][:,j],
                            lambdas[i]['beta'][:,j]).transpose().tolist()[0]
                            ) + (kkt_list[i] / self.sigma_h ** 2)
                        )
                    A_list[i]['mu'][:,j] = np.dot(
                        A_list[i]['sigma'][j,:,:],
                        np.dot(
                            kernel_list[i],
                            H_list[i]['mu'][j,].transpose()
                            ) / self.sigma_h ** 2
                        )

                # updates posterior distributions of representations
                H_list[i]['sigma'] = np.linalg.inv(
                    np.diag([self.sigma_h ** -2 for r in range(self.R)])
                    + np.outer(bw_mu[1:], bw_mu[1:])
                    + bw_sigma[1:,1:]
                    )
                H_list[i]['mu'] = np.dot(
                    H_list[i]['sigma'],
                    (np.dot(
                        A_list[i]['mu'].transpose(),
                        kernel_list[i]) / self.sigma_h ** 2)
                    + np.outer(bw_mu[1:], f_list[i]['mu'])
                    - np.array([
                        [x*bw_mu[0] + y for x,y in
                         zip(bw_mu[1:],
                             bw_sigma[1:,0].transpose().tolist()[0])]
                        for c in range(data_count[i])]).transpose()
                    )

            # updates posterior distributions of classification priors
            # in the shared subspace
            gamma_beta = (
                self.gamma_par['b'] ** -1
                + 0.5 * (bw_mu[0] ** 2 + bw_sigma[0,0])
                ) ** -1
            eta_beta = np.power(
                self.gamma_par['b'] ** -1
                + 0.5 * (np.power(bw_mu[1:], 2)
                         + np.diag(bw_sigma[1:,1:])),
                -1)

            # updates posterior distributions of classification parameters
            # in the shared subspace
            bw_sigma = [self.gamma_par['a'] * self.gamma_par['b']]
            bw_mu = [0 for r in range(self.R+1)]
            bw_sigma.extend([0 for r in range(self.R)])
            bw_sigma = np.vstack((
                bw_sigma,
                np.column_stack(
                    ([0 for r in range(self.R)],
                     np.diag([x*y for x,y in zip(eta_alpha, eta_beta)]))
                    )
                ))

            for i in range(len(X_list)):
                bw_sigma += np.vstack(
                    (np.column_stack(
                        (data_count[i],
                         np.sum(H_list[i]['mu'], axis=1).transpose())
                        ),
                     np.column_stack(
                         (np.sum(H_list[i]['mu'], axis=1),
                          np.dot(H_list[i]['mu'], H_list[i]['mu'].transpose())
                          + data_count[i] * H_list[i]['sigma'])
                        )
                    ))
                bw_mu += np.dot(
                    np.vstack((np.ones((1,data_count[i])),
                              H_list[i]['mu'])),
                    f_list[i]['mu']
                    )

            bw_sigma = np.matrix(np.linalg.inv(bw_sigma))
            bw_mu = np.dot(bw_sigma, bw_mu.transpose())

            # updates posterior distributions of predicted outputs
            f_dist = 0
            for i in range(len(X_list)):
                f_out = np.dot(
                    np.vstack(([1 for r in range(data_count[i])],
                               H_list[i]['mu'])).transpose(),
                    bw_mu
                    ).transpose()
                alpha_norm = (lu_list[i]['lower'] - f_out).tolist()[0]
                beta_norm = (lu_list[i]['upper'] - f_out).tolist()[0]
                norm_factor = [stats.norm.cdf(b) - stats.norm.cdf(a) if a != b
                               else 1
                               for a,b in zip(alpha_norm, beta_norm)]
                old_fpost = {'mu':f_list[i]['mu'],
                             'sigma':f_list[i]['sigma']}
                f_list[i]['mu'] = [
                    f + ((stats.norm.pdf(a) - stats.norm.pdf(b)) / n)
                    for a,b,n,f in
                    zip(alpha_norm, beta_norm,
                        norm_factor, f_out.tolist()[0])
                    ]
                f_list[i]['sigma'] = [
                    1 + (a * stats.norm.pdf(a) - b * stats.norm.pdf(b)) / n
                    - ((stats.norm.pdf(a) - stats.norm.pdf(b)) ** 2) / n**2
                    for a,b,n in zip(alpha_norm, beta_norm, norm_factor)
                    ]
                f_dist += np.mean(
                    [bhatta_dist(d1, d2)
                     for d2,d1 in zip(
                         [{'mu':mu2, 'sigma':sigma2} for mu2,sigma2
                          in zip(f_list[i]['mu'], f_list[i]['sigma'])],
                         [{'mu':mu1, 'sigma':sigma1} for mu1,sigma1
                          in zip(old_fpost['mu'], old_fpost['sigma'])]
                        )
                    ]) / float(len(X_list))

            bw_mu = bw_mu.transpose().tolist()[0]
            cur_iter += 1

        self.X = X_list
        self.lambdas = lambdas
        self.A = A_list
        self.gamma = {'a':gamma_alpha, 'b':gamma_beta}
        self.eta = {'a':eta_alpha, 'b':eta_beta}
        self.bw = {'sigma':bw_sigma, 'mu':bw_mu}
        self.f_list = f_list

    def predict_proba(self, X_list):
        data_count = [x.shape[0] for x in X_list]
        if self.kernel == 'rbf':
            X_gamma = [np.mean(metrics.pairwise.pairwise_distances(x))
                       for x in X_list]
            kernel_list = [metrics.pairwise.rbf_kernel(x,y,gamma=gam**-2.0)
                           for x,y,gam in zip(self.X, X_list, X_gamma)]
        elif self.kernel == 'linear':
            pass

        h_mu = [np.dot(a['mu'].transpose(), k)
                for a,k in zip(self.A, kernel_list)]
        f_mu = [np.dot(np.vstack(([1 for i in range(n)], h)).transpose(),
                       self.bw['mu'])
                for n,h in zip(data_count, h_mu)]
        f_sigma = [1.0 + np.diag(
            np.dot(
                np.dot(np.vstack(([1 for i in range(n)], h)).transpose(),
                       self.bw['sigma']),
                np.vstack(([1 for i in range(n)], h))
                ))
            for n,h in zip(data_count, h_mu)]

        pred_p = [(1-stats.norm.cdf((self.margin-mu) / sigma),
                   stats.norm.cdf((-self.margin-mu) / sigma))
                  for mu,sigma in zip(f_mu,f_sigma)]
        pred = [(p/(p+n))[0].tolist() for p,n in pred_p]
        return pred

    def score(self, X_list, y_list):
        pred_y = self.predict_proba(X_list)
        auc_scores = [metrics.roc_auc_score(x,y)
                      for x,y in zip(y_list, pred_y)]
        return np.mean(auc_scores)

    def get_params(self, deep=True):
        return {'sigma_h':self.sigma_h}

    def set_params(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)



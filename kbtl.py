
"""
KBTL (Kernelized Bayesian Transfer Learning)
Sharing knowledge between related but distinct tasks to improve
classification using an efficient variational approximation techniques.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

import numpy as np
from random import gauss as rnorm
from sklearn import metrics
from scipy.stats import norm


class KBTL(object):
    """Kernel based transfer learning classifier.
    """

    def __init__(self,
                 kernel='rbf', R=20, sigma_h=0.1,
                 lambda_par={'a':1.0,'b':1.0}, gamma_par={'a':1.0,'b':1.0},
                 eta_par={'a':1.0,'b':1.0}, margin=1.0,
                 max_iter=50, stopping_tol=1e-3):
        self.R = R
        self.sigma_h = sigma_h
        self.lambda_par = lambda_par
        self.gamma_par = gamma_par
        self.eta_par = eta_par
        self.margin = margin
        self.max_iter = max_iter

    def fit(self, X_list, y_list, verbose=False):
        # calculates kernel matrices
        data_count = [x.shape[0] for x in X_list]
        kernel_list = [metrics.pairwise.rbf_kernel(x,gamma=10**-5)
                       for x in X_list]
        y_list = [[1.0 if x else -1.0 for x in y] for y in y_list]

        # initializes matrix of priors for task-specific projection matrices
        lambdas = [{'alpha': np.matrix([[self.lambda_par['a'] + 0.5
                                         for i in xrange(self.R)]
                                        for j in xrange(d_count)]),
                    'beta': np.matrix([[self.lambda_par['b']
                                        for i in xrange(self.R)]
                                       for j in xrange(d_count)])}
                   for d_count in data_count]

        # initializes task-specific projection matrices
        A_list = [{'mu': np.matrix([[rnorm(0,1)
                                     for i in xrange(self.R)]
                                    for j in xrange(d_count)]),
                   'sigma': np.array([np.diag([1.0 for i in xrange(d_count)])
                                      for r in xrange(self.R)])}
                  for d_count in data_count]

        # initializes task-specific representations in shared sub-space
        H_list = [{'mu': np.matrix([[rnorm(0,1)
                                     for i in xrange(d_count)]
                                    for j in xrange(self.R)]),
                   'sigma': np.matrix(
                       np.diag([1.0 for i in xrange(self.R)]))}
                  for d_count in data_count]

        # initializes hyper-parameters
        gamma_alpha = self.gamma_par['a'] + 0.5
        gamma_beta = self.gamma_par['b']
        eta_alpha = [self.eta_par['a'] + 0.5 for i in xrange(self.R)]
        eta_beta = [self.eta_par['b'] for i in xrange(self.R)]
        bw_mu = [0]
        bw_mu.extend([rnorm(0,1) for i in xrange(self.R)])
        bw_sigma = np.matrix(np.diag([1 for i in xrange(self.R+1)]))

        # initializes predicted outputs
        f_list = [{'mu': [(abs(rnorm(0,1)) + self.margin)
                          * np.sign(lbl[i]) for i in xrange(d_count)],
                   'sigma': [1 for i in xrange(d_count)]}
                  for d_count,lbl in zip(data_count, y_list)]

        # precomputes kernel crossproducts, initializes lower-upper matrix
        kkt_list = [np.dot(x,x.transpose()) for x in kernel_list]
        lu_list = [{'lower': [-1e40 if y_list[j][i] <= 0 else self.margin
                              for i in xrange(d_count)],
                    'upper': [1e40 if y_list[j][i] >= 0 else -self.margin
                              for i in xrange(d_count)]}
                   for j,d_count in enumerate(data_count)]

        # does inference using variational Bayes for the given number
        # of iterations
        cur_iter = 1
        while(cur_iter <= self.max_iter):
            if verbose and (cur_iter % 10) == 0:
                print "On iteration " + str(cur_iter) + "...."
                print "gamma_beta: " + str(round(gamma_beta, 4))
            for i in xrange(len(X_list)):

                # updates posterior distributions of projection priors
                for j in xrange(self.R):
                    lambdas[i]['beta'][:,j] = np.power(
                        (self.lambda_par['b'] ** -1
                         + 0.5 * (
                             np.power(A_list[i]['mu'][:,j].transpose(), 2)
                             + np.diag(A_list[i]['sigma'][j,:,:])
                         )), -1).transpose()

                # updates posterior distributions of projection matrices
                for j in xrange(self.R):
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
                    np.diag([self.sigma_h ** -2 for r in xrange(self.R)])
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
                        for c in xrange(data_count[i])]).transpose()
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
            bw_mu = [0 for r in xrange(self.R+1)]
            bw_sigma.extend([0 for r in xrange(self.R)])
            bw_sigma = np.vstack((
                bw_sigma,
                np.column_stack(
                    ([0 for r in xrange(self.R)],
                     np.diag([x*y for x,y in zip(eta_alpha, eta_beta)]))
                    )
                ))
            for i in xrange(len(X_list)):
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
            for i in xrange(len(X_list)):
                f_out = np.dot(
                    np.vstack(([1 for r in xrange(data_count[i])],
                               H_list[i]['mu'])).transpose(),
                    bw_mu
                    ).transpose()
                alpha_norm = (lu_list[i]['lower'] - f_out).tolist()[0]
                beta_norm = (lu_list[i]['upper'] - f_out).tolist()[0]
                norm_factor = [norm.cdf(b) - norm.cdf(a) if a != b
                               else 1
                               for a,b in zip(alpha_norm, beta_norm)]
                f_list[i]['mu'] = [
                    f + ((norm.pdf(a) - norm.pdf(b)) / n)
                    for a,b,n,f in
                    zip(alpha_norm, beta_norm,
                        norm_factor, f_out.tolist()[0])
                    ]
                f_list[i]['sigma'] = [
                    1 + (a * norm.pdf(a) - b * norm.pdf(b)) / n
                    - ((norm.pdf(a) - norm.pdf(b)) ** 2) / n**2
                    for a,b,n in zip(alpha_norm, beta_norm, norm_factor)
                    ]

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
        kernel_list = [metrics.pairwise.rbf_kernel(x,y,gamma=10**-5)
                       for x,y in zip(self.X, X_list)]

        h_mu = [np.dot(a['mu'].transpose(), k)
                for a,k in zip(self.A, kernel_list)]
        f_mu = [np.dot(np.vstack(([1 for i in xrange(n)], h)).transpose(),
                       self.bw['mu'])
                for n,h in zip(data_count, h_mu)]
        f_sigma = [1.0 + np.diag(
            np.dot(
                np.dot(np.vstack(([1 for i in xrange(n)], h)).transpose(),
                       self.bw['sigma']),
                np.vstack(([1 for i in xrange(n)], h))
                ))
            for n,h in zip(data_count, h_mu)]

        pred_p = [(1-norm.cdf((self.margin-mu) / sigma),
                   norm.cdf((-self.margin-mu) / sigma))
                  for mu,sigma in zip(f_mu,f_sigma)]
        pred = [(p/(p+n))[0].tolist() for p,n in pred_p]
        return pred

    def get_params(self):
        return {'sigma_h':self.sigma_h}

    def set_params(self, **kwargs):
        for k,v in kwargs.iteritems():
            setattr(self, k, v)


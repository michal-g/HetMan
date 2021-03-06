
"""
HetMan (Heterogeneity Manifold)
Prediction of mutation sub-types using expression data.
This file contains the algorithms used to predict continuous mutation states.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

from math import exp
from scipy import stats

from .pipelines import RegrPipe
from .selection import PathwaySelect

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet as ENet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


class ElasticNet(RegrPipe):
    """A class corresponding to elastic net regression
       of gene gain/loss status.
    """

    tune_priors = (
        ('fit__alpha', stats.lognorm(scale=exp(1), s=exp(1))),
        ('fit__l1_ratio', (0.05,0.25,0.5,0.75,0.95))
        )

    def __init__(self, path_keys=None):
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = StandardScaler()
        fit_step = ENet(normalize=False, max_iter=5000)
        RegrPipe.__init__(self,
            [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
            path_keys)


class SVRrbf(RegrPipe):
    """A class corresponding to Support Vector Machine regression
       of gene gain/loss status using a radial basis kernel.
    """

    tune_priors = (
            ('fit__C', stats.lognorm(scale=exp(-1), s=exp(1))),
            ('fit__gamma', stats.lognorm(scale=1e-5, s=exp(2)))
        )

    def __init__(self, path_keys=None):
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = StandardScaler()
        fit_step = SVR(kernel='rbf', cache_size=500)
        RegrPipe.__init__(self,
            [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
            path_keys)


class rForest(RegrPipe):
    """A class corresponding to Random Forest regression
       of gene gain/loss status.
    """

    tune_priors = (
        ('fit__min_samples_leaf', (0.001,0.005,0.01,0.05)),
        ('fit__max_features', (5, 'sqrt', 'log2', 0.02))
        )

    def __init__(self, path_keys=None):
        feat_step = PathwaySelect(path_keys=path_keys)
        norm_step = StandardScaler()
        fit_step = RandomForestRegressor(n_estimators=1000)
        RegrPipe.__init__(self,
            [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
            path_keys)



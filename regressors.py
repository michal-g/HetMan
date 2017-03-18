
"""
HetMan (Heterogeneity Manifold)
Prediction of mutation sub-types using expression data.
This file contains the algorithms used to predict continuous mutation states.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

from math import exp
from scipy import stats

from pipelines import RegrPipe
from selection import PathwaySelect

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet as ENet
from sklearn.svm import SVR


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
        fit_step = ENet(normalize=False)
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
        UniPipe.__init__(self,
            [('feat', feat_step), ('norm', norm_step), ('fit', fit_step)],
            path_keys)




"""
Hetman (Heterogeneity Manifold)
Prediction of mutation sub-types using expression data.
This file contains the methods used to select genetic features for use
in downstream prediction algorithms.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

from data import get_pc2_neighb
from itertools import chain
import re
import dill as pickle
from functools import reduce

from math import log
import numpy as np
from scipy import stats
import dirichlet

from sklearn.feature_selection.base import SelectorMixin
from sklearn.feature_selection import GenericUnivariateSelect


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

    def __init__(self, path_keys=None):
        self.path_keys = path_keys
        super(PathwaySelect, self).__init__()

    def fit(self, X, y, **fit_params):
        mut_genes = fit_params['mut_genes']
        if self.path_keys is None:
            self.select_genes = set(X.columns)
        else:
            path_obj = fit_params['path_obj']
            select_genes = set()
            for gene in mut_genes:
                for pdirs, ptypes in self.path_keys:
                    if len(pdirs) == 0:
                        select_genes |= set(chain(*chain(
                            *[[g for t,g in v.items() if t in ptypes]
                            for v in path_obj[gene].values()]
                            )))
                    elif len(ptypes) == 0:
                        select_genes |= set(chain(*chain(
                            *[v.values() for k,v in path_obj[gene].items()
                            if k in pdirs]
                            )))
                    else:
                        select_genes |= set(chain(*chain(
                            *[[g for t,g in v.items() if t in ptypes]
                            for k,v in path_obj[gene].items() if k in pdirs]
                            )))
            self.select_genes = select_genes

        self.select_genes -= set(mut_genes)
        self.expr_genes = X.columns
        return self

    def _get_support_mask(self):
        return np.array([g in self.select_genes for g in self.expr_genes])
    
    def set_params(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

    def get_params(self, deep=True):
        return {'path_keys': self.path_keys}



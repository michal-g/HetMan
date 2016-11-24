
import hetman_classif as hclassif
import numpy as np
from sklearn import preprocessing, model_selection, pipeline, base
from sklearn.metrics import roc_auc_score
from sklearn.externals.joblib import Parallel, delayed
from kbtl import KBTL
from scipy import stats
from math import exp, log
from itertools import product
import dill
from pathos.multiprocessing import ProcessingPool


class LossOptimError(Exception):
    """Class for exceptions thrown by the Hetman Loss Optimizer module."""
    pass


class LossOptimizer(object):

    def __init__(self, cdata_list):
        if not all([hasattr(x, 'cnv_') for x in cdata_list]):
            raise LossOptimError("All given MutExpr objects must include "
                                 "CNV status.")
        if not all([len(x.cnv_) == 1 for x in cdata_list]):
            raise LossOptimError("All given MutExpr objects must have "
                                 "exactly one mutated gene.")
        mut_gene = set([x.cnv_.keys()[0] for x in cdata_list])
        if len(mut_gene) > 1:
            raise LossOptimError("All given MutExpr objects must have "
                                 "the same mutated gene.")
        else:
            mut_gene = list(mut_gene)[0]
        
        mut_chr = [v['chr'] for k,v in cdata_list[0].annot.items()
                   if v['gene_name'] == mut_gene][0]
        self.use_genes = [[v['gene_name'] for v in x.annot.values()
                           if v['chr'] != mut_chr]
                          for x in cdata_list]

        for i in xrange(len(cdata_list)):
            cdata_list[i].add_cnv_loss()
        self.data = cdata_list
        #self.uniclx = [hclassif.Lasso([mut_gene], x) for x in self.use_genes]
        self.multiclx = MultiKBTL(self.use_genes)
        self.history = []

    def __str__(self):
        pass

    def test_multiclx(self,
                      mset, test_indx=range(16), tune_indx=(50,75),
                      verbose=True):
        train_list = [x.training(mset=mset) for x in self.data]
        test_cvs = [[x for i,x in enumerate(tr_l[2])
                     if i in test_indx] for tr_l in train_list]
        if tune_indx is not None:
            tune_cvs = [[x for i,x in enumerate(tr_l[2])
                        if i in tune_indx] for tr_l in train_list]
            self.multiclx.tune(
                expr_list=[x[0] for x in train_list],
                mut_list=[x[1] for x in train_list],
                cv_samples=tune_cvs, verbose=verbose
                )

        return np.percentile(model_selection.cross_val_score(
            estimator=self.multiclx,
            X=[x[0] for x in train_list], y=[x[1] for x in train_list],
            scoring=_score_auc, cv=test_cvs, n_jobs=-1
            ), 25)

    def step(self):
        pass


class MultiKBTL(object):
    """A class corresponding to expression-based"""
    """classifiers of mutation status."""

    def __init__(self, expr_genes):
        self._tune_params = {'sigma_h': exp(-2)}
        self._expr_genes = expr_genes
        self._param_priors = {'sigma_h':stats.lognorm(scale=exp(-2), s=1)}
        self.scalers = [preprocessing.StandardScaler()
                        for i in xrange(len(expr_genes))]
        self.fitter = KBTL(sigma_h=self._tune_params['sigma_h'], max_iter=25)

    def __str__(self):
        print self._tune_params

    def tune(self, expr_list, mut_list, cv_samples, verbose=False):
        new_grid = {param:distr.rvs(16)
                    for param,distr in self._param_priors.items()}
        grid_test = self.grid_search(
            expr_list, mut_list, new_grid, cv_samples)
        for param in self._tune_params.keys():
            new_mean,new_sd = self._update_params(
                [(dict(x)[param],y) for x,y in grid_test.items()])
            self._param_priors[param] = stats.lognorm(
                scale=new_mean, s=new_sd)
        if verbose:
            print self

    def grid_search(self, expr_list, mut_list, new_grid, cv_samples):
        param_tbl = [{k:v for k,v in zip(new_grid.keys(), params)}
                     for params in product(*new_grid.values())]
        pool = ProcessingPool(nodes=16)
        test_list = pool.map(lambda p: self.test_param(
            expr_list, mut_list, cv_samples, p),
            param_tbl)
        return {tuple(k.items()):v for k,v in zip(param_tbl, test_list)}

    def test_param(self, expr_list, mut_list, cv_samples, param_list):
        out_list = []
        for i in range(len(cv_samples[0])):
            train_expr = [e.ix[c[i][0],:]
                          for e,c in zip(expr_list, cv_samples)]
            train_mut = [np.array(m)[c[i][0]]
                         for m,c in zip(mut_list, cv_samples)]
            test_expr = [e.ix[c[i][1],:]
                         for e,c in zip(expr_list, cv_samples)]
            test_mut = [np.array(m)[c[i][1]]
                        for m,c in zip(mut_list, cv_samples)]
            self.fit(train_expr, train_mut, param_list)
            out_list += [self.score(test_expr, test_mut)]
        return np.mean(out_list)

    def fit(self, expr_list, mut_list, params=None, verbose=True):
        norm_expr = [[] for i in xrange(len(expr_list))]
        for i in xrange(len(expr_list)):
            norm_expr[i] = self.scalers[i].fit_transform(
                expr_list[i].loc[:,self._expr_genes[i]])
        if params is not None:
            self.fitter.set_params(**params)
        self.fitter.fit(norm_expr, mut_list, verbose=verbose)

    def predict(self, expr_list):
        new_expr = [sc.transform(x.loc[:,g]) for x,g,sc
                    in zip(expr_list, self._expr_genes, self.scalers)]
        return self.fitter.predict_proba(new_expr)

    def transform(self, expr_list):
        trans_X = [[] for i in xrange(len(expr_list))]
        for i in xrange(len(expr_list)):
            trans_X[i] = self.scalers[i].transform(
                expr_list[i].loc[:,self.expr_genes[i]])
        return trans_X

    def fit_transform(self, expr_list, mut_list):
        trans_X = [[] for i in xrange(len(expr_list))]
        for i in xrange(len(expr_list)):
            trans_X[i] = self.scalers[i].fit_transform(
                expr_list[i].loc[:,self.expr_genes[i]])
        self.fitter.fit(trans_X, mut_list)
        return trans_X

    def score(self, expr_list, mut_list):
        """Computes the AUC score for a mutation classifier on a given
           expression matrix and a mutation state vector.
        
        Parameters
        ----------
        estimator : UniClassifier
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
        mut_scores = self.predict(expr_list)
        return np.mean([roc_auc_score(m,s) 
                        for m,s in zip(mut_list, mut_scores)])
    
    def _update_params(self, param_list):
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


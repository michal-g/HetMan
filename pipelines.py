
"""
HetMan (Heterogeneity Manifold)
Prediction of mutation sub-types using expression data.
This file contains classes used to organize feature selection, normalization,
and prediction methods into robust pipelines.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

from abc import abstractmethod

from math import log, exp
from functools import reduce
from operator import mul
import dill as pickle

from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score


class HetManPipeError(Exception):
    pass


class MutPipe(Pipeline):
    """A class corresponding to pipelines for predicting gene mutations
       using expression data.
    """

    # the parameters that are to be tuned, with either statistical
    # distributions or iterables to be sampled from as values
    tune_priors = ()

    def __init__(self, steps, path_keys):
        super(MutPipe, self).__init__(steps)
        self.set_params(path_keys=path_keys)
        self.cur_tuning = dict(self.tune_priors)

    def __str__(self):
        """Prints the tuned parameters of the pipeline."""
        param_str = type(self).__name__ + ' with '

        if self.tune_priors:
            param_list = self.get_params()
            param_str += reduce(
                lambda x,y: x + ', ' + y,
                [k + ': ' + '%s' % float('%.4g' % param_list[k])
                for k in list(self.cur_tuning.keys())]
                )
        else:
            param_str += 'no tuned parameters.'

        return param_str

    def predict_mut(self, expr):
        """Predict mutation status using the pipeline."""
        return self.predict(expr)

    @abstractmethod
    def score_mut(cls, estimator, expr, mut):
        """Score the accuracy of the pipeline in predicting the state
           of a given set of mutations. Used to ensure compatibility with
           scoring methods implemented in sklearn.
        """

    def tune(self, expr, mut, mut_genes, path_obj, cv_samples,
             test_count=16, update_priors=False, verbose=False):
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
                fit_params={'feat__mut_genes': mut_genes,
                            'feat__path_obj': path_obj},
                n_iter=test_count, scoring=self.score_mut,
                cv=cv_samples, n_jobs=-1, refit=False
                )
            grid_test.fit(expr, mut)

            # finds the best parameter combination and updates the classifier
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
        for param in self.tune_priors.keys():
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


class ClassPipe(MutPipe):
    """A class corresponding to pipelines for predicting
       discrete gene mutation states.
    """

    def __init__(self, steps, path_keys):
        if not hasattr(steps[-1][-1], 'predict_proba'):
            raise HetManPipeError(
                "Classifier pipelines must have a classification estimator"
                "with a 'predict_proba' method as their final step!")
        super(ClassPipe, self).__init__(steps, path_keys)

    def predict_mut(self, expr):
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
    def score_mut(cls, estimator, expr, mut):
        """Computes the AUC score using the classifier on a expr-mut pair.

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
        mut_scores = estimator.predict_mut(expr)
        return roc_auc_score(mut, mut_scores)


class RegrPipe(MutPipe):
    """A class corresponding to pipelines for predicting
       continuous gene mutation states.
    """

    def __init__(self, steps, path_keys):
        if not issubclass(type(steps[-1][-1]), RegressorMixin):
            raise HetManPipeError(
                "Regressor pipelines must have a regressor estimator"
                "inherting from the sklearn 'RegressorMixin' class as"
                "their final step!")
        super(RegrPipe, self).__init__(steps, path_keys)

    @classmethod
    def score_mut(cls, estimator, expr, mut):
        """Computes the R^2 score using the regressor on a expr-mut pair.

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
        return estimator.score(expr, mut)


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



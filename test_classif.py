
"""HetMan (Heterogeneity Manifold)
Classification of mutation sub-types using expression data.
This file contains unit tests for classifiers and regressors.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

from .cohorts import Cohort
from .kbtl import KBTL, MultiClf
from .mutation import MuType

import pytest
import numpy as np
import pandas as pd
import pickle
import sys
import synapseclient
from sklearn.datasets import make_classification

from itertools import combinations as combn
from itertools import chain, product


# .. objects that are re-used across many different tests ..
@pytest.fixture(scope='function')
def type_tester(request):
    """Create a set of mutation subtypes."""
    return TypeTester(request.param)

class TypeTester(object):
    # mutation types for each mutation collection
    mtypes = {
        'small': (
            MuType({('Gene', 'TTN'): None}),
            MuType({('Gene', 'TTN'):
                    {('Form', 'Missense_Mutation'):
                     {('Exon', ('326/363','10/363')): None}}}),
            MuType({('Gene', 'CDH1'): None,
                    ('Gene', 'TTN'):
                    {('Form', 'Missense_Mutation'):
                     {('Exon', ('302/363','10/363')): None}}}),
            MuType({('Form', 'Silent'): None}),
            MuType({('Gene', ('CDH1','TTN')): None})
            ),
        }

    def __init__(self, request):
        self.type_lbl = request

    def get_types(self):
        if self.type_lbl in self.mtypes:
            return self.mtypes[self.type_lbl]
        elif self.type_lbl == '_all':
            return reduce(lambda x,y: x+y, self.mtypes.values())
        else:
            raise KeyError("Unknown MuType test set!")

@pytest.fixture(scope='class')
def task_tester(request):
    """Create expression values and mutation labels from multiple tasks."""
    return TaskTester(request.param)

class TaskTester(object):

    def __init__(self, request):
        self.expr_type = request

    def get_tasks(self):
        if self.expr_type == 'separable':
            tasks = [
                make_classification(
                    n_samples=40, n_features=8, n_informative=8,
                    n_redundant=0, shuffle=False, class_sep=2.0,
                    n_clusters_per_class=1, random_state=3),
                make_classification(
                    n_samples=30, n_features=6, n_informative=6,
                    n_redundant=0, shuffle=False, class_sep=2.0,
                    n_clusters_per_class=1, random_state=7)
                ]

        return tasks

@pytest.fixture(scope='module')
def cdata_small():
    syn = synapseclient.Synapse()
    syn.login('grzadkow', 'W0w6g1i8A')
    return Cohort(syn, cohort='COAD', mut_genes=['BRAF','TP53','PIK3CA'],
                  mut_levels=('Gene', 'Form', 'Protein'))


class TestCaseKBTL:
    """Tests Kernelized Bayesian Transfer Learning classifier."""

    @pytest.mark.parametrize('task_tester', ['separable'],
                             indirect=True, scope="function")
    def test_basic(self, task_tester):
        """Can we correctly classify separable tasks?"""
        tasks = task_tester.get_tasks()
        X_list = [tsk[0] for tsk in tasks]
        y_list = [tsk[1] for tsk in tasks]
        kbtl_obj = KBTL(R=3)

        kbtl_obj.fit(X_list, y_list)
        pred_y = kbtl_obj.predict_proba(X_list)

        assert ([len(x['mu']) for x in kbtl_obj.f_list]
                == [x.shape[0] for x in X_list])
        for y, pred in zip(y_list, pred_y):
            assert (np.mean(pd.Series(pred)[y == 0])
                    < np.mean(pd.Series(pred)[y == 1]))


class TestCaseCohort:
    """Tests cohort functionality."""

    def test_clf(self, cdata_small):
        """Can we use a classifier on a Cohort?"""
        mtype = MuType({('Gene', 'BRAF'):{('Protein', 'p.V600E'): None}})
        clf = classifiers.Lasso()
        old_C = clf.named_steps['fit'].C

        cdata_small.tune_clf(clf, mtype=mtype)
        assert clf.named_steps['fit'].C != old_C
        cdata_small.score_clf(clf, mtype=mtype)
        cdata_small.fit_clf(clf, mtype=mtype)
        cdata_small.predict_clf(clf, use_test=False)
        cdata_small.eval_clf(clf, mtype=mtype)



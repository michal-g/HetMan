
import pytest
import numpy as np
import pandas as pd
import pickle
import sys
import synapseclient

sys.path += ['/home/users/grzadkow/compbio/scripts/HetMan']
from mutation import *
from cohorts import Cohort
import classifiers

from itertools import combinations as combn
from itertools import chain


@pytest.fixture(scope='module')
def muts_tester(request):
    """Create a mutation table."""
    return MutsTester(request.param)

def muts_id(val):
    """Get the ID of a file storing test mutation data."""
    return val[0].split('muts_')[-1].split('.p')[0]

class MutsTester(object):
    def __init__(self, arg):
        self.muts_file = arg[0]

    def get_muts_mtree(self, levels=None):
        with open(self.muts_file, 'rb') as fl:
            muts = pickle.load(fl)
        if levels is None:
            levels = tuple(muts.columns[:-1])
        mtree = MuTree(muts, levels=levels)
        return muts, mtree

@pytest.fixture(scope='module')
def type_tester(request):
    """Create a set of mutation subtypes."""
    return TypeTester(request.param)

class TypeTester(object):
    # mutation types for each mutation collection
    mtypes = {'small': (
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
        )}

    def __init__(self, arg):
        self.type_lbl = arg[0]

    def get_types(self):
        return self.mtypes[self.type_lbl]


@pytest.fixture(scope='module')
def cdata_small():
    syn = synapseclient.Synapse()
    syn.login('grzadkow', 'W0w6g1i8A')
    return Cohort(syn, cohort='COAD', mut_genes=['BRAF','TP53','PIK3CA'],
                  mut_levels=('Gene', 'Form', 'Protein'))


@pytest.mark.parametrize('muts_tester', [['test/muts_small.p']],
                         ids=muts_id, indirect=True, scope="class")
class TestCaseBasicMuTree:
    """Tests for basic functionality of MuTrees."""

    def test_keys(self, muts_tester):
        """Does the tree correctly implement key retrieval of subtrees?"""
        muts, mtree = muts_tester.get_muts_mtree()

        if len(mtree.levels) > 1:
            for vals, _ in muts.groupby(mtree.levels):
                assert mtree.child[vals[0]][vals[1:]] == mtree[vals]
                assert mtree[vals[:-1]].child[vals[-1]] == mtree[vals]
        else:
            for val in set(muts[mtree.levels[0]]):
                assert mtree.child[val] == mtree[val]

    def test_structure(self, muts_tester):
        """Is the internal structure of the tree correct?"""
        muts, mtree = muts_tester.get_muts_mtree()
        assert set(mtree.child.keys()) == set(muts[mtree.levels[0]])
        assert mtree.depth == 0

        lvl_sets = {i:mtree.levels[:i] for i in range(1, len(mtree.levels))}
        for i, lvl_set in lvl_sets.items():
            for vals, mut in muts.groupby(lvl_set):
                assert mtree[vals].depth == i
                assert (set(mtree[vals].child.keys())
                        == set(mut[mtree.levels[i]]))

    def test_print(self, muts_tester):
        """Can we print the tree?"""
        muts, mtree = muts_tester.get_muts_mtree()

        print(mtree)
        lvl_sets = [mtree.levels[:i] for i in range(1, len(mtree.levels))]
        for lvl_set in lvl_sets:
            for vals, _ in muts.groupby(lvl_set):
                print(mtree[vals])

    def test_iteration(self, muts_tester):
        """Does the tree correctly implement iteration over subtrees?"""
        muts, mtree = muts_tester.get_muts_mtree()

        for nm, mut in mtree:
            assert nm in set(muts[mtree.levels[0]])
            assert mut == mtree[nm]
            assert mut != mtree

        lvl_sets = {i:mtree.levels[:i] for i in range(1, len(mtree.levels))}
        for i, lvl_set in lvl_sets.items():
            for vals, _ in muts.groupby(lvl_set):
                if isinstance(vals, str):
                    vals = (vals, )
                for nm, mut in mtree[vals]:
                    assert nm in set(muts[mtree.levels[i]])
                    assert mut == mtree[vals][nm]
                    assert mut != mtree[vals[:-1]]

    def test_samples(self, muts_tester):
        """Does the tree properly store its samples?"""
        muts, mtree = muts_tester.get_muts_mtree()

        for vals, mut in muts.groupby(mtree.levels):
            assert set(mtree[vals]) == set(mut['Sample'])

    def test_get_samples(self, muts_tester):
        """Can we successfully retrieve the samples from the tree?"""
        muts, mtree = muts_tester.get_muts_mtree()

        lvl_sets = [mtree.levels[:i] for i in range(1, len(mtree.levels))]
        for lvl_set in lvl_sets:
            for vals, mut in muts.groupby(lvl_set):
                assert set(mtree[vals].get_samples()) == set(mut['Sample'])

    def test_allkeys(self, muts_tester):
        """Can we retrieve the mutation set key of the tree?"""
        muts, mtree = muts_tester.get_muts_mtree()

        assert mtree.allkey() == mtree.allkey(mtree.levels)
        lvl_sets = chain.from_iterable(
            combn(mtree.levels, r)
            for r in range(1, len(mtree.levels)+1))
        for lvl_set in lvl_sets:
            lvl_key = {}
            for vals, _ in muts.groupby(lvl_set):
                cur_key = lvl_key
                if isinstance(vals, str):
                    vals = (vals,)
                for i in range(len(lvl_set) - 1):
                    if (lvl_set[i], vals[i]) not in cur_key:
                        cur_key.update({(lvl_set[i], vals[i]):{}})
                    cur_key = cur_key[(lvl_set[i], vals[i])]
                cur_key.update({(lvl_set[-1], vals[-1]):None})

            assert mtree.allkey(lvl_set) == lvl_key


@pytest.mark.parametrize('type_tester', [['small']],
                         indirect=True, scope="class")
class TestCaseBasicMuType:
    """Tests for basic functionality of MuTypes."""

    def test_print(self, type_tester):
        """Can we print MuTypes?"""
        mtypes = type_tester.get_types()

        for mtype in mtypes:
            print(mtype)

    def test_hash(self, type_tester):
        """Can we get proper hash values of MuTypes?"""
        mtypes = type_tester.get_types()

        for mtype1, mtype2 in combn(mtypes, 2):
            assert (mtype1 == mtype2) == (hash(mtype1) == hash(mtype2))

    def test_rawkey(self, type_tester):
        """Can we get the raw key of MuTypes?"""
        mtypes = type_tester.get_types()

        for mtype in mtypes:
            mtype.raw_key()

    def test_samps(self, type_tester):
        """Can we retrieve the right samples using a MuType?"""
        assert (mtypes_small[4].get_samples(mtree_small)
                == set(muts_small['Sample']))
        assert (mtypes_small[0].get_samples(mtree_small)
                == set(muts_small['Sample'][muts_small['Gene'] == 'TTN']))
        assert (mtypes_small[3].get_samples(mtree_small)
                == set(muts_small['Sample'][muts_small['Form'] == 'Silent']))
        samp_indx = ((muts_small['Gene'] == 'TTN')
                     & (muts_small['Form'] == 'Missense_Mutation')
                     & (muts_small['Exon'].isin(['326/363','10/363'])))
        assert (mtypes_small[1].get_samples(mtree_small)
                == set(muts_small['Sample'][samp_indx]))
        samp_indx = ((muts_small['Gene'] == 'CDH1')
                     | ((muts_small['Gene'] == 'TTN')
                        & (muts_small['Form'] == 'Missense_Mutation')
                        & (muts_small['Exon'].isin(['302/363','10/363']))))
        assert (mtypes_small[2].get_samples(mtree_small)
                == set(muts_small['Sample'][samp_indx]))



class TestCaseMuTypeBinary:
    """Tests the binary operators defined for MuTypes."""

    def test_eq(self, mtypes_small):
        """Can we evaluate the equality of two MuTypes?"""
        for mtype in mtypes_small:
            assert mtype == mtype
        for pair in combn(mtypes_small, 2):
            assert pair[0] != pair[1]

    def test_and(self, mtypes_small):
        """Can we take the intersection of two MuTypes?"""
        for mtype in mtypes_small:
            assert mtype == (mtype & mtype)
        assert (mtypes_small[0] & mtypes_small[1]) == mtypes_small[1]
        assert (mtypes_small[0] & mtypes_small[4]) == mtypes_small[0]
        assert ((mtypes_small[1] & mtypes_small[2])
                == MuType(
                    {('Gene', 'TTN'):
                     {('Form', 'Missense_Mutation'):
                      {('Exon', '10/363'): None}}})
                    )

    def test_or(self, mtypes_small):
        """Can we take the union of two MuTypes?"""
        for mtype in mtypes_small:
            assert mtype == (mtype | mtype)
        assert (mtypes_small[0] | mtypes_small[1]) == mtypes_small[0]
        assert (mtypes_small[0] | mtypes_small[4]) == mtypes_small[4]
        assert ((mtypes_small[1] | mtypes_small[2])
                == MuType(
                    {('Gene', 'CDH1'): None,
                     ('Gene', 'TTN'):
                     {('Form', 'Missense_Mutation'):
                      {('Exon', ('326/363', '302/363', '10/363')): None}}})
                    )


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



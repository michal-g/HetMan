


import pytest
import numpy as np
import pandas as pd
import pickle
import sys

sys.path += ['/home/users/grzadkow/compbio/scripts/HetMan']
import data
from itertools import combinations as combn


@pytest.fixture(scope='module')
def muts_small():
    with open('test/muts_small.p', 'rb') as fl:
        muts = pickle.load(fl)
    return muts

@pytest.fixture(scope='module')
def mtree_small(muts_small):
    return data.MuTree(muts_small, levels=('Gene','Form','Exon'))

@pytest.fixture(scope='module')
def mtypes_small():
    mtype1 = data.MuType({('Gene', 'TTN'): None})
    mtype2 = data.MuType(
            {('Gene', 'TTN'):
             {('Form', 'Missense_Mutation'):
              {('Exon', ('326/363','10/363')): None}}}
            )
    mtype3 = data.MuType(
            {('Gene', 'CDH1'): None,
             ('Gene', 'TTN'):
             {('Form', 'Missense_Mutation'):
              {('Exon', ('302/363','10/363')): None}}}
            )
    mtype4 = data.MuType({('Form', 'Silent'): None})
    mtype5 = data.MuType({('Gene', ('CDH1','TTN')): None})
    return [mtype1, mtype2, mtype3, mtype4, mtype5]


class TestCaseBasicMuTree:
    """Basic tests for Hetman MuTrees."""

    def test_structure(self, muts_small, mtree_small):
        """Is the internal structure of the tree correct?"""
        assert set(list(mtree_small.child.keys())) == set(muts_small['Gene'])
        assert (set(list(mtree_small.child['TTN'].child.keys()))
                == set(muts_small['Form'][muts_small['Gene'] == 'TTN']))

    def test_samps(self, muts_small, mtree_small):
        """Does the tree properly store its samples?"""
        assert set(muts_small['Sample']) == mtree_small.get_samples()
        assert (mtree_small.get_samp_count(muts_small['Sample']) ==
                {k:list(muts_small.drop_duplicates()['Sample']).count(k)
                 for k in muts_small['Sample']})

    def test_status(self, muts_small, mtree_small):
        """Does the tree give the right mutation status?"""
        samps1 = (["dummy" + str(i) for i in range(3)]
                  + list(muts_small['Sample'])
                  + ["dummy" + str(i) for i in range(11,16)])
        assert (mtree_small.status(samps1)
                == pd.Series(samps1).isin(muts_small['Sample'])).all()

    def test_print(self, mtree_small):
        """Can we print the tree?"""
        print(mtree_small)

    def test_keys(self, mtree_small):
        """Can we retrieve the keys of the tree?"""
        key1 = mtree_small.allkey(['Gene'])
        key2 = mtree_small.allkey(['Gene','Form'])
        key3 = mtree_small.allkey()
        key4 = mtree_small.allkey(['Gene','Exon'])
        assert key1 == {(data.MutLevel.Gene, 'CDH1'): None,
                        (data.MutLevel.Gene, 'TTN'): None}
        assert key2 == {(data.MutLevel.Gene, 'CDH1'):
                        {(data.MutLevel.Form, 'Silent'): None},
                        (data.MutLevel.Gene, 'TTN'):
                        {(data.MutLevel.Form, 'Missense_Mutation'): None,
                         (data.MutLevel.Form, 'Silent'): None}}
        assert key3 == {(data.MutLevel.Gene, 'CDH1'):
                        {(data.MutLevel.Form, 'Silent'):
                         {(data.MutLevel.Exon, '7/16'): None}},
                        (data.MutLevel.Gene, 'TTN'):
                        {(data.MutLevel.Form, 'Missense_Mutation'):
                         {(data.MutLevel.Exon, '10/363'): None,
                          (data.MutLevel.Exon, '133/363'): None,
                          (data.MutLevel.Exon, '232/363'): None,
                          (data.MutLevel.Exon, '256/363'): None,
                          (data.MutLevel.Exon, '280/363'): None,
                          (data.MutLevel.Exon, '284/363'): None,
                          (data.MutLevel.Exon, '302/363'): None,
                          (data.MutLevel.Exon, '326/363'): None,
                          (data.MutLevel.Exon, '46/363'): None},
                         (data.MutLevel.Form, 'Silent'):
                         {(data.MutLevel.Exon, '26/363'): None}}}
        assert key4 == {(data.MutLevel.Gene, 'CDH1'):
                        {(data.MutLevel.Exon, '7/16'): None},
                        (data.MutLevel.Gene, 'TTN'):
                        {(data.MutLevel.Exon, '10/363'): None,
                         (data.MutLevel.Exon, '133/363'): None,
                         (data.MutLevel.Exon, '232/363'): None,
                         (data.MutLevel.Exon, '256/363'): None,
                         (data.MutLevel.Exon, '26/363'): None,
                         (data.MutLevel.Exon, '280/363'): None,
                         (data.MutLevel.Exon, '284/363'): None,
                         (data.MutLevel.Exon, '302/363'): None,
                         (data.MutLevel.Exon, '326/363'): None,
                         (data.MutLevel.Exon, '46/363'): None}}


class TestCaseBasicMuType:
    """Basic tests for Hetman MuTypes."""

    def test_samps(self, muts_small, mtree_small, mtypes_small):
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

    def test_print(self, mtypes_small):
        """Can we print MuTypes?"""
        for mtype in mtypes_small:
            print(mtype)

    def test_hash(self, mtypes_small):
        """Can we get the hash values of MuTypes?"""
        hash_test = [hash(mtype) for mtype in mtypes_small]
        assert len(set(hash_test)) == len(mtypes_small)


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
                == data.MuType(
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
                == data.MuType(
                    {('Gene', 'CDH1'): None,
                     ('Gene', 'TTN'):
                     {('Form', 'Missense_Mutation'):
                      {('Exon', ('326/363', '302/363', '10/363')): None}}})
                    )



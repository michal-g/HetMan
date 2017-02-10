


import pytest
import numpy as np
import pickle
import sys

sys.path += ['/home/users/grzadkow/compbio/scripts/HetMan']
import data


@pytest.fixture(scope='module')
def muts_small():
    with open('test/muts_small.p', 'rb') as fl:
        muts = pickle.load(fl)
    return muts


@pytest.fixture(scope='module')
def mtree_small(muts_small):
    return data.MuTree(muts_small, levels=('Gene','Form','Exon'))


class TestBasicMuTreeCase:
    """Basic tests for Hetman MuTrees."""

    def test_samps(self, muts_small, mtree_small):
        """Does the tree properly store its samples?"""
        assert set(muts_small['Sample']) == mtree_small.get_samples()
        assert (mtree_small.get_samp_count(muts_small['Sample']) ==
                {k:list(muts_small.drop_duplicates()['Sample']).count(k)
                 for k in muts_small['Sample']})

    def test_structure(self, muts_small, mtree_small):
        """Is the internal structure of the tree correct?"""
        assert set(list(mtree_small.child.keys())) == set(muts_small['Gene'])
        assert (set(list(mtree_small.child['TTN'].child.keys()))
                == set(muts_small['Form'][muts_small['Gene'] == 'TTN']))

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


class TesBasicMuTypeCase:
    """Basic tests for Hetman MuTypes."""

    def test_samps(self, muts_small, mtree_small):
        """Can we retrieve the right samples using a MuType?"""
        mtype1 = data.MuType({('Gene', 'TP53'):None})
        mtype2 = data.MuType(
            {('Gene', 'TP53'):{('Conseq', 'Splice_Site'):None}})
        mtype3 = data.MuType(
            {('Gene', 'TP53'):
             {('Conseq', 'Missense_Mutation'):{('Exon', '5/11'):None}}})
        assert mtype1.get_samples(self.mtree) == set(self.muts['Sample'])
        assert (
            mtype2.get_samples(self.mtree) ==
            set(self.muts['Sample'][self.muts['Conseq'] == 'Splice_Site']))
        assert (
            mtype3.get_samples(self.mtree) ==
            set(self.muts['Sample'][np.array(
                [x['Conseq'] == 'Missense_Mutation'
                 and x['Exon'] == '5/11' for x in self.muts])])
            )


class MuTypeBinaryTestCase:
    """Tests the binary operators defined for MuTypes."""

    def setUp(self):
        self.mtype1 = data.MuType({('Gene', 'TP53'):None})
        self.mtype2 = data.MuType(
            {('Gene', 'TP53'):{('Conseq', 'Missense_Mutation'):None}})
        self.mtype3 = data.MuType(
            {('Gene', 'TP53'):
             {('Conseq', ('Nonsense_Mutation','Missense_Mutation')):
              {('Exon', '5/11'):None}}}
            )

    def test_or(self):
        for mtype in [self.mtype1,self.mtype2,self.mtype3]:
            assert mtype == (mtype | mtype)
        assert self.mtype1 | self.mtype2 == self.mtype1
        assert self.mtype1 | self.mtype3 == self.mtype1



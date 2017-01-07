
import unittest
import pickle
import data


class BasicMuTreeTestCase(unittest.TestCase):
    """Tests for Hetman MuTrees."""

    def setUp(self):
        with open('test/muts_small.p', 'rb') as fl:
            muts = pickle.load(fl)
        self.mtree = data.MuTree(muts=muts, levels=('Gene','Conseq','Exon'))

    def tearDown(self):
        self.mtree = None

    def test_init(self):
        """Can we properly instantiate a MuTree?"""
        assert self.mtree.levels == ((), ('Gene', 'Conseq', 'Exon'))
        assert self.mtree.branches_ == {'TP53'}
        assert len(self.mtree) == 18
        assert self.mtree.child.keys() == ['TP53']
        assert self.mtree.child['TP53'].branches_ == {
            'Nonsense_Mutation', 'Missense_Mutation',
            'Splice_Site', 'Frame_Shift'}


class MuTypeTestCase(unittest.TestCase):
    """Tests for Hetman MuTypes."""

    def test_init(self):
        """Can we properly initiate a MuType?"""
        pass


if __name__ == '__main__':
    unittest.main()


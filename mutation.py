
"""
Hetman (Heterogeneity Manifold)
Classification of mutation sub-types using expression data.
This file contains classes for representing mutation subtypes in
formats that facilitate classification of mutation sub-types.
"""

import pandas as pd

from re import sub as gsub
from math import log, ceil, exp
from enum import Enum

from functools import reduce
from itertools import combinations as combn
from itertools import groupby, product

from sklearn.cluster import MeanShift


# .. supported mutation levels ..
MutLevel = Enum('MutLevel', 'Gene Form PolyPhen GISTIC Exon Protein')


class MuTree(object):
    """A class corresponding to a hierarchy of mutation types
       present in a set of samples.
       
    Parameters
    ----------
    muts : array-like, shape (n_muts,)
        Input mutation data, each record is a mutation occurring in a sample.
        
    levels : tuple, shape (child_levels,) or ((), (child_levels,)) or
             ((parent_levels,), (child_levels,))
        A list of mutation levels to include in the tree. Any tree that is a
        child of another tree will also list the levels of its parents here.

        The Gene level corresponds to which gene is mutated (i.e. TP53, AML3),
        the Conseq level corresponds to the consequence of the mutation on the
        transcribed protein (i.e. missense, frameshift), the Exon level
        corresponds to which Exon the mutation affects, if applicable.

        Mutation trees can either have other mutation trees as children,
        corresponding to lower levels in the hierarchy, or have sets of
        individual samples as children if they are at the very bottom of the
        hierarchy.

    samples : set, optional
        Which samples' mutation data to include in the tree. Note that
        samples without any mutations will not be in the tree regardless.
        Default is to use all of the available samples.
    
    genes : set, optional
        Which genes' mutation data are to be included in the tree.
        i.e. {'TP53', 'ATM'}
        Default is to use all available genes.

    Attributes
    ----------
    branches_ : set of strs
        The branches at this level of the hierarchy, i.e. the set of genes, set
        of possible consequences, etc.
    """
    # .. functions for parsing various levels of mutation properties
    def _muts_gene(muts):
        return muts
    
    def _muts_form(muts):
        muts.loc[:,'Form'] = muts.loc[:,'Form'].str.replace('_(Del|Ins)$', '')
        return muts

    def _muts_polyphen(muts):
        mshift = MeanShift(bandwidth=exp(-3))
        mshift.fit(pd.DataFrame(muts['PolyPhen']))
        clust_vec = ['PolyPhen_' + str(round(mshift.cluster_centers_[x,0], 2))
                     for x in mshift.labels_]
        muts['PolyPhen'] = clust_vec
        return muts

    def _muts_gistic(muts):
        mshift = MeanShift(bandwidth=exp(-2))
        mshift.fit(pd.DataFrame(muts['GISTIC']))
        clust_vec = ['GISTIC_' + str(round(mshift.cluster_centers_[x,0], 2))
                     for x in mshift.labels_]
        muts['GISTIC'] = clust_vec
        return muts

    def _muts_exon(muts):
        return muts

    def _muts_protein(muts):
        return muts

    # maps mutation data parsing functions to mutation levels
    mut_fxs = {
        MutLevel.Gene: _muts_gene,
        MutLevel.Form: _muts_form,
        MutLevel.PolyPhen: _muts_polyphen,
        MutLevel.GISTIC: _muts_gistic,
        MutLevel.Exon: _muts_exon,
        MutLevel.Protein: _muts_protein
        }

    def __init__(self,
                 muts, levels=('Gene', 'Form', 'Protein'),
                 samples=None, genes=None, depth=0):
        self.levels = [MutLevel[lvl] for lvl in levels]
        self.depth = depth
        self.cur_level = MutLevel[levels[depth]]

        # gets subset of mutation data corresponding to requested gene(s)
        if genes is None:
            genes = set(muts['Gene'])
        else:
            gene_indx = [g in genes for g in muts['Gene']]
            muts = muts.ix[gene_indx, ]

        # gets subset of mutation data corresponding to requested samples
        if samples is None:
            samples = set(muts['Sample'])
        else:
            samples = set(muts['Sample']) & set(samples)
            samp_indx = [s in samples for s in muts['Sample']]
            muts = muts.ix[samp_indx, ]

        # recursively builds the mutation hierarchy
        muts = MuTree.mut_fxs[self.cur_level](muts)
        self.child = {}
        for nm, mut in muts.groupby(self.cur_level.name):
            next_level = None
            if depth < (len(levels) - 1):
                for lvl in levels[(depth + 1):]:
                    if not all([x|y for x,y in
                                zip(mut[lvl].isin(['.']),
                                    mut[lvl].isnull())]):
                        next_level = lvl
                        break
            if next_level is None:
                self.child[nm] = frozenset(mut['Sample'])
            else:
                self.child[nm] = MuTree(
                    muts=mut, levels=levels, samples=samples,
                    genes=None, depth=levels.index(next_level))

    def __str__(self):
        """Printing a MuTree shows each of the branches of the tree and
           the samples at the end of each branch."""
        new_str = self.cur_level.name
        for k,v in self.child.items():
            new_str = new_str + ' IS ' + k
            if isinstance(v, MuTree):
                new_str = (new_str + ' AND '
                           + '\n' + '\t'*(self.depth+1) + str(v))
            else:
                if len(v) > 10:
                    new_str = new_str + ': (' + str(len(v)) + ' samples)'
                else:
                    new_str = (new_str + ': '
                               + reduce(lambda x,y: x + ',' + y, tuple(v)))
            new_str = new_str + '\n' + '\t'*self.depth
        new_str = gsub('\n$', '', new_str)
        return new_str

    def __len__(self):
        """Returns the number of unique samples this MuTree contains."""
        return len(self.get_samples())

    def get_samples(self):
        """Gets the set of unique samples contained within the tree."""
        samps = set()
        for v in list(self.child.values()):
            if isinstance(v, MuTree):
                samps |= v.get_samples()
            else:
                samps |= v
        return samps

    def get_samp_count(self, samps):
        """Gets the number of branches of this tree each of the given
           samples appears in."""
        samp_count = {s:0 for s in samps}
        for v in list(self.child.values()):
            if isinstance(v, MuTree):
                new_counts = v.get_samp_count(samps)
                samp_count.update(
                    {s:(samp_count[s] + new_counts[s]) for s in samps})
            else:
                samp_count.update({s:(samp_count[s] + 1) for s in v})
        return samp_count

    def get_overlap(self, mtype1, mtype2):
        """Gets the proportion of samples in one mtype that also fall under
           another, taking the maximum of the two possible mtype orders.

        Parameters
        ----------
        mtype1,mtype2 : MuTypes
            The mutation sets to be compared.

        Returns
        -------
        ov : float
            The ratio of overlap between the two given sets.
        """
        samps1 = self.get_samples(mtype1)
        samps2 = self.get_samples(mtype2)
        if len(samps1) and len(samps2):
            ovlp = float(len(samps1 & samps2))
            ov = max(ovlp / len(samps1), ovlp / len(samps2))
        else:
            ov = 0
        return ov

    def allkey(self, levels=None):
        """Gets the key corresponding to the MuType that contains all of the
           branches of the tree. A convenience function that makes it easier
           to list all of the possible branches present in the tree, and to
           instantiate MuType objects that correspond to all of the possible
           mutation types.

        Parameters
        ----------
        levels : tuple
            A list of levels corresponding to how far the output MuType
            should recurse.

        Returns
        -------
        new_key : dict
            A MuType key which can be used to instantiate
            a MuType object (see below).
        """
        if levels is None:
            levels = [lvl.name for lvl in self.levels]
        if not levels:
            new_key = None
        elif self.cur_level.name in levels:
            cur_indx = levels.index(self.cur_level.name)
            new_lvls = levels[:cur_indx] + levels[(cur_indx+1):]
            new_key = {(self.cur_level, k):
                       v.allkey(new_lvls) if isinstance(v, MuTree)
                       else None
                       for k,v in self.child.items()}
        else:
            new_key = reduce(lambda x,y: {**x,**y},
                             [v.allkey(levels) if isinstance(v, MuTree)
                              else {k:None}
                              for k,v in self.child.items()])
        return new_key

    def subsets(self, mtype=None, levels=None):
        """Gets all of the MuTypes corresponding to exactly one of the
           branches of the tree within the given mutation set and at the
           given mutation levels.

        Parameters
        ----------
        mtype : MuType, optional
            A set of mutations whose sub-branches are to be obtained.

        levels : tuple, optional
            A list of levels where the sub-branches are to be located.

        Returns
        -------
        mtypes : list
            A list of MuTypes, each corresponding to one of the
            branches of the tree.
        """
        if mtype is None:
            mtype = MuType(self.allkey(levels))
        if levels is None:
            levels = [lvl.name for lvl in self.levels]
        if self.cur_level.name in levels:
            mtypes = []
            for k,v in self.child.items():
                for l,w in mtype.child.items():
                    if k in l:
                        new_lvls = list(
                            set(levels) - set([self.cur_level.name]))
                        if isinstance(v, MuTree) and len(new_lvls) > 0:
                            mtypes += [MuType({(self.cur_level, k):s})
                                       for s in v.subsets(w, new_lvls)]
                        else:
                            mtypes += [MuType({(self.cur_level, k):None})]
        else:
            mtypes += [v.subsets(mtype, levels)
                       for k,v in self.child.items()
                       if isinstance(v, MuTree)]
        return mtypes

    def direct_subsets(self, mtype, branches=None):
        """Gets all of the MuTypes corresponding to direct descendants
           of the given branches of the given mutation set.

        Parameters
        ----------
        mtype : MuType
            A set of mutations whose direct descendants are to be obtained.

        branches : set of strs, optional
            A set of branches whose subsets are to be obtained, the default is
            to use all available branches.

        Returns
        -------
        mtypes : list
            A list of MuTypes.
        """
        mtypes = []
        if len(self.levels[1]) > 1:
            for k,v in list(self.child.items()):
                for l,w in list(mtype.child.items()):
                    if k in l:
                        if w is not None:
                            mtypes += [MuType({(self.cur_level, k):s})
                                      for s in v.direct_subsets(w, branches)]
                        elif branches is None or k in branches:
                            if isinstance(v, MuTree):
                                mtypes += [
                                    MuType({(self.cur_level, k):
                                            MuType({(v.cur_level, x):None})})
                                    for x in list(v.child.keys())
                                    ]
                            else:
                                mtypes += [MuType({(self.cur_level, k):None})]
        else:
            if branches is None:
                branches = self.branches_
            mtypes += [
                MuType({(self.cur_level, k):None})
                for k in (set(self.child.keys())
                          & reduce(lambda x,y: x|y, list(mtype.child.keys()))
                          & branches)
                ]
        return mtypes

    def combsets(self,
                 mtype=None, levels=None,
                 min_size=1, comb_sizes=(1,)):
        """Gets the MuTypes that are subsets of this tree and that contain
           at least the given number of samples and the given number of
           individual branches at the given hierarchy levels.

        Parameters
        ----------
        mtype : MuType
            A set of mutations whose subsets are to be obtained.

        levels : tuple
            The levels that the output sets are to contain.

        min_size : int
            The minimum number of samples each returned
            subset has to contain.

        comb_sizes : tuple of ints
            The number of individual branches each returned
            subset can contain.

        Returns
        -------
        csets : list
            A list of MuTypes satisfying the given criteria.
        """
        subs = self.subsets(mtype, levels)
        csets = []
        for csize in comb_sizes:
            for kc in combn(subs, csize):
                new_set = reduce(lambda x,y: x | y, kc)
                if len(new_set.get_samples(self)) >= min_size:
                    csets += [new_set]
        return csets

    def status(self, samples, mtype=None):
        """For a given set of samples and a MuType, finds if each sample
           has a mutation in the MuType in this tree.

        Parameters
        ----------
        samples : list
            A list of samples whose mutation status is to be retrieved.

        mtype : MuType, optional
            A set of mutations whose membership we want to test.
            The default is to check against any mutation
            contained in the tree.

        Returns
        -------
        S : list of bools
            For each input sample, whether or not it has a mutation in the
            given set.
        """
        if mtype is None:
            mtype = MuType(self.allkey())
        samp_list = mtype.get_samples(self)
        return [s in samp_list for s in samples]


class MutSet(object):
    """A class corresponding to the presence (or absence)
       of multiple mutation types.
    """

    def __init__(self, relation, muts1, muts2):
        if relation not in set(['AND', 'AND NOT', 'OR NOT']):
            raise HetmanDataError(
                "relation must be one of AND, AND NOT, OR NOT")
        if (not (isinstance(muts1, MuType) or isinstance(muts1, MutSet))
            or not (isinstance(muts1, MuType) or isinstance(muts1, MutSet))):
            raise HetmanDataError(
                "muts must both be either a MuType or MutSet")
        self.relation = relation
        if (isinstance(muts1, MutSet)
            and muts1.relation == 'AND NOT' and self.relation == 'AND NOT'):
            self.muts1 = muts1.muts1
            self.muts2 = muts1.muts2 | muts2
        else:
            self.muts1 = muts1
            self.muts2 = muts2

    def __str__(self):
        return ('\t' + str(self.muts1)
                + '\n' + self.relation + '\n'
                + '\t' + str(self.muts2))

    def __hash__(self):
        return ((hash(self.muts1) + hash(self.muts2)) ^ hash(self.relation))

    def __or__(self, other):
        if isinstance(other, MuType):
            if self.relation == 'AND':
                mset = MutSet('AND', self.muts1 | other, self.muts2 | other)
            elif self.relation == 'AND NOT':
                mset = MutSet('AND', self.muts1 | other,
                                     MutSet('OR NOT', other, self.muts2))
            elif self.relation == 'OR NOT':
                mset = MutSet('OR NOT', self.muts1 | other, self.muts2)
        elif isinstance(other, MutSet):
            if self.relation == 'AND' and other.relation == 'AND':
                mset = MutSet('AND',
                              MutSet('AND', self.muts1 | other.muts1,
                                            self.muts1 | other.muts2),
                              MutSet('AND', self.muts2 | other.muts1,
                                            self.muts2 | other.muts2)
                             )
            elif self.relation == 'AND NOT' and other.relation == 'AND NOT':
                mset = MutSet('AND',
                              MutSet('AND', self.muts1 | other.muts1,
                                     MutSet('OR NOT',
                                            self.muts1, other.muts2)),
                              MutSet('AND NOT',
                                     MutSet('OR NOT', other.muts1, self.muts2),
                                     MutSet('AND', self.muts2, other.muts2)))
            elif self.relation == 'OR NOT' and other.relation == 'OR NOT':
                mset = MutSet('OR NOT',
                              MutSet('OR NOT',
                                     self.muts1 | other.muts1, self.muts2),
                              other.muts2)
            elif self.relation == 'AND' and other.relation == 'AND NOT':
                mset = MutSet('AND',
                              MutSet('AND', self.muts1 | other.muts1,
                                     MutSet('OR NOT',
                                            self.muts1, other.muts2)),
                              MutSet('AND', self.muts2 | other.muts1,
                                     MutSet('OR NOT',
                                            self.muts2, other.muts2)))
            elif self.relation == 'AND NOT' and other.relation == 'AND':
                mset = MutSet('AND',
                              MutSet('AND', self.muts1 | other.muts1,
                                            self.muts1 | other.muts2),
                              MutSet('AND',
                                     MutSet('OR NOT',
                                            other.muts1, self.muts2),
                                     MutSet('OR NOT',
                                            other.muts2, self.muts2)))
            elif self.relation == 'AND' and other.relation == 'OR NOT':
                mset = MutSet('OR NOT',
                              MutSet('AND', self.muts1 | other.muts1,
                                            self.muts2 | other.muts1),
                              other.muts2)
            elif self.relation == 'OR NOT' and other.relation == 'AND':
                mset = MutSet('OR NOT',
                              MutSet('AND', self.muts1 | other.muts1,
                                            self.muts1 | other.muts2),
                              self.muts2)
            elif self.relation == 'AND NOT' and other.relation == 'OR NOT':
                mset = MutSet('OR NOT',
                              MutSet('AND', self.muts1 | other.muts1,
                                     MutSet('OR NOT',
                                            other.muts1, self.muts2)),
                              other.muts2)
            elif self.relation == 'OR NOT' and other.relation == 'AND NOT':
                mset = MutSet('OR NOT',
                              MutSet('AND', self.muts1 | other.muts1,
                                     MutSet('OR NOT',
                                            self.muts1, other.muts2)),
                              self.muts2)
        else:
            mset = NotImplemented
        return mset

    def __ror__(self, other):
        return self | other

    def __and__(self, other):
        if self == other:
            return self
        else:
            return MutSet('AND', self, other)

    def __rand__(self, other):
        return self & other

    def __xor__(self, other):
        return MutSet('AND NOT', self | other, MutSet('AND', self, other))

    def get_samples(self, mtree):
        """Gets the set of unique of samples contained within a particular
           branch or branches of the tree.

        Parameters
        ----------
        mtype : MuType or MutSet, optional
            The set of mutation types whose samples we want to retrieve.
            The default is to use all mutation types stored in the tree.

        Returns
        -------
        samps : set
            The list of samples that have the specified type of mutations.
        """
        if self.relation == 'AND':
            samps = (self.muts1.get_samples(mtree)
                     & self.muts2.get_samples(mtree))
        elif self.relation == 'AND NOT':
            samps = (self.muts1.get_samples(mtree)
                     - self.muts2.get_samples(mtree))
        elif self.relation == 'OR NOT':
            samps = (self.muts1.get_samples(mtree)
                     | (mtree.get_samples() - self.muts2.get_samples(mtree)))
        return samps

    def rationalize(self, mtree):
        new_muts1 = self.muts1.rationalize(mtree)
        new_muts2 = self.muts2.rationalize(mtree)
        if isinstance(new_muts1, MuType) and isinstance(new_muts2, MuType):
            if new_muts1 >= new_muts2 and self.relation == 'AND':
                return new_muts2
            elif new_muts2 >= new_muts1 and self.relation == 'AND':
                return new_muts1
            elif self.relation == 'OR NOT':
                new_set = new_muts1 | new_muts2.invert(mtree)
                return new_set.rationalize(mtree)
        return MutSet(self.relation, new_muts1, new_muts2)


class MuType(object):
    """A class corresponding to a subset of mutations defined through hierarchy
       of properties. Used in conjunction with the above MuTree class to
       navigate the space of possible mutation subsets.

    Parameters
    ----------
    set_key : dict
        Define the mutation sub-types that are to be included in this set.
        Takes the form {(Level,Sub-Type):None or set_key, ...}.

        A value of None denotes all of the samples with the given sub-type of
        mutation at the given level, otherwise another set-key which defines a
        further subset of mutations contained within the given sub-type.
        Sub-Type can consist of multiple values, in which case the
        corresponding value applies to all of the included sub-types.

        i.e. {('Gene','TP53'):None} is the subset containing any mutation
        of the TP53 gene.
        {('Gene','BRAF'):{('Conseq',('missense','frameshift')):None}} contains
        the mutations of BRAF that result in a missense variation or a shift of
        the reading frame.

        As with MuTrees, MuTypes are constructed recursively, and so each value
        in a set key is used to create another MuType, unless it is None
        signifying a leaf node in the hierarchy.

    Attributes
    ----------
    level_ : str
        The mutation level at the head of this mutation set.
    """

    def __init__(self, set_key):
        # gets the mutation hierarchy level of this set, makes sure
        # the key is properly specified
        level = set(k for k,_ in list(set_key.keys()))
        if len(level) > 1:
            raise HetmanDataError(
                "improperly defined MuType key (multiple mutation levels)")
        level = tuple(level)[0]
        if isinstance(level, MutLevel):
            self.level_ = level
        else:
            self.level_ = MutLevel[level]

        # gets the subsets of mutations defined at this level, and
        # their further subdivisions if they exist
        membs = [(k,) if isinstance(k, str) else k
                 for _,k in list(set_key.keys())]
        children = dict(
            tuple((v, ch)) if isinstance(ch, MuType) or ch is None else
            tuple((v, MuType(ch)))
            for v,ch in zip(membs, list(set_key.values()))
            )

        # merges subsets at this level if their children are the same, i.e.
        # missense:None, frameshift:None => (missense,frameshift):None
        uniq_ch = set(children.values())
        self.child = {frozenset(i for j in
                                [k for k,v in list(children.items())
                                 if v == ch]
                                for i in j):ch for ch in uniq_ch}

    def __eq__(self, other):
        """Two MuTypes are equal if and only if they have the same set
           of children MuTypes for the same subsets."""
        if isinstance(self, MuType) ^ isinstance(other, MuType):
            eq = False
        elif self.level_ != other.level_:
            eq = False
        else:
            eq = (self.child == other.child)
        return eq

    def __str__(self):
        """Printing a MuType shows the hierarchy of mutation
           properties contained within it."""
        new_str = ''
        if self.level_ == MutLevel.Gene:
            new_str += 'a mutation where '
        for k,v in list(self.child.items()):
            new_str += (self.level_.name + ' IS '
                        + reduce(lambda x,y: x + ' OR ' + y, k))
            if v is not None:
                new_str += ' AND ' + '\n\t' + str(v)
            new_str += '\nOR '
        return gsub('\nOR $', '', new_str)

    def _raw_key(self):
        "Returns the expanded key of a MuType."
        rmembs = reduce(lambda x,y: x|y, list(self.child.keys()))
        return {memb:reduce(lambda x,y: x|y,
                            [v for k,v in list(self.child.items())
                             if memb in k])
                for memb in rmembs}

    def __or__(self, other):
        """Returns the union of two MuTypes."""
        if not isinstance(other, MuType):
            return NotImplemented
        if self.level_ is not other.level_:
            if self.level_.value < other.level_.value:
                return self
            else:
                return other
        else:
            new_key = {}
            self_set = set(self.child.keys()) - set(other.child.keys())
            other_set = set(other.child.keys()) - set(self.child.keys())
            both_set = set(self.child.keys()) & set(other.child.keys())

            if self_set:
                new_key.update({(self.level_, k):self.child[k]
                                for k in self_set})
            if other_set:
                new_key.update({(other.level_, k):other.child[k]
                                for k in other_set})
            if both_set:
                new_key.update(dict(
                    tuple((tuple((self.level_, k)), self.child[k]))
                    if self.child[k] == other.child[k]
                    else tuple((tuple((self.level_, k)), None))
                    if self.child[k] is None or other.child[k] is None
                    else tuple((tuple((self.level_, k)),
                                self.child[k] | other.child[k]))
                    for k in both_set))
        return MuType(new_key)

    def __and__(self, other):
        """Finds the intersection of two MuTypes."""
        if not isinstance(other, MuType):
            return NotImplemented
        if self.level_ is not other.level_:
            if self.level_.value > other.level_.value:
                return self
            else:
                return other
        else:
            new_key = {}
            self_keys = self._raw_key()
            other_keys = other._raw_key()
            both_set = list(set(self_keys) & set(other_keys))
            for k in both_set:
                if self_keys[k] is None:
                    new_key.update({(self.level_, k):other_keys[k]})
                elif other_keys[k] is None:
                    new_key.update({(self.level_, k):self_keys[k]})
                elif self_keys[k] == other_keys[k]:
                    new_key.update({(self.level_, k):self_keys[k]})
                else:
                    intx = self_keys[k] & other_keys[k]
                    if intx is not None:
                        new_key.update({(self.level_, k):intx})
            if new_key:
                return MuType(new_key)
            else:
                return None

    def __add__(self, other):
        if self == other:
            return self
        else:
            return MutSet('AND', self, other)

    def __ge__(self, other):
        """Checks if one MuType is a subset of the other."""
        if self.level_ != other.level_:
            raise HetmanDataError('mismatching MuType levels')
        self_keys = reduce(lambda x,y: x|y, list(self.child.keys()))
        other_keys = reduce(lambda x,y: x|y, list(other.child.keys()))
        self_keys = {x:reduce(lambda x,y: x|y,
                                [v for k,v in list(self.child.items()) if x in k])
                       for x in self_keys}
        other_keys = {x:reduce(lambda x,y: x|y,
                                 [v for k,v in list(other.child.items()) if x in k])
                        for x in other_keys}
        if set(self_keys) >= set(other_keys):
            return all([True if self_keys[k] is None
                        else False if other_keys[k] is None
                        else self_keys[k] >= other_keys[k]
                       for k in list(set(self_keys) & set(other_keys))])
        else:
            return False

    def __gt__(self, other):
        """Checks if one MuType is a proper subset of the other."""
        if self.level_ != other.level_:
            raise HetmanDataError('mismatching MuType levels')
        self_keys = reduce(lambda x,y: x|y, list(self.child.keys()))
        other_keys = reduce(lambda x,y: x|y, list(other.child.keys()))
        self_keys = {x:reduce(lambda x,y: x|y,
                                [v for k,v in list(self.child.items()) if x in k])
                       for x in self_keys}
        other_keys = {x:reduce(lambda x,y: x|y,
                                 [v for k,v in list(other.child.items()) if x in k])
                        for x in other_keys}
        if set(self_keys) == set(other_keys):
            comp_keys = list(set(self_keys) & set(other_keys))
            gt_comp = [True if self_keys[k] is None
                       else False if other_keys[k] is None
                       else self_keys[k] >= other_keys[k]
                       for k in comp_keys]
            eq_comp = [self_keys[k] == other_keys[k] for k in comp_keys]
            return all(gt_comp) and not all(eq_comp)
        elif set(self_keys) > set(other_keys):
            comp_keys = list(set(self_keys) & set(other_keys))
            return all([self_keys[k] >= other_keys[k]
                    for k in comp_keys])
        else:
            return False

    def __sub__(self, other):
        """Subtracts one MuType from another."""
        if self.level_ != other.level_:
            raise HetmanDataError("mismatching MuType levels")
        self_keys = reduce(lambda x,y: x|y, list(self.child.keys()))
        other_keys = reduce(lambda x,y: x|y, list(other.child.keys()))
        self_keys = {x:reduce(lambda x,y: x|y,
                                [v for k,v in list(self.child.items()) if x in k])
                       for x in self_keys}
        other_keys = {x:reduce(lambda x,y: x|y,
                                 [v for k,v in list(other.child.items()) if x in k])
                        for x in other_keys}
        new_key = {}
        for k in self_keys:
            if k in other_keys:
                if (other_keys[k] is not None
                    and self_keys[k] != other_keys[k]):
                    sub_val = self_keys[k] - other_keys[k]
                    new_key.update({(self.level_, k):sub_val})
            else:
                new_key.update({(self.level_, k):self_keys[k]})
        if new_key:
            return MuType(new_key)
        else:
            return None

    def __hash__(self):
        """MuType hashes are defined in an analagous fashion to those of
           tuples, see for instance http://effbot.org/zone/python-hash.htm"""
        value = 0x163125
        for k,v in list(self.child.items()):
            value += eval(hex((int(value) * 1000007) & 0xFFFFFFFF)[:-1])
            value ^= hash(k) ^ hash(v)
            value ^= len(self.child)
        if value == -1:
            value = -2
        return value

    def get_samples(self, mtree):
        """Gets the set of unique of samples contained within a particular
           branch or branches of the tree.

        Parameters
        ----------
        mtree : MuTree
            A set of samples organized according to the mutations they have.

        Returns
        -------
        samps : set
            The list of samples that have the specified type of mutations.
        """
        samps = set()
        if self.level_ in mtree.levels:
            if self.level_ is not mtree.cur_level:
                for v in mtree.child.values():
                    if isinstance(v, MuTree):
                        samps |= self.get_samples(v)
            else:
                for k,v in mtree.child.items():
                    for l,w in self.child.items():
                        if k in l:
                            if isinstance(v, frozenset):
                                samps |= v
                            elif w is None:
                                samps |= v.get_samples()
                            else:
                                samps |= w.get_samples(v)
        return samps

    def invert(self, mtree):
        """Returns the mutation types not included in this set of types that
           are also in the given tree.
        """
        new_key = {}
        self_ch = self._raw_key()
        for k in (set(mtree.child.keys()) - set(self_ch.keys())):
            new_key[(self.level_, k)] = None
        for k in (set(mtree.child.keys()) & set(self_ch.keys())):
            if self_ch[k] is not None and isinstance(mtree.child[k], MuTree):
                new_key[(self.level_, k)] = self_ch[k].invert(mtree.child[k])
        return MuType(new_key)

    def pure_set(self, mtree):
        """Returns the set of mutations equivalent to the samples that have
           this type of mutation and no others.
        """
        return MutSet('AND NOT', self, self.invert(mtree))

    def subkeys(self):
        """Gets all of the possible subsets of this MuType that contain
           exactly one of the leaf properties."""
        mkeys = []
        for k,v in list(self.child.items()):
            if v is None:
                mkeys += [{(self.level_, i):None} for i in k]
            else:
                mkeys += [{(self.level_, i):s} for i in k for s in v.subkeys()]

        return mkeys

    def rationalize(self, mtree):
        """Simplifies the structure of MuType if it finds that some of its
           branches correspond to the full set of branches possible in the
           mutation hierarchy.
        """
        if self.child == {frozenset(mtree.branches_): None}:
            if mtree.levels[0]:
                new_set = None
            else:
                new_set = MuType(
                    {(mtree.cur_level,tuple(mtree.branches_)):None})
        elif len(mtree.levels[1]) == 1:
            new_set = self
        else:
            new_key = {}
            for k,v in list(mtree.child.items()):
                for l,w in list(self.child.items()):
                    if k in l:
                        if w is not None:
                            new_key.update([((mtree.cur_level,k),
                                            w.rationalize(v))])
                        else:
                            new_key.update([((mtree.cur_level,k), None)])
            new_set = MuType(new_key)
            if new_set.child == {frozenset(mtree.branches_): None}:
                if mtree.levels[0]:
                    new_set = None
                else:
                    new_set = MuType(
                        {(mtree.cur_level,tuple(mtree.branches_)):None})
        return new_set

    def prune(self, mtree, min_prop=2.0/3, max_part=25, min_size=8):
        """Gets the mutation subsets of this tree that are also subsets of
        the given mutation set and include or exclude at least the given
        proportion of samples contained herein. Only subsets at the same level
        or one level below the given set are considered.

        If the possible number of mutation subsets is too high, the space of
        subsets is pruned by merging the smallest subsets and by narrowing the
        sample proportion threshold. Subsets are also filtered against a list
        of subsets that are to be excluded.

        Parameters
        ----------
        mtype : MuType
            A set of mutations within which subsets are to be obtained.

        prop_use : float, optional
            A sample proportion threshold used to filter the set of output
            subsets: prop_use<=(set_size)<=(1-prop_use), where set size is
            relative to the given mutation set.

        max_part : int
            The maximum number of mutation subsets that can be returned.

        Returns
        -------
        psets : list of MuTypes and/or MutSets
            A list of MuTypes that satisfy the given criteria.
        """
        orig_size = len(self.get_samples(mtree))
        min_samps = max(int(round(orig_size * min_prop)), min_size)
        prune_sets = [x for x in [MutSet('AND NOT', self, MuType(m))
                                  for m in self.invert(mtree).subkeys()]
                      if min_samps <= len(x.get_samples(mtree)) < orig_size]
        sub_groups = [MuType(m) for m in self.subkeys()]
        sub_list = [mtree.direct_subsets(m) for m in sub_groups]

        for i in range(len(sub_list)):
            if not sub_list[i]:
                sub_list[i] = [sub_groups[i]]
        sub_lens = [len(x) for x in sub_list]
        if all([x == 1 for x in sub_lens]):
            sub_groups = [self]
            sub_list = [reduce(lambda x,y: x+y, sub_list)]
            sub_lens = [len(sub_groups)]

        sub_sizes = [len(m.get_samples(mtree)) for m in sub_groups]
        test_count = 1
        for x in sub_lens:
            test_count *= 2**x - 1
        test_count -= 1

        if test_count > 1000:
            max_subs = [1000 ** (float(x)/sum(sub_sizes)) for x in sub_sizes]

            for i in range(len(sub_list)):
                if max_subs[i] > (2**sub_lens[i] - 1):
                    for j in range(len(sub_list))[(i+1):]:
                        max_subs[j] = (
                            (max_subs[j] * (max_subs[i] / (2**sub_lens[i]-1)))
                            ** (1.0 / (len(sub_list)-i-1)))
                    max_subs[i] = 2**sub_lens[i] - 1
                sub_indx = sorted(
                    [(x, float(len(x.get_samples(mtree)))) for x in sub_list[i]],
                    key=lambda y: y[1],
                    reverse=True
                    )

                while len(sub_indx) > max(ceil(log(max_subs[i], 2)), 1):
                    new_sub = sub_indx[-2][0] | sub_indx[-1][0]
                    sub_indx = sub_indx[:-2]
                    new_indx = (new_sub, float(len(new_sub.get_samples(mtree))))
                    sort_indx = sum([new_indx[1] < v for _,v in sub_indx])
                    sub_indx.insert(sort_indx, new_indx)
                sub_list[i] = [x[0] for x in sub_indx]

        psets = {}
        prune_count = 1000 + len(prune_sets)
        while prune_count > max_part:
            use_sets = []
            for csizes in product(*[list(range(1, len(x)+1)) for x in sub_list]):
                for set_combn in product(
                    *[combn(sl, csize) for sl,csize in zip(sub_list,csizes)]):
                    set_comps = [reduce(lambda y,z: y|z, x) for x in set_combn]
                    new_set = reduce(lambda x,y: x|y, set_comps)
                    new_set = new_set.rationalize(mtree)
                    if new_set not in psets:
                        psets[new_set] = len(new_set.get_samples(mtree))
                    if orig_size > psets[new_set] >= min_samps:
                        use_sets += [new_set]
            use_sets = list(set(use_sets))
            prune_count = len(use_sets) + len(prune_sets)

            if prune_count > max_part:
                subs_prune = [(i,x[0]) for i,x in
                              enumerate(zip(sub_sizes,sub_list))
                              if len(x[1]) > 1]
                min_size = min([sz for i,sz in subs_prune])
                min_indx = subs_prune[[sz for i,sz
                                       in subs_prune].index(min_size)][0]
                sub_indx = sorted(
                    [(x, float(len(x.get_samples(mtree))))
                     for x in sub_list[min_indx]],
                    key=lambda y: y[1],
                    reverse=True
                    )

                new_sub = sub_indx[-2][0] | sub_indx[-1][0]
                sub_indx = sub_indx[:-2]
                new_indx = (new_sub, float(len(new_sub.get_samples(mtree))))
                sort_indx = sum([new_indx[1] < v for _,v in sub_indx])
                sub_indx.insert(sort_indx, new_indx)
                sub_list[min_indx] = [x[0] for x in sub_indx]

        return use_sets + prune_sets



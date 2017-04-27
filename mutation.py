
"""
HetMan (Heterogeneity Manifold)
Classification of mutation sub-types using expression data.
This file contains classes for representing and storing mutation sub-types.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

from .pipelines import ClassPipe, RegrPipe

import numpy as np
import pandas as pd

from re import sub as gsub
from math import log, ceil, exp
from scipy.stats import describe
from enum import Enum

from functools import reduce
from itertools import combinations as combn
from itertools import groupby, product

from sklearn.cluster import MeanShift


class HetManMutError(Exception):
    pass


class MuTree(object):
    """A class corresponding to a hierarchy of mutation type
       present in a set of samples.
       
    Parameters
    ----------
    muts : pandas DataFrame, shape (n_muts,)
        Input mutation data, each record is a mutation occurring in a sample.
        Must contain a 'Sample' column.
        
    levels : tuple
        A list of mutation levels to be included in the tree.
        All sub-trees will have list the same set of levels regardless of
        their depth in the hierarchy.

        Levels can either be fields in the 'muts' DataFrame, in which case
        the tree will have a branch for each unique value in the field, or
        one of the keys of the MuTree.mut_fields object, in which case they
        will be defined by the corresponding MuType.muts_<level> method.

        Mutation trees can either have other mutation trees as children,
        corresponding to lower levels in the hierarchy, or have lists of
        individual samples as children if they are at the very bottom of the
        hierarchy which are stored as frozensets in the case of discrete
        mutation types and dicts in the case of continuous mutations.
    """

    # mapping between mutation fields and custom mutation levels
    mut_fields = {
        'Type': ['Gene', 'Form', 'Protein']
        }

    # .. functions for finding available branches of mutation levels ..
    @classmethod
    def check_muts(cls, muts, levels):
        """Checks that at least one of the given levels can be found in the
           given list of mutations.
        """
        muts_left = False
        lvls_left = list(levels)

        while lvls_left and not muts_left:
            cur_lvl = lvls_left.pop(0).split('_')[0]
            if cur_lvl in muts:
                muts_left = not all(pd.isnull(muts[cur_lvl]))

            elif cur_lvl in cls.mut_fields:
                if not all([x in muts for x in cls.mut_fields[cur_lvl]]):
                    raise HetManMutError("For mutation level " + cur_lvl
                                         + ", " + str(cls.mut_fields[cur_lvl])
                                         + " need to be provided as fields "
                                         "in the input.")
                else:
                    muts_left = not all(pd.isnull(
                        muts.loc[:, cls.mut_fields[cur_lvl]]))

            else:
                raise HetManMutError("Unknown mutation level " + cur_lvl
                                     + " which is not in the given mutation "
                                     "data frame nor a custom-defined level!")

        return muts_left

    @classmethod
    def split_muts(cls, muts, lvl_name):
        """Splits mutations into tree branches for a given level."""

        # level names have to consist of a base level name and an optional
        # parsing label separated by an underscore
        lvl_info = lvl_name.split('_')
        if len(lvl_info) > 2:
            raise HetManMutError("Invalid level name " + lvl_name
                                 + " with more than two fields!")

        # if a parsing label is present, add the parsed level
        # to the table of mutations
        elif len(lvl_info) == 2:
            parse_lbl = lvl_info[1].lower()
            parse_fx = 'parse_' + parse_lbl

            if parse_fx in cls.__dict__:
                muts = eval('cls.' + parse_fx)(muts, lvl_info[0])
            else:
                raise HetManMutError("Custom parse label " + parse_lbl
                                     + " must have a corresponding <"
                                     + parse_fx + "> method defined in "
                                     + cls.__name__ + "!")

        # splits mutations according to values of the specified level
        if isinstance(muts, tuple):
            if all(pd.isnull(val) for _, val in muts):
                split_muts = {}
            else:
                split_muts = muts
        elif lvl_info[0] in muts:
            split_muts = dict(tuple(muts.groupby(lvl_info[0])))

        # if the specified level is not a column in the mutation table,
        # we assume it's a custom mutation level
        else:
            split_fx = 'muts_' + lvl_info[0].lower()
            if split_fx in cls.__dict__:
                split_muts = eval('cls.' + split_fx)(muts)
            else:
                raise HetManMutError("Custom mutation level " + lvl_name
                                     + " must have a corresponding <"
                                     + split_fx + "> method defined in "
                                     + cls.__name__ + "!")

        return split_muts

    # .. functions for defining custom mutation levels ..
    @staticmethod
    def muts_type(muts):
        """Parses mutations according to Type, which can be 'CNV' (Gain or
           Loss), 'Point' (missense and silent mutations), or 'Frame' (indels,
           frameshifts, nonsense mutations).
        """
        new_muts = {}

        cnv_indx = muts['Form'].isin(['Gain', 'Loss'])
        point_indx = muts['Protein'].str.match(
            pat='^p\\.[A-Z][0-9]+[A-Z]$', as_indexer=True, na=False)
        frame_indx = muts['Protein'].str.match(
            pat='^p\\..*(?:\\*|(?:ins|del))', as_indexer=True, na=False)
        other_indx = ~(cnv_indx | point_indx | frame_indx)

        if any(cnv_indx):
            new_muts['CNV'] = muts.loc[cnv_indx, :]
        if any(point_indx):
            new_muts['Point'] = muts.loc[point_indx, :]
        if any(frame_indx):
            new_muts['Frame'] = muts.loc[frame_indx, :]
        if any(other_indx):
            new_muts['Other'] = muts.loc[other_indx, :]

        return new_muts

    # .. functions for custom parsing of mutation levels ..
    @staticmethod
    def parse_base(muts, parse_lvl):
        """Removes trailing _Del and _Ins, merging insertions and deletions
           of the same type together.
        """
        new_lvl = parse_lvl + '_base'
        new_muts = muts.assign(**{new_lvl: muts.loc[:, parse_lvl]})
        new_muts = new_muts.replace(to_replace={new_lvl: {'_(Del|Ins)$': ''}},
                                    regex=True, inplace=False)

        return new_muts

    @staticmethod
    def parse_clust(muts, parse_lvl):
        """Clusters continuous mutation scores into discrete levels."""
        mshift = MeanShift(bandwidth=exp(-3))
        mshift.fit(pd.DataFrame(muts[parse_lvl]))
        clust_vec = [(parse_lvl + '_'
                      + str(round(mshift.cluster_centers_[x,0], 2)))
                     for x in mshift.labels_]
        new_muts = mut
        new_muts[parse_lvl + '_clust'] = clust_vec

        return new_muts

    @staticmethod
    def parse_scores(muts, parse_lvl):
        return tuple(zip(muts['Sample'], pd.to_numeric(muts[parse_lvl])))


    def __new__(cls, muts, levels=('Gene', 'Form'), **kwargs):
        new_muts = cls.check_muts(muts, levels)
        if new_muts:
            return super(MuTree, cls).__new__(cls)
        else:
            return frozenset(muts['Sample'])

    def __init__(self, muts, levels=('Gene', 'Form'), **kwargs):
        if 'Sample' not in muts:
            raise HetManMutError("Mutations must have a 'Sample' field!")
        if 'depth' in kwargs:
            depth = kwargs['depth']
        else:
            depth = 0
        self.depth = depth

        # recursively builds the mutation hierarchy
        lvls_left = list(levels)
        self.child = {}
        rel_depth = 0

        while lvls_left and not self.child:
            cur_lvl = lvls_left.pop(0)
            splat_muts = self.split_muts(muts, cur_lvl)

            if splat_muts:
                self.cur_level = levels[rel_depth]
                if isinstance(splat_muts, tuple):
                    self.child = dict(splat_muts)

                else:
                    for nm, mut in splat_muts.items():
                        self.child[nm] = MuTree(mut, lvls_left,
                                                depth=self.depth+1)
            else:
                rel_depth += 1

    def __iter__(self):
        """Allows iteration over mutation categories at the current level."""
        if isinstance(self.child, frozenset):
            return iter(self.child)
        else:
            return iter(self.child.items())

    def __getitem__(self, key):
        """Gets a particular category of mutations at the current level."""
        if not key:
            key_item = self
        elif isinstance(key, str):
            key_item = self.child[key]
        elif hasattr(key, '__getitem__'):
            sub_item = self.child[key[0]]
            if isinstance(sub_item, MuTree):
                key_item = sub_item[key[1:]]
            elif key[1:]:
                raise KeyError("Key has more levels than this MuTree!")
            else:
                key_item = sub_item
        else:
            raise TypeError("Unsupported key type " + type(key) + "!")

        return key_item

    def __str__(self):
        """Printing a MuTree shows each of the branches of the tree and
           the samples at the end of each branch."""
        new_str = self.cur_level

        # if the current level is continuous, print summary statistics
        if '_scores' in self.cur_level:
            if len(self) > 8:
                score_dict = np.percentile(tuple(self.get_scores().values()),
                                           [0, 25, 50, 75, 100])
                new_str = (new_str + ": {} samples with score distribution "
                           "Min({:05.4f}) 1Q({:05.4f}) Med({:05.4f}) "
                           "3Q({:05.4f}) Max({:05.4f})".format(
                               len(self), *score_dict))
            else:
                new_str = new_str + ': ' + str(self.get_scores())

        # otherwise, iterate over the branches, recursing when necessary
        else:
            for nm, mut in self:
                new_str = new_str + ' IS ' + nm
                if isinstance(mut, MuTree):
                    new_str = (new_str + ' AND '
                               + '\n' + '\t'*(self.depth+1) + str(mut))

                # if we have reached a root node, print the samples
                else:
                    if not hasattr(mut, '__len__'):
                        new_str = new_str + str(round(mut, 2))
                    elif len(mut) > 10:
                        new_str = (new_str
                                   + ': (' + str(len(mut)) + ' samples)')
                    elif isinstance(mut, frozenset):
                        new_str = (new_str + ': '
                                   + reduce(lambda x,y: x + ',' + y, mut))
                new_str = new_str + '\n' + '\t'*self.depth
            new_str = gsub('\n$', '', new_str)

        return new_str

    def __len__(self):
        """Returns the number of unique samples this MuTree contains."""
        return len(self.get_samples())

    def get_levels(self):
        """Gets all the levels present in this tree and its children."""
        levels = set([self.cur_level])
        for _, mut in self:
            if isinstance(mut, MuTree):
                levels |= set(mut.get_levels())
        return levels

    def get_samples(self):
        """Gets the set of unique samples contained within the tree."""
        samps = set()
        for nm, mut in self:
            if isinstance(mut, MuTree):
                samps |= mut.get_samples()
            elif isinstance(mut, frozenset):
                samps |= mut
            else:
                samps |= set([nm])
        return samps

    def get_scores(self):
        """Gets all the sample scores contained within the tree."""
        scores = {}
        for nm, mut in self:
            if isinstance(mut, MuTree):
                scores = {**scores, **mut.get_scores()}
            elif isinstance(mut, frozenset):
                pass
            else:
                scores = {**scores, **{nm: mut}}
        return scores

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
            levels = self.get_levels()
        new_lvls = set(levels) - set([self.cur_level])

        if self.cur_level in levels:
            if '_scores' in self.cur_level:
                new_key = {(self.cur_level, 'Value'): None}

            else:
                new_key = {(self.cur_level, nm):
                           (mut.allkey(new_lvls)
                            if isinstance(mut, MuTree) and new_lvls
                            else None)
                           for nm, mut in self}

        else:
            new_key = reduce(
                lambda x,y: dict(
                    tuple(x.items()) + tuple(y.items())
                    + tuple((k, None) if x[k] is None
                            else (k, {**x[k], **y[k]})
                            for k in set(x) & set(y))),
                [mut.allkey(new_lvls) if isinstance(mut, MuTree) and new_lvls
                 else {(self.cur_level, 'Value'): None}
                 if '_scores' in self.cur_level
                 else {(self.cur_level, nm): None}
                 for nm, mut in self]
                )

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
            levels = self.get_levels()
        mtypes = []

        if self.cur_level in levels:
            for nm,mut in self:
                for k,v in mtype:
                    if k in nm:
                        new_lvls = list(
                            set(levels) - set([self.cur_level]))
                        if isinstance(mut, MuTree) and len(new_lvls) > 0:
                            mtypes += [MuType({(self.cur_level, k):s})
                                       for s in mut.subsets(v, new_lvls)]
                        else:
                            mtypes += [MuType({(self.cur_level, k):None})]
        else:
            mtypes += [mut.subsets(mtype, levels) for _,mut in self
                       if isinstance(mut, MuTree)]
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
        all_subs = self.subsets(mtype, levels)
        csets = []
        for csize in comb_sizes:
            for kc in combn(all_subs, csize):
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

    def scores(self, samples, mtype=None):
        """For a given set of samples and a MuType, finds the mutation score
           for each sample if applicable (zero otherwise).
        """
        if mtype is None:
            mtype = MuType(self.allkey())
        score_list = mtype.get_scores(self)
        return [score_list[samp] if samp in score_list else 0
                for samp in samples]

    def mut_vec(self, clf, samples, mtype=None):
        """Gets the appropriate mutation output vector corresponding to the
           classifier.
        """
        if isinstance(clf, ClassPipe):
            muts = self.status(samples, mtype)
        elif isinstance(clf, RegrPipe):
            muts = self.scores(samples, mtype)
        else:
            raise HetManMutError("Classifier must be either a ClassPipe "
                                 "or a RegrPipe!")
        return muts


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
    cur_level : str
        The mutation level at the head of this mutation set.
    """

    def __init__(self, set_key):
        # gets the mutation hierarchy level of this set, makes sure
        # the key is properly specified
        level = set(k for k,_ in list(set_key.keys()))
        if len(level) > 1:
            raise HetmanDataError(
                "improperly defined MuType key (multiple mutation levels)")
        if level:
            self.cur_level = tuple(level)[0]
        else:
            self.cur_level = None

        # gets the subsets of mutations defined at this level, and
        # their further subdivisions if they exist
        membs = [(k,) if isinstance(k, str) else k
                 for _,k in list(set_key.keys())]
        children = {
            tuple(i for i in k):
            (ch if ch is None or isinstance(ch, MuType) else MuType(ch))
            for k,ch in zip(membs, set_key.values())
            }

        # merges subsets at this level if their children are the same:
        #   missense:None, frameshift:None => (missense,frameshift):None
        # or if they have the same keys:
        #   (missense, splice):M1, missense:M2, splice:M2
        #    => (missense, splice):(M1, M2)
        uniq_ch = set(children.values())
        uniq_vals = tuple((frozenset(i for j in
                              [k for k,v in children.items() if v == ch]
                              for i in j), ch) for ch in uniq_ch)

        self.child = {}
        for val, ch in uniq_vals:
            if val in self.child:
                if ch is None or self.child[val] is None:
                    self.child[val] = None
                else:
                    self.child[val] |= ch
            else:
                self.child[val] = ch

    def __iter__(self):
        """Returns an expanded representation of the set structure."""
        return iter((l,v) for k,v in self.child.items() for l in k)

    def __eq__(self, other):
        """Two MuTypes are equal if and only if they have the same set
           of children MuTypes for the same subsets."""
        if isinstance(self, MuType) ^ isinstance(other, MuType):
            eq = False
        elif self.cur_level != other.cur_level:
            eq = False
        else:
            eq = (self.child == other.child)

        return eq

    def __repr__(self):
        """Shows the hierarchy of mutation properties contained
           within the MuType."""
        new_str = ''

        for k,v in self:
            if isinstance(k, str):
                new_str += self.cur_level + ' IS ' + k
            else:
                new_str += (self.cur_level + ' IS '
                            + reduce(lambda x,y: x + ' OR ' + y, k))

            if v is not None:
                new_str += ' AND ' + repr(v)
            new_str += ' OR '

        return gsub(' OR $', '', new_str)

    def __str__(self):
        """Gets a condensed label for the MuType."""
        new_str = ''

        for k,v in self:
            if v is None:
                new_str = new_str + k
            else:
                new_str = new_str + k + '-' + str(v)
            new_str = new_str + ', '

        return gsub(', $', '', new_str)

    def raw_key(self):
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
        new_key = {}
        self_dict = dict(self)
        other_dict = dict(other)

        if self.cur_level == other.cur_level:
            for k in (self_dict.keys() - other_dict.keys()):
                new_key.update({(self.cur_level, k): self_dict[k]})
            for k in (other_dict.keys() - self_dict.keys()):
                new_key.update({(self.cur_level, k): other_dict[k]})

            for k in (self_dict.keys() & other_dict.keys()):
                if (self_dict[k] is None) or (other_dict[k] is None):
                    new_key.update({(self.cur_level, k): None})
                else:
                    new_key.update({
                        (self.cur_level, k): self_dict[k] | other_dict[k]})

        else:
            raise HetManMutError("Cannot take the union of two MuTypes with "
                                 "mismatching mutation levels "
                                 + self.cur_level + " and "
                                 + other.cur_level + "!")

        return MuType(new_key)

    def __and__(self, other):
        """Finds the intersection of two MuTypes."""
        if not isinstance(other, MuType):
            return NotImplemented
        new_key = {}
        self_dict = dict(self)
        other_dict = dict(other)

        if self.cur_level == other.cur_level:
            for k in (self_dict.keys() & other_dict.keys()):
                if self_dict[k] is None:
                    new_key.update({(self.cur_level, k): other_dict[k]})
                elif other_dict[k] is None:
                    new_key.update({(self.cur_level, k): self_dict[k]})
                else:
                    new_key.update({
                        (self.cur_level, k): self_dict[k] & other_dict[k]})

        else:
            raise HetManMutError("Cannot take the intersection of two "
                                 "MuTypes with mismatching mutation levels "
                                 + self.cur_level + " and " + other.cur_level + "!")

        return MuType(new_key)

    def __add__(self, other):
        """Returns a set representing the presence of both mutations."""
        if self == other:
            return self
        else:
            return MutSet('AND', self, other)

    def __ge__(self, other):
        """Checks if one MuType is a subset of the other."""
        if not isinstance(other, MuType):
            return NotImplemented
        self_dict = dict(self)
        other_dict = dict(other)

        if self.cur_level == other.cur_level:
            if self_dict.keys() >= other_dict.keys():
                for k in (self_dict.keys() & other_dict.keys()):
                    if self_dict[k] is not None:
                        if other_dict[k] is None:
                            return False
                        elif not (self_dict[k] >= other_dict[k]):
                            return False
                                
            else:
                return False
        else:
            return False

        return True

    def __gt__(self, other):
        """Checks if one MuType is a proper subset of the other."""
        if not isinstance(other, MuType):
            return NotImplemented
        self_dict = dict(self)
        other_dict = dict(other)

        if self.cur_level == other.cur_level:
            if self_dict.keys() > other_dict.keys():
                for k in (self_dict.keys() & other_dict.keys()):
                    if other_dict[k] is None:
                        return False
                    elif self_dict[k] is not None:
                        if not (self_dict[k] > other_dict[k]):
                            return False
            else:
                return False
        else:
            return False

        return True

    def __sub__(self, other):
        """Subtracts one MuType from another."""
        if not isinstance(other, MuType):
            return NotImplemented
        new_key = {}
        self_dict = dict(self)
        other_dict = dict(other)

        if self.cur_level == other.cur_level:
            for k in self_dict.keys():
                if k in other_dict:
                    if other_dict[k] is not None:
                        if self_dict[k] is not None:
                            sub_val = self_dict[k] - other_dict[k]
                            if sub_val is not None:
                                new_key.update({(self.cur_level, k): sub_val})
                        else:
                            new_key.update({(self.cur_level, k): self_dict[k]})
                else:
                    new_key.update({(self.cur_level, k): self_dict[k]})

        else:
            raise HetManMutError("Cannot subtract MuType with mutation level "
                                 + other.cur_level + " from MuType with "
                                 + "mutation level " + self.cur_level + "!")

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

    def get_levels(self):
        """Gets all the levels present in this type and its children."""
        levels = set([self.cur_level])
        for _, v in self:
            if isinstance(v, MuType):
                levels |= set(v.get_levels())
        return levels

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

        if self.cur_level == mtree.cur_level:
            if '_scores' in self.cur_level :
                samps |= set(mtree.child.keys())

            else:
                for nm, mut in mtree:
                    for k, v in self:
                        if k == nm:
                            if isinstance(mut, frozenset):
                                samps |= mut
                            elif isinstance(mut, MuTree):
                                if v is None:
                                    samps |= mut.get_samples()
                                else:
                                    samps |= v.get_samples(mut)
                            else:
                                raise HetManMutError("get_samples error!")

        else:
            for _, mut in mtree:
                if isinstance(mut, MuTree):
                    samps |= self.get_samples(mut)

        return samps

    def get_scores(self, mtree):
        """Gets the mutation scores (i.e. GISTIC) associated with the samples
           within a particular branch or branches of the tree.
        """
        samps = {}

        if self.cur_level == mtree.cur_level:
            if '_scores' in self.cur_level:
                samps = {**samps, **mtree.child}

            else:
                for nm, mut in mtree:
                    for k, v in self:
                        if k == nm:
                            if isinstance(mut, frozenset):
                                samps = {**samps, **{s:1 for s in mut}}
                            elif isinstance(mut, MuTree):
                                if v is None:
                                    samps = {**samps, **mut.get_scores()}
                                else:
                                    samps = {**samps, **v.get_scores(mut)}
                            else:
                                raise HetManMutError("get_scores error!")

        else:
            for _, mut in mtree:
                if isinstance(mut, MuTree):
                    samps = {**samps, **self.get_scores(mut)}

        return samps

    def invert(self, mtree):
        """Returns the mutation types not included in this set of types that
           are also in the given tree.
        """
        new_key = {}
        self_ch = self.raw_key()

        for k in (set(mtree.child.keys()) - set(self_ch.keys())):
            new_key[(self.cur_level, k)] = None
        for k in (set(mtree.child.keys()) & set(self_ch.keys())):
            if self_ch[k] is not None and isinstance(mtree.child[k], MuTree):
                new_key[(self.cur_level, k)] = self_ch[k].invert(mtree.child[k])
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
                mkeys += [{(self.cur_level, i):None} for i in k]
            else:
                mkeys += [{(self.cur_level, i):s} for i in k for s in v.subkeys()]

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
                    {(mtree.levels[0],tuple(mtree.branches_)):None})
        elif len(mtree.levels[1]) == 1:
            new_set = self
        else:
            new_key = {}
            for k,v in list(mtree.child.items()):
                for l,w in list(self.child.items()):
                    if k in l:
                        if w is not None:
                            new_key.update([((mtree.levels[0],k),
                                            w.rationalize(v))])
                        else:
                            new_key.update([((mtree.levels[0],k), None)])
            new_set = MuType(new_key)
            if new_set.child == {frozenset(mtree.branches_): None}:
                if mtree.levels[0]:
                    new_set = None
                else:
                    new_set = MuType(
                        {(mtree.levels[0],tuple(mtree.branches_)):None})
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



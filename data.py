
"""
Hetman (Heterogeneity Manifold)
Classification of mutation sub-types using expression data.
This file contains classes for representing mutation and expression data in
formats that facilitate classification of mutation sub-types.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

import synapseclient
import numpy as np
import pandas as pd
import re
import defunct
import operator

from itertools import combinations as combn
from itertools import groupby, product
from math import log, ceil
from sklearn import model_selection
from classif import _score_auc
from scipy.stats import fisher_exact


# .. directories containing raw -omics data and cross-validation samples ..
_base_dir = '/home/users/grzadkow/compbio/'
_data_dir = _base_dir + 'input-data/ICGC/raw/'
_cv_dir = _base_dir + 'auxiliary/HetMan/cv-samples/'


# .. mutation data types for use with MC3 TCGA data ..
_mc3_levels = {
    'Gene':((np.str_, 16), 0),
    'Conseq':((np.str_, 32), 8),
    'Exon':((np.str_, 8), 38),
    'Protein':((np.str_, 16), 36)
    }


# .. helper functions for reading in genomic data from files downloaded
#    via the ICGC portal and GENCODE ..
def _read_annot(version='v19'):
    """Gets annotation data for protein-coding genes on non-sex
       chromosomes from a Gencode file.

    Parameters
    ----------
    version : str, 'v*', optional (default='v19')
        Which version of the Gencode data to use, must correspond
        to a file in input-data/.
        Default is to use v19, which matches what is used for the data
        available at the ICGC data portal (11/09/2016).

    Returns
    -------
    gene_annot : dict
        Dictionary with keys corresponding to Ensembl gene IDs and values
        consisting of dicts with annotation fields.
    """
    dt = np.dtype(
        [('Chr', 'a64'), ('Type', 'a64'),
         ('Start', 'i4'), ('End', 'i4'), ('Annot', 'a2048')]
        )
    annot = np.loadtxt(
        fname=(_base_dir + 'input-data/gencode.'
               + version + '.annotation.gtf.gz'),
        dtype=dt, usecols=(0,2,3,4,8), delimiter='\t'
        )

    # filter out annotation records that aren't
    # protein-coding genes on non-sex chromosomes
    chroms_use = ['chr' + str(i+1) for i in xrange(22)]
    annot = annot[annot['Type'] == 'gene']
    annot = annot[np.array([chrom in chroms_use for chrom in annot['Chr']])]
    gn_annot = {
        re.sub('\.[0-9]+', '', z['gene_id']).replace('"',
                                                     ''):z
        for z in [dict([['chr', an['Chr']]] +
                       [['Start', an['Start']]] +
                       [['End', an['End']]] +
                       [y for y in [x.split(' ')
                                    for x in an['Annot'].split('; ')]
                        if len(y) == 2])
                  for an in annot]
        if z['gene_type'] == '"protein_coding"'
        }

    for g in gn_annot:
        gn_annot[g]['gene_name'] = gn_annot[g]['gene_name'].replace('"', '')
    return gn_annot


def _read_expr(expr_file):
    """Gets expression data as a matrix from an ICGC tsv.gz file.

    Parameters
    ----------
    expr_file : str
        A file containing expression data.

    Returns
    -------
    expr : array-like, shape (n_samples, n_features)
        An expression matrix with genes as features, in the case of duplicate
        gene names values are averaged.
    """
    expr = pd.read_table(
            expr_file,
            usecols=(4,7,8), header=0,
            names=('Sample', 'Gene', 'FPKM'),
            dtype={'Sample':'a64', 'Gene':'a64', 'FPKM':'f4'}
            )
    expr['Sample'] = [reduce(lambda x,y: x+'-'+y,
                             s.split('-', 3)[:3])
                      for s in expr['Sample']]
    
    # transforms raw long-format expression data into wide-format
    return expr.pivot_table(
        index='Sample', columns='Gene',
        values='FPKM', aggfunc=np.mean
        )


def _read_mut(syn, mut_levels=('Gene','Conseq','Exon')):
    """Reads ICGC mutation data from the MC3 synapse file.

    Parameters
    ----------
    syn : object
        An instance of Synapse that has already been logged into.

    mut_levels : tuple of strs
        A list of mutation levels, must be in the keys of the _mc3_levels
        dict defined above.

    Returns
    -------
    muts : array of shape (n_mutations, mut_levels+1)
        A mutation array, with a row for each mutation appearing in an
        individual sample.
    """
    mc3 = syn.get('syn7824274')
    data_types = [('Sample', np.str_, 32)]
    use_cols = [15]
    for lvl in mut_levels:
        data_types.append(((lvl,) + _mc3_levels[lvl][0]))
        use_cols += [_mc3_levels[lvl][1]]
    muts = np.loadtxt(
        fname=mc3.path, dtype=data_types, skiprows=1,
        delimiter='\t', usecols=use_cols
        )
    muts['Sample'] = [reduce(lambda x,y: x+'-'+y,
                             s.split('-', 3)[:3])
                      for s in muts['Sample']]
    return muts


def _read_mut_icgc(mut_file):
    """Gets mutation data as an numpy array from an ICGC tsv.gz file.
       Deprecated in favour of using MC3 mutation calls.

    Parameters
    ----------
    mut_file : str
        A file containing mutation data.

    Returns
    -------
    mut : array-like, shape (n_mutations,)
        A 1-D array with each entry corresponding to a single mutation
        affecting a single transcript in a single sample.
    """
    mut_dt = np.dtype(
        [('Name', np.str_, 16), ('Sample', np.str_, 16),
         ('Position', np.int), ('Consequence', np.str_, 46),
         ('Protein', np.str_, 16), ('Gene', np.str_, 16),
         ('Transcript', np.str_, 16)]
        )
    return np.loadtxt(
        fname=mut_file,
        dtype=mut_dt, skiprows=1,
        delimiter='\t',
        usecols=(0,1,9,25,26,28,29)
        )


def _read_cnv(cnv_file):
    """Gets copy number variation as an numpy array from an ICGC tsv.gz file.

    Parameters
    ----------
    cnv_file : str
        A file containing copy number variation data.

    Returns
    -------
    cnv_data : array-like, shape (n_cnvs,)
        A 1-D array with each entry corresponding to a single copy number
        variation affecting a single sample.
    """
    return np.loadtxt(
        fname=cnv_file,
        dtype=[('Sample',np.str_,64), ('Mean',np.float),
               ('Chr',np.str_,64), ('Start',np.int), ('End',np.int)],
        skiprows=1, delimiter='\t', usecols=(0,9,11,12,13)
        )


class HetmanDataError(Exception):
    """Class for exceptions thrown by classes in the Hetman data module."""
    pass


class MuTree(object):
    """A class corresponding to a tree representing a hierarchy of
       mutations present in given set of samples.
       
    Parameters
    ----------
    muts : ndarray, shape (n_muts,)
        Input mutation data, each record is a single mutation occurring in a
        sample.
        
    samples : array-like, shape (n_samples,)
        Which samples' mutation data to include in the tree. Note that samples
        without any mutations will not be in the tree regardless.
    
    genes : list
        A list of genes whose mutation data are to be included in the tree.
        i.e. ['TP53', 'ATM']

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

    Attributes
    ----------
    branches_ : set of strs
        The branches at this level of the hierarchy, i.e. the set of genes, set
        of possible consequences, etc.
    """

    def __init__(self,
                 muts, samples, genes,
                 levels=('Gene', 'Conseq', 'Exon')):
        if isinstance(levels[0], str):
            self.levels = ((),levels)
        else:
            self.levels = levels
        self.cur_level = self.levels[1][0]

        # handles the possible mutation hierarchy levels
        null_arr = np.empty(shape=0, dtype=muts.dtype)
        if self.cur_level == 'Gene':
            self.branches_ = set(genes)
            new_muts = {g:muts[muts['Gene'] == g] for g in genes}
            new_samps = {g:(samples & set(new_muts[g]['Sample']))
                         for g in genes}

        elif self.cur_level == 'Conseq':
            new_muts = {}
            new_samps = {}
            self.branches_ = set()
            for s in samples:
                mut_s = muts[muts['Sample'] == s]
                conseqs = set(mut_s['Conseq'])
                for c in conseqs:
                    mut_c = mut_s[mut_s['Conseq'] == c]
                    c_lab = re.sub('_(Del|Ins)', '', c)
                    self.branches_ |= set([c_lab])
                    add_samp = new_samps.get(c_lab, set()) | set([s])
                    new_samps.update({c_lab:add_samp})
                    add_mut = np.concatenate(
                        (new_muts.get(c_lab, null_arr), mut_c))
                    new_muts.update({c_lab:add_mut})

        elif self.cur_level == 'Exon':
            new_samps = {}
            new_muts = {}
            self.branches_ = set()
            for s in samples:
                mut_s = muts[muts['Sample'] == s]
                exons = set(mut_s['Exon'])
                self.branches_ |= exons
                for ex in exons:
                    mut_ex = mut_s[mut_s['Exon'] == ex]
                    add_samp = new_samps.get(ex, set()) | set([s])
                    new_samps.update({ex:add_samp})
                    add_mut = np.concatenate((new_muts.get(ex, null_arr),
                                              mut_ex))
                    new_muts.update({ex:add_mut})

        elif self.cur_level == 'Protein':
            new_samps = {}
            new_muts = {}
            self.branches_ = set()
            for s in samples:
                mut_s = muts[muts['Sample'] == s]
                proteins = set(mut_s['Protein'])
                self.branches_ |= proteins
                for p in proteins:
                    mut_p = mut_s[mut_s['Protein'] == p]
                    add_samp = new_samps.get(p, set()) | set([s])
                    new_samps.update({p:add_samp})
                    add_mut = np.concatenate((new_muts.get(p, null_arr),
                                              mut_p))
                    new_muts.update({p:add_mut})

        # unless we have reached the final level, recurse down
        # to the next level
        if len(self.levels[1]) > 1:
            self.child = {g:MuTree(
                muts=new_muts[g],
                samples=(samples & set(new_samps[g])), genes=genes,
                levels=(self.levels[0] + (self.cur_level,),
                        self.levels[1][1:])
                ) for g in new_muts.keys()}
        else:
            self.child = {g:frozenset(tuple(new_samps[g]))
                          for g in new_muts.keys()}

    def __str__(self):
        """Printing a MuTree shows each of the branches of the tree and
           the samples at the end of each branch."""
        new_str = self.cur_level
        for k,v in self.child.items():
            new_str = new_str + ' IS ' + k
            if isinstance(v, MuTree):
                new_str = (new_str + ' AND '
                           + '\n' + '\t'*(len(self.levels[0])+1) + str(v))
            else:
                if len(v) > 15:
                    new_str = new_str + ': (' + str(len(v)) + ' samples)'
                else:
                    new_str = (new_str + ': '
                               + reduce(lambda x,y: x + ',' + y, tuple(v)))
            new_str = new_str + '\n' + '\t'*len(self.levels[0])
        new_str = re.sub('\n$', '', new_str)
        return new_str

    def __len__(self):
        """The length of a MuTree is the number
           of unique samples it contains."""
        return len(self.get_samples())

    def get_samples(self):
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
        samps = set()
        for v in self.child.values():
            if isinstance(v, MuTree):
                samps |= v.get_samples()
            else:
                samps |= v
        return samps

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

    def add_cnvs(self, mut_gene, cnvs):
        """Adds a list of copy number variations for the given gene to the
           mutation hierarchy. CNVs are treated as Gain/Loss entries on the
           Consequence level.

        Parameters
        ----------
        mut_gene : str
            One of the genes in the mutation tree. An error will be raised
            otherwise.

        cnvs : dict
            A dictionary with "Gain" and/or "Loss" as keys and individual
            samples as values.
        """
        if self.cur_level != 'Gene':
            raise HetmanDataError("CNVs can only be added to the "
                                  "<Gene> level of a mutation tree.")
        if not mut_gene in self.child.keys():
            raise HetmanDataError("CNVs can only be added to a gene "
                                  "already in the tree.")
        for k,v in cnvs.items():
            if v:
                self.child[mut_gene].child['Loss'] = frozenset(v)

    def allkey(self, levels=None):
        """Gets the key corresponding to the MuType that contains all of the
           branches of the tree. A convenience function that makes it easier to
           list all of the possible branches present in the tree, and to
           instantiate MuType objects that correspond to all of the possible
           mutation types.

        Parameters
        ----------
        levels : tuple
            A list of levels corresponding to how far the output MuType should
            recurse.

        Returns
        -------
        new_key : dict
            A MuType key which can be used to instantiate
            a MuType object (see below).
        """
        if levels is None:
            levels = self.levels[1]
        return {
            (self.cur_level, k):(v.allkey(levels)
                                 if (isinstance(v, MuTree)
                                     and self.cur_level != levels[-1])
                                 else None)
            for k,v in self.child.items()
            }

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
            levels = self.levels[1]
        mtypes = []
        if self.cur_level != levels[-1]:
            for k,v in self.child.items():
                for l,w in mtype.child.items():
                    if k in l:
                        if isinstance(v, MuTree):
                            mtypes += [MuType({(self.cur_level, k):s})
                                      for s in v.subsets(w, levels)]
                        else:
                            mtypes += [MuType({(self.cur_level, k):None})
                                      for k in (set(self.child.keys())
                                                & reduce(lambda x,y: x|y,
                                                         mtype.child.keys()))]
        else:
            mtypes += [MuType({(self.cur_level, k):None})
                      for k in (set(self.child.keys())
                                & reduce(lambda x,y: x|y,
                                         mtype.child.keys()))]
        return mtypes

    def direct_subsets(self, mtype, branches=None):
        """Gets all of the MuTypes corresponding to direct descendants
           of the given branches of the given mutation set.

        Parameters
        ----------
        mtype : MuType
            A set of mutations whose direct descandants are to be obtained.

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
            for k,v in self.child.items():
                for l,w in mtype.child.items():
                    if k in l:
                        if w is not None:
                            mtypes += [MuType({(self.cur_level, k):s})
                                      for s in v.direct_subsets(w, branches)]
                        elif branches is None or k in branches:
                            if isinstance(v, MuTree):
                                mtypes += [
                                    MuType({(self.cur_level, k):
                                            MuType({(v.cur_level, x):None})})
                                    for x in v.child.keys()
                                    ]
                            else:
                                mtypes += [MuType({(self.cur_level, k):None})]
        else:
            if branches is None:
                branches = self.branches_
            mtypes += [
                MuType({(self.cur_level, k):None})
                for k in (set(self.child.keys())
                          & reduce(lambda x,y: x|y, mtype.child.keys())
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

    def partitions(self, mtype, prop_use=0.1, max_part=25):
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
        psets : list
            A list of MuTypes that satisfy the given criteria.
        """
        # Get the list of possible branches to use for constructing subsets,
        # merge the smallest branches if the number of branches is too high.
        sub_groups = [MuType(m) for m in mtype.subkeys()]
        sub_list = [self.direct_subsets(m) for m in sub_groups]
        for i in range(len(sub_list)):
            if not sub_list[i]:
                sub_list[i] = [sub_groups[i]]
        sub_lens = [len(x) for x in sub_list]
        if all([x == 1 for x in sub_lens]):
            sub_groups = [mtype]
            sub_list = [reduce(lambda x,y: x+y, sub_list)]
            sub_lens = [len(sub_groups)]
        if reduce(lambda x,y: (2**x - 1)*(2**y - 1), sub_lens + [1]) > max_part:
            sub_sizes = [len(w.get_samples(self)) for m in sub_groups]
            max_subs = [max_part ** (float(x)/sum(sub_sizes)) for x in sub_sizes]
            for i in range(len(sub_list)):
                if max_subs[i] > (2**sub_lens[i] - 1):
                    for j in range(len(sub_list))[(i+1):]:
                        max_subs[j] = (
                            (max_subs[j] * (max_subs[i] / (2**sub_lens[i]-1)))
                            ** (1.0 / (len(sub_list)-i-1)))
                    max_subs[i] = (2**sub_lens[i] - 1)
                sub_indx = sorted(
                    [(x, float(len(x.get_samples(self)))) for x in sub_list[i]],
                    key=lambda y: y[1],
                    reverse=True
                    )
                while len(sub_indx) > max(ceil(log(max_subs[i], 2)), 1):
                    new_sub = sub_indx[-2][0] | sub_indx[-1][0]
                    sub_indx = sub_indx[:-2]
                    new_indx = (new_sub, float(len(new_sub.get_samples(self))))
                    sort_indx = sum([new_indx[1] < v for _,v in sub_indx])
                    sub_indx.insert(sort_indx, new_indx)
                sub_list[i] = [x[0] for x in sub_indx]

        # Get all possible combinations of the branches that satisfy the sample
        # size criteria. Tighten the criteria and merge the smallest branches
        # and redo if the number of combinations turns out to be too high.
        tot_samps = len(mtype.get_samples(self))
        psets = []
        for csizes in product(*[range(1, len(x)+1) for x in sub_list]):
            for set_combn in product(
                *[combn(sl, csize) for sl,csize in zip(sub_list,csizes)]):
                set_comps = map(lambda x: reduce(lambda y,z: y|z, x),
                                set_combn)
                new_set = reduce(lambda x,y: x | y, set_comps)
                set_size = len(new_set.get_samples(self))
                if (set_size >= max((tot_samps*prop_use), 10)
                    and set_size <= (tot_samps*(1-prop_use))):
                    psets += [self.rationalize(new_set)]
        return list(set(psets))

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

    def rationalize(self, mtype):
        """Simplifies the structure of MuType if it finds that some of its
           branches correspond to the full set of branches possible in the
           mutation hierarchy.
        """
        if mtype.child == {frozenset(self.branches_): None}:
            if self.levels[0]:
                new_set = None
            else:
                new_set = MuType(
                    {(self.cur_level,tuple(self.branches_)):None})
        elif len(self.levels[1]) == 1:
            new_set = mtype
        else:
            new_key = {}
            for k,v in self.child.items():
                for l,w in mtype.child.items():
                    if k in l:
                        if w is not None:
                            new_key.update([((self.cur_level,k),
                                            v.rationalize(w))])
                        else:
                            new_key.update([((self.cur_level,k), None)])
            new_set = MuType(new_key)
            if new_set.child == {frozenset(self.branches_): None}:
                if self.levels[0]:
                    new_set = None
                else:
                    new_set = MuType(
                        {(self.cur_level,tuple(self.branches_)):None})
        return new_set


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
        level = set(k for k,_ in set_key.keys())
        if len(level) > 1:
            raise HetmanDataError(
                "improperly defined MuType key (multiple mutation levels)")
        self.level_ = tuple(level)[0]

        # gets the subsets of mutations defined at this level, and
        # their further subdivisions if they exist
        membs = [(k,) if isinstance(k, str) else k for _,k in set_key.keys()]
        children = dict(
            tuple((v, ch)) if isinstance(ch, MuType) or ch is None else
            tuple((v, MuType(ch)))
            for v,ch in zip(membs, set_key.values())
            )

        # merges subsets at this level if their children are the same, i.e.
        # missense:None, frameshift:None => (missense,frameshift):None
        uniq_ch = set(children.values())
        self.child = {frozenset(i for j in
                                [k for k,v in children.items() if v == ch]
                                for i in j):ch for ch in uniq_ch}

    def __eq__(self, other):
        """Two MuTypes are equal if and only if they have the same set
           of children MuTypes for the same subsets."""
        if isinstance(self, MuType) ^ isinstance(other, MuType):
            eq = False
        elif self.level_ != other.level_:
            raise HetmanDataError("can't compare MuTypes of different levels")
        else:
            eq = (self.child == other.child)
        return eq

    def __str__(self):
        """Printing a MuType shows the hierarchy of mutation
           properties contained within it."""
        new_str = ''
        if self.level_ == 'Gene':
            new_str += 'a mutation where '
        for k,v in self.child.items():
            new_str += (self.level_ + ' IS '
                        + reduce(lambda x,y: x + ' OR ' + y, k))
            if v is not None:
                new_str += ' AND ' + '\n\t' + str(v)
            new_str += '\nOR '
        return re.sub('\nOR $', '', new_str)

    def _raw_key(self):
        "Returns the expanded key of a MuType."
        rmembs = reduce(lambda x,y: x|y, self.child.keys())
        return {memb:reduce(lambda x,y: x|y,
                            [v for k,v in self.child.items() if memb in k])
                for memb in rmembs}

    def __or__(self, other):
        """Returns the union of two MuTypes."""
        if not isinstance(other, MuType):
            return NotImplemented
        if self.level_ != other.level_:
            raise HetmanDataError("can't join MuTypes of different levels")
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
        if not isinstance(other, MuType):
            return NotImplemented
        """Finds the intersection of two MuTypes."""
        if self.level_ != other.level_:
            raise HetmanDataError('mismatching MuType levels')
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
        self_keys = reduce(lambda x,y: x|y, self.child.keys())
        other_keys = reduce(lambda x,y: x|y, other.child.keys())
        self_keys = {x:reduce(lambda x,y: x|y,
                                [v for k,v in self.child.items() if x in k])
                       for x in self_keys}
        other_keys = {x:reduce(lambda x,y: x|y,
                                 [v for k,v in other.child.items() if x in k])
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
        self_keys = reduce(lambda x,y: x|y, self.child.keys())
        other_keys = reduce(lambda x,y: x|y, other.child.keys())
        self_keys = {x:reduce(lambda x,y: x|y,
                                [v for k,v in self.child.items() if x in k])
                       for x in self_keys}
        other_keys = {x:reduce(lambda x,y: x|y,
                                 [v for k,v in other.child.items() if x in k])
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
        self_keys = reduce(lambda x,y: x|y, self.child.keys())
        other_keys = reduce(lambda x,y: x|y, other.child.keys())
        self_keys = {x:reduce(lambda x,y: x|y,
                                [v for k,v in self.child.items() if x in k])
                       for x in self_keys}
        other_keys = {x:reduce(lambda x,y: x|y,
                                 [v for k,v in other.child.items() if x in k])
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
        for k,v in self.child.items():
            value += (eval(hex((long(value) * 1000007) & 0xFFFFFFFFL)[:-1])
                      ^ hash(k) ^ hash(v))
            value ^= len(self.child)
        if value == -1:
            value = -2
        return value

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
        samps = set()
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

    def subkeys(self):
        """Gets all of the possible subsets of this MuType that contain
           exactly one of the leaf properties."""
        mkeys = []
        for k,v in self.child.items():
            if v is None:
                mkeys += [{(self.level_, i):None} for i in k]
            else:
                mkeys += [{(self.level_, i):s} for i in k for s in v.subkeys()]

        return mkeys


class MutExpr(object):
    """A class corresponding to expression and mutation data for an ICGC
       project.

    Parameters
    ----------
    syn : synapseclient object
        A logged-into instance of the synapseclient.Synapse() class.

    project : str
        An ICGC/TCGA project, i.e. BRCA-US or LGG-US

    mut_genes : list of strs
        A list of genes whose mutations we want to consider,
        i.e. ['TP53','KRAS'].

    cv_info : dict, optional
        A dictionary with a Label field (i.e. 'two-thirds')
        and a Sample field (i.e. 45) which specifes which cross-validation
        sample this object will use for training and testing expression-based
        classifiers of mutation status.

    mut_levels : tuple, optional
        A list of mutation levels we want to consider, see
        MuTree and MuType above.

    load_cnv : bool, optional
        Whether CNV data should loaded, the default is to omit it.

    Attributes
    ----------
    project_ : str
        The ICGC project whose data is stored in this object.

    cv_index_ : int
        Which cross-validation sample this object uses.

    train_expr_ : array-like, shape=(n_samples,n_tr_features)
        The subset of expression data used for training of classifiers.

    test_expr_ : array-like, shape=(n_samples,n_tst_features)
        The subset of expression data used for testing of classifiers.

    train_mut_ : MuTree
        Hierarchy of mutations present in the training samples.

    test_mut_ : MuTree
        Hierarchy of mutations present in the testing samples.

    cnv_ : dict
        Mean CNV scores for genes whose mutations we want to consider.
    """

    def __init__(self, syn,
                 project, mut_genes, cv_info=None,
                 mut_levels = ('Gene', 'Conseq', 'Exon'), load_cnv=False):

        # loads gene expression and annotation data, mutation data
        self.project_ = project
        annot = _read_annot()
        expr = _read_expr(_data_dir + project + '/exp_seq.tsv.gz')
        muts = _read_mut(syn, mut_levels)

        # filters out genes that are not expressed in any samples, don't have
        # any variation across the samples, are not included in the
        # annotation data, or are not in the mutation dataset
        expr = expr.loc[:, expr.apply(
            lambda x: np.mean(x) > 0 and np.var(x) > 0,
            axis=0)]
        annot = {g:a for g,a in annot.items()
                 if a['gene_name'] in expr.columns}
        annot_genes = [a['gene_name'] for g,a in annot.items()]
        expr = expr.loc[:, annot_genes]
        annot_data = {mut_g:{'ID':g, 'Chr':a['chr'],
                             'Start':a['Start'], 'End':a['End']}
                      for g,a in annot.items() for mut_g in mut_genes
                      if a['gene_name'] == mut_g}
        annot_ids = {k:v['ID'] for k,v in annot_data.items()}
        self.annot = annot
        samples = frozenset(set(muts['Sample']) & set(expr.index))

        # if cross-validation info is specified, get list of samples used for
        # training and testing after filtering out those for which we don't
        # have expression and mutation data
        if cv_info is not None:
            cv_file = _cv_dir + project + '_' + cv_info['Label'] + '.txt'
            with open(cv_file, 'r') as f:
                cv_samples = f.readline().split('\t')
                for i in range(cv_info['Sample']):
                    train_indx = f.readline()

            train_indx = train_indx.split('\t')[1:]
            cv_samples[-1] = cv_samples[-1].replace('\n', '')
            self.train_samps_ = frozenset(
                set(key for key,val
                    in zip(samples, train_indx)
                    if val == 'TRUE')
                & samples)

            self.train_expr_ = expr.loc[self.train_samps_, :]
            self.test_expr_ = expr.loc[samples - self.train_samps_, :]
            self.train_mut_ = MuTree(
                muts=muts, samples=self.train_samps_,
                genes=mut_genes, levels=mut_levels
                )
            self.test_mut_ = MuTree(
                muts=muts, samples=(samples - self.train_samps_),
                genes=mut_genes, levels=mut_levels
                )
            self.cv_index_ = cv_info['Sample'] ** 2

        # if no cross-validation info is specified, treat
        # all samples as training
        else:
            self.train_samps_ = None
            self.train_expr_ = _norm_expr(expr.loc[samples, :])
            self.train_mut_ = MuTree(
                muts=muts, samples=samples,
                genes=mut_genes, levels=mut_levels
                )
            self.cv_index_ = 0

        # if applicable, get the samples' CNV scores for the genes whose
        # mutations we want to consider
        if load_cnv:
            cnv_data = _read_cnv(
                _data_dir + project +
                '/copy_number_somatic_mutation.tsv.gz'
                )
            cnv_stats = {}
            for g,an in annot_data.items():
                gene_cnv = cnv_data[cnv_data['Chr'] == re.sub('chr','',
                                                              an['Chr'])]
                gene_cnv = gene_cnv[gene_cnv['Start'] <= an['Start']]
                gene_cnv = gene_cnv[gene_cnv['End'] >= an['End']]
                gene_cnv = np.sort(gene_cnv, order='Sample')
                gene_cnv = {s:np.mean([y[1] for y in x]) for s,x in
                            groupby(gene_cnv, lambda x: x['Sample'])}
                cnv_stats[g] = {s:gene_cnv[s] if s in gene_cnv else 0
                                for s in samples}
            self.cnv_ = cnv_stats

    def add_cnv_loss(self):
        """Adds CNV loss inferred using a Gaussian mixture model
           to the mutation tree.
        """
        for gene in self.cnv_.keys():
            cnv_def = defunct.Defunct(self, gene)
            loss_samps = cnv_def.get_loss()
            self.train_mut_.add_cnvs(cnv_def.mut_gene_,
                                     {'Loss': loss_samps[0]})
            self.test_mut_.add_cnvs(cnv_def.mut_gene_,
                                    {'Loss': loss_samps[1]})

    def mutex_test(self, mtype1, mtype2):
        """Checks the mutual exclusivity of two mutation types in the
           training data using a one-sided Fisher's exact test.

        Parameters
        ----------
        mtype1,mtype2 : MuTypes
            The mutation types to be compared.

        Returns
        -------
        pval : float
            The p-value given by the test.
        """
        samps1 = self.train_mut_.get_samples(mtype1)
        samps2 = self.train_mut_.get_samples(mtype2)
        if not samps1 or not samps2:
            raise HetmanDataError("Both sets must be non-empty!")
        all_samps = set(self.train_expr_.index)
        both_samps = samps1 & samps2
        _,pval = fisher_exact(
            [[len(all_samps - (samps1 | samps2)),
              len(samps1 - both_samps)],
             [len(samps2 - both_samps),
              len(both_samps)]],
            alternative='less')
        return pval

    def training(self, mtype=None, gene_list=None):
        """Gets the expression data and the mutation status corresponding
           to a given mutation sub-type for the training samples in this
           dataset.

        Parameters
        ----------
        mtype : MuType, optional
            A mutation sub-type(s).
            The default is to use all available mutations.

        Returns
        -------
        expr : array-like, shape(n_tr_samples,n_features)
            The expression data for training samples.

        mut : list of bools, shape(n_tr_samples,)
            Mutation status for the training samples, True iff a sample has a
            mutation in the given set.

        cv : tuple
            A list of internal cross-validation splits to be used for
            classifier tuning, model selection, etc.
        """
        if gene_list is None:
            gene_list = self.train_expr_.columns
        mut_status = self.train_mut_.status(self.train_expr_.index, mtype)
        return (self.train_expr_.loc[:,gene_list],
                mut_status,
                [(x,y) for x,y
                 in model_selection.StratifiedShuffleSplit(
                     n_splits = 100, test_size = 0.2,
                     random_state=self.cv_index_
                 ).split(self.train_expr_, mut_status)])

    def testing(self, mtype=None, gene_list=None):
        """Gets the expression data and the mutation status corresponding
           to a given mutation sub-type for the testing samples in this
           dataset.

        Parameters
        ----------
        mtype : MuType, optional
            A mutation sub-type(s).
            The default is to use all available mutations.

        Returns
        -------
        expr : array-like, shape(n_tst_samples,n_features)
            The expression data for testing samples.

        mut : list of bools, shape(n_tst_samples,)
            Mutation status for the testing samples, True iff a sample has a
            mutation in the given set.
        """
        if self.train_samps_ is None:
            raise HetmanError("No testing set defined!")
        if gene_list is None:
            gene_list = self.test_expr_.columns
        mut_status = self.test_mut_.status(self.test_expr_.index, mtype)
        return (self.test_expr_.loc[:,gene_list],
                mut_status)

    def get_cnv(self, samples=None):
        """Gets the CNV data for the given samples if it is available."""

        if hasattr(self, 'cnv_'):
            cnv = {g:{s:self.cnv_[g][s] for s in samples}
                   for g in self.cnv_.keys()}
        else:
            cnv = None
        return cnv

    def test_classif_cv(self,
                        classif, mtype=None,
                        gene_list=None, exclude_samps=None,
                        test_indx=range(20), tune_indx=None,
                        final_fit=False, verbose=False):
        """Test a classifier using tuning and cross-validation
           within the training samples of this dataset.

        Parameters
        ----------
        classif : UniClassifier
            The classifier to test.

        mtype : MuType, optional
            The mutation sub-type to test the classifier on.
            Default is to use all of the mutations available.

        test_indx : list of ints, optional
            Which of the internal cross-validation samples to use for testing
            classifier performance.

        tune_indx : list of ints, optional
            Which of the internal cross-validation samples to use for tuning
            the hyper-parameters of the given classifier.
            Default is to not do any tuning and thus use the default
            hyper-parameter settings.

        final_fit : boolean
            Whether or not to fit the given classifier to all of the training
            data after tuning and testing is complete. Useful if, for
            instance, we want to learn about the coefficients of this
            classifier when predicting the given set of mutations.

        verbose : boolean
            Whether or not the classifier should print information about the
            optimal hyper-parameters found during tuning.

        Returns
        -------
        P : float
            The 1st quartile of tuned classifier performance across the
            cross-validation samples. Used instead of the mean of performance
            to take into account performance variation for "hard" samples.

            Performance is measured using the area under the receiver operator
            curve metric.
        """
        train_expr,train_mut,train_cv = self.training(mtype, gene_list)
        if exclude_samps is not None:
            use_samps = [s for s,m in zip(train_expr.index,train_mut)
                         if s not in exclude_samps or m]
            use_indx = set([i for i,s in enumerate(train_expr.index)
                        if s in use_samps])
            train_cv = [(np.array(list(set(tr) & use_indx)),
                         np.array(list(set(tst) & use_indx)))
                         for tr,tst in train_cv]
        test_cvs = [x for i,x in enumerate(train_cv)
                    if i in test_indx]
        if tune_indx is not None:
            tune_cvs = [x for i,x in enumerate(train_cv)
                        if i in tune_indx]
            classif.tune(expr=train_expr, mut=train_mut,
                         cv_samples=tune_cvs, test_count='auto',
                         verbose=verbose)

        perf = np.percentile(model_selection.cross_val_score(
            estimator=classif, X=train_expr, y=train_mut,
            scoring=_score_auc, cv=test_cvs, n_jobs=-1
            ), 25)
        if final_fit:
            if exclude_samps is not None:
                train_mut = [m for s,m in zip(train_expr.index,train_mut)
                             if s in use_samps]
                train_expr = train_expr.loc[use_samps,:]
            classif.fit(X=train_expr, y=train_mut)
        return perf

    def test_classif_full(self, classif, tune_indx=range(5), mtype=None):
        """Test a classifier using by tuning within the training samples,
           training on all of them, and then testing on the testing samples.

        Parameters
        ----------
        classif : MutClassifier
            The classifier to test.

        mtype : MuType, optional
            The mutation sub-type to test the classifier on.
            Default is to use all of the mutations available.

        tune_indx : list of ints, optional
            Which of the internal cross-validation samples to use for tuning
            the hyper-parameters of the given classifier.

        Returns
        -------
        P : float
            Performance of the classifier on the testing samples as measured
            using the AUC ROC metric.
        """
        train_expr,train_mut,train_cv = self.training(mtype)
        test_expr,test_mut = self.testing(mtype)
        if tune_indx is not None:
            tune_cvs = [x for i,x in enumerate(train_cv)
                        if i in tune_indx]
            classif.tune(train_expr, train_mut, tune_cvs)
        classif.fit(train_expr, train_mut)
        return _score_auc(classif, test_expr, test_mut)


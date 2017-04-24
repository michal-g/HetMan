
"""
HetMan (Heterogeneity Manifold)
Classification of mutation sub-types using expression data.
This file contains classes that optimize over mutation subset partitions.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

import dill as pickle
import numpy as np
import pulp
from itertools import product, combinations, takewhile


class LossOptimError(Exception):
    """Class for exceptions thrown by the Hetman Loss Optimizer module."""
    pass


class ClassTest(object):

    def __init__(self, cdata, classif, mtype,
                 test_indx=range(16), tune_indx=range(60,65)):
        cdata.add_cnv_loss()
        self.data = cdata
        self.classif = classif
        self.test_indx = test_indx
        self.tune_indx = tune_indx
        mut_chrs = [v['chr'] for k,v in cdata.annot.items()
                    if v['gene_name'] in cdata.train_mut_.child.keys()]
        self.loss_genes = list(set([v['gene_name'] for v in cdata.annot.values()
                           if v['chr'] not in mut_chrs]))

    def run_test(self):
        return test_score
    

class CombSingle(object):
    """A class for finding optimal subsets of mutations for use in
       classification for a single gene and a single classifier.

    Parameters
    ----------
    cdata : data.MutExpr object
        Expression and mutation data from an ICGC project.

    classif : classif.UniClassifier class
        A classification class for use with single datasets.

    test_indx : tuple of ints
        A list of cross-validation indices to use for testing subsets
        within the training dataset.

    tune_indx : tuple of ints
        A list of cross-validation indices to use for tuning classifier
        hyper-parameters within the training dataset.

    Attributes
    ----------
    """

    def __init__(self,
                 cdata, classif=classif.Lasso,
                 test_indx=range(16), tune_indx=range(70,75)):
        # checks that the given MutExpr object has a valid
        # collection of mutation data
        if not hasattr(cdata, 'cnv_'):
            raise LossOptimError(
                "Given MutExpr object must include CNV status.")

        # infers CNV loss, adds attributes
        cdata.add_cnv_loss()
        self.data = cdata
        self.mut_samples = cdata.train_mut_.get_samples()
        self.classif = classif
        self.test_indx = test_indx
        self.tune_indx = tune_indx

        # gets list of genes included in each dataset, finds those not on
        # the same chromosome as the loss gene
        mut_chrs = [v['chr'] for k,v in cdata.annot.items()
                    if v['gene_name'] in cdata.train_mut_.child.keys()]
        self.loss_genes = list(set([v['gene_name'] for v in cdata.annot.values()
                           if v['chr'] not in mut_chrs]))
        self.hist = {}

    def __repr__(self):
        """Displays current state of the optimizer."""
        return reduce(
            lambda x,y: x + y,
            [str(k) + ': ' + str(round(v,4)) + '\n'
             for k,v in self.hist.items()]
            )

    def __str__(self):
        """Displays current state of the optimizer."""
        out_str = ('All mutations AUC: '
                   + str(round(self.hist[self.full_set], 4))
                   + '\n---------------------\n')
        memb = self.best_comb()
        memb_score = self.score_comb(memb)
        out_str += ('Optimal partition has score '
                    + str(round(memb_score, 4)) + ' and consists of: \n')
        i = 1
        for m in self.hist:
            if m in memb and memb[m].value() == 1.0:
                out_str += (
                    str(i) + ') ' + str(m)
                    + ' (' + str(round(self.hist[m], 4)) + ', '
                    + str(len(m.get_samples(self.data.train_mut_)))
                    + ')' + '\n'
                    )
                i += 1
        return out_str

    def test_mtypes(self, mtypes):
        """Tests the performance of the classifier on the given
           mutation subsets.
        """
        print "Testing " + str(len(mtypes)) + " sets..."
        best_score = 0
        for mtype in mtypes:
            if mtype not in self.hist:
                self.hist[mtype] = self.data.test_classif_cv(
                    classif=self.classif(
                        mut_genes=self.data.train_mut_.child.keys()),
                    mtype=mtype, gene_list=self.loss_genes,
                    test_indx=self.test_indx, tune_indx=self.tune_indx,
                    final_fit=False
                    )
            if self.hist[mtype] > best_score:
                best_set = mtype
        return best_set

    def best_comb(self):
        sets_use = [m for m in self.hist.keys() if m != self.full_set]
        memb = pulp.LpVariable.dicts('memb', sets_use,
                                     lowBound=0, upBound=1,
                                     cat=pulp.LpInteger)
        memb_mat = [self.data.train_mut_.status(self.mut_samples, m)
                    for m in memb.keys()]
        perf_mat = [[x*(self.hist[m]-0.5) for x in mat]
                    for mat,m in zip(memb_mat,memb.keys())]
        mtype_model = pulp.LpProblem("Mutation Set Model", pulp.LpMaximize)
        mtype_model += pulp.lpSum([
            sum([memb[x]*m for x,m in zip(memb.keys(),
                                          [y[s] for y in perf_mat])])
            for s in xrange(len(self.mut_samples))])
        for s in xrange(len(self.mut_samples)):
            mtype_model += (
                pulp.lpSum([memb[x]*m for x,m in zip(memb.keys(),
                                              [y[s] for y in memb_mat])])
                <= 1, "Should_include_%s"%s)
        mtype_model.solve()
        return memb

    def score_comb(self, memb):
        samp_scores = {s:0.5 for s in self.mut_samples}
        for m in self.hist:
            if m in memb and memb[m].value() == 1.0:
                for s in m.get_samples(self.data.train_mut_):
                    samp_scores[s] = self.hist[m]
        return np.mean(samp_scores.values())

    def step(self):
        """Carries out a step of the algorithm that searches over potential
           mutation subsets to test, prune, and merge.
        """
        if not self.hist:
            self.gene = self.data.train_mut_.child.keys()
            self.full_set = data.MuType({('Gene', tuple(self.gene)):None})
            self.base_mtypes = self.data.train_mut_.direct_subsets(
                mtype=self.full_set)
            test_mtypes = filter(
                lambda x: len(x.get_samples(self.data.train_mut_)) >= 10,
                self.base_mtypes)
            self.test_mtypes(test_mtypes + [self.full_set])
            for m in list(set(self.base_mtypes) - set(test_mtypes)):
                self.hist[m] = 0
            self.goal_perf = self.hist[self.full_set] ** 0.5
            self.cur_sets = test_mtypes
            self.next_sets = []
            self.mode = 'Prune'
            return True

        elif self.mode == 'Prune':
            new_sets = []
            found_new = False
            for cur_set in self.cur_sets:
                if isinstance(cur_set, data.MuType):
                    test_mtypes = cur_set.prune(
                        mtree=self.data.train_mut_,
                        min_prop=2.0/3, max_part=15, min_size=10)
                    best_set = self.test_mtypes(test_mtypes)
                    if self.hist[best_set] > self.hist[cur_set]:
                        new_sets += [best_set]
                        found_new = True
                    else:
                        new_sets += [cur_set]
                else:
                    new_sets += [cur_set]
            self.cur_sets = new_sets
            self.mode = 'Merge'
            return found_new

        elif self.mode == 'Merge':
            new_sets = []
            test_sets = []
            found_new = False
            for m in self.base_mtypes:
                if all([(m & c) is None for c in self.cur_sets]):
                    self.cur_sets += [m]
            for kc in combinations(self.cur_sets, 2):
                new_set = kc[0] | kc[1]
                if len(new_set.get_samples(self.data.train_mut_)) >= 12:
                    test_sets += [
                        (new_set.rationalize(self.data.train_mut_),
                        kc[0], kc[1])]
            best_set = self.test_mtypes([x[0] for x in test_sets])
            for merge_set,s1,s2 in test_sets:
                if (self.hist[merge_set]
                    > max(self.hist[s1], self.hist[s2])):
                    new_sets += [merge_set]
                    found_new = True
            self.cur_sets = new_sets
            self.mode = 'Prune'
            return found_new


class CombSimple(object):

    def __init__(self,
                 cdata, classif=classif.LogReg,
                 test_indx=range(32), tune_indx=(60,70,80)):
        # checks that the given MutExpr object has a valid
        # collection of mutation data
        if not hasattr(cdata, 'cnv_'):
            raise LossOptimError(
                "Given MutExpr object must include CNV status.")

        # infers CNV loss, adds attributes
        cdata.add_cnv_loss()
        self.data = cdata
        self.mut_samples = cdata.train_mut_.get_samples()
        self.classif = classif
        self.test_indx = test_indx
        self.tune_indx = tune_indx

        # gets list of genes included in each dataset, finds those not on
        # the same chromosome as the loss gene
        mut_chrs = [v['chr'] for k,v in cdata.annot.items()
                    if v['gene_name'] in cdata.train_mut_.child.keys()]
        self.expr_genes = [v['gene_name'] for v in cdata.annot.values()]
        self.loss_genes = [v['gene_name'] for v in cdata.annot.values()
                           if v['chr'] not in mut_chrs]
        self.phist = {}
        self.fhist = {}

    def __str__(self):
        """Displays current state of the optimizer."""
        out_str = "Current set:\n"
        out_str += str(self.cur_set) + "\n----------------\n"
        if self.fhist:
            out_str += reduce(
                lambda x,y: x+y,
                [str(k) + ': '
                 + str(round(self.fhist[k],4)) + ' | '
                 + str(round(self.phist[k],4)) + '\n'
                 for k in self.fhist])
        return out_str

    def test_mtypes(self, mtypes):
        print "Testing " + str(len(mtypes)) + " sets..."
        for mtype in mtypes:
            if mtype not in self.fhist:
                full_perf = self.data.test_classif_cv(
                    classif=self.classif(), mtype=mtype,
                    gene_list=self.loss_genes, exclude_samps=None,
                    test_indx=self.test_indx, tune_indx=self.tune_indx,
                    final_fit=True
                    )
                part_perf = self.data.test_classif_cv(
                    classif=self.classif(), mtype=mtype,
                    gene_list=self.loss_genes, exclude_samps=self.mut_samples,
                    test_indx=self.test_indx, tune_indx=self.tune_indx,
                    final_fit=True
                    )
                self.fhist[mtype] = full_perf
                self.phist[mtype] = part_perf

    def step(self):
        if not self.fhist:
            self.genes = self.data.train_mut_.child.keys()
            self.full_set = data.MuType({('Gene', tuple(self.genes)):None})
            self.cur_set = self.full_set
            gene_mtypes = [data.MuType({('Gene',g):None}) for g in self.genes]
            test_mtypes = []
            for g in self.genes:
                test_mtypes += self.data.train_mut_.partitions(
                    mtype=data.MuType({('Gene',g):None}),
                    prop_use=0.1, max_part=12)
            self.test_mtypes(test_mtypes + gene_mtypes + [self.full_set])
            self.goal_perf = max([y[1] for y in filter(
                lambda x: x[0] in gene_mtypes or x[0] == self.full_set,
                self.fhist.items())]) ** 0.5
            self.mode = 'Gene'

        elif self.mode == 'Gene':
            self.chosen_sets = set()
            self.leftover = set()
            self.repl = set()
            self.prun = set()
            self.merg = set()
            self.disc = set()
            for g in self.genes:
                best_set = None
                gene_mtype = data.MuType({('Gene',g):None})
                for mtype in self.fhist:
                    if mtype < gene_mtype:
                        if self.fhist[mtype] > self.fhist[gene_mtype]:
                            if self.phist[mtype] > self.phist[gene_mtype]:
                                if best_set is None:
                                    best_set = mtype
                                elif self.fhist[mtype] > self.fhist[best_set]:
                                    self.repl |= set([best_set])
                                    best_set = mtype
                                else:
                                    self.repl |= set([mtype])
                            else:
                                self.prun |= set([mtype])
                        else:
                            if self.phist[mtype] > self.phist[gene_mtype]:
                                self.merg |= set([mtype])
                            else:
                                self.disc |= set([mtype])
                repl_new = set()
                prun_new = set()
                merg_new = set()
                disc_new = set()
                if best_set is not None:
                    for mtype in tuple(self.repl):
                        if mtype & best_set is None:
                            repl_new |= set([mtype])
                    for mtype in tuple(self.prun):
                        if mtype & best_set is None:
                            prun_new |= set([mtype])
                    for mtype in tuple(self.merg):
                        if mtype & best_set is None:
                            merg_new |= set([mtype])
                    for mtype in tuple(self.disc):
                        if mtype & best_set is None:
                            disc_new |= set([mtype])
                self.prun = prun_new
                self.merg = merg_new
                self.disc = disc_new
                self.chosen_sets |= set([best_set])
                self.leftover |= set([
                    reduce(
                        lambda x,y: x|y,
                        self.data.train_mut_.direct_subsets(gene_mtype))
                    - best_set
                    ])

        elif self.mode == 'Loss':
            best_mtype = self.tested.pop(0)[0]
            while best_mtype is not None:
                new_mtypes = [
                    best_mtype | m for m in
                    self.data.train_mut_.direct_subsets(self.loss_mtype)
                    if not m <= best_mtype and
                    not any([m <= n for n,_ in self.final])
                    ]
                best_set,best_val = self.test_mtypes(new_mtypes)
                if best_val > self.history[best_mtype]:
                    best_mtype = best_set
                else:
                    self.final.append((best_mtype,self.history[best_mtype]))
                    self.tested = filter(
                        lambda x: (x[0] & best_mtype) is None,
                        self.tested)
                    best_mtype = None
            if not self.tested:
                self.mode = 'Mutex'

        elif self.mode == 'Mutex':
            self.mut_samples = self.data.train_mut_.get_samples(
                data.MuType({('Gene', (self.loss_gene,self.mutex_gene)):None}))
            if not self.tested:
                mutex_mtypes = self.data.train_mut_.combsets(
                    mtype=data.MuType({('Gene', self.mutex_gene):None}),
                    levels=('Gene','Conseq'), min_size=12, comb_sizes=(1,2,3)
                    )
                best_set,best_val = self.test_mtypes(mutex_mtypes)
            best_mtype = self.final.pop(0)[0]
            while best_mtype is not None:
                new_mtypes = [
                    best_mtype | m for m,_ in self.tested
                    if not m <= best_mtype
                    ]
                self.mut_samples = None
                best_combset,best_combval = self.test_mtypes(new_mtypes)
                if best_combval > max(self.history[best_mtype], best_val):
                    best_mtype = best_combset
                else:
                    self.final.append((best_mtype,self.history[best_mtype]))
                    self.tested = filter(
                        lambda x: (x[0] & best_mtype) is None,
                        self.tested)
                    best_mtype = None
                    if not self.tested:
                        self.mode == 'Final'


class LossSimple(object):

    def __init__(self,
                 cdata, loss_gene, classif=classif.LogReg,
                 test_indx=range(16), tune_indx=(50,75)):
        # checks that the given MutExpr object has a valid
        # collection of mutation data
        if not hasattr(cdata, 'cnv_'):
            raise LossOptimError(
                "Given MutExpr object must include CNV status.")
        if loss_gene not in cdata.cnv_:
            raise LossOptimError("Given MutExpr object must have "
                                 "mutation data for the loss gene.")
        self.loss_gene = loss_gene
        mutex_gene = set(cdata.cnv_.keys()) - set([loss_gene])
        if len(mutex_gene) != 1:
            raise LossOptimError("Given MutExpr object must have mutation "
                                 "data for exactly one gene other than the "
                                 "loss gene.")
        self.mutex_gene = tuple(mutex_gene)[0]

        # infers CNV loss, adds attributes
        cdata.add_cnv_loss()
        self.data = cdata
        self.mut_samples = cdata.train_mut_.get_samples(
            data.MuType({('Gene', (self.loss_gene,self.mutex_gene)):None}))
        self.loss_mtype = data.MuType({('Gene',self.loss_gene):None})
        self.mutex_mtype = data.MuType({('Gene',self.mutex_gene):None})
        self.classif = classif
        self.test_indx = test_indx
        self.tune_indx = tune_indx

        # gets list of genes included in each dataset, finds those not on
        # the same chromosome as the loss gene
        mut_chrs = [v['chr'] for k,v in cdata.annot.items()
                    if v['gene_name'] == loss_gene
                    or v['gene_name'] == self.mutex_gene]
        self.expr_genes = [v['gene_name'] for v in cdata.annot.values()]
        self.loss_genes = [v['gene_name'] for v in cdata.annot.values()
                           if v['chr'] not in mut_chrs]
        self.part_hist = {}
        self.full_hist = {}
        self.tested = []
        self.final = []

    def __str__(self):
        """Displays current state of the optimizer."""
        out_str = ""
        if self.tested:
            out_str += ("Tested:\n" + reduce(
                lambda x,y: x+y,
                [str(k) + ': '
                 + str(round(v,4)) + ' | ' + str(round(w,4)) + '\n'
                 for k,v,w in self.tested]))
        if self.final:
            out_str += ("Final:\n" + reduce(
                lambda x,y: x+y,
                [str(k) + ': ' + str(round(v,4)) + '\n'
                 for k,v in self.final]))
        return out_str

    def test_mtypes(self, mtypes):
        print "Testing " + str(len(mtypes)) + " sets..."
        full_mtype = None
        full_val = 0
        part_mtype = None
        part_val = 0
        for mtype in mtypes:
            new_classif = self.classif()
            if mtype not in self.full_hist:
                full_perf = self.data.test_classif_cv(
                    classif=new_classif, mtype=mtype,
                    gene_list=self.loss_genes, exclude_samps=None,
                    test_indx=self.test_indx, tune_indx=self.tune_indx,
                    final_fit=True
                    )
                part_perf = self.data.test_classif_cv(
                    classif=new_classif, mtype=mtype,
                    gene_list=self.loss_genes, exclude_samps=self.mut_samples,
                    test_indx=self.test_indx, tune_indx=self.tune_indx,
                    final_fit=True
                    )
                if full_perf > full_val:
                    full_mtype = mtype
                    full_val = full_perf
                self.full_hist[mtype] = full_perf
                if part_perf > part_val:
                    part_mtype = mtype
                    part_val = part_perf
                self.part_hist[mtype] = part_perf
                if not self.tested:
                    self.tested = [(mtype, full_perf, part_perf)]
                else:
                    sort_indx = sum([full_perf < v for _,v,_ in self.tested])
                    self.tested.insert(sort_indx, (mtype, full_perf, part_perf))

    def step(self):
        if not self.full_hist:
            self.mode = 'Prune'
            test_mtypes = self.data.train_mut_.combsets(
                levels=('Gene','Conseq'), min_size=8, comb_sizes=(1,))
            test_mtypes += [self.loss_mtype, self.mutex_mtype,
                          self.loss_mtype | self.mutex_mtype]
            self.test_mtypes(test_mtypes)

        elif self.mode == 'Prune':
            best_mtype = self.tested.pop(0)[0]
            if best_mtype is not None:
                if self.full_hist[best_mtype] > self.part_hist[best_mtype]:
                    subsets = self.data.train_mut_.combsets(
                        mtype=best_mtype, levels=('Gene','Conseq','Exon'),
                        comb_sizes=(1,)
                        )
                    if len(subsets) > 1:
                        use_mtype = reduce(
                            lambda x,y: x|y,
                            self.data.train_mut_.direct_subsets(best_mtype))
                        new_mtypes = [use_mtype - x for x in subsets]
                        self.test_mtypes(new_mtypes)
                else:
                    self.final += [best_mtype]

        elif self.mode == 'Loss':
            best_mtype = self.tested.pop(0)[0]
            while best_mtype is not None:
                new_mtypes = [
                    best_mtype | m for m in
                    self.data.train_mut_.direct_subsets(self.loss_mtype)
                    if not m <= best_mtype and
                    not any([m <= n for n,_ in self.final])
                    ]
                best_set,best_val = self.test_mtypes(new_mtypes)
                if best_val > self.history[best_mtype]:
                    best_mtype = best_set
                else:
                    self.final.append((best_mtype,self.history[best_mtype]))
                    self.tested = filter(
                        lambda x: (x[0] & best_mtype) is None,
                        self.tested)
                    best_mtype = None
            if not self.tested:
                self.mode = 'Mutex'

        elif self.mode == 'Mutex':
            self.mut_samples = self.data.train_mut_.get_samples(
                data.MuType({('Gene', (self.loss_gene,self.mutex_gene)):None}))
            if not self.tested:
                mutex_mtypes = self.data.train_mut_.combsets(
                    mtype=data.MuType({('Gene', self.mutex_gene):None}),
                    levels=('Gene','Conseq'), min_size=12, comb_sizes=(1,2,3)
                    )
                best_set,best_val = self.test_mtypes(mutex_mtypes)
            best_mtype = self.final.pop(0)[0]
            while best_mtype is not None:
                new_mtypes = [
                    best_mtype | m for m,_ in self.tested
                    if not m <= best_mtype
                    ]
                self.mut_samples = None
                best_combset,best_combval = self.test_mtypes(new_mtypes)
                if best_combval > max(self.history[best_mtype], best_val):
                    best_mtype = best_combset
                else:
                    self.final.append((best_mtype,self.history[best_mtype]))
                    self.tested = filter(
                        lambda x: (x[0] & best_mtype) is None,
                        self.tested)
                    best_mtype = None
                    if not self.tested:
                        self.mode == 'Final'


class LossFinder(object):

    def __init__(self,
                 cdata, loss_gene, classif=classif.LogReg,
                 test_indx=range(16), tune_indx=(50,75)):
        # checks that the given MutExpr object has a valid
        # collection of mutation data
        if not hasattr(cdata, 'cnv_'):
            raise LossOptimError(
                "Given MutExpr object must include CNV status.")
        if loss_gene not in cdata.cnv_:
            raise LossOptimError("Given MutExpr object must have "
                                 "mutation data for the loss gene.")
        self.loss_gene = loss_gene
        mutex_gene = set(cdata.cnv_.keys()) - set([loss_gene])
        if len(mutex_gene) != 1:
            raise LossOptimError("Given MutExpr object must have mutation "
                                 "data for exactly one gene other than the "
                                 "loss gene.")
        self.mutex_gene = tuple(mutex_gene)[0]

        # infers CNV loss, adds attributes
        cdata.add_cnv_loss()
        self.data = cdata
        self.mut_samples = cdata.train_mut_.get_samples(
            data.MuType({('Gene', (self.loss_gene,self.mutex_gene)):None}))
        self.classif = classif
        self.test_indx = test_indx
        self.tune_indx = tune_indx

        # gets list of genes included in each dataset, finds those not on
        # the same chromosome as the loss gene
        mut_chrs = [v['chr'] for k,v in cdata.annot.items()
                    if v['gene_name'] == loss_gene
                    or v['gene_name'] == self.mutex_gene]
        self.expr_genes = [v['gene_name'] for v in cdata.annot.values()]
        self.loss_genes = [v['gene_name'] for v in cdata.annot.values()
                           if v['chr'] not in mut_chrs]
        self.history = {}
        self.coefs = {}
        self.comps = {}
        self.cur_sets = []

    def __str__(self):
        """Displays current state of the optimizer."""
        return reduce(
            lambda x,y: x+y,
            [str(k) + ': ' + str(round(v,4)) + '\n'
             for k,v in self.history.items()]
            )

    def test_mtypes(self, mtypes):
        for mtype in mtypes:
            if mtype not in self.history:
                new_classif = self.classif()
                self.history[mtype] = self.data.test_classif_cv(
                    classif=new_classif, mtype=mtype,
                    gene_list=self.loss_genes, exclude_samps=self.mut_samples,
                    test_indx=self.test_indx, tune_indx=self.tune_indx,
                    final_fit=True
                    )
                self.coefs[mtype] = dict(
                    [[g,c] for g,c in
                     zip(self.loss_genes,
                         new_classif.named_steps['fit'].coef_[0])
                     if c>0])

    def comp_mtypes(self, old_mtypes, new_mtype):
        old_mean = np.mean([self.history[mtype] for mtype in old_mtypes])
        if self.history[new_mtype] > old_mean:
            old_size = len(reduce(
                lambda x,y: x|y,
                [self.data.train_mut_.get_samples(mtype) for mtype in old_mtypes]
                ))
            new_size = len(self.data.train_mut_.get_samples(new_mtype))
            comp_val = ((self.history[new_mtype] / old_mean)
                        * (float(new_size) / old_size))
            self.comps[(old_mtypes, new_mtype)] = comp_val

    def step(self):
        if not self.history:
            self.mode = 'Divide'
            loss_mtypes = self.data.train_mut_.combsets(
                mtype=data.MuType({('Gene', self.loss_gene):None}),
                levels=('Gene','Conseq'),
                min_size=int(len(cdata.train_mut_) * 0.1), comb_sizes=(1,)
                )
            self.test_mtypes(loss_mtypes)
            self.cur_sets = loss_mtypes
        
        elif self.mode == 'Divide':
            self.mode = 'Combine'
            self.add_sets = []
            for head_mtype in self.cur_sets:
                new_mtypes = self.data.train_mut_.combsets(
                    mtype=head_mtype, levels=('Gene','Conseq','Exon'),
                    min_size=int(len(cdata.train_mut_) * 0.05, comb_sizes=(1,2)
                    ))
                new_mtypes = list(set(new_mtypes) - set([head_mtype]))
                self.test_mtypes(new_mtypes)
                self.add_sets += new_mtypes
                for new_mtype in new_mtypes:
                    self.comp_mtypes((head_mtype,), new_mtype)
            self.cur_sets += self.add_sets

        elif self.mode == 'Combine':
            comb_dists = {}
            mtype_combs = combinations(self.cur_sets, 2)
            for comb in mtype_combs:
                comb_dists[comb] = self.coef_sim(comb[0],comb[1])
            test_combs = takewhile(
                lambda x: x[1]>0,
                sorted(comb_dists.items(),
                       key=operator.itemgetter(1),
                       reverse=True)
                )
            for comb in test_combs:
                self.test_mtypes(comb[0][0] | comb[0][1])
                self.comp_mtypes(
                    (comb[0][0], comb[0][1]),
                    comb[0][0] | comb[0][1])

        elif self.mode == 'Inter':
            mutex_mtypes = self.data.train_mut_.combsets(
                mtype=data.MuType({('Gene', self.mutex_gene):None}),
                levels=('Gene','Conseq'), min_size=12, comb_sizes=(1,2)
                )

    def coef_sim(self, mtype1, mtype2):
        if mtype1 not in self.coefs or mtype2 not in self.coefs:
            raise LossOptimError("Both mutation sets must have "
                                 "already been tested!")
        coefs1 = self.coefs[mtype1]
        coefs2 = self.coefs[mtype2]
        mags1 = np.sum([x ** 2 for x in coefs1.values()]) ** 0.5
        mags2 = np.sum([x ** 2 for x in coefs2.values()]) ** 0.5
        ovlp = np.sum([coefs1[k] * coefs2[k]
                      for k in set(coefs1.keys()) & set(coefs2.keys())])
        return ovlp / (mags1*float(mags2))

    def load_history(self, history=None, coefs=None, comps=None):
        if history is not None:
            self.history.update(history)
        if coefs is not None:
            self.coefs.update(coefs)
        if comps is not None:
            self.comps.update(comps)


class LossOptimizer(object):

    def __init__(self, cdata_list, loss_gene):
        # checks that given MutExpr objects have a valid collection of
        # mutation data, infers CNV loss status
        if not all([hasattr(x, 'cnv_') for x in cdata_list]):
            raise LossOptimError("All given MutExpr objects must include "
                                 "CNV status.")
        if not all([loss_gene in x.cnv_ for x in cdata_list]):
            raise LossOptimError("All given MutExpr objects must have "
                                 "mutation data for the loss gene.")
        self.loss_gene = loss_gene
        self.loss_set = data.MuType(
            {('Gene',self.loss_gene):{('Conseq','Loss'):None}})
        self.mtype_cur = self.loss_set
        mutex_genes = reduce(
            lambda x,y: x&y,
            [set(x.cnv_.keys()) for x in cdata_list]
            ) - set([loss_gene])
        if not mutex_genes:
            raise LossOptimError("Given MutExpr objects must share mutation "
                                 "data for at least one gene other than the "
                                 "loss gene.")
        self.mutex_genes = tuple(mutex_genes)
        for i in xrange(len(cdata_list)):
            cdata_list[i].add_cnv_loss()
        self.data = cdata_list
       
        # gets list of genes included in each dataset, finds those not on
        # the same chromosome as the loss gene
        mut_chr = [v['chr'] for k,v in cdata_list[0].annot.items()
                   if v['gene_name'] == loss_gene]
        self.expr_genes = [[v['gene_name'] for v in x.annot.values()]
                           for x in cdata_list]
        self.loss_genes = [[v['gene_name'] for v in x.annot.values()
                            if v['chr'] != mut_chr
                            and v['gene_name'] not in self.mutex_genes]
                           for x in cdata_list]

        # initializes classifiers that will be used to predict loss of
        # functionality status
        self.lasso_clx = [classif.LogReg([loss_gene], x)
                          for x in self.expr_genes]
        self.kern_clx = [classif.PCgbc([loss_gene], x)
                         for x in self.expr_genes]
        self.uni_scores = {}
        self.multi_scores = {}
        self.coefs = {}

    def __str__(self):
        """Displays current state of the optimizer."""
        return reduce(
            lambda x,y: x+y,
            [str(k) + ': '
             + reduce(lambda x,y: str(x) + ' | ' + str(y),
                      map(lambda n: round(n,4), v)) + '\n'
             for k,v in self.uni_scores.items()])


    def step(self):
        if not self.uni_scores:
            print ("Initializing Loss Optimizer for gene "
                   + self.loss_gene + " in ICGC projects "
                   + reduce(lambda x,y: x + ", " + y,
                            [d.project_ for d in self.data])
                   + " and MutEx genes "
                   + reduce(lambda x,y: x + ", " + y, self.mutex_genes))
            self.test_uniclx(mtype=self.mtype_cur)

        elif not self.multi_scores:
            self.test_multiclx(mtype=self.mtype_cur, kern_label='All')
            self.test_multiclx(mtype=self.mtype_cur, kern_label='Union')

        elif (self.mtype_cur == data.MuType(
            {('Gene',self.loss_gene):{('Conseq','Loss'):None}})):
            self.mtype_cur = data.MuType(
                {('Gene',self.loss_gene):{('Conseq',('Loss','stop_gained')):None}})
            self.test_uniclx(mtype=self.mtype_cur)
            self.test_multiclx(mtype=self.mtype_cur, kern_label='All')
            self.test_multiclx(mtype=self.mtype_cur, kern_label='Union')

    def test_uniclx(self,
                    mtype, test_indx=range(16), tune_indx=(50,75),
                    gene_lists=None, verbose=False):
        if gene_lists is None:
            gene_lists = self.loss_genes
        self.uni_scores[mtype] = [
            d.test_classif_cv(
                classif=c, mtype=mtype,
                test_indx=test_indx, tune_indx=tune_indx,
                gene_list=g, final_fit=True, verbose=verbose)
            for d,c,g in zip(self.data, self.uniclx, gene_lists)]
        self.coefs[mtype] = [
            {g:c for g,c in zip(gn,cf) if c>0}
            for gn,cf in zip(
                gene_lists, [clx.named_steps['fit'].coef_[0]
                             for clx in self.uniclx])]

    def test_multiclx(self,
                      mtype, kern_label,
                      test_indx=range(16), tune_indx=(50,75),
                      verbose=False):
        if kern_label == 'All':
            kern_genes = self.loss_genes
        elif kern_label == 'Union':
            union_genes = reduce(
                lambda x,y: x|y,
                [set(c.keys()) for c in self.coefs[self.mtype_cur]])
            kern_genes = [list(union_genes & set(lg))
                          for lg in self.loss_genes]
        elif kern_label == 'Inter':
            inter_genes = reduce(
                lambda x,y: x&y,
                [set(c.keys()) for c in self.coefs[self.mtype_cur]])
            kern_genes = [set(ig) & set(lg)
                          for ig,lg in zip(inter_genes, self.expr_genes)]

        multiclx = MultiKBTL(kern_genes)
        train_list = [x.training(mtype=mtype) for x in self.data]
        test_cvs = [[tr_l[2][i] for tr_l in train_list] for i in test_indx]
        if tune_indx is not None:
            tune_cvs = [[x for i,x in enumerate(tr_l[2])
                        if i in tune_indx] for tr_l in train_list]
            multiclx.tune(
                expr_list=[x[0].loc[:,g]
                           for g,x in zip(kern_genes,train_list)],
                mut_list=[x[1] for x in train_list],
                cv_samples=tune_cvs, verbose=verbose
                )
        if not mtype in self.multi_scores:
            self.multi_scores[mtype] = {}
        self.multi_scores[mtype][kern_label] = multiclx.crossval_score(
            expr_list=[x[0].loc[:,g]
                       for g,x in zip(kern_genes,train_list)],
            mut_list=[x[1] for x in train_list],
            cv_list = test_cvs
            )

    def loss_score(self, mtype):
        for i in range(len(self.loss_genes)):
            test_expr,test_mut = self.data[i].testing(
                mtype, self.loss_genes[i])
            _,loss_mut = self.data[i].testing(
                self.loss_set, self.loss_genes[i])
            probs = self.uniclx[i].prob_mut(test_expr)
            print "Both:"
            print str(np.percentile(
                [p for p,ml,mg in zip(probs, loss_mut, test_mut)
                 if mg and ml], (25,75)))
            print "Loss Only:"
            print str(np.percentile(
                [p for p,ml,mg in zip(probs, loss_mut, test_mut)
                 if not mg and ml], (25,75)))
            print "Other Only:"
            print str(np.percentile(
                [p for p,ml,mg in zip(probs, loss_mut, test_mut)
                 if mg and not ml], (25,75)))
            print "Neither:"
            print str(np.percentile(
                [p for p,ml,mg in zip(probs, loss_mut, test_mut)
                 if not mg and not ml], (25,75)))

    def coef_sim(self, mtype1, mtype2):
        if mtype1 not in self.coefs or mtype2 not in self.coefs:
            raise LossOptimError("Both mutation sets must have "
                                 "already been tested!")
        coefs1 = self.coefs[mtype1]
        coefs2 = self.coefs[mtype2]
        mags1 = [np.sum([x ** 2 for x in c.values()]) ** 0.5 for c in coefs1]
        mags2 = [np.sum([x ** 2 for x in c.values()]) ** 0.5 for c in coefs2]
        ovlp = [np.sum(c1[k]*c2[k] for k in set(c1.keys()) & set(c2.keys()))
                for c1,c2 in zip(coefs1,coefs2)]
        return [ov / (mg1*mg2) for ov,mg1,mg2 in zip(ovlp,mags1,mags2)]



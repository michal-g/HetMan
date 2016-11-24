
"""
Hetman (Heterogeneity Manifold)
Classification of mutation sub-types using expression data.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

import itertools
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

class DefunctError(Exception):
    """Class for exceptions thrown by the Hetman defunct module."""
    pass


class Defunct(object):

    def __init__(self, mut_expr, mut_gene=None):
        if not hasattr(mut_expr, 'cnv_'):
            raise DefunctError("Need to use an MutExpr object with CNV data")
        if mut_gene is None:
            mut_gene=mut_expr.cnv_.keys()[0]
        self.mut_gene_ = mut_gene

        self.train_cnv_ = [mut_expr.cnv_[mut_gene][s] for s in
                           mut_expr.train_expr_.index]
        self.test_cnv_ = [mut_expr.cnv_[mut_gene][s] for s in
                          mut_expr.test_expr_.index]
        self.train_samps = mut_expr.train_expr_.index
        self.test_samps = mut_expr.test_expr_.index
        self.train_expr_ = mut_expr.train_expr_.loc[:,mut_gene]
        self.test_expr_ = mut_expr.test_expr_.loc[:,mut_gene]

    def infer_cnv(self):
        norm = StandardScaler()
        mix = GaussianMixture(
            n_components=3, covariance_type='full', n_init=20,
            init_params='random', weights_init=[0.1,0.8,0.1],
            means_init=[(-1,-1), (0,0), (1,1)]
            )
        train_data = np.column_stack((self.train_expr_, self.train_cnv_))
        train_data = norm.fit_transform(train_data)
        mix.fit(train_data)
        train_labels = mix.predict(train_data)
        train_probs = mix.predict_proba(train_data)
        return norm,mix,train_data,train_labels,train_probs

    def get_loss(self):
        norm,mix,train_data,train_labels,train_probs = self.infer_cnv()
        test_labels = mix.predict(norm.transform(
            np.column_stack((self.test_expr_, self.test_cnv_))))
        return (tuple(self.train_samps[train_labels == 0]),
                tuple(self.test_samps[test_labels == 0]))

    def plot_infer(self):
        mix,train_data,train_labels,train_probs = self.infer_cnv()
        Y_ = mix.predict(train_data)
        colour_iter = itertools.cycle(
            ['navy', 'c', 'cornflowerblue', 'gold', 'darkorange'])
        splot = plt.subplot(1, 1, 1)
        for i, (mean, covar, colour) in enumerate(
            zip(mix.means_, mix.covariances_, colour_iter)):
            v, w = linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            plt.scatter(train_data[Y_ == i, 0], train_data[Y_ == i, 1],
                        20, color=colour)
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi
            ell = mpl.patches.Ellipse(
                mean, v[0], v[1], 180. + angle, color=colour)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.6)
            splot.add_artist(ell)

        xlims = [min(train_data[:,0])-0.3, max(train_data[:,0])+0.3]
        ylims = [min(train_data[:,1])-0.1, max(train_data[:,1])+0.1]
        plt.xlim(xlims[0], xlims[1])
        plt.ylim(ylims[0], ylims[1])
        plt.xticks(())
        plt.yticks(())
        plt.title('Expr vs. CNV')
        plt.show()


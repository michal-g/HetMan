
"""
Utility functions for HetMan experiments.
"""

import re
import pickle

from os import listdir
from os.path import isfile, join
from functools import reduce

base_dir = '/home/users/grzadkow/compbio/scripts/HetMan/experiments/'


def get_set_plotlbl(lbls):
    if isinstance(lbls, str):
        return lbls
    else:
        return reduce(lambda x,y: x + '-' + y, lbls)

def get_set_regexp(lbls):
    if isinstance(lbls, str):
        return lbls
    else:
        return '(' + reduce(lambda x,y: x + '|' + y, lbls) + ')'

def load_output(experiment, clf_set='base', mtype_set='default'):
    """Loads data output from experiment runs."""
    output_dir = join(base_dir, experiment, 'output')
    clf_regexp = get_set_regexp(clf_set)
    mtype_regexp = get_set_regexp(mtype_set)

    file_list = [fl for fl in listdir(output_dir)
                 if isfile(join(output_dir, fl))
                 and re.search(clf_regexp + '_' + mtype_regexp
                               + '__run[0-9]+\\.p$', fl)]

    return [pickle.load(open(join(output_dir, fl), 'rb')) for fl in file_list]




from ...mutation import MuType
from ... import classifiers as classif


# which classifiers we want to consider in our test
clf_list = {
    'base': [classif.NaiveBayes, classif.Lasso,
             classif.SVCrbf, classif.rForest],
    'more': [classif.RobustNB, classif.LogReg,
             classif.SVCpoly, classif.KNeigh],
    }

# which mutations we want to consider
mtype_list = {
    'default': (
        ('BRCA', MuType({('Gene', 'TP53'):
                         {('Form', 'Missense_Mutation'): None}})),
        ('BRCA', MuType({('Gene', 'PIK3CA'):
                         {('Protein', 'p.H1047R'): None}})),
        ('BRCA', MuType({('Gene', 'CDH1'):
                         {('Form', ('Frame_Shift_Ins', 'Frame_Shift_Del')):
                          None}})),
        ),
    }

# list of possible feature selection methods based on neighbourhoods
# in the Pathway Commons graph
key_list = {
    'All': None,
    'Up': ((['Up'], ()), ),
    'Neigh': ((['Up', 'Down'], ()), ),
    'expr': (((), ['controls-expression-of']), ),
    'Down': ((['Down'], ()), )
    }



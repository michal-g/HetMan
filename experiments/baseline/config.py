
from ...mutation import MuType
from ... import classifiers as classif


# which classifiers we want to consider in our test
clf_list = {
    'base': [classif.NaiveBayes, classif.Lasso,
             classif.SVCrbf, classif.rForest],
    'more': [classif.KNeigh, classif.LogReg],
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
        ('SKCM', MuType({('Gene', 'BRAF'):
                         {('Protein', 'p.V600E'): None}})),
        ('COAD', MuType({('Gene', 'TTN'):
                         {('Form', 'Intron'): None}})),
        ('UCEC', MuType({('Gene', 'PTEN'):
                         {('Form', ('Frame_Shift_Del', 'Nonsense_Mutation')):
                          None}})),
        ),
    }



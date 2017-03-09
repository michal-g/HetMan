
"""
Hetman (Heterogeneity Manifold)
Classification of mutation sub-types using expression data.
This file contains functions for reading in transcriptomic and genomic data
downloaded from sources such as TCGA, ICGC, and Firehose.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

from mutation import MutLevel

import numpy as np
import pandas as pd

from re import sub as gsub
from itertools import groupby
from functools import reduce
from bioservices import PathwayCommons 


# .. directories containing raw -omics data and cross-validation samples ..
_base_dir = '/home/users/grzadkow/compbio/'
_icgc_dir = _base_dir + 'input-data/ICGC/raw/'
_cv_dir = _base_dir + 'auxiliary/HetMan/cv-samples/'


# .. helper functions for parsing -omic datasets ..
def parse_tcga_barcodes(barcodes):
    """Extracts the sample labels from TCGA barcodes."""
    return [reduce(lambda x,y: x + '-' + y,
                   s.split('-', 4)[:4])
            for s in barcodes]


def log_norm_expr(expr):
    """Log-normalizes expression data."""
    log_add = np.nanmin(expr[expr > 0].values) * 0.5
    return np.log2(expr + log_add)


def get_annot(version='v19'):
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
    annot : dict
        Dictionary with keys corresponding to Ensembl gene IDs and values
        consisting of dicts with annotation fields.
    """
    annot = pd.read_csv(
        _base_dir + 'input-data/gencode.' + version + '.annotation.gtf.gz',
        usecols=[0,2,3,4,8], names=['Chr', 'Type', 'Start', 'End', 'Info'],
        sep = '\t', header=None, comment='#'
        )

    # filter out annotation records that aren't
    # protein-coding genes on non-sex chromosomes
    chroms_use = ['chr' + str(i+1) for i in range(22)]
    annot = annot.loc[annot['Type'] == 'gene', ]
    chr_indx = np.array([chrom in chroms_use for chrom in annot['Chr']])
    annot = annot.loc[chr_indx, ]

    # parse the info field to get each gene's annotation data
    gn_annot = {
        gsub('\.[0-9]+', '', z['gene_id']).replace('"', ''):z
        for z in [dict([['chr', an[0]]] +
                       [['Start', an[2]]] +
                       [['End', an[3]]] +
                       [y for y in [x.split(' ')
                                    for x in an[4].split('; ')]
                        if len(y) == 2])
                  for an in annot.values]
        if z['gene_type'] == '"protein_coding"'
        }
    for g in gn_annot:
        gn_annot[g]['gene_name'] = gn_annot[g]['gene_name'].replace('"', '')

    return gn_annot


# .. functions for reading in mRNA expression datasets ..
def get_expr_firehose(cohort):
    """Loads expression data as a matrix from a Firehose GDAC fzile.
       Use ./firehose_get -tasks 'Merge_rnaseqv2__illuminahiseq_rnaseqv2__
       unc_edu__Level_3__RSEM_genes_normalized__data' stddata latest
       to download these files for all available cohorts.
    """
    expr_file = (
        _base_dir + 'input-data/stddata__2016_01_28/' + cohort
        + '/20160128/gdac.broadinstitute.org_' + cohort
        + '.Merge_rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__'
        + 'RSEM_genes_normalized__data.Level_3.2016012800.0.0.tar.gz'
        )
    row_skip = 0
    while True:
        try:
            raw_data = pd.read_csv(
                expr_file, skiprows=row_skip,
                header=None, sep = '\t', dtype=object)
            break
        except:
            row_skip += 1

    # parses gene and sample names to get expression matrix axis labels
    gene_names = [gsub('\|.*$', '', str(x)) for x in raw_data.ix[:,0]]
    gene_indx = [x not in ['gene_id','nan','?'] for x in gene_names]
    gene_indx[0] = False
    samp_names = raw_data.ix[0, :]
    samp_indx = [not isinstance(x, float) and x.find('-01A-') != -1
                 for x in samp_names]
    
    # transforms raw data into expression matrix
    expr_data = raw_data.ix[gene_indx, samp_indx]
    expr_data.index = pd.Series(gene_names).loc[gene_indx]
    expr_data.columns = parse_tcga_barcodes(samp_names.loc[samp_indx])
    expr_data = expr_data.T.astype('float')

    return expr_data


def get_expr_icgc(expr_file):
    """Gets expression data as a matrix from an ICGC tsv.gz file.

    Parameters
    ----------
    expr_file : str
        A file containing expression data.

    Returns
    -------
    expr : ndarray, shape (n_samples, n_features)
        An expression matrix with genes as features, in the case of
        duplicate gene names, values are averaged.
    """
    expr = pd.read_csv(
        expr_file, usecols=(4,7,8), header=0, sep = '\t',
        names=('Sample', 'Gene', 'FPKM')
        )

    # gets patient IDs from TCGA barcodes
    expr['Sample'] = [reduce(lambda x,y: x + '-' + y,
                             s.split('-', 3)[:3])
                      for s in expr['Sample']]
    
    # transforms raw long-format expression data into wide-format
    return expr.pivot_table(index='Sample', columns='Gene',
                            values='FPKM', aggfunc=np.mean)


# .. functions for reading in simple mutation datasets ..
def get_mut_mc3(syn, mut_levels=('Gene', 'Form', 'Exon')):
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
    muts : ndarray, shape (n_mutations, mut_levels+1)
        A mutation array, with a row for each mutation appearing in an
        individual sample.
    """
    # columns in the MC3 file containing each level in the mutation hierarchy
    mut_cols = {
        MutLevel.Gene: 0,
        MutLevel.Form: 8,
        MutLevel.PolyPhen: 72,
        MutLevel.Exon: 38,
        MutLevel.Protein: 36
        }

    # gets data from Synapse, figures out which columns to use
    mc3 = syn.get('syn7824274')
    data_names = ['Sample'] + list(mut_levels)
    use_cols = [15]
    for lvl in mut_levels:
        use_cols.append(mut_cols[MutLevel[lvl]])
    data_names = np.array(data_names)[np.array(use_cols).argsort()]

    # imports data into a DataFrame, parses TCGA sample barcodes
    # and PolyPhen scores
    muts = pd.read_csv(
        mc3.path, usecols=use_cols, sep='\t', header=None,
        names=data_names, comment='#', skiprows=1)
    muts['Sample'] = parse_tcga_barcodes(muts['Sample'])
    if 'PolyPhen' in mut_levels:
        muts['PolyPhen'] = [gsub('\)$', '', gsub('^.*\(', '', x))
                            if x != '.' else 0
                            for x in muts['PolyPhen']]

    return muts


def get_mut_icgc(mut_file):
    """Gets mutation data as an numpy array from an ICGC tsv.gz file.
       Deprecated in favour of using MC3 mutation calls.

    Parameters
    ----------
    mut_file : str
        A file containing mutation data.

    Returns
    -------
    mut : ndarray, shape (n_mutations,)
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


# .. functions for reading in copy number mutations ..
def get_cnv_icgc(cnv_file):
    """Gets copy number variation as an numpy array from an ICGC tsv.gz file.

    Parameters
    ----------
    cnv_file : str
        A file containing copy number variation data.

    Returns
    -------
    cnv_data : ndarray, shape (n_cnvs,)
        A 1-D array with each entry corresponding to a single copy number
        variation affecting a single sample.
    """
    cnv_dt = np.dtype([('Sample',np.str_,64), ('Mean',np.float),
                       ('Chr',np.str_,64), ('Start',np.int), ('End',np.int)])
    return np.loadtxt(fname=cnv_file, dtype=cnv_dt, skiprows=1,
                      delimiter='\t', usecols=(0,9,11,12,13))


def get_cnv_firehose(cohort):
    """Gets CNV data as a matrix from a Firehose GDAC file.
       Use ./firehose_get -tasks CopyNumber_Gistic2 analyses latest
       to download these files for all available cohorts.
    """
    cnv_file = (
        _base_dir + 'input-data/analyses__2016_01_28/' + cohort
        + '/20160128/gdac.broadinstitute.org_' + cohort
        + '-TP.CopyNumber_Gistic2.Level_4.2016012800.0.0/'
        + 'all_data_by_genes.txt'
        )
    raw_data = pd.read_csv(cnv_file, sep='\t')

    # parses gene and sample names into CNV matrix axis labels
    gene_names = [gsub('\|.*$', '', str(x))
                  for x in pd.Series(raw_data.ix[1:, 0])]
    cnv_data = raw_data.ix[1:, 3:]
    cnv_data.index = gene_names
    cnv_data.columns = parse_tcga_barcodes(cnv_data.columns)

    return cnv_data.T


# .. functions for reading in pathway data ..
def get_pc2_neighb(gene):
    """Gets the neighbourhood of a gene as defined by Pathway Commons."""
    pc2 = PathwayCommons(verbose=False)
    pc2.settings.TIMEOUT = 1000
    neighb = {}

    # sets the parameters of the PC2 query
    url = 'graph'
    gene_id = pc2.idmapping(gene)[gene]
    params = {'format': 'BINARY_SIF',
              'kind': 'neighborhood',
              'limit': 1,
              'source': 'http://identifiers.org/uniprot/' + gene_id}

    # runs the PC2 query, makes sure the output has the correct format
    raw_data = 0
    while isinstance(raw_data, int):
        print("Reading in Pathway Commons data for gene " + gene + "...")
        raw_data = pc2.http_get(url, frmt=None, params=params)
        print(type(raw_data))

    print("Moving on...")
    # parses interaction data according to direction
    sif_data = raw_data.splitlines()
    sif_data = [x.split() for x in sif_data]
    up_neighbs = sorted([(x[1], x[0]) for x in sif_data if x[2] == gene],
                        key=lambda x: x[0])
    down_neighbs = sorted([(x[1], x[2]) for x in sif_data if x[0] == gene],
                          key=lambda x: x[0])
    other_neighbs = sorted([x for x in sif_data
                            if x[0] != gene and x[2] != gene],
                           key=lambda x: x[1])

    # parses according to interaction type
    neighb['Up'] = {k:[x[1] for x in v] for k,v in
                    groupby(up_neighbs, lambda x: x[0])}
    neighb['Down'] = {k:[x[1] for x in v] for k,v in
                      groupby(down_neighbs, lambda x: x[0])}
    neighb['Other'] = {k:[(x[0],x[2]) for x in v] for k,v in
                       groupby(other_neighbs, lambda x: x[1])}

    return neighb


def read_sif(mut_genes,
             sif_file='input-data/babur-mutex/data-tcga/Network.sif'):
    """Gets the edges containing at least one of given genes from a SIF
       pathway file and arranges them according to the direction of the
       edge and the type of interaction it represents.

    Parameters
    ----------
    mut_genes : array-like, shape (n_genes,)
        A list of genes whose interactions are to be retrieved.

    sif_file : str, optional
        A file in SIF format describing gene interactions.
        The default is the set of interactions used in the MutEx paper.

    Returns
    -------
    link_data : dict
        A list of the interactions that involve one of the given genes.
    """
    if isinstance(mut_genes, str):
        mut_genes = [mut_genes]
    sif_dt = np.dtype(
        [('Gene1', np.str_, 16),
         ('Type', np.str_, 32),
         ('Gene2', np.str_, 16)])
    sif_data = np.loadtxt(
        fname = _base_dir + sif_file, dtype = sif_dt, delimiter = '\t')
    link_data = {g:{'in':None, 'out':None} for g in mut_genes}

    for gene in mut_genes:
        in_data = np.sort(sif_data[sif_data['Gene2'] == gene],
                          order='Type')
        out_data = np.sort(sif_data[sif_data['Gene1'] == gene],
                           order='Type')
        link_data[gene]['in'] = {k:[x['Gene1'] for x in v] for k,v in
                                 groupby(in_data, lambda x: x['Type'])}
        link_data[gene]['out'] = {k:[x['Gene2'] for x in v] for k,v in
                                  groupby(out_data, lambda x: x['Type'])}
    return link_data



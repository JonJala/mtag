#!/usr/bin/env python
'''
'''

from __future__ import division
import numpy as np
import pandas as pd
import scipy.optimize
import argparse
import itertools
import time
import os, re
import joblib
import sys, gzip, bz2
import logging
from argparse import Namespace

from ldsc_mod.ldscore import sumstats as sumstats_sig
from ldsc_mod.ldscore import allele_info

import ldsc_mod.munge_sumstats as munge_sumstats

__version__ = '1.0.3    '

borderline = "<><><<>><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"

header ="\n"
header += borderline +"\n"
header += "<>\n"
header += "<> MTAG: Multitrait Analysis of GWAS \n"
header += "<> Version: {}\n".format(str(__version__))
header += "<> (C) 2017 Omeed Maghzian, Raymond Walters, and Patrick Turley\n"
header += "<> Harvard University Department of Economics / Broad Institute of MIT and Harvard\n"
header += "<> GNU General Public License v3\n"
header += borderline + "\n"
header += "<> Note:  It is recommended to run your own QC on the input before using this program. \n"
header += "<> Software-related correspondence: maghzian@nber.org \n"
header += "<> All other correspondence: paturley@broadinstitute.org \n"
header += borderline +"\n"
header += "\n\n"

pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 800)
pd.set_option('precision', 12)
pd.set_option('max_colwidth', 800)
pd.set_option('colheader_justify', 'left')

np.set_printoptions(linewidth=800)
np.set_printoptions(precision=3)

## General helper functions
def safely_create_folder(folder_path):
    try:
        os.makedirs(folder_path)
    except OSError:
        if not os.path.isdir(folder_path):
            raise

class DisableLogger():
    '''
    For disabling the logging module when calling munge_sumstats
    '''
    def __enter__(self):
       logging.disable(logging.CRITICAL)
    def __exit__(self, a, b, c):
       logging.disable(logging.NOTSET)

## Read / Write functions
def _read_SNPlist(file_path, SNP_index):

    # TODO Add more possible ways of reading SNPlists
    snplist = pd.read_csv(file_path, header=0, index_col=False)
    if SNP_index not in snplist.columns:
        raise ValueError("SNPlist read from {} does include --snp_name {} in its columns.".format(file_path, SNP_index))
    return pd.read_csv(file_path, header=0, index_col=False)

def _read_GWAS_sumstats(GWAS_file_name):
    '''
    read GWAS summary statistics from file that is in one of the acceptable formats.
    '''
    # TODO read more file types
    (openfunc, compression) = munge_sumstats.get_compression(args.sumstats)
    dat_gen = pd.read_csv(GWAS_file_name, index_col=False, header=0,delim_whitespace=True, compression=compression, na_values=['.','NA'],
        iterator=True, chunksize=args.chunksize)
    dat_gen = list(dat_gen)
    dat_gen_unfiltered = pd.concat(dat_gen, axis=0).reset_index(drop=True)

    return  dat_gen_unfiltered, dat_gen

def _read_matrix(file_path):
    '''
    For reading 2-dimensional matrices. These files must be in .npy form or whitespace delimited .csv files
    '''
    ext = file_path[-4:]
    if ext == '.npy':
        return np.load(file_path)
    if ext == '.txt':
        return np.loadtxt(file_path)
    else:
        raise ValueError('{} is not one of the acceptable file paths for reading in matrix-valued objects.'.format(ext))

## LDSC related functions
def sec_to_str(t):
    '''Convert seconds to days:hours:minutes:seconds'''
    [d, h, m, s, n] = reduce(lambda ll, b : divmod(ll[0], b) + ll[1:], [(t, 1), 60, 60, 24])
    f = ''
    if d > 0:
        f += '{D}d:'.format(D=d)
    if h > 0:
        f += '{H}h:'.format(H=h)
    if m > 0:
        f += '{M}m:'.format(M=m)

    f += '{S}s'.format(S=s)
    return f

class Logger_to_Logging(object):
    """
    Logger class that write uses logging module and is needed to use munge_sumstats or ldsc from the LD score package.
    """
    def __init__(self):
        logging.info('created Logger instance to pass through ldsc.')
        super(Logger_to_Logging, self).__init__()

    def log(self,x):
        logging.info(x)

def _perform_munge(args, GWAS_df, GWAS_dat_gen,p):

    original_cols = GWAS_df.columns
    merge_alleles = None
    out=None
    zz= args.z_name if args.z_name is not None else 'z'
    ignore_list = ""
    if args.info_min is None:
        ignore_list += "info"

    # sumstats is set to null because generator passed manually
    argnames = Namespace(sumstats=None,N=None,N_cas=None,N_con=None,out=out,maf_min=args.maf_min_list[p], info_min =args.info_min_list[p],daner=False, no_alleles=False, merge_alleles=merge_alleles,n_min=args.n_min_list[p],chunksize=args.chunksize, snp=args.snp_name,N_col=args.n_name, N_cas_col=None, N_con_col = None, a1=None, a2=None, p=None,frq=args.eaf_name,signed_sumstats=zz+',0', info=None,info_list=None, nstudy=None,nstudy_min=None,ignore=ignore_list,a1_inc=False, keep_maf=True, daner_n=False, keep_str_ambig=True, input_datgen=GWAS_dat_gen, cnames=list(original_cols))

    logging.info(borderline)
    logging.info('Munging Trait {}  {}'.format(p+1,borderline[:-17]))
    logging.info(borderline)


    munged_results = munge_sumstats.munge_sumstats(argnames, write_out=False, new_log=False)
    GWAS_df = GWAS_df.merge(munged_results, how='inner',left_on =args.snp_name,right_on='SNP',suffixes=('','_ss'))
    GWAS_df = GWAS_df[original_cols]

    logging.info(borderline)
    logging.info('Munging of Trait {} complete. SNPs remaining:\t {}'.format(p+1, len(GWAS_df)))
    logging.info(borderline+'\n')

    return GWAS_df, munged_results

def _quick_mode(ndarray,axis=0):
    '''
    From stackoverflow: Efficient calculation of the mode of an array. Scipy.stats.mode is way too slow
    '''
    if ndarray.size == 1:
        return (ndarray[0],1)
    elif ndarray.size == 0:
        raise Exception('Attempted to find mode on an empty array!')
    try:
        axis = [i for i in range(ndarray.ndim)][axis]
    except IndexError:
        raise Exception('Axis %i out of range for array with %i dimension(s)' % (axis,ndarray.ndim))
    srt = np.sort(ndarray,axis=axis)
    dif = np.diff(srt,axis=axis)
    shape = [i for i in dif.shape]
    shape[axis] += 2
    indices = np.indices(shape)[axis]
    index = tuple([slice(None) if i != axis else slice(1,-1) for i in range(dif.ndim)])
    indices[index][dif == 0] = 0
    indices.sort(axis=axis)
    bins = np.diff(indices,axis=axis)
    location = np.argmax(bins,axis=axis)
    mesh = np.indices(bins.shape)
    index = tuple([slice(None) if i != axis else 0 for i in range(dif.ndim)])
    index = [mesh[i][index].ravel() if i != axis else location.ravel() for i in range(bins.ndim)]
    counts = bins[tuple(index)].reshape(location.shape)
    index[axis] = indices[tuple(index)]
    modals = srt[tuple(index)].reshape(location.shape)
    return (modals, counts)


def load_and_merge_data(args):
    '''
    TODO Add description
    Parses file names from MTAG command line arguments and returns the relevant used for method.
    '''

    GWAS_input_files = args.sumstats.split(',')
    P = len(GWAS_input_files)  # of phenotypes/traits
    if args.n_min is not None:
        args.n_min_list = [float(x) for x in args.n_min.split(',')]
        if len(args.n_min_list) == 1:
            args.n_min_list = args.n_min_list * P
    else:
        args.n_min_list = [None]*P

    if args.maf_min is not None:
        args.maf_min_list = [float(x) for x in args.maf_min.split(',')]
        if len(args.maf_min_list) == 1:
            args.maf_min_list = args.maf_min_list * P
    else:
        args.maf_min_list = [None]*P

    if args.info_min is not None:
        args.info_min_list = [float(x) for x in args.info_min.split(',')]
        if len(args.info_min_list) == 1:
            args.info_min_list = args.info_min_list * P
    else:
        args.info_min_list = [None]*P



    GWAS_d = dict()
    sumstats_format = dict()
    for p, GWAS_input in enumerate(GWAS_input_files):
        GWAS_d[p], gwas_dat_gen = _read_GWAS_sumstats(GWAS_input)
        # add suffix
        logging.info('Read in Trait {} summary statistics ({} SNPs) from {} ...'.format(p+1,len(GWAS_d[p]), GWAS_input))

        # perform munge sumstats
        GWAS_d[p], sumstats_format[p] = _perform_munge(args, GWAS_d[p], gwas_dat_gen, p)
        GWAS_d[p] = GWAS_d[p].add_suffix(p)

        # convert Alleles to uppercase
        for col in [col+str(p) for col in [args.a1_name, args.a2_name]]:
            GWAS_d[p][col] = GWAS_d[p][col].str.upper()

        GWAS_d[p] =GWAS_d[p].rename(columns={x+str(p):x for x in GWAS_d[p].columns})
        GWAS_d[p] = GWAS_d[p].rename(columns={args.snp_name+str(p):args.snp_name})

        # Drop SNPs that are missing or duplicated
        missing_snps = GWAS_d[p][args.snp_name].isin(['NA','.'])
        M0 = len(GWAS_d[p])
        GWAS_d[p] = GWAS_d[p][np.logical_not(missing_snps)]
        if M0-len(GWAS_d[p]) > 0:
            logging.info('Trait {}: Dropped {} SNPs for missing values in the "snp_name" column'.format(p+1, M0-len(GWAS_d[p])))

        # drop snps that are duplicated

        M0 = len(GWAS_d[p])
        GWAS_d[p] = GWAS_d[p].drop_duplicates(subset=args.snp_name, keep='first')
        if M0-len(GWAS_d[p]) > 0:
            logging.info('Trait {}: Dropped {} SNPs for duplicate values in the "snp_name" column'.format(p+1, M0-len(GWAS_d[p])))

    ## Merge summary statistics of GWA studies by snp index

    for p in range(P):

        if p == 0:
            GWAS_all = GWAS_d[p]

        else:
            GWAS_all = GWAS_all.merge(GWAS_d[p], how = 'inner', on=args.snp_name)

            M_0 = len(GWAS_all)
            logging.info('Trait {} summary statistics: \t {} SNPs remaining merging with previous traits.'.format(p+1, M_0))
            if True:
                snps_to_flip = np.logical_and(GWAS_all[args.a1_name+str(0)] == GWAS_all[args.a2_name+str(p)], GWAS_all[args.a2_name+str(0)] == GWAS_all[args.a1_name+str(p)])
                GWAS_all['flip_snps'+str(p)]= snps_to_flip

                logging.debug('Columns after merging :{}'.format(GWAS_all.columns))
                logging.debug(GWAS_all.head(15))
                snps_to_keep = np.logical_or(np.logical_and(GWAS_all[args.a1_name+str(0)]==GWAS_all[args.a1_name+str(p)], GWAS_all[args.a2_name+str(0)]==GWAS_all[args.a2_name+str(p)]), snps_to_flip)

                GWAS_all = GWAS_all[snps_to_keep]
                if len(GWAS_all) < M_0:
                    logging.info('Dropped {} SNPs due to inconsistent allele pairs from phenotype {}. {} SNPs remain.'.format(M_0 - len(GWAS_all),p+1, len(GWAS_all)))

                if np.sum(snps_to_flip) > 0:
                    zz= args.z_name if args.z_name is not None else 'z'
                    freq_name = args.eaf_name if args.eaf_name is not None else 'freq'

                    GWAS_all.loc[snps_to_flip, zz+str(p)] = -1*GWAS_all.loc[snps_to_flip, zz+str(p)]
                    GWAS_all.loc[snps_to_flip, freq_name + str(p)] = 1. - GWAS_all.loc[snps_to_flip, freq_name + str(p)]
                    store_allele = GWAS_all.loc[snps_to_flip, args.a1_name+str(p)]
                    GWAS_all.loc[snps_to_flip, args.a1_name+str(p)] = GWAS_all.loc[snps_to_flip, args.a2_name+str(p)]
                    GWAS_all.loc[snps_to_flip, args.a2_name+str(p)] = store_allele
                    logging.info('Flipped the signs of of {} SNPs to make them consistent with the effect allele orderings of the first trait.'.format(np.sum(snps_to_flip)))
        # tag strand ambiguous SNPs
        # logging.info(GWAS_all.head(15))
        STRAND_AMBIGUOUS_SET = [x for x in allele_info.STRAND_AMBIGUOUS.keys() if allele_info.STRAND_AMBIGUOUS[x]]

        GWAS_all['strand_ambig'] = (GWAS_all[args.a1_name+str(0)].str.upper() + GWAS_all[args.a2_name+str(0)].str.upper()).isin(STRAND_AMBIGUOUS_SET)
        if args.drop_ambig_snps:
            M_0 = len(GWAS_all)
            GWAS_all = GWAS_all[np.logical_not(GWAS_all['strand_ambig'])]
            logging.info('Dropped {} SNPs due to strand ambiguity, {} SNPs remain.'.format(M_0-len(GWAS_all),len(GWAS_all)))

    logging.info('... Merge of GWAS summary statistics complete. Number of SNPs:\t {}'.format(len(GWAS_all)))

    GWAS_orig_cols = GWAS_all.columns
    ## Parses include files
    if args.include is not None:
        for j, include_file in enumerate(args.include.split(',')):
            if j == 0:
                snps_include = _read_SNPlist(include_file, args.snp_name)
            else:
                snps_include = snps_include.merge(_read_SNPlist(include_file,args.snp_name),how='outer', on=args.snp_name)
        GWAS_all = GWAS_all.merge(snps_include, how="left", on = args.snp_name,  indicator="included_merge", suffixes=('','_incl'))
        GWAS_all = GWAS_all.loc[GWAS_all['included_merge']=='both']
        GWAS_all = GWAS_all.loc[:,GWAS_orig_cols]
        logging.info('(--include) Number of SNPs remaining after restricting to SNPs in the union of  {include_path}: \t {M} remain'.format(include_path=args.include,M=len(GWAS_all)))
    ## Parses exclude files
    if args.exclude is not None:
        for exclude_file in args.exclude.split(','):
            snps_exclude = _read_SNPlist(exclude_file, args.snp_name)
            GWAS_all = GWAS_all.merge(snps_exclude, how="left", on = args.snp_name,  indicator="excluded_merge", suffixes=('','_incl'))
            GWAS_all = GWAS_all.loc[GWAS_all['excluded_merge']=='left_only']
            GWAS_all = GWAS_all.loc[:,GWAS_orig_cols]
            logging.info('(-exclude) Number of SNPs remaining after excluding to SNPs in {exclude_path}: \t {M} remain'.format(exclude_path=exclude_file,M=len(GWAS_all)))


    ## Parse chromosomes
    if args.only_chr is not None:
        chr_toInclude = args.only_chr.split(',')
        chr_toInclude = [int(c) for c in chr_toInclude]
        GWAS_all = GWAS_all[GWAS_all[args.chr_name+str(0)].isin(chr_toInclude)]

    ## add information to Namespace
    args.P = P

    return GWAS_all, args

def ldsc_matrix_formatter(result_rg, output_var):
    ''' Key Arguments:
    result_rg - matrix w/ RG objects obtained from estimate_rg (w/ None's on the diagonal)
    output_var - interested variable in the form of '.[VAR_NAME]'
    '''
    output_mat = np.empty_like(result_rg, dtype=float)
    (nrow, ncol) = result_rg.shape
    for i in range(nrow):
        for j in range(ncol):
            if result_rg[i, j] is None:
                output_mat[i, j] = None
            else:
                exec('output_mat[i, j] = result_rg[i, j]{}'.format(output_var))
    return(output_mat)

def estimate_sigma(data_df, args):
    sigma_hat = np.empty((args.P,args.P))

    args.munge_out = args.out+'_ldsc_temp/'
    # Creates data files for munging
    # Munge data
    ignore_list = ""
    if args.info_min is None:
        ignore_list += "info"

    gwas_ss_df = dict()

    for p in range(args.P):
        logging.info('Preparing phenotype {} to estimate sigma'.format(p))

        ld_ss_name = {args.snp_name        : 'SNP',
                      args.a1_name + str(p):  'A1',
                      args.a2_name + str(p):  'A2',
                      args.z_name + str(p):   'Z',
                      args.n_name + str(p):   'N',
                      args.eaf_name + str(p): 'FRQ'}

        # single_colnames = [col for col in data_df.columns if col[-1] == str(p) or col in args.snp_name]
        gwas_ss_df[p] = data_df[ld_ss_name.keys()].copy()

        # gwas_filtered_df= gwas_filtered_df.rename(columns={args.snp_name:args.snp_name+str(p)})
        gwas_ss_df[p] = gwas_ss_df[p].rename(columns=ld_ss_name)
        ## remove phenotype index from names



    # run ldsc
    h2_files = None
    rg_files = args.sumstats
    rg_out = '{}_rg_misc'.format(args.out)
    rg_mat = True

    args_ldsc_rg =  Namespace(out=rg_out, bfile=None,l2=None,extract=None,keep=None, ld_wind_snps=None,ld_wind_kb=None, ld_wind_cm=None,print_snps=None, annot=None,thin_annot=False,cts_bin=None, cts_break=None,cts_names=None, per_allele=False, pq_exp=None, no_print_annot=False,maf=None,h2=h2_files, rg=rg_files,ref_ld=None,ref_ld_chr=args.ld_ref_panel, w_ld=None,w_ld_chr=args.ld_ref_panel,overlap_annot=False,no_intercept=False, intercept_h2=None, intercept_gencov=None,M=None,two_step=None, chisq_max=None,print_cov=False,print_delete_vals=False,chunk_size=50, pickle=False,invert_anyway=False,yes_really=False,n_blocks=200,not_M_5_50=False,return_silly_things=False,no_check_alleles=False,print_coefficients=False,samp_prev=None,pop_prev=None, frqfile=None, h2_cts=None, frqfile_chr=None,print_all_cts=False, sumstats_frames=[ gwas_ss_df[i] for i in range(args.P)], rg_mat=rg_mat)

    if args.no_overlap:
        sigma_hat = np.zeros((args.P, args.P))
        for t in range(args.P):
            args_ldsc_rg.sumstats_frames = [gwas_ss_df[t]]
            rg_results_t = sumstats_sig.estimate_rg(args_ldsc_rg, Logger_to_Logging())
            sigma_hat[t,t] =  ldsc_matrix_formatter(rg_results_t, '.gencov.intercept')[0]
    else:
        rg_results =  sumstats_sig.estimate_rg(args_ldsc_rg, Logger_to_Logging())

        sigma_hat = ldsc_matrix_formatter(rg_results, '.gencov.intercept')

    # if args.no_overlap:
    #     T = sigma_hat.shape[0]
    #     sigma_hat = sigma_hat * np.eye(T)

    # logging.info(type(sigma_hat))
    logging.info(sigma_hat)

    return sigma_hat

def _posDef_adjustment(mat, scaling_factor=0.99,max_it=1000):
    '''
    Checks whether the provided is pos semidefinite. If it is not, then it performs the the adjustment procedure descried in 1.2.2 of the Supplementary Note

    scaling_factor: the multiplicative factor that all off-diagonal elements of the matrix are scaled by in the second step of the procedure.
    max_it: max number of iterations set so that
    '''
    logging.info('Checking for positive definiteness ..')
    assert mat.ndim == 2
    assert mat.shape[0] == mat.shape[1]
    is_pos_semidef = lambda m: np.all(np.linalg.eigvals(m) >= 0)
    if is_pos_semidef(mat):
        return mat
    else:
        logging.info('matrix is not positive definite, performing adjustment..')
        P = mat.shape[0]
        for i in range(P):
            for j in range(i,P):
                if np.abs(mat[i,j]) > np.sqrt(mat[i,i] * mat[j,j]):
                    mat[i,j] = scaling_factor*np.sign(mat[i,j])*np.sqrt(mat[i,i] * mat[j,j])
                    mat[j,i] = mat[i,j]
        n=0
        while not is_pos_semidef(mat) and n < max_it:
            dg = np.diag(mat)
            mat = scaling_factor * mat
            mat[np.diag_indices(P)] = dg
            n += 1
        if n == max_it:
            logging.info('Warning: max number of iterations reached in adjustment procedure. Sigma matrix used is still non-positive-definite.')
        else:
            logging.info('Completed in {} iterations'.format(n))
        return mat

def extract_gwas_sumstats(DATA, args):
    '''

    Output:
    -------
    All matrices are of the shape MxP, where M is the number of SNPs used in MTAG and P is the number of summary statistics results used. Columns are ordered according to the initial ordering of GWAS input files.
    results_template = pd.Dataframe of snp_name chr bpos a1 a2
    Zs: matriix of Z scores
    Ns: matrix of sample sizes
    Fs: matrix of allele frequencies
    '''
    if args.n_name is not None:
        n_cols = [args.n_name +str(p) for p in range(args.P)]
        Ns = DATA.filter(items=n_cols).as_matrix()
    else:
        Ns = DATA.filter(regex='^[nN].').as_matrix()
        args.n_name = 'n'

    # Apply sample-size specific filters

    N_passFilter = np.ones(len(Ns), dtype=bool)

    N_nearMode = np.ones_like(Ns, dtype=bool)
    if args.homogNs_frac is not None or args.homogNs_dist is not None:
        N_modes, _ = _quick_mode(Ns)
        assert len(N_modes) == Ns.shape[1]
        if args.homogNs_frac is not None:
            logging.info('--homogNs_frac {} is on, filtering SNPs ...'.format(args.homogNs_frac))
            assert args.homogNs_frac >= 0.
            homogNs_frac_list = [float(x) for x in args.homogNs_frac.split(',')]
            if len(homogNs_frac_list) == 1:
                homogNs_frac_list = homogNs_frac_list*args.P
            for p in range(args.P):
                N_nearMode[:,p] = np.abs((Ns[:,p] - N_modes[p])) / N_modes[p] <= homogNs_frac_list[p]
        elif args.homogNs_dist is not None:
            logging.info('--homogNs_dist {} is on, filtering SNPs ...'.format(args.homogNs_dist))
            homogNs_dist_list = [float(x) for x in args.homogNs_dist.split(',')]
            if len(homogNs_dist_list) == 1:
                homogNs_dist_list = homogNs_dist_list*args.P

            assert np.all(np.array(homogNs_dist_list) >=0)
            for p in range(args.P):
                N_nearMode[:,p] =  np.abs(Ns[:,p] - N_modes[p]) <= homogNs_dist_list[p]
        else:
            raise ValueError('Cannot specify both --homogNs_frac and --homogNs_dist at the same time.')

        # report restrictions
        mode_restrictions = 'Sample size restrictions close to mode:\n'
        for p in range(Ns.shape[1]):
            mode_restrictions +="Phenotype {}: \t {} SNPs pass modal sample size filter \n".format(p+1,np.sum(N_nearMode[:,p]))

        mode_restrictions+="Intersection of SNPs that pass modal sample size filter for all traits:\t {}".format(np.sum(np.all(N_nearMode, axis=1)))
        logging.info(mode_restrictions)
        N_passFilter = np.logical_and(N_passFilter, np.all(N_nearMode,axis=1))

    if args.n_max is not None:
        n_max_restrictions = "--n_max used, removing SNPs with sample size greater than  {}".format(args.n_max)
        N_passMax = Ns <= args.n_max
        for p in range(Ns.shape[1]):
            n_max_restrictions +=  "Phenotype {}: \t {} SNPs pass modal sample size filter".format(p+1,np.sum(N_passMax[:,p]))
        n_max_restrictions += "Intersection of SNPs that pass maximum sample size filter for all traits:\t {}".format(np.sum(np.all(N_passMax, axis=1)))
        logging.info(n_max_restrictions)
        N_passFilter = np.logical_and(N_passFilter, np.all(N_passMax,axis=1))

    Ns = Ns[N_passFilter]
    DATA = DATA[N_passFilter]

    if args.z_name is not None:
        z_cols = [args.z_name +str(p) for p in range(args.P)]
        Zs = DATA.filter(items=z_cols).as_matrix()
    else:
        Zs = DATA.filter(regex='^[zZ].').as_matrix()
        args.z_name = 'z'
    if args.eaf_name is not  None:
        f_cols = [args.eaf_name + str(p) for p in range(args.P)]

        Fs =DATA.filter(items=f_cols).as_matrix()
    else:
        orig_case_cols = DATA.columns
        DATA.columns = map(str.upper, DATA.columns)

        Fs = DATA.filter(regex='^/MAF|FREQ|FRQ/.').as_matrix()

        args.eaf_name = 'freq'
        DATA.columns = orig_case_cols
    assert Zs.shape[1] == Ns.shape[1] == Fs.shape[1]


    results_template = pd.DataFrame(index=np.arange(len(DATA)))
    results_template.loc[:,args.snp_name] = DATA[args.snp_name]
    # args.chr args.bpos args.alelle_names
    for col in [args.chr_name, args.bpos_name, args.a1_name, args.a2_name]:
        results_template.loc[:,col] = DATA[col+str(0)]
    # TODO: non-error form of integer conversion
    # results_template[args.chr_name] = results_template[args.chr_name].astype(int)
    # results_template[args.bpos_name] = results_template[args.bpos_name].astype(int)

    return Zs, Ns, Fs, results_template, DATA

###########################################
## OMEGA ESTIMATION
##########################################

def jointEffect_probability(Z_score, omega_hat, sigma_hat,N_mats, S=None):
    ''' For each SNP m in each state s , computes the evaluates the multivariate normal distribution at the observed row of Z-scores
    Calculate the distribution of (Z_m | s ) for all s in S, m in M. --> M  x|S| matrix
    The output is a M x n_S matrix of joint probabilities
    '''

    DTYPE = np.float64
    (M,P) = Z_score.shape
    if S is None: # 2D dimensional form
        assert omega_hat.ndim == 2
        omega_hat = omega_hat.reshape(1,P,P)
        S = np.ones((1,P),dtype=bool)

    (n_S,_) = S.shape
    jointProbs = np.empty((M,n_S))

    xRinvs = np.zeros([M,n_S,P], dtype=DTYPE)
    logSqrtDetSigmas = np.zeros([M,n_S], dtype=DTYPE)
    Ls = np.zeros([M,n_S,P,P], dtype=DTYPE)
    cov_s = np.zeros([M,n_S,P,P], dtype=DTYPE)

    Zs_rep = np.einsum('mp,s->msp',Z_score,np.ones(n_S))  # functionally equivalent to repmat
    cov_s = np.einsum('mpq,spq->mspq',N_mats,omega_hat) + sigma_hat

    Ls = np.linalg.cholesky(cov_s)
    Rs = np.transpose(Ls, axes=(0,1,3,2))

    xRinvs = np.linalg.solve(Ls, Zs_rep)

    logSqrtDetSigmas = np.sum(np.log(np.diagonal(Rs,axis1=2,axis2=3)),axis=2).reshape(M,n_S)

    quadforms = np.sum(xRinvs**2,axis=2).reshape(M,n_S)
    jointProbs = np.exp(-0.5 * quadforms - logSqrtDetSigmas - P * np.log(2 * np.pi) / 2)

    if n_S == 1:
        jointProbs = jointProbs.flatten()

    return jointProbs

def gmm_omega(Zs, Ns, sigma_LD):
    logging.info('Using GMM estimator of Omega ..')
    N_mats = np.sqrt(np.einsum('mp,mq->mpq', Ns,Ns))
    Z_outer = np.einsum('mp,mq->mpq',Zs, Zs)
    return np.mean((Z_outer - sigma_LD) / N_mats, axis=0)


def analytic_omega(Zs,Ns,sigma_LD):
    '''
    Closed form solution for Omega when the sample size is constant across all snps for each phenotype. Can serve as an approximation in other cases.

    '''
    M,P = Zs.shape
    N_mean = np.mean(Ns, axis=0)
    N_mats = np.einsum('mp, mq -> mpq', np.sqrt(Ns), np.sqrt(Ns))

    Cov_mean = np.mean(np.einsum('mp,mq->mpq',Zs,Zs) / N_mats, axis=0)
    return Cov_mean - sigma_LD / np.sqrt(np.outer(N_mean,N_mean))

def numerical_omega(args, Zs,N_mats,sigma_LD,omega_start):
    M,P = Zs.shape
    solver_options = dict()
    solver_options['fatol'] = 1.0e-8
    solver_options['xatol'] = args.tol
    solver_options['disp'] = False
    solver_options['maxiter'] = P*250 if args.perfect_gencov else P*(P+1)*500
    if args.perfect_gencov:
        x_start = np.log(np.diag(omega_start))
    else:
        x_start = flatten_out_omega(omega_start)

    opt_results = scipy.optimize.minimize(_omega_neglogL,x_start,args=(Zs,N_mats,sigma_LD,args),method='Nelder-Mead',options=solver_options)

    if args.perfect_gencov:
        return np.sqrt(np.outer(np.exp(opt_results.x), np.exp(opt_results.x))), opt_results
    else:
        return rebuild_omega(opt_results.x), opt_results

def _omega_neglogL(x,Zs,N_mats,sigma_LD,args):
    if args.perfect_gencov:
        omega_it = np.sqrt(np.outer(np.exp(x),np.exp(x)))
    else:
        omega_it = rebuild_omega(x)
    joint_prob = jointEffect_probability(Zs,omega_it,sigma_LD,N_mats)
    return - np.sum(np.log(joint_prob))

def flatten_out_omega(omega_est):
    # stacks the lower part of the cholesky decomposition ROW_WISE [(0,0) (1,0) (1,1) (2,0) (2,1) (2,2) ...]
    P_c = len(omega_est)
    x_chol = np.linalg.cholesky(omega_est)

    # transform components of cholesky decomposition for better optimization
    lowTr_ind = np.tril_indices(P_c)
    x_chol_trf = np.zeros((P_c,P_c))
    for i in range(P_c):
        for j in range(i): # fill in lower triangular components not on diagonal
            x_chol_trf[i,j] = x_chol[i,j]/np.sqrt(x_chol[i,i]*x_chol[j,j])
    x_chol_trf[np.diag_indices(P_c)] = np.log(np.diag(x_chol))  # replace with log transformation on the diagonal
    return tuple(x_chol_trf[lowTr_ind])


def rebuild_omega(chol_elems, s=None):
    '''Rebuild state-dependent Omega given combination of causal states
       cholX_elements are the elements (entered row-wise) of the lower triangular cholesky decomposition of Omega_s

    '''
    if s is None:
        P = int((-1 + np.sqrt(1.+ 8.*len(chol_elems)))/2.)
        s = np.ones(P,dtype=bool)
        P_c = P
    else:
        P_c = int(np.sum(s))
        P = s.shape[1] if s.ndim == 2 else len(s)
    cholL = np.zeros((P_c,P_c))

    cholL[np.tril_indices(P_c)] = np.array(chol_elems)
    cholL[np.diag_indices(P_c)] = np.exp(np.diag(cholL))  # exponentiate the diagnoal so cholL unique
    for i in range(P_c):
        for j in range(i): # multiply by exponentiated diags
            cholL[i,j] = cholL[i,j]*np.sqrt(cholL[i,i]*cholL[j,j])

    omega_c = np.dot(cholL, cholL.T)

    # Expand to include zeros of matrix
    omega = np.zeros((P,P))
    s_caus_ind = np.argwhere(np.outer(s, s))
    omega[(s_caus_ind[:,0],s_caus_ind[:,1])] = omega_c.flatten()
    return omega


def estimate_omega(args,Zs,Ns,sigma_LD, omega_in=None):


    # start_time =time.time()
    logging.info('Beginning estimation of Omega ...')

    M,P = Zs.shape
    N_mats = np.sqrt(np.einsum('mp, mq -> mpq',Ns, Ns))


    if args.perfect_gencov and args.equal_h2:
        logging.info('--perfect_gencov and --equal_h2 option used')
        return np.ones((P,P))

    if args.numerical_omega:
        if omega_in is None: # omega_in serves as starting point
            omega_in = np.zeros((P,P))
            omega_in[np.diag_indices(P)] = np.diag(gmm_omega(Zs,Ns,sigma_LD))

        omega_hat = omega_in

        omega_hat, opt_results = numerical_omega(args, Zs,N_mats, sigma_LD,omega_hat)
        numerical_msg = "\n Numerical optimization of Omega complete:"
        numerical_msg += "\nSuccessful termination? {}".format("Yes" if opt_results.success else "No")
        numerical_msg += "\nTermination message:\t{}".format(opt_results.message)
        numerical_msg += "\nCompleted in {} iterations".format(opt_results.nit)
        logging.info(numerical_msg)
        return omega_hat



    #logL = lambda joint_probs: np.sum(np.log(joint_probs))
    if args.perfect_gencov:
        # if args.equal_h2: # already covered above.
        #     return np.ones((P,P))
        if args.analytic_omega: # if both closed-form solution and solution with perfect covariance desired, then we compute closed form solution and return the outerproduct of the square root of the diagonal with itself.
            logging.info('Using closed-form solution...')
            omega_hat = analytic_omega(Zs,Ns, sigma_LD)
            return np.sqrt(np.outer(np.diag(omega_hat),np.diag(omega_hat)))

        else: # gmm_omega
            omega_hat = _posDef_adjustment(gmm_omega(Zs,Ns,sigma_LD))
            return np.sqrt(np.outer(np.diag(omega_hat), np.diag(omega_hat)))


    elif args.analytic_omega: # analytic solution only.

        return _posDef_adjustment(analytic_omega(Zs,Ns,sigma_LD))

    # else: gmm_omega (default)
    return _posDef_adjustment(gmm_omega(Zs,Ns,sigma_LD))

########################
## MTAG CALCULATION ####
########################

def mtag_analysis(Zs, Ns, omega_hat, sigma_LD):
    logging.info('Beginning MTAG calculations...')
    M,P = Zs.shape

    W_N = np.einsum('mp,pq->mpq',np.sqrt(Ns),np.eye(P))
    W_N_inv = np.linalg.inv(W_N)
    Sigma_N =  np.einsum('mpq,mqr->mpr',np.einsum('mpq,qr->mpr',W_N_inv,sigma_LD),W_N_inv)

    mtag_betas = np.zeros((M,P))
    mtag_se =np.zeros((M,P))

    for p in range(P):
        # Note that in the code, what I call "gamma should really be omega", but avoid the latter term due to possible confusion with big Omega
        gamma_k = omega_hat[:,p]
        tau_k_2 = omega_hat[p,p]
        om_min_gam = omega_hat - np.outer(gamma_k,gamma_k)/tau_k_2

        xx = om_min_gam + Sigma_N
        inv_xx = np.linalg.inv(xx)
        yy = gamma_k/tau_k_2
        W_inv_Z = np.einsum('mqp,mp->mq',W_N_inv,Zs)

        beta_denom = np.einsum('mp,p->m',np.einsum('q,mqp->mp',yy,inv_xx),yy)
        mtag_betas[:,p] = np.einsum('mp,mp->m',np.einsum('q,mqp->mp',yy,inv_xx), W_inv_Z) / beta_denom

        var_denom = np.einsum('mq,q->m',np.einsum('p,mpq->mq',yy,inv_xx),yy)

        mtag_var_p = 1. / var_denom

        mtag_se[:,p] = np.sqrt(mtag_var_p)



    logging.info(' ... Completed MTAG calculations.')
    return mtag_betas, mtag_se


#################
## SAVING RESULTS ##
#########################

def save_mtag_results(args,results_template,Zs,Ns, Fs,mtag_betas,mtag_se):
    '''
    Output will be of the form:

    snp_name z n maf mtag_beta mtag_se mtag_zscore mtag_pval

   '''
    p_values = lambda z: 2*(scipy.stats.norm.cdf(-1.*np.abs(z)))

    M,P  = mtag_betas.shape

    if args.std_betas:
        logging.info('Outputting standardized betas..')

    for p in range(P):
        logging.info('Writing Phenotype {} to file ...'.format(p+1))
        out_df = results_template.copy()
        out_df[args.z_name] = Zs[:,p]
        out_df[args.n_name] = Ns[:,p]
        out_df[args.eaf_name] = Fs[:,p]

        if args.std_betas:
            weights = np.ones(M,dtype=float)
        else:
            weights = np.sqrt( 2*Fs[:,p]*(1. - Fs[:,p]))
        out_df['mtag_beta'] = mtag_betas[:,p] / weights
        out_df['mtag_se'] = mtag_se[:,p] / weights

        out_df['mtag_z'] = mtag_betas[:,p]/mtag_se[:,p]
        out_df['mtag_pval'] = p_values(out_df['mtag_z'])

        if P == 1:
            out_path = args.out +'_trait.txt'
        else:
            out_path = args.out +'_trait_' + str(p+1) + '.txt'


        out_df.to_csv(out_path,sep='\t', index=False)

    if not args.equal_h2:
        omega_out = "\nEstimated Omega:\n"
        omega_out += str(args.omega_hat)
        omega_out += '\n'
        np.savetxt(args.out +'_omega_hat.txt',args.omega_hat, delimiter ='\t')
    else:
        omega_out = "Omega hat not computed because --equal_h2 was used.\n"


    sigma_out = "\nEstimated Sigma:\n"
    sigma_out += str(args.sigma_hat)
    sigma_out += '\n'
    np.savetxt( args.out +'_sigma_hat.txt',args.sigma_hat, delimiter ='\t')

    summary_df = pd.DataFrame(index=np.arange(1,P+1))
    input_phenotypes = [ '...'+f[-16:] if len(f) > 20 else f for f in args.sumstats.split(',')]


    for p in range(P):

        summary_df.loc[p+1,'Trait'] = input_phenotypes[p]
        summary_df.loc[p+1, 'N (max)'] = np.max(Ns[:,p])
        summary_df.loc[p+1, 'N (mean)'] = np.mean(Ns[:,p])
        summary_df.loc[p+1, '# SNPs used'] = int(len(Zs[:,p]))
        summary_df.loc[p+1, 'GWAS mean chi^2'] = np.mean(np.square(Zs[:,p])) / args.sigma_hat[p,p]
        Z_mtag = mtag_betas[:,p]/mtag_se[:,p]
        summary_df.loc[p+1, 'MTAG mean chi^2'] = np.mean(np.square(Z_mtag))
        summary_df.loc[p+1, 'GWAS equiv. (max) N'] = int(summary_df.loc[p+1, 'N (max)']*(summary_df.loc[p+1, 'MTAG mean chi^2'] - 1) / (summary_df.loc[p+1, 'GWAS mean chi^2'] - 1))

    summary_df['N (max)'] = summary_df['N (max)'].astype(int)
    summary_df['N (mean)'] = summary_df['N (mean)'].astype(int)
    summary_df['# SNPs used'] = summary_df['# SNPs used'].astype(int)
    summary_df['GWAS equiv. (max) N'] = summary_df['GWAS equiv. (max) N'].astype(int)

    final_summary = "\nSummary of MTAG results:\n"
    final_summary +="------------------------\n"
    final_summary += str(summary_df.round(3))+'\n'
    final_summary += omega_out
    final_summary += sigma_out

    logging.info(final_summary)
    logging.info(' ')
    logging.info('MTAG results saved to file.')
'''
Functions for maxFDR parallelization
'''
create_S = lambda P: np.asarray(list(itertools.product([False,True], repeat=P)))

# def _FDR_par(func_args):
#     '''
#     FDR methods for parallelization
#     '''
#     probs, t,omega_hat, sigma_hat,S,Ns, coords = func_args
#     if coords[0] % 1000 == 0:
#         logging.info('Calculating for {}: {}'.format(coords, probs))
#     return -1.*compute_fdr(probs, t, omega_hat, sigma_hat, S, Ns)  , coords



def MTAG_var_Z_jt_c(t, Omega, Omega_c, sigma_LD, Ns):

    '''
    Omega: full Omega matrix
    Omega_c: conditional Omega
    Sigma_LD
    N_mean: vector of length of "sample sizes" (1/c**2).

    This formula only works with constant N, etc.
    '''


    T = Ns.shape[1]
    W_N = np.einsum('mp,pq->mpq',np.sqrt(Ns),np.eye(T))
    W_N_inv = np.linalg.inv(W_N)
    Sigma_j =  np.einsum('mpq,mqr->mpr',np.einsum('mpq,qr->mpr',W_N_inv,sigma_LD),W_N_inv)

    gamma_k = Omega[:,t]
    tau_k_2 = Omega[t,t]

    om_min_gam = Omega - np.outer(gamma_k, gamma_k) / tau_k_2
    xx = om_min_gam + Sigma_j
    inv_xx = np.linalg.inv(xx)

    # num_L / R are the same due to symmetry
    num_L = np.einsum('p,mpq->mq', gamma_k / tau_k_2, inv_xx)
    num_R = np.einsum('mpq,q->mp', inv_xx, gamma_k / tau_k_2)


    numer = np.einsum('mp,mp->m', num_L, np.einsum('mpq,mq->mp', Omega_c + Sigma_j, num_R))

    denom = np.einsum('p,mp->m', gamma_k / tau_k_2, np.einsum('mpq,q->mp', inv_xx, gamma_k /tau_k_2))

    return numer / denom


def simplex_walk(num_dims, samples_per_dim):
    """
    A generator that returns lattice points on an n-simplex.
    """
    max_ = samples_per_dim + num_dims - 1
    for c in itertools.combinations(range(max_), num_dims):
        #print(c)
        c = list(c)
        yield np.array([(y - x - 1.) / (samples_per_dim - 1.)
               for x, y in itertools.izip([-1] + c, c + [max_])])



def scale_omega(gen_corr_mat, priors, S=None):
    assert gen_corr_mat.shape[0] == gen_corr_mat.shape[1]
    T = gen_corr_mat.shape[1]
    omega = np.zeros_like(gen_corr_mat)
    if S is None:
        S = create_S(T)
    n_S = len(S)
    for p1 in range(T):
        for p2 in range(T):
            # indices of states that are casual for both traits p1 and p2.
            caus_state = np.arange(n_S)[np.logical_and(S[:, p1], S[:, p2])]
            # print(np.sum(priors[caus_state]))
            omega[p1,p2] = gen_corr_mat[p1,p2] / np.sum(priors[caus_state])

    return omega

def compute_fdr(prob, t, omega, sigma, S, Ns,N_counts, p_threshold):

    z_threshold = scipy.stats.norm.isf(p_threshold / 2.) # magnitude of z-score needed for statistical significance
    n_S, T = S.shape

    omega_TT = scale_omega(omega, prob, S)

    if not is_pos_semidef(omega_TT):
        return np.inf

    Omega_s = np.einsum('st,sr->str',S,S) * omega_TT



    Prob_signif_cond_t = np.zeros(n_S)
    power_state_t = np.zeros_like(Prob_signif_cond_t)


    for k in range(len(S)):
        sd = np.sqrt(MTAG_var_Z_jt_c(t, omega, Omega_s[k,:,:], sigma, Ns)) #ZZZ generalize to take in Omega_s rather than one state at a time .
        Prob_signif_cond_t[k] = np.sum(2*scipy.stats.norm.sf(z_threshold, loc=0, scale = sd)*N_counts) / float(np.sum(N_counts)) # produces m FDR estimates: take average by weighting each unique sample size row with the counts of SNPs with that sample size (weighted average, denominator will be equal to M)

        power_state_t[k] = Prob_signif_cond_t[k] * float(prob[k])

    FDR_val = np.sum(power_state_t[~S[:,t]]) / np.sum(power_state_t)

    return FDR_val


def is_pos_semidef(m):
    if m.shape[0] == 2 and m.shape[1] == 2:
        return np.sqrt(m[0, 0]*m[1,1]) >= np.abs(m[0, 1])
    else:
        eigs =  np.linalg.eigvals(m)

    return np.all(eigs >= 0)


def neglogL_single_SS(x, beta, se, transformed=True):
    '''
    Returns the negative loglikelihood of betas from a spike-slab
    distribution. Used in the numerical optimziation of the `ss_estimation`.

    Arguments:
    ----------
    x: 2-tuple (pi_null, tau). If transformed, `x` consists of bijective transformations of pi_null and tau so that the image of the mapping is all real numbers.
    betas: The Mx1 vector of betas
    se:    The Mx1 vector of standard errors. Must allign with the reported betas.
    transformed: boolean, default True,
                If True, will perform inverse transformation on pi_null, tau so that they return to their "correct" domain.

    '''
    if  transformed:
        prob_null = 1.0 / (1.0 + np.exp(-1 * x[0]))
        tau = np.exp(-x[1])
    else:
        prob_null, tau = x

    causal_pdf = scipy.stats.norm.pdf(beta, loc=0,scale=np.sqrt(tau**2 + se**2))
    noncausal_pdf = scipy.stats.norm.pdf(beta,loc=0, scale = se)

    return -1* np.sum(np.log( (1.0-prob_null)*causal_pdf + prob_null * noncausal_pdf))

    prob_null = 1.0 / (1.0 + np.exp(-1 * x[0]))
    tau = np.exp(-x[1])

    causal_pdf = scipy.stats.norm.pdf(beta, loc=0,scale=np.sqrt(tau**2 + se**2))
    noncausal_pdf = scipy.stats.norm.pdf(beta,loc=0, scale = se)

    return -1* np.sum(np.log( (1.0-prob_null)*causal_pdf + prob_null * noncausal_pdf))



def ss_estimation(args, betas, se, max_iter=1000, tol=1.0e-10,
                  starting_params =(0.1, 1.0e-3),
                  callback=False):
    '''
    Numerically fit the distribution of betas and standard errors to a spike slab distribution.

    Arguments:
    ----------
    betas: The Mx1 vector of betas
    se:    The Mx1 vector of standard errors. Must allign with the reported betas.
    max_iter: int,
            Maximum number of iterations
    tol:    float,
            Tolerance used in numerical optimization (for both fatol, xatol)

    starting_params: 2-tuple: (pi_0, tau_0)
            Starting parameters for optimization. Default is 0.5, 1.0e-3
    callback:       boolean ,default False
            If True, the parameters values will be printed at each step of optimization.
    '''
    M,T = betas.shape

    def cback_print(x):
        logging.info(x)


    def _optim_ss(f_args):
        start_pi, start_tau = starting_params
        x_0 = ( 1.0/(1.0 + np.exp(-start_pi)), -np.log(start_tau) )
        beta_t, se_t = f_args
        solver_opts = dict()
        solver_opts['maxiter'] = max_iter
        solver_opts['fatol'] = tol
        solver_opts['xatol'] = tol
        solver_opts['disp'] = True
        optim_results = scipy.optimize.minimize(neglogL_single_SS, x_0, args=(betas, se,True), method='Nelder-Mead', options=solver_opts, callback=None)

        t_pi, t_tau = optim_results.x
        pi_null =  1.0 / (1.0 + np.exp(-1 * t_pi))
        tau = np.exp(-t_tau)
        return pi_null, tau

    callback = cback_print if callback else None
    arg_list_ss = [(betas[:,t], se[:,t]) for t in range(T)]
    ss_results =  joblib.Parallel(n_jobs = args.cores,
                                          backend='multiprocessing',
                                          verbose=0,
                                          batch_size=1)(joblib.delayed(_optim_ss)(f_args) for f_args in arg_list_ss)
    return ss_results


def causal_prob(probs, S):
    n_S,T = S.shape

def some_causal_for_allT(probs, S):
    # probability of being causal is nonzero for all traits
    n_S, T = S.shape
    # print(probs)
    if not np.all([np.sum(probs[S[:,t]]) > 0 for t in range(T)]):
        return False
    for p1 in range(T):
        for p2 in range(T):
            # indices of states that are casual for both traits p1 and p2.
            caus_state = np.arange(n_S)[np.logical_and(S[:, p1], S[:, p2])]
            # print(np.sum(priors[caus_state]))
            if np.sum(probs[caus_state]) == 0:
                return False
    return True

def _FDR_par(func_args):
    '''
    FDR methods for parallelization
    omega_hat, sigma_hat, S, Ns,
    '''
    probs, omega_hat, sigma_hat, S, Ns, N_counts, p_sig, g, t = func_args
    return compute_fdr(probs, t, omega_hat, sigma_hat, S, Ns, N_counts, p_sig)  , (g,t)

def fdr(args, Ns_f, Zs):
    '''
     Ns: Mx T matrix of sample sizes
    '''
    # M,T = Ns.shape

    # only use unique values
    if not args.grid_file:
        if args.intervals <= 0:
            raise ValueError('spacing of grid points for the max FDR calculation must be a positive integer')

    Ns = np.round(Ns_f) # round to avoid decimals
    Ns_unique, Ns_counts = np.unique(Ns, return_counts=True, axis=0)

    M_eff, T = Ns_unique.shape

    logging.info('T='+str(T))
    S = create_S(T)
    causal_prob = lambda x, SS: np.sum(np.einsum('s,st->st',x,SS),axis=0)

    if args.grid_file is not None:
        prob_grid = np.loadtxt(args.grid_file)
        # exclude rows that don't sum to 1
        prob_grid = prob_grid[(np.sum(prob_grid, axis=1) > 1.) & np.sum(prob_grid, axis=1) < 0]
    else:
        # automate the creation of the probability grid
        # one_dim_interval = np.linspace(0., 1., args.intervals +1)
        prob_grid = simplex_walk(len(S)-1, args.intervals+1)
    # exclude probabilities that have at least one trait with zero pi_causal
    # exclude probabilities that don't yield a valid NPD matrix
    prob_grid = [x for x in prob_grid if some_causal_for_allT(x,S) and is_pos_semidef(scale_omega(args.omega_hat, x,S))]

    if args.fit_ss:
        gwas_se = 1. / np.sqrt(Ns)
        gwas_betas = gwas_se * Zs

        ss_params_list = ss_estimation(args, gwas_betas, gwas_se)
        pi_causal_ss = np.array([1.- x[0] for x in ss_params_list])
        logging.info('Completed estimation of spike-slab parameters resulting in the following causal probabilities')
        for t in range(T):
            logging.info('Trait {}: \t {:.3f}'.format(t, pi_causal_ss[t]))

        # P0 = len(prob_grid)
        prob_grid = [p for p in prob_grid if np.all(np.abs(causal_prob(p)-pi_causal_ss) < (1. / args.intervals) ) ]
        logging.info('{} probabilities remain after restricting to the grid points with causal probabilities within one unit for each trait'.format(len(prob_grid)))
        # P0 = len(prob_grid)


    logging.info('Number of gridpoints to search: {}'.format(len(prob_grid)))

    FDR = -1.23 * np.ones((len(prob_grid), T))

    # performing coarse grid search
    logging.info('Performing grid search using {} cores.'.format(args.cores))


    N_vals = np.mean(Ns, axis=0, keepdims=True) if args.n_approx else Ns_unique
    N_weights = np.ones(1) if args.n_approx else Ns_counts

    # # define parallelization function
    # def _FDR_par(func_args):
    #     '''
    #     FDR methods for parallelization
    #     omega_hat, sigma_hat, S, Ns,
    #     '''
    #     probs, g, t = func_args
    #     return compute_fdr(probs, t, args.omega_hat, args.sigma_hat, S, Ns, args.p_sig)  , (g,t)


    if not args.n_approx:
        assert np.sum(N_weights) == len(Ns)

    arg_list = [(probs, args.omega_hat, args.sigm_hat, S, N_vals,N_weights, args.p_sig, g, t) for t in range(T) for g, probs in enumerate(prob_grid)]
    NN = len(arg_list)
    K = 10
    start_fdr =time.time()
    for k in range(K):
        k0 = int(k*NN / K)
        k1 = int((k+1) * NN / K)
        sublist = arg_list[k0:k1] if k + 1 != K else arg_list[k0:]
        grid_results =  joblib.Parallel(n_jobs = args.cores,
                                          backend='multiprocessing',
                                          verbose=0,
                                          batch_size='auto')(joblib.delayed(_FDR_par)(f_args) for f_args in sublist)
        logging.info('Grid search: {} percent finished for . Time: \t{:.3f} min'.format((k+1)*100./K, (time.time()-start_fdr)/ 60.))
        for i in range(len(grid_results)):
            FDR_gt, coord = grid_results[i] # coord = (g,t)
            FDR[coord[0], coord[1]] = FDR_gt

        np.savetxt(args.out + '_fdr_mat.txt', FDR, delimiter='\t')
        np.savetxt(args.out + '_prob_grid.txt', prob_grid, delimiter='\t')

    # save FDR file once more
    np.savetxt(args.out+'_fdr_mat.txt', FDR, delimiter='\t')
    logging.info('Saved calculations of fdr over grid points in {}'.format(args.out+'_fdr_mat.txt'))

    logging.info(borderline)
    ind_max = np.argmax(FDR, axis=0)
    logging.info('grid point indices for max FDR for each trait: {}'.format(ind_max))
    max_FDR = np.max(FDR, axis=0)
    logging.info('Maximum FDR')
    for t in range(T):
        logging.info('Max FDR of Trait {}: {} at probs = {}'.format(t+1, max_FDR[t], prob_grid[ind_max[t]]))

    logging.info(borderline)
    logging.info('Completed FDR calculations.')

def mtag(args):

    #1. Administrative checks
    if args.equal_h2 and not args.perfect_gencov:
        raise ValueError("--equal_h2 option used without --perfect_gencov. To use --equal_h2, --perfect_gencov must be also be included.")

     ## Instantiate log file and masthead
    logging.basicConfig(format='%(asctime)s %(message)s', filename=args.out + '.log', filemode='w', level=logging.INFO,datefmt='%Y/%m/%d/%I:%M:%S %p')
    if args.stream_stdout:
        logging.getLogger().addHandler(logging.StreamHandler()) # prints to console

    header_sub = header
    header_sub += "Calling ./mtag.py \\\n"
    defaults = vars(parser.parse_args(''))
    opts = vars(args)
    non_defaults = [x for x in opts.keys() if opts[x] != defaults[x]]
    options = ['--'+x.replace('_','-')+' '+str(opts[x])+' \\' for x in non_defaults]
    header_sub += '\n'.join(options).replace('True','').replace('False','')
    header_sub = header_sub[0:-1] + '\n'

    if args.ld_ref_panel is None:
        mtag_path = re.findall(".*/",__file__)[0]
        args.ld_ref_panel = mtag_path+'ld_ref_panel/eur_w_ld_chr/'

    start_time = time.time()  # starting time of analysis
    # take output directory from --out path
    try :
        args.outdir = re.search('.+/', args.out).group(0)
    except AttributeError:
        logging.info('Invalid path used for --out: must have at least one / (use ./[tag] for current directory) and must not end in /')
        raise
    ## TODO Check all input paths
    if not os.path.isdir(args.outdir):
        if args.make_full_path or args.outdir[0] != '/':
            logging.info("Output folder provided does not exist, creating the directory")
            safely_create_folder(args.outdir)
        else:
            raise ValueError('Could not find output directory:\n {} \n at the specified absolute path. To create this directory, use the --make_full_path option.'.format(args.outdir))

    logging.info(header_sub)
    logging.info("Beginning MTAG analysis...")

    # 2. Load Data and perform restrictions
    DATA, args = load_and_merge_data(args)

    # 3. Extract core information from combined GWAS data
    Zs , Ns ,Fs, res_temp, DATA = extract_gwas_sumstats(DATA,args)


    if not args.drop_ambig_snps:
        logging.info('Using {} SNPs to estimate Omega ({} SNPs excluded due to strand ambiguity)'.format(len(Zs)- np.sum(DATA['strand_ambig']), np.sum(DATA['strand_ambig'])))
    not_SA = np.logical_not(np.array(DATA['strand_ambig']))

    # 4. Estimate Sigma
    if args.residcov_path is None:
        logging.info('Estimating sigma..')
        if args.verbose:
            args.sigma_hat = estimate_sigma(DATA[not_SA], args)
        else:
            with DisableLogger():
                args.sigma_hat = estimate_sigma(DATA[not_SA], args)

    else:
        args.sigma_hat = _read_matrix(args.residcov_path)
    args.sigm_hat = _posDef_adjustment(args.sigma_hat)
    logging.info('Sigma hat:\n{}'.format(args.sigm_hat))


    G_mean_c2_adj = np.mean(np.square(Zs),axis=0) / np.diag(args.sigma_hat)
    low_c2 = G_mean_c2_adj < 1.1
    if np.any(low_c2):
        low_c2_msg = 'Mean chi^2 of SNPs used to estimate Omega is low for some SNPs'
        #low_c2_msg += 'Traits {}'.format(' '.join(np.arange(1,args.P+1)[low_c2])) if np.sum(low_c2) > 1 else 'Trait {}'.format(' '.join(np.arange(1,args.P+1)[low_c2]))
        #low_c2_msg += '(= {})'.format(' '.join(G_mean_c2_adj[low_c2]))
        low_c2_msg += 'MTAG may not perform well in this situation.'
        logging.info(low_c2_msg)


    #5. Estimate Omega

    if args.gencov_path is None:
        not_SA = np.logical_not(np.array(DATA['strand_ambig']))
        args.omega_hat = estimate_omega(args, Zs[not_SA], Ns[not_SA], args.sigma_hat)
        logging.info('Completed estimation of Omega ...')
    else:
        args.omega_hat = _read_matrix(args.gencov_path)


    assert args.omega_hat.shape[0] == args.omega_hat.shape[1] == Zs.shape[1] == args.sigma_hat.shape[0] == args.sigma_hat.shape[1]

    #6. Perform MTAG
    mtag_betas, mtag_se = mtag_analysis(Zs, Ns, args.omega_hat, args.sigma_hat)

    #7. Output GWAS_results
    save_mtag_results(args, res_temp,Zs,Ns, Fs,mtag_betas,mtag_se)



    if args.fdr:

        logging.info('Beginning maxFDR calculations. Depending on the number of grid points specified, this might take some time...')

        fdr(args, Ns, Zs)
        ### ZZZ use function fdr(args, Ns)


    logging.info('MTAG complete. Time elapsed: {}'.format(sec_to_str(time.time()-start_time)))




parser = argparse.ArgumentParser(description="\n **mtag: Multitrait Analysis of GWAS**\n This program is the implementation of MTAG method described by Turley et. al. Requires the input of a comma-separated list of GWAS summary statistics with identical columns. It is recommended to pass the column names manually to the program using the options below. The implementation of MTAG makes use of the LD Score Regression (ldsc) for cleaning the data and estimating residual variance-covariance matrix, so the input must also be compatible ./munge_sumstats.py command in the ldsc distribution included with mtag. The default estimation method for the genetic covariance matrix Omega is GMM (as described in the paper). \n\n Note below: any list of passed to the options below must be comma-separated without whitespace.")

# input_formatting = parser.add_argument_group(title="Options")

in_opts = parser.add_argument_group(title='Input Files', description="Input files to be used by MTAG. The --sumstats option is required, while using the other two options take priority of their corresponding estimation routines, if used.")
in_opts.add_argument("--sumstats", metavar="{File1},{File2}...", type=str, nargs='?',required=False, help='Specify the list of summary statistics files to perform multitrait analysis. Multiple files paths must be separated by \",\". Please read the documentation  to find the up-to-date set of acceptable file formats. A general guideline is that any files you pass into MTAG should also be parsable by ldsc and you should take the additional step of specifying the names of the main columns below to avoid reading errors.')
in_opts.add_argument("--gencov_path",metavar="FILE_PATH", default=None, action="store", help="If specified, will read in the genetic covariance matrix saved in the file path below and skip the estimation routine. The rows and columns of the matrix must correspond to the order of the GWAS input files specified. FIles can either be in whitespace-delimited .txt  or .npy format. Use with caution as the genetic covariance matrix specified will be weakly nonoptimal.")
in_opts.add_argument("--residcov_path",metavar="FILE_PATH", default=None, action="store", help="If specified, will read in the residual covariance matrix saved in the file path below and skip the estimation routine. The rows and columns of the matrix must correspond to the order of the GWAS input files specified. FIles can either be in .txt  or .npy format. Use with caution as the genetic covariance matrix specified will be weakly nonoptimal. File must either be in whitespace-delimited .txt  or .npy")

out_opts = parser.add_argument_group(title="Output formatting", description="Set the output directory and common name of prefix files.")

out_opts.add_argument("--out", metavar='DIR/PREFIX', default='./mtag_results', type=str, help='Specify the directory and name prefix to output MTAG results. All mtag results will be prefixed with the corresponding tag. Default is ./mtag_results')
out_opts.add_argument("--make_full_path", default=False, action="store_true", help="option to make output path specified in -out if it does not exist.")

input_formatting = parser.add_argument_group(title="Column names of input files", description="These options manually pass the names of the relevant summary statistics columns used by MTAG. It is recommended to pass these names because only narrow searches for these columns are performed in the default cases. Moreover, it is necessary that these input files be readable by ldsc's munge_sumstats command.")
input_formatting.add_argument("--snp_name", default="snpid", action="store",type=str, help="Name of the single column that provides the unique identifier for SNPs in the GWAS summary statistics across all GWAS results. Default is \"snpid\". This the index that will be used to merge the GWAS summary statistics. Any SNP lists passed to ---include or --exclude should also contain the same name.")
input_formatting.add_argument("--z_name", default=None, help="The common name of the column of Z scores across all input files. Default is to search for columns beginning with the lowercase letter z.")
input_formatting.add_argument("--n_name", default=None, help="the common name of the column of sample sizes in the GWAS summary statistics files. Default is to search for columns beginning with the lowercase letter  n.")
input_formatting.add_argument('--eaf_name',default="freq", help="The common name of the column of minor allele frequencies (MAF) in the GWAS input files. The default is \"freq\".")
input_formatting.add_argument('--chr_name',default='chr', type=str, help="Name of the column containing the chromosome of each SNP in the GWAS input. Default is \"chr\".")
input_formatting.add_argument('--bpos_name',default='bpos', type=str, help="Name of the column containing the base pair of each SNP in the GWAS input. Default is \"bpos\".")
input_formatting.add_argument('--a1_name',default='a1', type=str, help="Name of the column containing the effect allele of each SNP in the GWAS input. Default is \"a1\".")
input_formatting.add_argument('--a2_name',default='a2', type=str, help="Name of the column containing the non-effect allele of each SNP in the GWAS input. Default is \"a2\".")


filter_opts = parser.add_argument_group(title="Filter Options", description="The input summary statistics files can be filtered using the options below. Note that there is some default filtering according to sample size and allele frequency, following the recommendations we make in the corresponding paper. All of these column-based options allow a list of values to be passed of the same length as the number of traits ")
filter_opts.add_argument("--include",default=None, metavar="SNPLIST1,SNPLIST2,..", type=str, help="Restricts MTAG analysis to the union of snps in the list of  snplists provided. The header line must match the SNP index that will be used to merge the GWAS input files.")
filter_opts.add_argument("--exclude", "--excludeSNPs",default=None, metavar="SNPLIST1,SNPLIST2,..", type=str, help="Similar to the --include option, except that the union of SNPs found in the specified files will be excluded from MTAG. Both -exclude and -include may be simultaneously specified, but -exclude will take precedent (i.e., SNPs found in both the -include and -exclude SNP lists will be excluded).")
filter_opts.add_argument('--only_chr', metavar="CHR_A,CHR_B,..", default=None, type=str, action="store", help="Restrict MTAG to SNPs on one of the listed, comma-separated chromosome. Can be specified simultaneously with --include and --exclude, but will take precedent over both. Not generally recommended. Multiple chromosome numbers should be separated by commas without whitespace. If this option is specified, the GWAS summary statistics must also list the chromosome of each SNPs in a column named \`chr\`.")

filter_opts.add_argument("--homogNs_frac", default=None, type=str, action="store", metavar="FRAC", help="Restricts to SNPs within FRAC of the mode of sample sizes for the SNPs as given by (N-Mode)/Mode < FRAC. This filter is not applied by default.")
filter_opts.add_argument("--homogNs_dist", default=None, type=str, action="store", metavar="D", help="Restricts to SNPs within DIST (in sample size) of the mode of sample sizes for the SNPs. This filter is not applied by default.")

filter_opts.add_argument('--maf_min', default='0.01', type=str, action='store', help="set the threshold below SNPs with low minor allele frequencies will be dropped. Default is 0.01. Set to 0 to skip MAF filtering.")
filter_opts.add_argument('--n_min', default=None, type=str, action='store', help="set the minimum threshold for SNP sample size in input data. Default is 2/3*(90th percentile). Any SNP that does not pass this threshold for all of the GWAS input statistics will not be included in MTAG.")
filter_opts.add_argument('--n_max', default=None, type=str, action='store', help="set the maximum threshold for SNP sample size in input data. Not used by default. Any SNP that does not pass this threshold for any of the GWAS input statistics will not be included in MTAG.")
filter_opts.add_argument("--info_min", default=None,type=str, help="Minimim info score for filtering SNPs for MTAG.")
filter_opts.add_argument("--drop_ambig_snps", default=False, action="store_true", help="Drop strand ambiguous SNPs when performing MTAG (they are already not used to estimate Omega or Sigma.")
filter_opts.add_argument("--no_allele_flipping", default=False, action="store_true", help="Prevents flipping the effect sizes of summary statistics when the effect and non-effect alleles are reversed (reletive the first summary statistics file.")

special_cases = parser.add_argument_group(title="Special Cases",description="These options deal with notable special cases of MTAG that yield improvements in runtime. However, they should be used with caution as they will yield non-optimal results if the assumptions implicit in each option are violated.")
special_cases.add_argument('--analytic_omega', default=False, action='store_true', help='Option to turn off the numerical estimation of the genetic VCV matrix in the presence of constant sample size within each GWAS, for which a closed-form solution exists. The default is to typically use the closed form solution as the starting point for the numerical solution to the maximum-likelihood genetic VCV, Use with caution! If any input GWAS does not have constant sample size, then the analytic solution employed here will not be a maximizer of the likelihood function.')
special_cases.add_argument('--no_overlap', default=False, action='store_true', help='Imposes the assumption that there is no sample overlap between the input GWAS summary statistics. MTAG is performed with the off-diagonal terms on the residual covariance matrix set to 0.')
special_cases.add_argument('--perfect_gencov', default=False, action='store_true', help='Imposes the assumption that all traits used are perfectly genetically correlated with each other. The off-diagonal terms of the genetic covariance matrix are set to the square root of the product of the heritabilities')
special_cases.add_argument('--equal_h2', default=False, action='store_true', help='Imposes the assumption that all traits passed to MTAG have equal heritability. The diagonal terms of the genetic covariance matrix are set equal to each other. Can only be used in conjunction with --perfect_gencov')

fdr_opts = parser.add_argument_group(title='Max FDR calculation', description="These options are used for the calculation of an upper bound on the false disovery under the model described in Supplementary Note 1.1.4 of Turley et al. (2017). Note that there is one of three ways to define the space of grid points over which the upper bound is searched. ")

fdr_opts.add_argument('--fdr', default=False, action='store_true', help='Perform max FDR calculations')
# fdr_opts.add_argument(title='--skip-mtag', default=False, action='store_true',) # XXX make option to skip mtag calculations if already done.
# make mutually exclusive group
fdr_opts.add_argument('--grid_file',default=None, action='store', help='Pre-set list of grid points. Users can define a list of grid points over which the search is conducted. The list of grid points should be passed in text file as a white-space delimited matrix of dimnesions, G x S, where G is the number of grid points and S = 2^T is the number of possible causal states for SNPs. States are ordered according to a tree-like recursive structure from right to left. For example, for 3 traits, with the triple TFT denoting the state for which SNPs are causal for State 1, not causal for state 2, and causal for state 3, then the column ordering of probabilities should be: \nFFF FFT FTF FTT TFF TFT TTF TTT\n There should be no headers, or row names in the file. Any rows for which (i) the probabilities do not sum to 1, the prior of a SNP being is causal is 0 for any of the traits, and (iii) the resulting genetic correlation matrix is non positive definite will excluded in the search.')
# XXX rounding to 1e-6 & restandardize.

fdr_opts.add_argument('--fit_ss', default=False, action='store_true', help='This estimates the prior probability that a SNP is null for each trait and then proceeds to restrict the grid search to the set of probability vectors that sum to the prior null for each trait. This is useful for restrict the search space of larger-dimensional traits.')

fdr_opts.add_argument('--intervals', default=10, action='store',type=int, help='Number of intervals that you would like to partition the [0,1] interval. For example example, with two traits and --intervals set 10, then maxFDR will calculated over the set of feasible points in {0., 0.1, 0.2,..,0.9,1.0}^2.')

fdr_opts.add_argument('--cores', default=1, action='store', type=int, help='Number of threads/cores use to compute the FDR grid points for each trait.')

fdr_opts.add_argument('--p_sig', default=5.0e-8, action='store', help='P-value threshold used for statistical signifiance. Default is p=5.0e-8 (genome-wide significance).' )
fdr_opts.add_argument('--n_approx', default=False, action='store_true', help='Speed up FDR calculation by replacing the sample size of a SNP for each trait by the mean across SNPs (for each trait). Recommended.')

# fdr_opts.add_argument('--binned_n', default=False, action='store_true', help='When --n_approx is off, this options allows for a sped-up version of the max_FDR calculation by weighting the power calculations of unique rows.')

# wc = parser.add_argument_group(title='Winner\'s curse adjustment', description='Options related to the winner\'s curse adjustment of estimates of effect sizes from MTAG that could be used when replicating analyses.')
# GWAS or MTAG results?
# maybe both?

misc = parser.add_argument_group(title="Miscellaneous")

misc.add_argument('--ld_ref_panel', default=None, action='store',metavar="FOLDER_PATH", type=str, help='Specify folder of the ld reference panel (split by chromosome) that will be used in the estimation of the error VCV (sigma). This option is passed to --ref-ld-chr and --w-ld-chr when running LD score regression. The default is to use the reference panel of LD scores computed from 1000 Genomes European subjects (eur_w_ld_chr) that is included with the distribution of MTAG')
misc.add_argument('--time_limit', default=100.,type=float, action="store", help="Set time limit (hours) on the numerical estimation of the variance covariance matrix for MTAG, after which the optimization routine will complete its current iteration and perform MTAG using the last iteration of the genetic VCV.")

misc.add_argument('--std_betas', default=False, action='store_true', help="Results files will have standardized effect sizes, i.e., the weights 1/sqrt(2*MAF*(1-MAF)) are not applied when outputting MTAG results, where MAF is the minor allele frequency.")
misc.add_argument("--tol", default=1e-6,type=float, help="Set the relative (x) tolerance when numerically estimating the genetic variance-covariance matrix. Not recommended to change unless you are facing strong runtime constraints for a large number of traits.")
misc.add_argument('--numerical_omega', default=False, action='store_true', help='Option to use the MLE estimator of the genetic VCV matrix, implemented through a numerical routine.')
misc.add_argument('--verbose', default=False, action='store_true', help='When used, will include output from running ldsc scripts as well additional information (such as optimization routine information.')
misc.add_argument('--chunksize', default=1e7, type=int,
                    help='Chunksize for reading in data.')
misc.add_argument('--stream_stdout', default=False, action='store_true', help='Will streat mtag processing on console in addition to writing to log file.')

if __name__ == '__main__':
    start_t = time.time()
    args = parser.parse_args()
    try:
        mtag(args)
    except Exception as e:
        logging.error(e,exc_info=True)
        logging.info('Analysis terminated from error at {T}'.format(T=time.ctime()))
        time_elapsed = round(time.time() - start_t, 2)
        logging.info('Total time elapsed: {T}'.format(T=sec_to_str(time_elapsed)))

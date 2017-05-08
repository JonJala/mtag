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
import os, gzip, bz2, re
import logging
from argparse import Namespace
import munge_sumstats_withSA
from ldsc_mod import munge_sumstats_withoutSA
from ldsc_mod.ldscore import sumstats as sumstats_sig


__version__ = '0.9.0'

borderline = "<><><<>><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"

header ="\n"
header += borderline +"\n"
header += "<>\n"
header += "<> MTAG: Multitrait Analysis of GWAS \n"
header += "<> Version: {}\n".format(str(__version__))
header += "<> (C) 2017 Omeed Maghzian, Raymond Walters, and Patrick Turley\n"
header += "<> Harvard University Department of Economics / Broad Institute of MIT and Harvard\n"
header += "<> GNU General Public License v3\n"
header += "<> ----------------BETA VERSION----------------\n"
header += "<> Note: this tool is currently undergoing rapid revision \n"
header += "<> and may still have critical bugs. It is recommended to run \n"
header += "<> your own QC on the input before using this program. \n"
header += "<> Please email us if you find any errors / are confused about \n"
header += "<> anything! [Email maghzian@nber.org for bugs in the code.]\n"
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

def _read_GWAS_sumstats(GWAS_file):
    '''
    read GWAS summary statistics from file that is in one of the acceptable formats.
    '''
    # TODO read more file types
    return  pd.read_csv(GWAS_file, index_col=False, header=0,delim_whitespace=True)


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
        raise ValueError("{} is not one of the acceptable file paths for reading in matrix-valued objects.")

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

def get_compression(fh):
    '''
    Code from ldsc/munge_sumstats.py
    Read filename suffixes and figure out whether it is gzipped,bzip2'ed or not compressed
    '''
    if fh.endswith('gz'):
        compression = 'gzip'
        openfunc = gzip.open
    elif fh.endswith('bz2'):
        compression = 'bz2'
        openfunc = bz2.BZ2File
    else:
        openfunc = open
        compression = None

    return openfunc, compression


class Logger_to_Logging(object):
    """
    Logger class that write uses logging module and is needed to use munge_sumstats or ldsc from the LD score package.
    """
    def __init__(self):
        logging.info('created Logger instance to pass through ldsc.')
        super(Logger_to_Logging, self).__init__()

    def log(self,x):
        logging.info(x)

def _perform_munge(args, merged_GWAS, GWAS_filepaths,GWAS_initial_input):
    '''
    Wrapper for use of modified munge_sumstats function from ldsc package. Creates ld_temp folder within specified output file path to store munge sumstats, which may be accessed by

    Parameters
    ----------
    args : argparse.Namespace
        Options passed through `mtag` wrapper function.
    merged_GWAS : pd.Dataframe
        The merged set of GWAS summary statistics after the `-include` and `-exclude` SNP list filters have been applied.
    GWAS_files : dict
        Dictionary of the full set of GWAS summary statistics read in (before merges to other SNP lists). Keys are `range(P)` where P is the number summary statistics files read in. Values are Pandas DataFrame objects.
    '''

    # Create folder to store munge sumstats within output folder

    P=len(GWAS_filepaths)

    original_cols = merged_GWAS.columns

    # n_min maf_min info_min

    zz= args.z_name if args.z_name is not None else 'z'

    if args.n_min is not None:
        n_min_list = [float(x) for x in args.n_min.split(',')]
        if len(n_min_list) == 1:
            n_min_list = n_min_list * P
    else:
        n_min_list = [None]*P
    
    if args.maf_min is not None:
        maf_min_list = [float(x) for x in args.maf_min.split(',')]
        if len(maf_min_list) == 1:
            maf_min_list = maf_min_list * P
    else:
        maf_min_list = [None]*P

    if args.info_min is not None:
        info_min_list = [float(x) for x in args.info_min.split(',')]
        if len(info_min_list) == 1:
            info_min_list = info_min_list * P
    else:
        info_min_list = [None]*P



    for p in range(P):
        merge_alleles=None

        # Default minimum n is the same as it is for munge sumstats
        ignore_list = ""
        if args.info_min is None:
            ignore_list += "info"

        argnames = Namespace(sumstats=GWAS_filepaths[p],N=None,N_cas=None,N_con=None,out=args.munge_out+'filtering',maf_min=maf_min_list[p], info_min =info_min_list[p],daner=False, no_alleles=False, merge_alleles=merge_alleles,n_min=n_min_list[p],chunksize=1e7, snp=args.snp_name,N_col=args.n_name, N_cas_col=None, N_con_col = None, a1=None, a2=None, p=None,frq=args.eaf_name,signed_sumstats=zz+',0', info=None,info_list=None, nstudy=None,nstudy_min=None,ignore=ignore_list,a1_inc=False, keep_maf=True, daner_n=False)

        # filtering done with a modified version of munge sumstats that allows for strand ambiguous SNPs. This is a different file than the munge sumstats used in preparation to estimate sigma hat.
        logging.info(borderline)
        logging.info('Munging Trait {}  {}'.format(p+1,borderline[:-17]))
        logging.info(borderline)
        
        filtered_results = munge_sumstats_withSA.munge_GWASinput(argnames)
        merged_GWAS = merged_GWAS.merge(filtered_results, how='inner',left_on =args.snp_name,right_on='SNP',suffixes=('','_ss'))
        merged_GWAS = merged_GWAS[original_cols]
        
        logging.info(borderline)
        logging.info('Munging of Trait {} complete.'.format(p+1))
        logging.info('Total SNPs remaining:\t {}'.format(len(merged_GWAS)))
        logging.info(borderline+'\n')

    return merged_GWAS

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
    GWAS_d = dict()
    for p, GWAS_input in enumerate(GWAS_input_files):
        GWAS_d[p] = _read_GWAS_sumstats(GWAS_input).add_suffix(p)
        logging.info('Read in Trait {} summary statistics from {} ...'.format(p+1,GWAS_input))

        # convert Alleles to uppercase
        for col in [col+str(p) for col in [args.a1_name, args.a2_name]]:
            GWAS_d[p][col] = GWAS_d[p][col].str.upper()
            

    ## Merge summary statistics of GWA studies by snp index
    
    for p in range(P):
        GWAS_d[p] =GWAS_d[p].rename(columns={x+str(p):x for x in GWAS_d[p].columns})
        GWAS_d[p] = GWAS_d[p].rename(columns={args.snp_name+str(p):args.snp_name})

        # Drop SNPs that are missing or duplicated
        missing_snps = GWAS_d[p][args.snp_name].isin(['.'])
        M0 = len(GWAS_d[p])
        GWAS_d[p] = GWAS_d[p][np.logical_not(missing_snps)]
        if M0-len(GWAS_d[p]) > 0:
            logging.info('Trait {}: Dropped {} SNPs for missing values in the "snp_name" column'.format(p+1, M0-len(GWAS_d[p])))

        # drop snps that are duplicated
        duplicate_snps = GWAS_d[p].duplicated(subset=args.snp_name, keep='first')
        M0 = len(GWAS_d[p])
        GWAS_d[p] = GWAS_d[p][np.logical_not( duplicate_snps )]
        if M0-len(GWAS_d[p]) > 0:
            logging.info('Trait {}: Dropped {} SNPs for duplicate values in the "snp_name" column'.format(p+1, M0-len(GWAS_d[p])))


        if p == 0:
            GWAS_all = GWAS_d[p]
            #logging.info('Trait {} summary statistics: \t {} SNPs.'.format(p+1, len(GWAS_d[p])))
            #VALID_SNPS = {x for x in map(lambda y: ''.join(y), itertools.product(BASES, BASES)) if x[0] != x[1] and not STRAND_AMBIGUOUS[x]}
        else:
            GWAS_all = GWAS_all.merge(GWAS_d[p], how = 'inner', on=args.snp_name)

            M_0 = len(GWAS_all)
            logging.info('Trait {} summary statistics: \t {} SNPs remain.'.format(p+1, M_0))
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
                    GWAS_all.loc[snps_to_flip, zz+str(p)] = -1*GWAS_all.loc[snps_to_flip, zz+str(p)]
                    store_allele = GWAS_all.loc[snps_to_flip, args.a1_name+str(p)]
                    GWAS_all.loc[snps_to_flip, args.a1_name+str(p)] = GWAS_all.loc[snps_to_flip, args.a2_name+str(p)]
                    GWAS_all.loc[snps_to_flip, args.a2_name+str(p)] = store_allele
                    logging.info('Flipped the signs of of {} SNPs to make them consistent with the effect allele orderings of the first trait.'.format(np.sum(snps_to_flip))) 
        # tag strand ambiguous SNPs
        # logging.info(GWAS_all.head(15))
        COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        # bases
        BASES = COMPLEMENT.keys()
        STRAND_AMBIGUOUS_SET = {''.join(x) for x in itertools.product(BASES, BASES) if x[0] != x[1] and x[0] == COMPLEMENT[x[1]]}
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

    ## Perform munge using modified ldsc code.

    args.munge_out = args.out+'_ldsc_temp/'

    if not os.path.isdir(args.munge_out):
        safely_create_folder(args.munge_out)
        #args.delete_ldtemp = True
    else:
        pass
        #args.delete_ldtemp = False


    GWAS_all = _perform_munge(args, GWAS_all, GWAS_input_files,GWAS_d)


    ## Parse chromosomes
    if args.only_chr is not None:
        chr_toInclude = args.only_chr.split(',')
        chr_toInclude = [int(c) for c in chr_toInclude]
        GWAS_all = GWAS_all[GWAS_all[args.chr_name+str(0)].isin(chr_toInclude)]

    ## add information to Namespace
    args.P = P

    return GWAS_all, args

def estimate_sigma(data_df, args):
    sigma_hat = np.empty((args.P,args.P))
    save_paths_premunge = dict()
    save_paths_postmunge = dict()
    # Creates data files for munging
    # Munge data
    ignore_list = ""
    if args.info_min is None:
        ignore_list += "info"

    for p in range(args.P):
        logging.info('Preparing phenotype {} to estimate sigma'.format(p))
        single_colnames = [col for col in data_df.columns if col[-1] == str(p) or col in args.snp_name]

        gwas_filtered_df = data_df[single_colnames]
        gwas_filtered_df= gwas_filtered_df.rename(columns={args.snp_name:args.snp_name+str(p)})
        gwas_filtered_df.columns = [col[:-1] for col in gwas_filtered_df.columns]
        ## remove phenotype index from names


        save_paths_premunge[p] = args.munge_out + '_sigma_est_preMunge' +str(p) +'.csv'
        save_paths_postmunge[p] = args.munge_out + '_sigma_est_postMunge' + str(p)
        gwas_filtered_df.to_csv(save_paths_premunge[p], sep='\t',index=False)

        # we can manually many of the munge filters because the summary statistics have already been filtered.
        args_munge_sigma = Namespace(sumstats=save_paths_premunge[p],N=None,N_cas=None,N_con=None,out=save_paths_postmunge[p],maf_min=0, info_min =0.9,daner=False, no_alleles=False, merge_alleles=None,n_min=0,chunksize=1e7, snp=args.snp_name,N_col=args.n_name, N_cas_col=None, N_con_col = None, a1=None, a2=None, p=None,frq=args.eaf_name,signed_sumstats=args.z_name+',0',info=None,info_list=None, nstudy=None,nstudy_min=None,ignore=ignore_list,a1_inc=False, keep_maf=True, daner_n=False)
        munge_sumstats_withoutSA.munge_sumstats(args_munge_sigma)

    # run ldsc
    for p1 in range(args.P):
        for p2 in range (p1,args.P): #TODO make p1->p1+1 and use h2 estimates
            if (p1 == p2 and args.no_overlap) or not args.no_overlap:
                h2_files = None
                rg_files = '{X}.sumstats.gz,{Y}.sumstats.gz'.format(X=save_paths_postmunge[p1],Y=save_paths_postmunge[p2])
                rg_out = '{}_rg_{}_{}'.format(args.munge_out, p1,p2)
                args_ldsc_rg =  Namespace(out=rg_out, bfile=None,l2=None,extract=None,keep=None, ld_wind_snps=None,ld_wind_kb=None, ld_wind_cm=None,print_snps=None, annot=None,thin_annot=False,cts_bin=None, cts_break=None,cts_names=None, per_allele=False, pq_exp=None, no_print_annot=False,maf=args.maf_min,h2=h2_files, rg=rg_files,ref_ld=None,ref_ld_chr=args.ld_ref_panel, w_ld=None,w_ld_chr=args.ld_ref_panel,overlap_annot=False,no_intercept=False, intercept_h2=None, intercept_gencov=None,M=None,two_step=None, chisq_max=None,print_cov=False,print_delete_vals=False,chunk_size=50, pickle=False,invert_anyway=False,yes_really=False,n_blocks=200,not_M_5_50=False,return_silly_things=False,no_check_alleles=False,print_coefficients=False,samp_prev=None,pop_prev=None, frqfile=None, h2_cts=None, frqfile_chr=None,print_all_cts=False)
                rg_results =  sumstats_sig.estimate_rg(args_ldsc_rg, Logger_to_Logging())[0]
                sigma_hat[p1,p2] = rg_results.gencov.intercept
                sigma_hat[p2,p1] = sigma_hat[p1,p2]




    return sigma_hat

def _posDef_adjustment(mat, scaling_factor=0.99,max_it=1000):
    '''
    Checks whether the provided is pos semidefinite. If it is not, then it performs the the adjustment procedure descried in 1.2.2 of the Supplementary Note

    scaling_factor: the multiplicative factor that all off-diagonal elements of the matrix are scaled by in the second step of the procedure.
    max_it: max number of iterations set so that 
    '''
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
    #logL = lambda joint_probs: np.sum(np.log(joint_probs))
    if args.perfect_gencov:
        if args.equal_h2:
            return np.ones((P,P))
        elif args.analytic_omega: # if both closed-form solution and solution with perfect covariance desired, then we compute closed form solution and return the outerproduct of the square root of the diagonal with itself.
            logging.info('Using closed-form solution...')
            omega_hat = analytic_omega(Zs,Ns, sigma_LD)
            return np.sqrt(np.outer(np.diag(omega_hat),np.diag(omega_hat)))

        elif args.gmm_omega: 
            omega_hat = _posDef_adjustment(gmm_omega(Zs,Ns,sigma_LD))
            return np.sqrt(np.outer(np.diag(omega_hat), np.diag(omega_hat)))
    # XXX add --gmm_omega implementation


    elif args.analytic_omega: # analytic solution only.

        return _posDef_adjustment(analytic_omega(Zs,Ns,sigma_LD))

    elif args.gmm_omega:
        return _posDef_adjustment(gmm_omega(Zs,Ns,sigma_LD))
    # want analytic solution
    if omega_in is None: # omega_in serves as starting point
        omega_in = np.zeros((P,P))
        omega_in[np.diag_indices(P)] = np.diag(gmm_omega(Zs,Ns,sigma_LD))

    #logL_list = [logL(jointEffect_probability(Zs,omega_in,sigma_LD,N_mats))]
    #print(omega_in)
    omega_hat = omega_in
    # num_iter =0
    #while (time.time()-start_time)/3600 <= args.time_limit:
        # numerical solution
    #    logging.info('Iteration: {} ..'.format(num_iter))
    omega_hat, opt_results = numerical_omega(args, Zs,N_mats, sigma_LD,omega_hat)
    numerical_msg = "\n Numerical optimization of Omega complete:"
    numerical_msg += "\nSuccessful termination? {}".format("Yes" if opt_results.success else "No")
    numerical_msg += "\nTermination message:\t{}".format(opt_results.message)
    numerical_msg += "\nCompleted in {} iterations".format(opt_results.nit)
    logging.info(numerical_msg)



    #joint_prob = jointEffect_probability(Zs,omega_hat,sigma_LD,N_mats)
        # logL_list.append(logL(joint_prob))
    #    num_iter += 1
        # check that logL increasing

    #    if np.abs(logL_list[-1]-logL_list[-2]) < args.tol:
     #       break

    return omega_hat

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


        # inv_xx_S_inv_xx = np.einsum('mpq,mqr->mpr',np.einsum('mpq,mqr->mpr',inv_xx,Sigma_N), inv_xx)
        # var_denom = np.square(np.einsum('mq,q->m',np.einsum('p,mpq->mq',yy,inv_xx),yy))
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
    p_values = lambda z: 2*(1.0-scipy.stats.norm.cdf(np.abs(z)))

    M,P  = mtag_betas.shape

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



def mtag(args):


    #1. Administrative checks
    if args.equal_h2 and not args.perfect_gencov:
        raise ValueError("--equal_h2 option used without --perfect_gencov. To use --equal_h2, --perfect_gencov must be also be included.")


    # args.outdir = args.outdir if args.outdir[-1] in ['/','\\'] else args.outdir + '/'



     ## Instantiate log file and masthead
    logging.basicConfig(format='%(asctime)s %(message)s', filename=args.out + '.log', filemode='w', level=logging.INFO,datefmt='%Y/%m/%d %I:%M:%S %p')

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

     #2. Load Data and perform restrictions
    DATA, args = load_and_merge_data(args)

    #3. Extract core information from combined GWAS data
    Zs , Ns ,Fs, res_temp, DATA = extract_gwas_sumstats(DATA,args)

    #4. Estimate Sigma
    if args.residcov_path is None:
        logging.info('Estimating sigma..')
        if args.verbose:
            args.sigma_hat = estimate_sigma(DATA, args)
        else:
            with DisableLogger():
                args.sigma_hat = estimate_sigma(DATA, args)

    else:
        args.sigma_hat = _read_matrix(args.residcov_path)
    args.sigm_hat = _posDef_adjustment(args.sigma_hat)
    logging.info('Sigma hat:\n{}'.format(args.sigm_hat))
    
    ## Warning of low chi2:

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
        if not args.drop_ambig_snps:
            logging.info('Using {} SNPs to estimate Omega ({} SNPs excluded due to strand ambiguity)'.format(len(Zs)- np.sum(DATA['strand_ambig']), np.sum(DATA['strand_ambig'])))
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

    logging.info('MTAG complete. Time elapsed: {}'.format(sec_to_str(time.time()-start_time)))

parser = argparse.ArgumentParser(description="\n **mtag: Multitrait Analysis of GWAS**\n This program is the implementation of MTAG method described by Turley et. al. Requires the input of a comma-separated list of GWAS summary statistics with identical columns. It is recommended to pass the column names manually to the program using the options below. The implementation of MTAG makes use of the LD Score Regression (ldsc) for cleaning the data and estimating residual variance-covariance matrix, so the input must also be compatible ./munge_sumstats.py command in the ldsc distribution included with mtag. \n\n Note below: any list of passed to the options below must be comma-separated without whitespace.")

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
misc = parser.add_argument_group(title="Miscellaneous")

misc.add_argument('--ld_ref_panel', default=None, action='store',metavar="FOLDER_PATH", type=str, help='Specify folder of the ld reference panel (split by chromosome) that will be used in the estimation of the error VCV (sigma). This option is passed to --ref-ld-chr and --w-ld-chr when running LD score regression. The default is to use the reference panel of LD scores computed from 1000 Genomes European subjects (eur_w_ld_chr) that is included with the distribution of MTAG')
misc.add_argument('--time_limit', default=100.,type=float, action="store", help="Set time limit (hours) on the numerical estimation of the variance covariance matrix for MTAG, after which the optimization routine will complete its current iteration and perform MTAG using the last iteration of the genetic VCV.")

misc.add_argument('--std_betas', default=False, action='store_true', help="Results files will have standardized effect sizes, i.e., the weights 1/sqrt(2*MAF*(1-MAF)) are not applied when outputting MTAG results, where MAF is the minor allele frequency.")
misc.add_argument("--tol", default=1e-6,type=float, help="Set the relative (x) tolerance when numerically estimating the genetic variance-covariance matrix. Not recommended to change unless you are facing strong runtime constraints for a large number of traits.")

to_add = parser.add_argument_group(title="Options to add", description="Options that will (probably) in newer versions in mtag. PLEASE DO NOT USE ANY OF THESE OPTIONS. They either (i) do nothing (ii) give errors, or (iii) have not been seriously tested.")
to_add.add_argument('--gmm_omega', default=False, action='store_true', help='Option to use the GMM estimator of the genetic VCV matrix. Much faster than using numerical estimation. This option is still being tested.')
to_add.add_argument('--cc_Z', default=None, metavar="pval_name", action='store', help="Option to use Z-scores backed out by the p-values in the input summary statistics. The --z_name column will be used to determine the sign of the effect. Useful when you would like to apply MTAG to case-control GWAS results but the p-values are derived from likelihood ratio tests. Must give --cc_Z the name of the column containing the p-values in each file.")
to_add.add_argument('--verbose', default=False, action='store_true', help='When used, will include output from running ldsc scripts as well additional information (such as optimization routine information.')





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

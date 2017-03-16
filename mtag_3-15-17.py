#!/usr/bin/env python
'''
'''

from __future__ import division
from analyze_with_mtgwas import mtgwas_completeAnalysis
import numpy as np
import pandas as pd
import itertools
import argparse
import time
import os, sys
import warnings
import logging
from argparse import Namespace
from ldsc_mod import munge_sumstats_withSA
from ldsc_mod import munge_sumstats_withoutSA
from ldsc_mod.ldscore import sumstats as sumstats_sig


# Check version of programs?
# add datat types for all options

__version__ = '1.0.0'
header = "<><><<>><><><><><><><><><><><><><><><><><><><><><><><><><><><>\n"
header += "<><>\n"
header += "<>\t MTAG: Multitrait Analysis of GWAS \n"
header += "<>\t (C) 2017 Patrick Turley and Omeed Maghzian\n"
header += "<>\t S.S. GAC\n"
header += "<><>\n"
header += "<><><<>><><><><><><><><><><><><><><><><><><><><><><><><><><><>\n"

pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 800)
pd.set_option('precision', 6)
pd.set_option('max_colwidth', 800)

np.set_printoptions(linewidth=800)
np.set_printoptions(precision=4)

## General helper functions
def safely_create_folder(folder_path):
    try:
        os.makedirs(folder_path)
    except OSError:
        if not os.path.isdir(folder_path):
            raise


## Analysis / Computation related functions



## Read / Write functions
def _read_SNPlist(file_path, SNP_index):

    # XXX Add more possible ways of reading SNPlists
    return pd.read_csv(file_path, head=0, index_col=False)
def _read_GWAS_sumstats(GWAS_file):
    '''
    read GWAS summary statistics from file that is in one of the acceptable formats
    '''
    # XXX
    return  pd.read_csv(GWAS_file, index_col=False, header=0,delim_whitespace=True)

## LDSC related functions

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

def _perform_munge(args, merged_GWAS, GWAS_filepaths,GWAS_intial_input):
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
    args.munge_out = args.outdir+'ldsc_temp/'

    if not os.path.isdir(args.munge_out):
        safely_create_folder(args.munge_out)


    original_cols = merged_GWAS.columns

    for p in range(len(GWAS_filepaths)):
        #if args.cptid:
        #    GWAS_files[p]['cptid'] = df[['chr', 'bpos']].apply(lambda x: ':'.join(x), axis=1)

        merge_alleles=None

        # Minimum n default different from that of ldsc
        n_min = GWAS_initial_input[p][args.n_name+str(p)].quantile(0.9)*.75 if args.n_min is None else args.n_min

        ignore_list = ""
        if info_min is None:
            ignore_list += "info"


        argnames = Namespace(sumstats=GWAS_filepaths[p],N=None,N_cas=None,N_con=None,out=args.munge_out+'filtering',maf_min=args.maf_min, info_min =args.info_min,daner=False, no_alleles=True, merge_alleles=merge_alleles,n_min=n_min,chunksize=1e7, snp=args.snp_name,N_col=args.n_name, N_cas_col=None, N_con_col = None, a1=None, a2=None, p=None,frq=args.maf_name,signed_sumstats=args.z_name,info=args.info_min,info_list=None, nstudy=None,nstudy_min=None,ignore=ignore_list,a1_inc=False, keep_maf=True)
        filtered_results = munge_sumstats_withSA.munge_sumstats(argnames)

        GWAS_files[p] = merged_GWAS.merge(filtered_results, how='inner',left_on =snp_name,right_on='SNP',suffixes=('','_ss'))
        merged_GWAS = merged_GWAS[original_cols]
        logging.info('Completed munging (modified ldsc code) of Phenotype {}...'.format(p))
        # munge sumstats with SA
    return merged_GWAS


def load_and_merge_data(args):
    '''
    XXX Add documentation
    Parses file names from MTAG command line arguments and returns the relevant used for method.
    '''
    args.munge_out = args.outdir+'ldsc_temp/'

    GWAS_input_files = args.GWAS_results.split(',')
    P = len(GWAS_input_files)  # of phenotypes
    GWAS_d = dict()
    for p, GWAS_input in enumerate(GWAS_input_files):
        GWAS_d[p] = _read_GWAS_sumstats(GWAS_input).add_suffix(p)
        logging.info('Read in Phenotype {} from {} ...'.format(p,GWAS_input))

    ## Merge summary statistics of GWA studies by snp index
    # SNP_index = ['chr', 'bpos'] if args.cptid else ['snpid']
    #SNP_index = 'cptid' if args.cptid else ['snpid'

    for p in range(P):
        GWAS_d[p] =GWAS_d[p].rename({x+str(p):x for x in GWAS_d[p].columns})
        GWAS_d[p] = GWAS_d[p].rename({args.snp_name+str(p):args.snp_name})
        if p == 0:
            GWAS_all = GWAS_d[p]
        else:
            GWAS_all = GWAS_all.merge(GWAS_d[p], how = 'inner', on=args.snp_name )

            # XXX Perform checks on consistency of data across summary statistics

            # A. Check if reference alleles swtiched
    logging.info('... Merge of GWAS summary statistics complete. Number of SNPs:\t {}'.format(len(GWAS_all)))

    GWAS_columns = GWAS_all.columns
    ## Parses include files
    if args.include is not None:
        for include_file in args.include.split(','):
            snps_include = _read_SNPlist(include_file, args.snp_name)
            GWAS_all = GWAS_all.merge(snps_include, how="left", on = args.snp_name,  indicator="included_merge", suffixes=('','_incl'))
            GWAS_all = GWAS_all.loc[GWAS_all['included_merge']=='both']
            GWAS_all = GWAS_all.loc[:,GWAS_columns]
            logging.info('(-include) Number of SNPs remaining after restricting to SNPs in {include_path}: \t {M} remain'.format(include_path=include_file,M=len(GWAS_all)))
    ## Parses exclude files
    if args.exclude is not None:
        for exclude_file in args.exclude.split(','):
            snps_exclude = _read_SNPlist(exclude_file, SNP_index)
            GWAS_all = GWAS_all.merge(snps_exclude, how="left", on = args.snp_name,  indicator="excluded_merge", suffixes=('','_incl'))
            GWAS_all = GWAS_all.loc[GWAS_all['excluded_merge']=='left_only']
            GWAS_all = GWAS_all.loc[:,GWAS_columns]
            logging.info('(-exclude) Number of SNPs remaining after excluding to SNPs in {exclude_path}: \t {M} remain'.format(exclude_path=exclude_file,M=len(GWAS_all)))

    ## Perform munge using modified ldsc code.

    GWAS_all = _perfom_munge(args, GWAS_all, GWAS_input_files,GWAS_d)


    ## Parse chromosomes
    if args.only_chr is not None:
        chr_toInclude = args.only_chr.split(',')
        chr_toInclude = [int(c) for c in chr_toInclude]
        GWAS_all = GWAS_all[GWAS_all['chr'].isin(chr_toInclude)]

    ## add information to Namespace
    args.P = P

    return GWAS_all, args

def estimate_sigma(data_df, args):
    sigma_hat = np.empty((args.P,args.P))
    save_paths_premunge = dict()
    save_paths_postmunge = dict()
    # Creates data files for munging
    # Munge data
    for p in range(args.P):
        logging.info('Preparing phenotype {} to estimate sigma'.format(p))
        single_colnames = [col for col in data_df.columns if col[-1] == str(p) or col in args.snp_index]
        gwas_filtered_df = data_df[single_colnames]
        gwas_filtered_df= gwas_filtered_df.rename({args.snp_index:args.snp_index+str(p)})
        gwas_filtered_df.columns = [col[:-1] for col in gwas_filtered_df.columns]
        ## remove phenotype index from names


        save_paths_premunge[p] = args.munge_out + '_sigma_est_preMunge' +str(p) +'.csv'
        save_paths_postmunge[p] = args.munge_out + '_sigma_est_postMunge' + str(p)
        gwas_filtered_df.to_csv(save_paths_premunge[p], sep='\t',index=False)

        args_munge_sigma = Namespace(sumstats=save_paths_premunge[p],N=None,N_cas=None,N_con=None,out=save_paths_postmunge[p],maf_min=args.maf_min, info_min =args.info_min,daner=False, no_alleles=True, merge_alleles=None,n_min=0,chunksize=1e7, snp=args.snp_name,N_col=args.n_name, N_cas_col=None, N_con_col = None, a1=None, a2=None, p=None,frq=args.maf_name,signed_sumstats=args.z_name,info=args.info_min,info_list=None, nstudy=None,nstudy_min=None,ignore=ignore_list,a1_inc=False, keep_maf=True)
        munge_sumstats_withoutSA.munge_sumstats(args_munge_sigma)

    # run ldsc
    for p1 in range(args.P):
        for p2 in range (p1,args.P): #XXX make p1->p1+1
            h2_files = None
            rg_files = '{X}.sumstats.gz,{Y}.sumstats.gz'.format(X=save_paths_postmunge[p1],Y=save_paths_postmunge[p2])
            args_ldsc_rg =  Namespace(out=out_est_path, bfile=None,l2=None,extract=None,keep=None, ld_wind_snps=None,ld_wind_kb=None, ld_wind_cm=None,print_snps=None, annot=None,thin_annot=False,cts_bin=None, cts_break=None,cts_names=None, per_allele=False, pq_exp=None, no_print_annot=False,maf=args.maf_min,h2=h2_files, rg=rg_f,ref_ld=None,ref_ld_chr=args.ld_ref_panel, w_ld=None,w_ld_chr=args.ld_ref_panel,overlap_annot=False,no_intercept=False, intercept_h2=None, intercept_gencov=None,M=None,two_step=None, chisq_max=None,print_cov=False,print_delete_vals=False,chunk_size=50, pickle=False,invert_anyway=False,yes_really=False,n_blocks=200,not_M_5_50=False,return_silly_things=False,no_check_alleles=False,print_coefficients=False,samp_prev=None,pop_prev=None, frqfile=None, h2_cts=None, frqfile_chr=None,print_all_cts=False)
            rg_results =  sumstats_sig.estimate_rg(args_ldsc_rg, Logger_to_Logging())[0]
            sigma_hat[p1,p2] = rg_results.gencov.intercept
            sigma_hat[p2,p1] = sigma_hat[p1,p2]

    return sigma_hat

def extract_gwas_sumstats(DATA, args):
    '''

    Output:
    -------
    All matrices are of the shape MxP, where M is the number of SNPs used in MTAG and P is the number of summary statistics results used. Columns are ordered according to the initial ordering of GWAS input files.
    Zs: matriix of Z scores
    Ns: matrix of sample sizes
    Fs: matrix of allele frequencies
    '''
    if args.z_name is not None:
        z_cols = [args.z_name +str(p) for p in range(args.P)]
        Zs = DATA.filter(items=z_cols).as_matrix()
    else:
        Zs = DATA.filter(regex='^[zZ].').as_matrix()

    if args.n_name is not None:
        n_cols = [args.n_name +str(p) for p in range(args.P)]
        Ns = DATA.filter(items=n_cols).as_matrix()
    else:
        Ns = DATA.filter(regex='^[nN].').as_matrix()

    if args.maf_name is None:
        f_cols = [args.maf_name + str(p) for p in range(args.P)]
        Fs =DATA.filter(items=f_cols).as_matrix()
    else:
        DATA.columsn = map(str.upper, DATA.columns)
        Fs = DATA.filter(regex='^[(MAF)(FREQ)(FRQ)].').as_matrix()

    return Zs, Ns, Fs

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



def analytic_omega(Zs,Ns,sigma_LD):
    '''
    Closed form solution for Omega when the sample size is constant across all snps for each phenotype. Can serve as an approximation in other cases.

    '''
    M,P = Zs.shape
    N_mean = np.mean(Ns, axis=0)
    N_mats = np.einsum('mp, mq -> mpq', np.sqrt(Ns), np.sqrt(Ns))
    N_mean_outer = np.outer(N_mean)

    Cov_mean = np.mean(np.einsum('mp,mq->mpq',Zs,Zs) / N_mats, axis=0)
    return Cov_mean - sigma_LD / np.sqrt(np.outer(N_mean,N_mean))
    W_N = np.einsum('mp,pq->mpq',np.sqrt(Ns),np.eye(P))

def numerical_omega(Zs,N_mats,sigma_LD,omega_start):
    M,P = Zs.shape

    solver_options = dict()
    solver_options['fatol'] = 1.0e-30
    solver_options['xatol'] = 1.0e-15
    solver_options['disp'] = True
    x_start = flatten_out_omega(omega_start)

    opt_results = scipy.optimize.minimize(_omega_neglogL,x_start,args=(Zs,N_mats,sigma_LD),method='Nelder-Mead',options-solver_options)

    return rebuild_omega(opt_results.x)


def _omega_neglogL(x,Zs,N_mats,sigma_LD):
    P = Zs.shape[1]
    omega_it = rebuild_omega(x)
    joint_prob = jointEffect_probability(Zs,omega_it,sigma_LD,N_mats)
    return - np.sum(np.log(joint_prob))

def flatten_out_omega(omega_est):
    # stacks the lower part of the cholesky decomposition ROW_WISE [(0,0) (1,0) (1,1) (2,0) (2,1) (2,2) ...]
    P_c = len(omega_est)
    x_chol = np.linalg.cholesky(omega_est)

    # transform components of cholesky decomposition for better optimization
    lowTr_ind = np.tril_indices(P_c)
    x_chol_trf = np.zeros((P_causal,P_causal))
    for i in range(P_causal):
        for j in range(i): # fill in lower triangular components not on diagonal
            x_chol_trf[i,j] = x_chol[i,j]/np.sqrt(x_chol[i,i]*x_chol[j,j])
    x_chol_trf[np.diag_indices(P_c)] = np.log(np.diag(x_chol))  # replace with log transformation on the diagonal
    return tuple(x_chol_trf[lowTr_ind])


def rebuild_omega(chol_elems, s=None):
    '''Rebuild state-dependent Omega given combination of causal states
       cholX_elements are the elements (entered row-wise) of the lower triangular cholesky decomposition of Omega_s

    '''
    if s is None:
        P = len(chol_elems)*(len(chol_elems)-1)/2
        s = np.ones(P,dtype=bool)
        P = P_c
    else:
        P_c = np.sum(s)
        P = s.shape[1] if s.ndim == 2 else len(s)
    cholL = np.zeros((P_c,P_c))
    offDiag_ind = np.tril_indices(P_c,k=-1)
    cholL[np.tril_indices(P_c)] = np.array(chol_elems)
    cholL[np.diag_indices(P_c)] = np.exp(np.diag(cholL))  # exponentiate the diagnol so cholL unique
    for i in range(P_c):
        for j in range(i): # multiply by exponentiated diags
            cholL[i,j] = cholL[i,j]*np.sqrt(cholL[i,i]*cholL[j,j])

    omega_c = np.dot(cholL, cholL.T)

    # Expand to include zeros of matrix
    omega = np.zeros((P,P))
    s_caus_ind = np.argwhere(np.outer(s, s))
    omega[(s_caus_ind[:,0],s_caus_ind[:,1])] = omega_c.flatten()
    return omega


def estimate_omega(args,Zs,Ns,sigma_LD, omega_in=None,time_limit=100):
    start_time =time.time()
    logging.info('Beginning estimation of Omega ...')
    M,P = Zs.shape
    N_mats = np.sqrt(np.einsum('mp, mq -> mpq',Ns, Ns))
    logL = lambda joint_probs: np.sum(np.log(joint_probs))
    if False: # analytic solution only.
        return analytic_omega(Zs,Ns,sigma_LD)

    # want analytic solution
    if omega_in is None: # omega_in serves as starting point
        omega_in = analytic_omega(Zs,Ns,sigma_LD)

    logL_list = [logL(jointEffect_probability(Zs,omega_in,sigma_LD,N_mats))]

    omega_hat = omega_in
    while (time.time()-start_time)/3600 <= time_limit:
        # numerical solution
        omega_hat = numerical_omega(Zs,N_mats,sigma_hat,omega_hat)
        joint_prob = jointEffect_probability(Zs,omega_hat,sigma_LD,N_mats)
        logL_list.append(logL(joint_prob))
        # check that logL increasing
        if np.abs(logL_list[-1]-logL_list[-2]) < args.tol:
            break


    logging.info('Completed estimation of Omega ...')

    return omega_hat

########################
## MTAG CALCULATION ####
########################

def mtag_analysis(args, Zs, Ns, omega_hat, sigma_LD):
    logging.info('Beginning MTAG calculations...')
    M,P = Zs.shape
    N_mats = np.sqrt(np.einsum('mp, mq -> mpq', Ns, Ns))

    W_N = np.einsum('mp,pq->mpq',np.sqrt(N_all_SNP),np.eye(P))
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
        W_inv_Z = np.einsum('mqp,mp->mq',W_N_inv,Z_score)

        beta_denom = np.einsum('mp,p->m',np.einsum('q,mqp->mp',yy,inv_xx),yy)
        mtag_betas[:,p] = np.einsum('mp,mp->m',np.einsum('q,mqp->mp',yy,inv_xx), W_inv_Z) / beta_denom

        inv_xx_S_inv_xx = np.einsum('mpq,mqr->mpr',np.einsum('mpq,mqr->mpr',inv_xx,Sigma_N), inv_xx)
        var_denom = np.square(np.einsum('mq,q->m',np.einsum('p,mpq->mq',yy,inv_xx),yy))
        mtag_var_p = np.einsum('mq,q->m',np.einsum('p,mpq ->mq',yy,inv_xx_S_inv_xx),yy) / var_denom

        mtag_se[:,p] = np.sqrt(mle_var_p)



    logging.info(' ... Completed MTAG calculations.')
    return mtag_betas, mtag_se


#################
## SAVING RESULTS ##
#########################

def save_mtag_results(args,DATA,Zs,Ns, Fs,mtag_betas,mtag_se,omega_hat,sigma_hat):
    '''
    Output will be of the form:

    snp_name z n maf mtag_beta mtag_se mtag_zscore mtag_pval

   '''
    p_values = lambda z: 2*(1.0-scipy.stats.norm.cdf(np.abs(z)))
    snps = DATA[args.snp_name]
    M,P  = mtag_betas.shape

    for p in range(P):
        logging.info('Writing Phenotype {} to file ...'.format(p))

        out_df = pd.DataFrame(index=range(M))
        out_df[args.snp_name] = snps
        out_df[args.z_name] = Zs[:,p]
        out_df[args.n_name] = Ns[:,p]
        out_df[args.maf_name] = Fs[:,p]
        ### XXXX standardized or unstandardized beta
        out_df['mtag_beta'] = mtag_betas[:,p]
        out_df['mtag_se'] = mtag_se[:,p]
        out_df['mtag_z'] = mtag_betas[:,p]/mtag_se[:,p]
        out_df['mtag_pval'] = p_values(out_df['mtag_z'])

        if P == 1:
            out_path = args.outdir + args.out +'_phenotype.csv'
        else:
            out_path = args.outdir + args.out +'_phenotype_' + str(p+1) + '.csv'
        out_df.to_csv(out_path,sep='\t', index=False)



    omega_out = "Estimated Omega:\n"
    omega_out += str(omega_hat)
    np.savetxt(args.outdir + args.out +'_omega_hat.csv',omega_hat, delimeter ='\t')


    sigma_out = "Estimated Sigma:\n"
    sigma_out += str(sigma_hat)
    np.savetxt(args.outdir + args.out +'_sigma_hat.csv',sigma_hat, delimeter ='\t')

        ### XXXX STOPPED HERE
    summary_df = pd.DataFrame(index=range(P))
    input_phenotypes = [ '.../'+f[:16] if len(f) > 20 else f for f in args.GWAS_results.split(',')]

    for p in range(P):
        summary_df.loc[p,'Phenotype'] = input_phenotypes[p]
        summary_df.loc[p, 'n (max)'] = np.max(Ns[:,p])
        summary_df.loc[p, '# SNPs used'] = len(Zs)
        summary_df.loc[p, 'GWAS mean chi^2'] = np.mean(np.square(Zs))
        Z_mtag = mtag_betas[:,p]/mtag_se[:,p]
        summary_df.loc[p, 'MTAG mean chi^2'] = np.mean(np.square(Z_mtag))
        summary_df.loc[p, 'GWAS equivalent '] = summary_df.loc[p, 'n (max)']*(summary_df.loc[p, 'MTAG mean chi^2'] - 1) / (summary_df.loc[p, 'GWAS mean chi^2'] - 1)



    final_summary = "Summary of MTAG results:\n"
    final_summary +="------------------------\n"
    final_summary += str(summary_df)
    final_summary += omega_out
    final_summary += sigma_out

    loggin.info(final_summary)



def mtag(args):


    #1. Administrative checks

    args.outdir = args.outdir if args.outdir[-1] in ['/','\\'] else args.outdir + '/'
    args.n_name = 'n' if args.n_name is None else args.n_name

    if args.ld_ref_panel is None:
        args.ld_ref_panel = '../ld_ref_panel/eur_w_ld_chr/'

    ## XXX Check all paths exist / well-formed

    if not os.path.isdir(args.outdir):
        if args.make_path:
            warnings.warn("Output folder provided is not found, creating the directory")
            safely_create_folder(args.outdir)
        else:
            raise ValueError('Output directory: {} could not be found and the -make_path option was not specified.'.format(args.outdir))

     ## Instantiate log file and masthead
    logging.basicConfig(format='%(asctime)s %(message)s', filename=args.outdir + args.out + '.log', filemode='w', level=logging.INFO,datefmt='%Y/%m/%d %I:%M:%S %p')

    header_sub = header
    header_sub += "Calling ./mtag.py \\\n"
    options = ['--{KEY} {VAL} \\'.format(KEY=x.replace('_','-'),VAL=var(args)[x]) for x in var(args).keys()]
    header_sub += '\n'.join(options).replace('True','').replace('False','')
    header_sub = header_sub[0:-1] + '\n'

    start_time = time.time()  # starting time of analysis

    logging.info(header_sub)
    logging.info("Beginning MTAG analysis...")

     #2. Load Data and perform restrictions
    DATA, args = load_and_merge_data(args)

    #4. Estimate Sigma
    args.sigma_hat = estimate_sigma(DATA, args)
    #5. Estimate Omega
    Zs , Ns ,Fs = extract_gwas_sumstats(DATA,args)
    omega_hat = estimate_omega(args, Zs, Ns, sigma)
    #6. Perform MTAG
    mtag_betas, mtag_se = mtag_analysis(args, Zs,Ns,omega_hat, sigma)
    #7. Output GWAS_results
    save_mtag_results(args,DATA,Zs,Ns, Fs,mtag_betas,mtag_se)

    logging.info('MTAG complete. Time elapsed: {}'.format(sec_to_str(time.time()-start_time)))

parser = argparse.ArgumentParser(description="MTAG: Perform Multitrait Analysis over a provided set of GWAS summary statitistics, following the method described by Turley et. al. (forthcoming). Requires the user to specify the paths for the GWAS input files with columns formatted according to the documentation and output folder path.")

parser.add_argument("GWAS_results", metavar="{File1},{File2}...", type=str, nargs='?', help='Specify the list of files to perform multitrait analysis. Multiple files paths must be seperated by \",\". Please read the documentation  to find the up-to-date set of acceptable file formats. A general guideline is that any files you pass into MTAG should also be parsable by ldsc and you should take the additional step of specifying the names of the main columns below to avoid reading errors.')
# Required arguments

parser.add_argument("--outdir", metavar="FOLDER_PATH",default=".", type=str, help= "Specify the directory to output MTAG results. All output files created in this folder will be prefixed by the name passed to --out. The default is the current directory.")
parser.add_argument("--out", metavar="NAME", default="mtag", type=str, nargs='?', help='Specify the name prefix that all will share. Default name is \'mtag_results\'')

# "OPTIONAL' Arguments

parser.add_argument('-munge_on', action="store_true", default=False, help="The residual covariance matrix Sigma will be estimated from LD score (ldsc) using the summary statistics that have been processed with ldsc's munge function. For consistency, it is recommended to perform all QC on the GWAS summary statistics before running MTAG.")

parser.add_argument("--tol", default=1e-7,type=float, help="Set the absolute tolerance when numerically estimating the genetic variance-covariance matrix. Not recommended to change unless you are facing strong runtime constraints for a large number of phenotypes.")

parser.add_argument("--snp_name", default="snpid", action="store",type="str", help="Name of the single column that provides the unique identifier for SNPs in the GWAS summary statistics across all GWAS results. Default is \"snpid\". This the index that will be used to merge the GWAS summary statistics.")

parser.add_argument("-z_name", default=None, help="The common name of the column of Z scores across all input files. Default is to search for the first column beginning with the lowercase letter z.")
parser.add_argument("-n_name", default=None, help="the common name of the column of sample sizes in the GWAS results file. Default is to search for the first column beginning with the lowercase letter  n.")
parser.add_argument('-maf_name',default=None, help="The common of the column of minor allele frequencies (MAF) in the GWAS input files. The default is to search for columns beginning with either \"maf\" or \"freq\".")
parser.add_argument("-make_path", default=False, action="store_true", help="option to make output path specified in -out if it does not exist.")
parser.add_argument("--info_min", default=None,type=float, help="Minimim info score for filtering SNPs for MTAG")

parser.add_argument("-include",default=None, metavar="SNPLIST1,SNPLIST2,..", type=str, help="Option to give file paths of a list of snp indices that will restrict MTAG. The first line must match the SNP index that will be used to merge the GWAS input files. MTAG will run on the intersection of SNPs in all of the GWAS input files and in the specified list of SNPs. Multiple SNP lists should be seperated by commas without whitespace.")
parser.add_argument("-exclude", "--excludeSNPs",default=None, metavar="FILEPATH1,FILEPATH2", type=str, help="Similar to the -include option, except that the SNPs found in the specified files will be excluded from MTAG. In other words, MTAG will run for the intersection of SNPs in the GWAS input files and included SNPlists minus the union of SNPs in the -exclude file paths. Both -exclude and -include may be simultaneously specified, but -exclude will take precedent (i.e., SNPs found in both the -include and -exclude SNP lists will be excluded). Multiple SNO lists should be separated by commas without whitespace.")

parser.add_argument("-gencov_path",metavar="FILE_PATH", default=None, action="store", help="If specified, will read in the genetic covariance matrix saved in the file path below and skip the estimation routine. The rows and columns of the matrix must correspond to the order of the GWAS input files specified. FIles can either be in .csv or .npy format. Use with caution as the genetic covariance matrix specified will be weakly nonoptimal.")

parser.add_argument("-homogNs_frac", default=None, type=float, action="store", metavar="FRAC", help="Restricts to SNPs within FRAC of the mode of sample sizes for the SNPs as given by (N-Mode)/Mode < FRAC. This filter is not applied by default.")

parser.add_argument("-homogNs_dist", default=None, type=float, action="store", metavar="FRAC", help="Restricts to SNPs within DIST (in sample size) of the mode of sample sizes for the SNPs. This filter is not applied by default.")

parser.add_argument('-only_chr', metavar="CHR_A,CHR_B,..", default=None, type=str, action="store", help="Restrict MTAG to SNPs on one of the listed chromosome above (and also pass the -include and -exclude filters) Not generally recommended. Multiple chromosome numbers should be seperated by commas without whitespace.")

parser.add_argument('-timeLimit', '--timeLimit', default=None,type=float, action="store", help="Set time limit (hours) on the numerical estimation of the variance covariance matrix for MTAG, after which the optimization routine will spot and perform MTAG using the last genetic VCV matrix it reached.")

parser.add_argument('-ld_ref_path', default=None, action='store', help="Specifies the folder containing the reference panel of LD scores to use in estimating the error variance covariance matrix. Passes the folder path to the -ref-ld-chr and -w-ld-chr when running ldsc. The default is the white European file included with the ldsc in the MTAG package.")

parser.add_argument('-std_betas', default=False, action='store_true', help="Results files will have standardized effect sizes, i.e. not weighted by 1/sqrt(2*MAF*(1-MAF)), where MAF is the minor allele frequency.")
parser.add_argument('-maf_min', default=0.01, type=float, action='store', help="set the threshold below SNPs with low minor allele frequencies will be dropped. Default is 0.01. Set to 0.5 to skip MAF filtering.")
parser.add_argument('-n_min', default=None, type=float, action='store', help="set the minimum threshold for SNP sample size in input data. Default is 0.75*(90th percentile), similar to the threshold used in LD score regression. Any SNP that does not pass this threshold for any of the GWAS input statistics will not be included in MTAG.")
parser.add_argument('-n_max', default=None, type=float, action='store', help="set the maximum threshold for SNP sample size in input data. Not used by default. Any SNP that does not pass this threshold for any of the GWAS input statistics will not be included in MTAG.")
parser.add_argument('-numerical_Off', default=False, action='store_true', help='Option to turn off the numerical estimation of the genetic VCV matrix in the presence of constant sample size within each GWAS, for which a closed-form solution exists. The default is to typically use the closed form solution as an approximation for the genetic VCV, which is then \"polished\" off by numerical methods. Use with caution! If any input GWAS does not have constant sample size, then the analytic solution employed here will not be a maximizer of the likelihood function.')
parser.add_argument('-no_overlap', default=False, action='store_true', help='Imposes the assumption that there is no sample overlap between the input GWAS summary staistics. MTAG is performed with the off-diagonal terms on the residual covariance matrix set to 0.')
parser.add_argument('-perfect_gencov', default=False, action='store_true', help='Imposes the assumption that all phenotypes used are perfectly genetically correlated with each other. The off-diagonal terms of the genetic covariance matrix are set to the square root of the product of the heritabilities')
parser.add_argument('-equal_h2', default=False, action='store_true', help='Imposes the assumption that all phenotypes passed to MTAG have equal heritability. The diagonal terms of the genetic covariance matrix are set equal to each other. Can only be used in conejunction with -perfect_gencov')

parser.add_argument('--ld_ref_panel', default=None,action='store',type='str', help='Specify folder of the ld reference panel (split by chromosome) that will be used in the estimation of the error VCV (sigma). The default is to the reference panel of LD scores computed from 1000 Genomes European subjects (eur_w_ld_chr) that is included with the distribution of mtag')

if __name__ == '__main__':

    mtag(parser.parse_args())

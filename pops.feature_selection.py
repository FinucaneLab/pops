#!/usr/bin/env python

import pandas as pd
import numpy as np
import argparse
import statsmodels.api as sm
import scipy


def munge_sigma(args):
	f = open(args.gene_results+'.genes.raw')
	lines = list(f)[2:]
	lines = [np.asarray(line.strip('\n').split(' ')) for line in lines]
	sigmas = []
	for chrom in range(1,23):
		chr_start = min(np.where([int(line[1])==chrom for line in lines])[0])
		chr_end = max(np.where([int(line[1])==chrom for line in lines])[0])
		lines_chr = lines[chr_start:chr_end+1]
		n_genes = len(lines_chr)
		sigma_chr = np.zeros([n_genes, n_genes])
		for i in range(n_genes):
			line = lines_chr[i]
			if line.shape[0] > 9:
				gene_corrs = np.asarray([float(c) for c in line[9:]])
				sigma_chr[i, i-gene_corrs.shape[0]:i] = gene_corrs
		sigma_chr = sigma_chr+sigma_chr.T+np.identity(n_genes)
		sigmas.append(sigma_chr)
	return sigmas

def compute_Ls(sigmas, args):
	Ls = []
	min_lambda = 0
	for sigma in sigmas:
		W = np.linalg.eigvalsh(sigma)
		min_lambda = min(min_lambda, min(W))
	Y = pd.read_table(args.gene_results+'.genes.out', delim_whitespace=True).ZSTAT.values
	ridge = abs(min(min_lambda, 0))+.05+.9*max(0, np.var(Y)-1)
	for sigma in sigmas:
		sigma = sigma+ridge*np.identity(sigma.shape[0])
		L = np.linalg.cholesky(np.linalg.inv(sigma))
		Ls.append(L)
	full_L = scipy.linalg.block_diag(Ls[0], Ls[1], Ls[2], Ls[3], Ls[4], Ls[5], Ls[6], Ls[7], Ls[8], Ls[9], Ls[10], Ls[11], Ls[12], Ls[13], Ls[14], Ls[15], Ls[16], Ls[17], Ls[18], Ls[19], Ls[20], Ls[21])
	return full_L

def get_transformation_matrix(args):
	sigmas = munge_sigma(args)
	L = compute_Ls(sigmas, args)
	return L

def marginal_ols(X, Y):
	model = sm.OLS(Y, X).fit()
	coef = model.params[0]
	std_er = model.bse[0]
	pval = model.pvalues[0]
	return coef, std_er, pval

def main(args):
	f_df = pd.read_table(args.features)
	gene_scores = pd.read_table(args.gene_results+'.genes.out', delim_whitespace=True)[['GENE', 'ZSTAT']].rename(columns={'GENE': 'ENSGID'})
	f_df = gene_scores[['ENSGID']].merge(f_df, on='ENSGID', how='left')
	gene_scores = gene_scores.ZSTAT.values - np.mean(gene_scores.ZSTAT.values)
	nf = f_df.shape[1]-1
	coefs = np.zeros(nf)
	std_ers = np.zeros(nf)
	pvals = np.zeros(nf)
	L = get_transformation_matrix(args)
	LY = np.matmul(L, gene_scores)
	for i in range(nf):
		LX = np.matmul(L, f_df.iloc[:,1+i].values.astype(float))
		coef, std_er, pval = marginal_ols(LX, LY)
		coefs[i] = coef
		std_ers[i] = std_er
		pvals[i] = pval
	sig_results = pd.DataFrame(data={'Feature': f_df.columns.values[1:], 'BETA': coefs, 'SE': std_ers, 'P': pvals})[['Feature', 'BETA', 'SE', 'P']]
	selected = sig_results.loc[sig_results.P<=.05, 'Feature'].values
	np.savetxt(args.out+'.features',  selected, fmt='%s')


parser = argparse.ArgumentParser()
parser.add_argument('--features', help='Path to gene features', type=str)
parser.add_argument('--gene_results', help='Prefix to gene analysis output from MAGMA', type=str)
parser.add_argument('--out', help='Prefix to output for selected features', type=str)

if __name__ == '__main__':
	args = parser.parse_args()
	main(args)

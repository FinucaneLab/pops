#!/usr/bin/env python

import pandas as pd
import numpy as np
import scipy as sc
import argparse
from sklearn.linear_model import LinearRegression,RidgeCV
from sklearn.metrics import make_scorer
import copy


def munge_sigma(args):
	f = open(args.gene_results+'.genes.raw')
	lines = list(f)[2:]
	lines = [np.asarray(line.strip('\n').split(' ')) for line in lines]
	sigmas = []
	gene_metadata = []
	gene_lists = []
	for chrom in range(1,23):
		chr_start = min(np.where([int(line[1])==chrom for line in lines])[0])
		chr_end = max(np.where([int(line[1])==chrom for line in lines])[0])
		lines_chr = lines[chr_start:chr_end+1]
		n_genes = len(lines_chr)
		sigma_chr = np.zeros([n_genes, n_genes])
		gene_NSNPs = np.zeros(n_genes)
		gene_NPARAM = np.zeros(n_genes)
		gene_MAC = np.zeros(n_genes)
		for i in range(n_genes):
			line = lines_chr[i]
			gene_NSNPs[i] = line[4]
			gene_NPARAM[i] = line[5]
			gene_MAC[i] = line[7]
			if line.shape[0] > 9:
				gene_corrs = np.asarray([float(c) for c in line[9:]])
				sigma_chr[i, i-gene_corrs.shape[0]:i] = gene_corrs
		sigma_chr = sigma_chr+sigma_chr.T+np.identity(n_genes)
		sigmas.append(sigma_chr)
		gene_metadata_chr = pd.DataFrame(data={'NSNPS': gene_NSNPs, 'NPARAM': gene_NPARAM, 'MAC': gene_MAC})
		gene_metadata.append(gene_metadata_chr)
		gene_list_chr = [line[0] for line in lines_chr]
		gene_lists.append(gene_list_chr)
	return sigmas, gene_metadata, gene_lists

def compute_Ls(sigmas, args):
	Ls = []
	min_lambda=0
	for sigma in sigmas:
		W = np.linalg.eigvalsh(sigma)
		min_lambda = min(min_lambda, min(W))
	Y = pd.read_table(args.gene_results+'.genes.out', delim_whitespace=True).ZSTAT.values
	ridge = abs(min_lambda)+.05+.9*max(0, np.var(Y)-1)
	for sigma in sigmas:
		sigma = sigma+ridge*np.identity(sigma.shape[0])
		L = np.linalg.cholesky(np.linalg.inv(sigma))
		Ls.append(L)
	return Ls

def munge_features(args):
	feature_df = pd.read_table(args.gene_loc)[['ENSGID', 'CHR']]
	selected_features = np.loadtxt(args.selected_features, dtype=str).tolist()
	control_features = np.loadtxt(args.control_features, dtype=str).tolist()
	if type(control_features)==str:
		f_cols = ['ENSGID']+[control_features]+selected_features
	else:
		f_cols = ['ENSGID']+control_features+selected_features
	print(f_cols)
	features = pd.read_table(args.features, delim_whitespace=True, usecols=f_cols)
	print(features.columns)
	feature_df = feature_df.merge(features, on='ENSGID', how='inner')
	return feature_df

def munge_gene_results(args):
	gene_scores = pd.read_table(args.gene_results+'.genes.out', delim_whitespace=True)[['GENE', 'ZSTAT']]
	gene_scores.rename(columns={'GENE': 'ENSGID'}, inplace=True)
	return gene_scores

def build_control_covariates(metadata):
	genesize = metadata.NPARAM.values.astype(float)
	genedensity = metadata.NPARAM.values/metadata.NSNPS.values
	inverse_mac = 1.0/metadata.MAC.values
	cov = np.stack((genesize, np.log(genesize), genedensity, np.log(genedensity), inverse_mac, np.log(inverse_mac)), axis=1)
	return cov

def transform_regression(feature_df, args):
	sigmas, metadata, gene_lists = munge_sigma(args)
	Ls = compute_Ls(sigmas, args)
	gene_scores = munge_gene_results(args)
	LXs = []
	LCs = []
	LYs = []
	for chrom in range(22):
		L = Ls[chrom]
		genes = pd.DataFrame(data={'ENSGID': gene_lists[chrom]})
		X = genes.merge(feature_df, on='ENSGID', how='inner').values[:,2:].astype(float)
		C = build_control_covariates(metadata[chrom])
		Y = genes.merge(gene_scores, on='ENSGID', how='inner').ZSTAT.values
		LX = np.matmul(L, X)
		LC = np.matmul(L, C)
		LY = np.matmul(L, Y)
		LXs.append(LX)
		LCs.append(LC)
		LYs.append(LY)
	return LXs, LCs, LYs

def build_training(matrix_list, chrom):
	full_matrix = np.concatenate(matrix_list[0:chrom-1]+matrix_list[chrom:22])
	full_matrix = np.real(full_matrix)
	return full_matrix

def corr_score(Y, Y_pred):
	score = sc.stats.pearsonr(Y, Y_pred)[0]
	return score

def initialize_regressor():
	scorer = make_scorer(corr_score)
	alphas = np.logspace(-2, 10, num=25)
	regr = RidgeCV(alphas=alphas, scoring=scorer, fit_intercept=False)
	return regr

def project_out_cov(Y, C):
	lm = LinearRegression(fit_intercept=False)
	lm.fit(C, Y)
	Y_adj = Y - np.matmul(C, lm.coef_)
	return Y_adj

def train(LXs, LCs, LYs, args):
	X_train = build_training(LXs, args.chromosome)
	C_control = build_training(LCs, args.chromosome)
	Y_train = build_training(LYs, args.chromosome)
	Y_train = project_out_cov(Y_train, C_control)
	Y_train = Y_train-np.mean(Y_train)
	regr = initialize_regressor()
	regr.fit(X_train, Y_train)
	betahat = regr.coef_
	return betahat

def predict(betahat, feature_df, args):
	X_predict = feature_df[feature_df.CHR==(args.chromosome)].values[:,2:].astype(float)
	Y_predict = np.matmul(X_predict, betahat)
	return Y_predict

def munge_results(betahat, gene_scores, feature_df, args):
	features = feature_df.columns.values[2:]
	genes = feature_df.ENSGID[feature_df.CHR==(args.chromosome)]
	results_df = pd.DataFrame(data={'ENSGID': genes, 'Score': gene_scores})
	coefs_df = pd.DataFrame(data={'Feature': features, 'beta_hat': betahat})
	return results_df, coefs_df
		
def write_output(results_df, coefs_df, args):
	results_df.to_csv(args.out+'.'+str(args.chromosome)+'.results', index=False, sep='\t')
	coefs_df.to_csv(args.out+'.'+str(args.chromosome)+'.coefs', index=False, sep='\t')


def main(args):
	feature_df = munge_features(args)
	LXs, LCs, LYs = transform_regression(feature_df, args)
	betas = train(LXs, LCs, LYs, args)
	predictions = predict(betas, feature_df, args)
	results_df, coefs_df = munge_results(betas, predictions, feature_df, args)
	write_output(results_df, coefs_df, args)


parser = argparse.ArgumentParser()
parser.add_argument('--gene_loc', help='Path to gene loc file', type=str)
parser.add_argument('--gene_results', help='Prefix to gene analysis output from MAGMA', type=str)
parser.add_argument('--features', help='Path to gene features', type=str)
parser.add_argument('--selected_features', help='Path to names of selected features', type=str)
parser.add_argument('--control_features', help='Path to names of control features', type=str)
parser.add_argument('--chromosome', help='Chromosome number', type=int)
parser.add_argument('--out', help='Prefix to output for selected features', type=str)

if __name__ == '__main__':
	args = parser.parse_args()
	main(args)

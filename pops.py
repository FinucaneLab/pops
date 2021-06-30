import pandas as pd
import numpy as np
import re
import scipy.linalg
import random
import logging
import argparse

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import make_scorer

### --------------------------------- PROGRAM INPUTS --------------------------------- ###

def get_pops_args(argv=None):
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument("--gene_annot_path", help="...")
    parser.add_argument("--feature_mat_prefix", help="...")
    parser.add_argument("--num_feature_chunks", type=int, help="...")
    parser.add_argument("--magma_prefix", help="...")
    parser.add_argument('--use_magma_covariates', dest='use_magma_covariates', action='store_true')
    parser.add_argument('--ignore_magma_covariates', dest='use_magma_covariates', action='store_false')
    parser.set_defaults(use_magma_covariates=True)
    parser.add_argument('--use_magma_error_cov', dest='use_magma_error_cov', action='store_true')
    parser.add_argument('--ignore_magma_error_cov', dest='use_magma_error_cov', action='store_false')
    parser.set_defaults(use_magma_error_cov=True)
    parser.add_argument("--y_path", help="...")
    parser.add_argument("--y_covariates_path", help="...")
    parser.add_argument("--y_error_cov_path", help="...")
    parser.add_argument("--project_out_covariates_chromosomes", nargs="*", help="...")
    parser.add_argument('--project_out_covariates_remove_hla', dest='project_out_covariates_remove_hla', action='store_true')
    parser.add_argument('--project_out_covariates_keep_hla', dest='project_out_covariates_remove_hla', action='store_false')
    parser.set_defaults(project_out_covariates_remove_hla=True)
    parser.add_argument("--subset_features_path", help="...")
    parser.add_argument("--control_features_path", help="...")
    parser.add_argument("--feature_selection_chromosomes", nargs="*", help="...")
    parser.add_argument("--feature_selection_p_cutoff", type=float, default=0.05, help="...")
    parser.add_argument("--feature_selection_max_num", type=int, help="...")
    parser.add_argument("--feature_selection_fss_num_features", type=int, help="...")
    parser.add_argument('--feature_selection_remove_hla', dest='feature_selection_remove_hla', action='store_true')
    parser.add_argument('--feature_selection_keep_hla', dest='feature_selection_remove_hla', action='store_false')
    parser.set_defaults(feature_selection_remove_hla=True)
    parser.add_argument("--training_chromosomes", nargs="*", help="...")
    parser.add_argument('--training_remove_hla', dest='training_remove_hla', action='store_true')
    parser.add_argument('--training_keep_hla', dest='training_remove_hla', action='store_false')
    parser.set_defaults(training_remove_hla=True)
    parser.add_argument("--method", default="ridge", help="...")
    parser.add_argument("--out_prefix", help="...")
    parser.add_argument('--save_matrix_files', dest='save_matrix_files', action='store_true')
    parser.add_argument('--no_save_matrix_files', dest='save_matrix_files', action='store_false')
    parser.set_defaults(save_matrix_files=False)
    parser.add_argument("--random_seed", type=int, default=42, help="...")
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')
    parser.set_defaults(verbose=False)
    return parser.parse_args(argv)


### --------------------------------- GENERAL --------------------------------- ###

def natural_key(string_):
    """See https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def get_hla_genes(gene_annot_df):
    sub_gene_annot_df = gene_annot_df[gene_annot_df.CHR == "6"]
    sub_gene_annot_df = sub_gene_annot_df[sub_gene_annot_df.TSS >= 20 * (10 ** 6)]
    sub_gene_annot_df = sub_gene_annot_df[sub_gene_annot_df.TSS <= 40 * (10 ** 6)]
    return sub_gene_annot_df.index.values


### Returns as vector of booleans of length len(Y_ids)
def get_gene_indices_to_use(Y_ids, gene_annot_df, use_chrs, remove_hla):
    all_chr_genes_set = set(gene_annot_df[gene_annot_df.CHR.isin(use_chrs)].index.values)
    if remove_hla == True:
        hla_genes_set = set(get_hla_genes(gene_annot_df))
        use_genes = [True if (g in all_chr_genes_set) and (g not in hla_genes_set) else False for g in Y_ids]
    else:
        use_genes = [True if g in all_chr_genes_set else False for g in Y_ids]
    return np.array(use_genes)


def get_indices_in_target_order(ref_list, target_names):
    ref_to_ind_mapper = {}
    for i, e in enumerate(ref_list):
        ref_to_ind_mapper[e] = i
    return np.array([ref_to_ind_mapper[t] for t in target_names])

### --------------------------------- READING DATA --------------------------------- ###

def read_gene_annot_df(gene_annot_path):
    gene_annot_df = pd.read_csv(gene_annot_path, delim_whitespace=True).set_index("ENSGID")
    gene_annot_df["CHR"] = gene_annot_df["CHR"].astype(str)
    return gene_annot_df


def read_magma(magma_prefix, use_magma_covariates, use_magma_error_cov):
    ### Get Y and Y_ids
    magma_df = pd.read_csv(magma_prefix + ".genes.out", delim_whitespace=True)
    Y = magma_df.ZSTAT.values
    Y_ids = magma_df.GENE.values
    if use_magma_covariates is not None or use_magma_error_cov is not None:
        ### Get covariates and error_cov
        sigmas, gene_metadata = munge_magma_covariance_metadata(magma_prefix + ".genes.raw")
        cov_df = build_control_covariates(gene_metadata)
        ### Process
        assert (cov_df.index.values == Y_ids).all(), "Covariate ids and Y ids don't match."
        covariates = cov_df.values
        error_cov = scipy.linalg.block_diag(*sigmas)
    if use_magma_covariates == False:
        covariates = None
    if use_magma_error_cov == False:
        error_cov = None
    return Y, covariates, error_cov, Y_ids


def munge_magma_covariance_metadata(magma_raw_path):
    sigmas = []
    gene_metadata = []
    with open(magma_raw_path) as f:
        ### Get all lines
        lines = list(f)[2:]
        lines = [np.asarray(line.strip('\n').split(' ')) for line in lines]
        ### Check that chromosomes are sequentially ordered
        all_chroms = np.array([l[1] for l in lines])
        all_seq_breaks = np.where(all_chroms[:-1] != all_chroms[1:])[0]
        assert len(all_seq_breaks) == len(set(all_chroms)) - 1, "Chromosomes are not sequentially ordered."
        ### Get starting chromosome and set up temporary variables
        curr_chrom = lines[0][1]
        curr_ind = 0
        num_genes_in_chr = sum([1 for line in lines if line[1] == curr_chrom])
        curr_sigma = np.zeros((num_genes_in_chr, num_genes_in_chr))
        curr_gene_metadata = []
        for line in lines:
            ### If we move to a new chromosome, we reset everything
            if line[1] != curr_chrom:
                ### Symmetrize and save
                sigmas.append(curr_sigma + curr_sigma.T + np.eye(curr_sigma.shape[0]))
                gene_metadata.append(curr_gene_metadata)
                ### Reset
                curr_chrom = line[1]
                curr_ind = 0
                num_genes_in_chr = sum([1 for line in lines if line[1] == curr_chrom])
                curr_sigma = np.zeros((num_genes_in_chr, num_genes_in_chr))
                curr_gene_metadata = []
            ### Add metadata; GENE, NSNPS, NPARAM, MAC
            curr_gene_metadata.append([line[0], float(line[4]), float(line[5]), float(line[7])])
            if len(line) > 9:
                ### Add covariance
                gene_corrs = np.array([float(c) for c in line[9:]])
                curr_sigma[curr_ind, curr_ind - gene_corrs.shape[0]:curr_ind] = gene_corrs
            curr_ind += 1
        ### Save last piece
        sigmas.append(curr_sigma + curr_sigma.T + np.eye(curr_sigma.shape[0]))
        gene_metadata.append(curr_gene_metadata)
    gene_metadata = pd.DataFrame(np.vstack(gene_metadata), columns=["GENE", "NSNPS", "NPARAM", "MAC"])
    gene_metadata.NSNPS = gene_metadata.NSNPS.astype(np.float64)
    gene_metadata.NPARAM = gene_metadata.NPARAM.astype(np.float64)
    gene_metadata.MAC = gene_metadata.MAC.astype(np.float64)
    return sigmas, gene_metadata


def build_control_covariates(metadata):
    genesize = metadata.NPARAM.values
    genedensity = metadata.NPARAM.values/metadata.NSNPS.values
    inverse_mac = 1.0/metadata.MAC.values
    cov = np.stack((genesize, np.log(genesize), genedensity, np.log(genedensity), inverse_mac, np.log(inverse_mac)), axis=1)
    cov_df = pd.DataFrame(cov, columns=["gene_size", "log_gene_size", "gene_density", "log_gene_density", "inverse_mac", "log_inverse_mac"])
    cov_df["GENE"] = metadata.GENE.values
    cov_df = cov_df.loc[:,["GENE", "gene_size", "log_gene_size", "gene_density", "log_gene_density", "inverse_mac", "log_inverse_mac"]]
    cov_df = cov_df.set_index("GENE")
    return cov_df


### --------------------------------- PROCESSING DATA --------------------------------- ###

def block_Linv(A, block_labels):
    block_labels = np.array(block_labels)
    Linv = np.zeros(A.shape)
    for l in set(block_labels):
        subset_ind = (block_labels == l)
        sub_A = A[np.ix_(subset_ind, subset_ind)]
        Linv[np.ix_(subset_ind, subset_ind)] = np.linalg.inv(np.linalg.cholesky(sub_A))
    return Linv


def block_AB(A, block_labels, B):
    block_labels = np.array(block_labels)
    new_B = np.zeros(B.shape)
    for l in set(block_labels):
        subset_ind = (block_labels == l)
        new_B[subset_ind] = A[np.ix_(subset_ind, subset_ind)].dot(B[subset_ind])
    return new_B


def block_BA(A, block_labels, B):
    block_labels = np.array(block_labels)
    new_B = np.zeros(B.shape)
    for l in set(block_labels):
        subset_ind = (block_labels == l)
        new_B[:,subset_ind] = B[:,subset_ind].dot(A[np.ix_(subset_ind, subset_ind)])
    return new_B


def regularize_error_cov(error_cov, Y, Y_ids, gene_annot_df):
    Y_chr = gene_annot_df.loc[Y_ids].CHR.values
    min_lambda = 0
    for c in set(Y_chr):
        subset_ind = Y_chr == c
        W = np.linalg.eigvalsh(error_cov[np.ix_(subset_ind, subset_ind)])
        min_lambda = min(min_lambda, min(W))
    ridge = abs(min(min_lambda, 0))+.05+.9*max(0, np.var(Y)-1)
    return error_cov + np.eye(error_cov.shape[0]) * ridge


def project_out_covariates(Y, covariates, error_cov, Y_ids, gene_annot_df, project_out_covariates_Y_gene_inds):
    ### If covariates doesn't contain intercept, add intercept
    if not np.isclose(covariates.var(axis=0), 0).any():
        covariates = np.hstack((covariates, np.ones((covariates.shape[0], 1))))
    X_train, y_train = covariates[project_out_covariates_Y_gene_inds], Y[project_out_covariates_Y_gene_inds]
    if error_cov is not None:
        sub_error_cov = error_cov[np.ix_(project_out_covariates_Y_gene_inds, project_out_covariates_Y_gene_inds)]
        sub_error_cov_labels = gene_annot_df.loc[Y_ids[project_out_covariates_Y_gene_inds]].CHR.values
        Linv = block_Linv(sub_error_cov, sub_error_cov_labels)
        X_train, y_train = block_AB(Linv, sub_error_cov_labels, X_train), block_AB(Linv, sub_error_cov_labels, y_train)
    reg = LinearRegression(fit_intercept=False).fit(X_train, y_train)
    Y_proj = Y - reg.predict(covariates)
    return Y_proj
    
    
def project_out_V(M, V):
    gram_inv = np.linalg.inv(V.T.dot(V))
    moment = V.T.dot(M)
    betas = gram_inv.dot(moment)
    M_res = M - V.dot(betas)
    return M_res

### --------------------------------- FEATURE SELECTION --------------------------------- ###

def batch_marginal_ols(Y, X):
    ### Save current error settings and set divide to ignore
    old_settings = np.seterr(divide='ignore')
    ### Does not include intercept; we assume that's been projected out already
    sum_sq_X = np.sum(np.square(X), axis=0)
    ### If near-constant to 0 then set to nan. Make a safe copy so we don't get divide by 0 errors.
    near_const_0 = np.isclose(sum_sq_X, 0)
    sum_sq_X_safe = sum_sq_X.copy()
    sum_sq_X_safe[near_const_0] = 1
    betas = Y.dot(X) / sum_sq_X_safe
    mse = np.mean(np.square(Y.reshape(-1,1) - X * betas), axis=0)
    se = np.sqrt(mse / sum_sq_X_safe)
    z = betas / se
    chi2 = np.square(z)
    pvals = scipy.stats.chi2.sf(chi2, 1)
    r2 = 1 - (mse / np.var(Y))
    ### Set everything that's near-constant to 0 to be nan
    betas[near_const_0] = np.nan
    se[near_const_0] = np.nan
    pvals[near_const_0] = np.nan
    r2[near_const_0] = np.nan
    ### Reset error settings to old
    np.seterr(**old_settings)
    return betas, se, pvals, r2


### Accepts covariates, error_cov = None
def compute_marginal_assoc(feature_mat_prefix, num_feature_chunks, Y, Y_ids, covariates, error_cov, gene_annot_df, feature_selection_Y_gene_inds):
    ### Get Y data
    feature_selection_genes = Y_ids[feature_selection_Y_gene_inds]
    sub_Y = Y[feature_selection_Y_gene_inds]
    ### Add intercept if no near-constant feature
    if covariates is not None and not np.isclose(covariates.var(axis=0), 0).any():
        covariates = np.hstack((covariates, np.ones((covariates.shape[0], 1))))
    elif covariates is None:
        ### If no covariates then make intercept as only covariate
        covariates = np.ones((Y.shape[0], 1)) 
    sub_covariates = covariates[feature_selection_Y_gene_inds]
    if error_cov is not None:
        sub_error_cov = error_cov[np.ix_(feature_selection_Y_gene_inds, feature_selection_Y_gene_inds)]
        sub_error_cov_labels = gene_annot_df.loc[feature_selection_genes].CHR.values
        Linv = block_Linv(sub_error_cov, sub_error_cov_labels)
        sub_Y = block_AB(Linv, sub_error_cov_labels, sub_Y)
        sub_covariates = block_AB(Linv, sub_error_cov_labels, sub_covariates)
    ### Project covariates out of sub_Y
    sub_Y = project_out_V(sub_Y.reshape(-1,1), sub_covariates).flatten()
    ### Get X training indices
    rows = np.loadtxt(feature_mat_prefix + ".rows.txt", dtype=str).flatten()
    X_train_inds = get_indices_in_target_order(rows, feature_selection_genes)
    ### Loop through and get marginal association data
    marginal_assoc_data = []
    all_cols = []
    for i in range(num_feature_chunks):
        mat = np.load(feature_mat_prefix + ".mat.{}.npy".format(i))
        mat = mat[X_train_inds]
        cols = np.loadtxt(feature_mat_prefix + ".cols.{}.txt".format(i), dtype=str).flatten()
        ### Apply error covariance transformation if available
        if error_cov is not None:
            mat = block_AB(Linv, sub_error_cov_labels, mat)
        ### Project out covariates
        mat = project_out_V(mat, sub_covariates)
        ### Compute marginal associations
        marginal_assoc_data.append(np.vstack(batch_marginal_ols(sub_Y, mat)).T)
        all_cols.append(cols)
    marginal_assoc_data = np.vstack(marginal_assoc_data)
    all_cols = np.hstack(all_cols)
    marginal_assoc_df = pd.DataFrame(marginal_assoc_data, columns=["beta", "se", "pval", "r2"], index=all_cols)
    return marginal_assoc_df


### Note that subset_features overrides control_features.
### That is: we do not include control features that are not contained in subset features
### Also, control features do not count toward feature_selection_max_num
def select_features_from_marginal_assoc_df(marginal_assoc_df,
                                           subset_features_path,
                                           control_features_path,
                                           feature_selection_p_cutoff,
                                           feature_selection_max_num):
    ### Subset to subset_features
    if subset_features_path is not None:
        subset_features = np.loadtxt(subset_features_path, dtype=str).flatten()
        marginal_assoc_df = marginal_assoc_df.loc[subset_features]
    ### Get control_features contained in currently subsetted features, and set those aside
    if control_features_path is not None:
        control_features = np.loadtxt(control_features_path, dtype=str).flatten()
        control_df = marginal_assoc_df[marginal_assoc_df.index.isin(control_features)]
        marginal_assoc_df = marginal_assoc_df[~marginal_assoc_df.index.isin(control_features)]
    ### Subset to features that pass p-value cutoff
    if feature_selection_p_cutoff is not None:
        marginal_assoc_df = marginal_assoc_df[marginal_assoc_df.pval < feature_selection_p_cutoff]
    ### Enforce maximum number of features
    if feature_selection_max_num is not None:
        marginal_assoc_df = marginal_assoc_df.sort_values("pval").iloc[:feature_selection_max_num]
    ### Get selected features
    selected_features = list(marginal_assoc_df.index.values)
    ### Combine with control features
    if control_features_path is not None:
        selected_features = selected_features + list(control_df.index.values)
    return selected_features


def load_feature_matrix(feature_mat_prefix, num_feature_chunks, selected_features):
    if selected_features is not None:
        selected_features_set = set(selected_features)
    rows = np.loadtxt(feature_mat_prefix + ".rows.txt", dtype=str).flatten()
    all_mats = []
    all_cols = []
    for i in range(num_feature_chunks):
        mat = np.load(feature_mat_prefix + ".mat.{}.npy".format(i))
        cols = np.loadtxt(feature_mat_prefix + ".cols.{}.txt".format(i), dtype=str).flatten()
        if selected_features is not None:
            keep_inds = [True if c in selected_features_set else False for c in cols]
            mat = mat[:,keep_inds]
            cols = cols[keep_inds]
        all_mats.append(mat)
        all_cols.append(cols)
    mat = np.hstack(all_mats)
    cols = np.hstack(all_cols)
    return mat, cols, rows


def add_feature_to_covariates(covariates, covariates_ids, feature_mat_prefix, num_feature_chunks, feature_name):
    ### Get X indices
    rows = np.loadtxt(feature_mat_prefix + ".rows.txt", dtype=str).flatten()
    X_inds = get_indices_in_target_order(rows, covariates_ids)
    for i in range(num_feature_chunks):
        cols = np.loadtxt(feature_mat_prefix + ".cols.{}.txt".format(i), dtype=str).flatten()
        if feature_name in cols:
            mat = np.load(feature_mat_prefix + ".mat.{}.npy".format(i))[X_inds]
            f = mat[:,np.where(cols == feature_name)[0]]
            break
    covariates = np.hstack((covariates, f))
    return covariates


def forward_stepwise_selection(feature_mat_prefix, num_feature_chunks, Y, Y_ids, covariates, error_cov, gene_annot_df, feature_selection_Y_gene_inds, num_features_to_select):
    if covariates is None:
        covariates = np.ones((Y.shape[0], 1))
    selected_features = []
    for i in range(num_features_to_select):
        logging.info("FORWARD STEPWISE SELECTION: {} features selected".format(len(selected_features)))
        marginal_assoc_df = compute_marginal_assoc(feature_mat_prefix, num_feature_chunks, Y, Y_ids, covariates, error_cov, gene_annot_df, feature_selection_Y_gene_inds)
        top_feature = marginal_assoc_df[~marginal_assoc_df.index.isin(selected_features)].sort_values("pval").index.values[0]
        selected_features.append(top_feature)
        covariates = add_feature_to_covariates(covariates, Y_ids, feature_mat_prefix, num_feature_chunks, top_feature)
    return selected_features
    
    
### --------------------------------- MODEL FITTING --------------------------------- ###

def build_training(mat, cols, rows, Y, Y_ids, error_cov, gene_annot_df, training_Y_gene_inds, project_out_intercept=True):
    ### Get training Y
    training_genes = Y_ids[training_Y_gene_inds]
    sub_Y = Y[training_Y_gene_inds]
    intercept = np.ones((sub_Y.shape[0], 1)) ### Make intercept
    ### Get training X
    X_train_inds = get_indices_in_target_order(rows, training_genes)
    X = mat[X_train_inds]
    assert (rows[X_train_inds] == training_genes).all(), "Something went wrong. This shouldn't happen."
    ### Apply error covariance
    if error_cov is not None:
        sub_error_cov = error_cov[np.ix_(training_Y_gene_inds, training_Y_gene_inds)]
        sub_error_cov_labels = gene_annot_df.loc[training_genes].CHR.values
        Linv = block_Linv(sub_error_cov, sub_error_cov_labels)
        sub_Y = block_AB(Linv, sub_error_cov_labels, sub_Y)
        X = block_AB(Linv, sub_error_cov_labels, X)
        intercept = block_AB(Linv, sub_error_cov_labels, intercept)
    if project_out_intercept == True:
        ### Project out intercept
        sub_Y = project_out_V(sub_Y.reshape(-1,1), intercept).flatten()
        X = project_out_V(X, intercept)
    return X, sub_Y


# def corr_score(Y, Y_pred):
#     score = scipy.stats.pearsonr(Y, Y_pred)[0]
#     return score


def initialize_regressor(method, random_state):
    # scorer = make_scorer(corr_score)
    if method == "ridge":
        alphas = np.logspace(-2, 10, num=25)
        # reg = RidgeCV(fit_intercept=False, alphas=alphas, scoring=scorer)
        # logging.info("Model = RidgeCV with 25 alphas, generalized leave-one-out cross-validation, held-out Pearson correlation as scoring metric.")
        reg = RidgeCV(fit_intercept=False, alphas=alphas)
        logging.info("Model = RidgeCV with 25 alphas, generalized leave-one-out cross-validation, NMSE as scoring metric.")
    elif method == 'lasso':
        # alphas = np.logspace(-2, 10, num=25)
        # reg = LassoCV(fit_intercept=False, alphas=alphas, random_state=random_state)
        reg = LassoCV(fit_intercept=False, n_alphas=200, random_state=random_state)
        logging.info("Model = LassoCV with 200 alphas, 5-fold cross-validation, mean-squared error as scoring metric.")
    elif method == 'linreg':
        ### Note that this solves using pseudo-inverse if # features > # samples, corresponding to minimum norm OLS
        reg = LinearRegression(fit_intercept=False)
        logging.info("Model = LinearRegression. Note that this solves using the pseudo-inverse if # features > # samples, corresponding to minimum norm OLS.")
    return reg

    
def compute_coefficients(X_train, Y_train, cols, method, random_state):
    if method not in ["ridge", "lasso", "linreg"]:
        raise ValueError("Invalid argument for \"method\". Must be one of \"ridge\", \"lasso\", or \"linreg\".")
    reg = initialize_regressor(method, random_state)
    logging.info("Fitting model.")
    reg.fit(X_train, Y_train)
    if method == "ridge":
        coefs_df = pd.DataFrame([["METHOD", "RidgeCV"],
                                 ["SELECTED_CV_ALPHA", reg.alpha_],
                                 ["BEST_CV_SCORE", reg.best_score_]])
        coefs_df = pd.concat([coefs_df, pd.DataFrame([cols, reg.coef_]).T])
        coefs_df.columns = ["parameter", "beta"]
        coefs_df = coefs_df.set_index("parameter")
    elif method == "lasso":
        best_score = reg.mse_path_[np.where(reg.alphas_ == reg.alpha_)[0][0]].mean()
        coefs_df = pd.DataFrame([["METHOD", "LassoCV"],
                                 ["SELECTED_CV_ALPHA", reg.alpha_],
                                 ["BEST_CV_SCORE", best_score]])
        coefs_df = pd.concat([coefs_df, pd.DataFrame([cols, reg.coef_]).T])
        coefs_df.columns = ["parameter", "beta"]
        coefs_df = coefs_df.set_index("parameter")
    elif method == "linreg":
        coefs_df = pd.DataFrame([["METHOD", "LinearRegression"]])
        coefs_df = pd.concat([coefs_df, pd.DataFrame([cols, reg.coef_]).T])
        coefs_df.columns = ["parameter", "beta"]
        coefs_df = coefs_df.set_index("parameter")
    return coefs_df
    
    
def pops_predict(mat, rows, cols, coefs_df):
    pred = mat.dot(coefs_df.loc[cols].beta.values)
    preds_df = pd.DataFrame([rows, pred]).T
    preds_df.columns = ["ENSGID", "PoPS_Score"]
    return preds_df

### --------------------------------- MAIN --------------------------------- ###

def main(config_dict):
    ### --------------------------------- Basic settings --------------------------------- ###
    ### Set logging settings
    if config_dict["verbose"]:
        logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)
        logging.info("Verbose output enabled.")
    else:
        logging.basicConfig(format="%(levelname)s: %(message)s")
    ### Set random seeds
    np.random.seed(config_dict["random_seed"])
    random.seed(config_dict["random_seed"])

    ### Display configs
    logging.info("Config dict = {}".format(str(config_dict)))
    
    ### --------------------------------- Reading/processing data --------------------------------- ###
    gene_annot_df = read_gene_annot_df(config_dict["gene_annot_path"])
    ### If chromosome arguments are None, replace their values in config_dict with all chromosomes
    all_chromosomes = sorted(gene_annot_df.CHR.unique(), key=natural_key)
    if config_dict["project_out_covariates_chromosomes"] is None:
        config_dict["project_out_covariates_chromosomes"] = all_chromosomes
        logging.info("--project_out_covariates_chromosomes is None, defaulting to all chromosomes")
    if config_dict["feature_selection_chromosomes"] is None:
        config_dict["feature_selection_chromosomes"] = all_chromosomes
        logging.info("--feature_selection_chromosomes is None, defaulting to all chromosomes")
    if config_dict["training_chromosomes"] is None:
        config_dict["training_chromosomes"] = all_chromosomes
        logging.info("--training_chromosomes is None, defaulting to all chromosomes")
    ### Make sure all chromosome arguments are fully contained in gene_annot_df's chromosome list
    assert set(config_dict["project_out_covariates_chromosomes"]).issubset(all_chromosomes), "Invalid --project_out_covariates_chromosomes argument."
    assert set(config_dict["feature_selection_chromosomes"]).issubset(all_chromosomes), "Invalid --feature_selection_chromosomes argument."
    assert set(config_dict["training_chromosomes"]).issubset(all_chromosomes), "Invalid --training_chromosomes argument."
    ### Read in scores
    if config_dict["magma_prefix"] is not None:
        logging.info("MAGMA scores provided, loading MAGMA.")
        Y, covariates, error_cov, Y_ids = read_magma(config_dict["magma_prefix"],
                                                     config_dict["use_magma_covariates"],
                                                     config_dict["use_magma_error_cov"])
        if config_dict["use_magma_covariates"] == True:
            logging.info("Using MAGMA covariates.")
        else:
            logging.info("Ignoring MAGMA covariates.")
        if config_dict["use_magma_error_cov"] == True:
            logging.info("Using MAGMA error covariance.")
        else:
            logging.info("Ignoring MAGMA error covariance.")
    elif config_dict["y_path"] is not None:
        raise ValueError("Not implemented yet.")
    else:
        raise ValueError("At least one of --magma_prefix or --y_path must be provided.")
    ### Get projection, feature selection, and training genes
    project_out_covariates_Y_gene_inds = get_gene_indices_to_use(Y_ids,
                                                                 gene_annot_df,
                                                                 config_dict["project_out_covariates_chromosomes"],
                                                                 config_dict["project_out_covariates_remove_hla"])
    feature_selection_Y_gene_inds = get_gene_indices_to_use(Y_ids,
                                                            gene_annot_df,
                                                            config_dict["feature_selection_chromosomes"],
                                                            config_dict["feature_selection_remove_hla"])
    training_Y_gene_inds = get_gene_indices_to_use(Y_ids,
                                                   gene_annot_df,
                                                   config_dict["training_chromosomes"],
                                                   config_dict["training_remove_hla"])
    ### Regularize error covariance if using
    if error_cov is not None:
        logging.info("Regularizing error covariance.")
        error_cov = regularize_error_cov(error_cov, Y, Y_ids, gene_annot_df)
    ### Project out covariates if using
    if covariates is not None:
        logging.info("Projecting {} covariates out of target scores using genes on chromosome {}. HLA region {}."
                     .format(covariates.shape[1],
                             ", ".join(sorted(gene_annot_df.loc[Y_ids[project_out_covariates_Y_gene_inds]].CHR.unique(), key=natural_key)),
                             "removed" if config_dict["project_out_covariates_remove_hla"] else "included"))
        Y_proj = project_out_covariates(Y,
                                        covariates,
                                        error_cov,
                                        Y_ids,
                                        gene_annot_df,
                                        project_out_covariates_Y_gene_inds)
    else:
        Y_proj = Y
    
    
    ### --------------------------------- Feature selection --------------------------------- ###
    ### Compute marginal association data frame
    logging.info("Computing marginal association table using genes on chromosome {}. HLA region {}."
                 .format(", ".join(sorted(gene_annot_df.loc[Y_ids[feature_selection_Y_gene_inds]].CHR.unique(), key=natural_key)),
                         "removed" if config_dict["feature_selection_remove_hla"] else "included"))
    marginal_assoc_df = compute_marginal_assoc(config_dict["feature_mat_prefix"],
                                               config_dict["num_feature_chunks"],
                                               Y_proj,
                                               Y_ids,
                                               None,
                                               error_cov,
                                               gene_annot_df,
                                               feature_selection_Y_gene_inds)
    ### Either do FSS or filter marginal_assoc_df
    if config_dict["feature_selection_fss_num_features"] is not None:
        logging.info("--feature_selection_fss_num_features set to {}, so performing forward stepwise selection (overriding all other feature selection settings).".format(config_dict["feature_selection_fss_num_features"]))
        selected_features = forward_stepwise_selection(config_dict["feature_mat_prefix"],
                                                       config_dict["num_feature_chunks"],
                                                       Y_proj,
                                                       Y_ids,
                                                       None,
                                                       error_cov,
                                                       gene_annot_df,
                                                       feature_selection_Y_gene_inds,
                                                       config_dict["feature_selection_fss_num_features"])
        marginal_assoc_df["selected"] = marginal_assoc_df.index.isin(selected_features)
        ### Annotate with selection rank
        marginal_assoc_df["selection_rank"] = np.nan
        for i in range(len(selected_features)):
            marginal_assoc_df.loc[selected_features[i], "selection_rank"] = i + 1
        logging.info("Forward stepwise selection complete, {} features in model.".format(len(selected_features)))
    else:
        ### Filter features based on settings
        selected_features = select_features_from_marginal_assoc_df(marginal_assoc_df,
                                                                   config_dict["subset_features_path"],
                                                                   config_dict["control_features_path"],
                                                                   config_dict["feature_selection_p_cutoff"],
                                                                   config_dict["feature_selection_max_num"])
        ### Annotate marginal_assoc_df with selected True/False
        marginal_assoc_df["selected"] = marginal_assoc_df.index.isin(selected_features)
        ### Explicitly set features with nan p-values to not-selected
        marginal_assoc_df["selected"] = marginal_assoc_df["selected"] & ~pd.isnull(marginal_assoc_df.pval)
        ### Redefine selected_features
        selected_features = marginal_assoc_df[marginal_assoc_df.selected].index.values
        ### Complex logging statement
        select_feat_logtxt_pieces = []
        if config_dict["subset_features_path"] is not None:
            select_feat_logtxt_pieces.append("subsetting to features at {}".format(config_dict["subset_features_path"]))
        if config_dict["feature_selection_p_cutoff"] is not None:
            if config_dict["feature_selection_max_num"] is not None:
                select_feat_logtxt_pieces.append("filtering to top {} features with p-value < {}"
                                                 .format(config_dict["feature_selection_max_num"],
                                                         config_dict["feature_selection_p_cutoff"]))
            else:
                select_feat_logtxt_pieces.append("filtering to features with p-value < {}"
                                                 .format(config_dict["feature_selection_p_cutoff"]))
        elif config_dict["feature_selection_max_num"] is not None:
            select_feat_logtxt_pieces.append("filtering to top {} features by p-value"
                                             .format(config_dict["feature_selection_max_num"]))
        if config_dict["control_features_path"] is not None:
            select_feat_logtxt_pieces.append("unioning with non-constant control features")
        ### Combine complex logging statement
        if len(select_feat_logtxt_pieces) == 0:
            select_feat_logtxt = ("{} features reamin in model.".format(len(selected_features)))
        if len(select_feat_logtxt_pieces) == 1:
            select_feat_logtxt = ("After {}, {} features remain in model."
                                  .format(select_feat_logtxt_pieces[0], len(selected_features)))
        elif len(select_feat_logtxt_pieces) == 2:
            select_feat_logtxt = ("After {} and {}, {} features remain in model."
                                  .format(select_feat_logtxt_pieces[0], select_feat_logtxt_pieces[1], len(selected_features)))
        elif len(select_feat_logtxt_pieces) == 3:
            select_feat_logtxt = ("After {}, {}, and {}, {} features remain in model."
                                  .format(select_feat_logtxt_pieces[0], select_feat_logtxt_pieces[1], select_feat_logtxt_pieces[2], len(selected_features)))
        logging.info(select_feat_logtxt)

    
    ### --------------------------------- Training --------------------------------- ###
    ### Load data
    ### Won't necessarily load in order of selected_features. Loads in order of matrix columns.
    ### Note: doesn't raise error if trying to select feature that isn't in columns
    mat, cols, rows = load_feature_matrix(config_dict["feature_mat_prefix"], config_dict["num_feature_chunks"], selected_features)
    logging.info("Building training X and Y using genes on chromosome {}. HLA region {}."
                 .format(", ".join(sorted(gene_annot_df.loc[Y_ids[training_Y_gene_inds]].CHR.unique(), key=natural_key)),
                         "removed" if config_dict["training_remove_hla"] else "included"))
    ### Build training X and Y
    ### Should be properly subsetted and have error_cov applied. We also explicitly project out intercept
    X_train, Y_train = build_training(mat, cols, rows,
                                      Y_proj, Y_ids, error_cov,
                                      gene_annot_df, training_Y_gene_inds,
                                      project_out_intercept=True)
    logging.info("X dimensions = {}. Y dimensions = {}".format(X_train.shape, Y_train.shape))
    ### Compute coefficients
    ### Output should contain at least one row for every column and additional rows for any metadata like method, regularization chosen by CV, etc.
    coefs_df = compute_coefficients(X_train, Y_train, cols, config_dict["method"], config_dict["random_seed"])
    ### Prediction
    logging.info("Computing PoPS scores.")
    preds_df = pops_predict(mat, rows, cols, coefs_df)
    ### Annotate Y, Y_proj, and gene used in feature selection + gene used in training
    preds_df = preds_df.merge(pd.DataFrame(np.array([Y_ids, Y]).T, columns=["ENSGID", "Y"]),
                              how="left",
                              on="ENSGID")
    if covariates is not None:
        preds_df = preds_df.merge(pd.DataFrame(np.array([Y_ids, Y_proj]).T, columns=["ENSGID", "Y_proj"]),
                                  how="left",
                                  on="ENSGID")
        preds_df["project_out_covariates_gene"] = preds_df.ENSGID.isin(Y_ids[project_out_covariates_Y_gene_inds])
    preds_df["feature_selection_gene"] = preds_df.ENSGID.isin(Y_ids[feature_selection_Y_gene_inds])
    preds_df["training_gene"] = preds_df.ENSGID.isin(Y_ids[training_Y_gene_inds])

    
    ### --------------------------------- Save --------------------------------- ###
    logging.info("Writing output files.")
    preds_df.to_csv(config_dict["out_prefix"] + ".preds", sep="\t", index=False)
    coefs_df.to_csv(config_dict["out_prefix"] + ".coefs", sep="\t")
    marginal_assoc_df.to_csv(config_dict["out_prefix"] + ".marginals", sep="\t")
    if config_dict["save_matrix_files"] == True:
        logging.info("Saving matrix files as well.")
        pd.DataFrame(np.hstack((Y_train.reshape(-1,1), X_train)),
                     index=Y_ids[training_Y_gene_inds],
                     columns=["Y_train"] + list(cols)).to_csv(config_dict["out_prefix"] + ".traindata", sep="\t")
        pd.DataFrame(mat,
                     index=rows,
                     columns=cols).to_csv(config_dict["out_prefix"] + ".matdata", sep="\t")
    
    
### Main
if __name__ == '__main__':
    args = get_pops_args()
    config_dict = vars(args)
    main(config_dict)

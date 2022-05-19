import numpy as np
import pandas as pd
import scipy
import scipy.stats
import itertools
import statsmodels.api as sm
import random
import logging
import argparse



### --------------------------------- PROGRAM INPUTS --------------------------------- ###

def get_args(argv=None):
    parser = argparse.ArgumentParser(description='A suite of validation tools for PoPS + NMF + network output.')
    parser.add_argument("--nmf_out_prefix", help="...")
    parser.add_argument("--net_out_prefix", help="...")
    parser.add_argument("--net_out_prefix", help="...")
    parser.add_argument("--gene_annot_path", help="Path to tab-separated gene annotation file...")
    parser.add_argument("--trait_magma_out_prefix", help="...")
    parser.add_argument("--nmf_thresh", type=float, default=0.95, help="...")
    parser.add_argument("--net_thresh", type=float, default=0.95, help="...")
    parser.add_argument("--magma_validation_table", help="...")
    parser.add_argument("--epigenetic_validation_table", help="...")
    parser.add_argument("--pathways_table", help="...")
    parser.add_argument("--ppi_pairs", help="...")
    parser.add_argument("--polygenic_contrast_score_table", help="...")
    parser.add_argument("--polygenic_contrast_num_permutations", type=int, default=10000, help="...")
    parser.add_argument("--network_purity_gene_table", help="...")
    parser.add_argument("--network_purity_thresh_interval", type=float, default=0.1, help="...")
    parser.add_argument("--network_purity_num_permutations", type=int, default=10000, help="...")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility. 42 by default")
    parser.add_argument('--verbose', dest='verbose', action='store_true', help="Set this flag to get verbose output")
    parser.add_argument('--no_verbose', dest='verbose', action='store_false', help="(Default) Set this flag to silence output")
    parser.set_defaults(verbose=False)
    return parser.parse_args(argv)

### --------------------------------- MAGMA PROCESSING --------------------------------- ###

def get_hla_genes(gene_annot_df):
    sub_gene_annot_df = gene_annot_df[gene_annot_df.CHR == "6"]
    sub_gene_annot_df = sub_gene_annot_df[sub_gene_annot_df.TSS >= 20 * (10 ** 6)]
    sub_gene_annot_df = sub_gene_annot_df[sub_gene_annot_df.TSS <= 40 * (10 ** 6)]
    return sub_gene_annot_df.index.values


def get_indices_in_target_order(ref_list, target_names):
    ref_to_ind_mapper = {}
    for i, e in enumerate(ref_list):
        ref_to_ind_mapper[e] = i
    return np.array([ref_to_ind_mapper[t] for t in target_names])


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


def regularize_error_cov(error_cov, Y_ids, gene_annot_df, max_ratio=10):
    assert max_ratio > 1, "max_ratio must be greater than 1"
    Y_chr = gene_annot_df.loc[Y_ids].CHR.values
    min_lambda = np.inf
    max_lambda = -np.inf
    for c in set(Y_chr):
        subset_ind = Y_chr == c
        W = np.linalg.eigvalsh(error_cov[np.ix_(subset_ind, subset_ind)])
        min_lambda = min(min_lambda, min(W))
        max_lambda = max(max_lambda, max(W))
    if np.isinf(min_lambda) or np.isinf(max_lambda):
        raise ValueError("No eigenvalues computed; probably Y_ids is an empty list.")
    ridge = max(0, (max_lambda - max_ratio * min_lambda) / (max_ratio - 1))
    return error_cov + np.eye(error_cov.shape[0]) * ridge
    
    
def project_out_V(M, V):
    gram_inv = np.linalg.inv(V.T.dot(V))
    moment = V.T.dot(M)
    betas = gram_inv.dot(moment)
    M_res = M - V.dot(betas)
    return M_res


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

### --------------------------------- BINARY ENRICHMENT --------------------------------- ###

def batch_one_sided_fisher_exact_test(X_df, Y_df):
    ### Check that tables are binary
    assert ((X_df == 0) | (X_df == 1)).all().all(), "X_df must be binary"
    assert ((Y_df == 0) | (Y_df == 1)).all().all(), "Y_df must be binary"
    ### Convert to float for faster math operations
    X_df = X_df.astype(np.float64)
    Y_df = Y_df.astype(np.float64)
    ### Construct contingency table values
    cont11 = X_df.T.dot(Y_df)
    cont10 = X_df.T.dot(1 - Y_df)
    cont01 = (1 - X_df).T.dot(Y_df)
    cont00 = (1 - X_df).T.dot(1 - Y_df)
    ### Compute test values
    odds_ratios = (cont11 * cont00) / (cont10 * cont01)
    fet_pval = scipy.stats.hypergeom.sf(cont11 - 1,
                                        cont11 + cont10 + cont01 + cont00,
                                        cont11 + cont10,
                                        cont11 + cont01)
    fet_pval = pd.DataFrame(fet_pval, index=cont11.index, columns=cont11.columns)
    ### Combine into one dataframe
    multicol = pd.MultiIndex.from_tuples(itertools.product(["odds_ratio", "pvalue"], X_df.columns),
                                         names=["statistic", "trait"])
    combined_df = pd.DataFrame(columns=multicol, index=Y_df.columns)
    combined_df.odds_ratio = odds_ratios.T
    combined_df.pvalue = fet_pval.T
    return combined_df


def jaccard_similarity(X_df, Y_df):
    ### Check that tables are binary
    assert ((X_df == 0) | (X_df == 1)).all().all(), "X_df must be binary"
    assert ((Y_df == 0) | (Y_df == 1)).all().all(), "Y_df must be binary"
    ### Convert to float for faster math operations
    X_df = X_df.astype(np.float64)
    Y_df = Y_df.astype(np.float64)
    ### Compute Jaccard
    overlap = Y_df.values.T.dot(X_df.values)
    J = overlap / (Y_df.values.T.sum(axis=1, keepdims=True) + X_df.values.sum(axis=0) - overlap)
    multicol = pd.MultiIndex.from_tuples(itertools.product(["jaccard", "overlap"], X_df.columns))
    jaccard_df = pd.DataFrame(columns=multicol, index=Y_df.columns.values)
    jaccard_df.jaccard = J
    jaccard_df.overlap = overlap.astype(int)
    return jaccard_df


def format_score_from_jaccard(jaccard_df):
    score_df = pd.DataFrame(index=jaccard_df.jaccard.index, columns=["closest_comp", "jaccard", "overlap"])
    max_jaccard_ids = jaccard_df.jaccard.idxmax(axis=1)
    for i in max_jaccard_ids.index:
        score_df.loc[i] = [max_jaccard_ids[i],
                           jaccard_df.jaccard.loc[i,max_jaccard_ids[i]],
                           jaccard_df.overlap.loc[i,max_jaccard_ids[i]]]
    return score_df

### --------------------------------- CONTINUOUS ENRICHMENT --------------------------------- ###

def batch_marginal_ols(Ys, Xs):
    ### Save current error settings and set divide to ignore
    old_settings = np.seterr(divide='ignore')
    ### Does not include intercept; we assume that's been projected out already
    sum_sq_Xs = np.sum(np.square(Xs), axis=0)
    sum_sq_Ys = np.sum(np.square(Ys), axis=0)
    near_const_0_x = np.isclose(sum_sq_Xs, 0)
    near_const_0_y = np.isclose(sum_sq_Ys, 0)
    ### If X near-constant to 0 then set to nan. Make a safe copy so we don't get divide by 0 errors.
    sum_sq_Xs_safe = sum_sq_Xs.copy()
    sum_sq_Xs_safe[near_const_0_x] = 1
    moments = Xs.T.dot(Ys)
    betas = moments / sum_sq_Xs_safe.reshape(-1,1)
    mse = (sum_sq_Ys - 2 * betas * moments + betas * betas * sum_sq_Xs.reshape(-1,1)) / Xs.shape[0]
    se = np.sqrt(mse / sum_sq_Xs_safe.reshape(-1,1))
    ### Impute nans wherever sum_sq_Xs or sum_sq_Ys is close to 0
    se[near_const_0_x] = np.nan
    mse[near_const_0_x] = np.nan
    betas[near_const_0_x] = np.nan
    se[:,near_const_0_y] = np.nan
    mse[:,near_const_0_y] = np.nan
    betas[:,near_const_0_y] = np.nan
    ### Reset error settings to old
    np.seterr(**old_settings)
    return betas, se, mse


def magma_enrichment_general(Ys_df, Xs_df, method, covariates_df=None, error_cov=None, error_cov_block_labels=None):
    ### Check arguments, check indices, and extract data
    assert method in ["marginal", "joint"], "Method must be marginal or joint."
    assert (Ys_df.index == Xs_df.index).all(), "Xs_df and Ys_df indices don't match"
    Ys = Ys_df.values
    Xs = Xs_df.values
    if covariates_df is not None:
        assert (Ys_df.index == covariates_df.index).all(), "Ys_df and covariates_df indices don't match"
        covariates = covariates_df.values
    else:
        covariates = np.ones((Xs.shape[0], 1))
    ### Add intercept to covariates if not there
    if not np.isclose(covariates.var(), 0).any():
        covariates = np.hstack((covariates, np.ones((covariates.shape[0],1))))
    ### Decorrelate if applicable
    if error_cov is not None:
        assert error_cov.shape[0] == error_cov.shape[1], "error_cov is not square"
        assert error_cov.shape[0] == error_cov_block_labels.shape[0], "Block label dimensions don't match error_cov dimensions"
        assert Ys.shape[0] == error_cov.shape[0], "error_cov dimensions must match Ys_df dimensions"
        Linv = block_Linv(error_cov, error_cov_block_labels)
        Ys = block_AB(Linv, error_cov_block_labels, Ys)
        Xs = block_AB(Linv, error_cov_block_labels, Xs)
        covariates = block_AB(Linv, error_cov_block_labels, covariates)
    ### Project out covariates
    Ys_proj = project_out_V(Ys, covariates)
    Xs_proj = project_out_V(Xs, covariates)
    ### Compute enrichments
    if method == "marginal":
        betas, se, mse = batch_marginal_ols(Ys_proj, Xs_proj)
        pvals = scipy.stats.norm.sf(betas / se)
        multicol = pd.MultiIndex.from_tuples(itertools.product(["beta", "se", "pval"], Xs_df.columns),
                                             names=["statistic", "trait"])
        magma_enrichments = pd.DataFrame(columns=multicol, index=Ys_df.columns.values)
        magma_enrichments.beta = betas.T
        magma_enrichments.se = se.T
        magma_enrichments.pval = pvals.T
    elif method == "joint":
        assert Ys_proj.shape[1] == 1, "Does not support multiple targets for Ys. Make sure Ys.shape[1] == 1"
        results = sm.OLS(Ys_proj.flatten(), Xs_proj, has_const=False).fit()
        pvals = scipy.stats.norm.sf(results.params / results.bse)
        magma_enrichments = pd.DataFrame(np.array([results.params, results.bse, pvals]).T,
                                         index=Xs_df.columns.values,
                                         columns=["beta", "se", "pval"])
    return magma_enrichments

### --------------------------------- DIFFERENTIAL ENRICHMENT --------------------------------- ###

def two_tail_p_to_one_tail_p(two_tail_p, coef, tail="upper"):
    half_p = 0.5 * two_tail_p
    if tail == "upper":
        one_tail_p = half_p * (coef > 0) + (1 - half_p) * (coef <= 0)
    elif tail == "lower":
        one_tail_p = half_p * (coef < 0) + (1 - half_p) * (coef >= 0)
    else:
        raise ValueError("tail argument must be upper or lower")
    return one_tail_p


def compute_enrichments(Ys, Xs, cov):
    multicol = pd.MultiIndex.from_tuples(itertools.product(["coef", "se", "pvalue"], Ys.columns),
                                         names=["statistic", "trait"])
    res_df = pd.DataFrame(columns=multicol, index=Xs.columns)
    for y in Ys.columns:
        for x in Xs.columns:
            model = sm.OLS(Ys.loc[:,y], pd.concat([Xs.loc[:,[x]], cov], axis=1))
            results = model.fit(hasconst=True)
            res_df.loc[x, ("coef", y)] = results.params[x]
            res_df.loc[x, ("se", y)] = results.bse[x]
            res_df.loc[x, ("pvalue", y)] = two_tail_p_to_one_tail_p(results.pvalues[x],
                                                                    results.params[x],
                                                                    tail="upper")
    return res_df.astype(np.float64)


def batch_simple_linreg(Ys, Xs):
    return Ys.T.dot(Xs) / np.square(Xs).sum(axis=0)


def differential_enrichment_test(Ys_df, Xs_df, cov_df, num_perms=10000):
    assert (Ys_df.index == Xs_df.index).all(), "Ys_df and Xs_df indices don't match"
    assert (cov_df.index == Ys_df.index).all(), "cov_df and Ys_df indices don't match"
    Ys = Ys_df.values
    Xs = Xs_df.values
    cov = cov_df.values
    Ys_proj = project_out_V(Ys, cov)
    Xs_proj = project_out_V(Xs, cov)
    coefs = batch_simple_linreg(Ys_proj, Xs_proj)
    diff_enrich = coefs[0] - coefs[1]
    null_diff_enrich = []
    for i in range(num_perms):
        perm = np.random.permutation(Xs_proj.shape[0])
        null_coefs = batch_simple_linreg(Ys_proj, Xs_proj[perm])
        null_diff_enrich.append(null_coefs[0] - null_coefs[1])
    null_diff_enrich = np.array(null_diff_enrich)
    num_exceed = (np.abs(null_diff_enrich) >= np.abs(diff_enrich)).sum(axis=0)
    pval = (num_exceed + 1) / (num_perms + 1)
    res_df = pd.DataFrame([diff_enrich, pval, null_diff_enrich.mean(axis=0), null_diff_enrich.std(axis=0)]).T
    res_df.columns = ["diff_enrich", "pval", "null_mean", "null_std"]
    res_df.index = Xs_df.columns
    return res_df

### --------------------------------- NETWORK ENRICHMENT --------------------------------- ###

def edge_purity_statistics(adj_matrix, group_table):
    pure = False
    all_group_stats = []
    for g in range(group_table.shape[1]):
        pure_g = (group_table[:,g] * adj_matrix * group_table[:,g].reshape(-1,1)).astype(bool)
        pure = pure | pure_g
        all_group_stats.append(int(pure_g.sum() / 2))
    return int(pure.sum() / 2), all_group_stats

def permute_adj_matrix(adj_matrix):
    perm_adj_matrix = np.array(adj_matrix.copy())
    perm = np.random.permutation(perm_adj_matrix.shape[0])
    perm_adj_matrix = perm_adj_matrix[np.ix_(perm, perm)]
    return perm_adj_matrix

def compute_edge_purity_null(adj_matrix, group_table, num_permutations=100, random_seed=42):
    np.random.seed(random_seed)
    all_null_pure = []
    for i in range(num_permutations):
        null_pure, _ = edge_purity_statistics(permute_adj_matrix(adj_matrix), group_table)
        all_null_pure.append(null_pure)
    return np.array(all_null_pure)

def edge_purity_hypothesis_test(adj_matrix, group_table, num_permutations=100, random_seed=42):
    pure, group_pure = edge_purity_statistics(adj_matrix, group_table)
    all_null_pure = compute_edge_purity_null(adj_matrix, group_table, num_permutations=num_permutations)
    pval = ((all_null_pure >= pure).sum() + 1) / (all_null_pure.shape[0] + 1)
    return pure, group_pure, pval

### --------------------------------- MAIN CONTROL --------------------------------- ###

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
    
    ### Load NMF data
    if config_dict["nmf_out_prefix"] is not None:
        logging.info("Loading NMF data from {}".format(config_dict["nmf_out_prefix"]))
        nmf_score_norm_df = pd.read_csv(config_dict["nmf_out_prefix"] + ".nmf_score_norm", sep="\t", index_col=0)
        logging.info("Defining high-confidence NMF genes (score >= {})".format(config_dict["nmf_thresh"], config_dict["nmf_thresh"]))
        nmf_thresh_df = (nmf_score_norm_df >= config_dict["nmf_thresh"]).astype(int)

    ### NMF + original MAGMA enrichments
    if config_dict["trait_magma_out_prefix"] is not None:
        assert config_dict["nmf_out_prefix"] is not None and config_dict["gene_annot_path"] is not None, "If trait_magma_out_prefix is provided, then nmf_out_prefix and gene_annot_path must be provided"
        logging.info("Loading gene annotation table and magma from {} and {}".format(config_dict["gene_annot_path"], config_dict["trait_magma_out_prefix"]))
        gene_annot_df = read_gene_annot_df(config_dict["gene_annot_path"])
        Y, covariates, error_cov, Y_ids = read_magma(config_dict["trait_magma_out_prefix"], True, True)
        error_cov = regularize_error_cov(error_cov, Y_ids, gene_annot_df)
        hla_genes = get_hla_genes(gene_annot_df)
        training_genes = np.array(sorted(list(set(Y_ids).difference(hla_genes))))
        Y_train_inds = get_indices_in_target_order(Y_ids, training_genes)
        ### Subset, format, and reorder data
        Y_train_df = pd.DataFrame(Y[Y_train_inds].reshape(-1,1), index=training_genes, columns=["magma_score"])
        covariates_df = pd.DataFrame(covariates[Y_train_inds], index=training_genes, columns=["COVAR{}".format(i+1) for i in range(covariates.shape[1])])
        train_error_cov = error_cov[np.ix_(Y_train_inds, Y_train_inds)]
        train_error_cov_block_labels = gene_annot_df.loc[training_genes].CHR.values
        ### Compute MAGMA enrichments
        logging.info("Computing enrichment of components and original MAGMA scores")
        nmf_marginal_assoc = magma_enrichment_general(Y_train_df,
                                                      nmf_score_norm_df.loc[training_genes],
                                                      "marginal",
                                                      covariates_df=covariates_df,
                                                      error_cov=train_error_cov,
                                                      error_cov_block_labels=train_error_cov_block_labels)
        nmf_joint_assoc = magma_enrichment_general(Y_train_df,
                                                   nmf_score_norm_df.loc[training_genes],
                                                   "joint",
                                                    covariates_df=covariates_df,
                                                    error_cov=train_error_cov,
                                                    error_cov_block_labels=train_error_cov_block_labels)
        nmf_thresh_marginal_assoc = magma_enrichment_general(Y_train_df,
                                                             nmf_thresh_df.loc[training_genes],
                                                             "marginal",
                                                             covariates_df=covariates_df,
                                                             error_cov=train_error_cov,
                                                             error_cov_block_labels=train_error_cov_block_labels)
        nmf_thresh_joint_assoc = magma_enrichment_general(Y_train_df,
                                                          nmf_thresh_df.loc[training_genes],
                                                          "joint",
                                                          covariates_df=covariates_df,
                                                          error_cov=train_error_cov,
                                                          error_cov_block_labels=train_error_cov_block_labels)
        logging.info("Saving component + original MAGMA enrichments")
        pass

    ### MAGMA validation
    if config_dict["magma_validation_table"] is not None:
        assert config_dict["trait_magma_out_prefix"] is not None, "If magma_validation_table is provided, then trait_magma_out_prefix must be provided"
        logging.info("Loading MAGMA validation from {}".format(config_dict["magma_validation_table"]))
        magma_validation_df = pd.read_csv(config_dict["magma_validation_table"], sep="\t", index_col=0)
        magma_validation_df = Y_train_df.loc[:,[]].merge(magma_validation_df, how="left",
                                                         left_index=True, right_index=True)
        magma_validation_df = magma_validation_df.fillna(magma_validation_df.mean())
        covariates_with_magma_df = covariates_df
        for power in range(1,4):
            Y_power_df = Y_train_df.pow(power).rename(columns={"magma_score":"Y{}".format(power)})
            covariates_with_magma_df = pd.concat([covariates_with_magma_df, Y_power_df], axis=1)
        logging.info("Computing MAGMA validation enrichments")
        nmf_magma_validation_assoc = magma_enrichment_general(magma_validation_df,
                                                              nmf_score_norm_df.loc[training_genes],
                                                              "marginal",
                                                              covariates_df=covariates_with_magma_df,
                                                              error_cov=train_error_cov,
                                                              error_cov_block_labels=train_error_cov_block_labels)
        nmf_thresh_magma_validation_assoc = magma_enrichment_general(magma_validation_df,
                                                                     nmf_thresh_df.loc[training_genes],
                                                                     "marginal",
                                                                     covariates_df=covariates_with_magma_df,
                                                                     error_cov=train_error_cov,
                                                                     error_cov_block_labels=train_error_cov_block_labels)
        logging.info("Saving MAGMA validation enrichments")
        pass

    ### Epigenetic validation
    if config_dict["epigenetic_validation_table"] is not None:
        assert config_dict["trait_magma_out_prefix"] is not None, "If epigenetic_validation_table is provided, then trait_magma_out_prefix must be provided"
        logging.info("Loading epigenetic validation from {}".format(config_dict["epigenetic_validation_table"]))
        epigenetic_df = pd.read_csv(config_dict["epigenetic_validation_table"], sep="\t", index_col=0)
        epigenetic_Y_df = Y_train_df.loc[:,[]].merge(epigenetic_df, how="left",
                                                     left_index=True, right_index=True)
        epigenetic_Y_df = epigenetic_Y_df.fillna(0)
        logging.info("Computing epigenetic enrichments")
        epi_magma_assoc = magma_enrichment_general(Y_train_df,
                                                   epigenetic_Y_df,
                                                   "marginal",
                                                   covariates_df=covariates_df,
                                                   error_cov=train_error_cov,
                                                   error_cov_block_labels=train_error_cov_block_labels)
        epi_magma_assoc_exps = epi_magma_assoc.pval.columns.values[epi_magma_assoc.pval.loc["magma_score"] < 0.05]
        epi_comp_fisher_df = batch_one_sided_fisher_exact_test(nmf_thresh_df.loc[epigenetic_df.index], epigenetic_df)
        epi_comp_jaccard_df = jaccard_similarity(nmf_thresh_df.loc[epigenetic_df.index], epigenetic_df)
        epi_magma_comp_exps = epi_magma_assoc_exps[(epi_comp_fisher_df.loc[epi_magma_assoc_exps].pvalue < 0.05 / epi_comp_fisher_df.shape[0]).any(axis=1)]
        logging.info("Saving epigenetic enrichments")
        pass

    ### Pathway enrichments
    if config_dict["pathways_table"] is not None:
        assert config_dict["trait_magma_out_prefix"] is not None, "If pathways_table is provided, then trait_magma_out_prefix must be provided"
        logging.info("Loading pathway data from {}".format(config_dict["pathways_table"]))
        pathways_df = pd.read_csv(config_dict["pathways_table"], sep="\t", index_col=0)
        pathways_Y_df = Y_train_df.loc[:,[]].merge(pathways_df, how="left",
                                                   left_index=True, right_index=True)
        pathways_Y_df = pathways_Y_df.fillna(0)
        logging.info("Computing pathway enrichments")
        pathway_magma_assoc = magma_enrichment_general(Y_train_df,
                                                       pathways_Y_df,
                                                       "marginal",
                                                       covariates_df=covariates_df,
                                                       error_cov=train_error_cov,
                                                       error_cov_block_labels=train_error_cov_block_labels)
        magma_assoc_pathways = pathway_magma_assoc.pval.columns.values[pathway_magma_assoc.pval.loc["magma_score"] < 0.05]
        pathway_comp_fisher_df = batch_one_sided_fisher_exact_test(nmf_thresh_df.loc[pathways_df.index], pathways_df)
        pathway_comp_jaccard_df = jaccard_similarity(nmf_thresh_df.loc[pathways_df.index], pathways_df)
        pathway_magma_comps = magma_assoc_pathways[(pathway_comp_fisher_df.loc[magma_assoc_pathways].pvalue < 0.05 / pathway_comp_fisher_df.shape[0]).any(axis=1)]
        logging.info("Saving pathway enrichments")
        pass

    ### Annotate high confidence network edges
    if config_dict["net_out_prefix"] is not None:
        logging.info("Loading network from {}".format(config_dict["net_out_prefix"]))
        net_df = pd.read_csv(config_dict["net_out_prefix"] + ".net", sep="\t", index_col=0)
        net_norm_df = pd.read_csv(config_dict["net_out_prefix"] + ".net_norm", sep="\t", index_col=0)
        logging.info("Annotating high confidence edges (score >= {})".format(config_dict["net_thresh"]))
        high_conf_net_edges = [(net_norm_df.index[r], net_norm_df.columns[c])
                               for r, c in zip(*np.where(net_norm_df >= config_dict["net_thresh"]))
                               if r < c]
        net_edge_summary_df = pd.DataFrame(columns=["gene1", "gene2", "score", "norm_score"])
        if config_dict["nmf_out_prefix"] is not None:
            net_edge_summary_df["co_occur_components"] = np.nan
        if config_dict["ppi_pairs"] is not None:
            ppi_pairs = np.loadtxt(config_dict["ppi_pairs"], dtype=str)
            ppi_set = set([tuple(p) for p in ppi_pairs])
            net_edge_summary_df["ppi_interaction"] = np.nan
        if config_dict["epigenetic_validation_table"] is not None:
            net_edge_summary_df["all_epi_data"] = np.nan
            net_edge_summary_df["all_epi_data"] = net_edge_summary_df["all_epi_data"].astype(object)
            epi_comp_jaccard_score_df = format_score_from_jaccard(epi_comp_jaccard_df)
            epi_overlap_net_df = epigenetic_df.loc[epigenetic_df.index.isin(net_df.index)].sum()
            epi_comp_jaccard_score_df["net_overlap"] = epi_overlap_net_df
            epi_comp_jaccard_score_df = epi_comp_jaccard_score_df.rename(columns={"jaccard":"nmf_jaccard", "overlap":"nmf_overlap"})
        if config_dict["pathways_table"] is not None:
            net_edge_summary_df["all_pathway_data"] = np.nan
            net_edge_summary_df["all_pathway_data"] = net_edge_summary_df["all_pathway_data"].astype(object)
            pathway_comp_jaccard_score_df = format_score_from_jaccard(pathway_comp_jaccard_df)
            pathway_overlap_net_df = pathways_df.loc[pathways_df.index.isin(net_df.index)].sum()
            pathway_comp_jaccard_score_df["net_overlap"] = pathway_overlap_net_df
            pathway_comp_jaccard_score_df = pathway_comp_jaccard_score_df.rename(columns={"jaccard":"nmf_jaccard", "overlap":"nmf_overlap"})
        for n1, n2 in high_conf_net_edges:
            curr_ind = len(net_edge_summary_df.index)
            ### Annotate basics
            net_edge_summary_df.loc[curr_ind] = np.nan
            net_edge_summary_df.loc[curr_ind,"gene1"] = n1
            net_edge_summary_df.loc[curr_ind,"gene2"] = n2
            net_edge_summary_df.loc[curr_ind,"score"] = net_df.loc[n1,n2]
            net_edge_summary_df.loc[curr_ind,"norm_score"] = net_norm_df.loc[n1,n2]
            if "co_occur_components" in net_edge_summary_df.columns:
                co_occuring_comps = nmf_thresh_df.columns.values[nmf_thresh_df.loc[[n1,n2]].all(axis=0)]
                net_edge_summary_df.loc[curr_ind,"co_occur_components"] = ", ".join(co_occuring_comps)
            if "ppi_interaction" in net_edge_summary_df.columns:
                net_edge_summary_df.loc[curr_ind,"ppi_interaction"] = (n1, n2) in ppi_set or (n2, n1) in ppi_set
            if "all_epi_data" in net_edge_summary_df.columns:
                co_occuring_epi = np.array([])
                if n1 in epigenetic_df.index and n2 in epigenetic_df.index:
                    co_occuring_epi = epigenetic_df.columns.values[epigenetic_df.loc[[n1,n2]].all(axis=0)]
                enriched_co_occuring_epi = np.array(list(set(epi_magma_comp_exps).intersection(co_occuring_epi)))
                net_edge_summary_df.at[curr_ind,"all_epi_data"] = [tuple(d) for d in epi_comp_jaccard_score_df.loc[enriched_co_occuring_epi].sort_values("nmf_jaccard", ascending=False).reset_index().values]
            if "all_pathway_data" in net_edge_summary_df.columns:
                co_occuring_pathway = np.array([])
                if n1 in pathways_df.index and n2 in pathways_df.index:
                    co_occuring_pathway = pathways_df.columns.values[pathways_df.loc[[n1,n2]].all(axis=0)]
                enriched_co_occuring_pathway = np.array(list(set(pathway_magma_comps).intersection(co_occuring_pathway)))
                net_edge_summary_df.at[curr_ind,"all_pathway_data"] = [tuple(d) for d in pathway_comp_jaccard_score_df.loc[enriched_co_occuring_pathway].sort_values("nmf_jaccard", ascending=False).reset_index().values]
        net_edge_summary_df = net_edge_summary_df.sort_values("score", ascending=False).reset_index().drop("index", axis=1)
        logging.info("Saving network summary")
        pass
        
    ### Polygenic contrast test
    if config_dict["polygenic_contrast_score_table"] is not None:
        assert config_dict["trait_magma_out_prefix"] is not None, "If polygenic_contrast_score_table is provided, then trait_magma_out_prefix must be provided"
        logging.info("Conducting polygenic contrast test with data at {}".format(config_dict["polygenic_contrast_score_table"]))
        polygenic_contrast_score_df = pd.read_csv(config_dict["polygenic_contrast_score_table"], sep="\t", index_col=0)
        covariates_with_magma_df = covariates_df
        for power in range(1,4):
            Y_power_df = Y_train_df.pow(power).rename(columns={"magma_score":"Y{}".format(power)})
            covariates_with_magma_df = pd.concat([covariates_with_magma_df, Y_power_df], axis=1)
        covariates_with_magma_df["INTERCEPT"] = 1.
        ### Get training genes that are common to all tables
        common_genes = set(polygenic_contrast_score_df.index)
        common_genes = common_genes.intersection(covariates_with_magma_df.index)
        common_genes = common_genes.intersection(nmf_score_norm_df.index)
        common_genes = np.array(sorted(list(common_genes)))
        ### Subset to training genes
        covariates_with_magma_df = covariates_with_magma_df.loc[common_genes]
        polygenic_contrast_score_df = polygenic_contrast_score_df.loc[common_genes]
        nmf_score_polygenic_contrast = nmf_score_norm_df.loc[common_genes]
        nmf_score_thresh_polygenic_contrast = nmf_thresh_df.loc[common_genes].astype(float)
        ### Compute enrichment tables
        enrich_df = compute_enrichments(polygenic_contrast_score_df, nmf_score_polygenic_contrast, covariates_with_magma_df)
        logging.info("Conducting {} permutations (continuous components)".format(config_dict["polygenic_contrast_num_permutations"]))
        diff_enrich_df = differential_enrichment_test(polygenic_contrast_score_df, nmf_score_polygenic_contrast, covariates_with_magma_df, num_perms=config_dict["polygenic_contrast_num_permutations"])
        enrich_thresh_df = compute_enrichments(polygenic_contrast_score_df, nmf_score_thresh_polygenic_contrast, covariates_with_magma_df)
        logging.info("Conducting {} permutations (binarized components)".format(config_dict["polygenic_contrast_num_permutations"]))
        diff_thresh_enrich_df = differential_enrichment_test(polygenic_contrast_score_df, nmf_score_thresh_polygenic_contrast, covariates_with_magma_df, num_perms=config_dict["polygenic_contrast_num_permutations"])
        logging.info("Saving polygenic contrast test results")
        pass

    ### Network purity test
    if config_dict["network_purity_gene_table"] is not None:
        assert config_dict["net_out_prefix"] is not None, "If network_purity_gene_table is provided, then net_out_prefix must be provided"
        logging.info("Conducting network purity test with data at {}".format(config_dict["network_purity_gene_table"]))
        ### Load data
        network_purity_genes_df = pd.read_csv(config_dict["network_purity_gene_table"], sep="\t", index_col=0)
        ### Subset to test genes
        network_purity_genes_background = network_purity_genes_df.index.values[network_purity_genes_df.any(axis=1)]
        test_genes = np.array(sorted(list(set(net_norm_df.index).intersection(network_purity_genes_background))))
        network_purity_genes_df = network_purity_genes_df.loc[test_genes].astype(int)
        net_norm_df = net_norm_df.loc[test_genes, test_genes]
        ### Run tests
        all_thresh = np.arange(0, 1, config_dict["network_purity_thresh_interval"])
        all_thresh = np.array([np.round(t, 4) for t in all_thresh])
        all_group_pure_edge_column_names = ["pure_edges_{}".format(g.replace(" ", "_")) for g in network_purity_genes_df.columns]
        all_purity_stats_df = pd.DataFrame(columns=["thresh", "pure_edges", "total_edges", "pval"] + all_group_pure_edge_column_names)
        logging.info("Conducting {} permutations for each of {} thresholds".format(config_dict["network_purity_num_permutations"], len(all_thresh)))
        for thresh in all_thresh:
            thresh_net = (net_norm_df >= thresh).values.astype(int)
            np.fill_diagonal(thresh_net, 0)
            thresh_net_total_edges = int(thresh_net.sum() / 2)
            pure, group_pure, pval = edge_purity_hypothesis_test(thresh_net, network_purity_genes_df.values, num_permutations=config_dict["network_purity_num_permutations"])
            curr_df_ind = len(all_purity_stats_df.index)
            all_purity_stats_df.loc[curr_df_ind] = [thresh, pure, thresh_net_total_edges, pval] + list(group_pure)
        all_purity_stats_df["pure_edges"] = all_purity_stats_df["pure_edges"].astype(int)
        all_purity_stats_df["total_edges"] = all_purity_stats_df["total_edges"].astype(int)
        all_purity_stats_df[all_group_pure_edge_column_names] = all_purity_stats_df[all_group_pure_edge_column_names].astype(int)
        logging.info("Saving network purity test results")
        pass



### Main
if __name__ == '__main__':
    ### Get arguments
    args = get_args()
    config_dict = vars(args)
    main(config_dict)

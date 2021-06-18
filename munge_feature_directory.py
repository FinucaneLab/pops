import numpy as np
import pandas as pd
import glob
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Converts a directory of feature files into efficient NumPy format, written out to multiple chunks, amenable for use downstream with PoPS.')
    parser.add_argument("--gene_annot_path", help="Path to gene annotation table. For the purposes of this script, only require that there is an ENSGID column.")
    parser.add_argument("--feature_dir", help="Directory where raw feature files live. Each feature file must be a tab-separated file with a header for column names and the first column must be the ENSGID. Will process every file in the directory so make sure every file is a feature file and there are no hidden files. Please also make sure the column names are unique across all feature files. The easiest way to ensure this is to prefix every column with the filename.")
    parser.add_argument("--nan_policy", default="raise", help="What to do if a feature file is missing ENSGIDs that are in gene_annot_path. Takes the values \"raise\" (raise an error), \"ignore\" (ignore and write out with nans), \"mean\" (impute the mean of the feature), and \"zero\" (impute 0). Default is \"raise\".")
    parser.add_argument("--save_prefix", help="Prefix to the output path. For each chunk i, 2 files will be written: {save_prefix}_mat.{i}.npy, {save_prefix}_cols.{i}.txt. Furthermore, row data will be written to {save_prefix}_rows.txt")
    parser.add_argument("--max_cols", default=5000, type=int, help="Maximum number of columns per output chunk. Default is 5000.")

    args = parser.parse_args()
    gene_annot_path = args.gene_annot_path
    feature_dir = args.feature_dir
    nan_policy = args.nan_policy
    save_prefix = args.save_prefix
    MAX_COLS = args.max_cols
    
    assert nan_policy in ["raise", "ignore", "mean", "zero"], "Invalid argument for flag --nan_policy. Accepts \"raise\", \"ignore\", \"mean\", and \"zero\"."

    gene_annot_df = pd.read_csv(gene_annot_path, sep="\t", index_col="ENSGID").iloc[:,0:0]
    row_data = gene_annot_df.index.values
    np.savetxt(save_prefix + ".rows.txt", row_data, fmt="%s")

    #### Sort for canonical ordering
    all_feature_files = sorted([f for f in glob.glob(feature_dir + "/*")])

    all_mat_data = []
    all_col_data = []
    curr_block_index = 0
    for f in all_feature_files:
        f_df = pd.read_csv(f, sep="\t", index_col=0).astype(np.float64)
        f_df = gene_annot_df.merge(f_df, how="left", left_index=True, right_index=True)
        if nan_policy == "raise":
            assert not f_df.isnull().values.any(), "Missing genes in feature matrix."
        elif nan_policy == "ignore":
            pass
        elif nan_policy == "mean":
            f_df = f_df.fillna(f_df.mean())
        elif nan_policy == "zero":
            f_df = f_df.fillna(0)
        mat = f_df.values
        cols = f_df.columns.values
        all_mat_data.append(mat)
        all_col_data += list(cols)
        while len(all_col_data) >= MAX_COLS:
            ### Flush MAX_COLS columns to disk at a time
            mat = np.hstack(all_mat_data)
            save_mat = mat[:,:MAX_COLS]
            keep_mat = mat[:,MAX_COLS:]
            save_cols = all_col_data[:MAX_COLS]
            keep_cols = all_col_data[MAX_COLS:]
            ### Save
            np.save(save_prefix + ".mat.{}.npy".format(curr_block_index), save_mat)
            np.savetxt(save_prefix + ".cols.{}.txt".format(curr_block_index), save_cols, fmt="%s")
            ### Update variables
            all_mat_data = [keep_mat]
            all_col_data = keep_cols
            curr_block_index += 1
    ### Flush last block
    if len(all_col_data) > 0:
        mat = np.hstack(all_mat_data)
        np.save(save_prefix + ".mat.{}.npy".format(curr_block_index), mat)
        np.savetxt(save_prefix + ".cols.{}.txt".format(curr_block_index), all_col_data, fmt="%s")


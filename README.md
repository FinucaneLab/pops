# Polygenic Priority Score (PoPS) v0.2

PoPS is a gene prioritization method that leverages genome-wide signal from GWAS summary statistics and incorporates data from an extensive set of public bulk and single-cell expression datasets, curated biological pathways, and predicted protein-protein interactions. The PoPS method is described in detail at

Weeks, E. M. et al. [Leveraging polygenic enrichments of gene features to predict genes underlying complex traits and diseases](https://www.medrxiv.org/content/10.1101/2020.09.08.20190561v1). *medRxiv* (2020).

Detailed below is an example workflow for running PoPS with a reduced set of features on 3 publicly available summary statistics (whose MAGMA [citation] scores are provided with the repository at `data/base_scores/`). Step 0 preprocesses the feature matrix and must be run once; it only needs to be rerun if the feature matrix changes.

## Step 0: Munge features

The script at src/munge_feature_files.py accepts a directory of feature files and processes them into a more efficient format for downstream usage. Example features and a gene annotation file are provided. To generate the data necessary for the example, run the snippet below. This will write features to the directory `data/features_munged/`

```
python munge_feature_files.py \
 --gene_annot_path data/utils/gene_annot_jun10.txt \
 --feature_dir data/features_raw/ \
 --save_prefix data/features_munged/pops_features \
 --max_cols 500
```

| Flag | Description |
|-|-|
| --gene_annot_path | Path to gene annotation table. For the purposes of this script, only require that there is an ENSGID column |
| --feature_dir | Directory where raw feature files live. Each feature file must be a tab-separated file with a header for column names and the first column must be the ENSGID. Will process every file in the directory so make sure every file is a feature file and there are no hidden files. Please also make sure the column names are unique across all feature files. The easiest way to ensure this is to prefix every column with the filename |
| --nan_policy | What to do if a feature file is missing ENSGIDs that are in gene_annot_path. Takes the values "raise" (raise an error), "ignore" (ignore and write out with nans), "mean" (impute the mean of the feature), and "zero" (impute 0). Default is "raise" |
| --save_prefix | Prefix to the output path. For each chunk i, 2 files will be written: {save_prefix}_mat.{i}.npy, {save_prefix}_cols.{i}.txt. Furthermore, row data will be written to {save_prefix}_rows.txt |
| --max_cols | Maximum number of columns per output chunk. Default is 5000 |

## Step 1: Generate MAGMA scores

## Step 2: Run PoPS


# Polygenic Priority Score (PoPS) v0.2

PoPS is a gene prioritization method that leverages genome-wide signal from GWAS summary statistics and incorporates data from an extensive set of public bulk and single-cell expression datasets, curated biological pathways, and predicted protein-protein interactions. The PoPS method is described in detail at

Weeks, E. M. et al. [Leveraging polygenic enrichments of gene features to predict genes underlying complex traits and diseases](https://www.medrxiv.org/content/10.1101/2020.09.08.20190561v1). *medRxiv* (2020).

Detailed below is an example workflow for running PoPS with a reduced set of features on a set of publicly available summary statistics for schizophrenia. Step 0 preprocesses the feature matrix and must be run once; it only needs to be rerun if the feature matrix changes.

To download the full set of feature files used to generate the results in the manuscript, please see [TODO: ADD LINK].

## Step 0: Munge features

The script at src/munge_feature_files.py accepts a directory of feature files and processes them into a more efficient format for downstream usage. Example features and a gene annotation file are provided. To generate the data necessary for the example, run the snippet below. This will write chunks of the feature matrix to the directory `data/features_munged/` with the prefix `pops_features`.

```
python munge_feature_directory.py \
 --gene_annot_path example/data/utils/gene_annot_jun10.txt \
 --feature_dir example/data/features_raw/ \
 --save_prefix example/data/features_munged/pops_features \
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

PoPS, by default, accepts gene-level association statistics outputted by the MAGMA [de Leeuw et al. 2015 PLOS Computational Biology] software as input. For this example, these files have already been generated and are provided in `example/data/magma_scores/`. For instructions on running MAGMA, please see the [MAGMA documentation](https://ctg.cncr.nl/software/magma), as well as the PoPS manuscript for the particular settings used. PoPS requires a .genes.out and a .genes.raw file.

Briefly, MAGMA scores can be generated with the following command:

```
./magma \
 --bfile {PATH_TO_REFERENCE_PANEL_PLINK} \
 --gene-annot {PATH_TO_MAGMA_ANNOT}.genes.annot \
 --pval {PATH_TO_SUMSTATS}.sumstats ncol=N \
 --gene-model snp-wise=mean \
 --out {OUTPUT_PREFIX}
```

## Step 2: Run PoPS

PoPS accepts many command line flags which can be used to customize the regression. To keep this tutorial simple, we will only describe how to use PoPS with the default (and recommended) settings. To learn more about the options that PoPS accepts, please run `python pops.py --help` from the command line.

This command will write `PASS_Schizophrenia.coefs`, `PASS_Schizophrenia.marginals`, and `PASS_Schizophrenia.preds` to `example/out/`. Add the `--verbose` flag to get verbose output.

```
python pops.py \
 --gene_annot_path example/data/utils/gene_annot_jun10.txt \
 --feature_mat_prefix example/data/features_munged/pops_features \
 --num_feature_chunks 2 \
 --magma_prefix example/data/magma_scores/PASS_Schizophrenia \
 --control_features_path example/data/utils/features_jul17_control.txt \
 --out_prefix example/out/PASS_Schizophrenia
```

| Flag | Description |
|-|-|
| --gene_annot_path | Path to tab-separated gene annotation file. Must contain ENSGID, CHR, and TSS columns |
| --feature_mat_prefix | Prefix to the split feature matrix files, such as those outputted by munge_feature_directory.py. There must be .mat.*.npy files, .cols.*.txt files, and a .rows.txt file |
| --num_feature_chunks | Prefix to the gene-level association statistics outputted by MAGMA. There must be a .genes.out file and a .genes.raw file |
| --magma_prefix | Prefix to the gene-level association statistics outputted by MAGMA. There must be a .genes.out file and a .genes.raw file |
| --control_features_path | Optional path to list of features (one per line) to always include |
| --out_prefix | Prefix that results will be saved with. Will write out a .preds, .coefs, and .marginals file |
| --verbose | Set this flag to get verbose output |

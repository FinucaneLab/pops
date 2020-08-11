# Polygeneic Priority Score (PoPS) v0.1
Please cite the following manuscript for the PoPS method:

Weeks et al. [Leveraging polygenic enrichments of gene features to predict genes underlying complex traits and diseases]() 2020 *bioRxiv*

## Getting started

#### Dependencies
- pandas
- numpy
- argparse
- statsmodels
- scipy
- sklearn

#### Software
[MAGMA](https://ctg.cncr.nl/software/magma) v1.07b

#### Data
1. SNP location
   - The SNP location file should contain three columns: SNP ID, chromosome, and base pair position. These should be the first three columns in that file (additional columns are ignored). The only exception is if you use a .bim file from binary PLINK data, this can be provided to MAGMA as a SNP location file without modification.
   - The data/1000G.EUR.bim file is a SNP location file for all SNPs in the 1000 Genomes Project.
2. Gene location
   - The gene location file must have four columns containing the gene ID, chromosome, start position and stop position, in that order. These columns should be labeled ENSGID, CHR, START, END.
   - The data/gene_loc.txt file is a gene location file for 18,383 protein coding genes.
3. MAGMA gene annotation
   - The MAGMA gene annotation file is created by running MAGMA with the --annotate flag.
   - Each row of the MAGMA annotation file corresponds to a gene and containings the gene ID, a specification of the gene's location, and a list of SNP IDs of SNPs mapped to that gene.
   - The data/magma_0kb.genes.annot file is a MAGMA annotation file for the 18,383 protein coding genes using SNPs in the 1000 Genomes EUR reference panel and a 0 Kb window around the gene body.
4. Reference panel
   - A binary PLINK format data set, consisting of a .bed, .bim and .fam trio of files, is required for the reference panel.
   - The data/1000G.EUR.bed/bim/fam files contain the necessary reference panel data for Europeans in the 1000 Genomes Project.
5. Gene features
   - The first column of the gene feature file must contain gene Ensemble IDs and be labeled ENSGID. Remaining columns must have unique column names and contain gene features.
   - We provide data for 57,543 gene features in the data/POPS.features.txt.gz: 40,546 derived from gene expression data, 8,718 extracted from a protein-protein interaction network, and 8,479 based on pathway membership.
6. Control features
   - A list of the names of control features always included in PoPS analyses.
   - The data/control.features contains the names of the relevant control features for the gene features provided.
7. Summary statistics
   - The summary statistics file must contain a column containing SNP IDs, p-values, the sample size used per SNP can be included in the analysis and be labeled SNP, P, and N.
   - An example summary statistics file is provided in the examples/AFib.sumstats file.


## Typical analysis
The typical PoPS analysis takes GWAS summary statistics together with gene features to compute PoP scores for each gene and is composed of 3 main steps.
1. Compute gene association statistics (z-scores) using MAGMA.
2. Select marginally associated features by performing enrichment analysis for each gene feature separately.
3. Estimate Polygenic Priority Scores (PoP scores) by fitting a joint model for the enrichment of all selected features.

### Step 1 - Compute gene assocaition statistics
#### Inputs
##### `--bfile`
This flag gives the location of the plink format file set for the reference panel described in `Data`.
##### `--gene_annot`
This flag gives the location of the MAGMA gene annotation file described in `Data`.
##### `--pval`
This flag gives the location of the summary statistics file described above and designates the name of the column containing the per SNP sample size.
##### `--gene-model`
This flag designates which model to use for computing gene association statistics. We recommend setting snp-wise=mean.
##### `--out`
This flag designates the prefix for where to print the output. MAGMA will append .genes.out and .genes.raw to this prefix.
#### Outputs
##### `.genes.out` file
The .genes.out file contains the MAGMA gene analysis results in human-readable format. This file contains the gene z-scores and relevant data to construct the control covariates in the joint prediction model.
##### `.genes.raw` file
The .genes.raw file is the intermediary file that serves as the input for subsequent analyses. This file contains the required data to consturct the gene-gene correlation matrix.
#### Sample command
```
./magma\
	--bfile 1000G.EUR\
	--gene-annot magma_0kb.genes.annot\
	--pval AFib.sumstats ncol=N\
	--gene-model snp-wise=mean\
	--out AFib
```
For more detail, see the [MAGMA manual](https://ctg.cncr.nl/software/MAGMA/doc/manual_v1.07.pdf).

### Step 2 - Select features
#### Inputs
##### `--features`
This flag gives the location of the gene feature file described in `Data`.
##### `--gene_results`
This flag gives the prefix for location of the gene association results from `Step 1`.
##### `--out`
This flag designates the prefix for where to print the output. It will append .features to this prefix.
#### Outputs
##### `.features` file
The .features file contains the names of the marginally selected features. This file has no header and contains the name of one feature per row.
#### Sample command
```
python POPS.feature_selection.py\
	--features POPS.features.txt.gz\
	--gene_results AFib\
	--out AFib
```

### Step 3 - Estimate PoP scores
#### Inputs
##### `--gene_loc`
This flag gives the location of the gene location file described in `Data`.
#####  `--gene_results`
This flag gives the prefix for location of the gene association results from `Step 1`.
##### `--features`
This flag gives the location of the gene feature file described in `Data`.
##### `--selected_features`
This flag gives the prefix for the location of the list of selected features from `Step 2`.
##### `--control_features`
This flag gives the location of the list of control fetures described in `Data`.
##### `--chromosome`
This flag designates the chromosome for which to compute PoP scores.
##### `--out`
This flag designates the prefix for where to print the output. It will append .coefs and .results to this prefix.
#### Outputs
##### `.{chomosome}.results` file
The .results file contains the predicted PoP scores for each gene on the designated chromosome.
##### `.{chomosome}.coefs` file
The .coefs file contains the estimated $\hat{\Beta}$ for each feature from fitting the PoPS model leaving out the designated chromosome.
#### Sample command
```
python POPS.predict_scores.py\
	--gene_loc gene_loc.txt\
	--gene_results AFib\
	--features POPS.features.txt.gz\
	--selected_features AFib.features\
	--control_features control.features\
	--chromosome 1\
	--out AFib

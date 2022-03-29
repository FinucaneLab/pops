import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse.csgraph import connected_components
import xlsxwriter
from collections import defaultdict

import scipy.stats

from fpdf import FPDF
from PIL import Image
from pathlib import Path

import argparse

### Helper functions

def cluster_components(sim_df, sim_thresh):
    graph = (sim_df >= sim_thresh).values
    _, labels = connected_components(graph)
    cluster_df = pd.DataFrame(np.array([labels]).T, index=sim_df.index, columns=["cluster_id"])
    return cluster_df

def group_sort(group_ids, values):
    group_ids = np.array(group_ids)
    values = np.array(values)
    argsort_vals = np.argsort(values)
    reorder_inds = []
    for i in argsort_vals:
        if i in reorder_inds:
            continue
        in_group = (group_ids == group_ids[i])
        reorder_inds += list(np.where(in_group)[0][np.argsort(values[in_group])])
    return reorder_inds

def load_df_from_npz(filename):
    with np.load(filename, allow_pickle=True) as f:
        obj = pd.DataFrame(**f)
    return obj

def prep_nlog_pval(nlog_pval, scores, effective_infinity):
    all_vals = nlog_pval.values.flatten()
    all_vals = all_vals[~np.isinf(all_vals) & ~pd.isnull(all_vals)]
    max_meaningful_val = all_vals.max()
    new_effective_infinity = max(effective_infinity, max_meaningful_val + 1)
    shifted_scores = scores - scores.min().min() + 1
    prep_nlog_pval = nlog_pval.replace(np.inf, new_effective_infinity)
    prep_nlog_pval = prep_nlog_pval + np.isinf(nlog_pval) * shifted_scores
    return prep_nlog_pval, max_meaningful_val

def bar_chart_general(labels, values, max_value, ylabel=None, hlines=None, groups=None, group_color_dict=None, legend_mask=0):
    fig, ax = plt.subplots(1,1,figsize=(20,4))
    pos = np.arange(len(values))
    clipped_values = np.minimum(values, max_value)
    if groups is not None:
        color = [group_color_dict[g] for g in groups]
    else:
        color = "#C3BEF7"
    if groups is not None and legend_mask > 0:
        ax.bar(pos, list(clipped_values[:-legend_mask]) + [0] * legend_mask, color=color)
        ax.set_xticks(pos[:-legend_mask])
        ax.set_xticklabels(labels[:-legend_mask], rotation=45, ha="right")
    else:
        ax.bar(pos, clipped_values, color=color)
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, rotation=45, ha="right")
    for i in range(len(values)):
        if values[i] > max_value:
            ax.scatter(pos[i], max_value + max_value * 0.01, s=80, marker=10, color="#999999")
    if groups is not None:
        if legend_mask:
            unique_groups = pd.DataFrame(groups[:-legend_mask]).iloc[:,0].unique()
        else:
            unique_groups = pd.DataFrame(groups).iloc[:,0].unique()
        handles = [plt.Rectangle((0,0),1,1, color=group_color_dict[g]) for g in unique_groups]
        ax.legend(handles, unique_groups, loc=1)
    if hlines is not None:
        for h in hlines:
            ax.axhline(h, linestyle="--", color="#787878")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return ax

def abbreviate(s, max_len=20):
    ### Rules: split words, recursively abbreviate longest words to first 3 letters + . until length is satisfied
    ### If all words are abbreviated and length still not satisfied, replace last three characters with ellipses
    s_words = s.split(" ")
    words_remain = True
    while words_remain and len(" ".join(s_words)) > max_len:
        longest_word = sorted(s_words, key=lambda x: -len(x))[0]
        if len(longest_word) <= 4:
            words_remain = False
        else:
            new_s_words = [word if word != longest_word else longest_word[:3] + "." for word in s_words]
            s_words = new_s_words
    new_s = " ".join(s_words)
    if len(new_s) > max_len:
        new_s = new_s[:max_len-3] + "..."
    return new_s

### Get arguments

parser = argparse.ArgumentParser()
parser.add_argument("--geneset_path", help="")
parser.add_argument("--geneset_sim_path", help="")
parser.add_argument("--usage_path", help="")
parser.add_argument("--component_path", help="")
parser.add_argument("--input_matrix_path", help="")
parser.add_argument("--magma_enrichment_path", help="")
parser.add_argument("--epigenetic_enrichment_path", help="")
parser.add_argument("--genes_of_interest_paths", nargs="*", default=[], help="")
parser.add_argument("--genes_of_interest_set_names", nargs="*", default=[], help="")
parser.add_argument("--out_prefix", help="")
parser.add_argument("--gene_annot_path", help="")
parser.add_argument("--feature_meta_path", help="")
parser.add_argument("--trait_meta_path", help="")
parser.add_argument("--epigenetic_meta_path", help="")
parser.add_argument("--sim_thresh", type=float, default=0.7, help="")
parser.add_argument("--bar_nlog10_ylim", type=float, default=50, help="")
args = parser.parse_args()
config_dict = vars(args)

### Generate tables and PDF

gene_annot_df = pd.read_csv(config_dict["gene_annot_path"], sep="\t", index_col=0)
feature_meta_df = pd.read_csv(config_dict["feature_meta_path"], sep="\t", index_col=0)
trait_meta_df = pd.read_csv(config_dict["trait_meta_path"], sep="\t")
trait_meta_df = trait_meta_df.loc[:,["id", "meta_id", "description", "category"]]
trait_meta_df = trait_meta_df.set_index("id")
epigenetic_meta_df = pd.read_csv(config_dict["epigenetic_meta_path"], sep="\t")
epigenetic_meta_df = epigenetic_meta_df.loc[:,["file_accession", "biosample_id", "biosample_name"]]
epigenetic_meta_df = epigenetic_meta_df.set_index("file_accession")

trait_meta_id_to_desc = {}
for meta_id in trait_meta_df.meta_id.unique():
    desc = trait_meta_df[trait_meta_df.meta_id == meta_id].description
    cat = trait_meta_df[trait_meta_df.meta_id == meta_id].category
    assert len(set(desc)) == 1, "Meta ID to description mapping not 1 to 1"
    assert len(set(cat)) == 1, "Meta ID to category mapping not 1 to 1"
    trait_meta_id_to_desc[meta_id] = desc[0]
epi_bio_id_to_desc = {}
for bio_id in epigenetic_meta_df.biosample_id.unique():
    desc = epigenetic_meta_df[epigenetic_meta_df.biosample_id == bio_id].biosample_name
    assert len(set(desc)) == 1, "Biosample ID to name mapping not 1 to 1"
    epi_bio_id_to_desc[bio_id] = desc[0]

magma_enrich_df = pd.read_csv(config_dict["magma_enrichment_path"], sep="\t", header=[0,1])
magma_enrich_df = magma_enrich_df.set_index(("statistic","trait"))
magma_enrich_df.index.name = "trait"
epi_enrich_df = pd.read_csv(config_dict["epigenetic_enrichment_path"], sep="\t", header=[0,1])
epi_enrich_df = epi_enrich_df.set_index(("statistic","trait"))
epi_enrich_df.index.name = "trait"
geneset_df = pd.read_csv(config_dict["geneset_path"], sep="\t", index_col=0)
sim_df = pd.read_csv(config_dict["geneset_sim_path"], sep="\t", index_col=0)
usage_df = pd.read_csv(config_dict["usage_path"], sep="\t", index_col=0)
component_df = pd.read_csv(config_dict["component_path"], sep="\t", index_col=0)
input_matrix = load_df_from_npz(config_dict["input_matrix_path"])

### Extract p-values
magma_pval_df = magma_enrich_df.pvalue
epi_pval_df = epi_enrich_df.pvalue
### Get Bonferroni significant
magma_pval_bon_df = (magma_pval_df < 0.05 / (~pd.isnull(magma_pval_df)).sum(axis=0))
epi_pval_bon_df = (epi_pval_df < 0.05 / (~pd.isnull(epi_pval_df)).sum(axis=0))
### Extract some quantitative score for ranking purposes
magma_scores_df = magma_enrich_df.coef / magma_enrich_df.se
epi_scores_df = epi_enrich_df.odds_ratio
### Construct negative log10 p-value dataframe
magma_nlog10_pval_df = -np.log10(magma_pval_df)
epi_nlog10_pval_df = -np.log10(epi_pval_df)
### Prep the negative log10 p-value data for ranking/visualization purposes
prep_magma_nlog10_pval_df, magma_nlog10_pval_max_val = prep_nlog_pval(magma_nlog10_pval_df, magma_scores_df, config_dict["bar_nlog10_ylim"] + 1)
prep_epi_nlog10_pval_df, epi_nlog10_pval_max_val = prep_nlog_pval(epi_nlog10_pval_df, epi_scores_df, config_dict["bar_nlog10_ylim"] + 1)

### Format component summary table

component_summary = magma_pval_bon_df.T.loc[:,[]].copy()
component_summary.columns.name = None
component_summary.index.name = "component_id"

cluster_df = cluster_components(sim_df, config_dict["sim_thresh"])

for c in component_summary.index:
    ### Add cluster id
    component_summary.loc[c,"cluster_id"] = cluster_df.loc[c].cluster_id
    ### Add gene set size
    component_summary.loc[c,"geneset_size"] = geneset_df.loc[:,c].sum()
    ### Add top genes
    gs_genes = geneset_df.index[geneset_df.loc[:,c] == 1]
    top_gs_genes = usage_df.loc[gs_genes,c].sort_values(ascending=False).index.values[:5]
    top_gs_gene_names = gene_annot_df.loc[top_gs_genes].NAME.values
    component_summary.loc[c,"top_genes"] = " | ".join(top_gs_gene_names)
    ### Add top features
    top_features = component_df.loc[c].sort_values(ascending=False).index.values[:5]
    top_features_names = feature_meta_df.loc[top_features].Long_Name.values
    component_summary.loc[c,"top_features"] = " | ".join(top_features_names)
    ### Add phewas hits
    phewas_hits = magma_pval_bon_df.index[magma_pval_bon_df.loc[:,c]]
    sorted_phewas_hits = prep_magma_nlog10_pval_df.loc[phewas_hits,c].sort_values(ascending=False).index.values
    top_phewas_hit_meta_ids = trait_meta_df.loc[sorted_phewas_hits].meta_id.unique()[:5]
    top_phewas_hit_names = [trait_meta_id_to_desc[i] for i in top_phewas_hit_meta_ids]
    component_summary.loc[c,"num_phewas_bon"] = len(phewas_hits)
    component_summary.loc[c,"top_phewas"] = " | ".join(top_phewas_hit_names)
    ### Add epigenetic hits
    epi_hits = epi_pval_bon_df.index[epi_pval_bon_df.loc[:,c]]
    sorted_epi_hits = prep_epi_nlog10_pval_df.loc[epi_hits,c].sort_values(ascending=False).index.values
    top_epi_hit_bio_ids = epigenetic_meta_df.loc[sorted_epi_hits].biosample_id.unique()[:5]
    top_epi_hit_names = [epi_bio_id_to_desc[i] for i in top_epi_hit_bio_ids]
    component_summary.loc[c,"num_epi_bon"] = len(epi_hits)
    component_summary.loc[c,"top_epi"] = " | ".join(top_epi_hit_names)

component_summary["geneset_size"] = component_summary["geneset_size"].astype(int)
component_summary["cluster_id"] = component_summary["cluster_id"].astype(int)
component_summary["num_phewas_bon"] = component_summary["num_phewas_bon"].astype(int)
component_summary["num_epi_bon"] = component_summary["num_epi_bon"].astype(int)
component_summary["num_total_bon"] = component_summary["num_phewas_bon"] + component_summary["num_epi_bon"]

reorder_inds = group_sort(component_summary.cluster_id.values, -component_summary.num_total_bon.values)
component_summary = component_summary.iloc[reorder_inds]

### Format phewas summary table

phewas_summary = trait_meta_df.groupby("meta_id").description.first().to_frame()

for p in phewas_summary.index:
    ### Add metadata
    phewas_summary.loc[p,"category"] = trait_meta_df[trait_meta_df.meta_id == p].category[0]
    traits_in_meta = trait_meta_df[trait_meta_df.meta_id == p].index.values
    phewas_summary.loc[p,"num_traits"] = len(traits_in_meta)
    ### Summarize significant components
    components_bon_any = magma_pval_bon_df.columns[magma_pval_bon_df.loc[traits_in_meta].any(axis=0)]
    phewas_summary.loc[p,"num_components_bon"] = len(components_bon_any)
    phewas_summary.loc[p,"eff_num_components_bon"] = component_summary.loc[components_bon_any].cluster_id.unique().shape[0]
    phewas_summary.loc[p,"best_pval"] = magma_pval_df.loc[traits_in_meta].min().min()
    top_components = prep_magma_nlog10_pval_df.loc[traits_in_meta].max().loc[components_bon_any].sort_values(ascending=False).index[:5]
    phewas_summary.loc[p,"top_components"] = " | ".join(top_components)    

phewas_summary["num_traits"] = phewas_summary["num_traits"].astype(int)
phewas_summary["num_components_bon"] = phewas_summary["num_components_bon"].astype(int)
phewas_summary["eff_num_components_bon"] = phewas_summary["eff_num_components_bon"].astype(int)

phewas_summary = phewas_summary.sort_values("best_pval", ascending=True, kind="stable")
phewas_summary = phewas_summary.sort_values("eff_num_components_bon", ascending=False, kind="stable")

### Format epigenetic summary table

epi_summary = epigenetic_meta_df.groupby("biosample_id").biosample_name.first().to_frame()

for p in epi_summary.index:
    ### Add metadata
    exp_in_meta = epigenetic_meta_df[epigenetic_meta_df.biosample_id == p].index.values
    epi_summary.loc[p,"num_experiments"] = len(exp_in_meta)
    ### Summarize significant components
    components_bon_any = epi_pval_bon_df.columns[epi_pval_bon_df.loc[exp_in_meta].any(axis=0)]
    epi_summary.loc[p,"num_components_bon"] = len(components_bon_any)
    epi_summary.loc[p,"eff_num_components_bon"] = component_summary.loc[components_bon_any].cluster_id.unique().shape[0]
    epi_summary.loc[p,"best_pval"] = epi_pval_df.loc[exp_in_meta].min().min()
    top_components = prep_epi_nlog10_pval_df.loc[exp_in_meta].max().loc[components_bon_any].sort_values(ascending=False).index[:5]
    epi_summary.loc[p,"top_components"] = " | ".join(top_components)

epi_summary["num_experiments"] = epi_summary["num_experiments"].astype(int)
epi_summary["num_components_bon"] = epi_summary["num_components_bon"].astype(int)
epi_summary["eff_num_components_bon"] = epi_summary["eff_num_components_bon"].astype(int)

epi_summary = epi_summary.sort_values("best_pval", ascending=True, kind="stable")
epi_summary = epi_summary.sort_values("eff_num_components_bon", ascending=False, kind="stable")

### If we have gene sets of interest, summarize those

assert len(config_dict["genes_of_interest_paths"]) == len(config_dict["genes_of_interest_set_names"]), "Must be one name per path"

if len(config_dict["genes_of_interest_paths"]) != 0:
    ### Format component overlap with genes of interest table
    comp_goi_summary = component_summary.loc[:,["cluster_id", "num_phewas_bon", "num_epi_bon", "num_total_bon"]]
    for name, path in zip(config_dict["genes_of_interest_set_names"], config_dict["genes_of_interest_paths"]):
        genes_of_interest = np.loadtxt(path, dtype=str).flatten()
        for c in comp_goi_summary.index:
            overlap_goi = list(set(genes_of_interest).intersection(geneset_df.index[geneset_df.loc[:,c] == 1]))
            overlap_goi_names = gene_annot_df.loc[usage_df.loc[overlap_goi,c].sort_values(ascending=False).index].NAME.values
            comp_goi_summary.loc[c,name] = " | ".join(overlap_goi_names)
    ### Format gene of interest nearest neighbor table
    all_genes_of_interest = set()
    goi_mapper = {}
    for name, path in zip(config_dict["genes_of_interest_set_names"], config_dict["genes_of_interest_paths"]):
        genes_of_interest = np.loadtxt(path, dtype=str).flatten()
        all_genes_of_interest = all_genes_of_interest.union(genes_of_interest)
        goi_mapper[name] = genes_of_interest
    all_genes_of_interest = list(all_genes_of_interest)
    goi_nn_summary = pd.DataFrame(index=all_genes_of_interest).merge(gene_annot_df.loc[:,["NAME"]],
                                                                 how="left", left_index=True, right_index=True)
    goi_nn_summary.index.name = "ENSGID"
    for name in config_dict["genes_of_interest_set_names"]:
        goi_nn_summary[name] = goi_nn_summary.index.isin(goi_mapper[name])
    subset_cosine = input_matrix.loc[goi_nn_summary.index].dot(input_matrix.loc[goi_nn_summary.index].T)
    subset_cosine = pd.DataFrame(subset_cosine.values / np.sqrt(np.diagonal(subset_cosine.values) * np.diagonal(subset_cosine.values).reshape(-1,1)),
                                 index = subset_cosine.index, columns=subset_cosine.columns)
    for g in goi_nn_summary.index:
        g_nn = subset_cosine.loc[g].sort_values(ascending=False).index.values[1:1+5]
        g_nn_names = gene_annot_df.loc[g_nn].NAME.values
        goi_nn_summary.loc[g,"nearest_neighbors"] = " | ".join(g_nn_names)
    goi_nn_summary = goi_nn_summary.sort_values("NAME")

### Write out summary tables

component_summary.to_csv(config_dict["out_prefix"] + ".comp_summary.tsv", sep="\t")
phewas_summary.to_csv(config_dict["out_prefix"] + ".phewas_summary.tsv", sep="\t")
epi_summary.to_csv(config_dict["out_prefix"] + ".epi_summary.tsv", sep="\t")
if len(config_dict["genes_of_interest_paths"]) != 0:
    comp_goi_summary.to_csv(config_dict["out_prefix"] + ".comp_goi_summary.tsv", sep="\t")
    goi_nn_summary.to_csv(config_dict["out_prefix"] + ".goi_nn_summary.tsv", sep="\t")

### Reformat

component_summary = component_summary.reset_index()
component_summary = component_summary.loc[:,["component_id", "cluster_id", "geneset_size",
                                             "num_phewas_bon", "num_epi_bon", "num_total_bon",
                                             "top_genes", "top_features", "top_phewas", "top_epi"]]
phewas_summary = phewas_summary.reset_index()
epi_summary = epi_summary.reset_index()

if len(config_dict["genes_of_interest_paths"]) != 0:
    comp_goi_summary = comp_goi_summary.reset_index()
    goi_nn_summary = goi_nn_summary.reset_index()
    goi_nn_summary = goi_nn_summary.drop("ENSGID", axis=1)
    num_neighbors = len(goi_nn_summary.iloc[0]["nearest_neighbors"].split(" | "))
    for n in range(num_neighbors):
        goi_nn_summary["neighbor_{}".format(n+1)] = (goi_nn_summary["nearest_neighbors"]
                                                     .apply(lambda x: x.split(" | ")[n] if n < len(x.split(" | ")) else np.nan))
    goi_nn_summary = goi_nn_summary.drop("nearest_neighbors", axis=1)
    for name in config_dict["genes_of_interest_set_names"]:
        goi_nn_summary.loc[:,name] = goi_nn_summary.loc[:,name].astype(str)

### Generate formatted Excel spreadsheets

workbook = xlsxwriter.Workbook(config_dict["out_prefix"] + ".xlsx")
comp_summary_worksheet = workbook.add_worksheet("Component summary")
phewas_summary_worksheet = workbook.add_worksheet("PheWAS summary")
epi_summary_worksheet = workbook.add_worksheet("Epigenetic summary")
if len(config_dict["genes_of_interest_paths"]) != 0:
    comp_goi_summary_worksheet = workbook.add_worksheet("Component GOI summary")
    goi_nn_summary_worksheet = workbook.add_worksheet("GOI nearest neighbors")

header_format_opt = {"bg_color":"#3688BF",
                     "font_color":"#ffffff",
                     "border":5,
                     "bold":True}
main_cell_format_opt = {"bg_color":"#ffffff",
                        "right":3}

### Component summary to Excel

longest_string_per_col = defaultdict(lambda: 0)

for col_num in range(component_summary.shape[1]):
    cell_fmt = workbook.add_format(header_format_opt)
    comp_summary_worksheet.write(0, col_num, component_summary.columns[col_num], cell_fmt)
    longest_string_per_col[col_num] = max(longest_string_per_col[col_num], len(component_summary.columns[col_num]))

current_row = 1
alt_state = 0
for i in range(component_summary.shape[0]):
    if i > 0 and component_summary.iloc[i].cluster_id != component_summary.iloc[i-1].cluster_id:
        alt_state = 1 - alt_state
    if alt_state == 0:
        main_cell_format_opt["bg_color"] = "#ffffff"
    else:
        main_cell_format_opt["bg_color"] = "#cfcfcf"
    row = component_summary.iloc[i]
    top_genes = row.loc["top_genes"].split(" | ")
    top_features = row.loc["top_features"].split(" | ")
    top_phewas = row.loc["top_phewas"].split(" | ")
    top_epi = row.loc["top_epi"].split(" | ")
    assert len(top_genes) == len(top_features), "Gene/feature length mismatch"
    assert len(top_phewas) <= len(top_genes), "More phewas than genes/features"
    assert len(top_epi) <= len(top_genes), "More epi than genes/features"
    N = len(top_genes)
    row_expand = pd.DataFrame(index=np.arange(N), columns=row.index.values)
    row_expand.iloc[0] = row
    for j in range(N):
        if j < len(top_genes):
            row_expand.iloc[j].loc["top_genes"] = top_genes[j]
        if j < len(top_features):
            row_expand.iloc[j].loc["top_features"] = top_features[j]
        if j < len(top_phewas):
            row_expand.iloc[j].loc["top_phewas"] = top_phewas[j]
        if j < len(top_epi):
            row_expand.iloc[j].loc["top_epi"] = top_epi[j]
    for j in range(row_expand.shape[0]):
        if j < row_expand.shape[0] - 1:
            main_cell_format_opt["bottom"] = 0
        else:
            main_cell_format_opt["bottom"] = 3
        for col_num in range(row_expand.shape[1]):
            if col_num == 0:
                main_cell_format_opt["bold"] = True
            else:
                main_cell_format_opt["bold"] = False
            cell_fmt = workbook.add_format(main_cell_format_opt)
            if pd.isnull(row_expand.iloc[j,col_num]):
                comp_summary_worksheet.write(current_row, col_num, "", cell_fmt)
            else:
                el = row_expand.iloc[j,col_num]
                comp_summary_worksheet.write(current_row, col_num, el, cell_fmt)
                longest_string_per_col[col_num] = max(longest_string_per_col[col_num], len(str(el)))
        current_row += 1

for i in longest_string_per_col.keys():
    comp_summary_worksheet.set_column(i, i, longest_string_per_col[i])

comp_summary_worksheet.freeze_panes(1, 1)

### Phewas summary to Excel

header_format_opt = {"bg_color":"#3688BF",
                     "font_color":"#ffffff",
                     "border":5,
                     "bold":True}
main_cell_format_opt = {"bg_color":"#ffffff",
                        "right":3,
                        "bottom":1,
                        "bottom_color":"#dedede"}

longest_string_per_col = defaultdict(lambda: 0)

for col_num in range(phewas_summary.shape[1]):
    cell_fmt = workbook.add_format(header_format_opt)
    phewas_summary_worksheet.write(0, col_num, phewas_summary.columns[col_num], cell_fmt)
    longest_string_per_col[col_num] = max(longest_string_per_col[col_num], len(phewas_summary.columns[col_num]))

current_row = 1
for i in range(phewas_summary.shape[0]):
    row = phewas_summary.iloc[i]
    for col_num in range(row.shape[0]):
        if col_num == 0:
            main_cell_format_opt["bold"] = True
        else:
            main_cell_format_opt["bold"] = False
        cell_fmt = workbook.add_format(main_cell_format_opt)
        if pd.isnull(row.iloc[col_num]):
            phewas_summary_worksheet.write(current_row, col_num, "", cell_fmt)
        else:
            el = row.iloc[col_num]
            phewas_summary_worksheet.write(current_row, col_num, el, cell_fmt)
            longest_string_per_col[col_num] = max(longest_string_per_col[col_num], len(str(el)))
    current_row += 1

for i in longest_string_per_col.keys():
    phewas_summary_worksheet.set_column(i, i, longest_string_per_col[i])

phewas_summary_worksheet.freeze_panes(1, 2)

### Epigenetic summary to Excel

header_format_opt = {"bg_color":"#3688BF",
                     "font_color":"#ffffff",
                     "border":5,
                     "bold":True}
main_cell_format_opt = {"bg_color":"#ffffff",
                        "right":3,
                        "bottom":1,
                        "bottom_color":"#dedede"}

longest_string_per_col = defaultdict(lambda: 0)

for col_num in range(epi_summary.shape[1]):
    cell_fmt = workbook.add_format(header_format_opt)
    epi_summary_worksheet.write(0, col_num, epi_summary.columns[col_num], cell_fmt)
    longest_string_per_col[col_num] = max(longest_string_per_col[col_num], len(epi_summary.columns[col_num]))

current_row = 1
for i in range(epi_summary.shape[0]):
    row = epi_summary.iloc[i]
    for col_num in range(row.shape[0]):
        if col_num == 0:
            main_cell_format_opt["bold"] = True
        else:
            main_cell_format_opt["bold"] = False
        cell_fmt = workbook.add_format(main_cell_format_opt)
        if pd.isnull(row.iloc[col_num]):
            epi_summary_worksheet.write(current_row, col_num, "", cell_fmt)
        else:
            el = row.iloc[col_num]
            epi_summary_worksheet.write(current_row, col_num, el, cell_fmt)
            longest_string_per_col[col_num] = max(longest_string_per_col[col_num], len(str(el)))
    current_row += 1

for i in longest_string_per_col.keys():
    epi_summary_worksheet.set_column(i, i, longest_string_per_col[i])

epi_summary_worksheet.freeze_panes(1, 2)

### Component GOI summary to Excel (if applicable)

if len(config_dict["genes_of_interest_paths"]) != 0:
    header_format_opt = {"bg_color":"#3688BF",
                     "font_color":"#ffffff",
                     "border":5,
                     "bold":True}
    main_cell_format_opt = {"bg_color":"#ffffff",
                            "right":3,
                            "bottom":1,
                            "bottom_color":"#dedede",
                            "valign":"top"}
    longest_string_per_col = defaultdict(lambda: 0)
    
    for col_num in range(comp_goi_summary.shape[1]):
        cell_fmt = workbook.add_format(header_format_opt)
        comp_goi_summary_worksheet.write(0, col_num, comp_goi_summary.columns[col_num], cell_fmt)
        longest_string_per_col[col_num] = max(longest_string_per_col[col_num], len(comp_goi_summary.columns[col_num]))

    current_row = 1
    alt_state = 0
    for i in range(comp_goi_summary.shape[0]):
        if i > 0 and component_summary.iloc[i].cluster_id != component_summary.iloc[i-1].cluster_id:
            alt_state = 1 - alt_state
        if alt_state == 0:
            main_cell_format_opt["bg_color"] = "#ffffff"
        else:
            main_cell_format_opt["bg_color"] = "#cfcfcf"
        row = comp_goi_summary.iloc[i].copy()
        for name in config_dict["genes_of_interest_set_names"]:
            row.loc[name] = row.loc[name].replace(" | ", ", ")
        for col_num in range(row.shape[0]):
            if col_num == 0:
                main_cell_format_opt["bold"] = True
            else:
                main_cell_format_opt["bold"] = False
            if row.index[col_num] in config_dict["genes_of_interest_set_names"]:
                main_cell_format_opt["text_wrap"] = True
            else:
                main_cell_format_opt["text_wrap"] = False
            cell_fmt = workbook.add_format(main_cell_format_opt)
            if pd.isnull(row.iloc[col_num]):
                comp_goi_summary_worksheet.write(current_row, col_num, "", cell_fmt)
            else:
                el = row.iloc[col_num]
                comp_goi_summary_worksheet.write(current_row, col_num, el, cell_fmt)
                longest_string_per_col[col_num] = max(longest_string_per_col[col_num], len(str(el)))
        current_row += 1

    for i in longest_string_per_col.keys():
        comp_goi_summary_worksheet.set_column(i, i, min(longest_string_per_col[i], 80))

    comp_goi_summary_worksheet.freeze_panes(1, 1)

### GOI NN summary to Excel (if applicable)

if len(config_dict["genes_of_interest_paths"]) != 0:
    header_format_opt = {"bg_color":"#3688BF",
                     "font_color":"#ffffff",
                     "border":5,
                     "bold":True}
    main_cell_format_opt = {"bg_color":"#ffffff",
                            "right":3,
                            "bottom":1,
                            "bottom_color":"#dedede"}
    longest_string_per_col = defaultdict(lambda: 0)
    
    for col_num in range(goi_nn_summary.shape[1]):
        cell_fmt = workbook.add_format(header_format_opt)
        goi_nn_summary_worksheet.write(0, col_num, goi_nn_summary.columns[col_num], cell_fmt)
        longest_string_per_col[col_num] = max(longest_string_per_col[col_num], len(goi_nn_summary.columns[col_num]))

    current_row = 1
    for i in range(goi_nn_summary.shape[0]):
        row = goi_nn_summary.iloc[i]
        for col_num in range(row.shape[0]):
            if col_num == 0:
                main_cell_format_opt["bold"] = True
            else:
                main_cell_format_opt["bold"] = False
            cell_fmt = workbook.add_format(main_cell_format_opt)
            if pd.isnull(row.iloc[col_num]):
                goi_nn_summary_worksheet.write(current_row, col_num, "", cell_fmt)
            else:
                el = row.iloc[col_num]
                goi_nn_summary_worksheet.write(current_row, col_num, el, cell_fmt)
                longest_string_per_col[col_num] = max(longest_string_per_col[col_num], len(str(el)))
        current_row += 1

    for i in longest_string_per_col.keys():
        goi_nn_summary_worksheet.set_column(i, i, longest_string_per_col[i])

    goi_nn_summary_worksheet.freeze_panes(1, 1)

workbook.close()

### Generate per-component plots

color_list = ["#EBA6A9", "#C6A15B", "#D3DFB8", "#574D68", "#C1D37F",
              "#3066BE", "#D2D4C8", "#F49F0A", "#00A6A6", "#BBDEF0",
              "#0A8754", "#B07C9E", "#C3BEF7", "#637074", "#E9C46A",
              "#F4A261", "#3A2D32", "#FFE3DC", "#A2AD91", "#D56062"]
cat_color_dict = {}
for i, cat in enumerate(trait_meta_df.category.unique()):
    cat_color_dict[cat] = color_list[i]

### Make directory for per-component plots
Path(config_dict["out_prefix"] + "_raw_fig_dir/").mkdir(parents=True, exist_ok=True)

for comp in magma_pval_df.columns:
    marg_nlog10_pval = -np.log10(0.05)
    bon_nlog10_pval = -np.log10(0.05 / (~magma_pval_df.loc[:,comp].isnull()).sum())
    plotting_df = prep_magma_nlog10_pval_df.loc[:,comp].to_frame().rename(columns={comp:"nlog10_pval"})
    plotting_df = plotting_df.merge(trait_meta_df.loc[:,["meta_id", "category"]],
                                    how="left", left_index=True, right_index=True)
    grouped_plotting_df = []
    for mi in plotting_df.meta_id.unique():
        sub_plotting_df = plotting_df[plotting_df.meta_id == mi]
        grouped_plotting_df.append(sub_plotting_df.iloc[sub_plotting_df.nlog10_pval.argmax()])
    grouped_plotting_df = pd.DataFrame(grouped_plotting_df).reset_index().drop("index", axis=1).set_index("meta_id")
    grouped_plotting_df = grouped_plotting_df.sort_values("nlog10_pval", ascending=False).iloc[:60]
    sorted_grouped_plotting_df = []
    for cat in grouped_plotting_df.category.unique():
        sorted_grouped_plotting_df.append(grouped_plotting_df[grouped_plotting_df.category == cat])
    sorted_grouped_plotting_df = pd.concat(sorted_grouped_plotting_df, axis=0)
    bar_chart_general([abbreviate(l.replace("_", " "), max_len=30) for l in sorted_grouped_plotting_df.index.values],
                      np.maximum(sorted_grouped_plotting_df.nlog10_pval.values, 0),
                      config_dict["bar_nlog10_ylim"],
                      ylabel="-log10 p-value",
                      hlines=[marg_nlog10_pval,bon_nlog10_pval],
                      groups=sorted_grouped_plotting_df.category.values,
                      group_color_dict=cat_color_dict,
                      legend_mask=4)
    plt.savefig(config_dict["out_prefix"] + "_raw_fig_dir/" + "{}.magma.jpeg".format(comp),
                dpi=300, bbox_inches="tight")
    plt.close()

for comp in epi_pval_df.columns:
    marg_nlog10_pval = -np.log10(0.05)
    bon_nlog10_pval = -np.log10(0.05 / (~epi_pval_df.loc[:,comp].isnull()).sum())

    plotting_df = prep_epi_nlog10_pval_df.loc[:,comp].to_frame().rename(columns={comp:"nlog10_pval"})
    plotting_df = plotting_df.merge(epigenetic_meta_df, how="left", left_index=True, right_index=True)
    grouped_plotting_df = []
    for bi in plotting_df.biosample_id.unique():
        sub_plotting_df = plotting_df[plotting_df.biosample_id == bi]
        grouped_plotting_df.append(sub_plotting_df.iloc[sub_plotting_df.nlog10_pval.argmax()])
    grouped_plotting_df = pd.DataFrame(grouped_plotting_df).reset_index().drop("index", axis=1).set_index("biosample_id")
    grouped_plotting_df = grouped_plotting_df.sort_values("nlog10_pval", ascending=False).iloc[:60]
    bar_chart_general([abbreviate(l, max_len=30) for l in grouped_plotting_df.biosample_name.values],
                      np.maximum(grouped_plotting_df.nlog10_pval.values, 0),
                      config_dict["bar_nlog10_ylim"],
                      ylabel="-log10 p-value",
                      hlines=[marg_nlog10_pval,bon_nlog10_pval])
    plt.savefig(config_dict["out_prefix"] + "_raw_fig_dir/" + "{}.epi.jpeg".format(comp),
                dpi=300, bbox_inches="tight")
    plt.close()

### Write PDF

a4_width = 210
a4_height = 297

pdf = FPDF('P', 'mm', 'A4')

for comp in geneset_df.columns:
    pdf.add_page()
    pdf.set_margins(left=10, top=10, right=10)
    pdf.set_xy(10, 10)
    pdf.set_font('arial', style="B", size=15.0)
    pdf.multi_cell(w=0,h=5,align="C",txt=comp,border=0)
    pdf.set_font('arial', size=3.0)
    pdf.multi_cell(w=0,h=3,align="C",txt="",border=0)
    
    magma_bar_path = config_dict["out_prefix"] + "_raw_fig_dir/" + "{}.magma.jpeg".format(comp)
    cover = Image.open(magma_bar_path)
    aspect_ratio = cover.size[0] / cover.size[1]
    pdf.image(magma_bar_path, w=a4_width - 20, h=(a4_width - 20) / aspect_ratio)

    epi_bar_path = config_dict["out_prefix"] + "_raw_fig_dir/" + "{}.epi.jpeg".format(comp)
    cover = Image.open(epi_bar_path)
    aspect_ratio = cover.size[0] / cover.size[1]
    pdf.image(epi_bar_path, w=a4_width - 20, h=(a4_width - 20) / aspect_ratio)
    
    top_features = component_df.loc[comp].sort_values(ascending=False).index.values[:10]
    top_features_names = feature_meta_df.loc[top_features].Long_Name.values
    top_features_text = "\n".join(top_features_names)
    pdf.set_font('arial', size=3.0)
    pdf.multi_cell(w=0,h=3,align="C",txt="",border=0)
    pdf.set_font('arial', style="B", size=6.0)
    pdf.multi_cell(w=0,h=5,align="L",txt="Top features:",border=0)
    pdf.set_font('arial', size=6.0)
    pdf.multi_cell(w=0,h=4,align="L",txt=top_features_text,border=0)
    
    gs_ensgid = geneset_df.index[geneset_df.loc[:,comp] == 1].values
    sorted_gs_ensgid = usage_df.loc[gs_ensgid,comp].sort_values(ascending=False).index.values
    sorted_gs_names = gene_annot_df.loc[sorted_gs_ensgid].NAME.values
    gs_text = ", ".join(sorted_gs_names)
    pdf.set_font('arial', size=3.0)
    pdf.multi_cell(w=0,h=3,align="C",txt="",border=0)
    pdf.set_font('arial', style="B", size=6.0)
    pdf.multi_cell(w=0,h=5,align="L",txt="Genes in component:",border=0)
    pdf.set_font('arial', size=6.0)
    pdf.multi_cell(w=0,h=4,align="L",txt=gs_text,border=0)
    
    if len(config_dict["genes_of_interest_paths"]) != 0:
        for name in config_dict["genes_of_interest_set_names"]:
            goi_gs = comp_goi_summary.set_index("component_id").loc[comp].loc[name]
            goi_gs_text = goi_gs.replace(" | ", ", ")
            pdf.set_font('arial', size=3.0)
            pdf.multi_cell(w=0,h=3,align="C",txt="",border=0)
            pdf.set_font('arial', style="B", size=6.0)
            pdf.multi_cell(w=0,h=5,align="L",txt="Overlap with {}:".format(name),border=0)
            pdf.set_font('arial', size=6.0)
            pdf.multi_cell(w=0,h=4,align="L",txt=goi_gs_text,border=0)

pdf.output(config_dict["out_prefix"] + '.pdf', 'F')


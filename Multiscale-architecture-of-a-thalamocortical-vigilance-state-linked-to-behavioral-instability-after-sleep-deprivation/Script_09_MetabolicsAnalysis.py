#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
metabolomics_gene_validation_analysis.py

Purpose
-------
Analyze paired serum metabolomics data for sleep deprivation and test whether
blood metabolic changes match TVS-related gene enrichment results.

Designed for your Excel format:
- Row 1: subject IDs, e.g., sub-001 ... sub-030
- Row 2: time labels, e.g., before / after
- Row 3: sample IDs
- Column 1: sample number
- Column 2: Component Name
- Column 3: Chinese metabolite name
- Data: metabolite concentration, usually nmol/L

Main outputs
------------
1. Paired before/after statistics for each metabolite
2. FDR-corrected significant metabolites
3. Gene-analysis-matching category summaries
4. Figures:
   - volcano plot
   - top changed metabolites bar plot
   - paired line plots
   - top log2FC heatmap
   - matching category plot
5. Optional: if you provide TVS/PVT file, it computes:
   - Δmetabolite ~ ΔTVS correlations
   - Δmetabolite ~ ΔPVT correlations
   - pathway/metabolic-index correlations
   - simple mediation screening

Recommended command
-------------------
python metabolomics_gene_validation_analysis.py \
    --input "BMSW20241212WAL1080-60 serum-定量结果表格.xlsx" \
    --outdir metabolomics_gene_validation_outputs

Optional with TVS/PVT:
python metabolomics_gene_validation_analysis.py \
    --input "BMSW20241212WAL1080-60 serum-定量结果表格.xlsx" \
    --tvs_csv tvs_pvt_values.csv \
    --outdir metabolomics_gene_validation_outputs

Optional TVS/PVT csv format
---------------------------
subject,tvs_before,tvs_after,pvt_before,pvt_after
sub-001,0.12,0.05,240,310
sub-002,...

You may also use:
subject,tvs_delta,pvt_delta

"""

import argparse
import re
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon, spearmanr, pearsonr, fisher_exact
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")


# ============================================================
# Utilities
# ============================================================

def ensure_outdir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def bh_fdr(pvals):
    pvals = pd.Series(pvals, dtype=float)
    out = np.full(len(pvals), np.nan)
    mask = pvals.notna().values
    if mask.sum() > 0:
        out[mask] = multipletests(pvals[mask].values, method="fdr_bh")[1]
    return out


def cohen_dz(diff):
    diff = pd.Series(diff, dtype=float).dropna()
    if len(diff) < 2:
        return np.nan
    sd = diff.std(ddof=1)
    if sd == 0:
        return np.nan
    return diff.mean() / sd


def clean_numeric(x):
    return pd.to_numeric(x, errors="coerce")


def safe_filename(s, max_len=80):
    s = str(s)
    s = re.sub(r"[^\w\-_\. ]", "_", s)
    return s[:max_len]


def get_pseudocount(values):
    """Half of the minimum positive value; fallback to 1e-9."""
    vals = pd.to_numeric(pd.Series(values).ravel(), errors="coerce")
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]
    if len(vals) == 0:
        return 1e-9
    return float(vals.min()) / 2.0


def classify_metabolite(name):
    """
    Lightweight keyword-based biological categories. This is not a replacement
    for HMDB/KEGG annotation, but it is useful for matching the gene analysis.
    """
    n = str(name).lower()

    # Important for your current gene-analysis match
    if "carnitine" in n:
        return "Acylcarnitine / fatty-acid oxidation"
    if any(k in n for k in [
        "sebacic", "suberic", "adipic", "pimelic", "azelaic",
        "dicarbox", "glutaric", "succinic", "methylmalonic",
        "ethylmalonic"
    ]):
        return "Dicarboxylic acid / fatty-acid oxidation"
    if any(k in n for k in ["cholesterol", "steroid", "bile", "cortisol", "cortisone"]):
        return "Cholesterol / steroid metabolism"
    if any(k in n for k in ["nicotinamide", "nad", "riboflavin", "flavin", "oxid", "glutathione"]):
        return "Oxidative / redox metabolism"

    # Other broad classes
    if any(k in n for k in [
        "lysine", "glutamine", "glutamate", "tryptophan", "tyrosine",
        "phenylalanine", "valine", "leucine", "isoleucine",
        "arginine", "ornithine", "citrulline", "alanine", "serine",
        "glycine", "histidine", "methionine", "threonine", "proline"
    ]):
        return "Amino acid metabolism"
    if any(k in n for k in ["guanosine", "uridine", "adenosine", "cytidine", "xanthine", "uric", "hypoxanthine"]):
        return "Nucleotide / nucleoside metabolism"
    if "acetyl" in n:
        return "Acetylated metabolite"
    if any(k in n for k in ["phosphatidyl", "sphingo", "ceramide", "lyso", "triglyceride", "diglyceride"]):
        return "Complex lipid metabolism"

    return "Other"


def gene_validation_match_category(category):
    """
    Collapse categories into whether they directly match the current AHBA/GO results:
    fatty acid metabolism, beta oxidation, biological oxidation, cholesterol/steroid,
    oxidative/redox processes.
    """
    if category in [
        "Acylcarnitine / fatty-acid oxidation",
        "Dicarboxylic acid / fatty-acid oxidation",
        "Cholesterol / steroid metabolism",
        "Oxidative / redox metabolism",
    ]:
        return "Matches TVS gene enrichment"
    return "Other metabolites"


# ============================================================
# Load and reshape Excel
# ============================================================

def load_serum_excel(input_xlsx, sheet_name=0):
    raw = pd.read_excel(input_xlsx, sheet_name=sheet_name, header=None)

    if raw.shape[0] < 4 or raw.shape[1] < 6:
        raise ValueError("The Excel sheet shape is too small. Please check the input file.")

    # Fixed structure from your file
    subject_row = raw.iloc[0, 3:]
    time_row = raw.iloc[1, 3:]
    sampleid_row = raw.iloc[2, 3:]

    metabolite_df = raw.iloc[3:, :].copy()
    metabolite_df = metabolite_df.rename(columns={
        0: "sample_number",
        1: "Component Name",
        2: "Chinese Name"
    })

    value_cols = list(range(3, raw.shape[1]))

    long_rows = []
    for col in value_cols:
        subject = str(subject_row[col]).strip()
        time = str(time_row[col]).strip().lower()
        sample_id = str(sampleid_row[col]).strip()

        if subject == "nan" or time == "nan":
            continue
        if time not in ["before", "after"]:
            # Allow flexible labels
            if "before" in time or "pre" in time:
                time = "before"
            elif "after" in time or "post" in time:
                time = "after"
            else:
                continue

        tmp = metabolite_df[["sample_number", "Component Name", "Chinese Name", col]].copy()
        tmp = tmp.rename(columns={col: "value"})
        tmp["subject"] = subject
        tmp["time"] = time
        tmp["sample_id"] = sample_id
        tmp["value"] = clean_numeric(tmp["value"])
        long_rows.append(tmp)

    long_df = pd.concat(long_rows, ignore_index=True)
    long_df["metabolite"] = long_df["Component Name"].astype(str)
    long_df["metabolite_cn"] = long_df["Chinese Name"].astype(str)
    long_df["category"] = long_df["metabolite"].apply(classify_metabolite)
    long_df["gene_validation_group"] = long_df["category"].apply(gene_validation_match_category)

    # Remove completely missing metabolite names
    long_df = long_df[long_df["metabolite"].notna()]
    long_df = long_df[long_df["metabolite"].str.lower() != "nan"]

    return raw, long_df


def make_paired_wide(long_df):
    """
    Return one row per subject-metabolite with before, after, delta, log2FC.
    """
    wide = long_df.pivot_table(
        index=["subject", "metabolite", "metabolite_cn", "category", "gene_validation_group"],
        columns="time",
        values="value",
        aggfunc="mean"
    ).reset_index()

    if "before" not in wide.columns or "after" not in wide.columns:
        raise ValueError("Could not find both before and after columns after reshaping.")

    pc = get_pseudocount(wide[["before", "after"]].values.ravel())
    wide["raw_delta"] = wide["after"] - wide["before"]
    wide["log2FC"] = np.log2((wide["after"] + pc) / (wide["before"] + pc))

    return wide, pc


# ============================================================
# Paired statistics
# ============================================================

def paired_metabolite_stats(paired_df):
    rows = []
    for metab, g in paired_df.groupby("metabolite"):
        before = clean_numeric(g["before"])
        after = clean_numeric(g["after"])
        mask = before.notna() & after.notna()
        before = before[mask]
        after = after[mask]
        diff = after - before

        category = g["category"].iloc[0]
        gene_group = g["gene_validation_group"].iloc[0]
        metab_cn = g["metabolite_cn"].iloc[0]

        n = int(mask.sum())
        if n >= 3:
            try:
                t_stat, t_p = ttest_rel(after, before, nan_policy="omit")
            except Exception:
                t_stat, t_p = np.nan, np.nan
            try:
                # Wilcoxon can fail if all differences are zero
                w_stat, w_p = wilcoxon(after, before, zero_method="wilcox", alternative="two-sided")
            except Exception:
                w_stat, w_p = np.nan, np.nan
        else:
            t_stat, t_p, w_stat, w_p = np.nan, np.nan, np.nan, np.nan

        rows.append({
            "metabolite": metab,
            "metabolite_cn": metab_cn,
            "category": category,
            "gene_validation_group": gene_group,
            "n_pairs": n,
            "mean_before": before.mean(),
            "mean_after": after.mean(),
            "median_before": before.median(),
            "median_after": after.median(),
            "mean_raw_delta": diff.mean(),
            "median_raw_delta": diff.median(),
            "mean_log2FC": g.loc[mask.values, "log2FC"].mean(),
            "median_log2FC": g.loc[mask.values, "log2FC"].median(),
            "cohen_dz": cohen_dz(diff),
            "paired_t_stat": t_stat,
            "paired_t_p": t_p,
            "wilcoxon_stat": w_stat,
            "wilcoxon_p": w_p,
        })

    res = pd.DataFrame(rows)
    res["paired_t_FDR"] = bh_fdr(res["paired_t_p"])
    res["wilcoxon_FDR"] = bh_fdr(res["wilcoxon_p"])
    res["significant_t_FDR05"] = res["paired_t_FDR"] < 0.05
    res["significant_wilcoxon_FDR05"] = res["wilcoxon_FDR"] < 0.05
    res["direction"] = np.where(res["mean_log2FC"] > 0, "after > before", "after < before")
    res = res.sort_values(["paired_t_FDR", "paired_t_p"], na_position="last").reset_index(drop=True)
    return res


def category_summary(stats_df):
    rows = []
    for cat, g in stats_df.groupby("category"):
        n_total = len(g)
        n_sig_t = int((g["paired_t_FDR"] < 0.05).sum())
        n_sig_w = int((g["wilcoxon_FDR"] < 0.05).sum())
        n_down_sig = int(((g["paired_t_FDR"] < 0.05) & (g["mean_log2FC"] < 0)).sum())
        n_up_sig = int(((g["paired_t_FDR"] < 0.05) & (g["mean_log2FC"] > 0)).sum())
        rows.append({
            "category": cat,
            "n_metabolites": n_total,
            "n_paired_t_FDR_lt_0.05": n_sig_t,
            "n_wilcoxon_FDR_lt_0.05": n_sig_w,
            "n_significant_down": n_down_sig,
            "n_significant_up": n_up_sig,
            "median_log2FC": g["mean_log2FC"].median(),
            "mean_log2FC": g["mean_log2FC"].mean(),
        })
    return pd.DataFrame(rows).sort_values("n_paired_t_FDR_lt_0.05", ascending=False)


def matching_enrichment(stats_df, fdr_col="paired_t_FDR", alpha=0.05):
    """
    Fisher exact test:
    Are gene-enrichment-matching metabolites overrepresented among significant metabolites?

    Robust version:
    Always forces a 2 x 2 table:
        rows:    match / other
        columns: significant / non-significant
    """
    df = stats_df.copy()

    df["is_sig"] = df[fdr_col] < alpha
    df["is_match"] = df["gene_validation_group"] == "Matches TVS gene enrichment"

    match_sig = int(((df["is_match"] == True) & (df["is_sig"] == True)).sum())
    match_nonsig = int(((df["is_match"] == True) & (df["is_sig"] == False)).sum())
    other_sig = int(((df["is_match"] == False) & (df["is_sig"] == True)).sum())
    other_nonsig = int(((df["is_match"] == False) & (df["is_sig"] == False)).sum())

    table = np.array([
        [match_sig, match_nonsig],
        [other_sig, other_nonsig]
    ], dtype=int)

    # Fisher exact test requires a real 2 x 2 table.
    # If one row or column is all zero, Fisher is not meaningful.
    if table.sum() == 0 or (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
        oddsratio, p = np.nan, np.nan
    else:
        oddsratio, p = fisher_exact(table)

    out = pd.DataFrame({
        "comparison": ["Gene-enrichment-matching metabolites enriched among significant metabolites"],
        "fdr_col": [fdr_col],
        "alpha": [alpha],
        "oddsratio": [oddsratio],
        "p_value": [p],
        "match_sig": [match_sig],
        "match_nonsig": [match_nonsig],
        "other_sig": [other_sig],
        "other_nonsig": [other_nonsig],
    })

    table_df = pd.DataFrame(
        table,
        index=["Matches TVS gene enrichment", "Other metabolites"],
        columns=["Significant", "Non-significant"]
    )

    return out, table_df


# ============================================================
# Optional TVS / PVT association
# ============================================================

def load_tvs_csv(tvs_csv):
    tvs = pd.read_csv(tvs_csv)
    if "subject" not in tvs.columns:
        raise ValueError("tvs_csv must contain a 'subject' column.")

    tvs["subject"] = tvs["subject"].astype(str)

    if {"tvs_before", "tvs_after"}.issubset(tvs.columns):
        tvs["tvs_delta"] = clean_numeric(tvs["tvs_after"]) - clean_numeric(tvs["tvs_before"])
    if {"pvt_before", "pvt_after"}.issubset(tvs.columns):
        tvs["pvt_delta"] = clean_numeric(tvs["pvt_after"]) - clean_numeric(tvs["pvt_before"])

    return tvs


def metabolite_tvs_correlations(paired_df, tvs_df, target_col="tvs_delta", method="spearman"):
    if target_col not in tvs_df.columns:
        return pd.DataFrame()

    merged = paired_df.merge(tvs_df[["subject", target_col]], on="subject", how="inner")
    rows = []
    for metab, g in merged.groupby("metabolite"):
        x = clean_numeric(g["log2FC"])
        y = clean_numeric(g[target_col])
        mask = x.notna() & y.notna()
        if mask.sum() >= 6:
            if method == "pearson":
                r, p = pearsonr(x[mask], y[mask])
            else:
                r, p = spearmanr(x[mask], y[mask])
        else:
            r, p = np.nan, np.nan

        rows.append({
            "metabolite": metab,
            "metabolite_cn": g["metabolite_cn"].iloc[0],
            "category": g["category"].iloc[0],
            "gene_validation_group": g["gene_validation_group"].iloc[0],
            "target": target_col,
            "method": method,
            "n": int(mask.sum()),
            "r": r,
            "p": p,
        })

    res = pd.DataFrame(rows)
    res["FDR"] = bh_fdr(res["p"])
    res = res.sort_values(["FDR", "p"], na_position="last").reset_index(drop=True)
    return res


def make_metabolic_indices(paired_df):
    """
    Subject-level metabolic indices:
    - mean log2FC of all acylcarnitines
    - mean log2FC of all fatty-acid oxidation categories
    - mean log2FC of gene-enrichment-matching metabolites
    """
    rows = []
    for subject, g in paired_df.groupby("subject"):
        def mean_for(mask):
            vals = g.loc[mask, "log2FC"]
            if vals.notna().sum() == 0:
                return np.nan
            return vals.mean()

        is_acyl = g["category"] == "Acylcarnitine / fatty-acid oxidation"
        is_fao = g["category"].isin([
            "Acylcarnitine / fatty-acid oxidation",
            "Dicarboxylic acid / fatty-acid oxidation",
        ])
        is_match = g["gene_validation_group"] == "Matches TVS gene enrichment"

        rows.append({
            "subject": subject,
            "acylcarnitine_index_mean_log2FC": mean_for(is_acyl),
            "fatty_acid_oxidation_index_mean_log2FC": mean_for(is_fao),
            "gene_matching_metabolism_index_mean_log2FC": mean_for(is_match),
        })
    return pd.DataFrame(rows)


def index_tvs_correlations(index_df, tvs_df):
    merged = index_df.merge(tvs_df, on="subject", how="inner")
    targets = [c for c in ["tvs_delta", "pvt_delta"] if c in merged.columns]
    index_cols = [c for c in index_df.columns if c != "subject"]
    rows = []
    for idx_col in index_cols:
        for target in targets:
            x = clean_numeric(merged[idx_col])
            y = clean_numeric(merged[target])
            mask = x.notna() & y.notna()
            if mask.sum() >= 6:
                r, p = spearmanr(x[mask], y[mask])
            else:
                r, p = np.nan, np.nan
            rows.append({
                "index": idx_col,
                "target": target,
                "n": int(mask.sum()),
                "spearman_r": r,
                "p": p,
            })
    res = pd.DataFrame(rows)
    res["FDR"] = bh_fdr(res["p"])
    return res


def simple_mediation_screen(index_df, tvs_df, x_col, m_col="tvs_delta", y_col="pvt_delta", n_boot=5000, seed=1234):
    """
    Simple bootstrap mediation screen:
    X = metabolic index
    M = TVS delta
    Y = PVT delta

    This is a lightweight exploratory function. For publication, consider using
    a formal package or robust regression/permutation tests.
    """
    if m_col not in tvs_df.columns or y_col not in tvs_df.columns:
        return pd.DataFrame()

    dat = index_df.merge(tvs_df[["subject", m_col, y_col]], on="subject", how="inner")
    dat = dat[["subject", x_col, m_col, y_col]].copy()
    for c in [x_col, m_col, y_col]:
        dat[c] = clean_numeric(dat[c])
    dat = dat.dropna()

    if dat.shape[0] < 10:
        return pd.DataFrame()

    x = dat[x_col].values
    m = dat[m_col].values
    y = dat[y_col].values

    # z-score for comparable coefficients
    x = (x - x.mean()) / x.std(ddof=1)
    m = (m - m.mean()) / m.std(ddof=1)
    y = (y - y.mean()) / y.std(ddof=1)

    # a path: M ~ X
    a = np.polyfit(x, m, 1)[0]
    # b path: Y ~ M + X
    Xmat = np.column_stack([np.ones(len(x)), m, x])
    coef = np.linalg.lstsq(Xmat, y, rcond=None)[0]
    b = coef[1]
    c_prime = coef[2]
    indirect = a * b

    rng = np.random.default_rng(seed)
    boots = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        xb, mb, yb = x[idx], m[idx], y[idx]
        try:
            ab = np.polyfit(xb, mb, 1)[0]
            Xb = np.column_stack([np.ones(len(xb)), mb, xb])
            coefb = np.linalg.lstsq(Xb, yb, rcond=None)[0]
            bb = coefb[1]
            boots.append(ab * bb)
        except Exception:
            pass

    boots = np.array(boots)
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
    p_boot = 2 * min((boots <= 0).mean(), (boots >= 0).mean())

    return pd.DataFrame({
        "x": [x_col],
        "m": [m_col],
        "y": [y_col],
        "n": [n],
        "a_path": [a],
        "b_path": [b],
        "c_prime": [c_prime],
        "indirect_ab": [indirect],
        "boot_ci_low": [ci_low],
        "boot_ci_high": [ci_high],
        "boot_p_approx": [p_boot],
    })


# ============================================================
# Figures
# ============================================================

def plot_volcano(stats_df, outdir, fdr_col="paired_t_FDR"):
    df = stats_df.copy()
    df["neglog10_FDR"] = -np.log10(df[fdr_col].replace(0, np.nan))
    df["is_sig"] = df[fdr_col] < 0.05

    fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=300)
    ax.scatter(df["mean_log2FC"], df["neglog10_FDR"], s=20, alpha=0.65)
    sig = df[df["is_sig"]]
    if not sig.empty:
        ax.scatter(sig["mean_log2FC"], sig["neglog10_FDR"], s=28, alpha=0.9)

    ax.axhline(-np.log10(0.05), linewidth=1, linestyle="--")
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Mean log2FC after / before")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title("Sleep-deprivation-induced serum metabolic changes")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Label top 10 by FDR
    lab = df.sort_values(fdr_col).head(10)
    for _, row in lab.iterrows():
        ax.text(row["mean_log2FC"], row["neglog10_FDR"], row["metabolite"], fontsize=7)

    fig.tight_layout()
    fig.savefig(outdir / "Fig_Metabolomics_A_volcano.png", dpi=300)
    fig.savefig(outdir / "Fig_Metabolomics_A_volcano.svg")
    plt.close(fig)


def plot_top_changed_bar(stats_df, outdir, top_n=20):
    df = stats_df.copy()
    df = df.sort_values("paired_t_FDR").head(top_n)
    df = df.sort_values("mean_log2FC")

    fig_h = max(4.8, 0.26 * len(df))
    fig, ax = plt.subplots(figsize=(7.2, fig_h), dpi=300)
    ax.barh(df["metabolite"], df["mean_log2FC"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Mean log2FC after / before")
    ax.set_title(f"Top {top_n} changed metabolites after sleep deprivation")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / "Fig_Metabolomics_B_top_changed_bar.png", dpi=300)
    fig.savefig(outdir / "Fig_Metabolomics_B_top_changed_bar.svg")
    plt.close(fig)


def plot_paired_lines(paired_df, stats_df, outdir, top_n=8):
    top = stats_df.sort_values("paired_t_FDR").head(top_n)["metabolite"].tolist()
    n = len(top)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0*ncols, 2.8*nrows), dpi=300)
    axes = np.array(axes).ravel()

    for ax, metab in zip(axes, top):
        g = paired_df[paired_df["metabolite"] == metab].copy()
        for _, row in g.iterrows():
            ax.plot([0, 1], [row["before"], row["after"]], marker="o", linewidth=0.8, alpha=0.55)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Before", "After"])
        ax.set_title(metab, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Paired changes of top metabolites", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "Fig_Metabolomics_C_top8_paired_lines.png", dpi=300)
    fig.savefig(outdir / "Fig_Metabolomics_C_top8_paired_lines.svg")
    plt.close(fig)


def plot_heatmap_top_log2fc(paired_df, stats_df, outdir, top_n=30):
    top = stats_df.sort_values("paired_t_FDR").head(top_n)["metabolite"].tolist()
    mat = paired_df[paired_df["metabolite"].isin(top)].pivot_table(
        index="metabolite", columns="subject", values="log2FC", aggfunc="mean"
    )
    mat = mat.loc[top]
    mat = mat.apply(lambda x: (x - x.mean()) / x.std(ddof=1), axis=1)

    fig, ax = plt.subplots(figsize=(10, max(5, 0.22 * len(mat))), dpi=300)
    im = ax.imshow(mat.values, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=7)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=90, fontsize=7)
    ax.set_title(f"Subject-level log2FC pattern of top {top_n} metabolites")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Row z-scored log2FC")
    fig.tight_layout()
    fig.savefig(outdir / "Fig_Metabolomics_D_top30_log2FC_heatmap.png", dpi=300)
    fig.savefig(outdir / "Fig_Metabolomics_D_top30_log2FC_heatmap.svg")
    plt.close(fig)


def plot_matching_categories(stats_df, outdir):
    df = stats_df.copy()
    df["sig"] = df["paired_t_FDR"] < 0.05
    summ = df.groupby("gene_validation_group").agg(
        n_metabolites=("metabolite", "count"),
        n_sig=("sig", "sum"),
        median_log2FC=("mean_log2FC", "median"),
    ).reset_index()
    summ["sig_percent"] = summ["n_sig"] / summ["n_metabolites"] * 100

    fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=300)
    ax.bar(summ["gene_validation_group"], summ["sig_percent"])
    ax.set_ylabel("Significant metabolites (%)")
    ax.set_title("Do changed metabolites match TVS gene-enrichment themes?")
    ax.set_xticklabels(summ["gene_validation_group"], rotation=20, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, row in summ.iterrows():
        ax.text(i, row["sig_percent"], f"{int(row['n_sig'])}/{int(row['n_metabolites'])}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(outdir / "Fig_Metabolomics_E_matching_categories.png", dpi=300)
    fig.savefig(outdir / "Fig_Metabolomics_E_matching_categories.svg")
    plt.close(fig)


def plot_index_correlation(index_df, tvs_df, outdir, index_col, target_col):
    if target_col not in tvs_df.columns:
        return
    dat = index_df.merge(tvs_df[["subject", target_col]], on="subject", how="inner")
    x = clean_numeric(dat[index_col])
    y = clean_numeric(dat[target_col])
    mask = x.notna() & y.notna()
    if mask.sum() < 6:
        return
    r, p = spearmanr(x[mask], y[mask])

    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=300)
    ax.scatter(x[mask], y[mask], s=35, alpha=0.75)
    m, b = np.polyfit(x[mask], y[mask], 1)
    xs = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(xs, m * xs + b, linewidth=1.5)
    ax.set_xlabel(index_col)
    ax.set_ylabel(target_col)
    ax.set_title("Metabolic index association")
    ax.text(0.05, 0.95, f"Spearman r = {r:.3f}\np = {p:.3g}\nn = {mask.sum()}",
            transform=ax.transAxes, ha="left", va="top")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    safe = safe_filename(f"Fig_Metabolomics_F_{index_col}_vs_{target_col}")
    fig.savefig(outdir / f"{safe}.png", dpi=300)
    fig.savefig(outdir / f"{safe}.svg")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input serum metabolomics Excel file")
    parser.add_argument("--sheet", default=0, help="Sheet name or index. Default: 0")
    parser.add_argument("--outdir", default="metabolomics_gene_validation_outputs")
    parser.add_argument("--tvs_csv", default=None, help="Optional CSV with subject, TVS/PVT values")
    parser.add_argument("--top_n", type=int, default=20)
    args = parser.parse_args()

    outdir = ensure_outdir(args.outdir)

    # Load and reshape
    raw, long_df = load_serum_excel(args.input, sheet_name=args.sheet)
    paired_df, pseudocount = make_paired_wide(long_df)
    stats_df = paired_metabolite_stats(paired_df)
    cat_df = category_summary(stats_df)
    enrich_t, table_t = matching_enrichment(stats_df, "paired_t_FDR", 0.05)
    enrich_w, table_w = matching_enrichment(stats_df, "wilcoxon_FDR", 0.05)

    # Subject-level matrices
    log2fc_mat = paired_df.pivot_table(index="metabolite", columns="subject", values="log2FC", aggfunc="mean")
    raw_delta_mat = paired_df.pivot_table(index="metabolite", columns="subject", values="raw_delta", aggfunc="mean")
    index_df = make_metabolic_indices(paired_df)

    # Save core tables
    long_df.to_csv(outdir / "serum_metabolomics_long_format.csv", index=False)
    paired_df.to_csv(outdir / "serum_metabolomics_paired_subject_level.csv", index=False)
    stats_df.to_csv(outdir / "metabolite_prepost_statistics.csv", index=False)
    cat_df.to_csv(outdir / "significant_metabolite_class_counts.csv", index=False)
    log2fc_mat.to_csv(outdir / "metabolite_subject_log2FC_matrix.csv")
    raw_delta_mat.to_csv(outdir / "metabolite_subject_raw_delta_matrix.csv")
    index_df.to_csv(outdir / "subject_metabolic_indices.csv", index=False)

    pd.concat([enrich_t, enrich_w], ignore_index=True).to_csv(
        outdir / "gene_validation_matching_enrichment_tests.csv", index=False
    )

    # Summary counts
    summary = pd.DataFrame({
        "metric": [
            "n_subjects",
            "n_metabolites",
            "pseudocount_for_log2FC",
            "paired_t_FDR_lt_0.05",
            "wilcoxon_FDR_lt_0.05",
            "paired_t_FDR_lt_0.10",
            "wilcoxon_FDR_lt_0.10",
            "n_gene_matching_metabolites",
            "n_gene_matching_sig_t_FDR05",
        ],
        "value": [
            paired_df["subject"].nunique(),
            paired_df["metabolite"].nunique(),
            pseudocount,
            int((stats_df["paired_t_FDR"] < 0.05).sum()),
            int((stats_df["wilcoxon_FDR"] < 0.05).sum()),
            int((stats_df["paired_t_FDR"] < 0.10).sum()),
            int((stats_df["wilcoxon_FDR"] < 0.10).sum()),
            int((stats_df["gene_validation_group"] == "Matches TVS gene enrichment").sum()),
            int(((stats_df["gene_validation_group"] == "Matches TVS gene enrichment") & (stats_df["paired_t_FDR"] < 0.05)).sum()),
        ]
    })
    summary.to_csv(outdir / "analysis_summary_counts.csv", index=False)

    # Figures
    plot_volcano(stats_df, outdir)
    plot_top_changed_bar(stats_df, outdir, top_n=args.top_n)
    plot_paired_lines(paired_df, stats_df, outdir, top_n=min(8, args.top_n))
    plot_heatmap_top_log2fc(paired_df, stats_df, outdir, top_n=min(30, max(args.top_n, 30)))
    plot_matching_categories(stats_df, outdir)

    # Optional TVS/PVT analysis
    tvs_corr_tables = {}
    mediation_tables = []
    if args.tvs_csv is not None:
        tvs_df = load_tvs_csv(args.tvs_csv)
        tvs_df.to_csv(outdir / "tvs_pvt_values_used.csv", index=False)

        for target in ["tvs_delta", "pvt_delta"]:
            if target in tvs_df.columns:
                for method in ["spearman", "pearson"]:
                    corr = metabolite_tvs_correlations(paired_df, tvs_df, target_col=target, method=method)
                    name = f"metabolite_log2FC_correlations_with_{target}_{method}"
                    corr.to_csv(outdir / f"{name}.csv", index=False)
                    tvs_corr_tables[name] = corr

        idx_corr = index_tvs_correlations(index_df, tvs_df)
        idx_corr.to_csv(outdir / "metabolic_index_correlations_with_TVS_PVT.csv", index=False)

        # Plot primary index correlations if available
        for idx_col in [
            "acylcarnitine_index_mean_log2FC",
            "fatty_acid_oxidation_index_mean_log2FC",
            "gene_matching_metabolism_index_mean_log2FC",
        ]:
            for target in ["tvs_delta", "pvt_delta"]:
                if target in tvs_df.columns:
                    plot_index_correlation(index_df, tvs_df, outdir, idx_col, target)

        # Simple mediation screens
        if {"tvs_delta", "pvt_delta"}.issubset(tvs_df.columns):
            for idx_col in [
                "acylcarnitine_index_mean_log2FC",
                "fatty_acid_oxidation_index_mean_log2FC",
                "gene_matching_metabolism_index_mean_log2FC",
            ]:
                med = simple_mediation_screen(index_df, tvs_df, idx_col)
                if not med.empty:
                    mediation_tables.append(med)
            if mediation_tables:
                pd.concat(mediation_tables, ignore_index=True).to_csv(
                    outdir / "simple_mediation_screen_metabolism_TVS_PVT.csv", index=False
                )

    # Save Excel workbook for easy inspection
    xlsx_out = outdir / "metabolomics_gene_validation_results.xlsx"
    with pd.ExcelWriter(xlsx_out, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary_counts", index=False)
        stats_df.to_excel(writer, sheet_name="prepost_stats", index=False)
        cat_df.to_excel(writer, sheet_name="category_summary", index=False)
        pd.concat([enrich_t, enrich_w], ignore_index=True).to_excel(writer, sheet_name="gene_match_tests", index=False)
        index_df.to_excel(writer, sheet_name="metabolic_indices", index=False)
        paired_df.to_excel(writer, sheet_name="paired_subject_level", index=False)
        # Keep matrix sheets manageable
        log2fc_mat.to_excel(writer, sheet_name="log2FC_matrix")
        raw_delta_mat.to_excel(writer, sheet_name="raw_delta_matrix")
        if args.tvs_csv is not None and "idx_corr" in locals():
            idx_corr.to_excel(writer, sheet_name="index_TVS_PVT_corr", index=False)
            for name, corr in tvs_corr_tables.items():
                # Excel sheet max 31 characters
                sheet = name.replace("metabolite_log2FC_", "")[:31]
                corr.to_excel(writer, sheet_name=sheet, index=False)
            if mediation_tables:
                pd.concat(mediation_tables, ignore_index=True).to_excel(writer, sheet_name="mediation_screen", index=False)

if __name__ == "__main__":
    main()

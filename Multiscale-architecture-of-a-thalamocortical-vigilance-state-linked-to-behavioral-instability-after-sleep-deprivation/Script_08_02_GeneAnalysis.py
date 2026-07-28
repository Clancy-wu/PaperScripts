import os
import numpy as np
import pandas as pd

from scipy.stats import zscore, pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import gseapy as gp

# ============================================================
# 0. User settings
# ============================================================

ATLAS_NII = "atlas/atlas_246.nii.gz"
ATLAS_INFO = "atlas/atlas_info_246.csv"
TVS_CSV = "atlas/tvs_246.csv"

OUTDIR = "tvs_gene_results"
os.makedirs(OUTDIR, exist_ok=True)

N_COMPONENTS = 1
N_BOOT = 1000
RANDOM_SEED = 20260702

# GO databases from Enrichr.
# You can change or add databases.
GENE_SETS = [
    "GO_Biological_Process_2023",
    "GO_Molecular_Function_2023",
    "GO_Cellular_Component_2023",
    "Reactome_2022",
    "KEGG_2021_Human"
]

# ============================================================
# 1. Load TVS values
# ============================================================
tvs = pd.read_csv(TVS_CSV)
required_cols = {"roi_id", "roi_name", "tvs_value"}
missing = required_cols - set(tvs.columns)
if len(missing) > 0:
    raise ValueError(f"TVS csv is missing columns: {missing}")

tvs = tvs.sort_values("roi_id").reset_index(drop=True)

if tvs.shape[0] != 246:
    raise ValueError(f"Expected 246 TVS values, but got {tvs.shape[0]}")

print("Loaded TVS values:")
print(tvs.head())
print(tvs.tail())

# ============================================================
# 2. Get AHBA gene expression using abagen
# ============================================================
import abagen

atlas_info = pd.read_csv(ATLAS_INFO)

expression = abagen.get_expression_data(
    atlas=ATLAS_NII,
    atlas_info=atlas_info,
    ibf_threshold=0.5,
    probe_selection="diff_stability",
    donor_probes="aggregate",
    sample_norm="srs",
    gene_norm="srs",
    norm_matched=True,
    missing="interpolate",
    return_donors=False,
    verbose=1
)

expression.to_csv(os.path.join(OUTDIR, "ahba_expression_246.csv"))

print("Expression matrix shape:", expression.shape)
print(expression.iloc[:5, :5])

# ============================================================
# 3. Align expression matrix and TVS vector
# ============================================================
# Convert expression index to integer if possible
try:
    expression.index = expression.index.astype(int)
except Exception:
    print("Warning: expression index could not be converted to int.")
    print("Please check whether expression index matches roi_id or roi_name.")

# Keep only ROIs with expression data
common_rois = sorted(set(tvs["roi_id"]).intersection(set(expression.index)))

if len(common_rois) < 200:
    print(f"Warning: only {len(common_rois)} ROIs matched between TVS and AHBA expression.")

tvs_matched = tvs[tvs["roi_id"].isin(common_rois)].copy()
tvs_matched = tvs_matched.sort_values("roi_id").reset_index(drop=True)

expr_matched = expression.loc[tvs_matched["roi_id"].values].copy()

# Remove genes with missing values
expr_matched = expr_matched.dropna(axis=1)

print("Matched ROIs:", len(common_rois))
print("Matched expression shape:", expr_matched.shape)

# Save matched data
tvs_matched.to_csv(os.path.join(OUTDIR, "tvs_matched_rois.csv"), index=False)
expr_matched.to_csv(os.path.join(OUTDIR, "ahba_expression_matched.csv"))

# ============================================================
# 4. Standardize TVS and gene expression
# ============================================================

y = tvs_matched["tvs_value"].values.reshape(-1, 1)
X = expr_matched.values

# z-score across ROIs
y_z = zscore(y, axis=0, nan_policy="omit")
X_z = zscore(X, axis=0, nan_policy="omit")

# Replace possible numerical NaNs
valid_genes = ~np.isnan(X_z).any(axis=0)
X_z = X_z[:, valid_genes]
gene_names = expr_matched.columns[valid_genes].to_numpy()

print("Final X shape:", X_z.shape)
print("Final y shape:", y_z.shape)

# ============================================================
# 5. PLS regression
# ============================================================

pls = PLSRegression(n_components=N_COMPONENTS)
pls.fit(X_z, y_z)

x_scores = pls.x_scores_[:, 0]
x_weights = pls.x_weights_[:, 0]

# Align sign so that positive PLS score correlates positively with TVS
r_pls_tvs, p_pls_tvs = pearsonr(x_scores, y_z.ravel())

if r_pls_tvs < 0:
    x_scores = -x_scores
    x_weights = -x_weights
    r_pls_tvs = -r_pls_tvs

print(f"PLS1 score ~ TVS correlation: r = {r_pls_tvs:.4f}, p = {p_pls_tvs:.4g}")

pls_roi_df = pd.DataFrame({
    "roi_id": tvs_matched["roi_id"].values,
    "roi_name": tvs_matched["roi_name"].values,
    "tvs_value": tvs_matched["tvs_value"].values,
    "pls1_score": x_scores
})

pls_roi_df.to_csv(os.path.join(OUTDIR, "pls1_roi_scores.csv"), index=False)

# ============================================================
# 6. Bootstrap gene weights
# ============================================================
rng = np.random.default_rng(RANDOM_SEED)

n_rois = X_z.shape[0]
n_genes = X_z.shape[1]

boot_weights = np.zeros((N_BOOT, n_genes), dtype=float)

for b in range(N_BOOT):
    idx = rng.choice(np.arange(n_rois), size=n_rois, replace=True)

    X_b = X_z[idx, :]
    y_b = y_z[idx, :]

    pls_b = PLSRegression(n_components=1)
    pls_b.fit(X_b, y_b)

    w_b = pls_b.x_weights_[:, 0]

    # Align bootstrap sign to original weights
    if np.corrcoef(w_b, x_weights)[0, 1] < 0:
        w_b = -w_b

    boot_weights[b, :] = w_b

    if (b + 1) % 100 == 0:
        print(f"Bootstrap {b + 1}/{N_BOOT} finished")

boot_mean = boot_weights.mean(axis=0)
boot_std = boot_weights.std(axis=0, ddof=1)

z_boot = x_weights / boot_std

# two-sided empirical normal approximation
from scipy.stats import norm
p_boot = 2 * (1 - norm.cdf(np.abs(z_boot)))

gene_rank_df = pd.DataFrame({
    "gene": gene_names,
    "pls1_weight": x_weights,
    "boot_mean": boot_mean,
    "boot_std": boot_std,
    "z_boot": z_boot,
    "p_boot": p_boot
})

gene_rank_df = gene_rank_df.sort_values("z_boot", ascending=False).reset_index(drop=True)

gene_rank_df.to_csv(os.path.join(OUTDIR, "pls1_gene_rank_bootstrap.csv"), index=False)

print("Top positive genes:")
print(gene_rank_df.head(20))

print("Top negative genes:")
print(gene_rank_df.tail(20))

# ============================================================
# 7. Prepare ranked gene list for GSEA prerank
# ============================================================

rnk = gene_rank_df[["gene", "z_boot"]].copy()
rnk = rnk.dropna()
rnk = rnk.sort_values("z_boot", ascending=False)

rnk_file = os.path.join(OUTDIR, "tvs_pls1_gene_rank.rnk")
rnk.to_csv(rnk_file, sep="\t", header=False, index=False)

# ============================================================
# 8. GO / KEGG / Reactome enrichment using GSEApy prerank
# ============================================================

for gene_set in GENE_SETS:
    print(f"Running prerank enrichment: {gene_set}")

    outdir_gsea = os.path.join(OUTDIR, f"gsea_{gene_set}")

    pre_res = gp.prerank(
        rnk=rnk_file,
        gene_sets=gene_set,
        outdir=outdir_gsea,
        permutation_num=1000,
        min_size=10,
        max_size=500,
        seed=RANDOM_SEED,
        threads=4,
        verbose=True
    )

    # Save main result table
    res = pre_res.res2d
    res.to_csv(os.path.join(OUTDIR, f"gsea_{gene_set}_results.csv"), index=False)

print("All analyses finished.")

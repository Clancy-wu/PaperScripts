import os
import re
import glob
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker
from scipy.stats import rankdata, zscore


# ============================================================
# User settings
# ============================================================

bold_dir = "validation_cohort/xcpd_24wmcfs03"
atlas_path = "/home/clancy/TemplateFlow/tpl-MNI152NLin2009cAsym/BN_Atlas246_2mm_tpl-MNI152NLin2009cAsym.nii.gz"
state_template_dir = "all_dynamic_brain_states"

output_dir = "validation_cohort_results_24WMCFS03"
os.makedirs(output_dir, exist_ok=True)

n_states = 10
tvs_state = 3

# File patterns
rw_pattern = "sub-*/func/sub-*_task-rest_run-01_space-MNI152NLin2009cAsym_res-2_desc-denoisedSmoothed_bold.nii.gz"
sd_pattern = "sub-*/func/sub-*_task-rest_run-02_space-MNI152NLin2009cAsym_res-2_desc-denoisedSmoothed_bold.nii.gz"

# ============================================================
# Helper functions
# ============================================================

def load_state_templates(state_template_dir, n_states=10):
    """
    Load raw state_1_mean.txt ~ state_10_mean.txt.

    Each file should contain 246 values.

    Output:
        templates: n_states × 246

    Important:
        For Spearman similarity, do NOT return the original Pearson-style
        z-scored templates. Spearman is Pearson correlation of ranked values.
        Therefore the rank-z transformation is performed later by
        prepare_template_rank_z().
    """

    templates = []

    for k in range(1, n_states + 1):
        file_path = os.path.join(
            state_template_dir,
            f"state_{k}_mean.txt"
        )

        v = np.loadtxt(file_path)
        templates.append(v)

    templates = np.vstack(templates)

    return templates


def prepare_template_rank_z(templates):
    """
    Convert raw state templates to ranked and z-scored templates.

    Spearman correlation = Pearson correlation of rank-transformed values.

    Input:
        templates: n_states × 246

    Output:
        templates_rank_z: n_states × 246
    """

    templates_rank = np.apply_along_axis(
        rankdata,
        axis=1,
        arr=templates
    )

    templates_rank_z = zscore(
        templates_rank,
        axis=1,
        ddof=0,
        nan_policy="omit"
    )

    return templates_rank_z


def extract_subject_id(file_path):
    """
    Extract subject ID from file name.

    Example:
        sub-001_task-rest_run-01_..._bold.nii.gz -> sub-001
    """

    fname = os.path.basename(file_path)
    subject = re.findall(r"(sub-\d+)_task", fname)[0]

    return subject


def extract_roi_timeseries(bold_path, atlas_path):
    """
    Extract 246 ROI time series from preprocessed BOLD image.

    Output:
        roi_ts: T × 246

    Note:
        standardize=True z-scores each ROI time series across time.
        This is different from Spearman spatial ranking across ROIs.
    """

    masker = NiftiLabelsMasker(
        labels_img=atlas_path,
        standardize=True,
        detrend=False,
        verbose=0
    )

    roi_ts = masker.fit_transform(bold_path)

    return roi_ts


def compute_state_assignment(roi_ts, templates_rank_z):
    """
    Compute Spearman spatial similarity between each TR and each state template.

    roi_ts:
        T × 246 ROI time series

    templates_rank_z:
        n_states × 246 ranked and z-scored state templates

    Output:
        similarity:
            T × n_states Spearman similarity matrix

        state_label:
            length T, values 1 to n_states
    """

    # ------------------------------------------------------------
    # 1. Rank-transform each TR spatial pattern across 246 ROIs
    # ------------------------------------------------------------
    # Spearman correlation is Pearson correlation after rank transform.
    roi_rank = np.apply_along_axis(
        rankdata,
        axis=1,
        arr=roi_ts
    )

    # ------------------------------------------------------------
    # 2. Z-score ranked TR patterns across ROIs
    # ------------------------------------------------------------
    # This is the "-z" variable mentioned earlier: roi_rank_z.
    # After rank transformation, z-scoring makes dot product equal
    # to Pearson correlation of ranks, i.e., Spearman correlation.
    roi_rank_z = zscore(
        roi_rank,
        axis=1,
        ddof=0,
        nan_policy="omit"
    )

    # ------------------------------------------------------------
    # 3. Spearman similarity by matrix multiplication
    # ------------------------------------------------------------
    # Both roi_rank_z and templates_rank_z are z-scored across ROIs.
    # Therefore Spearman correlation = dot product / n_roi.
    n_roi = roi_rank_z.shape[1]

    similarity = np.dot(
        roi_rank_z,
        templates_rank_z.T
    ) / n_roi

    # ------------------------------------------------------------
    # 4. Winner-take-all state assignment
    # ------------------------------------------------------------
    state_label = np.argmax(similarity, axis=1) + 1

    return similarity, state_label


def compute_fo_sr_for_all_states(state_label, similarity, n_states=10):
    """
    Compute FO and state-specific SR for all states.

    FO_k:
        fraction of TRs assigned to state k

    SR_k:
        state-specific switching rate based on binary occurrence sequence:
        y_t = 1 if state k, otherwise 0

        SR_k = mean(abs(diff(y_t)))

    global_SR:
        switching rate across all state labels:
        mean(state_label[t] != state_label[t-1])
    """

    results = []

    global_sr = np.mean(state_label[1:] != state_label[:-1])

    for k in range(1, n_states + 1):

        y = (state_label == k).astype(int)

        fo = np.mean(y)
        sr = np.mean(np.abs(np.diff(y)))

        n_entries = np.sum((y[1:] == 1) & (y[:-1] == 0))
        n_exits = np.sum((y[1:] == 0) & (y[:-1] == 1))

        mean_expression = np.mean(similarity[:, k - 1])
        mean_expression_when_assigned = (
            np.mean(similarity[y == 1, k - 1])
            if np.sum(y) > 0
            else np.nan
        )

        results.append(
            {
                "state": k,
                "FO": fo,
                "SR_state_specific": sr,
                "n_entries": n_entries,
                "n_exits": n_exits,
                "mean_expression": mean_expression,
                "mean_expression_when_assigned": mean_expression_when_assigned,
                "global_SR": global_sr
            }
        )

    result_df = pd.DataFrame(results)

    return result_df


def process_one_bold(
    bold_path,
    session,
    atlas_path,
    templates_rank_z,
    n_states=10
):
    """
    Process one BOLD file and calculate FO/SR for all states.
    """

    subject = extract_subject_id(bold_path)

    roi_ts = extract_roi_timeseries(
        bold_path=bold_path,
        atlas_path=atlas_path
    )

    similarity, state_label = compute_state_assignment(
        roi_ts=roi_ts,
        templates_rank_z=templates_rank_z
    )

    fo_sr_df = compute_fo_sr_for_all_states(
        state_label=state_label,
        similarity=similarity,
        n_states=n_states
    )

    fo_sr_df.insert(0, "session", session)
    fo_sr_df.insert(0, "subject", subject)

    tr_df = pd.DataFrame(
        {
            "subject": subject,
            "session": session,
            "TR": np.arange(1, len(state_label) + 1),
            "state": state_label
        }
    )

    for k in range(1, n_states + 1):
        tr_df[f"similarity_state_{k}"] = similarity[:, k - 1]

    return fo_sr_df, tr_df


# ============================================================
# Main analysis
# ============================================================

templates = load_state_templates(
    state_template_dir=state_template_dir,
    n_states=n_states
)

templates_rank_z = prepare_template_rank_z(templates)

rw_files = sorted(glob.glob(os.path.join(bold_dir, rw_pattern)))
sd_files = sorted(glob.glob(os.path.join(bold_dir, sd_pattern)))

print(f"Number of RW files: {len(rw_files)}")
print(f"Number of SD files: {len(sd_files)}")

all_files = []

for f in rw_files:
    all_files.append((f, "RW"))

for f in sd_files:
    all_files.append((f, "SD"))

all_fo_sr = []
all_tr = []

for bold_path, session in all_files:

    print(f"Processing: {bold_path}")

    fo_sr_df, tr_df = process_one_bold(
        bold_path=bold_path,
        session=session,
        atlas_path=atlas_path,
        templates_rank_z=templates_rank_z,
        n_states=n_states
    )

    all_fo_sr.append(fo_sr_df)
    all_tr.append(tr_df)

fo_sr_all_df = pd.concat(all_fo_sr, axis=0, ignore_index=True)
tr_all_df = pd.concat(all_tr, axis=0, ignore_index=True)


# ============================================================
# Save long-format results
# ============================================================

fo_sr_long_csv = os.path.join(
    output_dir,
    "state_FO_SR_long.csv"
)

tr_state_csv = os.path.join(
    output_dir,
    "TR_state_assignment.csv"
)

fo_sr_all_df.to_csv(
    fo_sr_long_csv,
    index=False
)

tr_all_df.to_csv(
    tr_state_csv,
    index=False
)


# ============================================================
# Save wide-format results
# ============================================================

fo_wide = fo_sr_all_df.pivot_table(
    index=["subject", "session"],
    columns="state",
    values="FO"
)
fo_wide.columns = [
    f"state_{int(c)}_FO" for c in fo_wide.columns
]

sr_wide = fo_sr_all_df.pivot_table(
    index=["subject", "session"],
    columns="state",
    values="SR_state_specific"
)
sr_wide.columns = [
    f"state_{int(c)}_SR" for c in sr_wide.columns
]

expr_wide = fo_sr_all_df.pivot_table(
    index=["subject", "session"],
    columns="state",
    values="mean_expression"
)
expr_wide.columns = [
    f"state_{int(c)}_mean_expression" for c in expr_wide.columns
]

expr_assigned_wide = fo_sr_all_df.pivot_table(
    index=["subject", "session"],
    columns="state",
    values="mean_expression_when_assigned"
)
expr_assigned_wide.columns = [
    f"state_{int(c)}_mean_expression_when_assigned" for c in expr_assigned_wide.columns
]

wide_df = pd.concat(
    [fo_wide, sr_wide, expr_wide, expr_assigned_wide],
    axis=1
).reset_index()

global_sr_df = fo_sr_all_df[
    ["subject", "session", "global_SR"]
].drop_duplicates()

wide_df = wide_df.merge(
    global_sr_df,
    on=["subject", "session"],
    how="left"
)

wide_csv = os.path.join(
    output_dir,
    "state_FO_SR_wide.csv"
)

wide_df.to_csv(
    wide_csv,
    index=False
)


# ============================================================
# Save delta SD - RW results
# ============================================================

wide_pivot = wide_df.pivot(
    index="subject",
    columns="session"
)

delta_columns = {}

for variable in wide_df.columns:
    if variable in ["subject", "session"]:
        continue

    delta_columns[f"delta_{variable}"] = (
        wide_pivot[(variable, "SD")] - wide_pivot[(variable, "RW")]
    )

delta_out = pd.DataFrame(delta_columns)
delta_out.index.name = "subject"
delta_out = delta_out.reset_index()

delta_csv = os.path.join(
    output_dir,
    "state_FO_SR_delta_SD_minus_RW.csv"
)

delta_out.to_csv(
    delta_csv,
    index=False
)

print("\nFinished.")
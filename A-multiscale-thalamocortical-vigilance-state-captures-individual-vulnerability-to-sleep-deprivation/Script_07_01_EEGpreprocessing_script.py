from __future__ import annotations

from pathlib import Path
from glob import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================================================
# 1. Setting
# =========================================================
PROJECT_ROOT = Path("EEG_reactivity_pipeline")
RAW_DIR = Path("EEG")

DIRS = {
    "preproc": PROJECT_ROOT / "preprocessed",
    "epochs": PROJECT_ROOT / "epochs",
    "features_epoch_qc": PROJECT_ROOT / "qc" / "feature_epoch_distributions",
    "qc_psd": PROJECT_ROOT / "qc" / "psd",
    "qc_ica": PROJECT_ROOT / "qc" / "ica",
    "stats": PROJECT_ROOT / "stats",
    "json": PROJECT_ROOT / "json",
}
for p in DIRS.values():
    p.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. Original Directories
# =========================================================
SESSION_LABELS = ["RW", "SD"]
PRIMARY_REST_CONDITION = "CloseEye"   
OPTIONAL_REST_CONDITIONS = ["OpenEye"]  
TASK_CONDITION = "PVT"

# ROI select
ROI_DICT = {
    "global": None,  # Using all electrodes 
    "posterior": ["P3", "P4", "Pz", "PO3", "PO4", "PO7", "PO8", "Oz", "O1", "O2"],
    "frontal": ["F3", "F4", "Fz", "F1", "F2"],
    "frontocentral": ["FC1", "FC2", "FC3", "FC4", "FCz"],
    "central": ["C1", "C2", "C3", "C4", "Cz"],
}

# frequency bands: theta / alpha / beta
BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
}

# Features of primary analysis
#   - global_theta
#   - posterior_alpha
#   - global_theta_alpha_ratio
#   - global_theta_cv
#   - posterior_alpha_cv
#   - global_theta_alpha_ratio_cv
#
PRIMARY_FEATURE_SPECS = [
    ("global", "theta"),
    ("posterior", "alpha"),
]
SUPPLEMENTARY_FEATURE_SPECS = [
    ("frontal", "theta"),
    ("frontocentral", "theta"),
    ("central", "alpha"),
    ("frontal", "beta"),
]

RATIO_SPECS = [
    ("global", "theta", "alpha", "theta_alpha_ratio"),
]

# Preprocessing parameters
RESAMPLE_SFREQ = 500
USE_ICA = True
ICA_N_COMPONENTS = 20
ICA_RANDOM_STATE = 97
NOTCH_FREQ = 50.0

# Filtering
REST_FILTER = (1.0, 40.0)
TASK_FILTER = (0.5, 40.0)

# epoch 
REST_EPOCH_LEN = 2.0
REST_EPOCH_OVERLAP = 1.0

# use 'pre stimilus' as the preparatory vigilance of PVT. 
PVT_EPOCH_TMIN = -1.0
PVT_EPOCH_TMAX = 0.0
PVT_EVENT_KEY = "Stimulus/S  1"

# reject parameters: keep data quality
REST_REJECT = dict(eeg=200e-6)
PVT_REJECT = dict(eeg=150e-6)

# to avoid potential erorrs from computing "ratio / log"
EPS = np.finfo(float).eps

# =========================================================
# 3. Functions
# =========================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_set_montage(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    try:
        raw.set_montage("standard_1020", on_missing="ignore")
    except Exception as e:
        print(f"[Warning] montage setting error: {e}")
    return raw


def get_eeg_picks(raw_or_epochs) -> List[int]:
    return mne.pick_types(raw_or_epochs.info, eeg=True, exclude="bads")


def locate_vhdr(sub_dir: Path, session: str, condition: str) -> Optional[Path]:
    pattern = sub_dir / session / condition / "*.vhdr"
    matches = sorted(glob(str(pattern)))
    if not matches:
        return None
    return Path(matches[0])


def save_psd_qc(raw: mne.io.BaseRaw, out_png: Path, title: str) -> None:
    try:
        spec = raw.compute_psd(fmin=1, fmax=40, verbose=False)
        fig = spec.plot(show=False)
        fig.suptitle(title)
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[warning] PSD QC saving failed: {title}: {e}")


def save_epoch_distribution_plot(epoch_values: np.ndarray, out_png: Path, title: str) -> None:
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(np.arange(len(epoch_values)), epoch_values, marker="o", ms=2, lw=1)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("value")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[warning] epoch distribution saving failed: {title}: {e}")


def robust_cv(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    mu = np.mean(x)
    if np.isclose(mu, 0.0):
        return np.nan
    return float(np.std(x, ddof=1) / (mu + EPS))


def safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.mean(x))


def safe_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.std(x, ddof=1))


def save_json(data: dict, out_file: Path) -> None:
    ensure_dir(out_file.parent)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    flat_cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            flat_cols.append("_".join([str(x) for x in c if str(x) != ""]).strip("_"))
        else:
            flat_cols.append(str(c))
    df = df.copy()
    df.columns = flat_cols
    return df

# =========================================================
# 4. Preprocessing
# =========================================================
def run_ica(raw: mne.io.BaseRaw, sub: str, session: str, condition: str) -> Tuple[mne.io.BaseRaw, List[int]]:
    """
    use Fp1/Fp2/Fpz as EOG proxy for ICA identification.
    """
    print(f"[ICA] {sub} {session} {condition}")

    raw_ica = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
    ica = ICA(
        n_components=ICA_N_COMPONENTS,
        random_state=ICA_RANDOM_STATE,
        max_iter="auto"
    )
    ica.fit(raw_ica, verbose=False)

    proxy = [ch for ch in ["Fp1", "Fp2", "Fpz"] if ch in raw_ica.ch_names]
    eog_inds: List[int] = []
    eog_scores = None

    if proxy:
        try:
            eog_inds, eog_scores = ica.find_bads_eog(raw_ica, ch_name=proxy, verbose=False)
        except Exception as e:
            print(f"[warning] ICA error: {sub} {session} {condition}: {e}")

    ica.exclude = eog_inds
    cleaned = ica.apply(raw.copy(), verbose=False)

    # QC
    try:
        figs = ica.plot_components(show=False)
        if not isinstance(figs, list):
            figs = [figs]
        for i, fig in enumerate(figs):
            fig.savefig(
                DIRS["qc_ica"] / f"{sub}_{session}_{condition}_ica_components_{i}.png",
                dpi=150,
                bbox_inches="tight"
            )
            plt.close(fig)
    except Exception as e:
        print(f"[warning] ICA component saving error: {sub} {session} {condition}: {e}")

    if eog_scores is not None:
        try:
            fig = ica.plot_scores(eog_scores, show=False)
            fig.savefig(
                DIRS["qc_ica"] / f"{sub}_{session}_{condition}_ica_eog_scores.png",
                dpi=150,
                bbox_inches="tight"
            )
            plt.close(fig)
        except Exception as e:
            print(f"[warning] ICA score saving error: {sub} {session} {condition}: {e}")

    return cleaned, list(ica.exclude)


def preprocess_raw(vhdr_path: Path, sub: str, session: str, condition: str, kind: str) -> Tuple[mne.io.BaseRaw, List[int]]:
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
    raw = safe_set_montage(raw)

    eeg_picks = get_eeg_picks(raw)
    if len(eeg_picks) == 0:
        raise RuntimeError(f"{sub} {session} {condition} no EEG channels")

    raw.set_eeg_reference("average", verbose=False)

    if RESAMPLE_SFREQ is not None and raw.info["sfreq"] > RESAMPLE_SFREQ:
        raw.resample(RESAMPLE_SFREQ, verbose=False)

    if kind == "rest":
        l_freq, h_freq = REST_FILTER
    elif kind == "task":
        l_freq, h_freq = TASK_FILTER
    else:
        raise ValueError(f"unknown kind: {kind}")

    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    raw.notch_filter(freqs=NOTCH_FREQ, verbose=False)

    removed_components: List[int] = []
    if USE_ICA:
        raw, removed_components = run_ica(raw, sub, session, condition)

    raw.pick(get_eeg_picks(raw))

    return raw, removed_components


# =========================================================
# 5. Epoching
# =========================================================
def make_rest_epochs(raw: mne.io.BaseRaw) -> mne.Epochs:
    events = mne.make_fixed_length_events(
        raw,
        duration=REST_EPOCH_LEN,
        overlap=REST_EPOCH_OVERLAP
    )
    epochs = mne.Epochs(
        raw,
        events,
        tmin=0.0,
        tmax=REST_EPOCH_LEN,
        baseline=None,
        preload=True,
        reject=REST_REJECT,
        reject_by_annotation=True,
        verbose=False
    )
    return epochs


def make_pvt_prestim_epochs(raw: mne.io.BaseRaw) -> Tuple[mne.Epochs, Dict[str, int]]:
    events, event_id_all = mne.events_from_annotations(raw, verbose=False)
    if PVT_EVENT_KEY not in event_id_all:
        raise RuntimeError(
            f"PVT: no event: {PVT_EVENT_KEY}, actually event：{list(event_id_all.keys())}"
        )

    selected = {PVT_EVENT_KEY: event_id_all[PVT_EVENT_KEY]}
    epochs = mne.Epochs(
        raw,
        events,
        event_id=selected,
        tmin=PVT_EPOCH_TMIN,
        tmax=PVT_EPOCH_TMAX,
        baseline=None,
        preload=True,
        reject=PVT_REJECT,
        reject_by_annotation=True,
        verbose=False
    )
    return epochs, event_id_all


# =========================================================
# 6. Features extraction
# ---------------------------------------------------------
def get_roi_picks(epochs: mne.Epochs, roi_name: str) -> List[int]:
    if ROI_DICT[roi_name] is None:
        return get_eeg_picks(epochs)
    picks = [epochs.ch_names.index(ch) for ch in ROI_DICT[roi_name] if ch in epochs.ch_names]
    return picks


def compute_epoch_psd_linear(epochs: mne.Epochs, fmin: float = 1.0, fmax: float = 40.0) -> Tuple[np.ndarray, np.ndarray]:
    psd = epochs.compute_psd(method="welch", fmin=fmin, fmax=fmax, verbose=False)
    psds, freqs = psd.get_data(return_freqs=True)
    psds = np.maximum(psds, EPS)
    return psds, freqs


def band_average_linear(psds_linear: np.ndarray, freqs: np.ndarray, band_range: Tuple[float, float]) -> np.ndarray:
    fmin, fmax = band_range
    idx = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(idx):
        raise RuntimeError(f"frequency band error: {band_range}. ")
    return psds_linear[:, :, idx].mean(axis=-1)


def summarize_epoch_vector(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    return {
        "mean_linear": safe_mean(values),
        "mean_db": float(10.0 * np.log10(safe_mean(values) + EPS)) if np.isfinite(safe_mean(values)) else np.nan,
        "sd_linear": safe_std(values),
        "cv_linear": robust_cv(values),
        "n_epochs_used": int(values.size),
    }


def extract_condition_features(
    epochs: mne.Epochs,
    sub: str,
    session: str,
    condition: str,
    save_epoch_qc: bool = True
) -> Tuple[Dict[str, float], pd.DataFrame]:
    psds_linear, freqs = compute_epoch_psd_linear(epochs, fmin=1.0, fmax=40.0)

    band_cache = {}
    for band_name, band_range in BANDS.items():
        band_cache[band_name] = band_average_linear(psds_linear, freqs, band_range)

    summary_features: Dict[str, float] = {
        "n_epochs_total": int(len(epochs)),
    }
    epoch_rows = []

    # 1) frequency band features
    all_feature_specs = PRIMARY_FEATURE_SPECS + SUPPLEMENTARY_FEATURE_SPECS
    for roi_name, band_name in all_feature_specs:
        picks = get_roi_picks(epochs, roi_name)
        if len(picks) == 0:
            print(f"[warning] ROI without valid channel: {roi_name} @ {sub} {session} {condition}")
            continue

        epoch_vals = band_cache[band_name][:, picks].mean(axis=1)

        stat = summarize_epoch_vector(epoch_vals)
        base_name = f"{roi_name}_{band_name}"

        summary_features[f"{base_name}_mean_linear"] = stat["mean_linear"] # linear means original value without db
        summary_features[f"{base_name}_mean_db"] = stat["mean_db"]
        summary_features[f"{base_name}_sd_linear"] = stat["sd_linear"]
        summary_features[f"{base_name}_cv_linear"] = stat["cv_linear"]
        summary_features[f"{base_name}_n_epochs_used"] = stat["n_epochs_used"]

        if save_epoch_qc:
            save_epoch_distribution_plot(
                epoch_vals,
                DIRS["features_epoch_qc"] / f"{sub}_{session}_{condition}_{base_name}.png",
                f"{sub} {session} {condition} {base_name}"
            )

        for i, v in enumerate(epoch_vals):
            epoch_rows.append({
                "sub": sub,
                "session": session,
                "condition": condition,
                "feature": base_name,
                "epoch_idx": i,
                "value_linear": float(v),
                "value_db": float(10.0 * np.log10(v + EPS)),
            })

    # 2) ratio features
    for roi_name, num_band, den_band, ratio_name in RATIO_SPECS:
        picks = get_roi_picks(epochs, roi_name)
        if len(picks) == 0:
            continue

        num_vals = band_cache[num_band][:, picks].mean(axis=1)
        den_vals = band_cache[den_band][:, picks].mean(axis=1)
        ratio_vals = num_vals / (den_vals + EPS)

        stat = summarize_epoch_vector(ratio_vals)
        base_name = f"{roi_name}_{ratio_name}"

        summary_features[f"{base_name}_mean_linear"] = stat["mean_linear"]

        summary_features[f"{base_name}_sd_linear"] = stat["sd_linear"]
        summary_features[f"{base_name}_cv_linear"] = stat["cv_linear"]
        summary_features[f"{base_name}_n_epochs_used"] = stat["n_epochs_used"]

        if save_epoch_qc:
            save_epoch_distribution_plot(
                ratio_vals,
                DIRS["features_epoch_qc"] / f"{sub}_{session}_{condition}_{base_name}.png",
                f"{sub} {session} {condition} {base_name}"
            )

        for i, v in enumerate(ratio_vals):
            epoch_rows.append({
                "sub": sub,
                "session": session,
                "condition": condition,
                "feature": base_name,
                "epoch_idx": i,
                "value_linear": float(v),
                "value_db": np.nan,
            })

    epoch_df = pd.DataFrame(epoch_rows)
    return summary_features, epoch_df


# =========================================================
# 7. Dataframe output
# =========================================================
def dict_to_long_rows(sub: str, session: str, condition: str, feature_dict: Dict[str, float]) -> List[dict]:
    rows = []
    for marker, value in feature_dict.items():
        rows.append({
            "sub": sub,
            "session": session,
            "condition": condition,
            "marker": marker,
            "value": value,
        })
    return rows


def compute_delta_tables(df_long: pd.DataFrame, rest_condition: str = PRIMARY_REST_CONDITION, task_condition: str = TASK_CONDITION) -> pd.DataFrame:
    """
    Δ = PVT - Rest 
    """
    if df_long.empty:
        return pd.DataFrame()

    wide = df_long.pivot_table(
        index=["sub", "session", "marker"],
        columns="condition",
        values="value",
        aggfunc="first"
    ).reset_index()

    if rest_condition not in wide.columns or task_condition not in wide.columns:
        print(f"[warning] cannot build delta： no {rest_condition} or {task_condition}")
        return pd.DataFrame()

    wide["delta_value"] = wide[task_condition] - wide[rest_condition]
    delta_df = wide[["sub", "session", "marker", "delta_value"]].copy()
    delta_df["delta_name"] = f"{task_condition}_minus_{rest_condition}"
    return delta_df


def compute_double_delta(delta_df: pd.DataFrame, rw_label: str = "RW", sd_label: str = "SD") -> pd.DataFrame:
    """
    ΔΔ = Δ_SD - Δ_RW
    """
    if delta_df.empty:
        return pd.DataFrame()

    wide = delta_df.pivot_table(
        index=["sub", "marker", "delta_name"],
        columns="session",
        values="delta_value",
        aggfunc="first"
    ).reset_index()

    if rw_label not in wide.columns or sd_label not in wide.columns:
        print(f"[warning] cannot build double-delta： no {rw_label} or {sd_label}")
        return pd.DataFrame()

    wide["double_delta_value"] = wide[sd_label] - wide[rw_label]
    out = wide[["sub", "marker", "delta_name", "double_delta_value"]].copy()
    out["double_delta_name"] = f"{sd_label}_minus_{rw_label}_of_{wide['delta_name'].iloc[0]}" if len(wide) > 0 else np.nan
    return out


def select_primary_analysis_markers(df_long: pd.DataFrame) -> pd.DataFrame:
    primary_markers = [
        "global_theta_mean_db",
        "posterior_alpha_mean_db",
        "global_theta_alpha_ratio_mean_linear",
        "global_theta_cv_linear",
        "posterior_alpha_cv_linear",
        "global_theta_alpha_ratio_cv_linear",
        "n_epochs_total",
    ]
    return df_long[df_long["marker"].isin(primary_markers)].copy()


# =========================================================
# 8. Main
# =========================================================
def main():
    condition_rows: List[dict] = []
    epoch_feature_dfs: List[pd.DataFrame] = []
    qc_rows: List[dict] = []

    subjects = sorted([x for x in os.listdir(RAW_DIR) if x.startswith("sub-")])
    print(f"found subjects: {len(subjects)}")

    for sub in subjects:
        print(f"\n========== preprocessing {sub} ==========")
        sub_dir = RAW_DIR / sub

        for session in SESSION_LABELS:
            print(f"------ session: {session} ------")

            # CloseEye as the primary resting condition；OpenEye is optional. 
            condition_plan = [(PRIMARY_REST_CONDITION, "rest")] + [(c, "rest") for c in OPTIONAL_REST_CONDITIONS] + [(TASK_CONDITION, "task")]

            for condition, kind in condition_plan:
                vhdr = locate_vhdr(sub_dir, session, condition)
                if vhdr is None:
                    print(f"[skip] {sub} {session} {condition} no vhdr")
                    continue

                print(f"[reading] {sub} {session} {condition}: {vhdr}")
                try:
                    raw, removed_components = preprocess_raw(vhdr, sub, session, condition, kind=kind)

                    out_raw = DIRS["preproc"] / f"{sub}_{session}_{condition}_preproc_raw.fif"
                    raw.save(out_raw, overwrite=True, verbose=False)

                    save_psd_qc(
                        raw,
                        DIRS["qc_psd"] / f"{sub}_{session}_{condition}_psd.png",
                        f"{sub} {session} {condition} PSD"
                    )

                    if kind == "rest":
                        epochs = make_rest_epochs(raw)
                    else:
                        epochs, event_id_all = make_pvt_prestim_epochs(raw)

                    out_epo = DIRS["epochs"] / f"{sub}_{session}_{condition}-epo.fif"
                    epochs.save(out_epo, overwrite=True, verbose=False)

                    summary_features, epoch_df = extract_condition_features(
                        epochs=epochs,
                        sub=sub,
                        session=session,
                        condition=condition,
                        save_epoch_qc=True
                    )

                    condition_rows.extend(
                        dict_to_long_rows(sub, session, condition, summary_features)
                    )
                    if not epoch_df.empty:
                        epoch_feature_dfs.append(epoch_df)

                    save_json(
                        summary_features,
                        DIRS["json"] / f"{sub}_{session}_{condition}_summary_features.json"
                    )

                    qc_rows.append({
                        "sub": sub,
                        "session": session,
                        "condition": condition,
                        "kind": kind,
                        "n_epochs": len(epochs),
                        "n_ica_removed": len(removed_components),
                        "ica_removed_components": ",".join(map(str, removed_components)) if removed_components else "",
                        "preproc_raw": str(out_raw),
                        "epochs_file": str(out_epo),
                    })

                except Exception as e:
                    print(f"[error] {sub} {session} {condition}: {e}")
                    qc_rows.append({
                        "sub": sub,
                        "session": session,
                        "condition": condition,
                        "kind": kind,
                        "error": str(e),
                    })

    # =====================================================
    # 9. output
    # =====================================================
    df_condition_long = pd.DataFrame(condition_rows)
    df_qc = pd.DataFrame(qc_rows)
    df_epoch = pd.concat(epoch_feature_dfs, axis=0, ignore_index=True) if epoch_feature_dfs else pd.DataFrame()

    # save condition-level long
    out_condition_long = DIRS["stats"] / "eeg_condition_features_long.csv"
    df_condition_long.to_csv(out_condition_long, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_condition_long}")

    # save condition-level wide
    if not df_condition_long.empty:
        df_condition_wide = df_condition_long.pivot_table(
            index=["sub", "session"],
            columns=["condition", "marker"],
            values="value",
            aggfunc="first"
        ).reset_index()
        df_condition_wide = flatten_multiindex_columns(df_condition_wide)
        out_condition_wide = DIRS["stats"] / "eeg_condition_features_wide.csv"
        df_condition_wide.to_csv(out_condition_wide, index=False, encoding="utf-8-sig")
        print(f"Saved: {out_condition_wide}")

    # save epoch-level
    out_epoch = DIRS["stats"] / "eeg_epoch_feature_values.csv"
    df_epoch.to_csv(out_epoch, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_epoch}")

    # save QC
    out_qc = DIRS["stats"] / "eeg_qc_summary.csv"
    df_qc.to_csv(out_qc, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_qc}")

    # Delta
    df_delta = compute_delta_tables(df_condition_long, rest_condition=PRIMARY_REST_CONDITION, task_condition=TASK_CONDITION)
    out_delta = DIRS["stats"] / "eeg_delta_task_minus_rest.csv"
    df_delta.to_csv(out_delta, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_delta}")

    # Double delta
    df_double_delta = compute_double_delta(df_delta, rw_label="RW", sd_label="SD")
    out_double_delta = DIRS["stats"] / "eeg_double_delta_sd_minus_rw_of_task_minus_rest.csv"
    df_double_delta.to_csv(out_double_delta, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_double_delta}")

    # marker
    df_primary = select_primary_analysis_markers(df_condition_long)
    out_primary = DIRS["stats"] / "eeg_primary_markers_condition_level.csv"
    df_primary.to_csv(out_primary, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_primary}")

    if not df_delta.empty:
        df_delta_primary = df_delta[df_delta["marker"].isin([
            "global_theta_mean_db",
            "posterior_alpha_mean_db",
            "global_theta_alpha_ratio_mean_linear",
            "global_theta_cv_linear",
            "posterior_alpha_cv_linear",
            "global_theta_alpha_ratio_cv_linear",
            "n_epochs_total",
        ])].copy()
        out_delta_primary = DIRS["stats"] / "eeg_primary_markers_delta.csv"
        df_delta_primary.to_csv(out_delta_primary, index=False, encoding="utf-8-sig")
        print(f"Saved: {out_delta_primary}")

    if not df_double_delta.empty:
        df_double_delta_primary = df_double_delta[df_double_delta["marker"].isin([
            "global_theta_mean_db",
            "posterior_alpha_mean_db",
            "global_theta_alpha_ratio_mean_linear",
            "global_theta_cv_linear",
            "posterior_alpha_cv_linear",
            "global_theta_alpha_ratio_cv_linear",
            "n_epochs_total",
        ])].copy()
        out_double_delta_primary = DIRS["stats"] / "eeg_primary_markers_double_delta.csv"
        df_double_delta_primary.to_csv(out_double_delta_primary, index=False, encoding="utf-8-sig")
        print(f"Saved: {out_double_delta_primary}")

    merge_hint = {
        "join_keys_condition_level": ["sub", "session"],
        "join_keys_delta_level": ["sub", "session"],
        "join_keys_double_delta_level": ["sub"],
        "recommended_primary_eeg_markers": [
            "global_theta_mean_db",
            "posterior_alpha_mean_db",
            "global_theta_alpha_ratio_mean_linear",
            "global_theta_cv_linear",
            "posterior_alpha_cv_linear",
            "global_theta_alpha_ratio_cv_linear",
        ],
        "main_contrast_definition": "delta = Task - CloseEye; double_delta = delta_SD - delta_RW",
        "recommended_downstream_statistics": [
            "double_delta_EEG ~ double_delta_rt_cv",
            "double_delta_EEG ~ double_delta_hr_cv",
            "double_delta_EEG ~ double_delta_FO",
        ],
    }
    save_json(merge_hint, DIRS["json"] / "analysis_merge_hint.json")
    print(f"Saved: {DIRS['json'] / 'analysis_merge_hint.json'}")


if __name__ == "__main__":
    main()

from pathlib import Path
import re
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt

# =========================
# 0. Users setting 
# =========================
ECG_ROOT = Path("ECG")
OUT_DIR = Path("ECG_preprocess")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLING_RATE = 1024
ECG_CLEAN_METHOD = "biosppy"   
SAVE_BAD_PLOTS = True        
BAD_PLOT_DIR = OUT_DIR / "bad_ecg_plots"
BAD_PLOT_DIR.mkdir(parents=True, exist_ok=True)

EYE_CONDITIONS = ["OpenEye", "CloseEye", "Task"] # The Task is PVT

# =========================
# 1. Functions
# =========================
def parse_sub_run(path: Path):

    folder = path.parts[-3]  # ECG/sub-009_run-01/OpenEye/ecg.csv
    m = re.match(r"(sub-\d+)_run-(\d+)", folder)
    if m is None:
        raise ValueError(f"File Errors to parse: {folder}")

    sub = m.group(1)
    run_num = m.group(2)

    if run_num == "01":
        state = "RW"
    elif run_num == "02":
        state = "SD"
    else:
        state = f"run-{run_num}"

    return sub, state


def load_ecg_csv(csv_path: Path):
    """
    Read ecg.csv
    """
    df = pd.read_csv(csv_path, header=None)
    ecg = df.iloc[:, 0].astype(float).to_numpy()
    return ecg


def detect_rpeaks(ecg, sampling_rate=1024, clean_method="biosppy"):
    """
    ECG clean + R peak detection
    """
    ecg_clean = nk.ecg_clean(ecg, sampling_rate=sampling_rate, method=clean_method)
    signals, info = nk.ecg_peaks(ecg_clean, sampling_rate=sampling_rate)
    rpeaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    return ecg_clean, rpeaks


def compute_rr_metrics(rpeaks, sampling_rate=1024):
    """
    Compute RR、HR、RMSSD、SDNN from R peaks
    """
    if len(rpeaks) < 3:
        return {
            "n_rpeaks": len(rpeaks),
            "mean_hr_bpm": np.nan,
            "hr_cv": np.nan,
            "rmssd_ms": np.nan,
            "sdnn_ms": np.nan,
            "mean_rr_ms": np.nan,
            "rr_cv": np.nan,
            "abnormal_rr_ratio": np.nan,
            "n_rr_total": 0,
            "n_rr_valid": 0
        }

    rr_ms = np.diff(rpeaks) / sampling_rate * 1000.0

    # Abnormal RR
    rr_valid_mask = (rr_ms >= 400) & (rr_ms <= 1500)
    rr_valid = rr_ms[rr_valid_mask]

    abnormal_rr_ratio = 1 - (len(rr_valid) / len(rr_ms))

    if len(rr_valid) < 3:
        return {
            "n_rpeaks": len(rpeaks),
            "mean_hr_bpm": np.nan,
            "hr_cv": np.nan, 
            "rmssd_ms": np.nan,
            "sdnn_ms": np.nan,
            "mean_rr_ms": np.nan,
            "rr_cv": np.nan,
            "abnormal_rr_ratio": abnormal_rr_ratio,
            "n_rr_total": len(rr_ms),
            "n_rr_valid": len(rr_valid)
        }

    mean_rr_ms = float(np.mean(rr_valid))
    mean_hr_bpm = float(60000.0 / mean_rr_ms)

    # RR_CV
    rr_sd_ms = float(np.std(rr_valid, ddof=1)) if len(rr_valid) >= 2 else np.nan
    rr_cv = float(rr_sd_ms / mean_rr_ms) if (mean_rr_ms > 0 and not np.isnan(rr_sd_ms)) else np.nan
    # HR series from valid RR
    hr_valid = 60000.0 / rr_valid
    # HR_CV
    hr_mean_bpm = float(np.mean(hr_valid))
    hr_sd_bpm = float(np.std(hr_valid, ddof=1)) if len(hr_valid) >= 2 else np.nan
    hr_cv = float(hr_sd_bpm / hr_mean_bpm) if (hr_mean_bpm > 0 and not np.isnan(hr_sd_bpm)) else np.nan

    diff_rr = np.diff(rr_valid)
    rmssd_ms = float(np.sqrt(np.mean(diff_rr ** 2))) if len(diff_rr) >= 1 else np.nan
    sdnn_ms = float(np.std(rr_valid, ddof=1)) if len(rr_valid) >= 2 else np.nan

    return {
        "n_rpeaks": len(rpeaks),
        "mean_hr_bpm": mean_hr_bpm,
        "hr_cv": hr_cv, 
        "rmssd_ms": rmssd_ms,
        "sdnn_ms": sdnn_ms,
        "mean_rr_ms": mean_rr_ms,
        "rr_cv": rr_cv,
        "abnormal_rr_ratio": abnormal_rr_ratio,
        "n_rr_total": len(rr_ms),
        "n_rr_valid": len(rr_valid)
    }


def judge_ecg_good(metrics):

    n_rpeaks = metrics["n_rpeaks"]
    hr = metrics["mean_hr_bpm"]
    rmssd = metrics["rmssd_ms"]
    abnormal_rr_ratio = metrics["abnormal_rr_ratio"]
    n_rr_valid = metrics["n_rr_valid"]

    if np.isnan(hr) or np.isnan(rmssd):
        return False

    if n_rpeaks < 100:
        return False

    if hr < 35 or hr > 130:
        return False

    if rmssd > 300:
        return False

    if abnormal_rr_ratio > 0.20:
        return False

    if n_rr_valid < 100:
        return False

    return True


def save_bad_ecg_plot(ecg_clean, rpeaks, out_png, sampling_rate=1024, seconds=30):

    n = min(len(ecg_clean), seconds * sampling_rate)
    ecg_seg = ecg_clean[:n]
    rpeaks_seg = rpeaks[rpeaks < n]

    plt.figure(figsize=(15, 4))
    plt.plot(ecg_seg, linewidth=1)
    if len(rpeaks_seg) > 0:
        plt.scatter(rpeaks_seg, ecg_seg[rpeaks_seg], color="red", s=20)
    plt.title(out_png.stem)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# =========================
# 2. Main
# =========================
def main():
    all_rows = []

    csv_1 = sorted(ECG_ROOT.glob("sub-*_*/OpenEye/ecg.csv"))
    csv_2 = sorted(ECG_ROOT.glob("sub-*_*/CloseEye/ecg.csv"))
    csv_3 = sorted(ECG_ROOT.glob("sub-*_*/Task/ecg.csv"))
    csv_files = csv_1 + csv_2 + csv_3

    print(f"Find ECG file: {len(csv_files)}")

    for csv_path in csv_files:
        try:
            sub, state = parse_sub_run(csv_path)
            condition = csv_path.parent.name

            if condition not in EYE_CONDITIONS:
                print(f"[Skip] No condition: {csv_path}")
                continue

            print(f"Preprocessing: {sub} | {state} | {condition}")

            # Read
            ecg = load_ecg_csv(csv_path)

            # Clean + R peaks
            ecg_clean, rpeaks = detect_rpeaks(
                ecg,
                sampling_rate=SAMPLING_RATE,
                clean_method=ECG_CLEAN_METHOD
            )

            # Evaluations
            metrics = compute_rr_metrics(rpeaks, sampling_rate=SAMPLING_RATE)

            # QC
            ecg_good = judge_ecg_good(metrics)

            row = {
                "sub": sub,
                "state": state,
                "condition": condition,
                "file": str(csv_path),
                "duration_sec": len(ecg) / SAMPLING_RATE,
                "ecg_good": ecg_good,
                **metrics
            }
            all_rows.append(row)

            # Save bad plot
            if SAVE_BAD_PLOTS and (not ecg_good):
                out_png = BAD_PLOT_DIR / f"{sub}_{state}_{condition}_bad.png"
                save_bad_ecg_plot(ecg_clean, rpeaks, out_png, sampling_rate=SAMPLING_RATE)

        except Exception as e:
            print(f"[Error] {csv_path}: {e}")
            all_rows.append({
                "sub": np.nan,
                "state": np.nan,
                "condition": np.nan,
                "file": str(csv_path),
                "duration_sec": np.nan,
                "ecg_good": False,
                "n_rpeaks": np.nan,
                "mean_hr_bpm": np.nan,
                "hr_cv": np.nan, 
                "rmssd_ms": np.nan,
                "sdnn_ms": np.nan,
                "mean_rr_ms": np.nan,
                "rr_cv": np.nan,
                "abnormal_rr_ratio": np.nan,
                "n_rr_total": np.nan,
                "n_rr_valid": np.nan,
                "error": str(e)
            })

    df = pd.DataFrame(all_rows)

    out_csv = OUT_DIR / "ecg_summary.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_csv}")

    df_good = df[df["ecg_good"] == True].copy()
    out_good_csv = OUT_DIR / "ecg_summary_good_only.csv"
    df_good.to_csv(out_good_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_good_csv}")

if __name__ == "__main__":
    main()
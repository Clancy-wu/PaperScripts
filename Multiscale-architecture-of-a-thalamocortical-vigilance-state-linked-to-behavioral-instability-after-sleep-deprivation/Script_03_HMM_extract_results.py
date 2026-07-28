from pathlib import Path
import re
import os
import pickle
import numpy as np
import pandas as pd

from osl_dynamics.data import Data
from osl_dynamics.models.hmm import Config, Model
from osl_dynamics.inference import modes

# =========================
# 1. Path
# =========================
DATA_DIR = "fMRI_HMM_24wmcfs03"   
OUT_DIR = Path("fMRI_HMM_24wmcfs03_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 2. Load data
# =========================
data = Data(DATA_DIR)

ts = data.time_series()
n_channels = ts[0].shape[1]
print("n_channels =", n_channels)

# =========================
# 3. HMM setting
# =========================
config = Config(
    n_states=10,
    n_channels=n_channels, 
    sequence_length=100,
    learn_means=True,
    learn_covariances=True,
    batch_size=64,
    learning_rate=1e-3,
    n_epochs=20,
)

# =========================
# 4. Modeling
# =========================
model = Model(config)

# =========================
# 5. Initial
# =========================
init_history = model.random_state_time_course_initialization(
    data,
    n_epochs=1,
    n_init=3
)

# =========================
# 6. training
# =========================
history = model.fit(data)

# =========================
# 7. Save model
# =========================
model_dir = OUT_DIR / "model"
model.save(str(model_dir))

free_energy = model.free_energy(data)
history["free_energy"] = free_energy

with open(OUT_DIR / "history.pkl", "wb") as f:
    pickle.dump(history, f)

print("Free energy:", free_energy)

pickle.dump(init_history, open(f"{model_dir}/init_history.pkl", "wb"))
pickle.dump(history, open(f"{model_dir}/history.pkl", "wb"))

# Inferred state probabilities
alp = model.get_alpha(data)

# HMM parameters
means, covs = model.get_means_covariances()
trans_prob = model.get_trans_prob()
initial_state_probs = model.get_initial_state_probs()

# Save
inf_params_dir = OUT_DIR / "inf_params"
os.makedirs(inf_params_dir, exist_ok=True)
pickle.dump(alp, open(f"{inf_params_dir}/alp.pkl", "wb"))
np.save(f"{inf_params_dir}/means.npy", means)
np.save(f"{inf_params_dir}/covs.npy", covs)
np.save(f"{inf_params_dir}/trans_prob.npy", trans_prob)
np.save(f"{inf_params_dir}/initial_state_probs.npy", initial_state_probs)

#### Calculate summary statistics
# State time course
stc = modes.argmax_time_courses(alp)

# =====================================
# 8. Calculate summary statistics
# =====================================
fo = modes.fractional_occupancies(stc)   # fractional occupancy
lt = modes.mean_lifetimes(stc)           # mean lifetime
intv = modes.mean_intervals(stc)         # mean interval
sr = modes.switching_rates(stc)          # switching rate

SUMMARY_STATS_DIR = OUT_DIR / "summary_stats"
os.makedirs(SUMMARY_STATS_DIR, exist_ok=True)

np.save(f"{SUMMARY_STATS_DIR}/fo.npy", fo)
np.save(f"{SUMMARY_STATS_DIR}/lt.npy", lt)
np.save(f"{SUMMARY_STATS_DIR}/intv.npy", intv)
np.save(f"{SUMMARY_STATS_DIR}/sr.npy", sr)

# =====================================
# 9. Acquire file name
# =====================================
sub_info = np.array([os.path.basename(i) for i in data.inputs])
np.save(f"{SUMMARY_STATS_DIR}/subjects_name.npy", sub_info)

# =====================================
# 10. Parse subject name and run (run-01=RW, run-02=SD)
# =====================================
def parse_sub_and_run(filename):
    base = os.path.splitext(filename)[0]
    sub_name = base.split('_')[0]
    run_label = base.split('_')[1]
    return sub_name, run_label

sub_name = []
sub_run = []
for fname in sub_info:
    s, r = parse_sub_and_run(fname)
    sub_name.append(s)
    sub_run.append(r)

# =====================================
# 11. export csv
# =====================================
n_rows, n_states = fo.shape

df = pd.DataFrame({
    "session": np.arange(1, n_rows + 1),
    "sub_name": sub_name,
    "run": sub_run,
    "condition": ["rest"] * n_rows
})

for state in range(n_states):
    df[f"fo_state{state+1}"] = fo[:, state]
    df[f"lt_state{state+1}"] = lt[:, state]
    df[f"intv_state{state+1}"] = intv[:, state]
    df[f"sr_state{state+1}"] = sr[:, state]

df.to_csv(f"{SUMMARY_STATS_DIR}/hmm_summary_stats_wide.csv", index=False)

print("Saved summary statistics csv:")
print(f"{SUMMARY_STATS_DIR}/hmm_summary_stats_wide.csv")

# =====================================
# 12. subject-level dual estimation
# =====================================
import os
import glob
import pickle
import numpy as np
from osl_dynamics.models import load
from osl_dynamics.data import Data
import shutil

def dual_estimation_by_file(
    input_dir='fMRI_HMM_24wmcfs03',
    model_dir='fMRI_HMM_24wmcfs03_results/model',
    alpha_path='fMRI_HMM_24wmcfs03_results/inf_params/alp.pkl',
    output_dir='fMRI_HMM_24wmcfs03_results/dual_estimates_subject',
    standardize=True,
    remove_tmp=True,
):
    os.makedirs(output_dir, exist_ok=True)

    model = load(model_dir)

    with open(alpha_path, "rb") as f:
        alpha = pickle.load(f)

    files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))

    if len(files) != len(alpha):
        raise ValueError(
            f"Error: Input file ({len(files)}) and alpha number ({len(alpha)}) are different. "
        )

    for i, file_path in enumerate(files):
        name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"Dual estimation: {i+1}/{len(files)} | {name}")

        tmp_dir = f"tmp_{name}"

        data_single = Data([file_path], store_dir=tmp_dir)

        if standardize:
            data_single.standardize()

        means, covs = model.dual_estimation(data_single, [alpha[i]])

        np.save(os.path.join(output_dir, f"{name}_means.npy"), means)
        np.save(os.path.join(output_dir, f"{name}_covs.npy"), covs)

        if remove_tmp and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print("Dual estimation finished.")

dual_estimation_by_file()

print('finished.')

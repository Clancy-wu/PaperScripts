from pathlib import Path
import re
import numpy as np
from nilearn.maskers import NiftiLabelsMasker
#############################
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
def run(f, this_iter):
    with ProcessPoolExecutor(max_workers=14) as executor:
        results = list(tqdm(executor.map(f, this_iter), total=len(this_iter)))
    return results
#############################
# =========================
# 1. Path
# =========================
CLEAN_DIR = Path("xcpd_24wmcfs03")
ATLAS_FILE = Path("/home/clancy/TemplateFlow/tpl-MNI152NLin2009cAsym/BN_Atlas246_2mm_tpl-MNI152NLin2009cAsym.nii.gz")
OUT_DIR = Path("fMRI_HMM_24wmcfs03")
OUT_DIR.mkdir(parents=True, exist_ok=True)

masker = NiftiLabelsMasker(
    labels_img=str(ATLAS_FILE),
    standardize=False,   
    detrend=False
)

# =========================
# 2. Find cleaned 4D files
# =========================
bold_files = sorted(CLEAN_DIR.glob("sub-*/func/sub-*_task-rest_run-*_space-MNI152NLin2009cAsym_res-2_desc-denoisedSmoothed_bold.nii.gz"))
print(f"Found {len(bold_files)} cleaned 4D files.")

# =========================
# 3. Extract ROI time series
# =========================
def generate_fmri_for_hmm(bf):

    sub_match = re.search(r"(sub-\d+)", bf.name)
    run_match = re.search(r"run-([A-Za-z0-9]+)", bf.name)

    if sub_match is None:
        raise ValueError(f"Cannot parse subject from filename: {bf.name}")

    sub = sub_match.group(1)
    run_org = run_match.group(1) if run_match else "run-unk"
    if run_org == '01':
        run = 'RW'
    elif run_org == '02':
        run = 'SD'

    # ROI time series: shape = (n_timepoints, n_rois)
    ts = masker.fit_transform(str(bf)).astype(np.float32)
    out_name = f"{sub}_{run}.npy"
    return np.save(OUT_DIR / out_name, ts)

future_results = run(generate_fmri_for_hmm, bold_files)
print('finished.')
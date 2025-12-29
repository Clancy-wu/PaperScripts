import os
import re
from glob import glob
import itertools
from nilearn import image, datasets, surface

#############################################
# The brain dysfunctional map for the two masks were computed in a public dataset, respectively.
#############################################
# 1. create normative fzmap from GSP 1,000 subjects
seed_files = ['meta_mask/deprivation_association-test_z_FDR_0.01_mask_fsl.nii.gz',
              'meta_mask/deprivation_uniformity-test_z_FDR_0.01_mask_fsl.nii.gz'] # fsl
all_subs = glob(f'/home/clancy/data/GSP1000/GSP1000_v2_dataset/sub-*') # 1000
out_dirs = ['association_gsp1000_fzmap', 'uniformity_gsp1000_fzmap']
my_items = list(itertools.product(seed_files, all_subs, out_dirs))
run(compute_seed_fzmap_batch, my_items) # parallel running with 30 cores

## 2. combine fzmap
association_fzmap = glob(f'association_gsp1000_fzmap/*.nii.gz')
uniformity_fzmap = glob(f'uniformity_gsp1000_fzmap/*.nii.gz')
image.concat_imgs(association_fzmap).to_filename('association_gsp1000_fzmap.nii')
image.concat_imgs(uniformity_fzmap).to_filename('uniformity_gsp1000_fzmap.nii')
### randomize test under 1,000 times permutation tests
os.system('randomise -i association_gsp1000_fzmap.nii -o association_OneSampT -1 -T -n 1000')
os.system('randomise -i uniformity_gsp1000_fzmap.nii -o uniformity_OneSampT -1 -T -n 1000')

#############################################
# Optimizing the brain dysfunctional map by merging the two lesions. 
#############################################
import pandas as pd
import numpy as np
fsaverage = datasets.load_fsaverage('fsaverage5')
def ss_scale(arr):
    # standardize scale
    scaled = (arr - arr.mean()) / arr.std()
    return scaled

def create_weight_map_from_pvalue_tvalue(pvalue_file, tvalue_file):
    pvalue_img = surface.SurfaceImage.from_volume(fsaverage["pial"], pvalue_file, inner_mesh=fsaverage["white_matter"]) # 20484
    pvalue_lh = pvalue_img.data.parts['left']; pvalue_rh = pvalue_img.data.parts['right']

    tvalue_img = surface.SurfaceImage.from_volume(fsaverage["pial"], tvalue_file, inner_mesh=fsaverage["white_matter"]) # 20484
    tvalue_lh = tvalue_img.data.parts['left']; tvalue_rh = tvalue_img.data.parts['right']
    tvalue_lh_sig = np.zeros_like(tvalue_lh); tvalue_rh_sig = np.zeros_like(tvalue_rh)
    tvalue_lh_sig[pvalue_lh > 0.999] = tvalue_lh[pvalue_lh > 0.999]
    tvalue_rh_sig[pvalue_rh > 0.999] = tvalue_rh[pvalue_rh > 0.999]
    data = {"left": tvalue_lh_sig, "right": tvalue_rh_sig}
    return data

association_data = create_weight_map_from_pvalue_tvalue(
    pvalue_file = 'association_OneSampT_tfce_corrp_tstat1.nii.gz', 
    tvalue_file = 'association_OneSampT_tstat1.nii.gz'
)
uniformity_data = create_weight_map_from_pvalue_tvalue(
    pvalue_file = 'uniformity_OneSampT_tfce_corrp_tstat1.nii.gz', 
    tvalue_file = 'uniformity_OneSampT_tstat1.nii.gz'
)

both_mask_data_org = association_data * uniformity_data
both_mask_data_mask = np.where(both_mask_data_org > 0, 1, 0)
both_mask_data_sum = association_data + uniformity_data
both_mask_data = np.zeros_like(both_mask_data_sum)
both_mask_data[both_mask_data_mask == 1] = ss_scale(both_mask_data_sum[both_mask_data_mask == 1]) # standard scale
# remove medial-wall
medial_wall_lh = pd.read_csv('/home/clancy/TemplateFlow/ENIGMA_surface/fsa5_lh_mask.csv', header=None)
medial_wall_rh = pd.read_csv('/home/clancy/TemplateFlow/ENIGMA_surface/fsa5_rh_mask.csv', header=None)
medial_wall = np.append(medial_wall_lh[0].values, medial_wall_rh[0].values) 
both_mask_data[medial_wall==0] = 0
pd.DataFrame(both_mask_data, columns=['weight']).to_csv('association_uniformity_corrp_sig0.001_sscale_weight_hemi-both.csv', index=None)

#############################################
# Result 1: 
#       Validations of ALFF and TASK in local participants
#############################################
def alff_compute_with_weight(alff_file):
    both_weight_file = pd.read_csv('association_uniformity_corrp_sig0.001_sscale_weight_hemi-both.csv')
    both_weight_data = both_weight_file['weight'].values

    alff_img = surface.SurfaceImage.from_volume(fsaverage["pial"], alff_file, inner_mesh=fsaverage["white_matter"]) # 20484
    alff_data = np.append(alff_img.data.parts['left'], alff_img.data.parts['right']) 
    sub_alff_mean = np.mean(alff_data * both_weight_data)
    # info
    sub_name = re.findall(r'(sub-\d+)_task', os.path.basename(alff_file))[0]
    sub_run = re.findall(r'run-(\d+)_space', os.path.basename(alff_file))[0]
    return np.append(np.array([sub_name, sub_run]), sub_alff_mean)

def task_compute_wtih_weight(task_file):
    both_weight_file = pd.read_csv('association_uniformity_corrp_sig0.001_sscale_weight_hemi-both.csv')
    both_weight_data = both_weight_file['weight'].values

    task_img = surface.SurfaceImage.from_volume(fsaverage["pial"], task_file, inner_mesh=fsaverage["white_matter"]) # 20484
    task_data = np.append(task_img.data.parts['left'], task_img.data.parts['right']) 
    sub_task_mean = np.mean(task_data * both_weight_data)
    # info
    sub_name = re.findall(r'(sub-\d+)_task', os.path.basename(task_file))[0]
    sub_run = re.findall(r'run-(\d+)_con', os.path.basename(task_file))[0]
    return np.append(np.array([sub_name, sub_run]), sub_task_mean)

alff_dir = '/home/clancy/ssd/SleepDisfunction/xcpd_rest_nifti/sub-*/func/sub-*_task-rest_run-*_space-MNI152NLin2009cAsym_res-2_stat-alff_desc-smooth_boldmap.nii.gz'
alff_files = glob(alff_dir) # 56

task_dir = '/home/clancy/ssd/SleepDisfunction/STWMdata/AvgActive/sub-*_task-rdm_run-*_con_z.nii.gz'
task_files = glob(task_dir) # 56

run(alff_compute_with_weight, alff_files)
run(task_compute_wtih_weight, task_files)

#############################################
# Result 2:
#       Architectures of the weight map
#############################################
# 1. & 2. cortical regions and functional networks
weight_map_file = pd.read_csv('association_uniformity_corrp_sig0.001_sscale_weight_hemi-both.csv')
weight_map = weight_map_file['weight'].values # (20484,)
bn_lh_file = 'lh.fs5.BN_Atlas.label.gii'
bn_lh = surface.load_surf_data(bn_lh_file)
bn_rh_file = 'rh.fs5.BN_Atlas.label.gii'
bn_rh = surface.load_surf_data(bn_rh_file)
bn_both = np.append(bn_lh, bn_rh) # (20484,)
bn_both = bn_both.astype(np.int16)

result_index = []; result_mean = []
for i in range(1, 211):
    result_index.append(i)
    result_mean.append(np.mean(weight_map[bn_both==i]))
df = pd.DataFrame({'bn_label': result_index, 'value': result_mean})
df = df.sort_values('value')
df.to_csv('weight_map_sort_by_bn.csv', index=None)

# 3. associations between neural transmitters and the weight map
from neuromaps import images, nulls, stats
weight_map = pd.read_csv('association_uniformity_corrp_sig0.001_sscale_weight_hemi-both.csv')
weight_data = weight_map['weight'].values
rotated = nulls.alexander_bloch(weight_data, atlas='fsaverage', density='10k',
                                n_perm=5000, seed=1234)
weight_mask = np.where(weight_data==0, 0, 1)
def permutation_cor_neurotransmitter(one_map):
    ## map info
    one_map_info = one_map.split('/')
    ## map_file
    map_lh = glob(f'{one_map}/*hemi-L*.gii')
    map_rh = glob(f'{one_map}/*hemi-R*.gii')
    if len(map_lh) == 1 and len(map_rh) == 1:
        map_lh_data = images.load_data(map_lh)
        map_rh_data = images.load_data(map_rh)
        map_both_data = np.append(map_lh_data, map_rh_data)
        map_both_data_masked = map_both_data * weight_mask # multiply with the weight map
        corr, pval = stats.compare_images(map_both_data_masked, weight_data, nulls=rotated)
        return one_map_info[-4], one_map_info[-3], one_map_info[-2], corr, pval
    else:
        pass

all_maps = glob(f'/home/clancy/neuromaps-data/annotations_fsaverage5/*/*/*')
run(permutation_cor_neurotransmitter, all_maps)

# 4. associations between neural transmitters and ALFF in individuals 
def compute_alff_diff_from_subject(sub_name):
    sub_run1 = f'/home/clancy/ssd/SleepDisfunction/xcpd_rest_nifti/{sub_name}/func/{sub_name}_task-rest_run-01_space-MNI152NLin2009cAsym_res-2_stat-alff_desc-smooth_boldmap.nii.gz'
    sub_run2 = f'/home/clancy/ssd/SleepDisfunction/xcpd_rest_nifti/{sub_name}/func/{sub_name}_task-rest_run-02_space-MNI152NLin2009cAsym_res-2_stat-alff_desc-smooth_boldmap.nii.gz'

    alff_run1 = surface.SurfaceImage.from_volume(fsaverage["pial"], sub_run1, inner_mesh=fsaverage["white_matter"]) # 20484
    alff_run1_data = np.append(alff_run1.data.parts['left'], alff_run1.data.parts['right']) 

    alff_run2 = surface.SurfaceImage.from_volume(fsaverage["pial"], sub_run2, inner_mesh=fsaverage["white_matter"]) # 20484
    alff_run2_data = np.append(alff_run2.data.parts['left'], alff_run2.data.parts['right']) 

    all_maps = glob(f'/home/clancy/neuromaps-data/annotations_fsaverage5/*/*/*') 

    alff_diff = alff_run1_data - alff_run2_data
    weight_data = alff_diff * weight_map # multiply with the weight map
    rotated = nulls.alexander_bloch(weight_data, atlas='fsaverage', density='10k',
                                n_perm=5000, seed=1234)
    weight_mask = np.where(weight_data==0, 0, 1)
    map_result = []
    for one_map in all_maps:
        one_result = permutation_cor_neurotransmitter(one_map, weight_mask, weight_data, rotated, sub_name)
        map_result.append(one_result)
    sub_df = pd.DataFrame(map_result, columns=['annotation', 'source', 'desc', 'r', 'p', 'subject'])
    return sub_df

run(compute_alff_diff_from_subject, all_subs)

#############################################
# Result 3:
#       Subcortical regions connecting to the weight map
#############################################
# 1. Correlations between ALFF changes in the weight map and subcortical nuclei
def extract_subregion_value(bold_file):
    bn_atlas = '/home/clancy/TemplateFlow/tpl-MNI152NLin2009cAsym/BN_Atlas246_2mm_tpl-MNI152NLin2009cAsym.nii.gz'
    bn_data = image.get_data(bn_atlas)
    bold_data = image.get_data(bold_file)
    bold_data = bold_data[:, :, :, 0]

    mAmyg = np.mean(bold_data[ (bn_data==211) | (bn_data==212)]) # 166 + 222 = 388
    lAmyg = np.mean(bold_data[ (bn_data==213) | (bn_data==214)])
    rHipp = np.mean(bold_data[ (bn_data==215) | (bn_data==216)])
    cHipp = np.mean(bold_data[ (bn_data==217) | (bn_data==218)])
    vCa = np.mean(bold_data[ (bn_data==219) | (bn_data==220)])
    GP = np.mean(bold_data[ (bn_data==221) | (bn_data==222)])
    NAC = np.mean(bold_data[ (bn_data==223) | (bn_data==224)])
    vmPu = np.mean(bold_data[ (bn_data==225) | (bn_data==226)])
    dCa = np.mean(bold_data[ (bn_data==227) | (bn_data==228)])
    dlPu = np.mean(bold_data[ (bn_data==229) | (bn_data==230)])
    mPFtha = np.mean(bold_data[ (bn_data==231) | (bn_data==232)])
    mPMtha = np.mean(bold_data[ (bn_data==233) | (bn_data==234)])
    Stha = np.mean(bold_data[ (bn_data==235) | (bn_data==236)])
    rTtha = np.mean(bold_data[ (bn_data==237) | (bn_data==238)])
    PPtha = np.mean(bold_data[ (bn_data==239) | (bn_data==240)])
    Otha = np.mean(bold_data[ (bn_data==241) | (bn_data==242)])
    cTtha = np.mean(bold_data[ (bn_data==243) | (bn_data==244)])
    lPFtha = np.mean(bold_data[ (bn_data==245) | (bn_data==246)])

    alff_map = surface.SurfaceImage.from_volume(fsaverage["pial"], bold_file, inner_mesh=fsaverage["white_matter"]) # 20484
    alff_map_data = np.append(alff_map.data.parts['left'], alff_map.data.parts['right']) # 20484
    weight_data = alff_map_data * weight_map # multiply with the weight map 
    weigth_mean = np.mean(weight_data)

    sub_name = re.findall(r'(sub-\d+)_task', os.path.basename(bold_file))[0]
    sub_run = re.findall(r'run-(\d+)_space', os.path.basename(bold_file))[0]

    # 18 subregions 
    return sub_name, sub_run, weigth_mean, mAmyg, lAmyg, rHipp, cHipp, vCa, GP, NAC, vmPu, dCa, dlPu, mPFtha, mPMtha, Stha, rTtha, PPtha, Otha, cTtha, lPFtha

weight_map_file = pd.read_csv('association_uniformity_corrp_sig0.001_sscale_weight_hemi-both.csv')
weight_map = weight_map_file['weight'].values 
all_bold_files = glob(f'/home/clancy/ssd/SleepDisfunction/xcpd_rest_nifti/*/func/*_task-rest_run-*_space-MNI152NLin2009cAsym_res-2_stat-alff_desc-smooth_boldmap.nii.gz')
run(extract_subregion_value, all_bold_files)

# 2. Causal correlations between time series of the weight map and subcortical nuclei
# based on granger analysis
# granger analysis


import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy import stats

def _build_lag_matrix_1d(series: np.ndarray, p: int) -> np.ndarray:
    """
    Build lag matrix for a 1D series with guaranteed alignment to target series[p:].
    Returns matrix of shape (T-p, p) with columns [lag1, lag2, ..., lagp],
    where lagk column is series[t-k] for t = p..T-1.
    """
    series = np.asarray(series).ravel()
    T = series.shape[0]
    if p < 1:
        raise ValueError("p must be >= 1")
    if T <= p:
        raise ValueError(f"Time series too short (T={T}) for p={p}")

    # For each lag k, take series[p-k : T-k]
    cols = [series[p - k : T - k] for k in range(1, p + 1)]
    X = np.column_stack(cols)  # (T-p, p)
    return X


def granger_gc_with_adf_bic(
    x,
    y,
    p_max=5,
    ic="bic",
    adf_alpha=0.05,
    adf_autolag="AIC",
    adf_regression="c",
):
    """
    Granger causality (X -> Y) on preprocessed time series with:
      (1) ADF stationarity checks (reported, not used for exclusion),
      (2) lag order selection via a single information criterion (default: BIC),
      (3) nested OLS GC test outputting ΔR², F, p-value.

    Parameters
    ----------
    x, y : array-like, shape (T,)
        Preprocessed 1D time series of equal length.
    p_max : int, default=5
        Maximum lag order to consider in IC search (lags 1..p_max).
    ic : {"bic","aic"}, default="bic"
        Information criterion used for lag selection.
    adf_alpha : float, default=0.05
        Threshold for reporting ADF stationarity.
    adf_autolag : {"AIC","BIC","t-stat", None}, default="AIC"
        Autolag strategy used by adfuller.
    adf_regression : {"c","ct","ctt","n"}, default="c"
        Deterministic terms in ADF regression ("c" = constant).

    Returns
    -------
    dict with ADF report, selected p, IC table, ΔR², F, p-value, dfs, etc.
    """
    # --- input checks ---
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays")
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    T = len(y)

    if p_max < 1:
        raise ValueError("p_max must be >= 1")
    if T <= (p_max + 2):
        raise ValueError(f"Time series too short (T={T}) for p_max={p_max}")

    ic = ic.lower()
    if ic not in ("bic", "aic"):
        raise ValueError("ic must be 'bic' or 'aic'")

    # --- (1) ADF stationarity checks (reported) ---
    adf_x = adfuller(x, regression=adf_regression, autolag=adf_autolag)
    adf_y = adfuller(y, regression=adf_regression, autolag=adf_autolag)
    adf_report = {
        "x": {
            "stat": float(adf_x[0]),
            "p_value": float(adf_x[1]),
            "used_lag": int(adf_x[2]),
            "nobs": int(adf_x[3]),
            "is_stationary_at_alpha": bool(adf_x[1] < adf_alpha),
        },
        "y": {
            "stat": float(adf_y[0]),
            "p_value": float(adf_y[1]),
            "used_lag": int(adf_y[2]),
            "nobs": int(adf_y[3]),
            "is_stationary_at_alpha": bool(adf_y[1] < adf_alpha),
        },
        "alpha": float(adf_alpha),
        "regression": adf_regression,
        "autolag": adf_autolag,
    }

    # --- helper: fit restricted/unrestricted for a given p (guaranteed alignment) ---
    def _fit_models_for_p(p: int):
        lagged_y = _build_lag_matrix_1d(y, p)  # (T-p, p)
        lagged_x = _build_lag_matrix_1d(x, p)  # (T-p, p)
        Y = y[p:]                              # (T-p,)

        # Guaranteed alignment checks
        if lagged_y.shape[0] != Y.shape[0] or lagged_x.shape[0] != Y.shape[0]:
            raise RuntimeError(
                f"Lag/target misalignment even after manual construction: "
                f"Y={Y.shape}, lagged_y={lagged_y.shape}, lagged_x={lagged_x.shape}"
            )

        X_r = sm.add_constant(lagged_y, has_constant="add")
        X_u = sm.add_constant(np.hstack([lagged_y, lagged_x]), has_constant="add")

        model_r = sm.OLS(Y, X_r).fit()
        model_u = sm.OLS(Y, X_u).fit()
        return Y, X_r, X_u, model_r, model_u

    # --- (2) Select lag order via IC using unrestricted model ---
    ic_table = []
    best_p = None
    best_ic = np.inf

    for p in range(1, p_max + 1):
        Y, X_r, X_u, model_r, model_u = _fit_models_for_p(p)
        ic_value = float(model_u.bic if ic == "bic" else model_u.aic)
        ic_table.append({"p": int(p), ic.upper(): ic_value, "n_obs": int(Y.shape[0])})

        if ic_value < best_ic:
            best_ic = ic_value
            best_p = p

    if best_p is None:
        raise RuntimeError("Failed to select lag order; check inputs.")

    # --- (3) Final GC test at selected lag ---
    Y, X_r, X_u, model_r, model_u = _fit_models_for_p(best_p)

    R2_r = float(model_r.rsquared)
    R2_u = float(model_u.rsquared)
    delta_R2 = R2_u - R2_r

    ssr_r = float(model_r.ssr)
    ssr_u = float(model_u.ssr)

    q = X_u.shape[1] - X_r.shape[1]  # number of lagged-X terms (should equal p)
    df_num = int(q)
    df_den = int(model_u.df_resid)

    F = ((ssr_r - ssr_u) / df_num) / (ssr_u / df_den)
    p_value = float(stats.f.sf(F, df_num, df_den))  # stable tail prob

    return {
        "adf": adf_report,
        "ic_used": ic.upper(),
        "p_selected": int(best_p),
        "ic_table": ic_table,
        "R2_restricted": R2_r,
        "R2_unrestricted": R2_u,
        "delta_R2": float(delta_R2),
        "ssr_restricted": ssr_r,
        "ssr_unrestricted": ssr_u,
        "F": float(F),
        "p_value": p_value,
        "df_num": df_num,
        "df_den": df_den,
        "n_obs": int(Y.shape[0]),
        "n_reg_restricted": int(X_r.shape[1]),
        "n_reg_unrestricted": int(X_u.shape[1]),
    }


def extract_subregion_value(bold_file):
    bn_atlas = '/home/clancy/TemplateFlow/tpl-MNI152NLin2009cAsym/BN_Atlas246_2mm_tpl-MNI152NLin2009cAsym.nii.gz'
    bn_data = image.get_data(bn_atlas)
    bold_data = image.get_data(bold_file) # (97, 115, 97, x)

    mAmyg=[]; lAmyg=[]; rHipp=[]; cHipp=[]; vCa=[]; GP=[]; NAC=[]; vmPu=[]; dCa=[];  
    dlPu=[]; mPFtha=[]; mPMtha=[]; Stha=[]; rTtha=[]; PPtha=[]; Otha=[]; cTtha=[]; lPFtha=[]
    for i in range(bold_data.shape[3]):
        mAmyg.append(np.mean(bold_data[:, :, :, i][ (bn_data==211) | (bn_data==212)])) # 166 + 222 = 388
        lAmyg.append(np.mean(bold_data[:, :, :, i][ (bn_data==213) | (bn_data==214)]))
        rHipp.append(np.mean(bold_data[:, :, :, i][ (bn_data==215) | (bn_data==216)]))
        cHipp.append(np.mean(bold_data[:, :, :, i][ (bn_data==217) | (bn_data==218)]))
        vCa.append(np.mean(bold_data[:, :, :, i][ (bn_data==219) | (bn_data==220)]))
        GP.append(np.mean(bold_data[:, :, :, i][ (bn_data==221) | (bn_data==222)]))
        NAC.append(np.mean(bold_data[:, :, :, i][ (bn_data==223) | (bn_data==224)]))
        vmPu.append(np.mean(bold_data[:, :, :, i][ (bn_data==225) | (bn_data==226)]))
        dCa.append(np.mean(bold_data[:, :, :, i][ (bn_data==227) | (bn_data==228)]))
        dlPu.append(np.mean(bold_data[:, :, :, i][ (bn_data==229) | (bn_data==230)]))
        mPFtha.append(np.mean(bold_data[:, :, :, i][ (bn_data==231) | (bn_data==232)]))
        mPMtha.append(np.mean(bold_data[:, :, :, i][ (bn_data==233) | (bn_data==234)]))
        Stha.append(np.mean(bold_data[:, :, :, i][ (bn_data==235) | (bn_data==236)]))
        rTtha.append(np.mean(bold_data[:, :, :, i][ (bn_data==237) | (bn_data==238)]))
        PPtha.append(np.mean(bold_data[:, :, :, i][ (bn_data==239) | (bn_data==240)]))
        Otha.append(np.mean(bold_data[:, :, :, i][ (bn_data==241) | (bn_data==242)]))
        cTtha.append(np.mean(bold_data[:, :, :, i][ (bn_data==243) | (bn_data==244)]))
        lPFtha.append(np.mean(bold_data[:, :, :, i][ (bn_data==245) | (bn_data==246)]))
    
    surf_map = surface.SurfaceImage.from_volume(fsaverage["pial"], bold_file, inner_mesh=fsaverage["white_matter"]) # 20484
    alff_map_data = np.concatenate((surf_map.data.parts['left'], surf_map.data.parts['right']), axis=0) # [20484, 223]
    weight_data = alff_map_data * weight_map[:, np.newaxis]
    weigth_mean = weight_data.mean(axis=0)

    # ganger analysis: p=10, lag order=10
    # only focus on 10 significant subcortical regions
    # dlPu=[]; vmPu=[]; PPtha=[]; cHipp=[]; GP=[]; Stha=[]; dCa=[]; cTtha=[]; Otha=[]; vCa=[];
    dlPu_map = granger_gc_with_adf_bic(np.array(dlPu), weigth_mean) 
    map_dlPu = granger_gc_with_adf_bic(weigth_mean, np.array(dlPu)) 
    vmPu_map = granger_gc_with_adf_bic(np.array(vmPu), weigth_mean) 
    map_vmPu = granger_gc_with_adf_bic(weigth_mean, np.array(vmPu)) 
    PPtha_map = granger_gc_with_adf_bic(np.array(PPtha), weigth_mean) 
    map_PPtha = granger_gc_with_adf_bic(weigth_mean, np.array(PPtha)) 
    cHipp_map = granger_gc_with_adf_bic(np.array(cHipp), weigth_mean) 
    map_cHipp = granger_gc_with_adf_bic(weigth_mean, np.array(cHipp)) 
    GP_map = granger_gc_with_adf_bic(np.array(GP), weigth_mean) 
    map_GP = granger_gc_with_adf_bic(weigth_mean, np.array(GP)) 
    Stha_map = granger_gc_with_adf_bic(np.array(Stha), weigth_mean) 
    map_Stha = granger_gc_with_adf_bic(weigth_mean, np.array(Stha)) 
    dCa_map = granger_gc_with_adf_bic(np.array(dCa), weigth_mean) 
    map_dCa = granger_gc_with_adf_bic(weigth_mean, np.array(dCa)) 
    cTtha_map = granger_gc_with_adf_bic(np.array(cTtha), weigth_mean) 
    map_cTtha = granger_gc_with_adf_bic(weigth_mean, np.array(cTtha)) 
    Otha_map = granger_gc_with_adf_bic(np.array(Otha), weigth_mean) 
    map_Otha = granger_gc_with_adf_bic(weigth_mean, np.array(Otha)) 
    vCa_map = granger_gc_with_adf_bic(np.array(vCa), weigth_mean) 
    map_vCa = granger_gc_with_adf_bic(weigth_mean, np.array(vCa)) 

    # infomation
    sub_name = re.findall(r'(sub-\d+)_task', os.path.basename(bold_file))[0]
    sub_run = re.findall(r'run-(\d+)_space', os.path.basename(bold_file))[0]
    sub_information = dict(subject=sub_name, run=sub_run)

    #   vCa=[];
    all_dict = {'information': sub_information, 
                'dlPu_map': dlPu_map, 'map_dlPu': map_dlPu, 
                'vmPu_map': vmPu_map, 'map_vmPu': map_vmPu, 
                'PPtha_map': PPtha_map, 'map_PPtha': map_PPtha,
                'cHipp_map': cHipp_map, 'map_cHipp': map_cHipp, 
                'GP_map': GP_map, 'map_GP': map_GP, 
                'Stha_map': Stha_map, 'map_Stha': map_Stha, 
                'dCa_map': dCa_map, 'map_dCa': map_dCa, 
                'cTtha_map': cTtha_map, 'map_cTtha': map_cTtha, 
                'Otha_map': Otha_map, 'map_Otha': map_Otha, 
                'vCa_map': vCa_map, 'map_vCa': map_vCa
                }

    return np.savez(f'granger_analysis_results_revise/{sub_name}_run-{sub_run}.npz', **all_dict)

# granger analysis
weight_map_file = pd.read_csv('association_uniformity_corrp_sig0.001_sscale_weight_hemi-both.csv')
weight_map = weight_map_file['weight'].values # (20484,)
all_bold_files = glob(f'/home/clancy/ssd/SleepDisfunction/xcpd_rest_nifti/*/func/*_task-rest_run-*_space-MNI152NLin2009cAsym_res-2_desc-denoisedSmoothed_bold.nii.gz')
run(extract_subregion_value, all_bold_files)
print('finished.')


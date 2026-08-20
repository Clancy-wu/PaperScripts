import os
import re
from glob import glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

dual_regression_dir = 'fMRI_HMM_24wmcfs03_results/dual_estimates_subject'
group_template = np.load('fMRI_HMM_24wmcfs03_results/inf_params/means.npy')
group_state = group_template[2, : ] # state 3
pvt = pd.read_csv('PVT_results/PVT_result.csv')

##############################
#### RAS based on spearman correaltion defination. 
import numpy as np
from scipy.stats import pearsonr, spearmanr

def extract_test(sub_file, state_num):
    sub_data = np.load(sub_file)
    sub_data_state = sub_data[ state_num-1, : ] # state3
    z = np.arctanh(spearmanr(group_state, sub_data_state)[0]) # RAS and fisherz transform
    ## basic info
    sub_name = re.findall(r"(sub-\d+)_", os.path.basename(sub_file))[0]
    sub_run = re.findall(r"_(\w+)_", os.path.basename(sub_file))[0]
    return sub_name, sub_run, z
sub_files = sorted(glob(f'{dual_regression_dir}/sub-*_*_means.npy'))
sub_data = []
for i in sub_files:
    sub_data.append(extract_test(i, state_num=3))
sub_df = pd.DataFrame(sub_data, columns=['subject', 'run', 'cor_z'])
sub_df = sub_df.sort_values('subject')
sub_df.to_csv('subjects_RSA.csv', index=None)

pearsonr(
    sub_df[sub_df['run']=='SD']['cor_z'].values - sub_df[sub_df['run']=='RW']['cor_z'].values, 
     pvt[pvt['run']=='SD']['rt_cv'].values - pvt[pvt['run']=='RW']['rt_cv'].values
) # 

############################################
## Similarity between TVS and SD-CWM based on spin-rotation permutation test
import pandas as pd
import numpy as np
import nibabel as nib
from scipy.stats import pearsonr, spearmanr
from neuromaps import nulls, parcellate, stats

parcellation=('Data/lh.fs5.BN_Atlas_fix.label.gii', 'Data/rh.fs5.BN_Atlas_fix.label.gii')
bn_atlas = parcellate.Parcellater(parcellation=parcellation, space='fsaverage').fit()
SD_CWM_data = pd.read_csv('SD-CWM.csv')
SD_CWM_vertex = np.asarray(SD_CWM_data['weight'].values, dtype=float)
SD_CWM = bn_atlas.transform(SD_CWM_vertex, space='fsaverage')

# -----------------------------
# 1. Define valid ROI mask and compute Pearson correlation
# -----------------------------
TVS_data = pd.read_csv('all_dynamic_brain_states/state_3_mean.txt', header=None)
TVS = np.asarray(TVS_data[0].values, dtype=float)[:210]
mask = (TVS != 0) & (SD_CWM != 0)
pearsonr(TVS[mask], SD_CWM[mask]) # raw value, r=-0.23, p=0.001
pearsonr(np.abs(TVS[mask]), np.abs(SD_CWM[mask])) # abs value, r=0.24, p<0.001

# -----------------------------
# 3. Generate spin nulls
# -----------------------------
## raw value
rotated = nulls.baum(SD_CWM, atlas='fsaverage', density='10k',
                                n_perm=5000, seed=1234, parcellation=parcellation)
raw_compare = stats.compare_images(SD_CWM, TVS, nulls=rotated, return_nulls=True) 
print(raw_compare) # r=-0.23, p=0.042
np.savez('TVS_SDCWM_SpinCorrelation_RawValueResult.npz', 
         r_value=raw_compare[0], spin_p_value=raw_compare[1], n_perm=5000, null_values=raw_compare[2])

## abs value
abs_rotated = nulls.baum(np.abs(SD_CWM), atlas='fsaverage', density='10k',
                                n_perm=5000, seed=1234, parcellation=parcellation)
abs_compare = stats.compare_images(np.abs(SD_CWM), np.abs(TVS), nulls=abs_rotated, return_nulls=True) 
print(abs_compare) # r=0.24, p=0.043
np.savez('TVS_SDCWM_SpinCorrelation_AbsValueResult.npz', 
         r_value=abs_compare[0], spin_p_value=abs_compare[1], n_perm=1000, null_values=abs_compare[2])

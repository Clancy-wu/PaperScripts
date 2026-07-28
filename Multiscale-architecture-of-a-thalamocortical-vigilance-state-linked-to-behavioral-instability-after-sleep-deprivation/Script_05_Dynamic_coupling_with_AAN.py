#############################
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
def run(f, this_iter):
    with ProcessPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(f, this_iter), total=len(this_iter)))
    return results
#############################

import os
import re
from glob import glob
import numpy as np
from nilearn import image
from nilearn.maskers import NiftiLabelsMasker
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA

def similarity_value(A, B): 
    # Representation Similarity Analysis
    # r to z
    return np.array([ np.arctanh(spearmanr(row, B).statistic) for row in A ])

def dynamic_coupling_state_aan(sub_file):
    result_dir = 'fMRI_state_bilateral_aan_coupling'
    os.makedirs(result_dir, exist_ok=True)

    sub_name = re.findall(r'(sub-\d+)_task', os.path.basename(sub_file))[0] # save
    sub_run = re.findall(r'run-(\d+)_space', os.path.basename(sub_file))[0] # save
    if sub_run=='02': 
        sub_condition = 'SD'
    elif sub_run=='01':
        sub_condition = 'RW'
    else:
        raise ValueError(f"Input file error.")

    brain_state = np.load('fMRI_HMM_24wmcfs03_results/inf_params/means.npy') # group brain state, [10, 246]

    bn_atlas_file = '/home/clancy/TemplateFlow/tpl-MNI152NLin2009cAsym/BN_Atlas246_2mm_tpl-MNI152NLin2009cAsym.nii.gz'
    bn_masker = NiftiLabelsMasker(labels_img=bn_atlas_file, background_label=0, mask_img=None, 
                                smoothing_fwhm=None, standardize=False, standardize_confounds=False, 
                                high_variance_confounds=False, resampling_target='data', strategy='mean')
    bn_time_data = bn_masker.fit_transform(sub_file) # (time point, 246)
    similar_state1 = similarity_value(bn_time_data, brain_state[0, :])
    similar_state2 = similarity_value(bn_time_data, brain_state[1, :])
    similar_state3 = similarity_value(bn_time_data, brain_state[2, :])
    similar_state4 = similarity_value(bn_time_data, brain_state[3, :])
    similar_state5 = similarity_value(bn_time_data, brain_state[4, :])
    similar_state6 = similarity_value(bn_time_data, brain_state[5, :])
    similar_state7 = similarity_value(bn_time_data, brain_state[6, :])
    similar_state8 = similarity_value(bn_time_data, brain_state[7, :])
    similar_state9 = similarity_value(bn_time_data, brain_state[8, :])
    similar_state10 = similarity_value(bn_time_data, brain_state[9, :])

    aan_file = 'atlas/AAN_bilateral_Brainstem_MNI152-a009c_1mm_res-2mm.nii.gz'
    aan_masker = NiftiLabelsMasker(labels_img=aan_file, background_label=0, mask_img=None, 
                                smoothing_fwhm=None, standardize=False, standardize_confounds=False, 
                                high_variance_confounds=False, resampling_target='data', strategy='mean')
    aan_time_data = aan_masker.fit_transform(sub_file) 

    pca = PCA(n_components=1)
    pc1_score = pca.fit_transform(aan_time_data).ravel()

    file_out = f'{result_dir}/{sub_name}_{sub_condition}_dynamic_coupling.npz'

    return np.savez(file_out, 
                    sub_name = sub_name, 
                    sub_condition = sub_condition, 
                    similar_state1 = similar_state1, 
                    similar_state2 = similar_state2, 
                    similar_state3 = similar_state3, 
                    similar_state4 = similar_state4, 
                    similar_state5 = similar_state5, 
                    similar_state6 = similar_state6, 
                    similar_state7 = similar_state7, 
                    similar_state8 = similar_state8, 
                    similar_state9 = similar_state9, 
                    similar_state10 = similar_state10, 
                    aan_signal = aan_time_data, 
                    aan_pc1 = pc1_score
                    )

sub_files = sorted(glob(f'/home/clancy/ssd/SD_BrainHear_xcpd/xcpd_24wmcfs03/sub-*/func/sub-*_task-rest_run-*_space-MNI152NLin2009cAsym_res-2_desc-denoisedSmoothed_bold.nii.gz'))
results = run(dynamic_coupling_state_aan, sub_files)
print('finished.')

#############################
# Extract data
import numpy as np
import pandas as pd
from glob import glob
from scipy.stats import pearsonr, spearmanr, zscore

def dynamic_coupling(time_A, time_B):
    A_z = zscore(time_A, nan_policy="omit")
    B_z = zscore(time_B, nan_policy="omit")
    cofluct = A_z * B_z
    return np.nanmean(cofluct)

def extract_dynamic_coupling(sub_file):
    sub_data = np.load(sub_file)
    sub_name = sub_data['sub_name']
    sub_condition = sub_data['sub_condition']
    state1 = dynamic_coupling(sub_data['similar_state1'], sub_data['aan_pc1'])
    state2 = dynamic_coupling(sub_data['similar_state2'], sub_data['aan_pc1'])
    state3 = dynamic_coupling(sub_data['similar_state3'], sub_data['aan_pc1'])
    state4 = dynamic_coupling(sub_data['similar_state4'], sub_data['aan_pc1'])
    state5 = dynamic_coupling(sub_data['similar_state5'], sub_data['aan_pc1'])
    state6 = dynamic_coupling(sub_data['similar_state6'], sub_data['aan_pc1'])
    state7 = dynamic_coupling(sub_data['similar_state7'], sub_data['aan_pc1'])
    state8 = dynamic_coupling(sub_data['similar_state8'], sub_data['aan_pc1'])
    state9 = dynamic_coupling(sub_data['similar_state9'], sub_data['aan_pc1'])
    state10 = dynamic_coupling(sub_data['similar_state10'], sub_data['aan_pc1'])

    return sub_name, sub_condition, state1, state2, state3, state4, state5, state6, state7, state8, state9, state10

all_files = sorted(glob(f'fMRI_state_bilateral_aan_coupling/*.npz'))
future_results = run(extract_dynamic_coupling, all_files)
df = pd.DataFrame(future_results, 
                  columns=['subject', 'run', 'state1_dcmean', 'state2_dcmean', 'state3_dcmean',
                           'state4_dcmean', 'state5_dcmean', 'state6_dcmean', 'state7_dcmean', 
                           'state8_dcmean', 'state9_dcmean',  'state10_dcmean' ])
df.to_csv('Dynamic_coupling/TVS_AAN_DynamicCoupling_results.csv', index=None)

from scipy.stats import ttest_1samp, ttest_rel
import statsmodels.formula.api as smf
df = pd.read_csv('Dynamic_coupling/TVS_AAN_DynamicCoupling_results.csv')
ttest_1samp(df[df['run']=='RW']['state3_dcmean'].values, 0) # p=0.432
ttest_1samp(df[df['run']=='SD']['state3_dcmean'].values, 0) # p<0.001
ttest_rel(df[df['run']=='RW']['state3_dcmean'].values, df[df['run']=='SD']['state3_dcmean'].values) # p=0.042

#########################################################################################3
#### different ROIs
import numpy as np
import pandas as pd
from glob import glob
from scipy.stats import pearsonr, spearmanr, zscore

def dynamic_coupling(time_A, time_B):
    A_z = zscore(time_A, nan_policy="omit")
    B_z = zscore(time_B, nan_policy="omit")
    cofluct = A_z * B_z
    return np.nanmean(cofluct)

def extract_dynamic_coupling_ROI(sub_file):
    sub_data = np.load(sub_file)
    sub_name = sub_data['sub_name']
    sub_condition = sub_data['sub_condition']

    # roi1: 7601-LC, roi2: 7602-LDTg, roi3: 7603-mRt, roi4: 7604-PBC, roi5: 7605-PnO, roi6: 7606-PTg
    roi1 = dynamic_coupling(sub_data['similar_state3'], sub_data['aan_signal'][:, 0])
    roi2 = dynamic_coupling(sub_data['similar_state3'], sub_data['aan_signal'][:, 1])
    roi3 = dynamic_coupling(sub_data['similar_state3'], sub_data['aan_signal'][:, 2])
    roi4 = dynamic_coupling(sub_data['similar_state3'], sub_data['aan_signal'][:, 3])
    roi5 = dynamic_coupling(sub_data['similar_state3'], sub_data['aan_signal'][:, 4])
    roi6 = dynamic_coupling(sub_data['similar_state3'], sub_data['aan_signal'][:, 5])

    return sub_name, sub_condition, roi1, roi2, roi3, roi4, roi5, roi6

all_files = sorted(glob(f'fMRI_state_bilateral_aan_coupling/*.npz'))
future_results = run(extract_dynamic_coupling_ROI, all_files)
df = pd.DataFrame(future_results, 
                  columns=['subject', 'run', 'LC', 'LDTg', 'mRt',
                           'PBC', 'PnO', 'PTg' ])
df.to_csv('Dynamic_coupling/TVS_AAN_ROIs_results.csv', index=None)

################################################################################################
## independent LC template
import numpy as np
import pandas as pd
from glob import glob
from scipy.stats import pearsonr, spearmanr, zscore

def similarity_value(A, B): 
    # Representation Similarity Analysis
    # r to z
    return np.array([ np.arctanh(spearmanr(row, B).statistic) for row in A ])

def dynamic_coupling_state3_LC(sub_file):

    sub_name = re.findall(r'(sub-\d+)_task', os.path.basename(sub_file))[0] # save
    sub_run = re.findall(r'run-(\d+)_space', os.path.basename(sub_file))[0] # save
    if sub_run=='02': 
        sub_condition = 'SD'
    elif sub_run=='01':
        sub_condition = 'RW'
    else:
        raise ValueError(f"Input file error.")

    brain_state = np.load('fMRI_HMM_24wmcfs03_results/inf_params/means.npy') # group brain state, [10, 246]

    bn_atlas_file = '/home/clancy/TemplateFlow/tpl-MNI152NLin2009cAsym/BN_Atlas246_2mm_tpl-MNI152NLin2009cAsym.nii.gz'
    bn_masker = NiftiLabelsMasker(labels_img=bn_atlas_file, background_label=0, mask_img=None, 
                                smoothing_fwhm=None, standardize=False, standardize_confounds=False, 
                                high_variance_confounds=False, resampling_target='data', strategy='mean')
    bn_time_data = bn_masker.fit_transform(sub_file) # (time point, 246)

    similar_state3 = similarity_value(bn_time_data, brain_state[2, :])

    lc_file = 'atlas/LCmetaMask_MNI152a2009c_s01f_plus50_2mm.nii.gz'
    lc_masker = NiftiLabelsMasker(labels_img=lc_file, background_label=0, mask_img=None, 
                                smoothing_fwhm=None, standardize=False, standardize_confounds=False, 
                                high_variance_confounds=False, resampling_target='data', strategy='mean')
    lc_time_data = lc_masker.fit_transform(sub_file).ravel()
    A_z = zscore(similar_state3, nan_policy="omit")
    B_z = zscore(lc_time_data, nan_policy="omit")

    return sub_name, sub_condition, np.nanmean(A_z * B_z)

sub_files = sorted(glob(f'/home/clancy/ssd/SD_BrainHear_xcpd/xcpd_24wmcfs03/sub-*/func/sub-*_task-rest_run-*_space-MNI152NLin2009cAsym_res-2_desc-denoisedSmoothed_bold.nii.gz'))
future_results = run(dynamic_coupling_state3_LC, sub_files)
df = pd.DataFrame(future_results, columns=['subject', 'run', 'value'])
#df.to_csv('TVS_LC_results.csv', index=None)
df = pd.read_csv('Dynamic_coupling/TVS_LC_results.csv')

ttest_1samp(df[df['run']=='RW']['value'].values, 0) # p=0.447
ttest_1samp(df[df['run']=='SD']['value'].values, 0) # p=0.0138
ttest_rel(df[df['run']=='RW']['value'].values, df[df['run']=='SD']['value'].values) 
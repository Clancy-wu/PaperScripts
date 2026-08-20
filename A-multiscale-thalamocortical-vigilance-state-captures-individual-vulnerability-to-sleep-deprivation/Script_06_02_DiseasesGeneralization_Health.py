# X-axis：TVS–AAN mean co-fluctuation
# 
# Y-axis：TVS pattern amplitude
## sqrt(mean(similarity(t)^2))

#############################
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
def run(f, this_iter):
    with ProcessPoolExecutor(max_workers=12) as executor:
        results = list(tqdm(executor.map(f, this_iter), total=len(this_iter)))
    return results
#############################
import os
import re
from glob import glob
import numpy as np
from nilearn.maskers import NiftiLabelsMasker
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA

def similarity_value(A, B): 
    # Representation Similarity Analysis
    # r to z
    return np.array([ np.arctanh(spearmanr(row, B).statistic) for row in A ])

def generate_sub_data_GSP(sub_bold):
    
    try:
        brain_state = np.load('fMRI_HMM_24wmcfs03_results/inf_params/means.npy') # group brain state, [10, 246]
        bn_atlas_file = '/home/clancy/TemplateFlow/tpl-MNI152NLin2009cAsym/BN_Atlas246_2mm_tpl-MNI152NLin2009cAsym.nii.gz'
        bn_masker = NiftiLabelsMasker(labels_img=bn_atlas_file, background_label=0, mask_img=None, 
                                    smoothing_fwhm=None, standardize=False, standardize_confounds=False, 
                                    high_variance_confounds=False, resampling_target='data', strategy='mean')
        bn_time_data = bn_masker.fit_transform(sub_bold) # (time point, 246)
        similar_state3 = similarity_value(bn_time_data, brain_state[2, :])

        aan_file = 'atlas/AAN_bilateral_Brainstem_MNI152-a009c_1mm_res-2mm.nii.gz'
        aan_masker = NiftiLabelsMasker(labels_img=aan_file, background_label=0, mask_img=None, 
                                    smoothing_fwhm=None, standardize=False, standardize_confounds=False, 
                                    high_variance_confounds=False, resampling_target='data', strategy='mean')
        aan_time_data = aan_masker.fit_transform(sub_bold) 
        pca = PCA(n_components=1)
        pc1_score = pca.fit_transform(aan_time_data).ravel()

        sub_name = re.findall(r'(sub-\d+)_', os.path.basename(sub_bold))[0]
        result_dir = f'MultipleDiseases/Health'
        os.makedirs(result_dir, exist_ok=True)
        sub_npz = f'{sub_name}_TVS-AAN_TimeData.npz'
        return np.savez(f'{result_dir}/{sub_npz}', similar_state3 = similar_state3, aan_pc1 = pc1_score)
    except:
        pass

subs_bold = sorted(glob(f'/home/clancy/data/GSP1000/GSP1000_v2_dataset/sub-*/func/*bld001*.nii.gz'))
run(generate_sub_data_GSP, subs_bold)

print('finished.')

##### SD
def generate_sub_data_SD(sub_bold):
    
    try:
        brain_state = np.load('fMRI_HMM_24wmcfs03_results/inf_params/means.npy') # group brain state, [10, 246]
        bn_atlas_file = '/home/clancy/TemplateFlow/tpl-MNI152NLin2009cAsym/BN_Atlas246_2mm_tpl-MNI152NLin2009cAsym.nii.gz'
        bn_masker = NiftiLabelsMasker(labels_img=bn_atlas_file, background_label=0, mask_img=None, 
                                    smoothing_fwhm=None, standardize=False, standardize_confounds=False, 
                                    high_variance_confounds=False, resampling_target='data', strategy='mean')
        bn_time_data = bn_masker.fit_transform(sub_bold) # (time point, 246)
        similar_state3 = similarity_value(bn_time_data, brain_state[2, :])

        aan_file = 'atlas/AAN_bilateral_Brainstem_MNI152-a009c_1mm_res-2mm.nii.gz'
        aan_masker = NiftiLabelsMasker(labels_img=aan_file, background_label=0, mask_img=None, 
                                    smoothing_fwhm=None, standardize=False, standardize_confounds=False, 
                                    high_variance_confounds=False, resampling_target='data', strategy='mean')
        aan_time_data = aan_masker.fit_transform(sub_bold) 
        pca = PCA(n_components=1)
        pc1_score = pca.fit_transform(aan_time_data).ravel()

        sub_name = re.findall(r'(sub-\d+)_', os.path.basename(sub_bold))[0]
        sub_run = re.findall(r'run-(.*)_space', os.path.basename(sub_bold))[0]
        
        if sub_run == '01':
            result_dir = f'MultipleDiseases/RW'
        elif sub_run == '02':
            result_dir = f'MultipleDiseases/SD'
        else:
            ValueError('error file input.')

        os.makedirs(result_dir, exist_ok=True)
        sub_npz = f'{sub_name}_TVS-AAN_TimeData.npz'
        return np.savez(f'{result_dir}/{sub_npz}', similar_state3 = similar_state3, aan_pc1 = pc1_score)
    except:
        pass

subs_bold = sorted(glob(f'/home/clancy/ssd/SleepDisfunction/xcpd_rest_gsr_03fd/sub-*/func/sub-*_task-rest_run-*_space-MNI152NLin2009cAsym_res-2_desc-denoisedSmoothed_bold.nii.gz'))
run(generate_sub_data_SD, subs_bold)
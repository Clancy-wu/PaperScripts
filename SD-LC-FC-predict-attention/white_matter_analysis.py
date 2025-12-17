import os
import re
from glob import glob
import numpy as np
import pandas as pd
from nilearn import image

# extract brainstem 
atlas = '/usr/local/fsl/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr50-1mm.nii.gz'
atlas_data = image.get_data(atlas)
new_data = np.zeros_like(atlas_data)
new_data[atlas_data == 8] = 1
image.new_img_like(ref_niimg=atlas, data=new_data).to_filename('brainstem_fsl.nii.gz')

# transfer brainstem from fsl_mni to a2009s_mni
input_file = 'brainstem_fsl.nii.gz' # 91x109x91
interpolation_para = 'NearestNeighbor'
output_file = 'brainstem_a2009c.nii.gz'
reference_file = 'tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz'
transform_file = 'tpl-MNI152NLin2009cAsym_from-MNI152NLin6Asym_mode-image_xfm.h5'
os.system(f'antsApplyTransforms --default-value 0 \
                                --float 0 \
                                --input {input_file} \
                                --interpolation {interpolation_para} \
                                --output {output_file} \
                                --reference-image {reference_file} \
                                --transform {transform_file}')
image.resample_to_img(source_img='brainstem_a2009c.nii.gz', 
                      target_img='tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz', 
                      interpolation='nearest', ).to_filename('brainstem_a2009c_2mm.nii.gz')

# extract white matter
atlas = '/usr/local/fsl/data/atlases/JHU/JHU-ICBM-labels-1mm.nii.gz'
atlas_data = image.get_data(atlas)
new_data = np.zeros_like(atlas_data)
new_data[atlas_data > 0] = 1
image.new_img_like(ref_niimg=atlas, data=new_data).to_filename('whitematter_fsl.nii.gz')
# transfer brainstem from fsl_mni to a2009s_mni
input_file = 'whitematter_fsl.nii.gz' # 91x109x91
interpolation_para = 'NearestNeighbor'
output_file = 'whitematter_a2009c.nii.gz'
reference_file = 'tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz'
transform_file = 'tpl-MNI152NLin2009cAsym_from-MNI152NLin6Asym_mode-image_xfm.h5'
os.system(f'antsApplyTransforms --default-value 0 \
                                --float 0 \
                                --input {input_file} \
                                --interpolation {interpolation_para} \
                                --output {output_file} \
                                --reference-image {reference_file} \
                                --transform {transform_file}')
image.resample_to_img(source_img='whitematter_a2009c.nii.gz', 
                      target_img='tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz', 
                      interpolation='nearest', ).to_filename('whitematter_a2009c_2mm.nii.gz')

from pingouin import partial_corr
def regressin_covariance_fzvalue(sub_bold):
    # sub_bold is from xcpd_rest_nongsr_05fd
    # atlas signals
    wm_mask = '/home/clancy/ssd/SD_LC_MultiResult/white_matter/whitematter_a2009c_2mm.nii.gz'
    brainstem_mask = '/home/clancy/ssd/SD_LC_MultiResult/brainstem_a2009c_2mm.nii.gz'
    lc_mask = '/home/clancy/TemplateFlow/LC_Atlas/LC_MNI152a2009c/LCmetaMask_MNI152a2009c_2mm.nii.gz'
    tha_mask = '/home/clancy/ssd/SD_LC/l_tha_radiu-6.nii.gz'
    # fz value
    bold_signal = image.get_data(sub_bold) # (97, 115, 97, 223)
    wm_data = image.get_data(wm_mask) # (97, 115, 97)
    brainstem_data = image.get_data(brainstem_mask) 
    lc_data = image.get_data(lc_mask)
    tha_data = image.get_data(tha_mask) 

    wm_value = bold_signal[wm_data == 1].mean(axis=0)
    brainstem_value = bold_signal[brainstem_data == 1].mean(axis=0)
    lc_value = bold_signal[lc_data == 1].mean(axis=0)
    tha_value = bold_signal[tha_data == 1].mean(axis=0)

    df = pd.DataFrame({
        'wm_value': wm_value, 'brainstem_value': brainstem_value, 'lc_value': lc_value, 'tha_value': tha_value
    })
    wm_res = partial_corr(data=df, x='lc_value', y='tha_value', covar='wm_value', method='pearson')
    wm_fz = np.arctanh(wm_res['r'].iloc[0])
    brainstem_res = partial_corr(data=df, x='lc_value', y='tha_value', covar='brainstem_value', method='pearson')
    brainstem_fz = np.arctanh(brainstem_res['r'].iloc[0])   

    # basic information
    sub_name = re.findall(r'(sub-\d+)_task-rest', os.path.basename(sub_bold))[0]
    sub_run = re.findall(r'run-(\d+)_space', os.path.basename(sub_bold))[0]

    return sub_name, sub_run, wm_fz, brainstem_fz

# run-01: RW, run-02: SD
all_subs = glob(f'/home/clancy/ssd/SleepDisfunction/xcpd_rest_nongsr_05fd/sub-*/func/sub-*_task-rest_run-*_space-MNI152NLin2009cAsym_res-2_desc-denoisedSmoothed_bold.nii.gz')
future_results = run(regressin_covariance_fzvalue, all_subs)
print('finished.')
df = pd.DataFrame(future_results, columns=['subject', 'run', 'wm_fz', 'brainstem_fz'])
df = df.sort_values(['subject', 'run' ])
df.to_csv('white_matter_results.csv', index=None)
# end. Author@KangWU

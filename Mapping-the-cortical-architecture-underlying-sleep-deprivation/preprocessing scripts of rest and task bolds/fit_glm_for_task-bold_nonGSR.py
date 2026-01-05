import os
import re
from glob import glob
import pandas as pd
import numpy as np
from nilearn import image
from nilearn.glm.first_level import FirstLevelModel
## to ignore UserWarning in the FirstLevelModel fitting because the time series have been centered in fmriprep.
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#############################
# 1. prepare atlas
mask_img = '/home/clancy/TemplateFlow/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz'

def compute_glm_results(sub_file):
    sub_name = re.findall(r'(sub-.*)_task-rdm', os.path.basename(sub_file))[0]
    sub_run = re.findall(r'task-rdm_(run.*)_space', os.path.basename(sub_file))[0]
    sub_motion_file = sub_file.replace('space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz', 'desc-confounds_timeseries.tsv')
    sub_motion = pd.read_csv(sub_motion_file, sep='\s+')
    sub_motion = sub_motion.fillna(0)
    column_24P = [
        'trans_x', 'trans_x_derivative1', 'trans_x_derivative1_power2', 'trans_x_power2',
        'trans_y', 'trans_y_derivative1', 'trans_y_derivative1_power2', 'trans_y_power2',
        'trans_z', 'trans_z_derivative1', 'trans_z_derivative1_power2', 'trans_z_power2',
        'rot_x', 'rot_x_derivative1', 'rot_x_power2', 'rot_x_derivative1_power2',
        'rot_y', 'rot_y_derivative1', 'rot_y_power2', 'rot_y_derivative1_power2',
        'rot_z', 'rot_z_derivative1', 'rot_z_power2', 'rot_z_derivative1_power2',
    ]
    sub_motion_24P = sub_motion.loc[:, column_24P]
    #####################################################
    ##    Avg response
    #####################################################
    ## load time event: run-01 is rested wakefulness (RW), run-02 is sleep deprivation (SD)
    if sub_run == 'run-01':
        time_event_file = f'STWMdata_nonGSR/STWM_time_event/{sub_name}/Before/time_event_avg_removeTenTimePoint.csv'
    elif sub_run == 'run-02':
        time_event_file = f'STWMdata_nonGSR/STWM_time_event/{sub_name}/After/time_event_avg_removeTenTimePoint.csv'
    else:
        pass
    ## do GLM
    t_r = 2.02
    time_event = pd.read_csv(time_event_file)
    sub_glm = FirstLevelModel(t_r=t_r, slice_time_ref=0.5, hrf_model="spm", 
                               high_pass=1/128, mask_img=mask_img, smoothing_fwhm=6,
                               noise_model='ar1', random_state=42)
    sub_glm.fit(run_imgs=sub_file, events=time_event, confounds=sub_motion_24P)
    sub_MemoryControl = sub_glm.compute_contrast("memory - control", output_type='z_score')
    sub_MemoryControl_out = f'STWMdata_nonGSR/AvgActive/{sub_name}_task-rdm_{sub_run}_con_z.nii.gz'
    os.makedirs(os.path.dirname(sub_MemoryControl_out), exist_ok=True)
    sub_MemoryControl.to_filename(sub_MemoryControl_out)
    #####################################################
    ##    Digit response
    #####################################################    
    ## load time event 
    if sub_run == 'run-01':
        time_event_file = f'STWMdata_nonGSR/STWM_time_event/{sub_name}/Before/time_event_digit_removeTenTimePoint.csv'
    elif sub_run == 'run-02':
        time_event_file = f'STWMdata_nonGSR/STWM_time_event/{sub_name}/After/time_event_digit_removeTenTimePoint.csv'
    else:
        pass
    ## do GLM
    t_r = 2.02
    time_event = pd.read_csv(time_event_file)
    sub_glm = FirstLevelModel(t_r=t_r, slice_time_ref=0.5, hrf_model="spm", 
                               high_pass=1/128, mask_img=mask_img, smoothing_fwhm=6,
                               noise_model='ar1', random_state=42)
    sub_glm.fit(run_imgs=sub_file, events=time_event, confounds=sub_motion_24P)
    #### NumOne
    sub_NumOne = sub_glm.compute_contrast("NumOne - Control", output_type='z_score')
    sub_NumOne_out = f'STWMdata_nonGSR/NumOne/{sub_name}_task-rdm_{sub_run}_con_z.nii.gz'
    os.makedirs(os.path.dirname(sub_NumOne_out), exist_ok=True)
    sub_NumOne.to_filename(sub_NumOne_out)
    #### NumThree
    sub_NumThree = sub_glm.compute_contrast("NumThree - Control", output_type='z_score')
    sub_NumThree_out = f'STWMdata_nonGSR/NumThree/{sub_name}_task-rdm_{sub_run}_con_z.nii.gz'
    os.makedirs(os.path.dirname(sub_NumThree_out), exist_ok=True)
    sub_NumThree.to_filename(sub_NumThree_out)
    #### NumSix
    sub_NumSix = sub_glm.compute_contrast("NumSix - Control", output_type='z_score')
    sub_NumSix_out = f'STWMdata_nonGSR/NumSix/{sub_name}_task-rdm_{sub_run}_con_z.nii.gz'
    os.makedirs(os.path.dirname(sub_NumSix_out), exist_ok=True)
    sub_NumSix.to_filename(sub_NumSix_out)
    return 0

## fit with all files
all_files = glob(f'fmriprep/sub-*/func/sub-*_task-rdm_run-*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')
for file in all_files:
    compute_glm_results(file)
print('finished.')
## end.

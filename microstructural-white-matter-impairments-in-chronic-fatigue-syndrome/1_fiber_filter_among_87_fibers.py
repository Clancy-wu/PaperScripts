import os
import re
from glob import glob
from nilearn import image, maskers
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

#############################################################
compare_dir = 'Results_mni/FA'
all_tracks = glob('/home/clancy/TemplateFlow/HCP_YA1065_tractography_all_tracks_trk/hcp1065_avg_tracts_nifti_2mm/*/*.nii.gz')
clinic_info = pd.read_csv('clinical_info_200.csv')
#############################################################
def compute_result(file, track):
    file_name = re.findall(r'(sub-sub\d+)_space', os.path.basename(file))[0]
    file_data = image.get_data(file)
    track_data = image.get_data(track)
    value = np.mean(file_data[track_data==1])
    value_logit = np.log(value / (1 - value))
    return file_name, value_logit
def compute_result_batch(args):
    return compute_result(*args)

def track_comparison_two_group(track, compare_dir):
    files = glob(f'{compare_dir}/*.nii.gz')
    my_list = list(itertools.product(files, [track]))
    future_results = run(compute_result_batch, my_list)
    my_result = pd.DataFrame(future_results, columns=['subject', 'value'])
    my_dataset = pd.merge(
        my_result,
        clinic_info[['participant_id', 'group', 'age', 'gender', 'BMI']],
        left_on='subject', right_on='participant_id', how='left'
    )
    # Ordinary Least Squares regression 普通最小二乘回归
    model = smf.ols('value ~ group + age + gender + BMI', data=my_dataset).fit()
    p_group = model.pvalues['group[T.patient]']
    t_group = model.tvalues['group[T.patient]']
    
    track_info = track.split('/')
    return track_info[-2], track_info[-1], p_group, t_group

track_results = []
for track in all_tracks:
    track_result = track_comparison_two_group(track, compare_dir)
    track_results.append(track_result)

df = pd.DataFrame(track_results, columns=['TrackType', 'TrackMask', 'Pvalue', 'Tvalue'])
df.to_csv('FA_results_OLSRtest.csv', index=None)

#############################################################
# get value
compare_dir = 'Results_mni/FA'

# C_FPH_L and C_PO_L
track = '/home/clancy/TemplateFlow/HCP_YA1065_tractography_all_tracks_trk/hcp1065_avg_tracts_nifti_2mm/association/C_PO_L.nii.gz'
clinic_info = pd.read_csv('clinical_info_200.csv')

def compute_result(file, track):
    file_name = re.findall(r'(sub-sub\d+)_space', os.path.basename(file))[0]
    file_data = image.get_data(file)
    track_data = image.get_data(track)
    value = np.mean(file_data[track_data==1])
    value_logit = np.log(value / (1 - value))
    return file_name, value_logit
def compute_result_batch(args):
    return compute_result(*args)

files = glob(f'{compare_dir}/*.nii.gz')
my_list = list(itertools.product(files, [track]))
future_results = run(compute_result_batch, my_list)
my_result = pd.DataFrame(future_results, columns=['subject', 'value'])
my_result.to_csv('FA_C_PO_L_results.csv', index=None)


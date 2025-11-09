import os
import re
from glob import glob
import pandas as pd
import numpy as np
from dipy.io.image import load_nifti
from dipy.io.streamline import load_trk
import dipy.stats.analysis as dsa
 
#############################################################
# C_FPH_L and C_PO_L
mni_track_file = 'C_PO_L.trk' 
measure_dir = '/home/clancy/data/BigData/CFS_baseline/Results_mni/FA'
segment_points = 50

#############################################################
mni_track = load_trk(mni_track_file, reference="same", bbox_valid_check=False).streamlines
# split to 50 nodes and make weights
track_weight = dsa.gaussian_weights(mni_track, n_points=segment_points)
all_measures = glob(f'{measure_dir}/sub-sub*.nii.gz')
def extrack_values(mni_track, track_weight, measure_file, segment_points):
    img_data, img_affine = load_nifti(measure_file)
    file_name = re.findall(r'(sub-sub\d+)_space', os.path.basename(measure_file))[0]
    value = dsa.afq_profile(img_data, mni_track, affine=img_affine, n_points=segment_points, weights=track_weight)
    value_logit = np.log(value / (1 - value))
    return np.append(file_name, value_logit)
def extrack_values_batch(args):
    return extrack_values(*args)

my_list = list(itertools.product([mni_track], [track_weight], all_measures, [segment_points]))
future_results = run(extrack_values_batch, my_list)
col_name = ['subject'] + ['seg_'+str(i) for i in range(1, segment_points+1)]
my_result = pd.DataFrame(future_results, columns=col_name)
my_result.to_csv(f'fa_{mni_track_file}_segment_50_points.csv', index=None)

#############################################
clinic_info = pd.read_csv('clinical_info_200.csv')
my_result = pd.read_csv(f'fa_{mni_track_file}_segment_50_points.csv')
col_name = ['subject'] + ['seg_'+str(i) for i in range(1, segment_points+1)]
my_dataset = pd.merge(
        my_result,
        clinic_info[['participant_id', 'group', 'age', 'gender', 'BMI']],
        left_on='subject', right_on='participant_id', how='left'
    )

import statsmodels.formula.api as smf
seg_name=[]; Pvalue=[]; Tvalue=[]
for seg_i in col_name[1:]:
    model = smf.ols(f'{seg_i} ~ group + age + gender + BMI', data=my_dataset).fit()
    p_group = model.pvalues['group[T.patient]']
    t_group = model.tvalues['group[T.patient]']
    seg_name.append(seg_i)
    Pvalue.append(p_group)
    Tvalue.append(t_group)
df = pd.DataFrame({
    'seg_name':seg_name,
    'Pvalue':Pvalue,
    'Tvalue':Tvalue
})
df.to_csv(f'fa_{mni_track_file}_segment_50_points_results.csv', index=None)
# end.



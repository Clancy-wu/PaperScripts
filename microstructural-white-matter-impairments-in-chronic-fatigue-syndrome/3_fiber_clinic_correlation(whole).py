import os
import re
import pingouin as pg
import pandas as pd
from scipy.stats import pearsonr, spearmanr

fa_seg = pd.read_csv('fa_C_FPH_L_segment_50_points.csv')
clinic_info = pd.read_csv('clinical_info_200.csv')
clinic_info['gender'] = clinic_info['gender'].replace({'female': 1, 'male': 0})

##################################################
df = pd.merge(fa_seg, clinic_info, left_on='subject', right_on='participant_id', how='left')
df = df[df['FS-14'] > 0]

def compute_each_seg(seg_name):

    t_1, p_1 = pearsonr(df[seg_name].values, df['FS-14'].values)
    t_2, p_2 = spearmanr(df[seg_name].values, df['FS-14'].values)
    result_3 = pg.partial_corr(data=df, x=seg_name, y='FS-14', covar=['age', 'gender', 'BMI'], method='pearson')
    t_3 = result_3['r'].values[0]; p_3 = result_3['p-val'].values[0]
    result_4 = pg.partial_corr(data=df, x=seg_name, y='FS-14', covar=['age', 'gender', 'BMI'], method='spearman')
    t_4 = result_4['r'].values[0]; p_4 = result_4['p-val'].values[0]

    return seg_name, t_1, p_1, t_2, p_2, t_3, p_3, t_4, p_4

all_segs = ['seg_'+str(i) for i in range(1, 51)]
my_results = []
for seg_name in all_segs:
    seg_result = compute_each_seg(seg_name)
    my_results.append(seg_result)

col_name = ['seg_name', 't_1', 'p_1', 't_2', 'p_2', 't_3', 'p_3', 't_4', 'p_4' ]
df = pd.DataFrame(my_results, columns=col_name)
df.to_csv('FPH_whole_segment_50_points_correlation.csv', index=None)

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
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.tsatools import lagmat
def granger_delta_r2(x, y, p):
    """
    Compute Granger test, ΔR² and F-test for X -> Y with lag p.
    x, y : 1D numpy arrays of same length T
    p : lag order (integer >= 1)
    Returns dict with R2_restricted, R2_unrestricted, delta_R2, F, p_value, df_denom, df_num
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    T = len(y)
    if p < 1:
        raise ValueError("p must be >= 1")

    lagged_y = lagmat(y, maxlag=p, trim='both')  
    lagged_x = lagmat(x, maxlag=p, trim='both')

    Y = y[p:]   # length = T - p

    # Create design matrices
    # Restricted: constant + lagged_y
    X_r = sm.add_constant(lagged_y)
    # Unrestricted: constant + lagged_y + lagged_x
    X_u = sm.add_constant(np.hstack([lagged_y, lagged_x]))
    # Fit OLS on same Y
    model_r = sm.OLS(Y, X_r).fit()
    model_u = sm.OLS(Y, X_u).fit()
    # R² values
    R2_r = model_r.rsquared
    R2_u = model_u.rsquared
    delta_R2 = R2_u - R2_r
    # Compute F-stat (manually) for the joint test that coefficients of lagged_x are zero
    # Use SSR difference and degrees of freedom
    ssr_r = model_r.ssr
    ssr_u = model_u.ssr
    df_r = model_r.df_resid
    df_u = model_u.df_resid
    q = X_u.shape[1] - X_r.shape[1]  # number of restrictions = number of lagged_x columns
    df_num = q
    df_den = int(df_u)   # residual dof for unrestricted
    F = ((ssr_r - ssr_u) / q) / (ssr_u / df_den)
    from scipy import stats
    p_value = 1 - stats.f.cdf(F, df_num, df_den)

    return dict(
        R2_restricted=R2_r,
        R2_unrestricted=R2_u,
        delta_R2=delta_R2,
        ssr_restricted=ssr_r,
        ssr_unrestricted=ssr_u,
        F=F,
        p_value=p_value,
        df_num=df_num,
        df_den=df_den,
        n_obs = Y.shape[0],
        n_reg_restricted = X_r.shape[1],
        n_reg_unrestricted = X_u.shape[1]
    )

def compute_granger_results_from_file(bold_file):
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
    dlPu_map = granger_delta_r2(np.array(dlPu), weigth_mean, p=10) 
    map_dlPu = granger_delta_r2(weigth_mean, np.array(dlPu), p=10) 
    vmPu_map = granger_delta_r2(np.array(vmPu), weigth_mean, p=10) 
    map_vmPu = granger_delta_r2(weigth_mean, np.array(vmPu), p=10) 
    PPtha_map = granger_delta_r2(np.array(PPtha), weigth_mean, p=10) 
    map_PPtha = granger_delta_r2(weigth_mean, np.array(PPtha), p=10) 
    cHipp_map = granger_delta_r2(np.array(cHipp), weigth_mean, p=10) 
    map_cHipp = granger_delta_r2(weigth_mean, np.array(cHipp), p=10) 
    GP_map = granger_delta_r2(np.array(GP), weigth_mean, p=10) 
    map_GP = granger_delta_r2(weigth_mean, np.array(GP), p=10) 
    Stha_map = granger_delta_r2(np.array(Stha), weigth_mean, p=10) 
    map_Stha = granger_delta_r2(weigth_mean, np.array(Stha), p=10) 
    dCa_map = granger_delta_r2(np.array(dCa), weigth_mean, p=10) 
    map_dCa = granger_delta_r2(weigth_mean, np.array(dCa), p=10) 
    cTtha_map = granger_delta_r2(np.array(cTtha), weigth_mean, p=10) 
    map_cTtha = granger_delta_r2(weigth_mean, np.array(cTtha), p=10) 
    Otha_map = granger_delta_r2(np.array(Otha), weigth_mean, p=10) 
    map_Otha = granger_delta_r2(weigth_mean, np.array(Otha), p=10) 
    vCa_map = granger_delta_r2(np.array(vCa), weigth_mean, p=10) 
    map_vCa = granger_delta_r2(weigth_mean, np.array(vCa), p=10) 

    # infomation
    sub_name = re.findall(r'(sub-\d+)_task', os.path.basename(bold_file))[0]
    sub_run = re.findall(r'run-(\d+)_space', os.path.basename(bold_file))[0]
    sub_information = dict(subject=sub_name, run=sub_run)

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

    return np.savez(f'granger_analysis_results/{sub_name}_run-{sub_run}.npz', **all_dict)

all_bold_files = glob(f'/home/clancy/ssd/SleepDisfunction/xcpd_rest_nifti/*/func/*_task-rest_run-*_space-MNI152NLin2009cAsym_res-2_desc-denoisedSmoothed_bold.nii.gz')
run(compute_granger_results_from_file, all_bold_files)




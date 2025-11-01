import os
# Dir_1: "Locus coeruleus meta mask [MNI 152 linear space; 0.5 mm resolution]"
# Dir_2: "Reference brains [MNI 152 linear space; 0.5 mm resolution]"
# Dir_3: "LC_tpl-MNI152NLin2009cAsym"

# The Dir_1 and Dir_2 are downloaded from https://osf.io/sf2ky/.
# We first co-registered the reference brain used by LC template and the brain used by MNI152NLin2009cAsym
# to generate transformation affine matrix. 
src_space = 'MPRAGE_Template_MNI05.nii.gz' # 0.5mm, the reference brain used by LC meta mask
trg_space = 'tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz' # 1mm, the reference brain used by MNI152NLin2009cAsym

os.system(
    f'antsRegistrationSyNQuick.sh -d 3 -f {trg_space} -m {src_space} -o mprage-a2009c_'
) # use antsRegistrationSyNQuick.sh from ANTs

file_input = 'LCmetaMask_MNI05_s01f_plus50.nii.gz' # 0.5mm, copy from Dir_1
file_out = 'LCmetaMask_MNI152a2009c_s01f_plus50.nii.gz' # defined by self
os.system(
    f'antsApplyTransforms -d 3 \
  -i {file_input} \
  -r tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz \
  -o {file_out} \
  -n NearestNeighbor \
  -t mprage-a2009c_1Warp.nii.gz \
  -t mprage-a2009c_0GenericAffine.mat'
)

os.system('cp LCmetaMask_MNI152a2009c_s01f_plus50.nii.gz LC_tpl-MNI152NLin2009cAsym/LCmetaMask_MNI152a2009c_1mm.nii.gz')

####################################################################
## downsample 
from nilearn import image
import numpy as np

### LC 1mm voxels
lc_1mm = 'LC_tpl-MNI152NLin2009cAsym/LCmetaMask_MNI152a2009c_1mm.nii.gz'
np.sum(image.get_data(lc_1mm)) # 84 voxels 

### LC 2mm voxels 
fzmap_2mm = 'fzmap_MNI152NLin2009cAsym_2mm-gsr-05fd-s6/sub-001_task-rest_run-01_fzmap.nii.gz'
lc_2mm = image.resample_to_img(
    source_img=lc_1mm, target_img=fzmap_2mm, interpolation='nearest', copy_header=True, force_resample=True
    )
np.sum(lc_2mm.get_fdata()==1) # 12 voxels

### LC 3.5mm voxels 
fzmap_35mm = 'fzmap_MNI152NLin2009cAsym_3.5mm-gsr-05fd-s6/sub-001_task-rest_run-01_fzmap.nii.gz'
lc_35mm = image.resample_to_img(
    source_img=lc_1mm, target_img=fzmap_35mm, interpolation='nearest', copy_header=True, force_resample=True
    )
np.sum(lc_35mm.get_fdata()==1) # 2 voxels

####################################################################
# Of note: Due to data privacy, here we only show codes. Therefore the fzmap files are not provided. 
# And nii.gz files large than 100 MB were deleted. 
# We encourage you download dataset and run these command locally. 
# This LC template with 1mm resolution is welcome to use on person dataset. 
# Please cite our article. 

# @Author: Kang Wu, clancy_wu@126.com

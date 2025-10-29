# author Kang Wu
import numpy as np
import pandas as pd
import glob
from nilearn import maskers, image
seed = np.random.RandomState(0)
work_space = 'E://2022_sci_task/MVPA_surface_manuscript/'
# files resulting from Dpabisruf software
gmv_org = glob.glob('E://2022_sci_task/Report_original_outcomes/ResultsS/AnatVolu/*/*GM_probseg.nii') 
group_mask = 'E://2022_sci_task/MVPA_surface_manuscript/GreyMask.nii'
masker = maskers.NiftiMasker(mask_img=group_mask)
label = pd.read_excel('subject_71.xlsx')['group'].values
gmv_prepare = [ ]
for i in range(len(gmv_org)):
    img_ = masker.fit_transform(image.load_img(gmv_org[i]))
    img = np.squeeze(img_)
    gmv_prepare.append(img)
gmv_prepare = np.asarray(gmv_prepare)
from sklearn.preprocessing import StandardScaler
standerd = StandardScaler()
gmv_prepare = standerd.fit_transform(gmv_prepare)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(gmv_prepare, label, test_size=0.5, random_state=seed, shuffle=True)

from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.svm import LinearSVC
scoring = make_scorer(balanced_accuracy_score)
svc = LinearSVC( tol=0.0001, C=1.0, random_state=seed, max_iter=1e5)
svc.fit(x_train, y_train)
svc.score(x_test, y_test)
from sklearn.model_selection import cross_val_score
cross_val_score(svc, x_train, y_train, cv=2, scoring=scoring, n_jobs=-1) 
cross_val_score(svc, x_test, y_test, cv=2, scoring=scoring, n_jobs=-1) # result = 0.70

from sklearn.model_selection import permutation_test_score
scores, perm_scores, pvalue = permutation_test_score(svc, x_test, y_test, cv=2,scoring=scoring, n_permutations=5000,n_jobs=-1)
# scores = 0.70, pvalue = 0.023
###################################################################################################
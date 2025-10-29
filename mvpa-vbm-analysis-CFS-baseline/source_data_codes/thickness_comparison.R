#---------------------- 1. demographics comparison ----------------------
library(readxl)
clinic_score = read_excel('subjects_71_demographics.xlsx')
clinic_score$disease_month = as.numeric(clinic_score$disease_month)
clinic_score$gender = as.factor(clinic_score$gender)
group = as.factor(clinic_score$group)
# age
wilcox.test(clinic_score$age ~ clinic_score$group) # p=0.159
# gender
prop.test(c(11, 10), c(37, 34), alternative = 'two.sided', conf.level = 0.95, correct = F) # p=0.977
# BMI
t.test(clinic_score$BMI~clinic_score$group, var.equal=T) # p=0.145
# CFS duration
# SF-36
wilcox.test(clinic_score$`SF-36` ~ clinic_score$group) # p<0.001
# pri
wilcox.test(clinic_score$PRI ~ clinic_score$group) # p<0.001
# fs-14
wilcox.test(clinic_score$`FS-14` ~ clinic_score$group) # p<0.001

#---------------------- 2. generate thickness ----------------------
"library(freesurferformats)
library(gifti)
## five target cortices
target.l = c( 'S_central',  'G_and_S_paracentral', 'G_precentral')
target.r = c( 'S_intrapariet_and_P_trans',  'S_temporal_sup')
## fsaverage template
fsaverage.l = 'freesurfer/fsaverage/label/lh.aparc.a2009s.annot'
fsaverage.r = 'freesurfer/fsaverage/label/rh.aparc.a2009s.annot'
template.l = read.fs.annot(fsaverage.l) # 163842
template.r = read.fs.annot(fsaverage.r) # 163842
## smoothed thickness (FWHM=6) for each subject
cfs_thick_l_path = 'Results_Smooth/AnatSurfLH/Thickness/fsaverage/'
cfs_thick_r_path = 'Results_Smooth/AnatSurfRH/Thickness/fsaverage/'
cfs_thick_l_files = list.files(cfs_thick_l_path, pattern = 'L', full.names = T)
cfs_thick_r_files = list.files(cfs_thick_r_path, pattern = 'R', full.names = T)
fatigue_thickness = data.frame()
subjects = length(cfs_thick_l_files)
# left hemisphere
for (i in 1:subjects){
  data_input.l = readgii(cfs_thick_l_files[i])[[1]][[1]][ ,1]
  for (k in 1:length(target.l)){
    fatigue_thickness[i,k] = mean(data_input.l[template.l$label_names == target.l[k]])
  }
}
# right hemisphere
for (i in 1:subjects){
  data_input.r = readgii(cfs_thick_r_files[i])[[1]][[1]][ ,1]
  for (k in 1:length(target.r)){
    new_k = k+3
    fatigue_thickness[i, new_k] = mean(data_input.r[template.r$label_names == target.r[k]])
  }
}
colnames(fatigue_thickness) = c(paste0('L-', target.l) , paste0('R-', target.r))
fatigue_thickness$mean = rowMeans(fatigue_thickness)"
fatigue_thickness <- read.csv('Five_thickness.csv', header = T)
#---------------------- 3. thickness fits with GLM ----------------------
library(tidyverse)
library(MASS)
# GLM: SF-36 for all subjects
base_info = clinic_score[, c('SF-36')]
mymodel_data = cbind(fatigue_thickness[,1:5], base_info)
summary(lm(`SF-36`~., mymodel_data)) # adjusted R2=0.219, p<0.001
# GLM: FS-14 for all subjects
base_info = clinic_score[, c('FS-14')]
mymodel_data = cbind(fatigue_thickness[,1:5], base_info)
summary(lm(`FS-14`~., mymodel_data)) # adjusted R2=0.122, p=0.019
# GLM: PRI for all subjects
base_info = clinic_score[, c('PRI')]
mymodel_data = cbind(fatigue_thickness[,1:5], base_info)
summary(lm(`PRI`~., mymodel_data)) # adjusted R2=0.114, p=0.024
####################### split into two groups
## GLM: SF-36 in CFS group
patient_mask = (group == 'patient')
base_info = clinic_score[, c('SF-36')]
mymodel_data = cbind(fatigue_thickness[,1:5], base_info)
patient_model_data = mymodel_data[patient_mask, ]
summary(lm(`SF-36`~., patient_model_data)) # adjusted R2=0.272, p<0.001
## GLM: SF-36 in HC group
health_mask = (group == 'health')
base_info = clinic_score[, c('SF-36')]
mymodel_data = cbind(fatigue_thickness[,1:5], base_info)
health_model_data = mymodel_data[health_mask, ]
summary(lm(`SF-36`~., health_model_data)) # adjusted R2=0.026, p=0.344

#---------------------- 4. cortical thicknesses comparison between two groups ----------------------
# independent sample test
wilcox.test(fatigue_thickness$mean ~ clinic_score$group) # p=0.011
# anova test
summary(aov(fatigue_thickness$mean ~ clinic_score$group + clinic_score$age +
            clinic_score$gender + clinic_score$BMI)) # p=0.037
# permutation test
library(coin)
group = as.factor(clinic_score$group)
pvalue(oneway_test(fatigue_thickness$mean ~ group, distribution=approximate(nresample=5000))) # 0.0392

######################## end.

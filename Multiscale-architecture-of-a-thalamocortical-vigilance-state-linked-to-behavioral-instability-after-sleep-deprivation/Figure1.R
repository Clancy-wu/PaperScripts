library(data.table)
library(fsbrain)
library(freesurferformats)
library(subcortexVisualizationR)
library(lmerTest)
library(dplyr)
library(ggplot2)
library(pheatmap)
library(RColorBrewer)

# Figure 1A ---------------------------------------------------------------
num_from_label <- function(data_index, data_value, label_file){
  # data is 210
  label_info = read.fs.annot(label_file)
  label_vertex = label_info$label_codes
  label_code = label_info$colortable_df$code
  label_index = label_info$colortable_df$struct_index
  empty_num = rep(0, length(label_vertex))
  for (i in seq(data_index)){
    i_code = label_code[label_index == data_index[i]]
    empty_num[label_vertex == i_code] = data_value[i]
  }
  empty_num[empty_num==0] = NA
  return(empty_num)
}

label_lh_file = 'atlas/lh.BN_Atlas.annot'; label_rh_file = 'atlas/rh.BN_Atlas.annot' ## BN template
state_mean = fread('all_dynamic_brain_states/state_3_mean.txt') ## the third state is TVS
lh_num = num_from_label(seq(1, 210), state_mean$V1[1:210], label_lh_file)
rh_num = num_from_label(seq(1, 210), state_mean$V1[1:210], label_rh_file)
colFn_blue_red = colorRampPalette(c("#013e7d", "#FFFFFF", "#760715"))
## plot cortex
coloredmeshes <- vis.data.on.fsaverage(
  vis_subject_id = "fsaverage",
  morph_data_lh = lh_num,
  morph_data_rh = rh_num,
  surface = "pial",
  views = c("si"),
  draw_colorbar = 'horizontal',
  makecmap_options = list('colFn'=colFn_blue_red, symm=T, 'col.na'='white', 'range'=c(-0.15,0.15)),
  bg = 'curv_light',
  morph_data_both = NULL,
  style = "default"
)

vislayout.from.coloredmeshes(
  coloredmeshes = coloredmeshes,
  view_angles = c( "sd_medial_lh", "sd_lateral_lh",  "sd_lateral_rh", "sd_medial_rh"),
  output_img = "fsbrain_custom_1x4.png",  grid_like = FALSE, background_color = "white")

# plot subcortex
bn_atlas = fread('atlas/brainnetome.csv')
atlas = bn_atlas[ , .(subregion, hemi, index, Yeo_7network)]
setorder(atlas, index)
state_atlas_data = copy(atlas)
state_atlas_data$value = state_mean
colnames(state_atlas_data) = c('region', 'Hemisphere', 'index', 'Yeo_7network', 'value')

plot_subcortical_data(subcortex_data = state_atlas_data,
                      atlas = 'Brainnetome', hemisphere='both', 
                      line_color='black', line_thickness=0.5,
                      cmap=colFn_blue_red, vmin=-0.15, vmax=0.15)

state_atlas_data[211:246, ][order(-value)] ## enrich in the thalamus
## Canonical functional networks 
state_covs = fread('all_dynamic_brain_states/state_3_cov.txt') ## The third state is TVS
bn_atlas = fread('atlas/brainnetome.csv')
atlas = bn_atlas[ , .(subregion, hemi, index, Yeo_7network)]
cov_mat = as.matrix(state_covs)
network_order <- c("Visual", "Dorsal Attention", "Somatomotor", "Default",
                   "Frontoparietal", "Ventral Attention",   "Limbic")
net_label <- factor(atlas$Yeo_7network, levels = network_order)
network_names <- levels(net_label)
n_net <- length(network_names)
net_mat <- matrix(NA, nrow = n_net, ncol = n_net)
rownames(net_mat) <- network_names
colnames(net_mat) <- network_names

for (i in 1:n_net) {
  for (j in 1:n_net) {
    idx_i <- which(net_label == network_names[i])
    idx_j <- which(net_label == network_names[j])
    sub_mat <- cov_mat[idx_i, idx_j, drop = FALSE]
    if (i == j) {
      sub_vals <- sub_mat[upper.tri(sub_mat, diag = FALSE)]
    } else {
      sub_vals <- as.vector(sub_mat)
    }
    net_mat[i, j] <- mean(sub_vals, na.rm = TRUE)
  }
}

net_mat <- as.matrix(net_mat)
mode(net_mat) <- "numeric"
# max: 5, min: 0.21
my_breaks <- c(
  seq(0, 1, length.out = 40),
  seq(1, 3, length.out = 35),
  seq(3, 5, length.out = 26)
)
my_breaks <- unique(my_breaks)
my_colors <- colorRampPalette(c("#FFF7EC", "#FEE08B", "#B2182B"))(length(my_breaks) - 1)

pheatmap(
  net_mat, cluster_rows = FALSE, cluster_cols = FALSE,  color = my_colors,  breaks = my_breaks,
  border_color = "white",  display_numbers = TRUE,  number_format = "%.2f",  fontsize_number = 10,
  show_rownames = TRUE,  show_colnames = TRUE,  fontsize = 14,  fontsize_row = 13,  fontsize_col = 13,
  cellwidth = 35,  cellheight = 35,  angle_col = 45)

## subcortex
cov_mat = as.matrix(state_covs)
network_order <- c("rTtha", "mPFtha", "cTtha", 
                   "PPtha",  "lPFtha",  "Otha", 
                   "mPMtha", "Stha" )
net_label <- factor(atlas$subregion, levels = network_order)
network_names <- levels(net_label)
n_net <- length(network_names)
net_mat <- matrix(NA, nrow = n_net, ncol = n_net)
rownames(net_mat) <- network_names
colnames(net_mat) <- network_names

for (i in 1:n_net) {
  for (j in 1:n_net) {
    idx_i <- which(net_label == network_names[i])
    idx_j <- which(net_label == network_names[j])
    sub_mat <- cov_mat[idx_i, idx_j, drop = FALSE]
    if (i == j) {
      sub_vals <- sub_mat[upper.tri(sub_mat, diag = FALSE)]
    } else {
      sub_vals <- as.vector(sub_mat)
    }
    net_mat[i, j] <- mean(sub_vals, na.rm = TRUE)
  }
}
# max: 1.56, min: 0.13
subcor_breaks <- c(
  seq(0, 0.6, length.out = 40),
  seq(0.6, 1.2, length.out = 35),
  seq(1.2, 2, length.out = 26)
)
subcor_breaks <- unique(subcor_breaks)
subcor_colors <- colorRampPalette(c("#FFF7EC", "#FEE08B", "#B2182B"))(length(subcor_breaks) - 1)

pheatmap(
  net_mat,  cluster_rows = FALSE,  cluster_cols = FALSE,  color = subcor_colors,  breaks = subcor_breaks,
  border_color = "white",  display_numbers = TRUE,  number_format = "%.2f",  fontsize_number = 10,
  show_rownames = TRUE,  show_colnames = TRUE,  fontsize = 14,  fontsize_row = 13,  fontsize_col = 13,
  cellwidth = 35,  cellheight = 35,  angle_col = 45)


# Figure 1B ---------------------------------------------------------------
pvt = fread('PVT_results/PVT_result.csv')
pvt$run = factor(pvt$run, levels = c('RW', 'SD')) ## run01=RW, run02=SD
model <- lmer(rt_cv ~ run + (1 | subject), data = pvt)
summary(model) ## RTCV, SD vs. RW, p=0.003

ggplot(pvt, aes(x = run, y = rt_cv, fill = run)) +
  geom_line(aes(group = subject), color = "grey50", alpha = 0.5) +
  geom_boxplot(width = 0.45, alpha = 0.6, outlier.shape = NA) +
  geom_jitter(width = 0.05, size = 2, alpha = 0.8) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3, fill = "white") +
  stat_summary(fun.data = mean_se, geom = "errorbar", width = 0.12, size = 0.8) +
  scale_fill_manual(values = c("RW" = "#8B7BB8", "SD" = "#5BAE7D")) +
  labs(x = NULL, y = NULL) +
  theme_test() + 
  theme(
    legend.position = 'none', 
    plot.margin = unit(c(1, 1, 1, 1), "cm"),
    axis.title.x = element_text(size = 30),
    axis.title.y = element_text(size = 30),
    axis.text = element_text(size = 24),
    axis.text.x = element_blank(), 
    plot.title = element_text(size = 32, hjust = 0.1), 
    strip.text = element_text(size = 32), 
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1.2), 
    ) 

df = fread('fMRI_HMM_24wmcfs03_results/summary_stats/hmm_summary_stats_wide.csv')
df$run = factor(df$run, levels = c('RW', 'SD'))
model <- lmer(fo_state3 ~ run + (1 | sub_name), data = df)
summary(model) ## FO, SD vs. RW, p=0.029
model <- lmer(sr_state3 ~ run + (1 | sub_name), data = df)
summary(model) # SR, SD vs. RW, p=0.003

ggplot(df, aes(x = run, y = sr_state3, fill = run)) +
  geom_line(aes(group = sub_name), color = "grey50", alpha = 0.5) +
  geom_boxplot(width = 0.45, alpha = 0.6, outlier.shape = NA) +
  geom_jitter(width = 0.05, size = 2, alpha = 0.8) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3, fill = "white") +
  stat_summary(fun.data = mean_se, geom = "errorbar", width = 0.12, size = 0.8) +
  scale_fill_manual(values = c("RW" = "#8B7BB8", "SD" = "#5BAE7D")) +
  labs(x = NULL, y = NULL) +
  theme_test() + 
  theme(
    legend.position = 'none', 
    plot.margin = unit(c(1, 1, 1, 1), "cm"),
    axis.title.x = element_text(size = 30),
    axis.title.y = element_text(size = 30),
    axis.text = element_text(size = 24),
    axis.text.x = element_blank(), 
    plot.title = element_text(size = 32, hjust = 0.1), 
    strip.text = element_text(size = 32), 
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1.2), 
  ) 

setorder(df, sub_name, run)
setorder(pvt, subject, run)
diff_df = data.table(
  X = df[run=='SD', ]$fo_state3 - df[run=='RW', ]$fo_state3, 
  Y = pvt[run=='SD', ]$rt_cv - pvt[run=='RW', ]$rt_cv
)
cor.test(diff_df$X, diff_df$Y) # FO ~ RTCV, r=0.42, p=0.021

diff_df = data.table(
  X = df[run=='SD', ]$sr_state3 - df[run=='RW', ]$sr_state3, 
  Y = pvt[run=='SD', ]$rt_cv - pvt[run=='RW', ]$rt_cv
)
cor.test(diff_df$X, diff_df$Y) # SR ~ RTCV, r=0.31, p=0.095

ggplot(diff_df, aes(x = X, y = Y)) +
  geom_point( size = 3.5, shape = 21, stroke = 0.8, color = "#4A4A4A", fill = "#4A4A4A", alpha=1) +
  geom_smooth( method = "lm", se = TRUE, color = "#1F6F78", fill = "grey60", linewidth = 4 ) +
  theme_test(base_size = 16) +
  theme(
    axis.title = element_text(size = 18, face = "bold", color = "black"),
    axis.text = element_text(size = 15, color = "black"),
    axis.ticks = element_line(linewidth = 0.9, color = "black"),
    axis.ticks.length = unit(0.22, "cm"),
    panel.grid = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.9 ),
    axis.line = element_blank() 
  ) +
  labs(x = NULL, y = NULL)


# Figure 1D ---------------------------------------------------------------
#### RSA: representation similarity analysis 
rsa = fread('subject_RSA_with_TVS/subjects_RSA.csv')
setorder(rsa, subject, run)
setorder(pvt, subject, run)
diff_df = data.table(
  X = rsa[run=='SD', ]$cor_z - rsa[run=='RW', ]$cor_z, 
  Y = pvt[run=='SD', ]$rt_cv - pvt[run=='RW', ]$rt_cv
)
cor.test(diff_df$X, diff_df$Y) # RSA with TVS ~ RTCV, r=0.39, p=0.036
ggplot(diff_df, aes(x = X, y = Y)) +
  geom_point( size = 3.5, shape = 21, stroke = 0.8, color = "#4A4A4A", fill = "#4A4A4A", alpha=1) +
  geom_smooth( method = "lm", se = TRUE, color = "#1F6F78", fill = "grey60", linewidth = 4 ) +
  theme_test(base_size = 16) +
  theme(
    axis.title = element_text(size = 18, face = "bold", color = "black"),
    axis.text = element_text(size = 15, color = "black"),
    axis.ticks = element_line(linewidth = 0.9, color = "black"),
    axis.ticks.length = unit(0.22, "cm"),
    panel.grid = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.9 ),
    axis.line = element_blank() 
  ) +
  labs(x = NULL, y = NULL)

# Figure 1E ---------------------------------------------------------------
#### TVS ~ SD-CWM, spin-rotation based permutation test for spatial correlation. 
#### Python was used for this analysis. 
"
## Similarity between TVS and SD-CWM based on spin-rotation permutation test
import pandas as pd
import numpy as np
import nibabel as nib
from scipy.stats import pearsonr, spearmanr
from neuromaps import nulls, parcellate, stats

parcellation=('atlas/lh.fs5.BN_Atlas_fix.label.gii', 'atlas/rh.fs5.BN_Atlas_fix.label.gii')
bn_atlas = parcellate.Parcellater(parcellation=parcellation, space='fsaverage').fit()
SD_CWM_data = pd.read_csv('SD-CWM.csv')
SD_CWM_vertex = np.asarray(SD_CWM_data['weight'].values, dtype=float)
SD_CWM = bn_atlas.transform(SD_CWM_vertex, space='fsaverage')

# -----------------------------
# 1. Define valid ROI mask and compute Pearson correlation
# -----------------------------
TVS_data = pd.read_csv('all_dynamic_brain_states/state_3_mean.txt', header=None)
TVS = np.asarray(TVS_data[0].values, dtype=float)[:210]
mask = (TVS != 0) & (SD_CWM != 0)
pearsonr(TVS[mask], SD_CWM[mask]) # raw value, r=-0.23, p=0.001
pearsonr(np.abs(TVS[mask]), np.abs(SD_CWM[mask])) # abs value, r=0.24, p<0.001

# -----------------------------
# 3. Generate spin nulls
# -----------------------------
## raw value
rotated = nulls.baum(SD_CWM, atlas='fsaverage', density='10k',
                                n_perm=5000, seed=1234, parcellation=parcellation)
raw_compare = stats.compare_images(SD_CWM, TVS, nulls=rotated, return_nulls=True) 
print(raw_compare) ## r=-0.23, p=0.042

## abs value
abs_rotated = nulls.baum(np.abs(SD_CWM), atlas='fsaverage', density='10k',
                                n_perm=5000, seed=1234, parcellation=parcellation)
abs_compare = stats.compare_images(np.abs(SD_CWM), np.abs(TVS), nulls=abs_rotated, return_nulls=True) 
print(abs_compare) # r=0.24, p=0.043
"





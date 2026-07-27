library(data.table)
library(fsbrain)
library(RColorBrewer)
library(freesurferformats)

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
label_lh_file = '/home/clancy/TemplateFlow/BN_Atlas_freesurfer/fsaverage/label/lh.BN_Atlas.annot'
label_rh_file = '/home/clancy/TemplateFlow/BN_Atlas_freesurfer/fsaverage/label/rh.BN_Atlas.annot'
########################################################################################
state_mean = fread('all_dynamic_brain_states/state_3_mean.txt')

lh_num = num_from_label(seq(1, 210), state_mean$V1[1:210], label_lh_file)
rh_num = num_from_label(seq(1, 210), state_mean$V1[1:210], label_rh_file)
colFn_blue_red = colorRampPalette(c("#013e7d", "#FFFFFF", "#760715"));
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


library(subcortexVisualizationR)
bn_atlas = fread('Data/brainnetome.csv')
atlas = bn_atlas[ , .(subregion, hemi, index, Yeo_7network)]
setorder(atlas, index)
state_atlas_data = copy(atlas)
state_atlas_data$value = state_mean
colnames(state_atlas_data) = c('region', 'Hemisphere', 'index', 'Yeo_7network', 'value')

plot_subcortical_data(subcortex_data = state_atlas_data,
                      atlas = 'Brainnetome', hemisphere='both', 
                      line_color='black', line_thickness=0.5,
                      cmap=colFn_blue_red, vmin=-0.15, vmax=0.15)

state_atlas_data[211:246, ][order(-value)]
########################################################################################
library(pheatmap)
library(RColorBrewer)
state_covs = fread('all_dynamic_brain_states/state_3_cov.txt')
bn_atlas = fread('Data/brainnetome.csv')
atlas = bn_atlas[ , .(subregion, hemi, index, Yeo_7network)]
cov_mat = as.matrix(state_covs)
network_order <- c("Visual", "Dorsal Attention", "Somatomotor", "Default",
                   "Frontoparietal", "Ventral Attention",   "Limbic")
net_label = atlas$Yeo_7network
net_label <- factor(net_label, levels = network_order)

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
  border_color = "white",
  # show values
  display_numbers = TRUE,  number_format = "%.2f",  fontsize_number = 10,
  # labels
  show_rownames = TRUE,  show_colnames = TRUE,
  # font sizes
  fontsize = 14,  fontsize_row = 13,  fontsize_col = 13,
  # cell size
  cellwidth = 35,  cellheight = 35,
  # top labels style
  angle_col = 45)


## subcortex
cov_mat = as.matrix(state_covs)
network_order <- c("rTtha", "mPFtha", "cTtha", 
                   "PPtha",  "lPFtha",  "Otha", 
                   "mPMtha", "Stha" )
net_label = atlas$subregion
net_label <- factor(net_label, levels = network_order)
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
  border_color = "white",
  # show values
  display_numbers = TRUE,  number_format = "%.2f",  fontsize_number = 10,
  # labels
  show_rownames = TRUE,  show_colnames = TRUE,
  # font sizes
  fontsize = 14,  fontsize_row = 13,  fontsize_col = 13,
  # cell size
  cellwidth = 35,  cellheight = 35,
  # top labels style
  angle_col = 45)

########################################################################################
library(ggplot2)
library(dplyr)
library(data.table)
library(lmerTest)

pvt = fread('PVT_results/PVT_result.csv')
pvt$run = factor(pvt$run, levels = c('RW', 'SD'))
model <- lmer(rt_cv ~ run + (1 | subject), data = pvt)
summary(model)

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
summary(model) # p=0.029
model <- lmer(sr_state3 ~ run + (1 | sub_name), data = df)
summary(model) # p=0.003

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
cor.test(diff_df$X, diff_df$Y) # p=0.021

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

############################
rsa = fread('subjects_RSA.csv')
rsa$run = factor(rsa$run, levels = c('RW', 'SD'))
model <- lmer(cor_z ~ run + (1 | subject), data = rsa)
summary(model)

ggplot(rsa, aes(x = run, y = cor_z, fill = run)) +
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

setorder(rsa, subject, run)
setorder(pvt, subject, run)
diff_df = data.table(
  X = rsa[run=='SD', ]$cor_z - rsa[run=='RW', ]$cor_z, 
  Y = pvt[run=='SD', ]$rt_cv - pvt[run=='RW', ]$rt_cv
)
cor.test(diff_df$X, diff_df$Y) # p=0.035
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

predict_df = data.table(
  X = rsa[run=='RW', ]$cor_z, 
  Y = pvt[run=='SD', ]$rt_cv - pvt[run=='RW', ]$rt_cv
)
cor.test(predict_df$X, predict_df$Y) # p=0.035

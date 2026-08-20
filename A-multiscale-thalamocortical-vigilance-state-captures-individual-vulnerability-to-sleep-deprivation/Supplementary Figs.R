library(data.table)


# Extended Figure 1 -------------------------------------------------------
## The participant number and study design were shown. 


# Extended Figure 2 -------------------------------------------------------
library(data.table)
library(fsbrain)
library(RColorBrewer)
library(freesurferformats)
num_from_label <- function(data_index, data_value, label_file){
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
state_mean = fread('all_dynamic_brain_states/state_10_mean.txt') ## state_1_mean.txt --> state_10_mean.txt
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
  grid_like = FALSE, background_color = "white")

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

# Extended Figure 3 -------------------------------------------------------
library(data.table)
library(ggplot2)
library(dplyr)
library(data.table)
library(lmerTest)
## Figure 3A
pvt = fread('PVT_results/PVT_result.csv')
pvt$run = factor(pvt$run, levels = c('RW', 'SD'))
model <- lmer(rt_sd ~ run + (1 | subject), data = pvt)
summary(model)

ggplot(pvt, aes(x = run, y = rt_sd, fill = run)) +
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

## Figure 3B
states = fread('fMRI_HMM_24wmcfs03_results/summary_stats/hmm_summary_stats_wide.csv')
states$run = factor(states$run, levels = c('RW', 'SD'))
t_mat = matrix(NA, 2, 10)
p_mat = matrix(NA, 2, 10)
for (i in seq(10)){
  ## fo
  i_fo <- paste0('fo_state', i)
  formula_text <- paste(i_fo, '~ run + (1 | sub_name)')
  i_fo_result <- summary(lmer(as.formula(formula_text), data = states))
  t_mat[1, i] <- i_fo_result$coefficients['runSD', 't value']
  p_mat[1, i] <- i_fo_result$coefficients['runSD', 'Pr(>|t|)']
  ## sr
  i_sr = paste0('sr_state', i)
  formula_text <- paste(i_sr, '~ run + (1 | sub_name)')
  i_sr_result <- summary(lmer(as.formula(formula_text), data = states))
  t_mat[2, i] <- i_sr_result$coefficients['runSD', 't value']
  p_mat[2, i] <- i_sr_result$coefficients['runSD', 'Pr(>|t|)']  
}

library(corrplot)
rownames(t_mat) = c('FO', 'SR')
colnames(t_mat) = paste0('State', seq(10))
rownames(p_mat) = c('FO', 'SR')
colnames(p_mat) = paste0('State', seq(10))

corrplot(t_mat, is.corr = FALSE, method = 'square',
         p.mat = p_mat, insig = 'label_sig', sig.level = c(0.001, 0.01, 0.05),
         pch.cex = 2, pch.col = 'grey30', col = rev(COL2('RdBu')), col.lim = c(-4, 4), 
         tl.pos = 'n')

## Figure 3C
states_pvt <- merge(states, pvt, by.x = c("sub_name", "run"), by.y = c("subject", "run"))
states_pvt$run = factor(states_pvt$run, levels = c('RW', 'SD'))
r_mat = matrix(NA, 3, 10)
p_mat = matrix(NA, 3, 10)

for (i in seq(10)){
  ## mean RT
  i_fo <- paste0('fo_state', i)
  i_fo_change = states_pvt[run=='SD', ..i_fo][[1]] - states_pvt[run=='RW', ..i_fo][[1]]
  rt_change = states_pvt[run=='SD', rt_mean] - states_pvt[run=='RW', rt_mean]
  i_fo_result <- cor.test(i_fo_change, rt_change, method='pearson')
  r_mat[1, i] <- i_fo_result$estimate[[1]]
  p_mat[1, i] <- i_fo_result$p.value
  ## RT CV
  rt_change = states_pvt[run=='SD', rt_cv] - states_pvt[run=='RW', rt_cv]
  i_fo_result <- cor.test(i_fo_change, rt_change, method='pearson')
  r_mat[2, i] <- i_fo_result$estimate[[1]]
  p_mat[2, i] <- i_fo_result$p.value
  ## RT SD
  rt_change = states_pvt[run=='SD', rt_sd] - states_pvt[run=='RW', rt_sd]
  i_fo_result <- cor.test(i_fo_change, rt_change, method='pearson')
  r_mat[3, i] <- i_fo_result$estimate[[1]]
  p_mat[3, i] <- i_fo_result$p.value
}

rownames(r_mat) = c('mean RT', 'RT CV', 'RT SD')
colnames(r_mat) = paste0('State', seq(10))
rownames(p_mat) = c('mean RT', 'RT CV', 'RT SD')
colnames(p_mat) = paste0('State', seq(10))

corrplot(r_mat, is.corr = TRUE, method = 'square',
         p.mat = p_mat, insig = 'label_sig', sig.level = c(0.001, 0.01, 0.05),
         pch.cex = 2, pch.col = 'grey30', col = rev(COL2('RdBu')), 
         tl.pos = 'n')

## Figure 3D
states_pvt <- merge(states, pvt, by.x = c("sub_name", "run"), by.y = c("subject", "run"))
states_pvt$run = factor(states_pvt$run, levels = c('RW', 'SD'))
r_mat = matrix(NA, 3, 10)
p_mat = matrix(NA, 3, 10)

for (i in seq(10)){
  ## mean RT
  i_sr <- paste0('sr_state', i)
  i_sr_change = states_pvt[run=='SD', ..i_sr][[1]] - states_pvt[run=='RW', ..i_sr][[1]]
  rt_change = states_pvt[run=='SD', rt_mean] - states_pvt[run=='RW', rt_mean]
  i_sr_result <- cor.test(i_sr_change, rt_change, method='pearson')
  r_mat[1, i] <- i_sr_result$estimate[[1]]
  p_mat[1, i] <- i_sr_result$p.value
  ## RT CV
  rt_change = states_pvt[run=='SD', rt_cv] - states_pvt[run=='RW', rt_cv]
  i_sr_result <- cor.test(i_sr_change, rt_change, method='pearson')
  r_mat[2, i] <- i_sr_result$estimate[[1]]
  p_mat[2, i] <- i_sr_result$p.value
  ## RT SD
  rt_change = states_pvt[run=='SD', rt_sd] - states_pvt[run=='RW', rt_sd]
  i_sr_result <- cor.test(i_sr_change, rt_change, method='pearson')
  r_mat[3, i] <- i_sr_result$estimate[[1]]
  p_mat[3, i] <- i_sr_result$p.value
}

rownames(r_mat) = c('mean RT', 'RT CV', 'RT SD')
colnames(r_mat) = paste0('State', seq(10))
rownames(p_mat) = c('mean RT', 'RT CV', 'RT SD')
colnames(p_mat) = paste0('State', seq(10))

corrplot(r_mat, is.corr = TRUE, method = 'square',
         p.mat = p_mat, insig = 'label_sig', sig.level = c(0.001, 0.01, 0.05),
         pch.cex = 2, pch.col = 'grey30', col = rev(COL2('RdBu')), 
         tl.pos = 'n')


# Extended Figure 4 -------------------------------------------------------








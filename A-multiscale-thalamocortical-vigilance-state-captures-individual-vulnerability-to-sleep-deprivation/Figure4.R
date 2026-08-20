library(data.table)
library(lmerTest)
library(ggplot2)
library(magrittr)

compute_dz_ci <- function(delta_value, R = 5000) {
  # bootstrap to compute paired cohen's d and 95% CI
  x <- delta_value[is.finite(delta_value)]
  n <- length(x)
  mean_delta <- mean(x)
  sd_delta <- sd(x)
  se_delta <- sd_delta / sqrt(n)
  
  # Cohen's dz
  dz <- mean_delta / sd_delta
  
  # 95% CI for mean delta
  t_crit <- qt(0.975, df = n - 1)
  mean_ci_low  <- mean_delta - t_crit * se_delta
  mean_ci_high <- mean_delta + t_crit * se_delta
  
  # Bootstrap 95% CI for dz
  set.seed(42)
  boot_dz <- replicate(R, {
    xb <- sample(x, size = n, replace = TRUE)
    if (sd(xb) == 0) return(NA_real_)
    mean(xb) / sd(xb)
  })
  
  dz_ci <- quantile(boot_dz, probs = c(0.025, 0.975), na.rm = TRUE)
  
  data.frame(
    n = n,
    mean_delta = mean_delta,
    mean_ci_low = mean_ci_low,
    mean_ci_high = mean_ci_high,
    cohen_dz = dz,
    dz_ci_low = unname(dz_ci[1]),
    dz_ci_high = unname(dz_ci[2])
  )
}

################################################ 
#### ECG
clean_data_by_quality_control = function(ecg_data){
  ecg_data_clean = ecg_data[ecg_good == TRUE, ] # ecg_good == True
  ecg_data_clean = ecg_data_clean[n_rr_valid >= 150, ] # n_rr_valid >= 150
  ecg_data_clean = ecg_data_clean[abnormal_rr_ratio <= 0.05, ] # abnormal_rr_ratio <= 0.05
  ecg_data_clean = ecg_data_clean[(mean_hr_bpm>=40)&(mean_hr_bpm<=110), ] #40 <= mean_hr_bpm <= 110
  ecg_data_clean = ecg_data_clean[(mean_rr_ms>=500)&(mean_rr_ms<=1300), ] #500 <= mean_rr_ms <= 1300
  ecg_data_clean = ecg_data_clean[(rmssd_ms>=10)&(rmssd_ms<=200), ] #10 <= rmssd_ms <= 200
  ecg_data_clean = ecg_data_clean[(sdnn_ms>=10)&(sdnn_ms<=200), ] #10 <= sdnn_ms <= 200
  return(ecg_data_clean)
}

ecg_data = fread('ECG_preprocess/ecg_summary.csv')
setnames(ecg_data, "sub", "subject")
ecg_clean = clean_data_by_quality_control(ecg_data)
ecg_hr = dcast(ecg_clean, subject + state ~ condition, value.var = "mean_hr_bpm")
ecg_hr$hr_reactivity = ecg_hr$Task - ecg_hr$CloseEye
ecg_hrcv = dcast(ecg_clean, subject + state ~ condition, value.var = "hr_cv")
ecg_hrcv$hrcv_reactivity = ecg_hrcv$Task - ecg_hrcv$CloseEye

################################################ 
#### EEG
eeg_data = fread('EEG_results/stats/eeg_primary_markers_delta.csv')
eeg_data = eeg_data[marker == 'global_theta_alpha_ratio_mean_linear', ]
setnames(eeg_data, "sub", "subject")
eeg_global = dcast(eeg_data, subject + session ~ marker, value.var = "delta_value")

################################################ 
#### fMRI
fmri_data = fread('fMRI_HMM_24wmcfs03_results/summary_stats/hmm_summary_stats_wide.csv')
fmri_data = fmri_data[, .(sub_name, run, fo_state3)]
setnames(fmri_data, "sub_name", "subject")

#### Merge ECG + EEG + fMRI
ecg_diff = dcast(ecg_hrcv, subject ~ state, value.var = "hrcv_reactivity")
ecg_diff$ecg_diff = ecg_diff$SD - ecg_diff$RW

eeg_diff = dcast(eeg_global, subject ~ session, value.var = "global_theta_alpha_ratio_mean_linear")
eeg_diff$eeg_diff = eeg_diff$SD - eeg_diff$RW

fmri_diff = dcast(fmri_data, subject ~ run, value.var = 'fo_state3')
fmri_diff$fo_diff = fmri_diff$SD - fmri_diff$RW

df_merge = merge(fmri_diff[, .(subject, fo_diff)], 
                 eeg_diff[, .(subject, eeg_diff)], 
                 by = "subject", all = FALSE) %>%
            merge(., ecg_diff[, .(subject, ecg_diff)], by = "subject", all = FALSE)
df_merge = na.omit(df_merge, cols = c('fo_diff', 'eeg_diff', 'ecg_diff'))

merge_subs = df_merge$subject # 10 subjects

# Figure 4B ---------------------------------------------------------------
## Q1: Can SD change EEG ? 
model <- lmer(global_theta_alpha_ratio_mean_linear ~ session + (1 | subject), data = eeg_global[subject %in% merge_subs, ])
summary(model) # p=0.010

dcast(eeg_global[subject %in% merge_subs, ], subject ~ session, value.var = 'global_theta_alpha_ratio_mean_linear') %>%
  .[, .(diff=SD-RW)]%>%
  .[, compute_dz_ci(diff)]

ggplot(eeg_global[subject %in% merge_subs, ], 
       aes(x = session, y = global_theta_alpha_ratio_mean_linear, fill = session)) +
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

cor.test(df_merge$eeg_diff, df_merge$fo_diff, method = 'pearson') # p=0.031
ggplot(df_merge, aes(x = fo_diff, y = eeg_diff)) +
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


# Figure 4C ---------------------------------------------------------------
## Q1: Can SD change ECG ? 
model <- lmer(hr_reactivity ~ state + (1 | subject), data = ecg_hr[subject %in% merge_subs, ])
summary(model) # p=0.056

dcast(ecg_hr[subject %in% merge_subs, ], subject ~ state, value.var = 'hr_reactivity') %>%
  .[, .(diff=SD-RW)]%>%
  .[, compute_dz_ci(diff)]

model <- lmer(hrcv_reactivity ~ state + (1 | subject), data = ecg_hrcv[subject %in% merge_subs, ])
summary(model) # 0.714

ggplot(ecg_hr[subject %in% merge_subs, ], 
       aes(x = state, y = hr_reactivity, fill = state)) +
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

cor.test(df_merge$ecg_diff, df_merge$fo_diff, method = 'spearman') # p=0.031
ggplot(df_merge, aes(x = fo_diff, y = ecg_diff)) +
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


# Figure 4D ---------------------------------------------------------------
cor.test(df_merge$ecg_diff, df_merge$eeg_diff, method = 'spearman') # p=0.060
ggplot(df_merge, aes(x = ecg_diff, y = eeg_diff)) +
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


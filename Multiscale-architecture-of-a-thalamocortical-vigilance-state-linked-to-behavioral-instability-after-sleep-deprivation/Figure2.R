library(ggplot2)
library(data.table)
library(ggsci)

# Figure 2B ---------------------------------------------------------------
dc_data = fread('Dynamic_coupling/TVS_AAN_DynamicCoupling_results.csv')
# state3 is the TVS
####    Dynamic coupling mean
## Question 1: Does DC matters ?
t.test(dc_data[run=='RW', ]$state3_dcmean, mu=0) # p=0.4318
t.test(dc_data[run=='SD', ]$state3_dcmean, mu=0) # p=0.0001392

ggplot(dc_data[run=='RW', ], aes(x = run, y = state3_dcmean, fill = run)) +
  geom_boxplot(width = 0.45, alpha = 0.6, outlier.shape = NA) +
  geom_jitter(width = 0.05, size = 2, alpha = 0.8) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3, fill = "white") +
  stat_summary(fun.data = mean_se, geom = "errorbar", width = 0.12, size = 0.8) +
  scale_fill_manual(values = c("RW" = "#8B7BB8")) +
  labs(x = NULL, y = NULL) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", linewidth =2) + 
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

ggplot(dc_data[run=='SD', ], aes(x = run, y = state3_dcmean, fill = run)) +
  geom_boxplot(width = 0.45, alpha = 0.6, outlier.shape = NA) +
  geom_jitter(width = 0.05, size = 2, alpha = 0.8) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3, fill = "white") +
  stat_summary(fun.data = mean_se, geom = "errorbar", width = 0.12, size = 0.8) +
  scale_fill_manual(values = c("SD" = "#5BAE7D")) +
  labs(x = NULL, y = NULL) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", linewidth =2) + 
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

## Question 2: Does SD can alter DC ?
library(lmerTest)
dc_data$run = factor(dc_data$run, levels = c('RW', 'SD'))
model <- lmer(state3_dcmean ~ run + (1 | subject), data = dc_data)
summary(model) # p=0.0425

ggplot(dc_data, aes(x = run, y = state3_dcmean, fill = run)) +
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

## Question 3: Does RT_CV relates to DC ? No, p=0.230
pvt = fread('PVT_results/PVT_result.csv')
df = merge(dc_data[, .(subject, run, state3_dcmean)], pvt, by=c('subject', 'run')) # 
cor.test(
  df[run=='SD', ]$state3_dcmean - df[run=='RW', ]$state3_dcmean, 
  df[run=='SD', ]$rt_cv - df[run=='RW', ]$rt_cv, 
  method = 'pearson' ) # r=-0.225, p=0.230


# Figure 2C ---------------------------------------------------------------
library(dplyr)
library(tidyr)
library(ggplot2)
library(purrr)

roi_coupling = fread('Dynamic_coupling/TVS_AAN_ROIs_results.csv')
roi_coupling$run <- factor(roi_coupling$run, levels = c("RW", "SD"))
roi_long = melt(roi_coupling, id.vars = c('subject', 'run'), variable.name = 'roi_name', 
                value.name = 'value')
roi_diff = roi_long %>% 
          pivot_wider( names_from = run, values_from = value ) %>% 
          mutate( delta = SD - RW )

get_roi_stats <- function(dat, R = 5000, seed = 42) {
  x_rw <- dat$RW
  x_sd <- dat$SD
  delta <- dat$delta
  n <- length(delta)
  
  mean_rw <- mean(x_rw)
  mean_sd <- mean(x_sd)
  mean_delta <- mean(delta)
  sd_delta <- sd(delta)
  se_delta <- sd_delta / sqrt(n)
  
  tt <- t.test(x_sd, x_rw, paired = TRUE)
  t_value <- unname(tt$statistic)
  p_value <- tt$p.value  
  
  # 95% CI
  t_crit <- qt(0.975, df = n - 1)
  mean_delta_ci_low  <- mean_delta - t_crit * se_delta
  mean_delta_ci_high <- mean_delta + t_crit * se_delta
  
  # paired Cohen's dz
  dz <- mean_delta / sd_delta
  
  # bootstrap 95% CI for dz
  set.seed(seed)
  boot_dz <- replicate(R, {
    idx <- sample.int(n, size = n, replace = TRUE)
    d_boot <- delta[idx]
    if (sd(d_boot) == 0) return(NA_real_)
    mean(d_boot) / sd(d_boot)
  })
  
  dz_ci <- quantile(boot_dz, probs = c(0.025, 0.975), na.rm = TRUE)
  
  tibble(
    n = n,
    mean_RW = mean_rw,
    mean_SD = mean_sd,
    mean_delta = mean_delta,
    mean_delta_ci_low = mean_delta_ci_low,
    mean_delta_ci_high = mean_delta_ci_high,
    dz = dz,
    dz_ci_low = unname(dz_ci[1]),
    dz_ci_high = unname(dz_ci[2]),
    t_value = t_value,
    p_value = p_value
  )
}
roi_split <- split(roi_diff, roi_diff$roi_name)
roi_stats <- imap_dfr(roi_split, ~{
  get_roi_stats(.x, R = 5000, seed = 123 + which(names(roi_split) == .y)) %>%
    mutate(roi = .y)
})

roi_stats <- roi_stats %>%
  mutate(
    p_fdr = p.adjust(p_value, method = "fdr"),
    sig_fdr = case_when(
      p_fdr < 0.001 ~ "***",
      p_fdr < 0.01 ~ "**",
      p_fdr < 0.05 ~ "*",
      TRUE ~ "n.s."
    )
  ) %>%
  arrange(desc(dz))

roi_stats <- roi_stats %>%
  mutate(
    roi = factor(roi, levels = rev(roi)),
    label_d = sprintf("d = %.2f", dz),
    label_p = ifelse(
      p_fdr < 0.001,
      "FDR p < 0.001",
      sprintf("FDR p = %.3f", p_fdr)
    ),
    label_all = paste0(label_d, ", ", label_p),
    hjust_val = ifelse(dz >= 0, -0.1, 1.1)
  )

x_min <- min(c(roi_stats$dz_ci_low, 0), na.rm = TRUE)
x_max <- max(c(roi_stats$dz_ci_high, 0), na.rm = TRUE)
x_pad <- 0.18 * (x_max - x_min)

x_limits <- c(x_min - x_pad, x_max + x_pad)

ggplot(roi_stats, aes(x = dz, y = roi)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey60", linewidth = 1) +
  geom_errorbarh( aes(xmin = dz_ci_low, xmax = dz_ci_high),  height = 0.2,  linewidth = 2,
    color = "grey30"  ) +
  geom_point( size = 8, shape = 21, fill = "#E18727FF", color = "black", stroke = 0.8 ) +
  coord_cartesian(xlim = x_limits) +
  labs(   x = NULL, y = NULL,  ) +
  theme_test() + 
  theme(
    legend.position = 'none', 
    plot.margin = unit(c(1, 1, 1, 1), "cm"),
    axis.title.x = element_text(size = 30),
    axis.title.y = element_text(size = 30),
    axis.text = element_text(size = 24),
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    plot.title = element_text(size = 32, hjust = 0.1), 
    strip.text = element_text(size = 32), 
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1.2), 
  ) 


# Figure 2D ---------------------------------------------------------------
#### all states comparison
library(corrplot)
dc_data = fread('Dynamic_coupling/TVS_AAN_DynamicCoupling_results.csv')
state_names = colnames(dc_data)[3:12]
rw_t = vector(); rw_p = vector()
sd_t = vector(); sd_p = vector()
sdrw_t = vector(); sdrw_p = vector()

for (state_name in state_names){
  select_col = c('subject', 'run', state_name)
  state_data = dc_data[, ..select_col]
  colnames(state_data) = c('subject', 'run', 'value')
  ## RW
  state_rw = t.test(state_data[run=='RW', ]$value, mu=0)
  rw_t = c(rw_t, state_rw$statistic[[1]])
  rw_p = c(rw_p, state_rw$p.value)
  ## SD
  state_sd = t.test(state_data[run=='SD', ]$value, mu=0)
  sd_t = c(sd_t, state_sd$statistic[[1]])
  sd_p = c(sd_p, state_sd$p.value)
  ## SD-RW
  state_data$run = factor(state_data$run, levels = c('RW', 'SD'))
  state_sdrw = summary(lmer(value ~ run + (1 | subject), data = state_data))
  sdrw_t = c(sdrw_t, state_sdrw$coefficients['runSD', 't value'])
  sdrw_p = c(sdrw_p, state_sdrw$coefficients['runSD', 'Pr(>|t|)'])
}
df = data.table(
  state_name = state_names, 
  rw_t = rw_t, 
  rw_p = rw_p, 
  sd_t = sd_t, 
  sd_p = sd_p, 
  sdrw_t = sdrw_t, 
  sdrw_p = sdrw_p
)
df_tvalue = df[, .(state_name, rw_t, sd_t, sdrw_t)]
df_pvalue = df[, .(state_name, rw_p, sd_p, sdrw_p)]

mat_tvalue = as.matrix(df_tvalue[, 2:4])
rownames(mat_tvalue) = df_tvalue$state_name
colnames(mat_tvalue) = c('RW', 'SD', 'SD-RW')
mat_pvalue = as.matrix(df_pvalue[, 2:4])
rownames(mat_pvalue) = df_pvalue$state_name
colnames(mat_pvalue) = c('RW', 'SD', 'SD-RW')

mat_tvalue_max = copy(mat_tvalue)
mat_tvalue_max[mat_tvalue_max > 6] = 6
mat_tvalue_max[mat_tvalue_max < -6] = -6
corrplot(t(mat_tvalue_max), is.corr = FALSE, method = 'square',
         p.mat = t(mat_pvalue), insig = 'label_sig', sig.level = c(0.001, 0.01, 0.05),
         pch.cex = 2, pch.col = 'grey30', col = rev(COL2('RdBu')), col.lim = c(-6, 6) )


# Figure 2E ---------------------------------------------------------------
#### independent LC template
lc = fread('Dynamic_coupling/TVS_LC_results.csv')
t.test(lc[run=='RW', ]$value, mu=0) # p=0.4472
t.test(lc[run=='SD', ]$value, mu=0) # p=0.0009054

ggplot(lc[run=='RW', ], aes(x = run, y = value, fill = run)) +
  geom_boxplot(width = 0.45, alpha = 0.6, outlier.shape = NA) +
  geom_jitter(width = 0.05, size = 2, alpha = 0.8) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3, fill = "white") +
  stat_summary(fun.data = mean_se, geom = "errorbar", width = 0.12, size = 0.8) +
  scale_fill_manual(values = c("RW" = "#8B7BB8")) +
  labs(x = NULL, y = NULL) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", linewidth =2) + 
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

ggplot(lc[run=='SD', ], aes(x = run, y = value, fill = run)) +
  geom_boxplot(width = 0.45, alpha = 0.6, outlier.shape = NA) +
  geom_jitter(width = 0.05, size = 2, alpha = 0.8) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3, fill = "white") +
  stat_summary(fun.data = mean_se, geom = "errorbar", width = 0.12, size = 0.8) +
  scale_fill_manual(values = c("SD" = "#5BAE7D")) +
  labs(x = NULL, y = NULL) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", linewidth =2) + 
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

lc$run = factor(lc$run, levels = c('RW', 'SD'))
model <- lmer(value ~ run + (1 | subject), data = lc)
summary(model) # p=0.0598

ggplot(lc, aes(x = run, y = value, fill = run)) +
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

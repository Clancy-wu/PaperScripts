# TVS amplitude space dissociates acute sleep deprivation from chronic sleep–fatigue disorders
library(data.table)
library(ggplot2)
library(ggpubr)
library(ggpp)

# Figure 3A ---------------------------------------------------------------
SWD = fread('MultipleDiseases/SWD.csv') # shift work disorder
SD = fread('MultipleDiseases/SD.csv')
Health = fread('MultipleDiseases/Health.csv')

ggplot() +
  geom_point(data=SWD, aes(x = tvs_express_amplitude, y = tvs_aan_amplitude), 
    color = "#4DBBD5FF", alpha = 0.8, size = 1.5) + 
  geom_point(data=SD, aes(x = tvs_express_amplitude, y = tvs_aan_amplitude), 
             color = "#F39B7FFF", alpha = 0.8, size = 1.5) +
  geom_point(data=Health, aes(x = tvs_express_amplitude, y = tvs_aan_amplitude), 
             color = "gray50", alpha = 0.3, size = 1.5) +
  
  stat_ellipse(data=SWD, aes(x = tvs_express_amplitude, y = tvs_aan_amplitude), 
               level = 0.68, geom = "polygon", color = "#4DBBD5FF", fill = '#4DBBD5FF', 
               alpha = 0.3, linetype = "solid", size = 1) +
  stat_ellipse(data=SD, aes(x = tvs_express_amplitude, y = tvs_aan_amplitude), 
               level = 0.68, geom = "polygon", color = "#F39B7FFF", fill='#F39B7FFF', 
               alpha = 0.3, linetype = "solid", size = 1) +
  stat_ellipse(data=Health, aes(x = tvs_express_amplitude, y = tvs_aan_amplitude), 
               level = 0.68, color = "gray50", alpha = 0.6, linetype = "solid", size = 1) +

  stat_centroid(data=SWD, aes(x = tvs_express_amplitude, y = tvs_aan_amplitude), 
                color = "#4DBBD5FF", size = 4, shape = 19) +
  stat_centroid(data=SD, aes(x = tvs_express_amplitude, y = tvs_aan_amplitude), 
                color = "#F39B7FFF", size = 4, shape = 19) +  
  stat_centroid(data=Health, aes(x = tvs_express_amplitude, y = tvs_aan_amplitude), 
                color = "black", size = 4, shape = 19) +  
  
  theme_test() +
  labs(x = NULL, y = NULL) +
  theme(
    legend.position = 'none', 
    plot.margin = unit(c(1, 1, 1, 1), "cm"),
    axis.text = element_text(size = 16),
    plot.title = element_text(size = 32, hjust = 0.1), 
    strip.text = element_text(size = 32), 
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1.2), 
  ) 


# Figure 3B ---------------------------------------------------------------
library(ICSNP)
# Q1: Whether the shape is different among SD, SWD, and Health ?
sd_swd = HotellingsT2(SD[, 2:3], SWD[, 2:3]) # T.2=3.70, p=0.029
sd_health = HotellingsT2(SD[, 2:3], Health[, 2:3]) # T.2=16.80, p<0.001
swd_health = HotellingsT2(SWD[, 2:3], Health[, 2:3]) # T.2=5.22, p=0.006
round(p.adjust(c(sd_swd$p.value[1,1], sd_health$p.value[1,1], swd_health$p.value[1,1]), 
         method = 'fdr'), 3) # 0.029 0.000 0.008

# Q2: How SD and Insomnia deviate from Health ?
# ============================================
# Step 1：HC-defined normative space
# ============================================
X <- as.matrix(Health[, c("tvs_express_amplitude", "tvs_aan_amplitude")])
center_HC <- colMeans(X)
cov_HC <- cov(X)

scale_HC <- function(data) {
  data_mat <- as.matrix(data[, c("tvs_express_amplitude", "tvs_aan_amplitude")])
  centered <- sweep(data_mat, 2, center_HC, "-")
  # Mahalanobis transformation
  chol_decomp <- chol(cov_HC + diag(1e-10, ncol(cov_HC)))
  scaled <- centered %*% solve(chol_decomp)
  colnames(scaled) <- c("X_scaled", "Y_scaled")
  return(as.data.frame(scaled))
}

# HC --> normative space
HC_scaled <- scale_HC(Health)

# ============================================
# Step 2：The centroid deviation vector of Insomnia and SD
# ============================================
SWD_scaled <- scale_HC(SWD)
SD_scaled <- scale_HC(SD)

center_SWD_scaled <- colMeans(SWD_scaled)
center_SD_scaled <- colMeans(SD_scaled)

deviation_SWD <- center_SWD_scaled
deviation_SD <- center_SD_scaled

# ============================================
# Step 3：centroid deviation vector --> X  + Y axis 
# ============================================
SWD_X_dev <- deviation_SWD[1]
SWD_Y_dev <- deviation_SWD[2]
SD_X_dev <- deviation_SD[1]
SD_Y_dev <- deviation_SD[2]

# ============================================
# X or Y direction ?
# ============================================
# deviation angle
angle_SWD <- atan2(SWD_Y_dev, SWD_X_dev) * 180 / pi
angle_SD <- atan2(SD_Y_dev, SD_X_dev) * 180 / pi

# deviation length 
magnitude_SWD <- sqrt(SWD_X_dev^2 + SWD_Y_dev^2)
magnitude_SD <- sqrt(SD_X_dev^2 + SD_Y_dev^2)

# ============================================
# Plot
# ============================================

vector_data <- data.frame(
  Group = c("SWD", "SD"),
  X = c(SWD_X_dev, SD_X_dev),
  Y = c(SWD_Y_dev, SD_Y_dev)
)

ggplot(vector_data, aes(x = X, y = Y, color = Group)) +
  geom_segment(aes(x = 0, y = 0, xend = X, yend = Y),
               arrow = arrow(type = "closed", length = unit(0.25, "cm")),
               size = 1.5, alpha = 0.9) +
  geom_text(aes(label = paste0("(", round(X, 3), ", ", round(Y, 3), ")"),
                x = X * 0.5, y = Y * 0.5),
            size = 4, fontface = "bold", vjust = -0.5) +
  geom_text(aes(label = Group, x = X * 1.1, y = Y * 1.1),
            size = 5, fontface = "bold") +
  geom_point(aes(x = 0, y = 0), color = "black", size = 3) +
  annotate("text", x = 0, y = 0, label = "HC", vjust = -0.8, fontface = "bold") +
  geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
  geom_vline(xintercept = 0, linetype = "dashed", alpha = 0.5) +
  scale_color_manual(values = c("SWD" = "#4DBBD5FF", "SD" = "#F39B7FFF")) +
  coord_fixed(ratio = 1) +
  labs(title = "Centroid Deviation Vectors in HC-Defined Space",
       subtitle = paste0("HC centroid at (0,0) | Space normalized by HC covariance"),
       x = "Deviation along X axis (standardized)",
       y = "Deviation along Y axis (standardized)") +
  theme_test() +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5))

# ============================================
# Bootstrap test for deviation on X and Y
# ============================================
set.seed(123)
n_boot <- 5000

# SWD：Deviations of X / Y ≠ 0 ? 
boot_SWD_X <- replicate(n_boot, {
  boot_sample <- SWD[sample(1:nrow(SWD), nrow(SWD), replace = TRUE), ]
  boot_center <- colMeans(scale_HC(boot_sample))
  return(boot_center[1])
})

boot_SWD_Y <- replicate(n_boot, {
  boot_sample <- SWD[sample(1:nrow(SWD), nrow(SWD), replace = TRUE), ]
  boot_center <- colMeans(scale_HC(boot_sample))
  return(boot_center[2])
})

# SD：Deviations of X / Y ≠ 0 ? 
boot_SD_X <- replicate(n_boot, {
  boot_sample <- SD[sample(1:nrow(SD), nrow(SD), replace = TRUE), ]
  boot_center <- colMeans(scale_HC(boot_sample))
  return(boot_center[1])
})

boot_SD_Y <- replicate(n_boot, {
  boot_sample <- SD[sample(1:nrow(SD), nrow(SD), replace = TRUE), ]
  boot_center <- colMeans(scale_HC(boot_sample))
  return(boot_center[2])
})

# Bootstrap test for 95% CI and P value 
# SWD
ci_X_SWD <- quantile(boot_SWD_X, c(0.025, 0.975))
ci_Y_SWD <- quantile(boot_SWD_Y, c(0.025, 0.975))
p_X_SWD <- 2 * min(mean(boot_SWD_X >= 0), mean(boot_SWD_X <= 0))
p_Y_SWD <- 2 * min(mean(boot_SWD_Y >= 0), mean(boot_SWD_Y <= 0))

# SD
ci_X_SD <- quantile(boot_SD_X, c(0.025, 0.975))
ci_Y_SD <- quantile(boot_SD_Y, c(0.025, 0.975))
p_X_SD <- 2 * min(mean(boot_SD_X >= 0), mean(boot_SD_X <= 0))
p_Y_SD <- 2 * min(mean(boot_SD_Y >= 0), mean(boot_SD_Y <= 0))

# ============================================
# Plot 
# ============================================

results_plot <- data.frame(
  Group = rep(c("SWD", "SD"), each = 2),
  Axis = rep(c("X", "Y"), 2),
  Deviation = c(deviation_SWD[1], deviation_SWD[2],
                deviation_SD[1], deviation_SD[2]),
  CI_lower = c(ci_X_SWD[1], ci_Y_SWD[1],
               ci_X_SD[1], ci_Y_SD[1]),
  CI_upper = c(ci_X_SWD[2], ci_Y_SWD[2],
               ci_X_SD[2], ci_Y_SD[2]),
  p_value = c(p_X_SWD, p_Y_SWD, p_X_SD, p_Y_SD)
)

results_plot$Significant <- ifelse(results_plot$p_value < 0.05, "p < 0.05", "p ≥ 0.05")
results_plot$Significant <- factor(results_plot$Significant, levels = c("p < 0.05", "p ≥ 0.05"))

ggplot(results_plot, aes(x = Axis, y = Deviation, fill = Group)) +
  geom_bar(stat = "identity", position = position_dodge(0.7), width = 0.6, alpha = 0.7) +
  geom_errorbar(aes(ymin = CI_lower, ymax = CI_upper),
                position = position_dodge(0.7), width = 0.2, size = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", size = 0.8) +
  scale_fill_manual(values = c("SWD" = "#4DBBD5FF", "SD" = "#F39B7FFF")) +
  labs(x=NULL, y=NULL) +
  theme_test() +
  theme(
    legend.position = 'none', 
    plot.margin = unit(c(1, 1, 1, 1), "cm"),
    axis.text = element_text(size = 16),
    plot.title = element_text(size = 32, hjust = 0.1), 
    strip.text = element_text(size = 32), 
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1.2), 
  ) 
# ============================================
# Summary table 
# ============================================
Bootstrap_p = round(c(p_X_SWD, p_Y_SWD, p_X_SD, p_Y_SD), 4)
Bootstrap_p
# 0.0448 0.0116 0.0000 0.6620


# Figure 3D ---------------------------------------------------------------
## SWD vs CFS vs Health 
SWD = fread('MultipleDiseases/SWD.csv') # shift work disorder
CFS = fread('MultipleDiseases/CFS.csv')
Health = fread('MultipleDiseases/Health.csv')

SWD = fread('MultipleDiseases/SWD.csv') # shift work disorder
CFS_org = fread('MultipleDiseases/CFS.csv')
Health = fread('MultipleDiseases/Health.csv')

CFS_info = fread('MultipleDiseases/clinical_info_200_fatigue.csv')
CFS = merge(CFS_org, CFS_info[, .(participant_id, group, `FS-14`)], by.x='subject', by.y='participant_id')

CFS = CFS[group=='patient', ]

ggplot() +
  geom_point(data=SWD, aes(x = tvs_express_amplitude, y = tvs_aan_amplitude), 
             color = "#4DBBD5FF", alpha = 0.8, size = 1.5) + 
  geom_point(data=CFS, aes(x = tvs_express_amplitude, y = tvs_aan_amplitude), 
             color = "#c785e2", alpha = 0.8, size = 1.5) +
  geom_point(data=Health, aes(x = tvs_express_amplitude, y = tvs_aan_amplitude), 
             color = "gray50", alpha = 0.3, size = 1.5) +
  
  stat_ellipse(data=SWD, aes(x = tvs_express_amplitude, y = tvs_aan_amplitude), 
               level = 0.68, geom = "polygon", color = "#4DBBD5FF", fill = '#4DBBD5FF', 
               alpha = 0.3, linetype = "solid", size = 1) +
  stat_ellipse(data=CFS, aes(x = tvs_express_amplitude, y = tvs_aan_amplitude), 
               level = 0.68, geom = "polygon", color = "#c785e2", fill='#c785e2', 
               alpha = 0.3, linetype = "solid", size = 1) +
  stat_ellipse(data=Health, aes(x = tvs_express_amplitude, y = tvs_aan_amplitude), 
               level = 0.68, color = "gray50", alpha = 0.6, linetype = "solid", size = 1) +
  
  stat_centroid(data=SWD, aes(x = tvs_express_amplitude, y = tvs_aan_amplitude), 
                color = "#4DBBD5FF", size = 4, shape = 19) +
  stat_centroid(data=CFS, aes(x = tvs_express_amplitude, y = tvs_aan_amplitude), 
                color = "#c785e2", size = 4, shape = 19) +  
  stat_centroid(data=Health, aes(x = tvs_express_amplitude, y = tvs_aan_amplitude), 
                color = "black", size = 4, shape = 19) +  
  
  theme_test() +
  labs(x = NULL, y = NULL) +
  theme(
    legend.position = 'none', 
    plot.margin = unit(c(1, 1, 1, 1), "cm"),
    axis.text = element_text(size = 16),
    plot.title = element_text(size = 32, hjust = 0.1), 
    strip.text = element_text(size = 32), 
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1.2), 
  ) 

library(ICSNP)
# Q1: Whether the shape is different among CFS, SWD, and Health ?
cfs_swd = HotellingsT2(CFS[, 2:3], SWD[, 2:3]) # T.2=0.23, p=0.797
cfs_health = HotellingsT2(CFS[, 2:3], Health[, 2:3]) # T.2=6.66, p=0.001
swd_health = HotellingsT2(SWD[, 2:3], Health[, 2:3]) # T.2=5.22, p=0.005
round(p.adjust(c(cfs_swd$p.value[1,1], cfs_health$p.value[1,1], swd_health$p.value[1,1]), 
               method = 'fdr'), 3) # Pfdr = 0.797, 0.004, 0.008

# Q2: How SD and SWD deviate from Health ?
# ============================================
# Step 1：HC-defined normative space
# ============================================
X <- as.matrix(Health[, c("tvs_express_amplitude", "tvs_aan_amplitude")])
center_HC <- colMeans(X)
cov_HC <- cov(X)

scale_HC <- function(data) {
  data_mat <- as.matrix(data[, c("tvs_express_amplitude", "tvs_aan_amplitude")])
  centered <- sweep(data_mat, 2, center_HC, "-")
  # Mahalanobis transformation
  chol_decomp <- chol(cov_HC + diag(1e-10, ncol(cov_HC)))
  scaled <- centered %*% solve(chol_decomp)
  colnames(scaled) <- c("X_scaled", "Y_scaled")
  return(as.data.frame(scaled))
}

# HC --> normative space
HC_scaled <- scale_HC(Health)

# ============================================
# Step 2：The centroid deviation vector of SWD and SD
# ============================================
SWD_scaled <- scale_HC(SWD)
CFS_scaled <- scale_HC(CFS)


# Figure 3F ---------------------------------------------------------------

Health$mahal_dist <- sqrt(HC_scaled$X_scaled^2 + HC_scaled$Y_scaled^2)
SWD$mahal_dist <- sqrt(SWD_scaled$X_scaled^2 + SWD_scaled$Y_scaled^2)
CFS$mahal_dist <- sqrt(CFS_scaled$X_scaled^2 + CFS_scaled$Y_scaled^2)

cor.test(CFS$`FS-14`, CFS$mahal_dist, method = 'spearman') # p=0.0014

ggplot(CFS, aes(x = mahal_dist, y = `FS-14`)) +
  geom_point( size = 3.5, shape = 21, stroke = 0.8, color = "gray50", fill='gray50',  alpha=0.5) +
  geom_smooth( method = "lm", se = TRUE, color = "#c785e2", fill = "#c785e2", alpha=0.4, linewidth = 4 ) +
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


# Figure 3E ---------------------------------------------------------------

# Q2: How SWD and CFS deviate from Health ?
# ============================================
# Step 1：HC-defined normative space
# ============================================
X <- as.matrix(Health[, c("tvs_express_amplitude", "tvs_aan_amplitude")])
center_HC <- colMeans(X)
cov_HC <- cov(X)

scale_HC <- function(data) {
  data_mat <- as.matrix(data[, c("tvs_express_amplitude", "tvs_aan_amplitude")])
  centered <- sweep(data_mat, 2, center_HC, "-")
  # Mahalanobis transformation
  chol_decomp <- chol(cov_HC + diag(1e-10, ncol(cov_HC)))
  scaled <- centered %*% solve(chol_decomp)
  colnames(scaled) <- c("X_scaled", "Y_scaled")
  return(as.data.frame(scaled))
}

# HC --> normative space
HC_scaled <- scale_HC(Health)

# ============================================
# Step 2：The centroid deviation vector of SWD and CFS
# ============================================
SWD_scaled <- scale_HC(SWD)
CFS_scaled <- scale_HC(CFS)

center_SWD_scaled <- colMeans(SWD_scaled)
center_CFS_scaled <- colMeans(CFS_scaled)

deviation_SWD <- center_SWD_scaled
deviation_CFS <- center_CFS_scaled

# ============================================
# Bootstrap test for deviation on X and Y
# ============================================

set.seed(123)
n_boot <- 5000

# SWD：Deviations of X / Y ≠ 0 ? 
boot_SWD_X <- replicate(n_boot, {
  boot_sample <- SWD[sample(1:nrow(SWD), nrow(SWD), replace = TRUE), ]
  boot_center <- colMeans(scale_HC(boot_sample))
  return(boot_center[1])
})

boot_SWD_Y <- replicate(n_boot, {
  boot_sample <- SWD[sample(1:nrow(SWD), nrow(SWD), replace = TRUE), ]
  boot_center <- colMeans(scale_HC(boot_sample))
  return(boot_center[2])
})

# CFS：Deviations of X / Y ≠ 0 ? 
boot_CFS_X <- replicate(n_boot, {
  boot_sample <- CFS[sample(1:nrow(CFS), nrow(CFS), replace = TRUE), ]
  boot_center <- colMeans(scale_HC(boot_sample))
  return(boot_center[1])
})

boot_CFS_Y <- replicate(n_boot, {
  boot_sample <- CFS[sample(1:nrow(CFS), nrow(CFS), replace = TRUE), ]
  boot_center <- colMeans(scale_HC(boot_sample))
  return(boot_center[2])
})

# Bootstrap test for 95% CI and P value 
# SWD
ci_X_SWD <- quantile(boot_SWD_X, c(0.025, 0.975))
ci_Y_SWD <- quantile(boot_SWD_Y, c(0.025, 0.975))
p_X_SWD <- 2 * min(mean(boot_SWD_X >= 0), mean(boot_SWD_X <= 0))
p_Y_SWD <- 2 * min(mean(boot_SWD_Y >= 0), mean(boot_SWD_Y <= 0))

# CFS
ci_X_CFS <- quantile(boot_CFS_X, c(0.025, 0.975))
ci_Y_CFS <- quantile(boot_CFS_Y, c(0.025, 0.975))
p_X_CFS <- 2 * min(mean(boot_CFS_X >= 0), mean(boot_CFS_X <= 0))
p_Y_CFS <- 2 * min(mean(boot_CFS_Y >= 0), mean(boot_CFS_Y <= 0))

# ============================================
# Plot 
# ============================================

results_plot <- data.frame(
  Group = rep(c("SWD", "CFS"), each = 2),
  Axis = rep(c("X", "Y"), 2),
  Deviation = c(deviation_SWD[1], deviation_SWD[2],
                deviation_CFS[1], deviation_CFS[2]),
  CI_lower = c(ci_X_SWD[1], ci_Y_SWD[1],
               ci_X_CFS[1], ci_Y_CFS[1]),
  CI_upper = c(ci_X_SWD[2], ci_Y_SWD[2],
               ci_X_CFS[2], ci_Y_CFS[2]),
  p_value = c(p_X_SWD, p_Y_SWD, p_X_CFS, p_Y_CFS)
)

results_plot$Significant <- ifelse(results_plot$p_value < 0.05, "p < 0.05", "p ≥ 0.05")
results_plot$Significant <- factor(results_plot$Significant, levels = c("p < 0.05", "p ≥ 0.05"))

results_plot$Group = factor(results_plot$Group, levels = c('SWD', 'CFS'))
ggplot(results_plot, aes(x = Axis, y = Deviation, fill = Group)) +
  geom_bar(stat = "identity", position = position_dodge(0.7), width = 0.6, alpha = 0.7) +
  geom_errorbar(aes(ymin = CI_lower, ymax = CI_upper),
                position = position_dodge(0.7), width = 0.2, size = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", size = 0.8) +
  scale_fill_manual(values = c("SWD" = "#4DBBD5FF", "CFS" = "#c785e2") ) +
  labs(x=NULL, y=NULL) +
  theme_test() +
  theme(
    legend.position = 'none', 
    plot.margin = unit(c(1, 1, 1, 1), "cm"),
    axis.text = element_text(size = 16),
    plot.title = element_text(size = 32, hjust = 0.1), 
    strip.text = element_text(size = 32), 
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1.2), 
  ) 

Bootstrap_p = round(c(p_X_SWD, p_Y_SWD, p_X_CFS, p_Y_CFS), 4)
Bootstrap_p
# [1] 0.0448 0.0116 0.0016 0.0012
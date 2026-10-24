library(data.table)
library(dplyr)
df = fread('gradient_df.csv')
dmn_df = df[subject=='sub-001' & time=='Before' & network=='Default', ] # 91, 13
dmn_roi = dmn_df$roi # 91
pcc_roi = dmn_df[name %like% "PCC"]$roi # 20
pfc_roi = dmn_df[name %ilike% "PFC" | name %ilike% "PFCm"]$roi # 41
temp_roi = dmn_df[name %ilike% "Temp"]$roi # 25
par_roi = dmn_df[name %ilike% "Par"]$roi # 5
df_subregion = df[network=='Default', .(subject, time, roi, G2)]
subregion_map <- list(
  PCC       = pcc_roi,
  PFC       = pfc_roi,
  Temporal  = temp_roi,
  Parietal  = par_roi
)
df_subregion[, subregion := NA_character_]
for (sr in names(subregion_map)) {
  df_subregion[roi %in% subregion_map[[sr]], subregion := sr]
}

df_subregion$time = factor(df_subregion$time, levels = c('Before', 'After'))
df_delta = dcast(df_subregion, subject + roi + subregion ~ time, value.var = "G2") %>%
  .[ , delta_G2 := After-Before]
df_core = df_delta[subregion %in% c('PFC', 'PCC'), .(subject, roi, subregion, delta_G2)]

#### Signed ΔG2 and Absolute ΔG2 in PFC + PCC, and average PFC and PCC (equal weights)
df_mean <- df_core[, .(Signed = mean(delta_G2), Absolute = mean(abs(delta_G2))), by = .(subject, subregion)]
df_result <- df_mean[, .( Signed_G2 = mean(Signed), Absolute_G2 = mean(Absolute)), by = subject]

df_clinic = fread('acetate_and_pvt.csv')
cor.test( log2(df_result$Absolute_G2),  
          log2(df_clinic$rtcv_after) - log2(df_clinic$rtcv_before),  method = "pearson" ) # r=-0.50, p=0.004708
cor.test( log2(df_result$Absolute_G2),  
          log2(df_clinic$acetate_after) - log2(df_clinic$acetate_before),  method = "pearson" ) # r=0.39, p=0.0355
p.adjust(c(0.004708, 0.0355), method = 'fdr')

cor.test( df_result$Signed_G2,  
          log2(df_clinic$rtcv_after) - log2(df_clinic$rtcv_before),  method = "pearson" ) # r=0.27, p=0.1494
cor.test( df_result$Signed_G2,  
          log2(df_clinic$acetate_after) - log2(df_clinic$acetate_before),  method = "pearson" ) # r=0.18, p=0.334
p.adjust(c(0.1494, 0.334), method = 'fdr')

#### supplementary, not shown in the manuscript
df_mean_baseline <- df_subregion[time=='Before' & subregion %in% c('PFC', 'PCC'), 
                                 .(Signed = mean(G2), Absolute = mean(abs(G2))), by = .(subject, subregion)]
df_result_baseline <- df_mean_baseline[, .( Signed_G2 = mean(Signed), Absolute_G2 = mean(Absolute)), by = subject]
cor.test( log2(df_result_baseline$Absolute_G2),  
          log2(df_clinic$rtcv_before),  method = "pearson" ) # r=-0.50, p=0.004708
cor.test( log2(df_result_baseline$Absolute_G2),  
          log2(df_clinic$acetate_before),  method = "pearson" ) # r=0.39, p=0.0355
p.adjust(c(0.3545, 0.04536), method = 'fdr')

cor.test( log2(df_result_baseline$Signed_G2),  
          log2(df_clinic$rtcv_before),  method = "pearson" ) # r=0.32, p=0.534
cor.test( log2(df_result_baseline$Signed_G2),  
          log2(df_clinic$acetate_before),  method = "pearson" ) # r=0.81, p=0.049
p.adjust(c(0.2567, 0.04276), method = 'fdr')


cor.test( log2(df_result$Absolute_G2),  
          log2(df_clinic$rtmean_after) - log2(df_clinic$rtmean_before),  method = "pearson" ) # r=-0.11, p=0.5584
cor.test( log2(df_result$Signed_G2),  
          log2(df_clinic$rtmean_after) - log2(df_clinic$rtmean_before),  method = "pearson" ) # r=0.37, p=0.464


# Figure 3A

# Figure 3B, Figure 3C
library(ggplot2)
df_cor = data.table(
  #X = log2(df_clinic$rtcv_after) - log2(df_clinic$rtcv_before), 
  X = log2(df_clinic$acetate_after) - log2(df_clinic$acetate_before), 
  Y = df_result$Signed_G2
)
cor.test(df_cor$X, df_cor$Y, method='pearson')
ggplot(df_cor, aes(x = X, y = Y)) +
  geom_point( size = 3.5, shape = 21, stroke = 0.8, color = "#4A4A4A", fill = "#4A4A4A", alpha=0.8) +
  geom_smooth( method = "lm", se = TRUE, color = "#D03F51", fill = "grey60", linewidth = 4 ) +
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

# Figure 3D, Figure 3E
df_cor = data.table(
  #X = log2(df_clinic$rtcv_after) - log2(df_clinic$rtcv_before), 
  X = log2(df_clinic$acetate_after) - log2(df_clinic$acetate_before), 
  Y = log2(df_result$Absolute_G2)
)
cor.test(df_cor$X, df_cor$Y, method='pearson')
ggplot(df_cor, aes(x = X, y = Y)) +
  geom_point( size = 3.5, shape = 21, stroke = 0.8, color = "#4A4A4A", fill = "#4A4A4A", alpha=0.8) +
  geom_smooth( method = "lm", se = TRUE, color = "#D03F51", fill = "grey60", linewidth = 4 ) +
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

# Figure 3G, 
library(mediation)
mediation_df = data.table(
  X = log2(df_clinic$acetate_after) - log2(df_clinic$acetate_before), 
  M = log2(df_result$Absolute_G2), 
  Y = log2(df_clinic$rtcv_after) - log2(df_clinic$rtcv_before)
)
model_c <- lm(Y ~ X, data = mediation_df) # c
model_a <- lm(M ~ X, data = mediation_df) # a 
model_bc <- lm(Y ~ M + X, data = mediation_df) # b and c`
# Bootstrap mediation analysis
set.seed(20260818)
mediation_results <- mediate( model.m = model_a, model.y = model_bc,  treat = "X",  mediator = "M", control.value = 0,
                              treat.value = 1,  boot = TRUE,  boot.ci.type = "perc",  sims = 10000,  conf.level = 0.95)
summary(mediation_results)  

path_table <- data.frame(
  Path = c( "a: X_to_M","b: M_to_Y_adjusted_X",  "c: X_to_Y_total",  "c_prime: X_to_Y_adjusted_M" ),
  Estimate = c(  coef(model_a)["X"],coef(model_bc)["M"],coef(model_c)["X"],  coef(model_bc)["X"]),
  SE = c( summary(model_a)$coefficients["X", "Std. Error"],
          summary(model_bc)$coefficients["M", "Std. Error"],
          summary(model_c)$coefficients["X", "Std. Error"],
          summary(model_bc)$coefficients["X", "Std. Error"]
  ),
  P_value = c(
    summary(model_a)$coefficients["X", "Pr(>|t|)"],
    summary(model_bc)$coefficients["M", "Pr(>|t|)"],
    summary(model_c)$coefficients["X", "Pr(>|t|)"],
    summary(model_bc)$coefficients["X", "Pr(>|t|)"]
  )
)

print(path_table)

# Figure 3H
mediation_df = data.table(
  M = log2(df_clinic$acetate_after) - log2(df_clinic$acetate_before), 
  X = log2(df_result$Absolute_G2), 
  Y = log2(df_clinic$rtcv_after) - log2(df_clinic$rtcv_before)
)
model_c <- lm(Y ~ X, data = mediation_df) # c
model_a <- lm(M ~ X, data = mediation_df) # a 
model_bc <- lm(Y ~ M + X, data = mediation_df) # b and c`

set.seed(20260818)
mediation_results <- mediate( model.m = model_a, model.y = model_bc,  treat = "X",  mediator = "M", control.value = 0,
                              treat.value = 1,  boot = TRUE,  boot.ci.type = "perc",  sims = 10000,  conf.level = 0.95)
summary(mediation_results)  

path_table <- data.frame(
  Path = c( "a: X_to_M","b: M_to_Y_adjusted_X",  "c: X_to_Y_total",  "c_prime: X_to_Y_adjusted_M" ),
  Estimate = c(  coef(model_a)["X"],coef(model_bc)["M"],coef(model_c)["X"],  coef(model_bc)["X"]),
  SE = c( summary(model_a)$coefficients["X", "Std. Error"],
          summary(model_bc)$coefficients["M", "Std. Error"],
          summary(model_c)$coefficients["X", "Std. Error"],
          summary(model_bc)$coefficients["X", "Std. Error"]
  ),
  P_value = c(
    summary(model_a)$coefficients["X", "Pr(>|t|)"],
    summary(model_bc)$coefficients["M", "Pr(>|t|)"],
    summary(model_c)$coefficients["X", "Pr(>|t|)"],
    summary(model_bc)$coefficients["X", "Pr(>|t|)"]
  )
)

print(path_table)

moderation_df = data.table(
  delta_RTCV = log2(df_clinic$rtcv_after) - log2(df_clinic$rtcv_before), 
  absolute_G2 = log2(df_result$Absolute_G2), 
  delta_acetate = log2(df_clinic$acetate_after) - log2(df_clinic$acetate_before)
)
model_moderation <- lm(
  delta_RTCV ~ absolute_G2 * delta_acetate,
  data = moderation_df
)
summary(model_moderation)




library(data.table)
library(ggplot2)
library(magrittr)
library(lmerTest)
library(papaja)

df = fread('acetate_and_pvt.csv')

# Figure 1A
df_acetate = df %>%
  .[, .(subject, acetate_before, acetate_after)] %>%
  melt(, id.vars='subject', variable.name = "timepoint", value.name="value") %>%
  .[, .(log2value = log2(value)), by=.(subject, timepoint)] 

summary(lmer(log2value ~ timepoint + (1 | subject), data = df_acetate))
apa_print(lmer(log2value ~ timepoint + (1 | subject), data = df_acetate))

ggplot(df_acetate, aes(x = timepoint, y = log2value)) +
  geom_boxplot(width = 0.35, outlier.shape = NA, aes(color = timepoint), linewidth = 3) +
  geom_line(aes(group = subject), color = "grey70", linewidth = 0.4) +
  geom_point(size = 3, color='gray70') +
  scale_color_manual(values = c("acetate_before" = "#B783AFB3", "acetate_after"  = "#F5A673B3") ) +
  theme_test(base_size = 16) +
  theme(
    legend.position = 'none', 
    axis.title = element_text(size = 25, face = "bold", color = "black"),
    axis.text = element_text(size = 25, color = "black"),
    axis.ticks = element_line(linewidth = 0.9, color = "black"),
    axis.ticks.length = unit(0.22, "cm"),
    panel.grid = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.9 ),
    axis.line = element_blank() 
  ) +
  labs(x = NULL, y = NULL)

# Figure 1B
df_rtcv = df %>%
  .[, .(subject, rtcv_before, rtcv_after)] %>%
  melt(, id.vars='subject', variable.name = "timepoint", value.name="value") %>%
  .[, .(log2value = log2(value)), by=.(subject, timepoint)] 

summary(lmer(log2value ~ timepoint + (1 | subject), data = df_rtcv))
apa_print(lmer(log2value ~ timepoint + (1 | subject), data = df_rtcv))

ggplot(df_rtcv, aes(x = timepoint, y = log2value)) +
  geom_boxplot(width = 0.35, outlier.shape = NA, aes(color = timepoint), linewidth = 3) +
  geom_line(aes(group = subject), color = "grey70", linewidth = 0.4) +
  geom_point(size = 3, color='gray70') +
  scale_color_manual(values = c("rtcv_before" = "#B783AFB3", "rtcv_after"  = "#F5A673B3") ) +
  theme_test(base_size = 16) +
  theme(
    legend.position = 'none', 
    axis.title = element_text(size = 25, face = "bold", color = "black"),
    axis.text = element_text(size = 25, color = "black"),
    axis.ticks = element_line(linewidth = 0.9, color = "black"),
    axis.ticks.length = unit(0.22, "cm"),
    panel.grid = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.9 ),
    axis.line = element_blank() 
  ) +
  labs(x = NULL, y = NULL)

# Figure 1C
df_rtmean = df %>%
  .[, .(subject, rtmean_before, rtmean_after)] %>%
  melt(, id.vars='subject', variable.name = "timepoint", value.name="value") %>%
  .[, .(log2value = log2(value)), by=.(subject, timepoint)] 

summary(lmer(log2value ~ timepoint + (1 | subject), data = df_rtmean))
apa_print(lmer(log2value ~ timepoint + (1 | subject), data = df_rtmean))

ggplot(df_rtmean, aes(x = timepoint, y = log2value)) +
  geom_boxplot(width = 0.35, outlier.shape = NA, aes(color = timepoint), linewidth = 3) +
  geom_line(aes(group = subject), color = "grey70", linewidth = 0.4) +
  geom_point(size = 3, color='gray70') +
  scale_color_manual(values = c("rtmean_before" = "#B783AFB3", "rtmean_after"  = "#F5A673B3") ) +
  theme_test(base_size = 16) +
  theme(
    legend.position = 'none', 
    axis.title = element_text(size = 25, face = "bold", color = "black"),
    axis.text = element_text(size = 25, color = "black"),
    axis.ticks = element_line(linewidth = 0.9, color = "black"),
    axis.ticks.length = unit(0.22, "cm"),
    panel.grid = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.9 ),
    axis.line = element_blank() 
  ) +
  labs(x = NULL, y = NULL)


# Figure 1D
df_cor = data.table(
  X = log2(df$rtcv_after) - log2(df$rtcv_before),  
  Y = log2(df$acetate_after) - log2(df$acetate_before)
)
cor.test(df_cor$X, df_cor$Y) # 
ggplot(df_cor, aes(x = X, y = Y)) +
  geom_point( size = 3.5, shape = 21, stroke = 0.8, color = "#4A4A4A", fill = "#4A4A4A", alpha=0.8) +
  geom_smooth( method = "lm", se = TRUE, color = "#365083", fill = "grey60", linewidth = 4 ) +
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

# Figure 1E
df_cor = data.table(
  X = log2(df$rtmean_after) - log2(df$rtmean_before),  
  Y = log2(df$acetate_after) - log2(df$acetate_before)
)
cor.test(df_cor$X, df_cor$Y) # 
ggplot(df_cor, aes(x = X, y = Y)) +
  geom_point( size = 3.5, shape = 21, stroke = 0.8, color = "#4A4A4A", fill = "#4A4A4A", alpha=0.8) +
  geom_smooth( method = "lm", se = TRUE, color = "#365083", fill = "grey60", linewidth = 4 ) +
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

# Figure 1F
df_cor = data.table(
  X = log2(df$rtcv_before),  
  Y = log2(df$acetate_before) 
)
cor.test(df_cor$X, df_cor$Y) # 
ggplot(df_cor, aes(x = X, y = Y)) +
  geom_point( size = 3.5, shape = 21, stroke = 0.8, color = "#4A4A4A", fill = "#4A4A4A", alpha=0.8) +
  geom_smooth( method = "lm", se = TRUE, color = "#B783AF", fill = "grey60", linewidth = 4 ) +
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

# Figure 1G
df_cor = data.table(
  X = log2(df$rtmean_before),  
  Y = log2(df$acetate_before) 
)
cor.test(df_cor$X, df_cor$Y) 
ggplot(df_cor, aes(x = X, y = Y)) +
  geom_point( size = 3.5, shape = 21, stroke = 0.8, color = "#4A4A4A", fill = "#4A4A4A", alpha=0.8) +
  geom_smooth( method = "lm", se = TRUE, color = "#B783AF", fill = "grey60", linewidth = 4 ) +
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


# Difference between dependent correlations
# William's test
library(cocor)
df_cocor = data.table(
  subject = df$subject, 
  log2_acetate_before = log2(df$acetate_before), 
  log2_rtcv_before = log2(df$rtcv_before), 
  log2_rtmean_before = log2(df$rtmean_before), 
  log2_acetate_delta = log2(df$acetate_after) - log2(df$acetate_before), 
  log2_rtcv_delta = log2(df$rtcv_after) - log2(df$rtcv_before), 
  log2_rtmean_delta = log2(df$rtmean_after) - log2(df$rtmean_before)
)

result <- cocor(
  ~ log2_acetate_delta + log2_rtcv_delta |
    log2_acetate_delta + log2_rtmean_delta, 
  data = df_cocor, alternative = "two.sided", test = "williams1959"
)
print(result)

result <- cocor(
  ~ log2_acetate_before + log2_rtcv_before |
    log2_acetate_before + log2_rtmean_before, 
  data = df_cocor, alternative = "two.sided", test = "williams1959"
)
print(result)




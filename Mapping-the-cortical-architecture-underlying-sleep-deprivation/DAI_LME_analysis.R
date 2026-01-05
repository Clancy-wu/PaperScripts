library(dplyr)
library(tidyr)
library(data.table)
library(afex)
library(emmeans)
library(broom.mixed)

df_org = fread('granger_analysis_10_subregions.csv')
df = data.table()  
df$Subject = df_org$subject 
df$Condition = df_org$run
df$Condition[df_org$run == 1] = 'RW'
df$Condition[df_org$run == 2] = 'SD'
df$Region = df_org$type
df$Direction = df_org$direct
df$delta_R2 = df_org$delta_R2

afex_options(method_mixed = "S")  # Satterthwaite df to provide p-value

# 1) compute DAI
dai_df <- df %>%
  mutate(
    Subject   = factor(Subject),
    Condition = factor(Condition, levels = c("RW","SD")),
    Region    = factor(Region, levels = c('dlPu','vmPu','PPtha','cHipp','GP','Stha','dCa','cTtha','Otha','vCa')),
    Direction = factor(Direction), 
  ) %>%
  pivot_wider(names_from = Direction, values_from = delta_R2) %>%
  # DAI = top - down 
  mutate(DAI = `top` - `down`) %>%
  select(Subject, Condition, Region, DAI)


# 2) global LME： test the interaction of Condition × Region 
m_global <- mixed(
  DAI ~ Condition * Region + (1 | Subject),
  data = dai_df,
  method = "S",     
  type = 3  
)

m_global

# 3) multiple correction
emm <- emmeans(m_global$full_model, ~ Condition | Region)
contr <- contrast(emm, method = list("SD_minus_RW" = c(-1, 1)))
contr_df <- as.data.frame(contr)

# BH-FDR across regions
contr_df$q_fdr <- p.adjust(contr_df$p.value, method = "BH")
contr_df$sig_fdr <- contr_df$q_fdr < 0.05

contr_df

# 4) details of DAI for supplementary materials 
mean_sign <- dai_df %>%
  group_by(Region, Condition) %>%
  summarise(mean_DAI = mean(DAI, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = Condition, values_from = mean_DAI) %>%
  mutate(mean_sign_flip = sign(RW) * sign(SD) == -1)

contr_df

final_res <- contr_df %>%
  rename(Region = Region) %>%
  left_join(mean_sign, by = "Region")

final_res

# plot
summary_sign <- dai_df %>%
  group_by(Region, Condition) %>%
  summarise(
    mean_DAI = mean(DAI, na.rm = TRUE),
    sd_DAI   = sd(DAI, na.rm = TRUE),
    n        = sum(!is.na(DAI)),
    se_DAI   = sd_DAI / sqrt(n),
    .groups = "drop"
  )

sign_info <- summary_sign %>%
  mutate(
    sign_mean = sign(mean_DAI)
  ) %>%
  select(Region, Condition, sign_mean) %>%
  pivot_wider(
    names_from  = Condition,
    values_from = sign_mean
  ) %>%
  mutate(
    mean_sign_flip = sign(RW) * sign(SD) == -1
  )

summary_sign_full <- summary_sign %>%
  left_join(sign_info %>% select(Region, mean_sign_flip),
            by = "Region")

summary_sign_full <- summary_sign_full %>%
  mutate(
    Condition = factor(Condition, levels = c("RW", "SD")),
    Region    = factor(Region, 
                       levels = c('dlPu','vmPu','PPtha','cHipp','GP','Stha','dCa','cTtha','Otha','vCa')), 
  )
library(ggplot2)


ggplot( summary_sign_full, aes(x = Condition, y = mean_DAI, group = Region) ) +
  geom_line(color = "black", linewidth = 0.6 ) +
  # error bar
  geom_errorbar(aes(ymin = mean_DAI - se_DAI, ymax = mean_DAI + se_DAI ),
    width = 0.15, linewidth = 0.6 ) +
  # point
  geom_point(aes( shape = Condition, color = mean_DAI > 0 ),
    size = 10, stroke = 0.8 ) +
  scale_shape_manual( values = c("RW" = 16, "SD" = 17) ) +
  scale_color_manual( 
    values = c("TRUE"  = "red", "FALSE" = "blue"), 
    labels = c("FALSE" = "DAI < 0", "TRUE" = "DAI > 0")
  ) +
  facet_wrap(~ Region, nrow = 1) +
  labs( x = "", y = "DAI", shape = "Condition", color = "DAI sign" ) +
  theme_test() + 
  theme(
    legend.position = 'none', 
    plot.margin = unit(c(1, 1, 1, 1), "cm"),
    axis.title.x = element_text(size = 30),
    axis.title.y = element_text(size = 30),
    axis.text = element_text(size = 24),
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 1), 
    plot.title = element_text(size = 32, hjust = 0.1), 
    strip.text = element_text(size = 32))  

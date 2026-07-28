library(readxl)
library(tidyverse)
library(stringr)
library(ggrepel)


# Figure 6B ---------------------------------------------------------------
file_path <- "metabolomics_gene_validation/metabolomics_gene_validation_results.xlsx"
df <- read_excel(file_path, sheet = "prepost_stats")
# =========================
# 2. Prepare volcano data
# =========================
volcano_df <- df %>%
  mutate(
    mean_log2FC = as.numeric(mean_log2FC),
    wilcoxon_FDR = as.numeric(wilcoxon_FDR),
    # Avoid -log10(0)
    FDR_plot = ifelse(
      wilcoxon_FDR == 0,
      min(wilcoxon_FDR[wilcoxon_FDR > 0], na.rm = TRUE) / 2,
      wilcoxon_FDR
    ),
    neg_log10_FDR = -log10(FDR_plot),
    # Identify fatty-acid / acylcarnitine-related metabolites
    is_gene_relevant = str_detect(
      str_to_lower(category),
      "acylcarnitine|fatty-acid|fatty acid|dicarboxylic acid|lipid|cholesterol"
    ),
    is_gene_relevant = replace_na(is_gene_relevant, FALSE),
    
    # Three color groups
    point_group = case_when(
      wilcoxon_FDR < 0.05 & is_gene_relevant ~ "FDR < 0.05, gene-analysis relevant",
      wilcoxon_FDR < 0.05 & !is_gene_relevant ~ "FDR < 0.05",
      TRUE ~ "Not FDR-significant"
    ),
    
    # Label all FDR-significant metabolites
    label = ifelse(
      wilcoxon_FDR < 0.05,
      metabolite,
      NA
    )
  )

# Make sure legend order is fixed
volcano_df$point_group <- factor(
  volcano_df$point_group,
  levels = c(
    "Not FDR-significant",
    "FDR < 0.05",
    "FDR < 0.05, gene-analysis relevant"
  )
)

# =========================
# 3. Plot volcano figure
# =========================
ggplot(
  volcano_df,
  aes(
    x = mean_log2FC,
    y = neg_log10_FDR
  )
) +
  geom_point(
    aes(color = point_group),
    size = 4,
    alpha = 0.75
  ) +
  
  # Threshold lines
  geom_hline(
    yintercept = -log10(0.05),
    linetype = "dashed",
    linewidth = 0.8,
    color = "#1f77b4"
  ) +
  geom_vline(
    xintercept = 0,
    linewidth = 0.8,
    color = "#1f77b4"
  ) +
  
  scale_color_manual(
    values = c(
      "Not FDR-significant" = "#4C9ED9",
      "FDR < 0.05" = "#F28E2B",
      "FDR < 0.05, gene-analysis relevant" = "#2CA02C"
    )
  ) +
  
  labs(
    x = NULL,
    y = NULL,
    color = NULL
  ) +
  
  theme_test(base_size = 16) +
  theme(
    plot.title = element_text(
      size = 24,
      hjust = 0.5,
      face = "plain"
    ),
    axis.title = element_text(size = 18),
    axis.text = element_text(size = 15),
    
    legend.position = c(0.3, 0.9),
    legend.background = element_blank(),
    legend.key = element_blank(),
    legend.text = element_text(size = 14),
    
    axis.line = element_line(linewidth = 0.8),
    plot.margin = margin(10, 20, 10, 10)
  )


# Figure 6C ---------------------------------------------------------------

file_path <- "metabolomics_gene_validation/metabolomics_gene_validation_results.xlsx"
paired_df <- read_excel(file_path, sheet = "paired_subject_level")
# =========================
# 2. Select acylcarnitine metabolites
# =========================
acyl_df <- paired_df %>%
  filter(
    str_detect(
      str_to_lower(category),
      "acylcarnitine"
    )
  )

# =========================
# 3. Convert before/after to long format
# =========================
long_df <- acyl_df %>%
  select(
    subject,
    metabolite,
    before,
    after
  ) %>%
  pivot_longer(
    cols = c(before, after),
    names_to = "time",
    values_to = "concentration"
  ) %>%
  mutate(
    subject = as.factor(subject),
    time = factor(
      time,
      levels = c("before", "after"),
      labels = c("Before SD", "After SD")
    ),
    concentration = as.numeric(concentration)
  )

# =========================
# 4. Log2 transform and z-score within each metabolite
# =========================
# Use a small pseudocount to avoid log2(0)
pseudocount <- min(long_df$concentration[long_df$concentration > 0], na.rm = TRUE) / 2

signature_long <- long_df %>%
  mutate(
    log2_concentration = log2(concentration + pseudocount)
  ) %>%
  group_by(metabolite) %>%
  mutate(
    z_value = as.numeric(scale(log2_concentration))
  ) %>%
  ungroup()

# =========================
# 5. Calculate acylcarnitine signature for each subject and time
# =========================
signature_df <- signature_long %>%
  group_by(subject, time) %>%
  summarise(
    acylcarnitine_signature = mean(z_value, na.rm = TRUE),
    .groups = "drop"
  )

# =========================
# 6. Paired statistical test
# =========================
signature_wide <- signature_df %>%
  pivot_wider(
    names_from = time,
    values_from = acylcarnitine_signature
  )

t_res <- t.test(
  signature_wide$`After SD`,
  signature_wide$`Before SD`,
  paired = TRUE
)

print(t_res)

# =========================
# 7. Plot paired dot-line figure
# =========================
# Calculate mean values for summary line
summary_df <- signature_df %>%
  group_by(time) %>%
  summarise(
    mean_signature = mean(acylcarnitine_signature, na.rm = TRUE),
    se_signature = sd(acylcarnitine_signature, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

# Fixed jitter position, so points and paired lines align better
pd <- position_jitter(width = 0.08, height = 0, seed = 123)

ggplot(
  signature_df,
  aes(
    x = time,
    y = acylcarnitine_signature
  )
) +
  
  # Paired lines for each subject
  geom_line(
    aes(group = subject),
    color = "grey75",
    linewidth = 0.7,
    alpha = 0.75
  ) +
  
  # Individual jittered points
  geom_point(
    position = pd,
    shape = 21,
    size = 10,
    fill = "#2f6fa3ff",
    color = "black",
    stroke = 0.7,
    alpha = 0.70
  ) +
  
  # Summary line linking mean Before and mean After
  geom_line(
    data = summary_df,
    aes(
      x = time,
      y = mean_signature,
      group = 1
    ),
    inherit.aes = FALSE,
    color = "black",
    linewidth = 3
  ) +
  
  # Mean points
  geom_point(
    data = summary_df,
    aes(
      x = time,
      y = mean_signature
    ),
    inherit.aes = FALSE,
    shape = 23,
    size = 5.5,
    fill = "white",
    color = "black",
    stroke = 1.2
  ) +
  
  # Mean ± SE error bars
  geom_errorbar(
    data = summary_df,
    aes(
      x = time,
      ymin = mean_signature - se_signature,
      ymax = mean_signature + se_signature
    ),
    inherit.aes = FALSE,
    width = 0.12,
    linewidth = 0.9,
    color = "black"
  ) +
  
  labs(
    x = NULL,
    y = NULL
  ) +
  
  theme_test(base_size = 16) +
  theme(
    plot.title = element_text(
      size = 22,
      hjust = 0.5,
      face = "plain"
    ),
    axis.title.y = element_text(size = 18),
    axis.text.x = element_text(size = 16),
    axis.text.y = element_text(size = 15),
    legend.position = "none",
    axis.line = element_line(linewidth = 0.8),
    plot.margin = margin(10, 20, 10, 10)
  )


# Figure 6D ---------------------------------------------------------------
library(data.table)
# signature_wide
brain_state = fread('fMRI_HMM_24wmcfs03_results/summary_stats/hmm_summary_stats_wide.csv')
diff_df <- data.table(
  tvs_sr = brain_state[run=='SD']$sr_state3 - brain_state[run=='RW']$sr_state3, 
  acylcarnitine = signature_wide$`After SD` - signature_wide$`Before SD`
)
cor.test(diff_df$tvs_sr, diff_df$acylcarnitine, method = 'pearson')

ggplot(diff_df, aes(x = acylcarnitine, y = tvs_sr)) +
  geom_point( size = 6, shape = 21, stroke = 0.8, color = "#4A4A4A", fill = "#4A4A4A", alpha=0.5) +
  geom_smooth( method = "lm", se = TRUE, color = "#2f6fa3ff", fill = "grey60", linewidth = 4 ) +
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


# Figure 6G ---------------------------------------------------------------

library(data.table)
library(lmerTest)
library(corrplot)

validate_state_df = fread('validation_cohort_results_24WMCFS03/state_FO_SR_wide.csv')
state_names = colnames(validate_state_df)[3:12]
sdrw_t = vector(); sdrw_p = vector()

for (state_name in state_names){
  select_col = c('subject', 'session', state_name)
  state_data = validate_state_df[, ..select_col]
  colnames(state_data) = c('subject', 'run', 'value')
  ## SD-RW
  state_data$run = factor(state_data$run, levels = c('RW', 'SD'))
  state_sdrw = summary(lmer(value ~ run + (1 | subject), data = state_data))
  sdrw_t = c(sdrw_t, state_sdrw$coefficients['runSD', 't value'])
  sdrw_p = c(sdrw_p, state_sdrw$coefficients['runSD', 'Pr(>|t|)'])
}
df = data.table(
  state_name = state_names, 
  sdrw_t = sdrw_t, 
  sdrw_p = sdrw_p
)
df_tvalue = df[, .(state_name, sdrw_t)]
df_pvalue = df[, .(state_name, sdrw_p)]
df_pvalue$sdrw_pfdr = p.adjust(df_pvalue$sdrw_p, method = 'fdr')

mat_tvalue = as.matrix(df_tvalue[, 2:2])
rownames(mat_tvalue) = df_tvalue$state_name
colnames(mat_tvalue) = c('SD-RW')
mat_pvalue = as.matrix(df_pvalue[, 3:3])
rownames(mat_pvalue) = df_pvalue$state_name
colnames(mat_pvalue) = c('SD-RW')

mat_tvalue_max = copy(mat_tvalue)
mat_tvalue_max[mat_tvalue_max > 6] = 6
mat_tvalue_max[mat_tvalue_max < -6] = -6
corrplot(t(mat_tvalue_max), is.corr = FALSE, method = 'square',
         p.mat = t(mat_pvalue), insig = 'label_sig', sig.level = c(0.001, 0.01, 0.05),
         pch.cex = 2, pch.col = 'grey30', col = rev(COL2('RdBu')), col.lim = c(-6, 6), 
         tl.pos = 'n')


# Figure 6H ---------------------------------------------------------------

validate_state_df = fread('validation_cohort_results_24WMCFS03/state_FO_SR_wide.csv')
validate_state_df$session = factor(validate_state_df$session, levels = c('RW', 'SD'))
model <- lmer(state_3_FO ~ session + (1 | subject), data = validate_state_df)
summary(model) # p<0.001

ggplot(validate_state_df, aes(x = session, y = state_3_FO, fill = session)) +
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


# Figure 6I ---------------------------------------------------------------

ant_alert = fread('validation_cohort_results_24WMCFS03/ANT_alert_effect_score.csv')
ant_alert_df = melt(ant_alert, id.vars = "Subject", measure.vars = c("RW", "SD"), variable.name = "session", value.name = "value")
model <- lmer(value ~ session + (1 | Subject), data = ant_alert_df)
summary(model) # p=0.002

ggplot(ant_alert_df, aes(x = session, y = value, fill = session)) +
  geom_line(aes(group = Subject), color = "grey50", alpha = 0.5) +
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


# Figure 6J ---------------------------------------------------------------
state_alert_df = merge(validate_state_df[, .(subject=subject, session=session, state_3_FO)], 
                       ant_alert_df[, .(subject=Subject, session=session, value)], 
                       by=c('subject', 'session'), all.y = TRUE)
diff_df = data.table(
  subject = state_alert_df[session=='RW']$subject, 
  state_diff = state_alert_df[session=='SD']$state_3_FO - state_alert_df[session=='RW']$state_3_FO, 
  alert_diff = (state_alert_df[session=='SD']$value - state_alert_df[session=='RW']$value) 
)

cor.test(diff_df$state_diff, diff_df$alert_diff, method = 'pearson') # p=0.012

ggplot(diff_df, aes(x = state_diff, y = alert_diff)) +
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


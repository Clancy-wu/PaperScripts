# Result 2. Bile-acid signatures and baseline metabolic burden
# Required packages: readxl, dplyr

library(readxl)
library(dplyr)

DATA_FILE <- "psoriasis_submission_final_data.xlsx"
STATE_ORDER <- c("Preserved", "Atherogenic", "Inflammatory", "Hepatobiliary")
baseline <- read_excel(DATA_FILE, sheet = "baseline") |>
  mutate(metabolic_state = factor(metabolic_state, levels = STATE_ORDER))

# Figure 2A-TBA
library(ggplot2)
baseline$metabolic_state <- factor(baseline$metabolic_state, 
                  levels = c("Preserved", "Atherogenic", "Inflammatory", "Hepatobiliary"))
state_colors <- c("#2fa66a", "#f47a3d", "#e94b5f", "#7253b6")

ggplot(baseline %>% filter(!is.na(TBA) & !is.na(metabolic_state)), 
       aes(x = metabolic_state, y = TBA, color = metabolic_state, fill = metabolic_state)) +
  geom_boxplot(width = 0.6, outlier.shape = 16, outlier.size = 3, alpha = 0.5, linewidth = 1.2) +
  scale_y_log10(breaks = c(0.5, 1, 2, 5, 10, 20, 50), limits = c(0.5, 50)) +
  scale_color_manual(values = state_colors) +
  scale_fill_manual(values = state_colors) +
  theme_test() +
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

# Figure 2A-CG
ggplot(baseline %>% filter(!is.na(CG) & !is.na(metabolic_state)), 
       aes(x = metabolic_state, y = CG, color = metabolic_state, fill = metabolic_state)) +
  geom_boxplot(width = 0.6, outlier.shape = 16, outlier.size = 3, alpha = 0.5, linewidth = 1.2) +
  scale_y_log10(breaks = c(0.2, 0.5, 1, 2, 5, 10), limits = c(0.2, 10)) +
  scale_color_manual(values = state_colors) +
  scale_fill_manual(values = state_colors) +
  theme_test() +
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

for (marker in c("TBA", "CG")) {
  cat("\n", marker, " medians:\n", sep = "")
  print(tapply(baseline[[marker]], baseline$metabolic_state, median, na.rm = TRUE))
  kw <- kruskal.test(baseline[[marker]] ~ baseline$metabolic_state)
  cat("Kruskal-Wallis P =", format.pval(kw$p.value, digits = 6), "\n")

  d <- baseline |> filter(!is.na(.data[[marker]]), !is.na(Age), !is.na(Sex))
  d$log_marker <- log1p(d[[marker]])
  fit <- lm(log_marker ~ relevel(metabolic_state, ref = "Preserved") + Age + Sex, data = d)
  cat("Age/sex-adjusted model:\n")
  print(summary(fit)$coefficients)
}

# Figure 2B-TBA
library(ggplot2)
library(dplyr)
library(tibble)
library(broom) 

marker <- "TBA" 
d <- baseline |> filter(!is.na(.data[[marker]]), !is.na(Age), !is.na(Sex))
d$log_marker <- log1p(d[[marker]])
fit <- lm(log_marker ~ relevel(metabolic_state, ref = "Preserved") + Age + Sex, data = d)
model_results <- tidy(fit, conf.int = TRUE) |>
  filter(grepl("metabolic_state", term)) |>
  mutate(
    comparison = case_when(
      grepl("Atherogenic", term) ~ "AL vs. MP",
      grepl("Inflammatory", term) ~ "ILA vs. MP",
      grepl("Hepatobiliary", term) ~ "HML vs. MP"
    ),
    label_text = paste0("β = ", sprintf("%.3f", estimate), "\nP = ", 
                        ifelse(p.value < 0.001, "< 0.001", sprintf("%.4f", p.value))),
    comparison = factor(comparison, levels = rev(c("AL vs. MP", "ILA vs. MP", "HML vs. MP")))
  )
point_colors <- c("AL vs. MP" = "#F59B56", "ILA vs. MP" = "#E97C9D", "HML vs. MP" = "#8B78C9")

ggplot(model_results, aes(y = comparison, x = estimate, color = comparison)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black", linewidth = 1.2) +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.15, linewidth = 4) +
  geom_point(size = 10) +
  scale_color_manual(values = point_colors) +
  scale_x_continuous(limits = c(-0.2, 0.35), 
                     breaks = seq(-0.2, 0.3, by = 0.1)) +
  theme_test() +
  theme(
    legend.position = 'none', 
    axis.text.y = element_blank(), 
    axis.title = element_text(size = 25, face = "bold", color = "black"),
    axis.text = element_text(size = 25, color = "black"),
    axis.ticks = element_line(linewidth = 0.9, color = "black"),
    axis.ticks.length = unit(0.22, "cm"),
    panel.grid = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.9 ),
    axis.line = element_blank(), 
  ) +
  labs(x = NULL, y = NULL)


marker <- "CG" 
d <- baseline |> filter(!is.na(.data[[marker]]), !is.na(Age), !is.na(Sex))
d$log_marker <- log1p(d[[marker]])
fit <- lm(log_marker ~ relevel(metabolic_state, ref = "Preserved") + Age + Sex , data = d)
model_results <- tidy(fit, conf.int = TRUE) |>
  filter(grepl("metabolic_state", term)) |>
  mutate(
    comparison = case_when(
      grepl("Atherogenic", term) ~ "AL vs. MP",
      grepl("Inflammatory", term) ~ "ILA vs. MP",
      grepl("Hepatobiliary", term) ~ "HML vs. MP"
    ),
    label_text = paste0("β = ", sprintf("%.3f", estimate), "\nP = ", 
                        ifelse(p.value < 0.001, "< 0.001", sprintf("%.4f", p.value))),
    comparison = factor(comparison, levels = rev(c("AL vs. MP", "ILA vs. MP", "HML vs. MP")))
  )
point_colors <- c("AL vs. MP" = "#F59B56", "ILA vs. MP" = "#E97C9D", "HML vs. MP" = "#8B78C9")

ggplot(model_results, aes(y = comparison, x = estimate, color = comparison)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black", linewidth = 1.2) +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.15, linewidth = 4) +
  geom_point(size = 10) +
  scale_color_manual(values = point_colors) +
  scale_x_continuous(limits = c(-0.2, 0.35), 
                     breaks = seq(-0.2, 0.3, by = 0.1)) +
  theme_test() +
  theme(
    legend.position = 'none', 
    axis.text.y = element_blank(), 
    axis.title = element_text(size = 25, face = "bold", color = "black"),
    axis.text = element_text(size = 25, color = "black"),
    axis.ticks = element_line(linewidth = 0.9, color = "black"),
    axis.ticks.length = unit(0.22, "cm"),
    panel.grid = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.9 ),
    axis.line = element_blank(), 
  ) +
  labs(x = NULL, y = NULL)


# Existing baseline disease burden: post-hoc clinical characterization only.
comorbidity_cols <- c(
  dyslipidemia = "prev_dyslipidemia",
  fatty_liver = "prev_fatty_liver",
  diabetes = "prev_diabetes",
  hypertension = "prev_hypertension",
  hyperuricemia = "prev_hyperuricemia"
)
for (nm in names(comorbidity_cols)) {
  col <- comorbidity_cols[[nm]]
  tab <- table(baseline$metabolic_state, baseline[[col]])
  p <- suppressWarnings(chisq.test(tab)$p.value)
  cat(nm, "P =", format.pval(p, digits = 6), "\n")
}

baseline <- baseline |> mutate(baseline_multimorbidity = as.integer(baseline_metabolic_n >= 2))
tab <- table(baseline$metabolic_state, baseline$baseline_multimorbidity)
cat("Baseline multimorbidity P =", format.pval(chisq.test(tab)$p.value, digits = 6), "\n")
cat("Baseline multimorbidity proportions:\n")
print(tapply(baseline$baseline_multimorbidity, baseline$metabolic_state, mean, na.rm = TRUE))

# Figure 2C
plot_data <- baseline %>% 
  filter(!is.na(metabolic_state)) %>% 
  mutate(
    metabolic_state = factor(metabolic_state, 
                             levels = c("Atherogenic", "Hepatobiliary", "Inflammatory", "Preserved")) 
  ) %>%
  mutate(
    metabolic_conditions = case_when(
      baseline_metabolic_n == 0 ~ "0",
      baseline_metabolic_n == 1 ~ "1",
      baseline_metabolic_n >= 2 ~ "≥ 2"
    ),
    metabolic_conditions = factor(metabolic_conditions, levels = c("0", "1", "≥ 2"))
  ) %>%
  group_by(metabolic_state, metabolic_conditions) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(metabolic_state) %>%
  mutate( prop = count / sum(count), total_n = sum(count)  ) %>%
  ungroup() 

ggplot(plot_data, aes(x = metabolic_state, y = prop, fill = metabolic_conditions)) +
  geom_col(width = 0.7) + 
  geom_text(aes(label = ifelse(prop > 0.05, scales::percent(prop, accuracy = 0.1), "")), 
            position = position_stack(vjust = 0.5), 
            color = "black", size = 4) +
  scale_fill_manual(
    values = c("0" = "#DCEEF7", "1" = "#6BAED6", "≥ 2" = "#08519C") ) +
  scale_y_continuous(labels = scales::percent, expand = c(0, 0)) +
  theme_test() +
  theme(
    #legend.position = 'none', 
    axis.title = element_text(size = 25, face = "bold", color = "black"),
    axis.text = element_text(size = 25, color = "black"),
    axis.ticks = element_line(linewidth = 0.9, color = "black"),
    axis.ticks.length = unit(0.22, "cm"),
    panel.grid = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.9 ),
    axis.line = element_blank(), 
  ) +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 1.1), clip = "off")

# Figure 2D
library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)

plot_data <- baseline %>%
  filter(!is.na(metabolic_state)) %>% 
  select(metabolic_state, all_of(unname(comorbidity_cols))) %>%
  pivot_longer(
    cols = -metabolic_state,
    names_to = "disease_col",
    values_to = "has_disease"
  ) %>%
  mutate( disease = names(comorbidity_cols)[match(disease_col, comorbidity_cols)] ) %>%
  group_by(metabolic_state, disease) %>%
  summarise(
    prop = mean(has_disease, na.rm = TRUE), total_n = n(),  .groups = "drop")

plot_data$disease <- factor(plot_data$disease, 
              levels = c("hyperuricemia", "dyslipidemia", "fatty_liver", "diabetes", "hypertension" ))

plot_data <- plot_data %>%
  mutate(
    metabolic_state = factor(metabolic_state, 
                      levels = c("Preserved", "Inflammatory", "Hepatobiliary", "Atherogenic")) ) %>%
  mutate(
    color = recode(as.character(metabolic_state),
                   "Preserved"     = "#2fa66a",
                   "Inflammatory"  = "#e94b5f",
                   "Hepatobiliary" = "#7253b6",
                   "Atherogenic"   = "#f47a3d" )
  )

ggplot(plot_data, aes(x = disease, y = prop, fill = metabolic_state)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.75) +
  scale_fill_manual(
    values = c(
      "Preserved"     = "#2fa66a",
      "Inflammatory"  = "#e94b5f",
      "Hepatobiliary" = "#7253b6",
      "Atherogenic"   = "#f47a3d" 
    ) ) +
  scale_y_continuous( expand = c(0, 0)  ) +
  theme_test() +
  theme(
    legend.position = 'none', 
    axis.title = element_text(size = 25, face = "bold", color = "black"),
    axis.text = element_text(size = 25, color = "black"),
    axis.ticks = element_line(linewidth = 0.9, color = "black"),
    axis.ticks.length = unit(0.22, "cm"),
    panel.grid = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.9 ),
    axis.line = element_blank(), 
  ) +
  labs(x = NULL, y = NULL) 


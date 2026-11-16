# Result 5. Baseline metabolic states and subsequent metabolic multimorbidity
# Required packages: readxl, dplyr, survival

library(readxl)
library(dplyr)
library(survival)
library(survminer)


DATA_FILE <- "psoriasis_submission_final_data.xlsx"
STATE_ORDER <- c("Preserved", "Atherogenic", "Inflammatory", "Hepatobiliary")
baseline <- read_excel(DATA_FILE, sheet="baseline") |>
  mutate(metabolic_state=factor(metabolic_state, levels=STATE_ORDER))

# Figure 5A
d <- baseline |> 
  filter(follow_days > 0) |> 
  tidyr::drop_na(follow_days, event, metabolic_state)

km_data <- survfit(Surv(follow_days / 365.25, event) ~ metabolic_state, data = d) %>%
  broom::tidy() %>%
  mutate(
    cum_event = 1 - estimate,
    strata = gsub("metabolic_state=", "", strata),
    strata = factor(strata, levels = c("Preserved", "Atherogenic", "Inflammatory", "Hepatobiliary"))
  )
custom_colors = c("#2fa66a", "#f47a3d", "#e94b5f", "#7253b6")

ggplot(km_data, aes(x = time, y = cum_event, color = strata)) +
  geom_step(linewidth = 1) + 
  geom_point(data = subset(km_data, n.censor > 0), shape = 3, size = 2, stroke = 1) +
  scale_color_manual(values = custom_colors, 
                     labels = c("Preserved", "Atherogenic", "Inflammatory", "Hepatobiliary")) +
  scale_x_continuous(breaks = seq(0, 3, 1), expand = c(0, 0)) +
  scale_y_continuous(breaks = seq(0, 0.5, 0.1), expand = c(0, 0)) +
  coord_cartesian(xlim = c(0, 3), ylim = c(0, 0.5)) + 
  labs(
    x = "Follow-up time (years)",
    y = "Cumulative probability of\nmetabolic multimorbidity",
    color = NULL
  ) +
  theme_test() +
  theme(
    axis.text = element_text(color = "#3b3b6d", size = 11),
    axis.title = element_text(color = "#3b3b6d", size = 12),
    legend.position = c(0.2, 0.8),
    legend.background = element_blank(),
    legend.text = element_text(color = "#3b3b6d", size = 11),
    axis.line = element_line(color = "black"),
    plot.margin = margin(t = 10, r = 10, b = 10, l = 10)
  )

ggsurvplot(
  km_fit,
  data = d,
  fun = "event",
  risk.table = TRUE,          
  risk.table.height = 0.25,    
  break.time.by = 1,    
  xlim = c(0, 3),
  ylim = c(0, 0.5),
  palette = custom_colors,
  legend.title = "",
  legend.labs = c("Metabolically preserved", "Atherogenic lipid", 
                  "Inflammatory low-ApoAI", "Hepatobiliary mixed-lipid"),
  xlab = "Follow-up time (years)",
  ylab = "Cumulative probability of\nmetabolic multimorbidity",
  ggtheme = theme_classic()
)

# Figure 5B
fit_cox <- function(data, time, event, burden, add_tba=FALSE) {
  vars <- c(time,event,"Age","Sex","Smoking","Alcohol",burden,"metabolic_state")
  if (add_tba) vars <- c(vars,"TBA")
  d <- data |> filter(.data[[time]] > 0) |> tidyr::drop_na(all_of(vars))
  if (add_tba) {
    d <- d |> mutate(logTBA=log1p(TBA))
    f <- as.formula(paste0("Surv(",time,",",event,") ~ Age + Sex + Smoking + Alcohol + ",burden,
                           " + logTBA + relevel(metabolic_state, ref='Preserved')"))
  } else {
    f <- as.formula(paste0("Surv(",time,",",event,") ~ Age + Sex + Smoking + Alcohol + ",burden,
                           " + relevel(metabolic_state, ref='Preserved')"))
  }
  list(data=d, fit=coxph(f, data=d, ties="breslow"))
}

main <- fit_cox(baseline,"follow_days","event","baseline_metabolic_n",FALSE)
cat("Main N/events =",nrow(main$data),sum(main$data$event),"\n")
print(summary(main$fit))

main_tba <- fit_cox(baseline,"follow_days","event","baseline_metabolic_n",TRUE)
cat("\nMain + TBA:\n")
print(summary(main_tba$fit))

nonlip <- fit_cox(baseline,"follow_days_nonlipid","event_nonlipid","baseline_nonlipid_n",FALSE)
cat("\nNon-lipid sensitivity N/events =",nrow(nonlip$data),sum(nonlip$data$event_nonlipid),"\n")
print(summary(nonlip$fit))

# Figure 5B
print(summary(main$fit))
print(summary(main_tba$fit))

library(ggplot2)
library(dplyr)

plot_data <- data.frame(
  y_pos = c(9, 8, 6, 5, 3, 2, 0), 
  group = c("Atherogenic", "Atherogenic", 
            "Inflammatory", "Inflammatory", 
            "Hepatobiliary", "Hepatobiliary", 
            "TBA"),
  model = c("Main", "TBA", "Main", "TBA", "Main", "TBA", "TBA"),
  hr = c(4.84, 4.63, 3.14, 3.09, 4.48, 4.61, 0.82),
  lo = c(2.56, 2.44, 1.65, 1.63, 2.02, 2.07, 0.60),
  hi = c(9.14, 8.77, 5.96, 5.87, 9.94, 10.24, 1.11)
)

plot_data$color <- case_when(
  plot_data$group == "Atherogenic" ~ "#f47a3d",
  plot_data$group == "Inflammatory" ~ "#e94b5f",
  plot_data$group == "Hepatobiliary" ~ "#7253b6",
  plot_data$group == "TBA" ~ "#3b3b6d"
)
plot_data$shape <- ifelse(plot_data$model == "Main", 16, 1)

ggplot(plot_data, aes(y = y_pos)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "black", linewidth = 0.8) +
  geom_segment(aes(x = lo, xend = hi, y = y_pos, yend = y_pos, color = color), linewidth = 1.2) +
  geom_segment(aes(x = lo, xend = lo, y = y_pos - 0.15, yend = y_pos + 0.15, color = color), linewidth = 1.2) +
  geom_segment(aes(x = hi, xend = hi, y = y_pos - 0.15, yend = y_pos + 0.15, color = color), linewidth = 1.2) +
  geom_point(aes(x = hr, color = color, fill = ifelse(shape == 16, color, "white"), shape = factor(shape)), 
             size = 4.5, stroke = 1.5) +
  scale_x_log10(breaks = c(0.1, 1, 10, 100), limits = c(0.1, 100)) +
  scale_color_identity() +
  scale_fill_identity() +
  scale_shape_manual(values = c("16" = 16, "1" = 1)) +
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
  labs(x = NULL, y = NULL)

# Figure 5C
print(summary(nonlip$fit))

library(ggplot2)
library(dplyr)
library(tibble)
s <- summary(nonlip$fit)
coefs <- s$coefficients
conf_int <- s$conf.int

vars <- c(
  "Atherogenic"   = "relevel(metabolic_state, ref = \"Preserved\")Atherogenic",
  "Inflammatory"  = "relevel(metabolic_state, ref = \"Preserved\")Inflammatory",
  "Hepatobiliary" = "relevel(metabolic_state, ref = \"Preserved\")Hepatobiliary"
)

plot_data <- data.frame(
  y_pos = c(3, 2, 1), 
  hr = conf_int[vars, "exp(coef)"],
  lo = conf_int[vars, "lower .95"],
  hi = conf_int[vars, "upper .95"],
  color = c("#f47a3d", "#e94b5f", "#7253b6") 
)


ggplot(plot_data) +
  geom_segment(aes(x = lo, xend = hi, y = y_pos, yend = y_pos, color = color), 
               linewidth = 1.2) +
  geom_segment(aes(x = lo, xend = lo, y = y_pos - 0.15, yend = y_pos + 0.15, color = color), 
               linewidth = 1.2) +
  geom_segment(aes(x = hi, xend = hi, y = y_pos - 0.15, yend = y_pos + 0.15, color = color), 
               linewidth = 1.2) +
  geom_point(aes(x = hr, y = y_pos, color = color), 
             size = 5, shape = 16) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "black", linewidth = 0.8) +
  scale_color_identity() +
  scale_x_log10(
    breaks = c(0.1, 1, 10, 100),
    labels = c("0.1", "1", "10", "100"),
    limits = c(0.1, 100), 
    expand = c(0.02, 0) 
  ) +
  scale_y_continuous(
    limits = c(0.5, 3.8), 
    expand = c(0, 0)
  ) +
  theme_test() + 
  theme(
    axis.title = element_blank(),
    axis.text.x = element_text(size = 20, color = "black", face = "bold"), 
    axis.text.y = element_blank(), 
    axis.ticks.x = element_line(linewidth = 0.9, color = "black"),
    axis.ticks.y = element_blank(), 
    axis.ticks.length.x = unit(0.25, "cm"),
    panel.grid = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1.2), 
    plot.margin = margin(t = 20, r = 20, b = 20, l = 20) 
  ) +
  labs(x = NULL, y = NULL)




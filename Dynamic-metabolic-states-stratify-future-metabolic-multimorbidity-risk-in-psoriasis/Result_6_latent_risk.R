# Result 6. Latent risk before overt metabolic multimorbidity
# Required packages: readxl, dplyr, survival

library(readxl)
library(dplyr)
library(survival)

DATA_FILE <- "psoriasis_submission_final_data.xlsx"
STATE_ORDER <- c("Preserved", "Atherogenic", "Inflammatory", "Hepatobiliary")
baseline <- read_excel(DATA_FILE, sheet="baseline") |>
  mutate(metabolic_state=factor(metabolic_state, levels=STATE_ORDER))

run_stratum <- function(burden_value) {
  d <- baseline |>
    filter(follow_days > 0, baseline_metabolic_n == burden_value) |>
    tidyr::drop_na(Age,Sex,Smoking,Alcohol,metabolic_state,event)

  full <- coxph(Surv(follow_days,event) ~ Age + Sex + Smoking + Alcohol +
                  relevel(metabolic_state, ref="Preserved"), data=d, ties="breslow")
  reduced <- coxph(Surv(follow_days,event) ~ Age + Sex + Smoking + Alcohol,
                   data=d, ties="breslow")
  LR <- 2*(as.numeric(logLik(full))-as.numeric(logLik(reduced)))
  global_p <- pchisq(LR,df=3,lower.tail=FALSE)
  rates <- d |> group_by(metabolic_state) |>
    summarise(n=n(),events=sum(event),event_rate=mean(event),.groups="drop")

  cat("\nBaseline burden =",burden_value,"\n")
  cat("N/events =",nrow(d),sum(d$event),"\n")
  cat("Global state LRT P =",global_p,"\n")
  print(summary(full))
  cat("Observed event rates:\n"); print(rates)
}

run_stratum(0)
run_stratum(1)

# Figure 6A
library(survival)
library(ggplot2)
library(dplyr)
library(broom)

# baseline_metabolic_n == 0
d_A <- baseline |> 
  filter(follow_days > 0, baseline_metabolic_n == 0) |> 
  tidyr::drop_na(follow_days, event, metabolic_state)
km_data_A <- survfit(Surv(follow_days / 365.25, event) ~ metabolic_state, data = d_A) %>%
  tidy() %>%
  mutate(
    cum_event = 1 - estimate,
    strata = gsub("metabolic_state=", "", strata),
    strata = factor(strata, levels = c("Preserved", "Atherogenic", "Inflammatory", "Hepatobiliary"))
  )

ggplot(km_data_A, aes(x = time, y = cum_event, color = strata)) +
  geom_step(linewidth = 1) + 
  scale_color_manual(
    values = c("#2fa66a", "#f47a3d", "#e94b5f", "#7253b6"), 
    labels = c("Metabolically preserved", "Atherogenic lipid", 
               "Inflammatory low-ApoAI", "Hepatobiliary mixed-lipid")
  ) +
  scale_x_continuous(breaks = seq(0, 3, 1), expand = c(0, 0)) +
  scale_y_continuous(breaks = seq(0, 0.5, 0.1), expand = c(0, 0)) +
  coord_cartesian(xlim = c(0, 3), ylim = c(0, 0.5)) + 
  labs(
    title = "B   No baseline metabolic comorbidity",
    x = "Follow-up time (years)",
    y = "Cumulative probability of\nmetabolic multimorbidity",
    color = NULL
  ) +
  theme_test() +
  theme(
    plot.title = element_text(face = "bold", size = 13, hjust = 0),
    axis.text = element_text(color = "#3b3b6d", size = 11),
    axis.title = element_text(color = "#3b3b6d", size = 12),
    legend.position = c(0.25, 0.8), # 图例放在内部左上角
    legend.background = element_blank(),
    legend.key.size = unit(0.5, "cm"),
    legend.text = element_text(color = "#3b3b6d", size = 10),
    plot.margin = margin(t = 10, r = 10, b = 10, l = 10)
  )


# Figure 6B
d_B <- baseline |> 
  filter(follow_days > 0, baseline_metabolic_n == 1) |> 
  tidyr::drop_na(follow_days, event, metabolic_state)
km_data_B <- survfit(Surv(follow_days / 365.25, event) ~ metabolic_state, data = d_B) %>%
  tidy() %>%
  mutate(
    cum_event = 1 - estimate,
    strata = gsub("metabolic_state=", "", strata),
    strata = factor(strata, levels = c("Preserved", "Atherogenic", "Inflammatory", "Hepatobiliary"))
  )

ggplot(km_data_B, aes(x = time, y = cum_event, color = strata)) +
  geom_step(linewidth = 1) + 
  scale_color_manual(
    values = c("#2fa66a", "#f47a3d", "#e94b5f", "#7253b6"), 
    labels = c("Metabolically preserved", "Atherogenic lipid", 
               "Inflammatory low-ApoAI", "Hepatobiliary mixed-lipid")
  ) +
  scale_x_continuous(breaks = seq(0, 3, 1), expand = c(0, 0)) +
  scale_y_continuous(breaks = seq(0, 0.5, 0.1), expand = c(0, 0)) +
  coord_cartesian(xlim = c(0, 3), ylim = c(0, 0.5)) + 
  labs(
    title = "C   One baseline metabolic comorbidity",
    x = "Follow-up time (years)",
    y = "Cumulative probability of\nmetabolic multimorbidity",
    color = NULL
  ) +
  theme_test() +
  theme(
    plot.title = element_text(face = "bold", size = 13, hjust = 0),
    axis.text = element_text(color = "#3b3b6d", size = 11),
    axis.title = element_text(color = "#3b3b6d", size = 12),
    legend.position = c(0.25, 0.8), 
    legend.background = element_blank(),
    legend.key.size = unit(0.5, "cm"),
    legend.text = element_text(color = "#3b3b6d", size = 10),
    plot.margin = margin(t = 10, r = 10, b = 10, l = 10)
  )

# Figure 6C
run_stratum(0)

plot_data <- data.frame(
  y_pos = c(3, 2, 1), 
  group = c("Atherogenic", "Inflammatory", "Hepatobiliary"),
  hr = c(3.306, 2.715, 4.965),
  lo = c(1.3455, 1.1364, 1.5997), 
  hi = c(8.122, 6.486, 15.412), 
  color = c("#f47a3d", "#e94b5f", "#7253b6") 
)

ggplot(plot_data) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "black", linewidth = 0.8) +
  geom_segment(aes(x = lo, xend = hi, y = y_pos, yend = y_pos, color = color), linewidth = 1.2) +
  geom_segment(aes(x = lo, xend = lo, y = y_pos - 0.15, yend = y_pos + 0.15, color = color), linewidth = 1.2) +
  geom_segment(aes(x = hi, xend = hi, y = y_pos - 0.15, yend = y_pos + 0.15, color = color), linewidth = 1.2) +
  geom_point(aes(x = hr, y = y_pos, color = color), size = 5, shape = 16) +
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


# Figure 6D
run_stratum(1)

plot_data <- data.frame(
  y_pos = c(3, 2, 1), 
  group = c("Atherogenic", "Inflammatory", "Hepatobiliary"),
  hr = c(7.2615, 3.7661, 4.6996),
  lo = c(2.7954, 1.4129, 1.4600), 
  hi = c(18.863, 10.038, 15.128), 
  color = c("#f47a3d", "#e94b5f", "#7253b6") 
)

ggplot(plot_data) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "black", linewidth = 0.8) +
  geom_segment(aes(x = lo, xend = hi, y = y_pos, yend = y_pos, color = color), linewidth = 1.2) +
  geom_segment(aes(x = lo, xend = lo, y = y_pos - 0.15, yend = y_pos + 0.15, color = color), linewidth = 1.2) +
  geom_segment(aes(x = hi, xend = hi, y = y_pos - 0.15, yend = y_pos + 0.15, color = color), linewidth = 1.2) +
  geom_point(aes(x = hr, y = y_pos, color = color), size = 5, shape = 16) +
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

# Result 4. State-specific within-person TBA-TG coupling
# Required packages: readxl, dplyr, lme4

library(readxl)
library(dplyr)
library(lme4)

DATA_FILE <- "psoriasis_submission_final_data.xlsx"
STATE_ORDER <- c("Preserved", "Atherogenic", "Inflammatory", "Hepatobiliary")

long <- read_excel(DATA_FILE, sheet = "longitudinal") |>
  mutate(metabolic_state = factor(metabolic_state, levels = STATE_ORDER)) |>
  filter(!is.na(metabolic_state), !is.na(TBA), !is.na(TG)) |>
  mutate(logTBA = log1p(TBA), logTG = log1p(TG)) |>
  group_by(patient_SN) |>
  mutate(TBA_between = mean(logTBA), TBA_within = logTBA - TBA_between) |>
  ungroup()

long$metabolic_state <- factor(long$metabolic_state, levels = c("Preserved", "Atherogenic", "Inflammatory", "Hepatobiliary"))

full <- lmer(logTG ~ TBA_between + TBA_within * relevel(metabolic_state, ref="Preserved") +
               (1 | patient_SN), data=long, REML=FALSE,
             control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=200000)))
reduced <- lmer(logTG ~ TBA_between + TBA_within + relevel(metabolic_state, ref="Preserved") +
                  (1 | patient_SN), data=long, REML=FALSE,
                control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=200000)))

LR <- 2*(as.numeric(logLik(full)) - as.numeric(logLik(reduced)))
interaction_p <- pchisq(LR, df=3, lower.tail=FALSE)
cat("N observations =", nrow(long), "\n")
cat("N patients =", n_distinct(long$patient_SN), "\n")
cat("State x TBA-within LRT P =", interaction_p, "\n")

b <- fixef(full); V <- as.matrix(vcov(full))
normal_test <- function(est,se) c(beta=est, lo=est-1.96*se, hi=est+1.96*se, p=2*pnorm(-abs(est/se)))

cat("\nBetween-person TBA effect:\n")
print(normal_test(b["TBA_between"], sqrt(V["TBA_between","TBA_between"])))

# ============================================================
# Figure 4B. Overall tests of mixed-effects model terms
library(ggplot2)
library(dplyr)
wald_test <- function(model, terms) {
  beta <- fixef(model)
  V <- as.matrix(vcov(model))
  stopifnot(all(terms %in% names(beta)))
  est <- beta[terms]
  cov_sub <- V[terms, terms, drop = FALSE]
  chi2 <- as.numeric(
    t(est) %*% solve(cov_sub, est)
  )
  df <- length(terms)
  p <- pchisq(chi2, df = df, lower.tail = FALSE)
  data.frame( Chisq = chi2,   df = df,  P_value = p )
}

lrt_test <- function(full_model, reduced_model) {
  LR <- 2 * (
    as.numeric(logLik(full_model)) -
      as.numeric(logLik(reduced_model))
  )
  
  df <- attr(logLik(full_model), "df") -
    attr(logLik(reduced_model), "df")
  
  data.frame( Chisq = LR, df = df, P_value = pchisq(LR, df, lower.tail = FALSE)  )
}

print(names(fixef(full)))

state_terms <- grep(
  "^relevel\\(metabolic_state",
  names(fixef(full)),
  value = TRUE
)

state_terms <- state_terms[
  !grepl(":", state_terms, fixed = TRUE)
]

interaction_terms <- grep("TBA_within:",  names(fixef(full)),  value = TRUE,  fixed = TRUE)

stopifnot(
  length(state_terms) == 3,
  length(interaction_terms) == 3
)

no_between <- update( full,  . ~ . - TBA_between)

test_between <- lrt_test(full, no_between)
test_within <- wald_test(full,  terms = "TBA_within")
test_state <- wald_test(full,  terms = state_terms)

test_interaction <- lrt_test(full, reduced)
figure4B <- bind_rows(
  cbind(Term = "TBA_between",   Test = "LRT",   test_between),
  cbind(Term = "TBA_within (MP reference)",  Test = "Wald",  test_within ),
  cbind(Term = "State",  Test = "Joint Wald",  test_state ),
  cbind(Term = "TBA_within x State", Test = "LRT",  test_interaction)
)

figure4B <- figure4B |>
  mutate(
    P_label = ifelse(
      P_value < 0.001,
      "<0.001",
      sprintf("%.4f", P_value)
    )
  )

print(figure4B, row.names = FALSE)

plot_data <- figure4B |>
  mutate(
    neg_log10_p = -log10(ifelse(P_value == 0, 1e-15, P_value)),
    y_label = paste0(Term, " (", Test, ")"),
    y_label = factor(y_label, levels = rev(y_label)),
    sig = ifelse(P_value < 0.05, "Significant", "Not Significant")
  )

ggplot(plot_data, aes(y = y_label, x = neg_log10_p)) +
  geom_vline(xintercept = -log10(0.05), linetype = "dashed", color = "red", linewidth = 0.8) +
  geom_segment(aes(x = 0, xend = neg_log10_p, y = y_label, yend = y_label, color = sig), 
               linewidth = 1.2) +
  geom_point(aes(color = sig), size = 5) +
  geom_text(aes(label = paste0("P = ", P_label)), 
            hjust = -0.2, vjust = -0.8, size = 4, color = "black") +
  scale_color_manual(values = c("Significant" = "#D93F4C", "Not Significant" = "gray50")) +
  scale_x_continuous(
    limits = c(-5, 20), 
    breaks = seq(-5, 20, by = 5),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(x = NULL, y = NULL)+
  theme_test() +
  theme(
    legend.position = "none",
    axis.text.y = element_text(size = 12, color = "black"),
    axis.text.x = element_text(size = 12, color = "black"),
    axis.title.x = element_text(size = 14, face = "bold", margin = margin(t = 10)),
    axis.ticks.y = element_blank(), 
    axis.line.y = element_line(color = "black", linewidth = 0.8),
    axis.line.x = element_line(color = "black", linewidth = 0.8),
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 11, color = "gray40"),
    plot.margin = margin(t = 15, r = 30, b = 15, l = 15)
  )


# ============================================================
# State-specific within-person slopes.
base <- "TBA_within"
get_int_name <- function(state) grep(paste0("TBA_within:.*", state), names(b), value=TRUE)
slopes <- list()
for (state in STATE_ORDER) {
  cvec <- rep(0,length(b)); names(cvec) <- names(b); cvec[base] <- 1
  if (state != "Preserved") cvec[get_int_name(state)] <- 1
  est <- sum(cvec*b); se <- sqrt(as.numeric(t(cvec) %*% V %*% cvec))
  slopes[[state]] <- normal_test(est,se)
}
cat("\nState-specific within-person slopes:\n")
print(do.call(rbind, slopes))

# ============================================================
# Figure 4C. Exact mixed-effects model predictions
figure4c <- as.data.frame(do.call(rbind, slopes))
figure4c$metabolic_state <- rownames(figure4c)
rownames(figure4c) <- NULL
figure4c$metabolic_state <- factor(figure4c$metabolic_state, 
                                   levels = rev(figure4c$metabolic_state))

ggplot(figure4c, aes(x = beta, y = metabolic_state)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
  geom_errorbarh(aes(xmin = lo, xmax = hi), width = 0.2, color = "steelblue", linewidth = 0.8) +
  geom_point(aes(color = p < 0.05), size = 3.5, shape = 18) +
  scale_color_manual(values = c(`TRUE` = "firebrick", `FALSE` = "grey50"),
                     labels = c(`TRUE` = "p < 0.05", `FALSE` = "NS"),
                     name = "Significance") +
  #labs(x = "Beta (95% CI)", y = NULL, title = "Forest plot of slopes") +
  labs(x = NULL, y = NULL) +
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
  coord_cartesian(clip = "off") +
  expand_limits(y = c(0.5, nrow(figure4c) + 0.9))

# Figure 4D

# Individual interaction increments versus Preserved.
cat("\nInteraction increments vs Preserved:\n")
states <- c("Atherogenic", "Inflammatory", "Hepatobiliary")
slopes <- lapply(states, function(state) {
  nm  <- get_int_name(state)
  res <- normal_test(b[nm], sqrt(V[nm, nm]))
  tibble(
    state = state,
    beta  = as.numeric(res[grep("^beta", names(res))]),
    lo    = as.numeric(res[grep("^lo",   names(res))]),
    hi    = as.numeric(res[grep("^hi",   names(res))]),
    p     = as.numeric(res[grep("^p\\.", names(res))])
  )
}) |> bind_rows()
# Prespecified bridge to Result 3:
# average coupling increment of the two preferentially convergent states
# (Inflammatory + Hepatobiliary) versus Preserved.
cvec <- rep(0,length(b)); names(cvec) <- names(b)
cvec[get_int_name("Inflammatory")] <- 0.5
cvec[get_int_name("Hepatobiliary")] <- 0.5
est <- sum(cvec*b); se <- sqrt(as.numeric(t(cvec) %*% V %*% cvec))
cat("\nAverage Inflammatory + Hepatobiliary coupling increment vs Preserved:\n")
print(normal_test(est,se))

new_row <- normal_test(est, se) |>
  as.list() |>
  as_tibble() |>
  mutate(state = "Inflammatory+Hepatobiliary", .before = 1)
slopes <- bind_rows(slopes, new_row)

slopes$state <- factor(slopes$state, 
                       levels = c("Inflammatory+Hepatobiliary", "Hepatobiliary", 
                                  "Inflammatory", "Atherogenic" ))

ggplot(slopes, aes(x = beta, y = state)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
  geom_errorbarh(aes(xmin = lo, xmax = hi), width = 0.2, color = "steelblue", linewidth = 0.8) +
  geom_point(aes(color = p < 0.05), size = 3.5, shape = 18) +
  scale_color_manual(values = c(`TRUE` = "firebrick", `FALSE` = "grey50"),
                     labels = c(`TRUE` = "p < 0.05", `FALSE` = "NS"),
                     name = "Significance") +
  labs(x = NULL, y = NULL) +
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
  coord_cartesian(clip = "off") +
  expand_limits(y = c(0.5, nrow(figure4c) + 0.9))





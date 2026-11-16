# Result 1. Four distinct biochemical metabolic states
# Required packages: readxl, reticulate, dplyr
# Exact GMM reproduction uses scikit-learn through reticulate because the finalized model
# was originally fitted with sklearn GaussianMixture (full covariance, random_state=42, n_init=20).

library(readxl)
library(reticulate)
library(dplyr)

DATA_FILE <- "psoriasis_submission_final_data.xlsx"
STATE_ORDER <- c("Preserved", "Atherogenic", "Inflammatory", "Hepatobiliary")
FEATURES <- c("TG", "HDL-C", "LDL-C", "apoB", "apoAI", "UA", "CRP", "ALT", "GGT")
LOG_FEATURES <- c("TG", "CRP", "ALT", "GGT")

baseline <- read_excel(DATA_FILE, sheet = "baseline")
b0 <- baseline |> tidyr::drop_na(dplyr::all_of(FEATURES))

X <- as.matrix(b0[, FEATURES])
mode(X) <- "numeric"
for (v in LOG_FEATURES) X[, v] <- log1p(X[, v])

reticulate::py_require("scikit-learn")
sk_pre <- import("sklearn.preprocessing")
sk_mix <- import("sklearn.mixture")
scaler <- sk_pre$StandardScaler()$fit(X)
Z <- scaler$transform(X)

models <- list(); bics <- numeric()
for (k in 2:6) {
  gm <- sk_mix$GaussianMixture(
    n_components = as.integer(k), covariance_type = "full",
    random_state = 42L, n_init = 20L, max_iter = 500L
  )$fit(Z)
  models[[as.character(k)]] <- gm
  bics[as.character(k)] <- gm$bic(Z)
}

gmm <- models[["4"]]
component <- as.integer(gmm$predict(Z)) + 1L
posterior <- apply(gmm$predict_proba(Z), 1, max)

# Figure 1B
library(ggplot2)
bic_table = tibble(k = names(bics), BIC = as.vector(bics))
bic_table$k = as.numeric(bic_table$k)
bic_table$BIC = as.numeric(bic_table$BIC)
ggplot(bic_table, aes(x = k, y = BIC)) +
  geom_line(color = "steelblue", linewidth = 2) + 
  geom_point(color = "steelblue", size = 7) + 
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
# Figure 1C
probability_table = tibble(max_post = posterior)
ggplot(probability_table, aes(x = max_post)) +
  geom_histogram(binwidth = 0.02, fill = "steelblue",
                 color = "white", boundary = 0) +
  geom_vline(xintercept = 0.817, linetype = "dashed", color = "steelblue", linewidth = 2) +
  geom_vline(xintercept = 0.871, linetype = "solid", color = "steelblue", linewidth = 2) +  
  scale_y_continuous(expand = c(0, 0)) +
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


# Biological state naming follows the finalized manuscript rules.
tmp <- b0 |> mutate(component = component)
med <- tmp |> group_by(component) |> summarise(across(all_of(FEATURES), median, na.rm = TRUE), .groups = "drop")
inflammatory_comp <- med$component[which.max(med$CRP)]
remaining <- setdiff(med$component, inflammatory_comp)
hepatobiliary_comp <- med$component[match(max(med$GGT[med$component %in% remaining]), med$GGT)]
remaining <- setdiff(remaining, hepatobiliary_comp)
atherogenic_comp <- med$component[match(max(med$TG[med$component %in% remaining]), med$TG)]
preserved_comp <- setdiff(remaining, atherogenic_comp)
map <- setNames(c("Preserved", "Atherogenic", "Inflammatory", "Hepatobiliary"),
                c(preserved_comp, atherogenic_comp, inflammatory_comp, hepatobiliary_comp))
state_reproduced <- unname(map[as.character(component)])

cat("Complete GMM sample:", nrow(b0), "\n")
cat("BIC (2-6 states):\n")
print(round(bics, 3))
cat("Mean posterior:", mean(posterior), "\n")
cat("Median posterior:", median(posterior), "\n\n")
cat("State counts:\n")
print(table(factor(state_reproduced, levels = STATE_ORDER)))

profile <- b0 |> mutate(state_reproduced = factor(state_reproduced, levels = STATE_ORDER)) |>
  group_by(state_reproduced) |>
  summarise(across(all_of(FEATURES), median, na.rm = TRUE), .groups = "drop")
cat("\nState medians:\n")
print(profile)

# Verify agreement with the fixed analysis-ready labels stored in the Excel file.
agree <- mean(state_reproduced == b0$metabolic_state)
cat("\nAgreement with stored final labels:", agree, "\n")

# Figure 1E
library(dplyr)
library(fmsb)

var_order <- c("TG", "HDL-C", "LDL-C", "apoB", "apoAI", "UA", "CRP", "ALT", "GGT")
mat <- profile %>% select(all_of(var_order)) %>%  as.data.frame()
mat_scaled <- as.data.frame(scale(mat))

df_radar <- rbind(
  rep(2.5, ncol(mat_scaled)),
  rep(-2.5, ncol(mat_scaled)),
  mat_scaled
)
rownames(df_radar) <- c("Max", "Min", as.character(profile$state_reproduced))
colnames(df_radar) <- c("TG", "HDL-C", "LDL-C", "ApoB", "ApoAI", "UA", "CRP", "ALT", "GGT")
df_radar_fixed <- df_radar[, c(1, ncol(df_radar):2)]

# "Preserved" / "Atherogenic" / "Inflammatory" / "Hepatobiliary"
# "#2fa66a" / "#f47a3d" / "#e94b5f" / "#7253b6" / 
i <- which(profile$state_reproduced == "Preserved")
radarchart(  df_radar_fixed[c(1, 2, i + 2), ],
  axistype = 0,  pcol  = "#2fa66a", pfcol = adjustcolor("#2fa66a", alpha.f = 0.4),
  plwd  = 3,  cglcol = "grey70", cglty = 1, cglwd = 0.8,
  axislabcol = "grey40",  vlcex = 0.8 )

i <- which(profile$state_reproduced == "Atherogenic")
radarchart(  df_radar_fixed[c(1, 2, i + 2), ],
             axistype = 0,  pcol  = "#f47a3d", pfcol = adjustcolor("#f47a3d", alpha.f = 0.4),
             plwd  = 3,  cglcol = "grey70", cglty = 1, cglwd = 0.8,
             axislabcol = "grey40",  vlcex = 0.8 )

i <- which(profile$state_reproduced == "Inflammatory")
radarchart(  df_radar_fixed[c(1, 2, i + 2), ],
             axistype = 0,  pcol  = "#e94b5f", pfcol = adjustcolor("#e94b5f", alpha.f = 0.4),
             plwd  = 3,  cglcol = "grey70", cglty = 1, cglwd = 0.8,
             axislabcol = "grey40",  vlcex = 0.8 )

i <- which(profile$state_reproduced == "Hepatobiliary")
radarchart(  df_radar_fixed[c(1, 2, i + 2), ],
             axistype = 0,  pcol  = "#7253b6", pfcol = adjustcolor("#7253b6", alpha.f = 0.4),
             plwd  = 3,  cglcol = "grey70", cglty = 1, cglwd = 0.8,
             axislabcol = "grey40",  vlcex = 0.8 )


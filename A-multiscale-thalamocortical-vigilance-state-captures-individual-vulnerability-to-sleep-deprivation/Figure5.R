library(tidyverse)

# Figure 5B ---------------------------------------------------------------
input_csv <- "tvs_Neurosynth_decoding/TVS_nimare_neurosynth_decoding_results.csv"
df <- read_csv(input_csv, show_col_types = FALSE)
df_clean <- df %>%
  mutate(
    r = as.numeric(r),
    feature_clean = feature %>%
      str_replace_all("_", " ") %>%
      str_replace_all("-", " ")
  ) %>%
  filter(!is.na(r), !is.na(feature_clean))

top_pos <- df_clean %>%
  arrange(desc(r)) %>%
  slice_head(n = 5) %>%
  mutate(direction = "Positive")

top_neg <- df_clean %>%
  arrange(r) %>%
  slice_head(n = 5) %>%
  mutate(direction = "Negative")

plot_df <- bind_rows(top_neg, top_pos) %>%
  arrange(r) %>%
  mutate(
    feature_clean = factor(feature_clean, levels = feature_clean),
    r_label = sprintf("%.3f", r)
  )

ggplot(plot_df, aes(x = reorder(feature_clean, -r), y = r, fill = direction)) +
  geom_col(width = 0.75) +
  geom_hline(yintercept = 0, linewidth = 0.4) +
  geom_text(
    aes(
      label = r_label,
      vjust = ifelse(r >= 0, -0.3, 1.3)
    ),
    size = 5
  ) +
  scale_fill_manual(
    values = c("Positive" = "#dc9a8f", "Negative" = "#66aad3")
  ) +
  labs(
    y = NULL,
    x = NULL
  ) +
  theme_test() +
  theme(
    legend.position = 'none', 
    plot.margin = unit(c(1, 1, 1, 1), "cm"),
    axis.title.x = element_text(size = 30),
    axis.title.y = element_text(size = 30),
    axis.text = element_text(size = 24),
    axis.text.x = element_text(size = 15, angle = 20, hjust = 1, vjust = 1),  # 添加 vjust，值越大越靠下
    plot.title = element_text(size = 32, hjust = 0.1), 
    strip.text = element_text(size = 32), 
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1.2), 
  ) +
  coord_cartesian(clip = "off")
### vigilance-domain decoding


# Figure 5C ---------------------------------------------------------------

df_clean <- df %>%
  mutate(
    feature_raw = feature,
    feature_clean = feature %>%
      str_remove("^terms_abstract_tfidf__") %>%
      str_remove("^LDA\\d+_abstract_weight__\\d+_") %>%
      str_replace_all("_", " ") %>%
      str_replace_all("-", " ") %>%
      str_squish(),
    feature_lower = str_to_lower(feature_clean)
  )

vigilance_keywords <- c(
  "arousal",
  "alert",
  "alertness",
  "vigilance",
  "vigilant",
  "attention",
  "attentional",
  "sustained attention",
  "sleep",
  "sleepiness",
  "fatigue",
  "tired",
  "wakefulness",
  "wakeful",
  "reaction time",
  "psychomotor",
  "thalamus",
  "thalamic",
  "brainstem"
)

vigilance_df <- df_clean %>%
  filter(str_detect(feature_lower, str_c(vigilance_keywords, collapse = "|"))) %>%
  mutate(
    canonical_domain = case_when(
      str_detect(feature_lower, "arousal") ~ "Arousal",
      str_detect(feature_lower, "alert|alertness") ~ "Alertness",
      str_detect(feature_lower, "vigilance|vigilant") ~ "Vigilance",
      str_detect(feature_lower, "sustained attention") ~ "Sustained attention",
      str_detect(feature_lower, "attentional|attention") ~ "Attention",
      str_detect(feature_lower, "sleepiness|sleep") ~ "Sleep",
      str_detect(feature_lower, "fatigue|tired") ~ "Fatigue",
      str_detect(feature_lower, "wakefulness|wakeful") ~ "Wakefulness",
      str_detect(feature_lower, "reaction time|psychomotor") ~ "Reaction time",
      str_detect(feature_lower, "thalamus|thalamic") ~ "Thalamus",
      str_detect(feature_lower, "brainstem") ~ "Brainstem",
      TRUE ~ "Other vigilance-related"
    )
  ) %>%
  arrange(desc(abs(r)))

vigilance_representative <- vigilance_df %>%
  group_by(canonical_domain) %>%
  slice_max(order_by = abs(r), n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  arrange(r)

print(vigilance_representative)

ggplot(vigilance_representative, aes(x = reorder(canonical_domain, -r), y = r)) +
  geom_col(width = 0.75) +
  geom_hline(yintercept = 0, linewidth = 0.4) +
  geom_text(
    aes(
      label = round(r, 3),
      vjust = ifelse(r >= 0, -0.3, 1.3)
    ),
    size = 5
  ) +
  labs(
    y = NULL,
    x = NULL
  ) +
  scale_y_continuous(
    breaks = seq(-0.4, 0.4, by = 0.2), limits = c(-0.3, 0.25)
  )+
  theme_test() +
  theme(
    legend.position = 'none', 
    plot.margin = unit(c(1, 1, 1, 1), "cm"),
    axis.title.x = element_text(size = 30),
    axis.title.y = element_text(size = 30),
    axis.text = element_text(size = 24),
    axis.text.x = element_text(size = 15, angle = 20, hjust = 1, vjust = 1),  # 添加 vjust，值越大越靠下
    plot.title = element_text(size = 32, hjust = 0.1), 
    strip.text = element_text(size = 32), 
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1.2), 
  ) +
  coord_cartesian(clip = "off")

# Figure 5A ---------------------------------------------------------------
#### Cloud word display
library(tidyverse)
library(ggwordcloud)

df_clean <- df %>%
  mutate(
    r = as.numeric(r),
    feature_clean = feature %>%
      str_remove("^terms_abstract_tfidf__") %>%
      str_remove("^LDA\\d+_abstract_weight__\\d+_") %>%
      str_replace_all("_", " ") %>%
      str_replace_all("-", " ") %>%
      str_squish(),
    abs_r = abs(r),
    direction = ifelse(r >= 0, "Positive", "Negative")
  ) %>%
  filter(!is.na(r), !is.na(feature_clean))

# Select terms with abs(r) > 0.10
word_df <- df_clean %>%
  filter(abs_r > 0.15) %>%
  arrange(desc(abs_r))

ggplot(
  word_df,
  aes(
    label = feature_clean,
    size = abs_r,
    color = direction
  )
) +
  geom_text_wordcloud_area(
    rm_outside = TRUE,
    eccentricity = 0.65
  ) +
  scale_size_area(
    max_size = 12
  ) +
  scale_color_manual(
    values = c("Positive" = "#dc9a8f",
               "Negative" = "#66aad3")
  ) +
  labs(
    subtitle = "Neurosynth terms with |r| > 0.15",
    size = "|r|",
    color = NULL
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(size = 24, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 14, hjust = 0.5),
    legend.position = "bottom",
    panel.grid = element_blank()
  )


# Figure 5D ---------------------------------------------------------------
###### Neurotransmitter Architecture
df_transmitter <- read.csv("tvs_neurotransmitter_annotation/TVS_neurotransmitter_annotation.csv")
df_filtered <- df_transmitter %>%
  group_by(neurotransmitter) %>%
  slice_max(abs(rValue), n = 1) %>%
  ungroup() %>%
  arrange(desc(rValue))

df_filtered_plot <- df_filtered %>%
  select(source, desc, rValue, neurotransmitter) %>%
  mutate(direction = case_when(
    rValue > 0 ~ "positive",
    rValue < 0 ~ "negative",
  ))

df_filtered_plot

ggplot(df_filtered_plot, aes(x = reorder(neurotransmitter, -rValue), y = rValue, fill = direction)) +
  geom_col(width = 0.75) +
  geom_hline(yintercept = 0, linewidth = 0.4) +
  geom_text(
    aes(
      label = round(rValue, 2),
      vjust = ifelse(rValue >= 0, -0.3, 1.3)
    ),
    size = 5
  ) +
  scale_fill_manual(
    values = c("positive" = "#dc9a8f", "negative" = "#66aad3")
  ) +
  labs(
    y = NULL,
    x = NULL
  ) +
  theme_test() +
  theme(
    legend.position = 'none', 
    plot.margin = unit(c(1, 1, 1, 1), "cm"),
    axis.title.x = element_text(size = 30),
    axis.title.y = element_text(size = 30),
    axis.text = element_text(size = 24),
    axis.text.x = element_text(size = 15, angle = 20, hjust = 1, vjust = 1),  # 添加 vjust，值越大越靠下
    plot.title = element_text(size = 32, hjust = 0.1), 
    strip.text = element_text(size = 32), 
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1.2), 
  ) +
  coord_cartesian(clip = "off")


# Figure 5E ---------------------------------------------------------------
library(neuromapr)
library(freesurferformats)
library(data.table)
library(magrittr)

compare_two_whole_brain_maps <- function(map_1, map_2){
  ## Moran spectral randomization to test spatial correlations between maps. 
  bn_info <- read.csv('atlas/brainnetome.csv')
  bn_coords <- as.matrix(bn_info[, c("x.mni", "y.mni", "z.mni")]) # maps should in BN space
  ## Construct the whole-brain Euclidean distance matrix
  distmat <- as.matrix(
    dist(bn_coords, method = "euclidean")
    )# Euclidean distance in MNI space, measured in millimetres
  ## Ordinary whole-brain correlation
  ordinary_test <- cor.test(map_1, map_2, method = "pearson", use = "complete.obs" )
  observed_r <- unname(ordinary_test$estimate)
  parametric_p <- ordinary_test$p.value
  # Whole-brain Moran spatial-null test
  moran_result <- compare_maps(
    x = map_1, y = map_2, method = "pearson", null_method = "burt2020", distmat = distmat,
    n_perm = 5000L, seed = 1234L, verbose = TRUE ) 
  return(moran_result)
}

tvs = read.csv("all_dynamic_brain_states/state_3_mean.txt",  header = FALSE)
tvs_num <- tvs$V1
bn_info = read.csv('atlas/brainnetome_thalamus.csv')
all_neurotransmitter = read.csv('atlas/annotations_mni152_parc_246_BN.csv')
gabaa <- all_neurotransmitter %>%
  filter(source == 'dukart2018' & desc == 'flumazenil') 
gabaa_num <- as.numeric(gabaa[, 3:ncol(gabaa)])
a4b2 <- all_neurotransmitter %>%
  filter(source == 'hillmer2016' & desc == 'flubatine')
a4b2_num <- as.numeric(a4b2[, 3:ncol(a4b2)])

yeo7_colors_extended <- c(
  "Visual" = "#781286",
  "Somatomotor" = "#4682B4",
  "Dorsal Attention" = "#00760E",
  "Ventral Attention" = "#C43AFA",
  "Limbic" = "#DCF8A4",
  "Frontoparietal" = "#E69422",
  "Default" = "#CD3E4E",
  "Thalamus" = "#8C564B",
  "Subcortex" = "#7F7F7F"
)

#### a4b2 ~ TVS
tibble(X = a4b2_num, Y = tvs_num, Label = bn_info$Yeo_7network) %>%
ggplot(., aes(x = X, y = Y, fill=Label) ) +
  geom_point( size = 3.5, shape = 21, stroke = 0.8,  alpha=0.7) +
  geom_smooth( method = "lm", se = TRUE, color = "#dc9a8f", fill = "grey60", linewidth = 4 ) +
  scale_fill_manual(values = yeo7_colors_extended) +
  theme_test(base_size = 16) +
  theme(
    legend.position = 'none',
    axis.title = element_text(size = 18, face = "bold", color = "black"),
    axis.text = element_text(size = 15, color = "black"),
    axis.ticks = element_line(linewidth = 0.9, color = "black"),
    axis.ticks.length = unit(0.22, "cm"),
    panel.grid = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.9 ),
    axis.line = element_blank() 
  ) +
  labs(x = NULL, y = NULL)

compare_two_whole_brain_maps(a4b2_num, tvs_num) # p=0.052

# Figure 5F ---------------------------------------------------------------

#### gabaa ~ TVS
tibble(X = gabaa_num, Y = tvs_num, Label = bn_info$Yeo_7network) %>%
  ggplot(., aes(x = X, y = Y, fill=Label) ) +
  geom_point( size = 3.5, shape = 21, stroke = 0.8,  alpha=0.7) +
  geom_smooth( method = "lm", se = TRUE, color = "#66aad3", fill = "grey60", linewidth = 4 ) +
  scale_fill_manual(values = yeo7_colors_extended) +
  theme_test(base_size = 16) +
  theme(
    legend.position = 'none',
    axis.title = element_text(size = 18, face = "bold", color = "black"),
    axis.text = element_text(size = 15, color = "black"),
    axis.ticks = element_line(linewidth = 0.9, color = "black"),
    axis.ticks.length = unit(0.22, "cm"),
    panel.grid = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.9 ),
    axis.line = element_blank() 
  ) +
  labs(x = NULL, y = NULL)

compare_two_whole_brain_maps(gabaa_num, tvs_num) # p=0.096

# Figure 5G ---------------------------------------------------------------

#### gradient ~ TVS
tibble(X = scale(a4b2_num) - scale(gabaa_num), Y = tvs_num, Label = bn_info$Yeo_7network) %>%
  ggplot(., aes(x = X, y = Y, fill=Label) ) +
  geom_point( size = 3.5, shape = 21, stroke = 0.8,  alpha=0.7) +
  geom_smooth( method = "lm", se = TRUE, color = "grey40", fill = "grey60", linewidth = 4 ) +
  scale_fill_manual(values = yeo7_colors_extended) +
  theme_test(base_size = 16) +
  theme(
    #legend.position = 'none',
    axis.title = element_text(size = 18, face = "bold", color = "black"),
    axis.text = element_text(size = 15, color = "black"),
    axis.ticks = element_line(linewidth = 0.9, color = "black"),
    axis.ticks.length = unit(0.22, "cm"),
    panel.grid = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.9 ),
    axis.line = element_blank() 
  ) +
  labs(x = NULL, y = NULL)

compare_two_whole_brain_maps(scale(a4b2_num) - scale(gabaa_num), tvs_num) # p=0.032 

# Figure 5H ---------------------------------------------------------------

#### Gene Expression
pls_df <- read.csv('tvs_gene_results/pls1_roi_scores.csv')
cor.test(pls_df$pls1_score, pls_df$tvs_value, method = 'pearson') # r=0.28, 

tibble(X = pls_df$pls1_score, Y = pls_df$tvs_value, Label = bn_info$Yeo_7network) %>%
  ggplot(., aes(x = X, y = Y, fill=Label) ) +
  geom_point( size = 3.5, shape = 21, stroke = 0.8,  alpha=0.7) +
  geom_smooth( method = "lm", se = TRUE, color = "#dc9a8f", fill = "grey60", linewidth = 4 ) +
  scale_fill_manual(values = yeo7_colors_extended) +
  scale_x_continuous(limits = c(-100, 200)) +
  theme_test(base_size = 16) +
  theme(
    legend.position = 'none',
    axis.title = element_text(size = 18, face = "bold", color = "black"),
    axis.text = element_text(size = 15, color = "black"),
    axis.ticks = element_line(linewidth = 0.9, color = "black"),
    axis.ticks.length = unit(0.22, "cm"),
    panel.grid = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.9 ),
    axis.line = element_blank() 
  ) +
  labs(x = NULL, y = NULL)

# ============================================================
# Fast permutation test for TVS ~ PLS1
# One-component, one-response PLS using matrix algebra
# ============================================================
set.seed(20260729)
tvs_df <- read.csv( "tvs_gene_results/tvs_matched_rois.csv",check.names = FALSE)
expr_df <- read.csv("tvs_gene_results/ahba_expression_matched.csv", check.names = FALSE)
tvs_df <- tvs_df[  ,  !grepl("^Unnamed|^X$", colnames(tvs_df)),  drop = FALSE]
expr_roi_id <- expr_df[[1]]

X <- as.matrix(expr_df[, -1, drop = FALSE])
y <- tvs_df$tvs_value

storage.mode(X) <- "double"
y <- as.numeric(y)

X_z <- scale(X, center = TRUE, scale = TRUE)
y_z <- as.numeric(scale(y, center = TRUE, scale = TRUE))
# ------------------------------------------------------------
# Fast one-component PLS1 function
# ------------------------------------------------------------

fast_pls1 <- function(X, y) {
  w <- as.numeric(crossprod(X, y))
  w_norm <- sqrt(sum(w^2))
  w <- w / w_norm
  score <- as.numeric(X %*% w)
  # Sign alignment, matching Python script
  r <- cor(score, y)
  
  if (r < 0) {
    score <- -score
    w <- -w
    r <- -r
  }
  
  list(
    score = score,
    weight = w,
    r = r
  )
}

# ------------------------------------------------------------
# Calculate observed PLS1
# ------------------------------------------------------------
observed_fit <- fast_pls1(X_z, y_z)
r_observed <- observed_fit$r
cat("Observed PLS1 ~ TVS correlation:", r_observed, "\n")

# ============================================================
# Batch permutation
# ============================================================
set.seed(20260729)
n_perm <- 5000
batch_size <- 100
n_roi <- nrow(X_z)
n_batch <- ceiling(n_perm / batch_size)
r_null <- numeric(n_perm)
counter <- 1L
start_time <- Sys.time()
for (b in seq_len(n_batch)) {
  current_batch <- min(   batch_size,   n_perm - counter + 1L  )
  # Each column is one independently permuted TVS vector
  Y_perm <- replicate(current_batch,sample(y_z, size = n_roi, replace = FALSE)))
W_perm <- crossprod(X_z, Y_perm)
weight_norm <- sqrt(colSums(W_perm^2))
W_perm <- sweep(  W_perm,   MARGIN = 2,   STATS = weight_norm,  FUN = "/" )
# ROI-level PLS1 scores for every permutation
T_perm <- X_z %*% W_perm
# Since every column is centered, calculate correlations directly
numerator <- colSums(T_perm * Y_perm)
denominator <- sqrt(  colSums(T_perm^2) *     colSums(Y_perm^2) )
r_batch <- numerator / denominator
# Component sign is arbitrary
r_batch <- abs(r_batch)
index <- counter:(counter + current_batch - 1L)
r_null[index] <- r_batch
counter <- counter + current_batch
cat(  "Completed:",   min(b * batch_size, n_perm),  "/",  n_perm,    "\n"  )
}

end_time <- Sys.time()
cat("Total running time:", as.numeric(difftime(end_time, start_time, units = "secs")),"seconds\n")

# ============================================================
# Empirical permutation P value
# ============================================================
r_observed_abs <- abs(r_observed)
n_extreme <- sum( r_null >= r_observed_abs,  na.rm = TRUE)
p_perm <- (  n_extreme + 1) / (  n_perm + 1)

cat("\nObserved r =", r_observed, "\n") # r = 0.28
cat("Extreme permutations =", n_extreme, "\n")
cat("Number of permutations =", n_perm, "\n")
cat("Empirical permutation P =", p_perm, "\n") # P = 0.036


# Figure 5 I, J -----------------------------------------------------------

library(tidyverse)
library(stringr)
library(viridis)

gsea <- read.csv("tvs_gene_results/gsea_KEGG_2021_Human_results.csv",
                 check.names = FALSE)
#gsea <- read.csv("tvs_gene_results/gsea_Reactome_2022_results.csv",
#                 check.names = FALSE)

plot_df <- gsea %>%
  mutate(
    NES = as.numeric(NES),
    FDR = as.numeric(`FDR q-val`),
    # Extract leading-edge gene count from Tag %, e.g. "21/33" -> 21
    gene_count = as.numeric(str_extract(`Tag %`, "^[0-9]+")),
    # Avoid -log10(0) = Inf
    FDR_plot = ifelse(FDR == 0, min(FDR[FDR > 0], na.rm = TRUE) / 2, FDR),
    neg_log10_FDR = -log10(FDR_plot),
    # Clean pathway names
    Term_clean = Term %>%
      str_replace_all("_", " ") %>%
      str_squish()
  ) %>%
  filter(!is.na(NES), !is.na(FDR)) %>%
  arrange(desc(NES)) %>%
  slice_head(n = 10) %>%
  arrange(NES) %>%
  mutate(
    Term_clean = factor(Term_clean, levels = Term_clean)
  )

ggplot(
  plot_df,
  aes(
    x = NES,
    y = Term_clean,
    size = gene_count,
    color = neg_log10_FDR
  )
) +
  geom_point(alpha = 0.95) +
  geom_vline(
    xintercept = 0,
    color = "#8ab6d6",
    linewidth = 0.8
  ) +
  scale_color_viridis_c(
    option = "D",
    name = expression(-log[10](FDR))
  ) +
  scale_size_continuous(
    range = c(4, 12),
    name = "Gene count"
  ) +
  labs(
    x = "Normalized enrichment score",
    y = NULL
  ) +
  theme_classic(base_size = 14) +
  theme(
    plot.title = element_text(size = 20, hjust = 0.5),
    axis.title.x = element_text(size = 16),
    axis.text.x = element_text(size = 13),
    axis.text.y = element_text(size = 13),
    legend.title = element_text(size = 13),
    legend.text = element_text(size = 11),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.8),
    plot.margin = margin(10, 20, 10, 10)
  )


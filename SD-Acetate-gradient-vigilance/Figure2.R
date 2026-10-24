library(data.table)
library(ggplot2)
library(gifti)
library(ciftiTools)

# Figure 2A

# Figure 2B
component_df = fread('gradient_results/reference_gradient_lambda_spectrum.csv')

ggplot(component_df, aes(x = component, y = lambda)) +
  geom_line(color = "#B783AFB3", linewidth = 3) +
  geom_point(color = "#CC3E4E", size = 2) +
  theme_test(base_size = 16) +
  geom_vline(xintercept = 6, linetype = "dashed",
             color = "grey40", linewidth = 1) +
  labs(x=NULL, y=NULL) +
  theme(
    legend.position = 'none', 
    axis.title = element_text(size = 25, face = "bold", color = "black"),
    axis.text = element_text(size = 25, color = "black"),
    axis.ticks = element_line(linewidth = 0.9, color = "black"),
    axis.ticks.length = unit(0.22, "cm"),
    panel.grid = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.9 ),
    axis.line = element_blank() 
  ) 

# Figure 2C
eigenvalue_df = fread('gradient_results/reference_eigenvalues.csv')

ggplot(eigenvalue_df, aes(x = gradient, y = eigenvalue)) +
  geom_col(fill = "#B783AFB3", color = "black", width = 0.7) +
  labs(x=NULL, y=NULL) +
  theme(
    legend.position = 'none', 
    axis.title = element_text(size = 25, face = "bold", color = "black"),
    axis.text = element_text(size = 25, color = "black"),
    axis.ticks = element_line(linewidth = 0.9, color = "black"),
    axis.ticks.length = unit(0.22, "cm"),
    panel.grid = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.9 ),
    axis.line = element_blank() 
  ) 

# Figure 2D
gradient_ref = fread('gradient_results/reference_gradients.csv')
file_lh <- "Schaefer2018_7Networks_400.32k.L.label.gii"
file_rh <- "Schaefer2018_7Networks_400.32k.R.label.gii"
lh_gii <- read_gifti(file_lh)
rh_gii <- read_gifti(file_rh)
lh_data <- lh_gii$data$labels[, 1]
rh_data <- rh_gii$data$labels[, 1]
lh_data[lh_data == 0] <- NA
rh_data[rh_data == 0] <- NA
surf_lh <- load_surf(hemisphere = "left", name = "inflated")
surf_rh <- load_surf(hemisphere = "right", name = "inflated")

#### plot G1 & G2
lh_gradient = rep(NA, length(lh_data))
rh_gradient = rep(NA, length(rh_data))
for (i in seq(200)){
  lh_gradient[lh_data == i] = gradient_ref$V2[i]
}
for (i in seq(200)){
  rh_gradient[rh_data == i] = gradient_ref$V2[i+200]
}

gradient_xii <- as.xifti(cortexL = lh_gradient, cortexR = rh_gradient, 
                         surfL = surf_lh, surfR = surf_rh)
gradient_limit <- max(abs(c(lh_gradient, rh_gradient)),  na.rm = TRUE)

gradient_colors <- c(
  "#452978",  # purple
  "#3F4889",  # dark blue
  "#31678D",  # blue
  "#267F8F",  # blue-green
  "#1D9888",  # teal
  "#2CB17C",  # green
  "#5CC864",  # light green
  "#A3D936",  # yellow-green
  "#E9E418",  # yellow
  "#FDDC75",  # yellow-orange
  "#FFAB37",  # orange
  "#EC7316",  # orange-red
  "#BC4202",  # red-brown
  "#7E351A"   # dark sienna
)

# relative position of the gradient_colors
gradient_positions <- c(
  0.000,
  0.111,
  0.222,
  0.333,
  0.444,
  0.556,
  0.667,
  0.778,
  0.889,
  0.911,
  0.933,
  0.956,
  0.978,
  1.000
)

make_nonuniform_palette <- function(colors, positions, n = 256) {
  rgb_matrix <- col2rgb(colors) / 255
  output_position <- seq(0, 1, length.out = n)
  red <- approx( positions, rgb_matrix[1, ], xout = output_position )$y
  green <- approx( positions, rgb_matrix[2, ], xout = output_position )$y
  blue <- approx( positions, rgb_matrix[3, ], xout = output_position )$y
  rgb(red, green, blue)
}

gradient_palette <- make_nonuniform_palette(colors = gradient_colors,
  positions = gradient_positions,  n = 256)

view_xifti( gradient_xii,  what = "surface",  hemisphere = "both", view = "both",
  color_mode = "sequential",  colors = gradient_palette,
  zlim = c(-gradient_limit, gradient_limit),  widget = FALSE,  fname = 'Gradient_2'
)

# Figure 2F
gradient_ref = fread('reference_gradients.csv')
plot_ref = gradient_ref[, .(G1 = V1, G2 = V2)]
net_df = fread('tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-400Parcels7Networks_dseg.tsv')
net_df[, network := sub("^7Networks_(LH|RH)_", "", name)]
net_df[, network := sub("_.*$", "", network)]
##### "Vis"         "SomMot"      "DorsAttn"    "SalVentAttn" "Limbic"      "Cont"        "Default" 
plot_ref$network = net_df$network

network_colors <- c(
  Vis         = "#781180",
  SomMot      = "#3E6CB5",
  DorsAttn    = "#1E7A44",
  SalVentAttn = "#C43F8C",
  Limbic      = "#D9C9B0",
  Cont        = "#E8A33D",
  Default     = "#D03F51"
)

ggplot(plot_ref, aes(x = G2, y = G1,  color = network, fill = network)) +
  geom_point(size = 2, alpha = 0.5,) +
  stat_summary(
    aes(group = network),
    fun = mean,
    geom = "point",
    size = 5,
    shape = 21,
    fill = "white",   
    stroke = 1.2, 
  ) +
  scale_color_manual(values = network_colors) +
  scale_fill_manual(values  = network_colors) +
  labs(x=NULL, y=NULL) +
  theme(
    legend.position = 'none', 
    axis.title = element_text(size = 25, face = "bold", color = "black"),
    axis.text = element_text(size = 25, color = "black"),
    axis.ticks = element_line(linewidth = 0.9, color = "black"),
    axis.ticks.length = unit(0.22, "cm"),
    panel.grid = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.9 ),
    axis.line = element_blank() 
  ) 

# Figure 2G
net_df = fread('gradient_df.csv')
net_df = net_df[, .(subject, time, roi, G2, network)]
network_colors <- c(
  "Visual"            = "#781286",
  "Somatomotor"       = "#4682B4",
  "Dorsal Attention"  = "#00A651",
  "Ventral Attention" = "#C43CB4",
  "Limbic"            = "#F2C14E",
  "Frontoparietal"    = "#D6E3B5",
  "Default"           = "#D66A7A"
)

net_order <- net_df[
  , .(med = median(G2, na.rm = TRUE)), by = network
][order(-med), network]

net_df[, network := factor(network, levels = net_order)]
net_df[, time    := factor(time, levels = c("Before", "After"))]

ggplot(net_df, aes(x = network, y = G2, color = network,
                   fill  = interaction(network, time))) +
  geom_boxplot(
    position      = position_dodge(width = 0.75),
    width         = 0.65,
    linewidth     = 0.7,
    outlier.shape = NA,
    coef          = 1.5
  ) +
  stat_summary(
    aes(group = time),
    fun = mean, geom = "point",
    shape = 18, size = 2.5, color = "black",
    position = position_dodge(width = 0.75),
    show.legend = FALSE
  ) +
  scale_fill_manual(
    values = c(
      # Before boxes: transparent
      setNames(rep(NA, 7), paste0(levels(net_df$network), ".Before")),
      # After boxes: network colors
      setNames(network_colors, paste0(names(network_colors), ".After"))
    ),
    na.value = NA
  ) +
  scale_color_manual(values = network_colors) +
  labs(x = NULL, y = NULL) +
  theme(
    legend.position   = 'none',
    axis.title        = element_text(size = 25, face = "bold", color = "black"),
    axis.text         = element_text(size = 25, color = "black"),
    axis.ticks        = element_line(linewidth = 0.9, color = "black"),
    axis.ticks.length = unit(0.22, "cm"),
    panel.grid        = element_blank(),
    panel.border      = element_rect(color = "black", fill = NA, linewidth = 0.9),
    axis.line         = element_blank()
  )

library(lmerTest)
df = fread('gradient_df.csv')
net_df = df[, .(mean_G2 = mean(G2)), by = .(subject, time, network)]
net_df$time = factor(net_df$time, levels = c('Before', 'After'))
net_type = unique(net_df$network)
net_result_t = vector()
net_result_p = vector()
for (net in net_type){
  net_result = summary(lmer(mean_G2 ~ time + (1 | subject), data = net_df[network==net, ]))
  net_result_t = c(net_result_t, net_result$coefficients['timeAfter', 't value'])
  net_result_p = c(net_result_p, net_result$coefficients['timeAfter', 'Pr(>|t|)'])
}
data.table(
  network = net_type, 
  tvalue = net_result_t, 
  pvalue = net_result_p, 
  fdrp = p.adjust(net_result_p, method = 'fdr')
)
library(papaja)
summary(lmer(mean_G2 ~ time + (1 | subject), data = net_df[network=="Default", ]))
apa_print(lmer(mean_G2 ~ time + (1 | subject), data = net_df[network=="Default", ]))

# Figure 2H
library(magrittr)
df_subregion_org = fread('gradient_df.csv')
dmn_df = df_subregion_org[subject=='sub-001' & time=='Before' & network=='Default', ] # 91, 13
dmn_roi = dmn_df$roi # 91
pcc_roi = dmn_df[name %like% "PCC"]$roi # 20
pfc_roi = dmn_df[name %ilike% "PFC" | name %ilike% "PFCm"]$roi # 41
temp_roi = dmn_df[name %ilike% "Temp"]$roi # 25
par_roi = dmn_df[name %ilike% "Par"]$roi # 5
df_subregion = df_subregion_org[network=='Default', .(subject, time, roi, G2)]
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
df_subregion$subregion <- factor( df_subregion$subregion,
  levels = c("Parietal", "Temporal", "PFC", "PCC")
)

model_dmn <- lmer(
  G2 ~ time * subregion + (1 + time | subject) + (1 | roi),
  data = df_subregion,   REML = FALSE
)

anova(model_dmn)
summary(model_dmn)

library(emmeans)
emm_B <- emmeans( model_dmn, ~ time | subregion )
contrast_B <- contrast( emm_B,  method = list("After - Before" = c(-1, 1)), adjust = "fdr", cross.adjust = "fdr" )
plot_B_data <- as.data.frame( summary(contrast_B,  infer = c(TRUE, TRUE)) )

ggplot( plot_B_data, aes( x = subregion, y = estimate,  color = subregion )) +
  geom_hline( yintercept = 0,  linetype = "dashed",  color = "grey50", linewidth = 2 ) +
  geom_pointrange(aes(ymin = asymp.LCL,  ymax = asymp.UCL  ),  linewidth = 2, size = 3) +
  labs(x = NULL, y = NULL) +
  theme(
    # legend.position   = 'none',
    axis.title        = element_text(size = 25, face = "bold", color = "black"),
    axis.text         = element_text(size = 25, color = "black"),
    axis.ticks        = element_line(linewidth = 0.9, color = "black"),
    axis.ticks.length = unit(0.22, "cm"),
    panel.grid        = element_blank(),
    panel.border      = element_rect(color = "black", fill = NA, linewidth = 0.9),
    axis.line         = element_blank()
  )

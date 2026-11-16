# Result 3. Continuous-time multi-state transition model
# Required packages: readxl, dplyr, tidyr, expm

library(readxl)
library(dplyr)
library(tidyr)
library(expm)

DATA_FILE <- "psoriasis_submission_final_data.xlsx"
STATE_ORDER <- c("Preserved", "Atherogenic", "Inflammatory", "Hepatobiliary")
S <- setNames(1:4, STATE_ORDER)

long <- read_excel(DATA_FILE, sheet = "longitudinal") |>
  mutate(
    date = as.Date(date),
    metabolic_state = factor(metabolic_state, levels = STATE_ORDER)
  ) |>
  filter(!is.na(metabolic_state)) |>
  arrange(patient_SN, date)

length(unique(long$patient_SN)) # 2558 patients >=1 visit
sum(table(long$patient_SN)>1) # 837 patients >=2 visit

tr <- long |>
  group_by(patient_SN) |>
  mutate(next_state = lead(metabolic_state), next_date = lead(date)) |>
  ungroup() |>
  filter(!is.na(next_state), !is.na(next_date)) |>
  mutate(interval_days = as.numeric(next_date - date)) |>
  filter(interval_days > 0)

cat("Visits =", nrow(long), "\n")
cat("Patients =", n_distinct(long$patient_SN), "\n")
cat("Patients with >=2 visits =", n_distinct(tr$patient_SN), "\n")
cat("Adjacent transitions =", nrow(tr), "\n")
cat("Interval median/IQR =", median(tr$interval_days), quantile(tr$interval_days, .25), quantile(tr$interval_days, .75), "\n")

obs <- prop.table(table(tr$metabolic_state, tr$next_state), 1)
cat("\nObserved transition proportions:\n")
print(obs)

# Figure 3B
library(corrplot)
target_order <- c("Preserved", "Atherogenic", "Inflammatory", "Hepatobiliary")
mat <- as.matrix(obs)
mat <- mat[target_order, target_order]
my_colors <- colorRampPalette(c("white", "#6D9EC1", "#08519C"))(200)

corrplot(
  mat,
  method = "color",         
  type = "full",            
  order = "original",       
  addCoef.col = "black",    
  number.cex = 1,         
  tl.col = "black",         
  tl.srt = 45,              
  tl.cex = 1,               
  col = my_colors, cl.pos = "n", 
  is.corr = FALSE,          
  cl.lim = c(0, 1),         
  cl.cex = 0.8, 
  addgrid.col = "black",  addgrid.lwd = 1
)

colorlegend(colbar = my_colors,  labels = c("0%", "25%", "50%", "75%", "100%"), 
  at = c(0, 0.25, 0.5, 0.75, 1),  vertical = TRUE,  
  xlim = c(1.05, 1.15),   ylim = c(0.2, 0.8),  cex = 0.9 )


# Continuous-time homogeneous Markov model: exact likelihood analogue of the finalized analysis.
i_arr <- unname(S[as.character(tr$metabolic_state)])
j_arr <- unname(S[as.character(tr$next_state)])
dt_year <- tr$interval_days / 365.25
pairs <- which(row(matrix(0,4,4)) != col(matrix(0,4,4)), arr.ind = TRUE)
pairs <- pairs[order(pairs[,1], pairs[,2]), , drop = FALSE]

counts <- matrix(0,4,4); exposure <- rep(0,4)
for (z in seq_along(i_arr)) {
  exposure[i_arr[z]] <- exposure[i_arr[z]] + dt_year[z]
  if (i_arr[z] != j_arr[z]) counts[i_arr[z], j_arr[z]] <- counts[i_arr[z], j_arr[z]] + 1
}
initial_rates <- apply(pairs, 1, function(x) max(counts[x[1],x[2]] / max(exposure[x[1]],1e-8), 1e-6))

agg <- tr |> transmute(i=i_arr, j=j_arr, dt=dt_year) |> count(i,j,dt,name="n")

theta_to_Q <- function(theta) {
  Q <- matrix(0,4,4)
  rates <- exp(theta)
  for (z in seq_len(nrow(pairs))) Q[pairs[z,1], pairs[z,2]] <- rates[z]
  diag(Q) <- -rowSums(Q)
  Q
}
neg_loglik <- function(theta) {
  Q <- theta_to_Q(theta); total <- 0
  for (z in seq_len(nrow(agg))) {
    P <- expm(Q * agg$dt[z])
    prob <- max(P[agg$i[z], agg$j[z]], 1e-300)
    total <- total - agg$n[z] * log(prob)
  }
  total
}

set.seed(42)
start0 <- log(initial_rates)
starts <- list(start0)
for (z in 1:4) starts[[z+1]] <- start0 + rnorm(length(start0), 0, 0.5)
fits <- lapply(starts, function(st) optim(st, neg_loglik, method="L-BFGS-B",
                                         control=list(maxit=2500, factr=1e4)))
best <- fits[[which.min(sapply(fits, `[[`, "value"))]]
Q <- theta_to_Q(best$par)
rownames(Q) <- colnames(Q) <- STATE_ORDER
P90 <- expm(Q * (90/365.25)); rownames(P90) <- colnames(P90) <- STATE_ORDER

# Figure 3D
library(corrplot)
target_order <- c("Preserved", "Atherogenic", "Inflammatory", "Hepatobiliary")
P90_mat <- as.matrix(P90)
P90_mat <- P90_mat[target_order, target_order]
my_colors <- colorRampPalette(c("white", "#6D9EC1", "#08519C"))(200)

corrplot(
  P90_mat,
  method = "color",         
  type = "full",            
  order = "original",       
  addCoef.col = "black",    
  number.cex = 1,         
  tl.col = "black",         
  tl.srt = 45,              
  tl.cex = 1,               
  col = my_colors, cl.pos = "n", 
  is.corr = FALSE, 
  cl.cex = 0.8, 
  addgrid.col = "black"
)

colorlegend(colbar = my_colors,  labels = c("0%", "25%", "50%", "75%", "100%"), 
            at = c(0, 0.25, 0.5, 0.75, 1),  vertical = TRUE,  
            xlim = c(1.05, 1.15),   ylim = c(0.2, 0.8),  cex = 0.9 )


eig <- eigen(t(Q))
idx <- which.min(abs(eig$values))
pi <- Re(eig$vectors[,idx]); pi <- pi / sum(pi); names(pi) <- STATE_ORDER

cat("\nQ matrix (per year):\n"); print(Q)
cat("\n90-day transition probabilities:\n"); print(P90)
cat("\nStationary distribution:\n"); print(pi)

transition_contrast <- function(start, dest1, dest2) {
  sub <- tr |> filter(metabolic_state == start, next_state != start)
  n1 <- sum(sub$next_state == dest1); n2 <- sum(sub$next_state == dest2)
  ratio <- n1/n2; se <- sqrt(1/n1 + 1/n2); z <- log(ratio)/se
  c(n1=n1,n2=n2,ratio=ratio,lo=exp(log(ratio)-1.96*se),hi=exp(log(ratio)+1.96*se),
    p=2*pnorm(-abs(z)))
}
cat("\nFormal transition-frequency contrasts:\n")
print(transition_contrast("Inflammatory","Atherogenic","Preserved"))
print(transition_contrast("Inflammatory","Atherogenic","Hepatobiliary"))
print(transition_contrast("Hepatobiliary","Atherogenic","Preserved"))
print(transition_contrast("Hepatobiliary","Atherogenic","Inflammatory"))

# Figue 3E
library(ggplot2)
library(dplyr)

contra_1 = transition_contrast("Inflammatory","Atherogenic","Preserved")
contra_2 = transition_contrast("Inflammatory","Atherogenic","Hepatobiliary")
contra_3 = transition_contrast("Hepatobiliary","Atherogenic","Preserved")
contra_4 = transition_contrast("Hepatobiliary","Atherogenic","Inflammatory")
merged_tibble <-bind_rows(contra_1, contra_2, contra_3, contra_4)
merged_tibble$contrast = c('contra_1', 'contra_2', 'contra_3', 'contra_4')
merged_tibble$contrast = factor(merged_tibble$contrast, 
                                levels = rev(c('contra_1', 'contra_2', 'contra_3', 'contra_4')))

ggplot(merged_tibble, aes(y = contrast, x = ratio)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "#2C5985", linewidth = 1) +
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.15, linewidth = 1, color = "#D93F4C") +
  geom_point(size = 5, color = "#D93F4C") +
  scale_x_log10(
    limits = c(0.1, 100), 
    breaks = c(0.1, 1, 10, 100),
    labels = c("0.1", "1", "10", "100")
  ) +
  annotate(
    "segment", 
    x = 10, xend = 100,  # Arrow goes from 10 to 100
    y = -0.5, yend = -0.5, # Position below the lowest row
    arrow = arrow(length = unit(0.15, "cm"), type = "closed"),
    color = "#2C5985", 
    linewidth = 1
  ) +
  theme_test() +
  theme(
    axis.text.x = element_text(color = "#2C5985", size = 12, face = "bold", margin = margin(t = 5)),
    axis.ticks.x = element_line(color = "#2C5985", linewidth = 0.8),
    axis.ticks.length = unit(0.2, "cm"),
    axis.line.x = element_line(color = "#2C5985", linewidth = 0.8),
    plot.margin = margin(10, 20, 30, 10)
  )

# Figure 3F
eig <- eigen(t(Q))
idx <- which.min(abs(eig$values))
pi <- Re(eig$vectors[,idx]); pi <- pi / sum(pi); names(pi) <- STATE_ORDER

cat("\nStationary distribution:\n"); print(pi)

plot_data <- tibble(
  state = names(pi),
  percentage = as.numeric(pi) 
) |>
  mutate(state = factor(state, levels = c("Preserved", "Atherogenic", "Inflammatory", "Hepatobiliary")))

bar_colors <- c(
  "Preserved" = "#2fa66a",    
  "Atherogenic" = "#f47a3d",    
  "Inflammatory" = "#e94b5f",   
  "Hepatobiliary" = "#7253b6"   
)

ggplot(plot_data, aes(x = state, y = percentage, fill = state)) +
  geom_col(width = 0.65) +
  geom_text(
    aes(label = sprintf("%.2f", percentage)), 
    vjust = -0.8, 
    color = "#1B2B4B",
    fontface = "bold",
    size = 5
  ) +
  scale_fill_manual(values = bar_colors) +
  scale_y_continuous(
    limits = c(0, 1),
    expand = expansion(mult = c(0, 0)) 
  ) +
  theme_test() +
  theme(
    legend.position = 'none', 
    axis.title = element_text(size = 25, face = "bold", color = "black"),
    axis.text = element_text(size = 25, color = "black"),
    axis.ticks = element_line(linewidth = 0.9, color = "black"),
    axis.ticks.length = unit(0.22, "cm"),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.9 ),
    axis.line = element_blank() 
  ) +
  labs(x = NULL, y = NULL)





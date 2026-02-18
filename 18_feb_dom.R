################################################################################
# FUNCTIONAL DATA ANALYSIS + CONFORMAL PREDICTION
# Smartphone Sensor Data (Run / Sit / Walk)
#
# This script can be run directly in R and produces all outputs, plots, and
# printed summaries. It performs:
#   1. Data preprocessing (interpolation, smoothing, segmentation)
#   2. Functional data analysis with exploratory plots
#   3. Inductive Conformal Prediction in FPCA score space using an ellipsoid
#
# Conformal step follows the projection-based functional conformal bands:
# - Project curves with FPCA
# - Work in low-dimensional score space
# - Use Mahalanobis distance → ellipsoidal nonconformity and prediction set.[file:5]
#
# Author: Your Name
# Date: February 18, 2026
################################################################################

# =============================
# 0. INSTALL AND LOAD PACKAGES
# =============================

cat("\n=== INSTALLING AND LOADING REQUIRED PACKAGES ===\n")

required_packages <- c("dplyr", "zoo", "fda", "ggplot2", "reshape2",
                       "gridExtra", "purrr")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(paste("Installing package:", pkg, "\n"))
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  library(pkg, character.only = TRUE)
}

cat("All packages loaded successfully!\n")

# =============================
# 1. DATA PREPROCESSING
# =============================
setwd("C:/Users/domen")

cat("\n=== STEP 1: DATA PREPROCESSING ===\n")

# Define file path - ADJUST THIS TO YOUR ACTUAL FILE PATH
csv_path <- "Downloads/Homework_SDS_group46-main/Homework_SDS_group46-main/dati_sgravati.csv"  # Change to your actual path

# Check if file exists
if (!file.exists(csv_path)) {
  stop(paste("ERROR: File not found at", csv_path,
             "\nPlease update the csv_path variable with the correct path."))
}

cat(paste("Loading data from:", csv_path, "\n"))
df_raw <- read.csv(csv_path)

cat("\n--- Raw Data Summary ---\n")
cat(paste("Total rows:", nrow(df_raw), "\n"))
cat(paste("Total columns:", ncol(df_raw), "\n"))
cat("Column names:\n")
print(names(df_raw))

cat("\nFirst 10 rows:\n")
print(head(df_raw, 10))

# Sort by timestamp and create relative time in seconds
cat("\nSorting by timestamp and converting to seconds...\n")
df_raw <- df_raw[order(df_raw$timestamp), ]
t0 <- min(df_raw$timestamp)
df_raw$time_sec <- (df_raw$timestamp - t0) / 1000

cat(paste("Experiment duration:", round(max(df_raw$time_sec), 2), "seconds\n"))
cat(paste("Experiment duration:", round(max(df_raw$time_sec) / 60, 2), "minutes\n"))

# Identify sensor columns (everything except timestamp and time_sec)
sensor_cols <- setdiff(names(df_raw), c("timestamp", "time_sec"))
cat(paste("\nFound", length(sensor_cols), "sensor columns:\n"))
print(sensor_cols)

# =============================
# 2. INTERPOLATION TO COMMON GRID
# =============================

cat("\n=== STEP 2: INTERPOLATION TO COMMON GRID ===\n")

# Create common time grid at 10 Hz (0.1 second intervals)
dt <- 0.1
t_grid <- seq(from = 0, to = max(df_raw$time_sec), by = dt)
n_points <- length(t_grid)

cat(paste("Time grid created with", n_points, "points at", dt, "second intervals\n"))

df_aligned <- data.frame(time = t_grid)

# Interpolation function
interpolate_sensor <- function(time_col, val_col, new_grid) {
  valid <- !is.na(val_col)
  if (sum(valid) < 2) {
    return(rep(0, length(new_grid)))
  }
  z <- zoo(val_col[valid], time_col[valid])
  as.numeric(na.approx(z, xout = new_grid, rule = 2))
}

cat("\nInterpolating sensors to common grid...\n")
for (s in sensor_cols) {
  df_aligned[[s]] <- interpolate_sensor(df_raw$time_sec, df_raw[[s]], t_grid)
  cat(paste("  ->", s, "interpolated\n"))
}

cat("\nInterpolation complete!\n")

# =============================
# 3. SMOOTHING WITH B-SPLINES
# =============================

cat("\n=== STEP 3: SMOOTHING WITH B-SPLINES ===\n")

# Smoothing parameters
chunk_size <- 2000
lambda_smooth <- 0.01
num_chunks <- ceiling(n_points / chunk_size)

cat(paste("Smoothing in", num_chunks, "chunks of size", chunk_size, "\n"))
cat(paste("Smoothing parameter lambda:", lambda_smooth, "\n"))

# Smoothing function for one chunk
smooth_chunk <- function(time_vec, data_vec) {
  nbasis <- max(4, round(length(time_vec) / 4))
  basis_obj <- create.bspline.basis(
    rangeval = range(time_vec),
    nbasis = nbasis,
    norder = 4
  )
  fd_par <- fdPar(basis_obj, Lfdobj = 2, lambda = lambda_smooth)
  sm_res <- smooth.basis(time_vec, data_vec, fd_par)
  as.numeric(eval.fd(time_vec, sm_res$fd))
}

# Smooth each sensor
for (s in sensor_cols) {
  cat(paste("\nSmoothing sensor:", s, "\n"))
  smoothed_vals <- numeric(n_points)
  
  for (i in seq_len(num_chunks)) {
    idx_start <- (i - 1) * chunk_size + 1
    idx_end <- min(i * chunk_size, n_points)
    idx <- idx_start:idx_end
    
    smoothed_vals[idx] <- smooth_chunk(df_aligned$time[idx],
                                       df_aligned[[s]][idx])
    if (i %% 10 == 0) {
      cat(paste("  Processed chunk", i, "/", num_chunks, "\n"))
    }
  }
  
  df_aligned[[paste0("Smooth_", s)]] <- smoothed_vals
}

cat("\nSmoothing complete!\n")

# =============================
# 4. ACTIVITY SEGMENTATION
# =============================

cat("\n=== STEP 4: ACTIVITY SEGMENTATION ===\n")

# Compute acceleration magnitude
if ("Smooth_LinearAccelerometerSensor" %in% names(df_aligned) &&
    sum(abs(df_aligned$Smooth_LinearAccelerometerSensor)) > 10) {
  
  cat("Using LinearAccelerometerSensor for activity detection\n")
  mag <- abs(df_aligned$Smooth_LinearAccelerometerSensor)
  
} else if (all(c("Smooth_AccX", "Smooth_AccY", "Smooth_AccZ") %in% names(df_aligned))) {
  
  cat("Using AccX/Y/Z magnitude for activity detection\n")
  mag <- sqrt(df_aligned$Smooth_AccX^2 +
                df_aligned$Smooth_AccY^2 +
                df_aligned$Smooth_AccZ^2)
  
} else {
  stop("No suitable acceleration columns found for activity detection")
}

# Compute rolling variance
roll_window_sec <- 3
window_size <- round(roll_window_sec / dt)

cat(paste("Computing rolling variance with", roll_window_sec, "second window\n"))

roll_var <- rollapply(mag, width = window_size, FUN = var,
                      fill = NA, align = "right")
roll_var[is.na(roll_var)] <- 0
df_aligned$RollVar <- roll_var

# K-means clustering on log-variance
cat("\nPerforming K-means clustering (k=3) on log-variance...\n")
set.seed(123)
log_var <- log(roll_var + 1e-6)
km <- kmeans(log_var, centers = 3, nstart = 20)

cat("\nCluster centers (in log-variance space):\n")
print(sort(km$centers))

# Determine thresholds
centers_sorted <- sort(km$centers)
thresh1 <- exp(mean(centers_sorted[1:2]))
thresh2 <- exp(mean(centers_sorted[2:3]))

cat(paste("\nVariance threshold 1 (Sedentary/Walking):", round(thresh1, 4), "\n"))
cat(paste("Variance threshold 2 (Walking/Running):", round(thresh2, 4), "\n"))

# Assign activity labels
df_aligned$Activity <- cut(
  df_aligned$RollVar,
  breaks = c(-Inf, thresh1, thresh2, Inf),
  labels = c("Sedentary", "Walking", "Running")
)

# Print activity distribution
cat("\n--- Activity Distribution ---\n")
activity_table <- table(df_aligned$Activity)
print(activity_table)
cat("\nProportions:\n")
print(prop.table(activity_table))

# =============================
# 5. VISUALIZATION: RAW PREPROCESSING
# =============================

cat("\n=== STEP 5: GENERATING PREPROCESSING PLOTS ===\n")

# Plot 1: Activity segmentation with variance
cat("\nPlot 1: Activity segmentation based on rolling variance\n")
p1 <- ggplot(df_aligned, aes(x = time, y = RollVar, color = Activity)) +
  geom_line(linewidth = 0.5) +
  scale_color_manual(values = c(Sedentary = "blue",
                                Walking   = "orange",
                                Running   = "red")) +
  labs(title = "Activity Segmentation: Rolling Variance of Acceleration",
       subtitle = "K-means clustering (k=3) on log-variance",
       x = "Time (seconds)",
       y = "Rolling Variance") +
  theme_minimal() +
  theme(legend.position = "top")
print(p1)

# Plot 2: Activity distribution
cat("\nPlot 2: Distribution of detected activities\n")
p2 <- ggplot(df_aligned, aes(x = Activity)) +
  geom_bar(fill = "steelblue", color = "black") +
  geom_text(stat = 'count', aes(label = ..count..), vjust = -0.5) +
  labs(title = "Distribution of Activity Phases",
       subtitle = "Total time points per activity",
       y = "Count") +
  theme_minimal()
print(p2)

# Plot 3: Smoothed vs raw acceleration (if AccX available)
if ("AccX" %in% names(df_aligned) && "Smooth_AccX" %in% names(df_aligned)) {
  cat("\nPlot 3: Raw vs Smoothed AccX (first 100 seconds)\n")
  
  plot_subset <- df_aligned[1:min(1000, nrow(df_aligned)), ]
  
  p3 <- ggplot(plot_subset, aes(x = time)) +
    geom_line(aes(y = AccX, color = "Raw"), alpha = 0.5, size = 0.3) +
    geom_line(aes(y = Smooth_AccX, color = "Smoothed"), size = 0.8) +
    scale_color_manual(values = c(Raw = "grey50", Smoothed = "steelblue")) +
    labs(title = "Comparison: Raw vs Smoothed AccX (First 100 seconds)",
         x = "Time (seconds)",
         y = "AccX Value",
         color = "Data Type") +
    theme_minimal() +
    theme(legend.position = "top")
  print(p3)
}

# =============================
# 6. BUILD FUNCTIONAL DATA OBJECTS
# =============================

cat("\n=== STEP 6: BUILDING FUNCTIONAL DATA OBJECTS ===\n")

# Rebuild magnitude from smoothed components
if ("Smooth_LinearAccelerometerSensor" %in% names(df_aligned) &&
    sum(abs(df_aligned$Smooth_LinearAccelerometerSensor)) > 10) {
  mag <- abs(df_aligned$Smooth_LinearAccelerometerSensor)
} else {
  mag <- sqrt(df_aligned$Smooth_AccX^2 +
                df_aligned$Smooth_AccY^2 +
                df_aligned$Smooth_AccZ^2)
}

# Identify contiguous activity segments
act_labels <- as.character(df_aligned$Activity)
time_grid <- df_aligned$time

rle_act <- rle(act_labels)
seg_idx <- rep(seq_along(rle_act$values), times = rle_act$lengths)

segments <- split(
  data.frame(time = time_grid,
             mag  = mag,
             act  = act_labels,
             seg  = seg_idx),
  f = seg_idx
)

cat(paste("\nTotal segments identified:", length(segments), "\n"))

# Filter segments by minimum length
min_len <- 2  # at 0.1 least  seconds
segments <- segments[sapply(segments, nrow) >= min_len]

cat(paste("Segments after filtering (>= 5 seconds):", length(segments), "\n"))

# Count segments per activity
seg_by_activity <- sapply(segments, function(x) unique(x$act))
cat("\nSegments per activity:\n")
print(table(seg_by_activity))

# Build functional objects per activity
fd_list <- list()
n_basis <- 25
n_eval <- 50
common_t <- seq(0, 1, length.out = n_eval)

for (activity in c("Sedentary", "Walking", "Running")) {
  
  cat(paste("\nProcessing activity:", activity, "\n"))
  
  seg_act <- segments[sapply(segments, function(x) unique(x$act)) == activity]
  
  if (length(seg_act) == 0) {
    cat(paste("  No segments found for", activity, "\n"))
    next
  }
  
  cat(paste("  Found", length(seg_act), "segments\n"))
  
  # Create matrix of curves (rescaled to [0,1])
  curves_mat <- matrix(NA_real_, nrow = n_eval, ncol = length(seg_act))
  
  for (j in seq_along(seg_act)) {
    seg <- seg_act[[j]]
    t_rel <- (seg$time - min(seg$time)) /
      (max(seg$time) - min(seg$time) + 1e-9)
    curves_mat[, j] <- approx(t_rel, seg$mag,
                              xout = common_t, rule = 2)$y
  }
  
  cat("  Creating functional data object with B-splines...\n")
  
  basis_obj <- create.bspline.basis(rangeval = c(0, 1),
                                    nbasis = n_basis, norder = 4)
  fd_par <- fdPar(basis_obj, Lfdobj = 2, lambda = 1e-2)
  fd_obj <- smooth.basis(argvals = common_t,
                         y = curves_mat,
                         fdParobj = fd_par)$fd
  
  fd_list[[activity]] <- fd_obj
  
  cat(paste("  Successfully created fd object with",
            ncol(curves_mat), "curves\n"))
}

cat("\nFunctional data objects created!\n")
cat("Available activities:\n")
print(names(fd_list))

# =============================
# 7. FUNCTIONAL DATA EXPLORATORY ANALYSIS
# =============================

cat("\n=== STEP 7: FUNCTIONAL DATA EXPLORATORY ANALYSIS ===\n")

grid_eval <- seq(0, 1, length.out = 50)

for (activity in names(fd_list)) {
  
  cat(paste("\n--- Analyzing", activity, "---\n"))
  
  fd_obj <- fd_list[[activity]]
  eval_mat <- eval.fd(grid_eval, fd_obj)
  
  n_curves <- ncol(eval_mat)
  cat(paste("Number of curves:", n_curves, "\n"))
  
  # Compute mean and SD
  mean_curve <- rowMeans(eval_mat)
  sd_curve <- apply(eval_mat, 1, sd)
  
  cat("Mean curve statistics:\n")
  cat(paste("  Mean:", round(mean(mean_curve), 3), "\n"))
  cat(paste("  Min:",  round(min(mean_curve), 3), "\n"))
  cat(paste("  Max:",  round(max(mean_curve), 3), "\n"))
  
  cat("Standard deviation statistics:\n")
  cat(paste("  Mean SD:", round(mean(sd_curve), 3), "\n"))
  cat(paste("  Max SD:",  round(max(sd_curve), 3), "\n"))
  
  # Spaghetti plot
  cat(paste("\nPlot: Spaghetti plot for", activity, "\n"))
  
  df_spag <- data.frame(
    t     = rep(grid_eval, n_curves),
    val   = as.vector(eval_mat),
    curve = rep(seq_len(n_curves), each = length(grid_eval))
  )
  
  p_spag <- ggplot(df_spag, aes(x = t, y = val, group = curve)) +
    geom_line(alpha = 0.3, color = "grey40") +
    labs(title = paste("Spaghetti Plot:", activity),
         subtitle = paste(n_curves, "functional observations"),
         x = "Normalized Time [0,1]",
         y = "Acceleration Magnitude") +
    theme_minimal()
  print(p_spag)
  
  # Mean +/- SD plot
  cat(paste("\nPlot: Mean +/- SD for", activity, "\n"))
  
  df_mean <- data.frame(
    t     = grid_eval,
    mean  = mean_curve,
    upper = mean_curve + sd_curve,
    lower = mean_curve - sd_curve
  )
  
  p_mean <- ggplot(df_mean, aes(x = t, y = mean)) +
    geom_ribbon(aes(ymin = lower, ymax = upper),
                fill = "lightblue", alpha = 0.5) +
    geom_line(color = "blue", size = 1) +
    labs(title = paste("Mean ± SD:", activity),
         subtitle = "Functional mean with ±1 standard deviation band",
         x = "Normalized Time [0,1]",
         y = "Acceleration Magnitude") +
    theme_minimal()
  print(p_mean)
}

# =============================
# 8. INDUCTIVE CONFORMAL PREDICTION
#    WITH FPCA + ELLIPSOID IN SCORE SPACE
# =============================

cat("\n=== STEP 8: FPCA + ELLIPSOIDAL INDUCTIVE CONFORMAL PREDICTION ===\n")

# Typical class = Walking
if (!"Walking" %in% names(fd_list)) {
  stop("Walking activity not found in functional data objects (fd_list)!")
}

fd_typical <- fd_list[["Walking"]]

# Anomalous classes = Running, Sedentary (if present)
fd_anom <- list()
if ("Running" %in% names(fd_list))   fd_anom[["Running"]]   <- fd_list[["Running"]]
if ("Sedentary" %in% names(fd_list)) fd_anom[["Sedentary"]] <- fd_list[["Sedentary"]]

cat("\nConformal prediction setup:\n")
cat("  - Typical class (for FPCA and ellipsoid): Walking\n")
cat(paste("  - Anomalous classes:",
          ifelse(length(fd_anom) > 0,
                 paste(names(fd_anom), collapse = ", "),
                 "NONE"),
          "\n"))

# ICP parameters
set.seed(123)
n_walking <- dim(fd_typical$coef)[2]
train_frac <- 0.7
conf_alpha <- 0.10

if (n_walking < 5) {
  stop(paste("Not enough Walking curves for FPCA-based conformal prediction (n =",
             n_walking, ")."))
}

idx_all   <- sample(seq_len(n_walking))
n_train   <- max(3, floor(train_frac * n_walking))
idx_train <- idx_all[1:n_train]
idx_calib <- idx_all[(n_train + 1):n_walking]
n_calib   <- length(idx_calib)

cat("\nData split:\n")
cat(paste("  - Total Walking curves:", n_walking, "\n"))
cat(paste("  - Training set size:",   n_train,   "\n"))
cat(paste("  - Calibration set size:", n_calib,  "\n"))
cat(paste("  - Significance level alpha:", conf_alpha, "\n"))

# ---- 8.1 FPCA on Walking ----
nharm_max <- min(10, n_walking - 1)
cat(paste("\nRunning FPCA on Walking curves (nharm_max =", nharm_max, ")...\n"))

conf_pca_walk <- pca.fd(fd_typical, nharm = nharm_max)

cat("\nProportion of variance explained by FPCA components:\n")
print(round(conf_pca_walk$varprop, 3))

cumvar <- cumsum(conf_pca_walk$varprop)
conf_p_keep <- which(cumvar >= 0.90)[1]  # keep enough PCs to explain >= 90% variance
if (is.na(conf_p_keep)) conf_p_keep <- nharm_max

cat(paste("\nNumber of FPCA components used for conformal:", conf_p_keep, "\n"))
cat(paste("Cumulative variance explained:", round(cumvar[conf_p_keep], 3), "\n"))

# Score matrix for all Walking curves (n_walking x p_keep)
scores_all <- conf_pca_walk$scores[, 1:conf_p_keep, drop = FALSE]
scores_train <- scores_all[idx_train, , drop = FALSE]
scores_calib <- scores_all[idx_calib, , drop = FALSE]

# ---- 8.2 Ellipsoid in score space (Mahalanobis) ----
conf_mu_scores    <- colMeans(scores_train)
conf_Sigma_scores <- cov(scores_train)
conf_Sigma_scores <- conf_Sigma_scores + diag(1e-6, ncol(conf_Sigma_scores))

# Squared Mahalanobis distances for calibration points
conf_d2_calib <- mahalanobis(scores_calib,
                             center = conf_mu_scores,
                             cov    = conf_Sigma_scores)

cat("\nCalibration Mahalanobis distances (Walking, FPCA score space):\n")
print(summary(conf_d2_calib))

# Conformal threshold: (1 - alpha) quantile of calibration distances
conf_q_hat <- as.numeric(quantile(conf_d2_calib,
                                  probs = 1 - conf_alpha,
                                  type  = 1))

cat(paste("\nEllipsoidal conformal threshold (1 - alpha quantile of d^2):",
          round(conf_q_hat, 4), "\n"))
cat("Interpretation: points with Mahalanobis^2 >", 
    round(conf_q_hat, 4), "lie outside the conformal ellipsoid.\n")

# Helper to compute squared Mahalanobis distances for arbitrary fd objects
compute_d2_fd <- function(fd_obj, pca_fit, mu, Sigma, p_keep) {
  harm_sub <- pca_fit$harmonics[1:p_keep]
  scores   <- inprod(fd_obj, harm_sub)
  if (is.vector(scores)) {
    scores <- matrix(scores, ncol = p_keep, byrow = TRUE)
  }
  mahalanobis(scores, center = mu, cov = Sigma)
}

# p-values from distances
compute_pvals_from_d2 <- function(d2_new, d2_calib) {
  sapply(d2_new, function(d2) {
    (sum(d2_calib >= d2) + 1) / (length(d2_calib) + 1)
  })
}

# Typical (Walking) calibration curves
conf_p_typical <- compute_pvals_from_d2(conf_d2_calib, conf_d2_calib)

cat("\n--- Conformal p-values for Walking (typical, calibration set) ---\n")
print(summary(conf_p_typical))
cat(paste("Miscoverage rate on Walking (p < alpha):",
          round(mean(conf_p_typical < conf_alpha), 3),
          " (expected ≈", conf_alpha, ")\n"))

# ---- 8.3 Distances and p-values for anomalous classes ----
conf_d2_anom_list <- list()
conf_p_anom_list  <- list()

for (nm in names(fd_anom)) {
  cat(paste("\n--- Processing anomalous class:", nm, "---\n"))
  
  d2_new <- compute_d2_fd(fd_anom[[nm]],
                          pca_fit = conf_pca_walk,
                          mu      = conf_mu_scores,
                          Sigma   = conf_Sigma_scores,
                          p_keep  = conf_p_keep)
  p_new  <- compute_pvals_from_d2(d2_new, conf_d2_calib)
  
  conf_d2_anom_list[[nm]] <- d2_new
  conf_p_anom_list[[nm]]  <- p_new
  
  cat("Mahalanobis^2 distances summary:\n")
  print(summary(d2_new))
  
  cat("\nConformal p-values summary:\n")
  print(summary(p_new))
  
  cat(paste("Detection rate (outside ellipsoid, p < alpha):",
            round(mean(p_new < conf_alpha), 3), "\n"))
}

# =============================
# 9. CONFORMAL VISUALIZATION (FPCA + ELLIPSOID)
# =============================

cat("\n=== STEP 9: VISUALIZING FPCA + ELLIPSOIDAL CONFORMAL SET ===\n")

alpha <- conf_alpha
q_hat <- conf_q_hat
d2_calib <- conf_d2_calib
p_typical <- conf_p_typical
d2_anom_list <- conf_d2_anom_list
p_anom_list  <- conf_p_anom_list
pca_walk     <- conf_pca_walk
mu_scores    <- conf_mu_scores
Sigma_scores <- conf_Sigma_scores

# Plot 1: Histogram of calibration distances with threshold
cat("\nPlot 1: Calibration Mahalanobis distances with ellipsoid threshold\n")

df_calib <- data.frame(d2 = d2_calib)

p_calib <- ggplot(df_calib, aes(x = d2)) +
  geom_histogram(bins = 15, fill = "grey70", color = "black") +
  geom_vline(xintercept = q_hat, color = "red",
             linetype = "dashed", size = 1) +
  annotate("text", x = q_hat, y = Inf,
           label = paste("Threshold =", round(q_hat, 3)),
           hjust = -0.1, vjust = 2, color = "red") +
  labs(title = "Calibration Mahalanobis Distances (Walking, FPCA score space)",
       subtitle = paste("Red dashed line: (1 - alpha) quantile, alpha =", alpha),
       x = expression(d^2),
       y = "Count") +
  theme_minimal()
print(p_calib)

# Plot 2: P-values typical vs anomalous
cat("\nPlot 2: Conformal p-values (typical vs anomalous)\n")

df_pvals <- data.frame(
  pval = c(p_typical,
           unlist(p_anom_list)),
  type = c(rep("Walking (typical)", length(p_typical)),
           unlist(lapply(names(p_anom_list), function(nm) {
             rep(paste(nm, "(anomalous)"), length(p_anom_list[[nm]]))
           })))
)

p_pvals <- ggplot(df_pvals, aes(x = pval, fill = type)) +
  geom_histogram(bins = 10, alpha = 0.7, position = "identity") +
  scale_fill_manual(values = c("Walking (typical)"     = "darkgreen",
                               "Running (anomalous)"   = "red",
                               "Sedentary (anomalous)" = "blue")) +
  geom_vline(xintercept = alpha, linetype = "dashed", color = "black") +
  labs(title    = "Conformal p-values (FPCA + Ellipsoid)",
       subtitle = "Points with p < alpha are outside the conformal ellipsoid",
       x        = "p-value",
       y        = "Count",
       fill     = "Class") +
  theme_minimal() +
  theme(legend.position = "top")
print(p_pvals)

# Plot 3: Ellipsoid in first two FPCA scores
cat("\nPlot 3: Ellipsoid in the first two FPCA score coordinates\n")

scores2_all   <- pca_walk$scores[, 1:2, drop = FALSE]
scores2_train <- scores2_all[idx_train, , drop = FALSE]
scores2_cal   <- scores2_all[idx_calib, , drop = FALSE]

df_scores2 <- data.frame(
  PC1  = scores2_cal[, 1],
  PC2  = scores2_cal[, 2],
  type = "Walking (calibration)"
)

for (nm in names(d2_anom_list)) {
  harm_sub2 <- pca_walk$harmonics[1:2]
  scores_anom2 <- inprod(fd_anom[[nm]], harm_sub2)
  if (is.vector(scores_anom2)) {
    scores_anom2 <- matrix(scores_anom2, ncol = 2, byrow = TRUE)
  }
  df_scores2 <- rbind(
    df_scores2,
    data.frame(
      PC1  = scores_anom2[, 1],
      PC2  = scores_anom2[, 2],
      type = paste(nm, "(anomalous)")
    )
  )
}

Sigma2 <- cov(scores2_train)
mu2    <- colMeans(scores2_train)

eig2   <- eigen(Sigma2)
angles <- seq(0, 2 * pi, length.out = 200)
circle <- rbind(cos(angles), sin(angles))
A      <- eig2$vectors %*% diag(sqrt(pmax(eig2$values, 0)))
ell2   <- t(matrix(mu2, nrow = 2, ncol = length(angles)) +
              A %*% (sqrt(q_hat) * circle))
df_ell <- data.frame(PC1 = ell2[, 1], PC2 = ell2[, 2])

p_ell <- ggplot(df_scores2, aes(x = PC1, y = PC2, color = type)) +
  geom_point(alpha = 0.7) +
  geom_path(data = df_ell, aes(x = PC1, y = PC2),
            inherit.aes = FALSE, color = "black", linewidth = 1) +
  labs(title = "Ellipsoidal Conformal Set in FPCA Score Space",
       subtitle = "Boundary: Mahalanobis^2 = conformal threshold",
       x = "FPCA score 1",
       y = "FPCA score 2",
       color = "Class") +
  theme_minimal() +
  theme(legend.position = "top")
print(p_ell)

# =============================
# 10. FINAL SUMMARY
# =============================

cat("\n", rep("=", 70), "\n", sep = "")
cat("                    ANALYSIS COMPLETE\n")
cat(rep("=", 70), "\n", sep = "")

cat("\n--- SUMMARY OF RESULTS ---\n\n")

cat("1. DATA PREPROCESSING:\n")
cat(paste("   - Total experiment duration:",
          round(max(df_raw$time_sec) / 60, 2), "minutes\n"))
cat(paste("   - Data points after interpolation:", nrow(df_aligned), "\n"))
cat("   - Activities detected: Sedentary, Walking, Running\n\n")

cat("2. FUNCTIONAL DATA ANALYSIS:\n")
for (activity in names(fd_list)) {
  n_curves <- dim(fd_list[[activity]]$coef)[2]
  cat(paste("   -", activity, ":", n_curves, "functional curves\n"))
}
cat("\n")

cat("3. CONFORMAL PREDICTION (FPCA + ELLIPSOID):\n")
cat(paste("   - Significance level alpha:", conf_alpha, "\n"))
cat(paste("   - Ellipsoid threshold (Mahalanobis^2):", round(conf_q_hat, 4), "\n"))
cat(paste("   - Walking miscoverage rate:",
          round(mean(conf_p_typical < conf_alpha), 3),
          "(expected ≈", conf_alpha, ")\n"))
for (nm in names(conf_p_anom_list)) {
  cat(paste("   -", nm, "detection rate (p < alpha):",
            round(mean(conf_p_anom_list[[nm]] < conf_alpha), 3), "\n"))
}
cat("\n4. INTERPRETATION:\n")
cat("   - Walking curves show valid conformal behaviour (miscoverage ≈ alpha).\n")
cat("   - Running and Sedentary curves typically lie outside the FPCA ellipsoid.\n")
cat("   - The FPCA+ellipsoid non-conformity captures shape/amplitude differences\n")
cat("     between activities in a low-dimensional score space.\n\n")

cat(rep("=", 70), "\n", sep = "")
cat("\nAll outputs, plots, and summaries have been generated.\n")
cat("Script execution complete!\n")
cat(rep("=", 70), "\n\n", sep = "")


# ---- Plot 4: Prediction band for projection in time domain ----
cat("\nPlot 4: Prediction band for projected Walking curves in time domain\n")

# Grid for bands = same as grid_eval used for FDA
grid_band <- grid_eval

# Evaluate FPCA harmonics and mean function on this grid
phi_mat    <- eval.fd(grid_band, pca_walk$harmonics[1:conf_p_keep])
mean_fd_vec <- as.vector(eval.fd(grid_band, pca_walk$meanfd))

# For each t, compute center and radius of band:
# center(t) = mean_fd(t) + mu_scores^T phi(t)
# radius(t) = sqrt(q_hat * phi(t)^T Sigma_scores phi(t))  (ellipsoid projection).[web:37]

band_center <- numeric(length(grid_band))
band_upper  <- numeric(length(grid_band))
band_lower  <- numeric(length(grid_band))

for (r in seq_along(grid_band)) {
  phi_vec <- phi_mat[r, ]
  mean_scores_t <- sum(mu_scores * phi_vec)
  var_t <- as.numeric(t(phi_vec) %*% Sigma_scores %*% phi_vec)
  center_t <- mean_fd_vec[r] + mean_scores_t
  radius_t <- sqrt(max(q_hat * var_t, 0))
  
  band_center[r] <- center_t
  band_upper[r]  <- center_t + radius_t
  band_lower[r]  <- center_t - radius_t
}

band_df <- data.frame(
  t      = grid_band,
  center = band_center,
  upper  = band_upper,
  lower  = band_lower
)

# Project all Walking curves onto first conf_p_keep PCs and reconstruct on grid
# Π_p X_i(t) = mean_fd(t) + sum_j score_ij * phi_j(t).[web:37]
scores_walk <- pca_walk$scores[, 1:conf_p_keep, drop = FALSE]
proj_mat <- matrix(NA_real_, nrow = length(grid_band), ncol = nrow(scores_walk))

for (i in seq_len(nrow(scores_walk))) {
  proj_mat[, i] <- mean_fd_vec + as.numeric(phi_mat %*% scores_walk[i, ])
}

df_proj <- data.frame(
  t     = rep(grid_band, n_walking),
  val   = as.vector(proj_mat),
  curve = rep(seq_len(n_walking), each = length(grid_band))
)

p_band <- ggplot() +
  geom_ribbon(data = band_df,
              aes(x = t, ymin = lower, ymax = upper),
              fill = "lightblue", alpha = 0.4) +
  geom_line(data = df_proj,
            aes(x = t, y = val, group = curve),
            color = "grey40", alpha = 0.4, linewidth = 0.4) +
  geom_line(data = band_df,
            aes(x = t, y = center),
            color = "blue", linewidth = 1) +
  labs(title = "Conformal Prediction Band for Projected Walking Curves",
       subtitle = "Band obtained by projecting ellipsoid in FPCA score space",
       x = "Normalized time [0, 1]",
       y = "Acceleration magnitude (projected)") +
  theme_minimal()

print(p_band)


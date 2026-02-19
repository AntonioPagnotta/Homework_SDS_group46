################################################################################
# FUNCTIONAL DATA ANALYSIS + CONFORMAL PREDICTION (FINAL PIPELINE)
# Smartphone Sensor Data (Run / Sit / Walk)
#
# Homework Steps Addressed:
# 2. Describe your data (Spaghetti plots, Functional Means, Jerk Analysis)
# 3. Conformal step (Section 3: FPCA Projections & Ellipsoidal Prediction Bands)
################################################################################

# =============================
# 0. INSTALL AND LOAD PACKAGES
# =============================
cat("\n=== LOADING REQUIRED PACKAGES ===\n")
required_packages <- c("dplyr", "zoo", "fda", "ggplot2", "reshape2", "gridExtra")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  library(pkg, character.only = TRUE)
}

# =============================
# 1. DATA PREPROCESSING & ALIGNMENT
# =============================
cat("\n=== STEP 1: PREPROCESSING & INTERPOLATION ===\n")
df_raw <- read.csv("dati_sgravati (1).csv")

# Sort by timestamp and convert to relative seconds
df_raw <- df_raw[order(df_raw$timestamp), ]
t0 <- min(df_raw$timestamp)
df_raw$time_sec <- (df_raw$timestamp - t0) / 1000

sensor_cols <- setdiff(names(df_raw), c("timestamp", "time_sec"))

# Interpolate to common 10Hz grid
dt <- 0.1
t_grid <- seq(from = 0, to = max(df_raw$time_sec), by = dt)
df_aligned <- data.frame(time = t_grid)

interpolate_sensor <- function(time_col, val_col, new_grid) {
  valid <- !is.na(val_col)
  if (sum(valid) < 2) return(rep(0, length(new_grid)))
  z <- zoo(val_col[valid], time_col[valid])
  as.numeric(na.approx(z, xout = new_grid, rule = 2))
}

for (s in sensor_cols) {
  df_aligned[[s]] <- interpolate_sensor(df_raw$time_sec, df_raw[[s]], t_grid)
}

# =============================
# 2. B-SPLINE SMOOTHING & SEGMENTATION
# =============================
cat("\n=== STEP 2: CHUNKED SMOOTHING & ACTIVITY SEGMENTATION ===\n")

# Smooth Acceleration magnitude to remove high-frequency jitter
if("LinearAccelerometerSensor" %in% names(df_aligned)) {
  raw_mag <- abs(df_aligned$LinearAccelerometerSensor)
} else {
  raw_mag <- sqrt(df_aligned$AccX^2 + df_aligned$AccY^2 + df_aligned$AccZ^2)
}

# Smoothing in chunks to save memory
chunk_size <- 2000
num_chunks <- ceiling(length(t_grid) / chunk_size)
smooth_mag <- numeric(length(t_grid))

for(i in 1:num_chunks) {
  idx <- ((i-1)*chunk_size + 1):min(i*chunk_size, length(t_grid))
  nbasis <- max(4, round(length(idx) / 4))
  basis_obj <- create.bspline.basis(rangeval = range(t_grid[idx]), nbasis = nbasis, norder = 4)
  fd_par <- fdPar(basis_obj, Lfdobj = 2, lambda = 0.01)
  sm_res <- smooth.basis(t_grid[idx], raw_mag[idx], fd_par)
  smooth_mag[idx] <- as.numeric(eval.fd(t_grid[idx], sm_res$fd))
}
df_aligned$SmoothMag <- smooth_mag

# Segment via Rolling Variance + K-Means
window_size <- round(5 / dt) # 5 second rolling window
roll_var <- rollapply(df_aligned$SmoothMag, width = window_size, FUN = var, fill = 0, align = "right")

set.seed(123)
log_var <- log(roll_var + 1e-6)
km <- kmeans(log_var, centers = 3, nstart = 20)
centers_sorted <- sort(km$centers)

thresh1 <- exp(mean(centers_sorted[1:2]))
thresh2 <- exp(mean(centers_sorted[2:3]))

df_aligned$Activity <- cut(roll_var, 
                           breaks = c(-Inf, thresh1, thresh2, Inf),
                           labels = c("Sedentary", "Walking", "Running"))

# Clean labels with median filter to remove micro-glitches
act_smooth <- runmed(as.integer(df_aligned$Activity), k = round(1/dt)+1)
df_aligned$Activity <- factor(act_smooth, levels=1:3, labels=c("Sedentary", "Walking", "Running"))


# =============================
# 3. FDA EXPLORATION (HOMEWORK STEP 2)
# =============================
cat("\n=== STEP 3: FUNCTIONAL DATA EXPLORATION ===\n")

# Slice into 5-second fixed functional observations
pts_per_window <- round(5 / dt) 
window_duration <- 5
n_windows <- floor(nrow(df_aligned) / pts_per_window)
trunc_len <- n_windows * pts_per_window

acc_matrix <- matrix(df_aligned$SmoothMag[1:trunc_len], nrow = pts_per_window, ncol = n_windows)
activity_matrix <- matrix(as.character(df_aligned$Activity[1:trunc_len]), nrow=pts_per_window)

get_majority_label <- function(v) {
  tab <- table(v)
  lab <- names(tab)[which.max(tab)]
  prop <- max(tab) / sum(tab)
  c(label = lab, prop = prop)
}

# Filter for pure activity windows
maj_stats <- apply(activity_matrix, 2, get_majority_label)
maj_labels <- maj_stats["label", ]
maj_props  <- as.numeric(maj_stats["prop", ])

keep <- maj_props >= 0.8  # Keep windows that are at least 80% one activity
acc_matrix <- acc_matrix[, keep, drop = FALSE]
window_labels <- factor(maj_labels[keep], levels = c("Sedentary","Walking","Running"))

# Create FDA Object
window_time <- seq(0, window_duration, length.out = pts_per_window)
fda_basis <- create.bspline.basis(rangeval = c(0, window_duration), nbasis = 15, norder = 4)
acc_fd <- Data2fd(window_time, acc_matrix, fda_basis)

# =========================================================
# Plot 3.1: Spaghetti Plot (Separated 1x3 Grid)
# =========================================================
# Find global y-axis limits so all 3 plots share the same scale
plot_grid_eval <- seq(0, window_duration, length.out = 100)
ylim_acc <- range(eval.fd(plot_grid_eval, acc_fd))

par(mfrow = c(1, 3)) # Set up 1x3 grid
for(act in c("Sedentary", "Walking", "Running")) {
  idx <- which(window_labels == act)
  if(length(idx) > 0) {
    plot(acc_fd[idx], col = col_map[act], lty = 1, lwd = 0.5,
         main = paste("Acceleration:", act), 
         xlab = "Time (s)", ylab = "Acc (m/s^2)",
         ylim = ylim_acc) # Force shared y-axis
  } else {
    plot.new()
    title(main = paste(act, "- No Data"))
  }
}
mtext("FDA: 5-Second Acceleration Windows", side = 3, line = -2, outer = TRUE, font = 2)
par(mfrow = c(1, 1)) # Reset layout


# =========================================================
# Plot 3.2: 1st Derivative / Jerk Analysis (Separated 1x3 Grid)
# =========================================================
acc_deriv_fd <- deriv.fd(acc_fd, 1)

# Find global y-axis limits for the derivative
ylim_jerk <- range(eval.fd(plot_grid_eval, acc_deriv_fd))

par(mfrow = c(1, 3)) # Set up 1x3 grid
for(act in c("Sedentary", "Walking", "Running")) {
  idx <- which(window_labels == act)
  if(length(idx) > 0) {
    plot(acc_deriv_fd[idx], col = col_map[act], lwd = 0.5,
         main = paste("Jerk:", act), 
         xlab = "Time (s)", ylab = "Jerk (m/s^3)",
         ylim = ylim_jerk) # Force shared y-axis
  } else {
    plot.new()
    title(main = paste(act, "- No Data"))
  }
}
mtext("FDA Dedicated Plot: 1st Derivative (Jerk)", side = 3, line = -2, outer = TRUE, font = 2)
par(mfrow = c(1, 1)) # Reset layout


# Plot 3.3: Unified Functional Means (NEW) - CORRECTED
plot_grid <- seq(0, window_duration, length.out = 100)
eval_acc <- eval.fd(plot_grid, acc_fd)

df_eval <- as.data.frame(eval_acc)
# Force consistent column names just to be safe
colnames(df_eval) <- paste0("Curve_", 1:ncol(df_eval))
df_eval$Time <- plot_grid

df_melt <- melt(df_eval, id.vars = "Time", variable.name = "Window", value.name = "Acc")

# Because melt stacks curve 1, then curve 2, etc., we can assign labels directly
# by repeating each label exactly 'length(plot_grid)' times (100 times)
df_melt$Activity <- rep(window_labels, each = length(plot_grid))

df_summary <- df_melt %>%
  group_by(Time, Activity) %>%
  summarise(
    Mean_Acc = mean(Acc, na.rm = TRUE),
    SD_Acc = sd(Acc, na.rm = TRUE),
    .groups = "drop"
  )

p_means <- ggplot(df_summary, aes(x = Time, y = Mean_Acc, color = Activity, fill = Activity)) +
  geom_line(linewidth = 1.2) +
  geom_ribbon(aes(ymin = Mean_Acc - SD_Acc, ymax = Mean_Acc + SD_Acc), alpha = 0.2, color = NA) +
  scale_color_manual(values = col_map) +
  scale_fill_manual(values = col_map) +
  labs(title = "Functional Means & Variance by Activity",
       subtitle = "Solid line: Mean | Shaded area: ±1 Standard Deviation",
       x = "Time (s)", y = expression("Acceleration Magnitude " (m/s^2))) +
  theme_minimal()

print(p_means)

# Plot 3.4: Functional Boxplots
par(mfrow = c(1, 3)) # Set up 3-panel plot
for(act in c("Sedentary", "Walking", "Running")) {
  idx <- which(window_labels == act)
  if(length(idx) > 5) {
    # FIX: Added 'xlim = range(plot_grid)' to force the correct x-axis scale (0-5s)
    fbplot(eval_acc[, idx], plot_grid, main=paste("Functional Boxplot:", act),
           xlab="Time (s)", ylab="Acc", color=col_map[act], barcol="black",
           xlim = range(plot_grid)) 
  }
}
par(mfrow = c(1, 1)) # Reset plotting layout

# =============================
# 4. MULTI-CLASS CONFORMAL PREDICTION
#    Following Section 3.1: Gaussian Mixture Approximation Components
# =============================
cat("\n=== STEP 4: MULTI-CLASS CONFORMAL PREDICTION (FPCA + ELLIPSOID) ===\n")

activities <- c("Sedentary", "Walking", "Running")
conf_alpha <- 0.10

# Initialize data frames to collect plotting data
df_bands_all <- data.frame()
df_proj_all <- data.frame()
grid_eval <- seq(0, window_duration, length.out = 100)

# Diagnostic print to check labels
cat("\nCurrent distribution of functional windows:\n")
print(table(window_labels, useNA = "always"))

coverage_results <- data.frame(
  Activity = character(),
  Coverage = numeric(),
  Detection_Other = numeric(),
  stringsAsFactors = FALSE
)

for(act in activities) {
  cat(sprintf("\n--- Processing Class: %s ---\n", act))
  
  # Force character matching to avoid factor level issues
  idx_act <- which(as.character(window_labels) == act)
  
  if(length(idx_act) < 10) {
    cat(sprintf("Not enough data for %s (Found %d windows). Skipping...\n", act, length(idx_act)))
    next
  }
  
  fd_act <- acc_fd[idx_act]
  
  # 4.1 Split into Train and Calibration
  set.seed(42)
  n_train <- max(5, floor(0.6 * length(idx_act)))
  idx_train <- sample(1:length(idx_act), n_train)
  idx_calib <- setdiff(1:length(idx_act), idx_train)
  
  fd_train <- fd_act[idx_train]
  fd_calib <- fd_act[idx_calib]
  
  # 4.2 Perform FPCA on Training Data
  nharm_max <- min(10, n_train - 1)
  pca_act <- pca.fd(fd_train, nharm = nharm_max)
  
  # Keep PCs explaining at least 90% variance
  cumvar <- cumsum(pca_act$varprop)
  p_keep <- which(cumvar >= 0.90)[1]
  if (is.na(p_keep)) p_keep <- nharm_max
  cat(sprintf("  Using %d PCs (explains %.1f%% var)\n", p_keep, cumvar[p_keep]*100))
  
  # Plot 4.2.1: FPCA Harmonics Plot (NEW)
  par(mfrow = c(1, min(2, p_keep)))
  plot.pca.fd(pca_act, nx = 100, pointplot = TRUE, harm = 1:min(2, p_keep),
              expand = 0, cycle = FALSE)
  mtext(paste("FPCA Perturbations:", act), side = 3, line = -2, outer = TRUE, font = 2)
  par(mfrow = c(1, 1))
  
  # 4.3 Define conformity scores (Mahalanobis distance)
  scores_train <- pca_act$scores[, 1:p_keep, drop = FALSE]
  mu_scores    <- colMeans(scores_train)
  Sigma_scores <- cov(scores_train) + diag(1e-6, p_keep)
  
  # Project Calibration data
  scores_calib <- inprod(fd_calib, pca_act$harmonics[1:p_keep])
  if(is.vector(scores_calib)) scores_calib <- matrix(scores_calib, ncol=p_keep, byrow=TRUE)
  
  d2_calib <- mahalanobis(scores_calib, center = mu_scores, cov = Sigma_scores)
  
  # 4.4 Calculate Threshold (alpha = 0.10)
  q_hat <- as.numeric(quantile(d2_calib, probs = 1 - conf_alpha, type = 1))
  cat(sprintf("  Ellipsoidal Threshold (q_hat): %.3f\n", q_hat))
  
  # 1) Empirical coverage for this activity (calibration windows)
  coverage_act <- mean(d2_calib <= q_hat)
  cat(sprintf("  Empirical coverage for %s (should be ≈ %.2f): %.3f\n",
              act, 1 - conf_alpha, coverage_act))
  
  # 2) Detection: how often do other-activity windows fall OUTSIDE this band's ellipsoid?
  
  # All other windows from acc_fd with labels != act
  idx_other_global <- which(window_labels != act)
  if (length(idx_other_global) > 0) {
    fd_other <- acc_fd[idx_other_global]
    
    # Project other windows into this class's PC space
    scores_other <- inprod(fd_other, pca_act$harmonics[1:p_keep])
    if (is.vector(scores_other)) {
      scores_other <- matrix(scores_other, ncol = p_keep, byrow = TRUE)
    }
    
    d2_other <- mahalanobis(scores_other,
                            center = mu_scores,
                            cov    = Sigma_scores)
    
    detection_other <- mean(d2_other > q_hat)
    cat(sprintf("  Detection of non-%s windows (d2 > q_hat): %.3f\n",
                act, detection_other))
  } else {
    detection_other <- NA
  }
  
  # Store results
  coverage_results <- rbind(coverage_results,
                            data.frame(Activity = act,
                                       Coverage = coverage_act,
                                       Detection_Other = detection_other))
  # Identify outliers for this class in score space
  is_outlier <- d2_calib > q_hat  # or use a higher cutoff like quantile(d2_calib, 0.99)
  cat(sprintf("  Found %d outliers in %s calibration set.\n",
              sum(is_outlier), act))
  
  # Build a data frame with first two PCs for calibration windows (Safe for 1D)
  if (ncol(scores_calib) >= 2) {
    pc1_vals <- scores_calib[, 1]
    pc2_vals <- scores_calib[, 2]
  } else {
    pc1_vals <- scores_calib[, 1]
    pc2_vals <- rep(0, length(pc1_vals)) # Pad with 0s if only 1 PC is kept
  }
  
  df_scores_calib <- data.frame(
    PC1 = pc1_vals,
    PC2 = pc2_vals,
    Outlier = factor(ifelse(is_outlier, "Outlier", "Inlier"),
                     levels = c("Inlier", "Outlier"))
  )
  
  # Plot 4.4.1: Outlier Plot (Original)
  p_out <- ggplot(df_scores_calib, aes(x = PC1, y = PC2, color = Outlier)) +
    geom_point(size = 2, alpha = 0.8) +
    scale_color_manual(values = c(Inlier = "grey60", Outlier = "red")) +
    labs(title = sprintf("FPCA Scores for %s (Calibration)", act),
         subtitle = "Outliers defined by Mahalanobis distance > q_hat",
         x = "PC1 score", y = "PC2 score") +
    theme_minimal() +
    theme(legend.position = "top")
  
  print(p_out)
  
  # 4.5 Build Prediction Band for Time Domain Plotting
  phi_mat   <- eval.fd(grid_eval, pca_act$harmonics[1:p_keep])
  mean_vec  <- as.vector(eval.fd(grid_eval, pca_act$meanfd))
  
  band_center <- numeric(length(grid_eval))
  band_upper  <- numeric(length(grid_eval))
  band_lower  <- numeric(length(grid_eval))
  
  for (r in seq_along(grid_eval)) {
    phi_vec <- phi_mat[r, ]
    center_t <- mean_vec[r] + sum(mu_scores * phi_vec)
    var_t <- as.numeric(t(phi_vec) %*% Sigma_scores %*% phi_vec)
    radius_t <- sqrt(max(q_hat * var_t, 0))
    
    band_center[r] <- center_t
    band_upper[r]  <- center_t + radius_t
    band_lower[r]  <- center_t - radius_t
  }
  
  # Store band data
  df_bands_all <- rbind(df_bands_all, data.frame(
    Activity = act, t = grid_eval, 
    center = band_center, upper = band_upper, lower = band_lower
  ))
  
  # Reconstruct calibration curves for plotting within the band
  proj_calib <- matrix(NA, nrow=length(grid_eval), ncol=nrow(scores_calib))
  for (i in 1:nrow(scores_calib)) {
    proj_calib[, i] <- mean_vec + as.numeric(phi_mat %*% scores_calib[i, ])
  }
  
  df_proj_all <- rbind(df_proj_all, data.frame(
    Activity = act,
    t = rep(grid_eval, ncol(proj_calib)),
    val = as.vector(proj_calib),
    curve = rep(paste0(act, "_", 1:ncol(proj_calib)), each = length(grid_eval))
  ))
}

cat("\n=== SUMMARY: COVERAGE & DETECTION ===\n")
print(coverage_results)

# =============================
# 5. CONFORMAL VISUALIZATION (3-PANEL)
# =============================
cat("\n=== STEP 5: VISUALIZING PREDICTION BANDS ===\n")

if (nrow(df_bands_all) == 0) {
  cat("\n[ERROR] No data was generated for the bands. Please re-run Steps 1-3 to ensure your data is loaded and segmented correctly.\n")
} else {
  
  # Ensure factors for consistent plotting order
  df_bands_all$Activity <- factor(df_bands_all$Activity, levels = activities)
  df_proj_all$Activity <- factor(df_proj_all$Activity, levels = activities)
  
  # Plot 5.1: Conformal Bands (Original)
  p_bands_multi <- ggplot() +
    # Draw the Conformal Bands
    geom_ribbon(data = df_bands_all, aes(x = t, ymin = lower, ymax = upper, fill = Activity), alpha = 0.3) +
    # Draw the Projected Curves
    geom_line(data = df_proj_all, aes(x = t, y = val, group = curve, color = Activity), alpha = 0.4, linewidth = 0.4) +
    # Draw the Mean Center Line
    geom_line(data = df_bands_all, aes(x = t, y = center), color = "black", linewidth = 0.8, linetype = "dashed") +
    # Formatting
    facet_wrap(~Activity, scales = "free_y") + 
    scale_fill_manual(values = col_map) +
    scale_color_manual(values = col_map) +
    labs(title = "Conformal Prediction Bands for All Activities (90% Coverage)",
         subtitle = "Gaussian Mixture Components: Projections from FPCA Mahalanobis Ellipsoids",
         x = "Relative Time (seconds)", y = "Acceleration Magnitude") +
    theme_bw() +
    theme(legend.position = "none",
          strip.background = element_rect(fill="grey90"),
          strip.text = element_text(face="bold", size=12))
  
  print(p_bands_multi)
  
  # Plot 5.2: Global Latent Space Conformal Ellipsoids (NEW)
  pca_global <- pca.fd(acc_fd, nharm = 2)
  df_latent <- data.frame(
    PC1 = pca_global$scores[, 1],
    PC2 = pca_global$scores[, 2],
    Activity = window_labels
  )
  
  p_latent <- ggplot(df_latent, aes(x = PC1, y = PC2, color = Activity)) +
    geom_point(alpha = 0.6, size = 1.5) +
    stat_ellipse(type = "norm", level = 1 - conf_alpha, linewidth = 1.2, aes(fill = Activity), geom = "polygon", alpha = 0.1) +
    scale_color_manual(values = col_map) +
    scale_fill_manual(values = col_map) +
    labs(title = "Latent Space Conformal Regions (90% Coverage)",
         subtitle = "Global PC1 vs PC2 Scores with Gaussian Ellipsoids",
         x = "Principal Component 1", y = "Principal Component 2") +
    theme_bw() +
    theme(legend.position = "bottom")
  
  print(p_latent)
  
  cat("\nAnalysis Complete. Multi-class prediction bands generated successfully!\n")
}
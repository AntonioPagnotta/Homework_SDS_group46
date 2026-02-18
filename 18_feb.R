# ==========================================
# DATA PREPROCESSING & CONFORMAL PREDICTION (FULL PIPELINE)
# ==========================================

# Load necessary libraries
if(!require(dplyr)) install.packages("dplyr")
if(!require(zoo)) install.packages("zoo")
if(!require(fda)) install.packages("fda")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(reshape2)) install.packages("reshape2")
if(!require(ks)) install.packages("ks")

library(dplyr)
library(zoo)
library(fda)
library(ggplot2)
library(reshape2)
library(ks)

# ==========================================
# 1. Load and Inspect
# ==========================================
# In this foundational step, we first load the raw sensor data and immediately sort it by 
# timestamp. This is critical because time series analysis relies on the strict chronological 
# order of events to detect trends and patterns. We then convert the raw timestamps—which 
# are typically massive integers representing milliseconds since the Unix epoch—into a relative 
# time scale (seconds starting from zero). This normalization is essential to prevent numerical 
# instability and overflow errors during the complex matrix calculations required for Functional 
# Data Analysis later on. Finally, we employ a dynamic column detection strategy to identify 
# all available sensor streams automatically. This makes the script robust and reusable for 
# different datasets without needing to hardcode specific sensor names like "AccX" or "Compass".

print("Loading data...")
df <- read.csv("dati_sgravati (1).csv")

# Sort and convert time
df <- df[order(df$timestamp), ]
start_time <- min(df$timestamp)
df$time_sec <- (df$timestamp - start_time) / 1000

# Identify sensors
sensor_cols <- setdiff(names(df), c("timestamp", "time_sec"))
print(paste("Found sensors:", paste(sensor_cols, collapse=", ")))

# ==========================================
# 2. Handle Sparsity (Interpolation)
# ==========================================
# The raw data comes from asynchronous sensors that fire at irregular intervals, resulting in 
# a "sparse" dataset filled with NaNs (gaps) where rows align to one sensor but miss data for 
# others. Standard statistical algorithms cannot handle these gaps. To resolve this, we impose 
# a structured 10Hz time grid (0.1s intervals) upon the data. We chose 10Hz because human 
# voluntary movements (like walking) typically occur at frequencies below 5Hz; according to 
# the Nyquist-Shannon sampling theorem, sampling at double that frequency (10Hz) is sufficient 
# to capture the signal perfectly without aliasing. We then use linear interpolation to estimate 
# the sensor values at these exact grid points, transforming the messy, sparse input into a 
# dense, synchronized matrix ready for mathematical analysis.

print("Aligning and Interpolating...")
dt <- 0.1 
t_grid <- seq(from = 0, to = max(df$time_sec), by = dt)
df_aligned <- data.frame(time = t_grid)

interpolate_sensor <- function(time_col, val_col, new_grid) {
  valid_idx <- !is.na(val_col)
  if(sum(valid_idx) < 2) return(rep(0, length(new_grid)))
  z <- zoo(val_col[valid_idx], time_col[valid_idx])
  as.numeric(na.approx(z, xout = new_grid, rule = 2))
}

for(sensor in sensor_cols) {
  df_aligned[[sensor]] <- interpolate_sensor(df$time_sec, df[[sensor]], t_grid)
}

# ==========================================
# 3. Memory-Safe Smoothing (Chunked)
# ==========================================
# Attempting to smooth the entire 1.6-hour dataset in one pass would require constructing a 
# basis matrix of enormous size (approx. 60,000 x 60,000), which would exceed the RAM limits 
# of most computers (the "vector memory limit" error). To solve this, we process the data in 
# manageable "chunks" of 2000 points. Within each chunk, we apply B-spline smoothing. We select 
# B-splines because they provide local control and are ideal for non-periodic data like human 
# movement, unlike Fourier bases which assume periodicity. We also apply a roughness penalty 
# on the second derivative to filter out high-frequency sensor noise (jitter), leaving us with 
# a smooth curve that represents the true physical motion.

print("Smoothing in chunks to save memory...")
chunk_size <- 2000 
n_points <- length(t_grid)
num_chunks <- ceiling(n_points / chunk_size)

smooth_chunk <- function(time_vec, data_vec) {
  nbasis <- max(4, round(length(time_vec) / 4))
  basis_obj <- create.bspline.basis(rangeval = range(time_vec), nbasis = nbasis, norder = 4)
  fdPar_obj <- fdPar(basis_obj, Lfdobj = 2, lambda = 0.01)
  smooth_res <- smooth.basis(time_vec, data_vec, fdPar_obj)
  return(eval.fd(time_vec, smooth_res$fd))
}

for(sensor in sensor_cols) {
  # print(paste("Processing sensor:", sensor)) # Commented to reduce output spam
  smoothed_vals <- numeric(n_points)
  for(i in 1:num_chunks) {
    idx_start <- (i-1) * chunk_size + 1
    idx_end <- min(i * chunk_size, n_points)
    idx <- idx_start:idx_end
    smoothed_vals[idx] <- smooth_chunk(df_aligned$time[idx], df_aligned[[sensor]][idx])
  }
  df_aligned[[paste0("Smooth_", sensor)]] <- smoothed_vals
}

# ==========================================
# 4. Segmentation (Improved & Robust)
# ==========================================
# To enable supervised analysis without manual labels, we must automatically segment the time 
# series into distinct activities (Sedentary, Walking, Running). We rely on the rolling 
# variance of the acceleration magnitude because variance is a strong proxy for physical 
# intensity (gravity is constant, but movement creates variance). We use a 5-second window 
# instead of a shorter one to smooth out momentary glitches, ensuring that a single stumble 
# doesn't register as a "run". We apply a log-transformation to the variance to handle its 
# exponential scaling, then use K-Means clustering to find natural intensity thresholds. 
# Finally, we apply a median filter to the resulting labels to remove "flickering" (e.g., 
# a single millisecond of "Running" appearing in a "Walking" stream), ensuring contiguous 
# and clean activity segments.

print("Segmenting Activity Phases (Robust)...")

# A. Calculate Magnitude
if("Smooth_LinearAccelerometerSensor" %in% names(df_aligned) && 
   sum(abs(df_aligned$Smooth_LinearAccelerometerSensor)) > 10) {
  mag <- abs(df_aligned$Smooth_LinearAccelerometerSensor)
} else {
  mag <- sqrt(df_aligned$Smooth_AccX^2 + df_aligned$Smooth_AccY^2 + df_aligned$Smooth_AccZ^2)
}

# B. Rolling Variance (Increased to 5 seconds for stability)
window_size <- round(5 / dt) 
roll_var <- rollapply(mag, width = window_size, FUN = var, fill = NA, align = "right")
roll_var[is.na(roll_var)] <- 0
df_aligned$RollVar <- roll_var

# C. Clustering (K-Means)
set.seed(123)
log_var <- log(roll_var + 1e-6) 
kmeans_res <- kmeans(log_var, centers = 3, nstart = 50) # nstart=50 for better convergence

# Sort centers: 1=Sedentary, 2=Walking, 3=Running
centers_sorted <- sort(kmeans_res$centers)

# D. Thresholding (Using standard means for robustness)
thresh1 <- exp(mean(centers_sorted[1:2])) # Sedentary vs Walking
thresh2 <- exp(mean(centers_sorted[2:3])) # Walking vs Running

df_aligned$Activity <- cut(df_aligned$RollVar, 
                           breaks = c(-Inf, thresh1, thresh2, Inf),
                           labels = c("Sedentary", "Walking", "Running"))

# E. Post-Processing: Label Smoothing (Remove Glitches)
smooth_window <- round(1.0 / dt) 
if(smooth_window %% 2 == 0) smooth_window <- smooth_window + 1 # Must be odd

act_int <- as.integer(df_aligned$Activity)
act_smooth <- runmed(act_int, k = smooth_window) # Running Median
df_aligned$Activity <- factor(act_smooth, levels=1:3, labels=c("Sedentary", "Walking", "Running"))

# Print check to see distribution
print("Activity Distribution after cleaning:")
print(table(df_aligned$Activity))


# ==========================================
# 5. Visualization
# ==========================================
# Visualization is crucial for validation. We iterate through every smoothed sensor column 
# and generate individual plots. This is necessary because different sensors operate on vastly 
# different scales (e.g., light vs. acceleration), so combining them into one plot would 
# obscure details. By coloring the curves according to the automatically detected activity 
# phases, we can visually confirm that the "Running" segments align with high-amplitude 
# data regions, providing a sanity check for our segmentation logic.

print("Generating Plots for ALL Sensors...")
smooth_cols <- grep("Smooth_", names(df_aligned), value = TRUE)

for(sensor_col in smooth_cols) {
  if(max(abs(df_aligned[[sensor_col]]), na.rm=TRUE) == 0) next
  
  # print(paste("Plotting:", sensor_col)) # Commented for brevity
  p <- ggplot(df_aligned, aes_string(x = "time", y = sensor_col, color = "Activity", group = 1)) +
    geom_line(linewidth = 0.5) +
    labs(title = paste("Smoothed Data:", sensor_col),
         subtitle = "Color indicates activity phase", y = "Value", x = "Time (s)") +
    theme_minimal() +
    scale_color_manual(values = c("Sedentary" = "#377eb8", "Walking" = "#4daf4a", "Running" = "#e41a1c")) +
    guides(color = guide_legend(override.aes = list(linewidth = 3))) +
    theme(legend.position = "top")
  print(p)
}

# Distribution Plot
p_dist <- ggplot(df_aligned, aes(x = Activity)) +
  geom_bar(fill = "steelblue", color = "black") +
  labs(title = "Distribution of Activity Phases", y = "Count") +
  theme_minimal()
print(p_dist)

# ==========================================
# 6. EXPLORATORY DATA ANALYSIS (FDA)
# ==========================================
# In this phase, we transition from continuous signal processing to Functional Data Analysis 
# (FDA). We slice the long time series into fixed 5-second windows to create distinct 
# "functional observations." This allows us to treat each 5-second interval as a single 
# mathematical object, $X_i(t)$. We visualize these observations using a Spaghetti Plot to 
# inspect the variability within each activity. We also calculate the Functional Mean to 
# establish a "canonical profile" for each activity, and the First Derivative to analyze 
# "Jerk" (the rate of change of acceleration), which reveals dynamic nuances—like the 
# smoothness or abruptness of movement—that raw magnitude cannot show.

print("Starting FDA: Slicing data into functional observations...")

# A. Data Slicing (Creating Observations)
# We define a 5-second window to match the segmentation logic.
pts_per_window <- 50 
window_duration <- 5 # seconds

# Select the best available acceleration signal
if("Smooth_LinearAccelerometerSensor" %in% names(df_aligned)) {
  acc_data <- abs(df_aligned$Smooth_LinearAccelerometerSensor)
} else {
  acc_data <- sqrt(df_aligned$Smooth_AccX^2 + df_aligned$Smooth_AccY^2 + df_aligned$Smooth_AccZ^2)
}

# Calculate how many full windows fit in the data
n_windows <- floor(length(acc_data) / pts_per_window)
print(paste("Total Windows:", n_windows))
trunc_len <- n_windows * pts_per_window

# Reshape data into a Matrix (Rows = Time Points, Cols = Observations)
acc_matrix <- matrix(acc_data[1:trunc_len], nrow = pts_per_window, ncol = n_windows)

# Identify the Activity Label for each window (using the Mode)
activity_vec <- as.integer(df_aligned$Activity[1:trunc_len])
activity_matrix <- matrix(activity_vec, nrow = pts_per_window, ncol = n_windows)

get_mode <- function(v) {
  uniqv <- unique(na.omit(v))
  if(length(uniqv)==0) return(NA)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

window_labels <- apply(activity_matrix, 2, get_mode)
window_factors <- factor(window_labels, levels=1:3, labels=c("Sedentary", "Walking", "Running"))

# B. Create Functional Data Objects
# We create B-spline basis objects with nbasis=12 to allow enough flexibility to capture
# the complex shape of movement over a 5-second window.
window_time <- seq(0, window_duration, length.out = pts_per_window)
fda_basis <- create.bspline.basis(rangeval = c(0, window_duration), nbasis = 12, norder = 4)
acc_fd <- Data2fd(window_time, acc_matrix, fda_basis)

# C. Visualization: Spaghetti Plot
print("Generating Spaghetti Plot...")
col_map <- c("Sedentary" = "#377eb8", "Walking" = "#4daf4a", "Running" = "#e41a1c")
curve_colors <- col_map[window_factors]
curve_colors[is.na(curve_colors)] <- "gray"

plot(acc_fd, col = curve_colors, lty = 1, lwd = 0.5,
     main = "Spaghetti Plot: Acceleration Magnitude (5s Windows)",
     xlab = "Relative Time (seconds)", ylab = "Acceleration (m/s^2)")
legend("topright", legend = names(col_map), col = col_map, lwd = 2)

# D. Functional Mean and Variance
print("Computing Functional Means and SDs...")
par(mfrow=c(1,3)) 
for(act in levels(window_factors)) {
  idx <- which(window_factors == act)
  if(length(idx) > 5) { 
    subset_fd <- acc_fd[idx]
    mean_curve <- mean.fd(subset_fd)
    std_curve <- std.fd(subset_fd)
    
    eval_t <- seq(0, window_duration, length.out=100)
    mean_vals <- eval.fd(eval_t, mean_curve)
    std_vals <- eval.fd(eval_t, std_curve)
    
    plot(eval_t, mean_vals, type="l", lwd=2, col=col_map[act],
         ylim=c(0, max(acc_matrix, na.rm=TRUE)),
         main=paste("Mean Profile:", act), xlab="Time (s)", ylab="Acc (m/s^2)")
    lines(eval_t, mean_vals + 2*std_vals, lty=2, col="darkgray")
    lines(eval_t, mean_vals - 2*std_vals, lty=2, col="darkgray")
  }
}
par(mfrow=c(1,1)) 

# E. Derivative Analysis (Jerk)
print("Calculating Derivatives (Jerk)...")
acc_deriv_fd <- deriv.fd(acc_fd, 1)
plot(acc_deriv_fd, col = curve_colors, lwd = 0.5,
     main = "1st Derivative (Jerk) by Activity",
     xlab = "Relative Time (seconds)", ylab = "Jerk (m/s^3)")
legend("topright", legend = names(col_map), col = col_map, lwd = 2)

print("FDA Exploration Complete.")

# ==========================================
# 7. CONFORMAL PREDICTION (Full Approach 1: FPCA + KDE)
# ==========================================
# Here we implement the Inductive Conformal Prediction (ICP) using a Density-Based approach,
# which corresponds to Approach 1 in the Lei et al. paper. Instead of a simple distance metric, 
# we project the functional curves onto a lower-dimensional space using Functional PCA (FPCA). 
# This reduces the complex curves to a few score variables. We then estimate the Probability 
# Density Function (PDF) of the "Normal" data in this PCA space using Kernel Density Estimation 
# (KDE). The non-conformity score is defined as the negative log-likelihood; points in low-density 
# regions (anomalies) receive high scores. This method captures the "shape" of normality much 
# better than a simple mean-based distance. We use an Inductive split (Train/Calibration/Test) 
# to ensure computational efficiency.

print("Starting Inductive Conformal Prediction...")

# A. Dynamic Training Class Selection
# We prioritize "Walking" as the normal class. If data is insufficient, we fallback to "Sedentary".
training_label <- "Walking"
if (sum(window_factors == "Walking", na.rm=TRUE) < 20) {
  training_label <- "Sedentary"
  print("Switching to Sedentary for training.")
}
if (sum(window_factors == training_label, na.rm=TRUE) < 20) stop("Not enough data to train.")

# Indices
idx_normal <- which(window_factors == training_label)
idx_anomaly <- which(window_factors == "Running") 

set.seed(123)
idx_cal <- sample(idx_normal, size = floor(length(idx_normal) * 0.5))
idx_train <- setdiff(idx_normal, idx_cal)

fd_train <- acc_fd[idx_train]   
fd_cal   <- acc_fd[idx_cal]     
fd_test  <- acc_fd[idx_anomaly] 

print(paste("Training Set:", length(idx_train)))
print(paste("Calibration Set:", length(idx_cal)))
print(paste("Test Set (Running):", length(idx_anomaly)))

# B. Functional PCA
# We project the curves onto 2 principal components, capturing the main modes of variation.
nharm <- 2 
pca_model <- pca.fd(fd_train, nharm = nharm)
var_prop <- sum(pca_model$varprop) * 100
print(paste("FPCA Variance Explained:", round(var_prop, 2), "%"))

# Function to manually project new curves onto the PCA basis
project_fd_to_pca <- function(new_fd, pca_obj) {
  # Evaluate on fine grid
  t_eval <- seq(0, 5, length.out=100)
  mat_new <- eval.fd(t_eval, new_fd)
  vec_mean <- eval.fd(t_eval, pca_obj$meanfd)
  
  # Center the data (Subtract Training Mean)
  mat_centered <- sweep(mat_new, 1, vec_mean, "-") 
  
  # Project via inner product
  fd_centered <- Data2fd(t_eval, mat_centered, pca_obj$harmonics$basis)
  return(inprod(fd_centered, pca_obj$harmonics))
}

# Get Scores
scores_train <- pca_model$scores
scores_cal   <- project_fd_to_pca(fd_cal, pca_model)
scores_test  <- project_fd_to_pca(fd_test, pca_model)

# C. Density Estimation (KDE)
# We estimate the density of the training scores and define the conformity score as -log(density).
H_band <- Hpi(x = scores_train) 
kde_model <- kde(x = scores_train, H = H_band)

get_density_score <- function(new_scores, kde_obj) {
  dens_vals <- predict(kde_obj, x = new_scores)
  dens_vals[dens_vals < 1e-300] <- 1e-300 # Avoid log(0)
  return(-log(dens_vals))
}

alpha_cal <- get_density_score(scores_cal, kde_model)
alpha_test <- get_density_score(scores_test, kde_model)

# D. P-Values
calculate_p_value <- function(new_score, cal_scores) {
  (sum(cal_scores >= new_score) + 1) / (length(cal_scores) + 1)
}
p_values <- sapply(alpha_test, calculate_p_value, cal_scores = alpha_cal)

# E. Results
detection_rate <- mean(p_values < 0.05) * 100
print(paste("Anomaly Detection Rate:", round(detection_rate, 2), "%"))

# Visual 1: PCA Space
df_pca <- data.frame(
  PC1 = c(scores_train[,1], scores_cal[,1], scores_test[,1]),
  PC2 = c(scores_train[,2], scores_cal[,2], scores_test[,2]),
  Group = c(rep("Train", nrow(scores_train)),
            rep("Cal", nrow(scores_cal)),
            rep("Test (Run)", nrow(scores_test)))
)

p_pca <- ggplot(df_pca, aes(x=PC1, y=PC2, color=Group)) +
  geom_point(alpha=0.6) +
  stat_density_2d(data=subset(df_pca, Group=="Train"), color="black", alpha=0.5) +
  labs(title = "PCA Space: Normal Density vs Anomalies") + theme_minimal()
print(p_pca)

# Visual 2: Boxplot
df_scores <- data.frame(
  Score = c(alpha_cal, alpha_test),
  Group = c(rep("Normal", length(alpha_cal)), rep("Anomaly", length(alpha_test)))
)

p_scores <- ggplot(df_scores, aes(x = Group, y = Score, fill = Group)) +
  geom_boxplot() +
  geom_hline(yintercept = quantile(alpha_cal, 0.95), linetype="dashed", color="red") +
  labs(title = "Conformity Scores (-Log Density)", y = "Score") + theme_minimal()
print(p_scores)

print("Implementation Complete.")

# ==========================================
# 8. OUTLIER INVESTIGATION
# ==========================================
# Anomalies in the calibration set (outliers) can skew the threshold. We identify them using 
# standard boxplot logic (1.5 * IQR) and visualize them against the mean curve. This helps 
# us understand if these "Normal" outliers are actually mislabeled events (e.g., stumbles) 
# or just natural variations in walking intensity.

# 1. Identify Outlier Indices
bp_stats <- boxplot.stats(alpha_cal)
outlier_values <- bp_stats$out
outlier_indices <- which(alpha_cal %in% outlier_values)

print(paste("Found", length(outlier_indices), "outliers in the Calibration set."))

# 2. Visualize the Top 2 Extreme Outliers
if(length(outlier_indices) > 0) {
  sorted_outliers <- outlier_indices[order(alpha_cal[outlier_indices], decreasing = TRUE)]
  top_outliers <- head(sorted_outliers, 2)
  
  par(mfrow=c(1,1))
  t_eval <- seq(0, 5, length.out=100)
  mean_vals <- eval.fd(t_eval, pca_model$meanfd) # Use PCA mean
  plot(t_eval, mean_vals, type="l", lwd=4, col="black", 
       ylim=c(0, max(mean_vals)*3),
       main = "Why are these Outliers?",
       xlab = "Time (s)", ylab = "Acceleration Magnitude (m/s^2)")
  
  outlier_curves <- eval.fd(t_eval, fd_cal[top_outliers])
  matlines(t_eval, outlier_curves, col="red", lwd=2, lty=1)
  
  normal_indices <- setdiff(1:length(alpha_cal), outlier_indices)
  some_normals <- head(normal_indices, 5)
  normal_curves <- eval.fd(t_eval, fd_cal[some_normals])
  matlines(t_eval, normal_curves, col="gray", lwd=1, lty=2)
  
  legend("topright", legend=c("Mean Walking", "The Outliers", "Typical Walking"), 
         col=c("black", "red", "gray"), lwd=c(4, 2, 1), lty=c(1,1,2))
} else {
  print("No statistical outliers found.")
}

# ==========================================
# 9. VISUALIZATION: ZOOMED BOXPLOT (Focus on Normal)
# ==========================================
# Often, the "Anomaly" scores are so high that they squash the "Normal" boxplot into a flat 
# line, making it impossible to see the threshold or spread. Here, we force the Y-axis to 
# zoom in on the Normal distribution. We calculate a view limit slightly above the 95% 
# threshold, effectively cutting off the massive Anomaly scores, to provide a clear view 
# of the decision boundary and the normal data variance.

print("Generating Zoomed Boxplot...")

# Define Zoom Limits based on 'Normal' Data only
threshold_val <- quantile(alpha_cal, 0.95)
q3_cal <- quantile(alpha_cal, 0.75)
iqr_cal <- IQR(alpha_cal)
upper_fence_cal <- q3_cal + (1.5 * iqr_cal)

y_limit_upper <- max(threshold_val, upper_fence_cal) * 1.5 
y_limit_lower <- min(alpha_cal) * 0.9

p_zoomed <- ggplot(df_scores, aes(x = Group, y = Score, fill = Group)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.7) + 
  geom_hline(yintercept = threshold_val, linetype = "dashed", color = "red", linewidth = 1) +
  coord_cartesian(ylim = c(y_limit_lower, y_limit_upper)) +
  labs(title = "Conformity Scores (Zoomed Focus)",
       subtitle = paste("Y-axis limited to", round(y_limit_upper, 2), 
                        "to show Normal spread vs Threshold"),
       y = "Score (-log Density)") +
  theme_minimal() +
  scale_fill_manual(values = c("gray", "#e41a1c")) +
  annotate("text", x = 1, y = threshold_val, label = "95% Threshold", 
           vjust = -1, color = "red", fontface = "bold")

print(p_zoomed)
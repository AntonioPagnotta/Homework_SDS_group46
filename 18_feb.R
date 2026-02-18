# ==========================================
# DATA PREPROCESSING & CONFORMAL PREDICTION
# ==========================================

# Load necessary libraries
if(!require(dplyr)) install.packages("dplyr")
if(!require(zoo)) install.packages("zoo")
if(!require(fda)) install.packages("fda")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(reshape2)) install.packages("reshape2")

library(dplyr)
library(zoo)
library(fda)
library(ggplot2)
library(reshape2)

# ==========================================
# 1. Load and Inspect
# ==========================================
# In this initial step, we load the raw sensor data and immediately sort it by timestamp 
# because time series analysis strictly requires ordered data to function correctly. 
# We then convert the raw timestamps—which are massive integers representing milliseconds—into 
# a relative time scale (seconds from the start) to ensure numerical stability during 
# complex matrix calculations later on. Finally, we dynamically identify all sensor columns 
# present in the file, making the script robust enough to automatically handle any sensor 
# configuration without requiring manual code adjustments.

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
# Because the sensors are asynchronous and event-based, they fire at different times, 
# resulting in a sparse dataset full of gaps (NaNs) that would break standard statistical 
# algorithms. To resolve this, we create a common, continuous time grid sampled at 10Hz.
# Human movement frequencies (walking, hand gestures) generally fall below 5Hz. According
# to the Nyquist-Shannon sampling theorem, you need a sampling rate at least double the 
# highest frequency to capture the signal without aliasing. We then 
# use linear interpolation to align every sensor onto this shared grid, transforming 
# the disparate, hole-riddled columns into a dense, synchronized matrix suitable for 
# joint analysis.

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
# Processing the entire 1.6-hour dataset at once would require creating a massive basis matrix 
# that exceeds standard RAM limits, causing crashes. To prevent this, we split the data into 
# manageable chunks and smooth them sequentially. Within each chunk, we apply B-spline 
# smoothing with a roughness penalty on the second derivative. We choose B-splines because 
# human movement is non-periodic and transient, unlike Fourier waves, and the penalty ensures 
# we capture the underlying physical trend while filtering out the high-frequency jitter and 
# noise inherent in raw sensor data.

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
  print(paste("Processing sensor:", sensor))
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
# REASONING FOR CHANGES:
# 1. Larger Window (5s): A 2-second window was too sensitive, causing short 
#    bursts of walking intensity to look like running. Increasing to 5s smoothes 
#    the variance estimate.
# 2. Weighted Thresholds: We moved the Running threshold higher (closer to the 
#    Running cluster center) to reduce false positives.
# 3. Label Smoothing: We apply a "majority vote" filter to remove single-point 
#    glitches (e.g., Walk-Walk-Run-Walk-Walk becomes all Walk).

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

# D. Conservative Thresholding
# Instead of a simple mean (0.5/0.5), we weight the upper threshold towards Running.
# Formula: 0.3 * Walk_Center + 0.7 * Run_Center
# This pushes the boundary UP, requiring higher variance to classify as Running.
thresh1 <- exp(mean(centers_sorted[1:2])) # Sedentary vs Walking (Standard Mean)
thresh2 <- exp(0.3 * centers_sorted[2] + 0.7 * centers_sorted[3]) # Walking vs Running (Conservative)

df_aligned$Activity <- cut(df_aligned$RollVar, 
                           breaks = c(-Inf, thresh1, thresh2, Inf),
                           labels = c("Sedentary", "Walking", "Running"))

# E. Post-Processing: Label Smoothing (Remove Glitches)
# We use a running median (window = 1 second) to filter out flickering labels.
# (e.g., Walk-Walk-Run-Walk -> Walk-Walk-Walk-Walk)
# Convert factor to integer, smooth, convert back.
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
# Finally, we iterate through every smoothed sensor column and generate individual plots 
# to visually validate our results. This step is crucial because different sensors operate 
# on vastly different scales, making a single combined plot unreadable. By plotting them 
# separately and coloring the curves according to the detected activity phases, we can 
# confirm that the "Running" segments align with high-amplitude data regions and assess 
# the signal quality of every sensor in isolation.

print("Generating Plots for ALL Sensors...")
smooth_cols <- grep("Smooth_", names(df_aligned), value = TRUE)

for(sensor_col in smooth_cols) {
  if(max(abs(df_aligned[[sensor_col]]), na.rm=TRUE) == 0) next
  
  print(paste("Plotting:", sensor_col))
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
# In this analytical phase, we transition from continuous signal processing to true 
# Functional Data Analysis. We slice the long, continuous time series into fixed-length 
# windows (5 seconds) to create distinct "functional observations" or curves. This allows 
# us to treat every 5-second interval as a single mathematical object X_i(t). 
# We then visualize these curves using a "Spaghetti Plot" to inspect the variability 
# within each activity phase. Furthermore, we calculate the Functional Mean and Standard 
# Deviation for each activity to establish a "canonical profile" for Sedentary vs. Running 
# behaviors. Finally, we compute the first derivative of these curves to analyze "Jerk" 
# (the rate of change of acceleration), which often reveals dynamic nuances—like the 
# smoothness or abruptness of movement—that raw acceleration magnitude cannot show.

print("Starting FDA: Slicing data into functional observations...")

# -------------------------------------------------------
# A. Data Slicing (Creating Observations)
# -------------------------------------------------------
# We define a 5-second window to match the segmentation logic.
# At 10Hz (dt=0.1s), this equals 50 data points per curve.
window_duration <- 5 # seconds
pts_per_window <- round(window_duration / dt)

print(paste("Window Duration:", window_duration, "s"))
print(paste("Points per Window:", pts_per_window))

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
# If a window contains mostly "Running", we label the whole window "Running"
activity_vec <- as.integer(df_aligned$Activity[1:trunc_len])
activity_matrix <- matrix(activity_vec, nrow = pts_per_window, ncol = n_windows)

get_mode <- function(v) {
  uniqv <- unique(na.omit(v))
  if(length(uniqv)==0) return(NA)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

window_labels <- apply(activity_matrix, 2, get_mode)
window_factors <- factor(window_labels, levels=1:3, labels=c("Sedentary", "Walking", "Running"))

# -------------------------------------------------------
# B. Create Functional Data Objects
# -------------------------------------------------------
# We create the 'fd' objects. We use a B-spline basis with enough flexibility (nbasis=12)
# to capture the shape of movement within a 5-second window.
window_time <- seq(0, window_duration, length.out = pts_per_window)
fda_basis <- create.bspline.basis(rangeval = c(0, window_duration), nbasis = 12, norder = 4)
acc_fd <- Data2fd(window_time, acc_matrix, fda_basis)

# -------------------------------------------------------
# C. Visualization: Spaghetti Plot
# -------------------------------------------------------
print("Generating Spaghetti Plot...")

# Define colors
col_map <- c("Sedentary" = "#377eb8", "Walking" = "#4daf4a", "Running" = "#e41a1c")
curve_colors <- col_map[window_factors]
# Handle NAs if any windows were unclassified
curve_colors[is.na(curve_colors)] <- "gray"

# Plot all curves overlaid
plot(acc_fd, col = curve_colors, lty = 1, lwd = 0.5,
     main = "Spaghetti Plot: Acceleration Magnitude (5s Windows)",
     xlab = "Relative Time (seconds)", ylab = "Acceleration (m/s^2)")
legend("topright", legend = names(col_map), col = col_map, lwd = 2)

# -------------------------------------------------------
# D. Functional Mean and Variance
# -------------------------------------------------------
print("Computing Functional Means and SDs...")

# Set up a 1x3 plotting grid to show each activity clearly
par(mfrow=c(1,3)) 

for(act in levels(window_factors)) {
  # Find windows belonging to this activity
  idx <- which(window_factors == act)
  
  if(length(idx) > 5) { # Only plot if we have enough data
    subset_fd <- acc_fd[idx]
    
    # Compute functional statistics
    mean_curve <- mean.fd(subset_fd)
    std_curve <- std.fd(subset_fd)
    
    # Evaluate for plotting confidence bands (Mean +/- 2SD)
    eval_t <- seq(0, window_duration, length.out=100)
    mean_vals <- eval.fd(eval_t, mean_curve)
    std_vals <- eval.fd(eval_t, std_curve)
    
    # Plot
    plot(eval_t, mean_vals, type="l", lwd=2, col=col_map[act],
         ylim=c(0, max(acc_matrix, na.rm=TRUE)),
         main=paste("Mean Profile:", act), 
         xlab="Time (s)", ylab="Acc (m/s^2)")
    
    # Add Standard Deviation bands (dashed lines)
    lines(eval_t, mean_vals + 2*std_vals, lty=2, col="darkgray")
    lines(eval_t, mean_vals - 2*std_vals, lty=2, col="darkgray")
  }
}
par(mfrow=c(1,1)) # Reset plot layout

# -------------------------------------------------------
# E. Derivative Analysis (Jerk)
# -------------------------------------------------------
print("Calculating Derivatives (Jerk)...")

# The first derivative of Acceleration is Jerk (m/s^3)
# High jerk indicates jerky, uncontrolled, or high-impact movement.
acc_deriv_fd <- deriv.fd(acc_fd, 1)

plot(acc_deriv_fd, col = curve_colors, lwd = 0.5,
     main = "1st Derivative (Jerk) by Activity",
     xlab = "Relative Time (seconds)", ylab = "Jerk (m/s^3)")
legend("topright", legend = names(col_map), col = col_map, lwd = 2)

print("FDA Exploration Complete.")
# ==========================================
# DATA PREPROCESSING & CONFORMAL PREDICTION (FINAL CORRECTED)
# ==========================================

# Load necessary libraries
if(!require(dplyr)) install.packages("dplyr")
if(!require(zoo)) install.packages("zoo")
if(!require(fda)) install.packages("fda")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(reshape2)) install.packages("reshape2")
if(!require(ks)) install.packages("ks") # Required for Density CP

library(dplyr)
library(zoo)
library(fda)
library(ggplot2)
library(reshape2)
library(ks)

# ==========================================
# 1. Load and Inspect
# ==========================================
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
print("Smoothing in chunks...")
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
  # print(paste("Processing sensor:", sensor)) # Commented to reduce spam
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
# 4. Segmentation (Robust 5s Window)
# ==========================================
print("Segmenting Activity Phases...")

# A. Calculate Magnitude
if("Smooth_LinearAccelerometerSensor" %in% names(df_aligned) && 
   sum(abs(df_aligned$Smooth_LinearAccelerometerSensor)) > 10) {
  mag <- abs(df_aligned$Smooth_LinearAccelerometerSensor)
} else {
  mag <- sqrt(df_aligned$Smooth_AccX^2 + df_aligned$Smooth_AccY^2 + df_aligned$Smooth_AccZ^2)
}

# B. Rolling Variance (5s Window)
window_size <- round(5 / dt) 
roll_var <- rollapply(mag, width = window_size, FUN = var, fill = NA, align = "right")
roll_var[is.na(roll_var)] <- 0
df_aligned$RollVar <- roll_var

# C. Clustering
set.seed(123)
log_var <- log(roll_var + 1e-6) 
kmeans_res <- kmeans(log_var, centers = 3, nstart = 50)
centers_sorted <- sort(kmeans_res$centers)

# D. Thresholding (Reverted to Standard Mean for Robustness)
# Using simple midpoints prevents "Running" from being missed if the threshold is too high
thresh1 <- exp(mean(centers_sorted[1:2])) 
thresh2 <- exp(mean(centers_sorted[2:3])) 

df_aligned$Activity <- cut(df_aligned$RollVar, 
                           breaks = c(-Inf, thresh1, thresh2, Inf),
                           labels = c("Sedentary", "Walking", "Running"))

# E. Label Smoothing
smooth_window <- round(1.0 / dt) 
if(smooth_window %% 2 == 0) smooth_window <- smooth_window + 1
act_int <- as.integer(df_aligned$Activity)
act_smooth <- runmed(act_int, k = smooth_window)
df_aligned$Activity <- factor(act_smooth, levels=1:3, labels=c("Sedentary", "Walking", "Running"))

print("Activity Distribution:")
print(table(df_aligned$Activity))

# ==========================================
# 5. Visualization
# ==========================================
print("Generating Sensor Plots...")
smooth_cols <- grep("Smooth_", names(df_aligned), value = TRUE)

for(sensor_col in smooth_cols) {
  if(max(abs(df_aligned[[sensor_col]]), na.rm=TRUE) == 0) next
  
  p <- ggplot(df_aligned, aes_string(x = "time", y = sensor_col, color = "Activity", group = 1)) +
    geom_line(linewidth = 0.5) +
    labs(title = paste("Smoothed:", sensor_col), y = "Value", x = "Time (s)") +
    theme_minimal() +
    scale_color_manual(values = c("Sedentary" = "#377eb8", "Walking" = "#4daf4a", "Running" = "#e41a1c")) +
    guides(color = guide_legend(override.aes = list(linewidth = 3))) +
    theme(legend.position = "top")
  print(p)
}

# ==========================================
# 6. EXPLORATORY DATA ANALYSIS (FDA)
# ==========================================
print("Starting FDA Slicing...")

# A. Data Slicing (5s Windows)
window_duration <- 5 
pts_per_window <- round(window_duration / dt)

# Use Acc Magnitude for FDA
if("Smooth_LinearAccelerometerSensor" %in% names(df_aligned)) {
  acc_data <- abs(df_aligned$Smooth_LinearAccelerometerSensor)
} else {
  acc_data <- sqrt(df_aligned$Smooth_AccX^2 + df_aligned$Smooth_AccY^2 + df_aligned$Smooth_AccZ^2)
}

n_windows <- floor(length(acc_data) / pts_per_window)
trunc_len <- n_windows * pts_per_window
acc_matrix <- matrix(acc_data[1:trunc_len], nrow = pts_per_window, ncol = n_windows)

# Get Window Labels
activity_vec <- as.integer(df_aligned$Activity[1:trunc_len])
activity_matrix <- matrix(activity_vec, nrow = pts_per_window, ncol = n_windows)
get_mode <- function(v) {
  uniqv <- unique(na.omit(v))
  if(length(uniqv)==0) return(NA)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
window_labels <- apply(activity_matrix, 2, get_mode)
window_factors <- factor(window_labels, levels=1:3, labels=c("Sedentary", "Walking", "Running"))

# B. Create FD Objects
window_time <- seq(0, window_duration, length.out = pts_per_window)
fda_basis <- create.bspline.basis(rangeval = c(0, window_duration), nbasis = 12, norder = 4)
acc_fd <- Data2fd(window_time, acc_matrix, fda_basis)

# C. Plotting
col_map <- c("Sedentary" = "#377eb8", "Walking" = "#4daf4a", "Running" = "#e41a1c")
curve_colors <- col_map[window_factors]
curve_colors[is.na(curve_colors)] <- "gray"

plot(acc_fd, col = curve_colors, lty = 1, lwd = 0.5,
     main = "Spaghetti Plot (5s Windows)", xlab = "Time (s)", ylab = "Acc (m/s^2)")
legend("topright", legend = names(col_map), col = col_map, lwd = 2)

# ==========================================
# 7. CONFORMAL PREDICTION (Full Approach 1: FPCA + KDE)
# ==========================================
# FIX: Added 'project_fd_to_pca' function to solve the "function not found" error.
# This manually calculates scores for new data.

print("Starting Inductive Conformal Prediction...")

# A. Dynamic Training Class Selection
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
nharm <- 2 
pca_model <- pca.fd(fd_train, nharm = nharm)
var_prop <- sum(pca_model$varprop) * 100
print(paste("FPCA Variance Explained:", round(var_prop, 2), "%"))

# Function to manually project new curves onto the PCA basis
project_fd_to_pca <- function(new_fd, pca_obj) {
  # 1. Evaluate curves and mean on a fine grid
  t_eval <- seq(0, 5, length.out=100)
  mat_new <- eval.fd(t_eval, new_fd)
  vec_mean <- eval.fd(t_eval, pca_obj$meanfd)
  
  # 2. Center the data (Subtract Training Mean)
  mat_centered <- sweep(mat_new, 1, vec_mean, "-") 
  
  # 3. Create FD object for centered data
  fd_centered <- Data2fd(t_eval, mat_centered, pca_obj$harmonics$basis)
  
  # 4. Inner product with Harmonics -> Scores
  return(inprod(fd_centered, pca_obj$harmonics))
}

# Get Scores
scores_train <- pca_model$scores
scores_cal   <- project_fd_to_pca(fd_cal, pca_model)
scores_test  <- project_fd_to_pca(fd_test, pca_model)

# C. Density Estimation (KDE)
# Use Hpi for optimal bandwidth selection
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
# You noticed outliers in the boxplot. Let's find out exactly what they are.
# These are likely "Walking" windows that had unusually high energy (e.g., a stumble).

# 1. Identify Outlier Indices using standard Boxplot logic (1.5 * IQR)
# We look at 'alpha_cal' (The scores of the Calibration/Walking set)
bp_stats <- boxplot.stats(alpha_cal)
outlier_values <- bp_stats$out
outlier_indices <- which(alpha_cal %in% outlier_values)

print(paste("Found", length(outlier_indices), "outliers in the Calibration (Walking) set."))
print("Outlier Scores:")
print(outlier_values)

# 2. Visualize the Top 2 Extreme Outliers
# We plot them against the "Mean Walking Curve" to see WHY they were flagged.
if(length(outlier_indices) > 0) {
  # Sort to get the most extreme ones (highest scores)
  sorted_outliers <- outlier_indices[order(alpha_cal[outlier_indices], decreasing = TRUE)]
  top_outliers <- head(sorted_outliers, 2) # Grab top 2
  
  # Prepare Plotting
  par(mfrow=c(1,1))
  t_eval <- seq(0, 5, length.out=100)
  
  # A. Plot the Ideal Mean Curve (Black)
  mean_vals <- eval.fd(t_eval, mean_curve_train)
  plot(t_eval, mean_vals, type="l", lwd=4, col="black", 
       ylim=c(0, max(mean_vals)*3), # Scale y-axis to fit potential spikes
       main = "Why are these Outliers?",
       xlab = "Time (s)", ylab = "Acceleration Magnitude (m/s^2)")
  
  # B. Plot the Outlier Curves (Red)
  outlier_curves <- eval.fd(t_eval, fd_cal[top_outliers])
  matlines(t_eval, outlier_curves, col="red", lwd=2, lty=1)
  
  # C. Plot a few "Normal" curves for context (Gray)
  normal_indices <- setdiff(1:length(alpha_cal), outlier_indices)
  some_normals <- head(normal_indices, 5)
  normal_curves <- eval.fd(t_eval, fd_cal[some_normals])
  matlines(t_eval, normal_curves, col="gray", lwd=1, lty=2)
  
  legend("topright", 
         legend=c("Mean Walking", "The Outliers", "Typical Walking"), 
         col=c("black", "red", "gray"), lwd=c(4, 2, 1), lty=c(1,1,2))
  
  print("Plot generated. The RED lines are the outliers you saw.")
} else {
  print("No statistical outliers found in the Calibration set based on 1.5*IQR rule.")
}

# ==========================================
# 9. VISUALIZATION: ZOOMED BOXPLOT (Focus on Normal)
# ==========================================
# The "Running" scores are likely huge (e.g., 100x larger), which squashes the 
# "Walking" boxplot. Here, we force the Y-axis to zoom in on the Normal distribution 
# and the Threshold, cutting off the top of the Anomaly box.

print("Generating Zoomed Boxplot...")

# 1. Define Zoom Limits based on 'Normal' Data only
# We want to see the 95% Threshold and the bulk of Normal data.
# We ignore the massive Running scores for the upper limit calculation.
threshold_val <- quantile(alpha_cal, 0.95)

# Calculate Upper Fence of Normal data (Standard Boxplot logic)
q3_cal <- quantile(alpha_cal, 0.75)
iqr_cal <- IQR(alpha_cal)
upper_fence_cal <- q3_cal + (1.5 * iqr_cal)

# Set the View Limit to slightly above the Threshold or the Normal Fence
# (whichever is higher), but not all the way to the Running scores.
y_limit_upper <- max(threshold_val, upper_fence_cal) * 1.5 
y_limit_lower <- min(alpha_cal) * 0.9

# 2. Plot with Coordinate Zoom (coord_cartesian)
# NOTE: We use coord_cartesian instead of scale_y_continuous limits 
# because coord_cartesian zooms in without removing the data points from 
# the statistics calculations.
p_zoomed <- ggplot(df_scores, aes(x = Group, y = Score, fill = Group)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.7) + # Hide outliers dots
  
  # Add the Threshold Line
  geom_hline(yintercept = threshold_val, linetype = "dashed", color = "red", linewidth = 1) +
  
  # FORCE ZOOM
  coord_cartesian(ylim = c(y_limit_lower, y_limit_upper)) +
  
  # Labels and Theme
  labs(title = "Conformity Scores (Zoomed Focus)",
       subtitle = paste("Y-axis limited to", round(y_limit_upper, 2), 
                        "to show Normal spread vs Threshold"),
       y = "Score (-log Density)") +
  theme_minimal() +
  scale_fill_manual(values = c("gray", "#e41a1c")) +
  annotate("text", x = 1, y = threshold_val, label = "95% Threshold", 
           vjust = -1, color = "red", fontface = "bold")

print(p_zoomed)
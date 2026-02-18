# ==========================================
# DATA PREPROCESSING
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
# highest frequency to capture the signal without aliasing.We then 
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
# 4. Segmentation
# ==========================================
# To distinguish between Sedentary, Walking, and Running phases without manual labels, 
# we rely on the variance of the acceleration magnitude, as the mean acceleration is 
# dominated by constant gravity. Since variance scales exponentially across these activities 
# (from nearly zero to very high), we apply a log-transformation to make the values linearly 
# separable. We then use K-Means clustering to automatically detect the three natural intensity 
# levels in the data, assigning an activity label to every time point based on these 
# discovered thresholds.

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
# EXPLORATORY DATA ANALYSIS (FDA)
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
# We define a 5-second window. At 10Hz, this equals 50 data points per curve.
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
cat(n_windows)
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
# We create the 'fd' objects. We use a B-spline basis with enough flexibility (nbasis=10)
# to capture the shape of movement within a 5-second window.
window_time <- seq(0, window_duration, length.out = pts_per_window)
fda_basis <- create.bspline.basis(rangeval = c(0, window_duration), nbasis = 10, norder = 4)
acc_fd <- Data2fd(window_time, acc_matrix, fda_basis)

# -------------------------------------------------------
# C. Visualization: Spaghetti Plot
# -------------------------------------------------------
print("Generating Spaghetti Plot...")

# Define colors
col_map <- c("Sedentary" = "red", "Walking" = "darkgreen", "Running" = "lightblue")
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
# 7. CONFORMAL PREDICTION (Inductive Approach)
# ==========================================
# In this final computational step, we implement Inductive Conformal Prediction (ICP) 
# to build an anomaly detection system. Our goal is to "learn" the normal behavior of 
# the "Walking" phase and then rigorously test if the "Running" curves are statistically 
# distinct (non-conformal).
#
# We choose ICP over Transductive CP because calculating the functional mean is expensive.
# ICP splits the "Normal" data into a Training Set (to find the mean) and a Calibration 
# Set (to learn the threshold), allowing us to score new observations instantly.

print("Starting Inductive Conformal Prediction...")

# -------------------------------------------------------
# A. Data Preparation & Splitting
# -------------------------------------------------------
# We assume 'acc_fd' (the functional object of all 5s windows) and 'window_labels' 
# (the activity label for each window) exist from Section 6 (EDA).

# 1. Define "Normal" (Walking) and "Anomaly" (Running) indices
idx_walking <- which(window_labels == "Walking")
idx_running <- which(window_labels == "Running")

# Check if we have enough data
if(length(idx_walking) < 20) stop("Not enough 'Walking' data to train CP.")

# 2. Split "Normal" (Walking) into Proper Training and Calibration Sets
# We use a 50/50 split. 
# - Proper Training: Used to calculate the Mean Curve.
# - Calibration: Used to calculate the distribution of non-conformity scores (residuals).
set.seed(123)
n_walk <- length(idx_walking)
idx_cal <- sample(idx_walking, size = floor(n_walk * 0.5))
idx_train <- setdiff(idx_walking, idx_cal)

# Create subsets of the functional data
fd_train <- acc_fd[idx_train] # Normal behavior reference
fd_cal   <- acc_fd[idx_cal]   # Used to set the threshold
fd_test  <- acc_fd[idx_running] # Anomalies we want to detect

print(paste("Training Curves:", length(idx_train)))
print(paste("Calibration Curves:", length(idx_cal)))
print(paste("Test Curves (Running):", length(idx_running)))

# -------------------------------------------------------
# B. Define Non-Conformity Measure (Score)
# -------------------------------------------------------
# We calculate the Functional Mean of the Proper Training set.
# This represents the "ideal" Walking curve.
mean_curve_train <- mean.fd(fd_train)

# Define the Scoring Function: Integrated Squared Error (ISE)
# Formula: Integral( (X_i(t) - Mean(t))^2 dt )
# Reasoning: We want to flag curves that differ in Amplitude (energy) or Shape. 
# ISE captures the total deviation over the entire 5-second window.

get_conformity_score <- function(target_fd, reference_mean_fd) {
  # Evaluate both functions on a fine grid (100 points over 5 seconds)
  eval_t <- seq(0, 5, length.out = 100)
  
  # Get values matrix
  mat_target <- eval.fd(eval_t, target_fd) # Cols = curves
  vec_ref    <- eval.fd(eval_t, reference_mean_fd) # Single vector
  
  # Calculate Squared Errors
  # (Observation - Mean)^2
  sq_diff <- (mat_target - as.vector(vec_ref))^2
  
  # Integrate (Approximate via Riemann Sum)
  # Sum of squared diffs * dt (time step)
  dt <- 5 / 100
  scores <- colSums(sq_diff) * dt
  
  return(scores)
}

# -------------------------------------------------------
# C. Calibration Phase
# -------------------------------------------------------
# We calculate the scores (alpha) for the Calibration set.
# These scores tell us "How much does a normal Walking curve typically deviate from the mean?"
alpha_cal <- get_conformity_score(fd_cal, mean_curve_train)

# -------------------------------------------------------
# D. Prediction / Testing Phase
# -------------------------------------------------------
# Now we test the "Running" curves. We hypothesize that if our CP model works,
# running curves will have huge deviation scores compared to the calibration scores.

# 1. Calculate scores for the Test Set (Running)
alpha_test <- get_conformity_score(fd_test, mean_curve_train)

# 2. Compute P-Values
# For each test curve, the p-value is the fraction of Calibration scores 
# that are GREATER than the test score.
# If p-value is very low (e.g., < 0.05), it means the test score is higher than 
# 95% of the normal data, so we reject the "Normal" hypothesis.
calculate_p_value <- function(new_score, cal_scores) {
  # Formula: (|{cal_scores >= new_score}| + 1) / (n_cal + 1)
  (sum(cal_scores >= new_score) + 1) / (length(cal_scores) + 1)
}

# Apply to all test curves
p_values <- sapply(alpha_test, calculate_p_value, cal_scores = alpha_cal)

# -------------------------------------------------------
# E. Results and Visualization
# -------------------------------------------------------
# Define significance level (epsilon)
epsilon <- 0.05

# Classify: If p < epsilon, it is an Anomaly (Non-Conformal)
is_anomaly <- p_values < epsilon
detection_rate <- sum(is_anomaly) / length(is_anomaly) * 100

print(paste("Detection Rate (Running classified as Anomaly):", round(detection_rate, 2), "%"))

# VISUALIZATION 1: Score Distribution
# We visualize the separation between Normal (Calibration) scores and Anomaly (Test) scores.
df_scores <- data.frame(
  Score = c(alpha_cal, alpha_test),
  Type = c(rep("Calibration (Walking)", length(alpha_cal)), 
           rep("Test (Running)", length(alpha_test)))
)

p_scores <- ggplot(df_scores, aes(x = Type, y = Score, fill = Type)) +
  geom_boxplot() +
  scale_y_log10() + # Use Log scale because running energy is much higher
  labs(title = "Conformity Scores: Walking vs Running",
       subtitle = "Higher score = More different from Mean Walking Curve",
       y = "Integrated Squared Error (Log Scale)") +
  theme_minimal() +
  geom_hline(yintercept = quantile(alpha_cal, 0.95), linetype="dashed", color="red") +
  annotate("text", x=1, y=quantile(alpha_cal, 0.95), label="95% Threshold", vjust=-1)

print(p_scores)

# VISUALIZATION 2: Functional Plot of Detected Anomalies
# We plot the Mean Walking Curve vs. a few detected Anomalies
par(mfrow=c(1,1))
eval_t <- seq(0, 5, length.out=100)
mean_vals <- eval.fd(eval_t, mean_curve_train)

# Plot Mean Walking Curve (Black)
plot(eval_t, mean_vals, type="l", lwd=3, col="black", ylim=c(0, max(mean_vals)*3),
     main="CP Result: Normal Mean vs Detected Anomalies",
     xlab="Time (s)", ylab="Acceleration Magnitude")

# Overlay first 5 detected anomalies (Red)
anom_idx <- which(is_anomaly)[1:5]
if(length(anom_idx) > 0) {
  anom_vals <- eval.fd(eval_t, fd_test[anom_idx])
  matlines(eval_t, anom_vals, col="red", lty=1, lwd=1)
}
legend("topright", legend=c("Mean Walking (Normal)", "Detected Running (Anomaly)"),
       col=c("black", "red"), lwd=c(3, 1))

print("Conformal Prediction Implementation Complete.")
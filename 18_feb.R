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
# algorithms. To resolve this, we create a common, continuous time grid sampled at 10Hz, 
# which is sufficient to capture human movement without overloading memory. We then 
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

print("Segmenting Activity Phases...")

# Prioritize Linear Acc (no gravity) if available, else raw Acc Magnitude
if("Smooth_LinearAccelerometerSensor" %in% names(df_aligned) && 
   sum(abs(df_aligned$Smooth_LinearAccelerometerSensor)) > 10) {
  mag <- abs(df_aligned$Smooth_LinearAccelerometerSensor)
} else {
  mag <- sqrt(df_aligned$Smooth_AccX^2 + df_aligned$Smooth_AccY^2 + df_aligned$Smooth_AccZ^2)
}

window_size <- round(2 / dt) 
roll_var <- rollapply(mag, width = window_size, FUN = var, fill = NA, align = "right")
roll_var[is.na(roll_var)] <- 0
df_aligned$RollVar <- roll_var

set.seed(123)
log_var <- log(roll_var + 1e-6) 
kmeans_res <- kmeans(log_var, centers = 3, nstart = 20)

centers_sorted <- sort(kmeans_res$centers)
thresh1 <- exp(mean(centers_sorted[1:2]))
thresh2 <- exp(mean(centers_sorted[2:3]))

df_aligned$Activity <- cut(df_aligned$RollVar, 
                           breaks = c(-Inf, thresh1, thresh2, Inf),
                           labels = c("Sedentary", "Walking", "Running"))

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
    scale_color_manual(values = c("Sedentary" = "blue", "Walking" = "orange", "Running" = "red")) +
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
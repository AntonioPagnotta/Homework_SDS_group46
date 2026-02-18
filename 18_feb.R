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
print("Loading data...")
df <- read.csv("dati_sgravati (1).csv")

# Sort and convert time
df <- df[order(df$timestamp), ]
start_time <- min(df$timestamp)
df$time_sec <- (df$timestamp - start_time) / 1000

# Identify sensors
sensor_cols <- setdiff(names(df), c("timestamp", "time_sec"))

# ==========================================
# 2. Handle Sparsity (Interpolation)
# ==========================================
print("Aligning and Interpolating...")

# Define grid: 10Hz (dt = 0.1s)
dt <- 0.1 
t_grid <- seq(from = 0, to = max(df$time_sec), by = dt)

# Initialize aligned dataframe
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
print("Smoothing in chunks to save memory...")

# Define Chunk Size (e.g., 2000 points = 200 seconds)
chunk_size <- 2000 
n_points <- length(t_grid)
num_chunks <- ceiling(n_points / chunk_size)

# Function to smooth a specific vector chunk
smooth_chunk <- function(time_vec, data_vec) {
  # Create basis for this specific short time range
  # nbasis ~ 4 per second (adjust density as needed)
  nbasis <- max(4, round(length(time_vec) / 4))
  basis_obj <- create.bspline.basis(rangeval = range(time_vec), nbasis = nbasis, norder = 4)
  fdPar_obj <- fdPar(basis_obj, Lfdobj = 2, lambda = 0.01)
  
  # Smooth
  smooth_res <- smooth.basis(time_vec, data_vec, fdPar_obj)
  
  # Return evaluated curve (vector)
  return(eval.fd(time_vec, smooth_res$fd))
}

# Apply chunked smoothing to each sensor
for(sensor in sensor_cols) {
  print(paste("Processing sensor:", sensor))
  
  # Initialize result vector
  smoothed_vals <- numeric(n_points)
  
  for(i in 1:num_chunks) {
    # Define indices
    idx_start <- (i-1) * chunk_size + 1
    idx_end <- min(i * chunk_size, n_points)
    idx <- idx_start:idx_end
    
    # Extract chunk
    t_chunk <- df_aligned$time[idx]
    y_chunk <- df_aligned[[sensor]][idx]
    
    # Smooth chunk and store
    smoothed_vals[idx] <- smooth_chunk(t_chunk, y_chunk)
  }
  
  # Store in dataframe
  col_name <- paste0("Smooth_", sensor)
  df_aligned[[col_name]] <- smoothed_vals
}

# ==========================================
# 4. Segmentation
# ==========================================
print("Segmenting Activity Phases...")

# Calculate Magnitude from Smoothed Accelerometer
# (Using LinearAccelerometer or manual Acc calculation)
# We prioritize 'LinearAccelerometerSensor' if available and valid, else AccX/Y/Z
if("Smooth_LinearAccelerometerSensor" %in% names(df_aligned) && sum(df_aligned$Smooth_LinearAccelerometerSensor) != 0) {
  mag <- abs(df_aligned$Smooth_LinearAccelerometerSensor)
} else {
  mag <- sqrt(df_aligned$Smooth_AccX^2 + df_aligned$Smooth_AccY^2 + df_aligned$Smooth_AccZ^2)
}

# Rolling Variance
window_size <- round(2 / dt) # 2 seconds
roll_var <- rollapply(mag, width = window_size, FUN = var, fill = NA, align = "right")
roll_var[is.na(roll_var)] <- 0
df_aligned$RollVar <- roll_var

# Clustering (K-Means on Log Variance)
set.seed(123)
# Add small constant to avoid log(0)
log_var <- log(roll_var + 1e-6) 
kmeans_res <- kmeans(log_var, centers = 3, nstart = 20)

# Sort centers to align labels (Low -> Sedentary, Med -> Walking, High -> Running)
centers_sorted <- sort(kmeans_res$centers)
thresh1 <- exp(mean(centers_sorted[1:2]))
thresh2 <- exp(mean(centers_sorted[2:3]))

df_aligned$Activity <- cut(df_aligned$RollVar, 
                           breaks = c(-Inf, thresh1, thresh2, Inf),
                           labels = c("Sedentary", "Walking", "Running"))

# ==========================================
# 5. Visualization
# ==========================================
print("Plotting...")

# Plot a 5-minute snippet to verify quality
snippet <- df_aligned[1:3000, ] 
if(nrow(snippet) > 0) {
  p1 <- ggplot(snippet, aes(x = time, y = Smooth_AccX, color = Activity, group=1)) +
    geom_line() +
    labs(title = "Smoothed AccX (First 5 mins)", y = "Acceleration") +
    theme_minimal() +
    scale_color_manual(values = c("Sedentary" = "blue", "Walking" = "orange", "Running" = "red"))
  print(p1)
}

# Plot Activity Distribution
p2 <- ggplot(df_aligned, aes(x = Activity)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Distribution of Activity Phases") +
  theme_minimal()
print(p2)

print("Done! Data available in 'df_aligned'.")
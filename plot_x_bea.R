library(tidyverse)
library(lubridate)
library(gridExtra)

# 1. Load the data
df <- read_csv("dati_sgravati (1).csv")

# 2. Convert timestamp (milliseconds to datetime)
df <- df %>%
  mutate(datetime = as_datetime(timestamp / 1000))

# 3. Define Custom X-Axis Breaks
# Calculate exact start and end
start_time <- min(df$datetime)
end_time <- max(df$datetime)

# Generate regular 15-minute intervals strictly between start and end
# ceiling_date rounds up to the next 15 mins; floor_date rounds down
breaks_seq <- seq(
  from = ceiling_date(start_time, "15 mins"), 
  to = floor_date(end_time, "15 mins"), 
  by = "15 mins"
)

# Combine start, regular ticks, and end into one sorted, unique vector
final_breaks <- sort(unique(c(start_time, breaks_seq, end_time)))

# 4. Define a reusable theme/scale for the x-axis
# We use final_breaks for the ticks and rotate labels 45 degrees for readability
my_x_axis <- list(
  scale_x_datetime(breaks = final_breaks, date_labels = "%H:%M"),
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
)

# 5. Create Plots for each sensor

# Ambient Light
p1 <- df %>%
  filter(!is.na(AmbientLightSensor)) %>%
  ggplot(aes(x = datetime, y = AmbientLightSensor)) +
  geom_line(color = "orange") +
  labs(title = "Ambient Light Sensor", y = "Value", x = NULL) +
  theme_minimal() +
  my_x_axis

# Linear Accelerometer
p2 <- df %>%
  filter(!is.na(LinearAccelerometerSensor)) %>%
  ggplot(aes(x = datetime, y = LinearAccelerometerSensor)) +
  geom_line(color = "purple") +
  labs(title = "Linear Accelerometer Sensor", y = "m/s^2", x = NULL) +
  theme_minimal() +
  my_x_axis

# Accelerometer (X, Y, Z)
p3 <- df %>%
  select(datetime, AccX, AccY, AccZ) %>%
  pivot_longer(cols = c(AccX, AccY, AccZ), names_to = "Axis", values_to = "Value") %>%
  filter(!is.na(Value)) %>%
  ggplot(aes(x = datetime, y = Value, color = Axis)) +
  geom_line(alpha = 0.8) +
  labs(title = "Accelerometer (X, Y, Z)", y = "m/s^2", x = NULL) +
  theme_minimal() +
  theme(legend.position = "right") +
  my_x_axis

# Compass Sensor
p4 <- df %>%
  filter(!is.na(CompassSensor)) %>%
  ggplot(aes(x = datetime, y = CompassSensor)) +
  geom_line(color = "blue") +
  labs(title = "Compass Sensor", y = "Degrees", x = NULL) +
  theme_minimal() +
  my_x_axis

# Magnetic Rotation Sensor
p5 <- df %>%
  filter(!is.na(MagneticRotationSensor)) %>%
  ggplot(aes(x = datetime, y = MagneticRotationSensor)) +
  geom_line(color = "green") +
  labs(title = "Magnetic Rotation Sensor", y = "Value", x = "Time") +
  theme_minimal() +
  my_x_axis

# 6. Arrange all plots in one grid
grid.arrange(p1, p2, p3, p4, p5, ncol = 1)
from river import anomaly
from river import stats
from datetime import datetime
import json
from datetime import datetime, timedelta
from collections import defaultdict
import os

# Path to the logs file (adjust this path as necessary)
log_file_path = 'E_drive_operations_log.json'

# Define time periods for analysis (e.g., weekly or monthly)
time_period = timedelta(weeks=1)  # Weekly analysis
threshold_multiplier = 5  # Anomaly threshold (e.g., 5 times the average activity)

def read_logs(file_path):
    with open(file_path, 'r') as f:
        try:
            logs = json.load(f)  # Load the entire JSON array
        except json.JSONDecodeError as e:
            print(f"Error loading JSON array from file: {e}")
            logs = []  # Return an empty list if there's an error
    return logs

# Function to convert logs into features for online learning
category_map = {}
operation_map = {}
last_timestamp = {}

# Function to convert log entries into numerical features for model input
def log_to_features(log_entry):
    # Existing feature extraction...
    category = log_entry.get("category", "Unknown")
    if category not in category_map:
        category_map[category] = len(category_map)
    category_code = category_map[category]

    operation = log_entry.get("operation", "Unknown")
    if operation not in operation_map:
        operation_map[operation] = len(operation_map)
    operation_code = operation_map[operation]

    bytes_modified = log_entry.get("bytes_modified", 0)
    timestamp = datetime.fromisoformat(log_entry.get("timestamp"))

    # Calculate time difference (in seconds) from the last entry for the same file
    file_path = log_entry.get("file")
    time_diff = 0
    if file_path in last_timestamp:
        time_diff = (timestamp - last_timestamp[file_path]).total_seconds()
    last_timestamp[file_path] = timestamp  # Update last seen timestamp

    # Return feature dictionary with time difference included
    return {
        'category_code': category_code,
        'operation_code': operation_code,
        'bytes_modified': float(bytes_modified),
        'hour': timestamp.hour,
        'time_diff': time_diff  # New feature
    }
# Initialize Isolation Forest for online anomaly detection
model = anomaly.HalfSpaceTrees(seed=42, n_trees=10, height=5, window_size=50)

# Function to detect anomalies using online learning
def detect_anomalies(logs, model):
    anomalies = []

    for log_entry in logs:
        features = log_to_features(log_entry)
        #print("FEATURES:", features)
        # Score anomaly (scores closer to 1 mean high anomaly)
        anomaly_score = model.score_one(features)
        #print("ANOMALY SCORE:", anomaly_score)
        # Train model with the new data (no need to reassign `model`)
        model.learn_one(features)

        # Thresholding (use an empirical threshold or dynamically adjust based on anomaly history)
        if anomaly_score > 0.9:  # Example threshold
            anomalies.append({
                "log": log_entry,
                "anomaly_score": anomaly_score
            })

    return anomalies
# Example usage
log_file_path = '../synthetic_log_file.json'
logs = read_logs(log_file_path)
anomalies = detect_anomalies(logs, model)
print("ANOMALIES: " , anomalies)
# Print anomalies detected
for anomaly in anomalies:
    print(f"Anomaly detected with score {anomaly['anomaly_score']}: {anomaly['log']}")

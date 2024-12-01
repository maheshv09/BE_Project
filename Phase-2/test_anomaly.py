from river import anomaly
from datetime import datetime
from collections import deque
import json
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Initialize mappings and state tracking
category_map = {}
operation_map = {}
last_timestamp = {}

# Read logs from the file
def read_logs(file_path):
    with open(file_path, 'r') as f:
        try:
            logs = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error loading JSON: {e}")
            logs = []
    return logs

# Feature extraction
def log_to_features(log_entry):
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
    file_path = log_entry.get("file")

    # Calculate time difference
    time_diff = 0
    if file_path in last_timestamp:
        time_diff = (timestamp - last_timestamp[file_path]).total_seconds()
    last_timestamp[file_path] = timestamp

    path_depth = file_path.count("/") + file_path.count("\\")
    return {
        "category_code": category_code,
        "operation_code": operation_code,
        "bytes_modified": float(bytes_modified),
        "hour": timestamp.hour,
        "time_diff": time_diff,
        "path_depth": path_depth,
    }

# Prepare dataset for batch models
def prepare_batch_data(logs):
    return np.array([list(log_to_features(log).values()) for log in logs])

# Dynamic Thresholding
recent_scores = deque(maxlen=100)  # Track the last 100 anomaly scores

def dynamic_threshold(score, base_threshold=0.9):
    if len(recent_scores) < 10:  # If insufficient history, use base threshold
        return base_threshold
    
    mean_score = np.mean(recent_scores)
    std_score = np.std(recent_scores)
    
    # Dynamic threshold: Mean + 1.5 * Std Dev (tunable factor)
    dynamic_threshold_value = mean_score + 1.5 * std_score
    return max(base_threshold, dynamic_threshold_value)

# HalfSpaceTrees with dynamic thresholding
def detect_with_halfspacetrees_dynamic(logs, model):
    anomalies = []

    for log_entry in logs:
        features = log_to_features(log_entry)
        anomaly_score = model.score_one(features)
        model.learn_one(features)

        # Update recent scores deque
        recent_scores.append(anomaly_score)

        # Calculate dynamic threshold
        threshold = dynamic_threshold(anomaly_score)

        if anomaly_score > threshold:
            anomalies.append({
                "log": log_entry,
                "anomaly_score": anomaly_score,
                "threshold": threshold,
                "method": "HalfSpaceTrees (Dynamic)"
            })

    return anomalies

# One-Class SVM
def detect_with_oneclasssvm(X, model):
    predictions = model.fit_predict(X)
    return [
        {"index": i, "features": X[i], "method": "OneClassSVM"} 
        for i, pred in enumerate(predictions) if pred == -1
    ]

# Local Outlier Factor
def detect_with_lof(X, model):
    predictions = model.fit_predict(X)
    return [
        {"index": i, "features": X[i], "method": "LOF"} 
        for i, pred in enumerate(predictions) if pred == -1
    ]

# Autoencoder
def build_autoencoder(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(input_dim, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def detect_with_autoencoder(X, model):
    reconstruction = model.predict(X)
    reconstruction_error = np.mean((X - reconstruction) ** 2, axis=1)
    threshold = np.percentile(reconstruction_error, 95)  # Set 95th percentile as threshold
    return [
        {"index": i, "features": X[i], "method": "Autoencoder", "error": err} 
        for i, err in enumerate(reconstruction_error) if err > threshold
    ]

# Evaluation Metrics
def evaluate_anomalies(all_anomalies):
    counts = {}
    for anomaly in all_anomalies:
        method = anomaly.get("method", "Unknown")
        counts[method] = counts.get(method, 0) + 1
    print("Anomalies detected by each method:")
    for method, count in counts.items():
        print(f"{method}: {count}")
    return counts

# Example usage
log_file_path = '../synthetic_log_file.json'
logs = read_logs(log_file_path)

# Prepare dataset for batch models
X = prepare_batch_data(logs)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize Models
hforest = anomaly.HalfSpaceTrees(seed=42, n_trees=10, height=5, window_size=50)
ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
autoencoder = build_autoencoder(X_scaled.shape[1])

# Train Autoencoder (on normal data)
autoencoder.fit(X_scaled, X_scaled, epochs=10, batch_size=32, verbose=0)

# Run anomaly detection
anomalies_hforest_dynamic = detect_with_halfspacetrees_dynamic(logs, hforest)
anomalies_ocsvm = detect_with_oneclasssvm(X_scaled, ocsvm)
anomalies_lof = detect_with_lof(X_scaled, lof)
anomalies_autoencoder = detect_with_autoencoder(X_scaled, autoencoder)

# Consolidate anomalies
all_anomalies = anomalies_hforest_dynamic + anomalies_ocsvm + anomalies_lof + anomalies_autoencoder

# Evaluate and print results
evaluate_anomalies(all_anomalies)

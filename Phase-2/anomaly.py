import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from keras.models import Model
from keras.layers import Input, Dense
from datetime import datetime
import json

# Import River models
from river import anomaly
from river import preprocessing

def read_logs(file_path):
    with open(file_path, 'r') as f:
        try:
            logs = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error loading JSON: {e}")
            logs = []
    return logs

log_file_path = 'synthetic_log_file.json'  # Make sure you have this file
logs = read_logs(log_file_path)

def timestamp_to_features(timestamp):
    features = []
    try:
        timestamp_obj = datetime.fromisoformat(timestamp)
        
        features.append(timestamp_obj.hour)  # Hour of the day
        features.append(timestamp_obj.day)   # Day of the month
        features.append(timestamp_obj.weekday())  # Day of the week (0=Monday, 6=Sunday)
        return features
    except ValueError:
        print(f"Failed to parse timestamp: {timestamp}")
        return []  # Return empty list if timestamp format is incorrect or can't be parsed


def log_to_features(log):
    feature = []
    
    # Extract timestamp-based features
    time_features = timestamp_to_features(log['timestamp'])
    feature.extend(time_features)
    
    # Normalize operation using a simple hash
    operation_map = {"deletion": 0, "insertion": 1, "rename": 2, "update": 3}
    feature.append(operation_map.get(log['operation'], -1))  # Default -1 for unknown operations

    # Hashing category to normalize
    feature.append(hash(log['category']) % 1000)  # Use modulo for normalization

    # Add 'bytes_modified' if present
    feature.append(log.get('bytes_modified', 0))  # Default to 0 if not present
    
    # Add 'new_name' for rename operations
    feature.append(hash(log.get('new_name', '')) % 1000)  # Default to 0 if not present
    
    return feature

def extract_features_from_log(logs):
    features = []
    labels = []
    for log in logs:
        feature = log_to_features(log)
        features.append(feature)
        labels.append(log['operation']) 
    return np.array(features), np.array(labels)

def create_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder

# Train Autoencoder
def train_autoencoder(data):
    autoencoder = create_autoencoder(data.shape[1])
    autoencoder.fit(data, data, epochs=50, batch_size=64, validation_split=0.2, verbose=0)
    return autoencoder


recent_scores = []  # Store recent anomaly scores

def dynamic_threshold(score, base_threshold=0.9, window_size=100):
    # Maintain recent scores within the window size limit
    recent_scores.append(score)
    if len(recent_scores) > window_size:
        recent_scores.pop(0)

    if len(recent_scores) < 10:  # If not enough history, use the base threshold
        return base_threshold

    mean_score = np.mean(recent_scores)
    std_score = np.std(recent_scores)

    # Dynamic threshold: Mean + 1.5 * Std Dev
    dynamic_threshold_value = mean_score + 1.5 * std_score
    return max(base_threshold, dynamic_threshold_value)


# Function to detect anomalies with HalfSpaceTrees and dynamic thresholding
def detect_anomalies_with_halfspacetree_dynamic(data_scaled):
    river_model = anomaly.HalfSpaceTrees()  # River model for anomaly detection
    data_river = [{f'feature_{i}': value for i, value in enumerate(row)} for row in data_scaled]  # Convert to River format

    anomalies = []
    
    # Incrementally learn and predict anomalies
    for i, x in enumerate(data_river):
        score = river_model.score_one(x)
        river_model.learn_one(x)

        # Apply dynamic thresholding
        threshold = dynamic_threshold(score)

        # If the anomaly score exceeds the threshold, classify it as an anomaly
        if score > threshold:
            anomalies.append({
                "index": i,
                "features": x,
                "anomaly_score": score,
                "threshold": threshold,
                "method": "HalfSpaceTrees (Dynamic)"
            })

    return anomalies

def detect_anomalies_with_scores(data):
    # Ensure data is a 2D array
    data = np.array(data)
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)  # Reshape if it's a 1D array

    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 1. Isolation Forest
    isolation_forest = IsolationForest(contamination=0.1, random_state=42)
    isolation_forest.fit(data_scaled)
    score_iforest = isolation_forest.decision_function(data_scaled)  # Higher is normal
    predictions_iforest = isolation_forest.predict(data_scaled)  # 1 = normal, -1 = anomaly
    count_iforest = np.sum(predictions_iforest == -1)

    # 2. One-Class SVM
    ocsvm = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    ocsvm.fit(data_scaled)
    score_ocsvm = ocsvm.decision_function(data_scaled)  # Higher is normal
    predictions_ocsvm = ocsvm.predict(data_scaled)  # 1 = normal, -1 = anomaly
    count_ocsvm = np.sum(predictions_ocsvm == -1)

    # 3. LOF
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    predictions_lof = lof.fit_predict(data_scaled)  # 1 = normal, -1 = anomaly
    score_lof = -lof.negative_outlier_factor_  # LOF uses negative scores, convert to positive
    count_lof = np.sum(predictions_lof == -1)

    # 4. Autoencoder (Assumed function 'train_autoencoder')
    autoencoder = train_autoencoder(data_scaled[:-1])  # Train on normal data (use all except last row)
    reconstruction_error = np.mean((data_scaled - autoencoder.predict(data_scaled))**2, axis=1)
    predictions_autoencoder = reconstruction_error > np.percentile(reconstruction_error, 90)  # Top 10% as anomalies
    count_autoencoder = np.sum(predictions_autoencoder)

     # 5. River Model (HalfSpaceTrees with dynamic thresholding)
    anomalies_river = detect_anomalies_with_halfspacetree_dynamic(data_scaled)
    count_river = len(anomalies_river)
   
    anomaly_counts = {
        "IsolationForest": count_iforest,
        "OneClassSVM": count_ocsvm,
        "LOF": count_lof,
        "Autoencoder": count_autoencoder,
        "RiverModel": count_river,  # Include River anomaly detection
    }

    # Store scores and predictions
    scores = {
        "IsolationForest": score_iforest,
        "OneClassSVM": score_ocsvm,
        "LOF": score_lof,
        "Autoencoder": reconstruction_error,
         "RiverModel": [anomaly['anomaly_score'] for anomaly in anomalies_river],  # Include River scores
    }
    predictions = {
        "IsolationForest": predictions_iforest,
        "OneClassSVM": predictions_ocsvm,
        "LOF": predictions_lof,
        "Autoencoder": predictions_autoencoder,
         "RiverModel": [1 if anomaly['anomaly_score'] > anomaly['threshold'] else 0 for anomaly in anomalies_river],   # Include River predictions
    }

    return scores, predictions, anomaly_counts



import numpy as np

def combine_anomaly_scores(scores, weights, threshold=0.5):
    """
    Combines anomaly scores using weighted average and classifies based on threshold.
    Ensures that the scores from each model have the same length.
    """
    max_length = max(len(scores[model]) for model in scores)

    aggregated_score = np.zeros(max_length, dtype=float)

    for model, weight in weights.items():
        model_scores = np.array(scores[model], dtype=float)

        if len(model_scores) < max_length:
            padding = np.zeros(max_length - len(model_scores))
            model_scores = np.concatenate([model_scores, padding])

        aggregated_score += model_scores * weight

    # Normalize aggregated score
    aggregated_score = (aggregated_score - np.min(aggregated_score)) / (np.max(aggregated_score) - np.min(aggregated_score))

    # Classify based on threshold
    return aggregated_score > threshold, aggregated_score


features, labels = extract_features_from_log(logs)
scores, predictions, anomaly_counts = detect_anomalies_with_scores(features)

weights = {
    "IsolationForest": 0.35,  # Performs well on datasets with independent features
    "OneClassSVM": 0.25,     # Handles complex, non-linear relationships
    "LOF": 0.15,             # Good for local density-based anomaly detection
    "Autoencoder": 0.1,      # Useful for capturing global patterns, but may overfit
    "RiverModel": 0.1,      # Lightweight, incremental learning model
}

combined_predictions, aggregated_scores = combine_anomaly_scores(scores, weights)


print("Anomaly counts by each model:", anomaly_counts)

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
def evaluate_model_predictions(predictions, logs, scores):
    metrics = {}

    for model, pred in predictions.items():
        # Use the model's predictions against its thresholds
        actual = pred  # Directly use the predictions as the evaluation target
        metrics[model] = {
            "Precision": precision_score(actual, pred),
            "Recall": recall_score(actual, pred),
            "F1-Score": f1_score(actual, pred),
            "Accuracy": accuracy_score(actual, pred),
        }

    return metrics

metrics = evaluate_model_predictions(predictions, logs, scores)

    # Print results
for model, metric_values in metrics.items():
    print(f"Metrics for {model}:")
    for metric, value in metric_values.items():
        print(f"  {metric}: {value:.2f}")
    print()

def filter_logs_by_operation(logs, operation_type):
    """
    Filters logs based on the given operation type.
    """
    return [log for log in logs if log['operation'] == operation_type]


def save_anomalies(logs, predictions, file_path="anomalies.json"):
    """
    Saves all anomalies across models to a single JSON file.
    """
    anomalies = []
    for model, pred in predictions.items():
        # Collect logs marked as anomalies for this model
        model_anomalies = [
            {"model": model, "log": log} for log, pred_value in zip(logs, pred) if pred_value == -1
        ]
        anomalies.extend(model_anomalies)

    with open(file_path, 'w') as f:
        json.dump(anomalies, f, indent=4)
    print(f"Saved {len(anomalies)} anomalies to {file_path}")

# Save anomalies for all models to one file
save_anomalies(logs, predictions, "anomalies.json")


# ---------------------------------------------------------------------

# ANOMALY DETECTION FOR EACH OPERATION 

def filter_logs_by_operation(logs, operation_type):
    """
    Filters logs for the given operation type.
    """
    return [log for log in logs if log['operation'] == operation_type]

import json

operation_types = ["deletion", "insertion", "rename", "update"]
anomalies_by_operation = {}

for operation in operation_types:
    print(f"\nProcessing operation: {operation}")

    # Filter logs for the current operation
    filtered_logs = filter_logs_by_operation(logs, operation)

    if not filtered_logs:
        print(f"No logs found for operation: {operation}")
        # Include the operation with zero anomalies and empty logs
        anomalies_by_operation[operation] = {
            "anomaly_count": 0,
            "logs": [],
        }
        continue

    # Extract features and labels
    features, labels = extract_features_from_log(filtered_logs)

    # Detect anomalies
    scores, o_predictions, anomaly_count = detect_anomalies_with_scores(features)

    # Debug: Check anomaly_count structure
    print(f"Debug - anomaly_count for {operation}: {anomaly_count}")

    # Save only anomaly count and logs for the current operation
    anomalies_by_operation[operation] = {
        "anomaly_count": anomaly_count,
        "logs": filtered_logs,
    }

    print(f"Anomalies detected for {operation}: {anomaly_count}")

# Save anomalies_by_operation to a JSON file
output_file = "anomalies_by_operation.json"

# Convert all NumPy types to native Python types
def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

with open(output_file, 'w') as f:
    json.dump(anomalies_by_operation, f, indent=4, default=convert_to_serializable)

print(f"Anomalies by operation saved to {output_file}")


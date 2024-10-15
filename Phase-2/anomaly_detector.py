import json
from datetime import datetime, timedelta
from collections import defaultdict
import os

# Path to the logs file (adjust this path as necessary)
log_file_path = 'F:/mahesh/BE Project/Phase-2/E_drive_operations_log.json'

# Define time periods for analysis (e.g., weekly or monthly)
time_period = timedelta(weeks=1)  # Weekly analysis
threshold_multiplier = 5  # Anomaly threshold (e.g., 5 times the average activity)

def read_logs(file_path):
    logs = []
    with open(file_path, 'r') as f:
        data = f.read()
        # Split the data by '}{' and then handle each as an individual JSON entry
        log_entries = data.split('}{')
        
        for i, entry in enumerate(log_entries):
            # Ensure each log is a complete JSON object by adding back '{' and '}'
            if i == 0:
                entry = entry + '}'
            elif i == len(log_entries) - 1:
                entry = '{' + entry
            else:
                entry = '{' + entry + '}'

            # Convert each entry to a JSON object and append to logs
            logs.append(json.loads(entry))

    return logs


# Function to parse logs and categorize them by operation type and category over time
def parse_logs(logs):
    # Store frequency of operations by category and date
    operation_frequencies = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for log_entry in logs:
        file_category = log_entry.get("category", "Unknown")
        operation = log_entry.get("operation", "Unknown")
        timestamp = log_entry.get("timestamp", None)

        # Convert timestamp to datetime object
        if timestamp:
            log_time = datetime.fromisoformat(timestamp)
            # Group by day for now (you can adjust this to week, month, etc.)
            log_date = log_time.date()
            operation_frequencies[file_category][operation][log_date] += 1

    return operation_frequencies

# Function to calculate average frequencies and detect anomalies
def detect_anomalies(operation_frequencies, threshold_multiplier=5):
    anomalies = []

    for category, operations in operation_frequencies.items():
        for operation, daily_frequencies in operations.items():
            # Calculate the average frequency over time
            total_operations = sum(daily_frequencies.values())
            days_tracked = len(daily_frequencies)
            if days_tracked == 0:
                continue
            avg_frequency = total_operations / days_tracked

            # Check for anomalies by comparing daily frequencies to the average
            for date, frequency in daily_frequencies.items():
                if frequency > threshold_multiplier * avg_frequency:
                    anomalies.append({
                        "category": category,
                        "operation": operation,
                        "date": str(date),
                        "frequency": frequency,
                        "average_frequency": avg_frequency,
                        "anomaly_factor": frequency / avg_frequency
                    })

    return anomalies

# Function to print detected anomalies
def print_anomalies(anomalies):
    if not anomalies:
        print("No anomalies detected.")
        return
    
    print("Anomalies Detected:")
    for anomaly in anomalies:
        print(f"Category: {anomaly['category']}, Operation: {anomaly['operation']}")
        print(f"Date: {anomaly['date']}, Frequency: {anomaly['frequency']}")
        print(f"Average Frequency: {anomaly['average_frequency']:.2f}")
        print(f"Anomaly Factor: {anomaly['anomaly_factor']:.2f}x")
        print("-" * 40)

# Main function to run anomaly detection
def run_anomaly_detection(log_file_path, threshold_multiplier=5):
    # Read the logs from file
    logs = read_logs(log_file_path)

    # Parse the logs to get operation frequencies
    operation_frequencies = parse_logs(logs)

    # Detect anomalies
    anomalies = detect_anomalies(operation_frequencies, threshold_multiplier)

    # Print the results
    print_anomalies(anomalies)

# Run the anomaly detection on the provided log file
if __name__ == "__main__":
    run_anomaly_detection(log_file_path, threshold_multiplier)

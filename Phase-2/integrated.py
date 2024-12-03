import os
import time
import json
import datetime
import pickle
import magic
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from keras.models import Model
from keras.layers import Input, Dense
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ------------------- Categorizer Code ------------------- #

def load_models():
    with open('decision_tree_model.pkl', 'rb') as file:
        decision_tree = pickle.load(file)

    with open('le_extension.pkl', 'rb') as ext_file:
        le_ext = pickle.load(ext_file)

    with open('le_mime.pkl', 'rb') as mime_file:
        le_mime = pickle.load(mime_file)

    with open('le_category.pkl', 'rb') as category_file:
        le_category = pickle.load(category_file)

    return decision_tree, le_ext, le_mime, le_category

def predict_category(decision_tree, le_ext, le_mime, le_category, extension, mime_type):
    try:
        ext_encoded = le_ext.transform([extension])
        mime_encoded = le_mime.transform([mime_type])
        new_data = [[ext_encoded[0], mime_encoded[0]]]
        prediction = decision_tree.predict(new_data)
        category_predicted = le_category.inverse_transform(prediction)
        return category_predicted[0]
    except Exception as e:
        return f"Prediction error: {e}"

class Categorizer:
    def __init__(self):
        self.decision_tree, self.le_ext, self.le_mime, self.le_category = load_models()

    def categorize(self, file_path):
        extension = os.path.splitext(file_path)[1]
        mime_type = magic.Magic(mime=True).from_file(file_path)
        if mime_type == 'inode/x-empty':
            return "Empty File"
        return predict_category(self.decision_tree, self.le_ext, self.le_mime, self.le_category, extension, mime_type)

# ------------------- Observer Code ------------------- #

class FileHandler(FileSystemEventHandler):
    def __init__(self, categorizer):
        super().__init__()
        self.categorizer = categorizer
        self.category_cache = {}
        self.file_size_cache = {}
        self.logs = []

    def process_event(self, event_type, file_path, bytes_modified=None):
        if "$RECYCLE.BIN" in file_path:
            return

        log_entry = {
            "file": file_path,
            "operation": event_type,
            "timestamp": str(datetime.datetime.now())
        }

        if event_type in ["insertion", "update"]:
            if os.path.isfile(file_path):
                file_category = self.categorizer.categorize(file_path)
                self.category_cache[file_path] = file_category
                log_entry["category"] = file_category
                if event_type == "update" and bytes_modified is not None:
                    log_entry["bytes_modified"] = bytes_modified
                self.file_size_cache[file_path] = os.path.getsize(file_path)

        elif event_type == "deletion":
            file_category = self.category_cache.get(file_path, "Unknown")
            log_entry["category"] = file_category
            self.category_cache.pop(file_path, None)
            self.file_size_cache.pop(file_path, None)

        elif event_type == "rename":
            log_entry["new_name"] = file_path
            log_entry["category"] = self.category_cache.get(file_path, "Unknown")

        self.logs.append(log_entry)
        self.log_to_file(log_entry)

    def log_to_file(self, log_entry):
        log_file = get_log_file_for_drive(log_entry["file"])
        with open(log_file, "a") as log:
            log.write(json.dumps(log_entry) + "\n")

    def on_created(self, event):
        if not event.is_directory:
            self.process_event("insertion", event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            self.process_event("deletion", event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            old_size = self.file_size_cache.get(event.src_path, 0)
            new_size = os.path.getsize(event.src_path)
            bytes_modified = new_size - old_size
            self.process_event("update", event.src_path, bytes_modified)

    def on_moved(self, event):
        if not event.is_directory:
            self.process_event("rename", event.dest_path)

def get_log_file_for_drive(file_path):
    drive = os.path.splitdrive(file_path)[0]
    return f"{drive.replace(':', '')}_drive_operations_log.json"

# ------------------- Anomaly Detection Code ------------------- #

def extract_features_from_logs(logs):
    features = []
    for log in logs:
        feature = log_to_features(log)
        features.append(feature)
    return np.array(features)

def log_to_features(log):
    features = timestamp_to_features(log['timestamp'])
    operation_map = {"deletion": 0, "insertion": 1, "rename": 2, "update": 3}
    features.append(operation_map.get(log['operation'], -1))
    features.append(hash(log['category']) % 1000)
    features.append(log.get('bytes_modified', 0))
    features.append(hash(log.get('new_name', '')) % 1000 if 'new_name' in log else 0)
    return features

def timestamp_to_features(timestamp):
    try:
        timestamp_obj = datetime.datetime.fromisoformat(timestamp)
        return [timestamp_obj.hour, timestamp_obj.day, timestamp_obj.weekday()]
    except ValueError:
        return []

def train_hybrid_model(features):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(features)

    isolation_forest = IsolationForest(random_state=42)
    isolation_forest.fit(data_scaled)

    lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
    lof.fit(data_scaled)

    autoencoder = build_autoencoder(data_scaled.shape[1])
    autoencoder.fit(data_scaled, data_scaled, epochs=50, batch_size=16, verbose=0)

    one_class_svm = OneClassSVM(kernel='rbf', gamma='auto').fit(data_scaled)

    return scaler, isolation_forest, lof, autoencoder, one_class_svm

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(16, activation='relu')(input_layer)
    encoded = Dense(8, activation='relu')(encoded)
    decoded = Dense(16, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def detect_anomalies(logs, scaler, models):
    features = extract_features_from_logs(logs)
    scaled_features = scaler.transform(features)
    anomaly_logs = []

    for feature, log in zip(scaled_features, logs):
        isolation_score = models[0].predict([feature])[0]
        lof_score = models[1].predict([feature])[0]
        autoencoder_score = models[2].predict([feature])
        one_class_score = models[3].predict([feature])[0]

        if any(score == -1 for score in [isolation_score, lof_score, one_class_score]):
            anomaly_logs.append(log)
    return anomaly_logs

def log_anomalies_to_file(anomalies):
    with open('anomalies.json', 'a') as f:
        for anomaly in anomalies:
            f.write(json.dumps(anomaly) + "\n")

# ------------------- Main Execution ------------------- #

if __name__ == "__main__":
    categorizer = Categorizer()
    path_to_watch = "D:/"
    event_handler = FileHandler(categorizer)
    observer = Observer()
    observer.schedule(event_handler, path=path_to_watch, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(10)  # Periodic anomaly detection
            logs = event_handler.logs
            features = extract_features_from_logs(logs)
            if len(features) > 0:
                scaler, *models = train_hybrid_model(features)
                anomalies = detect_anomalies(logs, scaler, models)
                log_anomalies_to_file(anomalies)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()

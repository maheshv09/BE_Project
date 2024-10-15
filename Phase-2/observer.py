import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import json
import datetime
import pickle
import magic

# Function to create a log file name based on the drive
def get_log_file_for_drive(file_path):
    drive = os.path.splitdrive(file_path)[0]
    log_file_name = f"{drive.replace(':', '')}_drive_operations_log.json"
    return log_file_name

category_cache = {}  # Cache to store file categories on creation/update
file_size_cache = {}  # Cache to store file sizes before modification
last_logged_event = {}  # Cache to store the last modification event

# ---- CATEGORIZER CODE START ---- #
# Load the model and label encoders
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

# Predict file category
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

# Class for the categorizer that encapsulates model logic
class Categorizer:
    def __init__(self):
        self.decision_tree, self.le_ext, self.le_mime, self.le_category = load_models()

    def categorize(self, file_path):
        extension = os.path.splitext(file_path)[1]
        mime_type = magic.Magic(mime=True).from_file(file_path)

        # Handle empty files with the MIME type 'inode/x-empty'
        if mime_type == 'inode/x-empty':
            return "Empty File"
        
        return predict_category(self.decision_tree, self.le_ext, self.le_mime, self.le_category, extension, mime_type)

# ---- CATEGORIZER CODE END ---- #

# ---- DIRECTORY MONITORING CODE ---- #
class FileHandler(FileSystemEventHandler):
    def __init__(self, categorizer):
        super().__init__()
        self.categorizer = categorizer

    def process_event(self, event_type, file_path, bytes_modified=None):
        # Skip logging for files in the Recycle Bin
        if "$RECYCLE.BIN" in file_path:
            print(f"Skipping logging for Recycle Bin file: {file_path}")
            return

        log_entry = {
            "file": file_path,
            "operation": event_type,
            "timestamp": str(datetime.datetime.now())
        }

        if event_type in ["insertion", "update"]:
            if os.path.isfile(file_path):
                file_category = self.categorizer.categorize(file_path)
                category_cache[file_path] = file_category
                log_entry["category"] = file_category

                if event_type == "update" and bytes_modified is not None:
                    log_entry["bytes_modified"] = bytes_modified

                file_size_cache[file_path] = os.path.getsize(file_path)

        elif event_type == "deletion":
            # Retrieve the file category from cache on deletion
            file_category = category_cache.get(file_path, "Unknown")
            log_entry["category"] = file_category
            
            if file_path in category_cache:
                del category_cache[file_path]
            if file_path in file_size_cache:
                del file_size_cache[file_path]

        elif event_type == "rename":
            # Handle file renaming events
            log_entry["new_name"] = file_path
            log_entry["category"] = category_cache.get(file_path, "Unknown")

        # Determine the log file for the drive the file belongs to
        log_file = get_log_file_for_drive(file_path)

        # Append log entry to the drive-specific log file
        with open(log_file, "a") as log:
            log.write(json.dumps(log_entry) + "\n")

        print(f"Logged: {log_entry} to {log_file}")  # Print to console for monitoring


    def on_created(self, event):
        if not event.is_directory:
            file_size_cache[event.src_path] = os.path.getsize(event.src_path)
            self.process_event("insertion", event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            try:
                # Introduce a delay to allow time for the file write to complete
                time.sleep(0.0001)

                # Get the current file size after modification
                current_size = os.path.getsize(event.src_path)

                # Retrieve previous file size from cache; default to current_size if not found
                previous_size = file_size_cache.get(event.src_path)

                # If previous size is None (first modification), initialize it
                if previous_size is None:
                    previous_size = current_size
                    file_size_cache[event.src_path] = previous_size  # Cache the initial size

                # Calculate bytes modified
                bytes_modified = current_size - previous_size

                # Debug output for clarity
                print(f"CURR: {current_size} / Previous Size: {previous_size} / Bytes Modified: {bytes_modified}")

                # If no bytes were modified, skip logging
                if bytes_modified == 0:
                    print(f"Skipping logging for {event.src_path}: 0 bytes modified (possible metadata change).")
                    return

                # Log the update event with bytes modified
                self.process_event("update", event.src_path, bytes_modified=bytes_modified)

                # Update the cache with the new size after logging the change
                file_size_cache[event.src_path] = current_size

            except FileNotFoundError:
                print(f"File not found: {event.src_path}, likely deleted.")

    # Method to handle deletions
    def on_deleted(self, event):
        if not event.is_directory:
            self.process_event("deletion", event.src_path)

    # Method to handle renaming of files
    def on_moved(self, event):
        if not event.is_directory:
            print(f"File moved from {event.src_path} to {event.dest_path}")
            self.process_event("rename", event.dest_path)


def monitor_directory(directory_to_monitor, categorizer):
    event_handler = FileHandler(categorizer)
    observer = Observer()
    observer.schedule(event_handler, directory_to_monitor, recursive=True)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# ---- START MONITORING ---- #
if __name__ == '__main__':
    categorizer = Categorizer()

    # Example: Monitor the F:/ directory or change this to any path
    directory_to_monitor = 'E:/'
    monitor_directory(directory_to_monitor, categorizer)

import json
import random
from datetime import datetime, timedelta

# Define possible categories, operations, and other fields
categories = ["Email related", "Document/Letter", "Program executable", "Media File", "System Log", "Backup", "Configuration File", "Empty File", "Unknown"]
operations = ["insertion", "update", "deletion", "rename"]
base_timestamp = datetime(2024, 10, 15, 10, 0, 0)

# Generate synthetic log entries
def generate_log_entries(num_entries):
    log_entries = []
    for i in range(num_entries):
        # Randomly select category and operation
        category = random.choice(categories)
        operation = random.choice(operations)

        # Generate a random timestamp by adding random minutes/hours to the base timestamp
        random_time = base_timestamp + timedelta(
            hours=random.randint(0, 48),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )

        # Randomize bytes_modified for 'update' operations and set to 0 for others
        if operation == "update":
            bytes_modified = random.randint(-5000, 5000)
        else:
            bytes_modified = 0

        # Construct the log entry
        log_entry = {
            "file": f"E:/file_{i}.txt",
            "operation": operation,
            "timestamp": random_time.isoformat(),
            "category": category
        }
        
        # Add bytes_modified only for updates
        if operation == "update":
            log_entry["bytes_modified"] = bytes_modified

        # For renaming, add a new_name field
        if operation == "rename":
            log_entry["new_name"] = f"E:/file_renamed_{i}.txt"

        log_entries.append(log_entry)
    
    return log_entries

# Generate log entries as an array of objects and save them to a file
log_entries = generate_log_entries(10000)
log_file_path = 'synthetic_log_file.json'

with open(log_file_path, 'w') as log_file:
    json.dump(log_entries, log_file, indent=4)  # Save entire list as a JSON array

print(f"Generated {len(log_entries)} log entries and saved to {log_file_path}")

import json
import csv
import requests

url = "https://cdn.jsdelivr.net/gh/jshttp/mime-db@master/db.json"
response = requests.get(url)
data = response.json()

mime_extensions = {}

for file_type, attributes in data.items():
    if 'extensions' in attributes:
        if file_type not in mime_extensions:
            mime_extensions[file_type] = []
        mime_extensions[file_type].extend(attributes['extensions'])

with open('mime_extensions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['File Type', 'Extensions'])

    for file_type, extensions in mime_extensions.items():
        writer.writerow([file_type, extensions])

from flask import Flask, request, jsonify
import os
import pickle
import magic
from flask_cors import CORS
import pandas as pd
import sqlite3

app = Flask(__name__)
CORS(app)

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


def initialize_cache():
    conn = sqlite3.connect('file_cache.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS cache (
                        extension TEXT PRIMARY KEY,
                        mime_type TEXT,
                        category TEXT)''')
    conn.commit()
    return conn

def add_to_cache(conn, extension, mime_type, category):
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM cache')
    cache_size = cursor.fetchone()[0]
    
    # Add new entry to cache
    cursor.execute('INSERT OR REPLACE INTO cache (extension, mime_type, category) VALUES (?, ?, ?)', 
                   (extension, mime_type, category))
    conn.commit()

    # Check if the cache size has reached the limit (1000 entries)
    if cache_size >= 1000:
        print("Cache limit reached, merging with original dataset.")
        merge_cache_with_dataset(conn)

global df
df = pd.read_pickle('merged_file.pkl')

def merge_cache_with_dataset(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cache")
    cached_data = cursor.fetchall()

    # Convert cached data to DataFrame
    cached_df = pd.DataFrame(cached_data, columns=['Extension', 'MIME type', 'Category'])

    # Append new data to the original CSV file
  # Load your original dataset
    df = df.append(cached_df, ignore_index=True)
    df.to_csv('merged_file.csv', index=False)

    # Clear the cache
    cursor.execute('DELETE FROM cache')
    conn.commit()
    print("Cache merged with dataset and cleared.")

    # Update encoders and retrain model with new data
    retrain_model()

# Function to retrain the model after merging cache data
def retrain_model():
    global le_ext, le_mime, le_category, decision_tree, X, y
    
    le_ext.fit(df['Extension'])
    df['Extension_encoded'] = le_ext.transform(df['Extension'])

    le_mime.fit(df['MIME type'])
    df['MIME_encoded'] = le_mime.transform(df['MIME type'])

    le_category.fit(df['Category'])
    df['Category_encoded'] = le_category.transform(df['Category'])

    X = df[['Extension_encoded', 'MIME_encoded']]
    y = df['Category_encoded']

    decision_tree.fit(X, y)
    print("Model retrained with updated dataset.")


# Predict file category
def predict_category(decision_tree, le_ext, le_mime, le_category, extension, mime_type, conn=None):
    try:
        ext_encoded = le_ext.transform([extension])
        mime_encoded = le_mime.transform([mime_type])
        new_data = [[ext_encoded[0], mime_encoded[0]]]
        prediction = decision_tree.predict(new_data)
        category_predicted = le_category.inverse_transform(prediction)
        
        print(f"Predicted Category for {extension}: {category_predicted[0]}")
        return category_predicted[0]

    except ValueError as e:
        # Check if the MIME type indicates an unknown extension

        if mime_type == 'application/x-empty':
            print(f"MIME type '{mime_type}' detected. Categorizing as 'Empty File'.")
            return "Empty File"
        
        if mime_type == 'application/octet-stream':
            print(f"Invalid file extension '{extension}'. MIME type is '{mime_type}'.")
            return jsonify({"message": f"Invalid file extension '{extension}'."}), 400
        else:
            print("Mime: ", mime_type)
            # print("Mime_type: ", df['MIME type'].values)
            # print(mime_type in df['MIME type'].values)
            if mime_type in df['MIME type'].values:
                associated_category = df.loc[df['MIME type'] == mime_type, 'Category'].values[0]
                print("Associated: ", associated_category)
                print(f"MIME type '{mime_type}' found in dataset with associated category '{associated_category}'.")
                add_to_cache(conn, extension, mime_type, associated_category)
                return associated_category  # Return this message
            else:
                # MIME type not in dataset, check cache
                cursor = conn.cursor()
                cursor.execute("SELECT category FROM cache WHERE extension = ? AND mime_type = ?", (extension, mime_type))
                cached_result = cursor.fetchone()

                if cached_result:
                    cached_category = cached_result[0]
                    print(f"MIME type '{mime_type}' found in cache with associated category '{cached_category}'.")
                    return cached_category

                else:
                    # MIME type not in dataset or cache, determine using logic
                    mime_to_category_mapping = df[['MIME type', 'Category']].drop_duplicates().set_index('MIME type')['Category'].to_dict()
                    determined_category = mime_to_category_mapping.get(mime_type, 'Unknown')
                    print("Determined: ", determined_category)
                    
                    # If the determined category is unknown, log it as such
                    if determined_category == 'Unknown':
                        print(f"MIME type '{mime_type}' not found in dataset or cache. Setting category to 'Unknown'.")
                        return "Unknown"
                    else:
                        # If a category is determined, cache it for future use
                        print(f"MIME type '{mime_type}' not found in dataset. Determined category is '{determined_category}'.")
                        add_to_cache(conn, extension, mime_type, determined_category)
                        return determined_category


@app.route('/classify-dir', methods=['POST'])
def classify_directory():
    try:
        # Load models
        decision_tree, le_ext, le_mime, le_category = load_models()
        
        # Check if files are present in the request
        if 'files' not in request.files:
            return jsonify({"error": "No files uploaded"}), 400

        files = request.files.getlist('files')
        classified_files = {}

        # Process each file
        for file in files:
            file_path = file.filename
            extension = os.path.splitext(file_path)[1]  # Extract the file extension
            
            # Get the MIME type of the file
            mime_type = magic.Magic(mime=True).from_buffer(file.read(1024))  # Read only the first 1KB
            
            # Use the classifier to predict the category
            conn = initialize_cache()
            predicted_category = predict_category(decision_tree, le_ext, le_mime, le_category, extension, mime_type, conn=conn)
            
            # Store the result by category
            print("Predicted: ", predicted_category)
            if predicted_category not in classified_files:
                classified_files[predicted_category] = []
            
            classified_files[predicted_category].append({
                "file": file_path,
                "mime_type": mime_type,
            })

        return jsonify(classified_files)

    except Exception as e:
        return jsonify({"API Error": str(e)}), 500
    

def display_cache(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cache")
    cached_data = cursor.fetchall()

    # Check if the cache is empty
    if not cached_data:
        print("Cache is empty.")
        return jsonify({"message": "Cache is empty"}), 200

    # Format and display cache data
    cache_list = []
    for row in cached_data:
        cache_entry = {
            "extension": row[0],
            "mime_type": row[1],
            "category": row[2]
        }
        cache_list.append(cache_entry)

    return jsonify({"cache_data": cache_list}), 200


@app.route('/display-cache', methods=['GET'])
def get_cache_data():
    try:
        conn = initialize_cache()
        return display_cache(conn)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    
@app.route('/classify-file', methods=['POST'])
def classify_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files['file']
        extension = os.path.splitext(file.filename)[1]
        mime_type = magic.Magic(mime=True).from_buffer(file.read(1024))  # Read the first 1KB of the file to get MIME

        decision_tree, le_ext, le_mime, le_category = load_models()

        predicted_category = predict_category(decision_tree, le_ext, le_mime, le_category, extension, mime_type)

        return jsonify({
            "file": file.filename,
            "extension": extension,
            "mime_type": mime_type,
            "predicted_category": predicted_category 
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
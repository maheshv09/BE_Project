from flask import Flask, request, jsonify
import os
import pickle
import magic
from flask_cors import CORS

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

# Predict file category
def predict_category(decision_tree, le_ext, le_mime, le_category, extension, mime_type):
    try:
        # Transform the extension and MIME type to their encoded values
        ext_encoded = le_ext.transform([extension])
        mime_encoded = le_mime.transform([mime_type])
        
        # Prepare data for prediction
        new_data = [[ext_encoded[0], mime_encoded[0]]]
        
        # Make prediction
        prediction = decision_tree.predict(new_data)
        category_predicted = le_category.inverse_transform(prediction)
        return category_predicted[0]
    
    except Exception as e:
        return f"Prediction error: {e}"

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
            predicted_category = predict_category(decision_tree, le_ext, le_mime, le_category, extension, mime_type)
            
            # Store the result by category
            if predicted_category not in classified_files:
                classified_files[predicted_category] = []
            
            classified_files[predicted_category].append({
                "file": file_path,
                "mime_type": mime_type,
            })

        return jsonify(classified_files)

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

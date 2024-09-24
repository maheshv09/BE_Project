import os
import magic
import pickle
import typer
from rich.console import Console
from rich.table import Table

# Initialize Typer and Rich console
app = typer.Typer()
console = Console()

# Load your trained models and encoders
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

def get_mime_type(file_path):
    mime = magic.Magic(mime=True)
    mime_type = mime.from_file(file_path)
    return mime_type

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

# CLI command to classify a single file
@app.command()
def classify(file: str):
    """Classify the file type based on its extension and MIME type."""
    decision_tree, le_ext, le_mime, le_category = load_models()
    extension = os.path.splitext(file)[1]
    mime_type = get_mime_type(file)
    
    predicted_category = predict_category(decision_tree, le_ext, le_mime, le_category, extension, mime_type)

    # Display the result in a table format
    table = Table(title="File Classification Result")
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Extension", style="magenta")
    table.add_column("MIME Type", style="green")
    table.add_column("Predicted Category", style="yellow")

    table.add_row(file, extension, mime_type, predicted_category)
    console.print(table)

# CLI command to classify all files in a directory
@app.command(name="classify-dir")
def classify_directory(directory: str):
    """Classify all files in the given directory based on their extension and MIME type."""
    decision_tree, le_ext, le_mime, le_category = load_models()
    
    if os.path.isdir(directory):
        table = Table(title="Directory Classification Results")
        table.add_column("File", style="cyan", no_wrap=True)
        table.add_column("Extension", style="magenta")
        table.add_column("MIME Type", style="green")
        table.add_column("Predicted Category", style="yellow")

        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                extension = os.path.splitext(file_path)[1]
                mime_type = get_mime_type(file_path)
                predicted_category = predict_category(decision_tree, le_ext, le_mime, le_category, extension, mime_type)
                
                table.add_row(file_path, extension, mime_type, predicted_category)

        console.print(table)
    else:
        console.print(f"[red]'{directory}' is not a valid directory. Please provide a valid directory path.[/red]")

if __name__ == "__main__":
    app()

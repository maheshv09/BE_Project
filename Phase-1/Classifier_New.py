import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import mimetypes
import magic
import pickle

# Load the data
df = pd.read_csv('F:\mahesh\BE Project\Phase-1\merged_fileNew.csv')

# Encode categorical features
le_ext = LabelEncoder()
df['Extension_encoded'] = le_ext.fit_transform(df['Extension'])

le_mime = LabelEncoder()
df['MIME_encoded'] = le_mime.fit_transform(df['MIME type'])

# le_meta = LabelEncoder()
# df['Metadata_encoded'] = le_meta.fit_transform(df['Description'])

le_category = LabelEncoder()
df['Category_encoded'] = le_category.fit_transform(df['Category'])

# Features and target
X = df[['Extension_encoded', 'MIME_encoded']]
y = df['Category_encoded']

# Function to evaluate models and print evaluation metrics
def evaluate_model(model):
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
    confusion = confusion_matrix(y, y_pred)
    
    print(f"{model.__class__.__name__} Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(confusion)
    print("\nClassification Report:")
    print(classification_report(y, y_pred, zero_division=0))

# 1. Decision Tree
decision_tree = DecisionTreeClassifier()
evaluate_model(decision_tree)

# 2. Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
#evaluate_model(random_forest)

# 3. Support Vector Machine (SVM)
svm_model = SVC()
evaluate_model(svm_model)

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Define an improved neural network model
# neural_network = MLPClassifier(
#     hidden_layer_sizes=(128, 128, 64),  # 3 hidden layers with more neurons
#     activation='relu',                  # ReLU activation function
#     solver='adam',                      # Adam optimizer
#     alpha=0.001,                        # L2 regularization (to prevent overfitting)
#     learning_rate='adaptive',           # Learning rate adapts based on learning progress
#     max_iter=1000,                      # Increase iterations to allow for better training
#     batch_size=64,                      # Batch size for training
#     random_state=42,
#     verbose=True                        # Output training progress
# )

# # Train the model on the full dataset
# neural_network.fit(X_scaled, y)

# # Predict on the same data (since no splitting)
# y_pred = neural_network.predict(X_scaled)

# def evaluate_full_data_model(model, X_scaled, y):
#     y_pred = model.predict(X_scaled)
    
#     # Metrics
#     print("Full Data Evaluation:")
#     print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
#     print(f"Precision: {precision_score(y, y_pred, average='weighted', zero_division=0):.4f}")
#     print(f"Recall: {recall_score(y, y_pred, average='weighted', zero_division=0):.4f}")
#     print(f"F1 Score: {f1_score(y, y_pred, average='weighted', zero_division=0):.4f}")
    
#     # Confusion Matrix
#     print("\nConfusion Matrix:")
#     print(confusion_matrix(y, y_pred))

# # Evaluate the neural network on full dataset
# evaluate_full_data_model(neural_network, X_scaled, y)

# Save the models and encoders
with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(decision_tree, file)

with open('le_extension.pkl', 'wb') as ext_file:
    pickle.dump(le_ext, ext_file)

with open('le_mime.pkl', 'wb') as mime_file:
    pickle.dump(le_mime, mime_file)

with open('le_category.pkl', 'wb') as category_file:
    pickle.dump(le_category, category_file)


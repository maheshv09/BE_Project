import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv('Phase-1/file_extensions_with_mime.csv')
df['MIME Type'] = df['MIME Type'].fillna('unknown')

le_ext = LabelEncoder()
df['Extension_encoded'] = le_ext.fit_transform(df['Extension'])

le_mime = LabelEncoder()
df['MIME_encoded'] = le_mime.fit_transform(df['MIME Type'])

le_meta = LabelEncoder()
df['Metadata_encoded'] = le_meta.fit_transform(df['Metadata'])

le_category = LabelEncoder()
df['Category_encoded'] = le_category.fit_transform(df['Category'])

X = df[['Extension_encoded', 'MIME_encoded', 'Metadata_encoded']]
y = df['Category_encoded']

def evaluate_model(model):
    model.fit(X, y)
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"{model.__class__.__name__} Accuracy: {accuracy:.4f}")

decision_tree = DecisionTreeClassifier()
evaluate_model(decision_tree)

random_forest = RandomForestClassifier(n_estimators=100)
evaluate_model(random_forest)

svm_model = SVC()
#evaluate_model(svm_model)

def predict_new_extension(extension, mime_type, metadata):
    ext_encoded = le_ext.transform([extension])
    mime_encoded = le_mime.transform([mime_type])
    meta_encoded = le_meta.transform([metadata])
    print(ext_encoded, mime_encoded, meta_encoded)
    new_data = [[ext_encoded[0], mime_encoded[0], meta_encoded[0]]]
    prediction = decision_tree.predict(new_data)
    category_predicted = le_category.inverse_transform(prediction)
    
    print(f"Predicted Category for {extension}: {category_predicted[0]}")

new_extension = '.pls'
new_mime_type = 'application/pls+xml'
new_metadata = 'Pro/ENGINEER temporary data'

predict_new_extension(new_extension, new_mime_type, new_metadata)
# .pls,Pro/ENGINEER temporary data,Temporary,application/pls+xml
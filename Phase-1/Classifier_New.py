import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import mimetypes
import magic
import pickle

df = pd.read_csv('merged_file.csv')
# df['MIME Type'] = df['MIME Type'].fillna('unknown')

le_ext = LabelEncoder()
df['Extension_encoded'] = le_ext.fit_transform(df['Extension'])

le_mime = LabelEncoder()
df['MIME_encoded'] = le_mime.fit_transform(df['MIME type'])

le_meta = LabelEncoder()
df['Metadata_encoded'] = le_meta.fit_transform(df['Description'])

le_category = LabelEncoder()
df['Category_encoded'] = le_category.fit_transform(df['Category'])

X = df[['Extension_encoded', 'MIME_encoded']]
y = df['Category_encoded']

def evaluate_model(model):
    model.fit(X, y)
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"{model.__class__.__name__} Accuracy: {accuracy:.4f}")

decision_tree = DecisionTreeClassifier()
evaluate_model(decision_tree)

random_forest = RandomForestClassifier(n_estimators=100)
#evaluate_model(random_forest)

svm_model = SVC()
# evaluate_model(svm_model)

def get_mime_type_from_extension(extension):
    mime_type, _ = mimetypes.guess_type(f"file{extension}")
    return mime_type if mime_type else "Unknown"    

def get_mime_type(file_path):
    mime = magic.Magic(mime=True)
    mime_type = mime.from_file(file_path)
    return mime_type


def get_mime_type_fast(file_path, read_size=1024):
    with open(file_path, 'rb') as file:
        file_start = file.read(read_size)
    mime = magic.Magic(mime=True)
    mime_type = mime.from_buffer(file_start)
    return mime_type


def predict_new_extension(extension, mime_type):
    ext_encoded = le_ext.transform([extension])
    mime_encoded = le_mime.transform([mime_type])
    new_data = [[ext_encoded[0], mime_encoded[0]]]
    prediction = decision_tree.predict(new_data)
    category_predicted = le_category.inverse_transform(prediction)
    
    print(f"Predicted Category for {extension}: {category_predicted[0]}")

new_extension = '.mp4'
new_mime_type = get_mime_type_fast("F:\mahesh\BE Project\Phase-1\Classifier.py")
print("New mime:" + new_mime_type)                                                   

predict_new_extension(new_extension, new_mime_type)
# .pls,Pro/ENGINEER temporary data,Temporary,application/pls+xml

with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(decision_tree, file)

with open('le_extension.pkl', 'wb') as ext_file:
    pickle.dump(le_ext, ext_file)

with open('le_mime.pkl', 'wb') as mime_file:
    pickle.dump(le_mime, mime_file)

with open('le_category.pkl', 'wb') as category_file:
    pickle.dump(le_category, category_file)
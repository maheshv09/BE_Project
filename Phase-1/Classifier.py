import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv('Phase-1\mime_extensions.csv')

def categorize_file_type(file_type):
    if 'application' in file_type:
        return 'application'
    elif 'audio' in file_type:
        return 'audio'
    elif 'image' in file_type:
        return 'image'
    elif 'video' in file_type:
        return 'video'
    elif 'text' in file_type:
        return 'text'
    else:
        return 'other'

df['Category'] = df['File Type'].apply(categorize_file_type)

df['Extensions'] = df['Extensions'].apply(lambda x: x.strip("[]").replace("'", "").split(",")[0])

le_ext = LabelEncoder()
df['Extensions_encoded'] = le_ext.fit_transform(df['Extensions'])

le_cat = LabelEncoder()
df['Category_encoded'] = le_cat.fit_transform(df['Category'])

X = df[['Extensions_encoded']]
y = df['Category_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model.__class__.__name__} Accuracy: {accuracy:.4f}")

decision_tree = DecisionTreeClassifier()
evaluate_model(decision_tree)

random_forest = RandomForestClassifier(n_estimators=100)
evaluate_model(random_forest)

svm_model = SVC()
evaluate_model(svm_model)

def predict_new_extension(extension):
    extension_encoded = le_ext.transform([extension])
    prediction = decision_tree.predict([extension_encoded])
    file_type_predicted = le_cat.inverse_transform(prediction)
    print(f"Predicted Category: {file_type_predicted[0]}")

new_extension = 'mp3'  
predict_new_extension(new_extension)

asx
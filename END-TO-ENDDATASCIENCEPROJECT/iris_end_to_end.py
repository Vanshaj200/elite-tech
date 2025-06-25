import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from flask import Flask, request, jsonify
import numpy as np
import os

# 1. Data Collection & Preprocessing
data = load_iris(as_frame=True)
df = data.frame
X = df.drop('target', axis=1)
y = df['target']

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 4. Evaluation
preds = clf.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Test Accuracy: {acc:.4f}")

# 5. Model Serialization
model_path = 'project_2/iris_rf_model.joblib'
os.makedirs('project_2', exist_ok=True)
joblib.dump(clf, model_path)
print(f"Model saved to {model_path}")

# 6. Flask API
def create_app():
    app = Flask(__name__)
    model = joblib.load(model_path)
    feature_names = X.columns.tolist()

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json()
        features = [data.get(f, 0) for f in feature_names]
        arr = np.array(features).reshape(1, -1)
        pred = model.predict(arr)[0]
        class_name = data.target_names[pred] if hasattr(data, 'target_names') else str(pred)
        return jsonify({'prediction': int(pred), 'class_name': class_name})

    @app.route('/')
    def home():
        return 'Iris Classifier API is running!'

    return app

if __name__ == '__main__':
    # To run the API: python project_2/iris_end_to_end.py
    app = create_app()
    app.run(debug=True, port=5000)

"""
# Example API usage (with requests):
import requests
sample = {
    'sepal length (cm)': 5.1,
    'sepal width (cm)': 3.5,
    'petal length (cm)': 1.4,
    'petal width (cm)': 0.2
}
r = requests.post('http://127.0.0.1:5000/predict', json=sample)
print(r.json())
""" 
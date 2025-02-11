from flask import Flask, request, jsonify
import pickle
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS globally

# Load trained models with absolute paths
model_dir = os.path.dirname(os.path.abspath(__file__))  # Get current directory

try:
    diabetes_model = pickle.load(open(os.path.join(model_dir, "diabetes.pkl"), "rb"))
    stress_model = pickle.load(open(os.path.join(model_dir, "stress.pkl"), "rb"))
    thyroid_model = pickle.load(open(os.path.join(model_dir, "thyroid.pkl"), "rb"))
    heart_model = pickle.load(open(os.path.join(model_dir, "heart.pkl"), "rb"))
    calorie_model = pickle.load(open(os.path.join(model_dir, "calorie.pkl"), "rb"))
except Exception as e:
    print(f"Error loading models: {e}")
    diabetes_model = None
    stress_model = None
    thyroid_model = None
    heart_model = None
    calorie_model = None

@app.route('/')
def home():
    return "Welcome to the Health Prediction API!"

# Diabetes Prediction
@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    if not diabetes_model:
        return jsonify({'error': 'Diabetes model not loaded'}), 500

    data = request.json
    features = np.array([list(data.values())]).astype(float)
    prediction = diabetes_model.predict(features)[0]
    return jsonify({'Diabetes Prediction': int(prediction)})

# Stress Level Prediction
@app.route('/predict/stress', methods=['POST'])
def predict_stress():
    if not stress_model:
        return jsonify({'error': 'Stress model not loaded'}), 500

    data = request.json
    features = np.array([list(data.values())]).astype(float)
    prediction = stress_model.predict(features)[0]
    return jsonify({'Stress Level': int(prediction)})

# Thyroid Prediction
@app.route('/predict/thyroid', methods=['POST'])
def predict_thyroid():
    if not thyroid_model:
        return jsonify({'error': 'Thyroid model not loaded'}), 500

    data = request.json
    features = np.array([list(data.values())]).astype(float)
    prediction = thyroid_model.predict(features)[0]
    return jsonify({'Thyroid Condition': int(prediction)})

# Heart Disease Prediction
@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    if not heart_model:
        return jsonify({'error': 'Heart model not loaded'}), 500

    data = request.json
    features = np.array([list(data.values())]).astype(float)
    prediction = heart_model.predict(features)[0]
    return jsonify({'Heart Disease Prediction': int(prediction)})

# Calorie Burn Prediction (Fix: Change from "/predict/calories" to "/predict/calorie")
@app.route('/predict/calorie', methods=['POST'])
def predict_calorie():
    if not calorie_model:
        return jsonify({'error': 'Calorie model not loaded'}), 500

    data = request.json
    features = np.array([list(data.values())]).astype(float)
    prediction = calorie_model.predict(features)[0]
    return jsonify({'Estimated Calories Burnt': float(prediction)})

if __name__ == '_main_':
    app.run(debug=True)
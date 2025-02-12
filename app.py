from flask import Flask, request, jsonify
import pickle
import numpy as np
import os
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict/*": {"origins": "*"}})  # Restrict CORS

# Load trained models with absolute paths
model_dir = os.path.dirname(os.path.abspath(__file__))  # Get current directory

def load_model(filename):
    try:
        model = pickle.load(open(os.path.join(model_dir, filename), "rb"))
        print(f"{filename} loaded successfully!")
        return model
    except FileNotFoundError:
        print(f"{filename} model file not found.")
        return None
    except Exception as e:
        print(f"Error loading {filename} model: {e}")
        return None

# Load models
diabetes_model = load_model("diabetesnew.pkl")
stress_model = load_model("stress_model.pkl")
thyroid_model = load_model("thyroid.pkl")
heart_model = load_model("heart.pkl")
calorie_model = load_model("calorie.pkl")

@app.route('/')
def home():
    return "Welcome to the Health Prediction API!"

def make_prediction(model, data, expected_features):
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    if not isinstance(data, dict) or len(data) != len(expected_features):
        return jsonify({'error': 'Invalid input data'}), 400

    try:
        features_df = pd.DataFrame([data], columns=expected_features)
        prediction = model.predict(features_df)[0]
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    expected_features = ["Age", "Family_Diabetes", "highBP", "BMI", "Alcohol", "Sleep", 
                         "RegularMedicine", "JunkFood", "Stress", "BPLevel", "Pregancies", 
                         "Pdiabetes", "UriationFreq"]
    return make_prediction(diabetes_model, request.json, expected_features)

@app.route('/predict/stress', methods=['POST'])
def predict_stress():
    expected_features = ["Gender", "Age", "Occupation", "Sleep Duration", "BMI Category", 
                         "Heart Rate", "Daily Steps", "Systolic BP"]
    return make_prediction(stress_model, request.json, expected_features)

@app.route('/predict/thyroid', methods=['POST'])
def predict_thyroid():
    expected_features = ["age", "sex", "TT4", "T3", "T4U", "FTI", "TSH", "pregnant"]
    return make_prediction(thyroid_model, request.json, expected_features)

@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    expected_features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "thal"]
    return make_prediction(heart_model, request.json, expected_features)

@app.route('/predict/calorie', methods=['POST'])
def predict_calorie():
    if not calorie_model:
        return jsonify({'error': 'Calorie model not loaded'}), 500

    expected_features = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", "Gender"]
    data = request.json

    if not isinstance(data, dict) or len(data) != len(expected_features):
        return jsonify({'error': 'Invalid input data'}), 400

    try:
        features_df = pd.DataFrame([data], columns=expected_features)
        prediction = calorie_model.predict(features_df)[0]
        return jsonify({'Calorie Burnt Prediction': float(prediction)})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == "__main__":  # ✅ Fixed the typo here
    app.run(debug=False)

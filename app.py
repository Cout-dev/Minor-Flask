from flask import Flask, request, jsonify
import pickle
import numpy as np
import os
import pandas as pd
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS globally

# Load trained models with absolute paths
model_dir = os.path.dirname(os.path.abspath(__file__))  # Get current directory

try:
    diabetes_model = pickle.load(open(os.path.join(model_dir, "diabetes.pkl"), "rb"))
    stress_model = pickle.load(open(os.path.join(model_dir, "stress.pkl"), "rb"))
    lungs_model = pickle.load(open(os.path.join(model_dir, "lungs.pkl"), "rb"))
    covid_model = pickle.load(open(os.path.join(model_dir, "covid.pkl"), "rb"))  # COVID Model
    genhealth_model = pickle.load(open(os.path.join(model_dir, "genhealth.pkl"), "rb"))  # General Health Model
    sleephealth_model = pickle.load(open(os.path.join(model_dir, "sleep2.pkl"), "rb"))  # Sleep Health Model
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    diabetes_model = None
    stress_model = None
    lungs_model = None
    covid_model = None
    genhealth_model = None
    sleephealth_model = None

@app.route('/')
def home():
    return "Welcome to the Health Prediction API!"

# Test endpoint
@app.route('/test', methods=['GET'])
def test_api():
    return jsonify({'message': 'API is working fine!'})

# Diabetes Prediction
@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    if not diabetes_model:
        return jsonify({'error': 'Diabetes model not loaded'}), 500
    
    data = request.json
    try:
        features = np.array([list(data.values())]).astype(float)
        prediction = int(diabetes_model.predict(features)[0])
        return jsonify({'Diabetes Prediction': prediction})
    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

@app.route('/predict/stress', methods=['POST'])
def predict_stress():
    if not stress_model:
        return jsonify({'error': 'Stress model not loaded'}), 500

    data = request.json
    try:
        features = np.array([list(data.values())]).astype(float)
        prediction = stress_model.predict(features)[0]
        return jsonify({'Stress Level': int(prediction)})
    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

# Lungs Disease Prediction
@app.route('/predict/lungs', methods=['POST'])
def predict_lungs():
    if not lungs_model:
        return jsonify({'error': 'Lungs model not loaded'}), 500

    data = request.json
    try:
        feature_order = [
            "AGE", "GENDER", "SMOKING", "FINGER_DISCOLORATION", "MENTAL_STRESS",
            "EXPOSURE_TO_POLLUTION", "LONG_TERM_ILLNESS", "IMMUNE_WEAKNESS",
            "BREATHING_ISSUE", "ALCOHOL_CONSUMPTION", "THROAT_DISCOMFORT",
            "CHEST_TIGHTNESS", "FAMILY_HISTORY", "SMOKING_FAMILY_HISTORY", "STRESS_IMMUNE"
        ]

        features = np.array([[float(data[feature]) for feature in feature_order]])
        prediction = int(lungs_model.predict(features)[0])
        return jsonify({'Lungs Prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

# COVID Prediction
@app.route('/predict/covid', methods=['POST'])
def predict_covid():
    if not covid_model:
        return jsonify({'error': 'COVID model not loaded'}), 500

    data = request.json
    try:
        feature_order = [
            "Breathing Problem", "Fever", "Dry Cough", "Sore throat", "Hyper Tension",
            "Abroad travel", "Contact with COVID Patient", "Attended Large Gathering",
            "Visited Public Exposed Places", "Family working in Public Exposed Places"
        ]

        features = np.array([[float(data[feature]) for feature in feature_order]])
        prediction = covid_model.predict(features)[0]
        return jsonify({'COVID Risk Prediction': int(prediction)})
    
    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

# General Health Prediction
@app.route('/predict/genhealth', methods=['POST'])
def predict_genhealth():
    if not genhealth_model:
        return jsonify({'error': 'General health model not loaded'}), 500

    data = request.json
    try:
        features = np.array([list(data.values())]).astype(float)
        prediction = int(genhealth_model.predict(features)[0])
        return jsonify({'General Health Prediction': prediction})
    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500
    

import pandas as pd  # Add this import if not already present

# Mapping for Sleep Disorder Predictions
sleep_disorder_mapping = {0: "Insomnia", 1: "Sleep Apnea", 2: "None"}

@app.route('/predict/sleep', methods=['POST'])
def predict_sleephealth():
    if not sleephealth_model:
        return jsonify({'error': 'Sleep health model not loaded'}), 500

    data = request.json
    try:
        # Define expected features (from model training)
        expected_features = [
            'Gender', 'Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 
            'Stress Level', 'BMI Category', 'Heart Rate', 'Daily Steps', 'Systolic BP', 'Diastolic BP',
            'Occupation_Doctor', 'Occupation_Engineer', 'Occupation_Lawyer', 'Occupation_Manager', 
            'Occupation_Nurse', 'Occupation_Sales Representative', 'Occupation_Salesperson', 
            'Occupation_Scientist', 'Occupation_Software Engineer', 'Occupation_Teacher'
        ]

        # Ensure all expected features are present, fill missing ones with 0
        for feature in expected_features:
            if feature not in data:
                data[feature] = 0  # Assign default value 0 if missing

        # Convert input to DataFrame with correct order
        features = pd.DataFrame([data], columns=expected_features)

        # Perform prediction
        predicted_label = sleephealth_model.predict(features)[0]  # Model returns an integer

        # Convert numeric prediction to readable label
        predicted_disorder = sleep_disorder_mapping.get(int(predicted_label), "Unknown")

        return jsonify({'Sleep Disorder Prediction': predicted_disorder})
    
    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

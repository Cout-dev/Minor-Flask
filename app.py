from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained models
diabetes_model = pickle.load(open("diabetes.pkl", "rb"))
stress_model = pickle.load(open("stress.pkl", "rb"))
thyroid_model = pickle.load(open("thyroid.pkl", "rb"))
heart_model = pickle.load(open("heart.pkl", "rb"))
calorie_model = pickle.load(open("calorie.pkl", "rb"))

@app.route('/')
def home():
    return "Welcome to the Health Prediction API!"

# Diabetes Prediction
@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    data = request.json
    features = np.array([list(data.values())]).astype(float)
    prediction = diabetes_model.predict(features)[0]
    return jsonify({'Diabetes Prediction': int(prediction)})

# Stress Level Prediction
@app.route('/predict/stress', methods=['POST'])
def predict_stress():
    data = request.json
    features = np.array([list(data.values())]).astype(float)
    prediction = stress_model.predict(features)[0]
    return jsonify({'Stress Level': int(prediction)})

# Thyroid Prediction
@app.route('/predict/thyroid', methods=['POST'])
def predict_thyroid():
    data = request.json
    features = np.array([list(data.values())]).astype(float)
    prediction = thyroid_model.predict(features)[0]
    return jsonify({'Thyroid Condition': int(prediction)})

# Heart Disease Prediction
@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    data = request.json
    features = np.array([list(data.values())]).astype(float)
    prediction = heart_model.predict(features)[0]
    return jsonify({'Heart Disease Prediction': int(prediction)})

# Calorie Burn Prediction
@app.route('/predict/calories', methods=['POST'])
def predict_calories():
    data = request.json
    features = np.array([list(data.values())]).astype(float)
    prediction = calorie_model.predict(features)[0]
    return jsonify({'Estimated Calories Burnt': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
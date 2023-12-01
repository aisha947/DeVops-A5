from flask import Flask, request, jsonify
import cloudpickle
import pandas as pd
import numpy as np  # Added import for NumPy

app = Flask(__name__)

# Load pre-trained models
with open('model_min_t.pkl', 'rb') as f:
    model_min_t = cloudpickle.load(f)

with open('model_max_t.pkl', 'rb') as f:
    model_max_t = cloudpickle.load(f)

# API endpoint for predicting Min Temperature
@app.route('/predict/min_t', methods=['POST'])
def predict_min_t():
    try:
        # Get input data
        input_data = request.get_json()
        num_days = input_data['num_days']

        # Create dummy data for prediction (adjust as needed)
        dummy_data = pd.DataFrame({'Humidity': [50], 'Wind_Speed': [10], 'Weather_Condition_Clear': [1]})

        # Make predictions
        predictions = []
        for day in range(num_days):
            prediction = {'date': f'Day {day + 1}', 'min_temperature': model_min_t.predict(dummy_data.values)[0]}
            predictions.append(prediction)

        return jsonify({'predictions': predictions})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# API endpoint for predicting Max Temperature
@app.route('/predict/max_t', methods=['POST'])
def predict_max_t():
    try:
        # Get input data
        input_data = request.get_json()
        num_days = input_data['num_days']

        # Create dummy data for prediction (adjust as needed)
        dummy_data = pd.DataFrame({'Humidity': [50], 'Wind_Speed': [10], 'Weather_Condition_Clear': [1]})

        # Make predictions
        predictions = []
        for day in range(num_days):
            prediction = {'date': f'Day {day + 1}', 'max_temperature': model_max_t.predict(dummy_data.values)[0]}
            predictions.append(prediction)

        return jsonify({'predictions': predictions})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)


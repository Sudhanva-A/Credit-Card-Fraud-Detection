from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
from joblib import load
import os
from flask_cors import CORS
import pandas as pd
from io import BytesIO 

app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')
CORS(app)

# Load scaler and models
scaler = load('../models/best-models/scaler.joblib')
model_paths = {
    "xgboost": "../models/best-models/xgboost.joblib",
    "random_forest": "../models/best-models/best_rf_model.joblib",
    "logistic": "../models/best-models/logisticregression.joblib"
}
current_model = load(model_paths["xgboost"])
current_model_name = "xgboost"

# 🧠 In-memory storage for predictions
predictions_log = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    scaled = scaler.transform(features)
    proba = current_model.predict_proba(scaled)[0][1]
    is_fraud = proba > 0.2

    result = {
        'features': data['features'],
        'probability': round(float(proba), 4),
        'is_fraud': bool(is_fraud),
        'model': current_model_name
    }

    # Save to global log
    predictions_log.append(result)

    return jsonify({
        'probability': result['probability'],
        'is_fraud': result['is_fraud']
    })

@app.route('/change-model', methods=['POST'])
def change_model():
    global current_model, current_model_name
    model_name = request.json.get('model')
    if model_name in model_paths:
        current_model = load(model_paths[model_name])
        current_model_name = model_name
        return jsonify({'message': f'{model_name} model loaded successfully.', 'model': model_name})
    else:
        return jsonify({'error': 'Model not found'}), 404

@app.route('/download', methods=['GET'])
def download():
    if not predictions_log:
        return jsonify({"error": "No data to download."}), 400

    df = pd.DataFrame(predictions_log)
    csv_bytes = df.to_csv(index=False).encode('utf-8')  # convert to bytes
    buffer = BytesIO(csv_bytes)  # wrap in BytesIO
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype='text/csv',
        as_attachment=True,
        download_name='fraud_predictions.csv'
    )
@app.route('/stats', methods=['GET'])
def stats():
    total = len(predictions_log)
    frauds = sum(1 for p in predictions_log if p['is_fraud'])
    rate = round((frauds / total) * 100, 2) if total else 0.0
    latest_model = predictions_log[-1]['model'] if predictions_log else current_model_name
    return jsonify({
        'total_predictions': total,
        'fraud_predictions': frauds,
        'fraud_rate': rate,
        'model': latest_model
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

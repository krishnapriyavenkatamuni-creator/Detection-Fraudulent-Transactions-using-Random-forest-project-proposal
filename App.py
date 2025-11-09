from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature names (must match training data)
FEATURE_NAMES = [
    'transaction_amount', 'transaction_hour', 'day_of_week', 'age',
    'num_transactions_last_month', 'account_age_days', 'distance_from_home',
    'distance_from_last_transaction', 'ratio_to_median_purchase',
    'used_chip', 'used_pin_number', 'online_order'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Extract features
        features = [
            float(data.get('transaction_amount', 0)),
            int(data.get('transaction_hour', 0)),
            int(data.get('day_of_week', 0)),
            int(data.get('age', 0)),
            int(data.get('num_transactions_last_month', 0)),
            int(data.get('account_age_days', 0)),
            float(data.get('distance_from_home', 0)),
            float(data.get('distance_from_last_transaction', 0)),
            float(data.get('ratio_to_median_purchase', 0)),
            int(data.get('used_chip', 0)),
            int(data.get('used_pin_number', 0)),
            int(data.get('online_order', 0))
        ]
        
        # Create DataFrame with proper feature names
        features_df = pd.DataFrame([features], columns=FEATURE_NAMES)
        
        # Scale features
        features_scaled = scaler.transform(features_df)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Prepare response
        result = {
            'prediction': 'FRAUD' if prediction == 1 else 'LEGITIMATE',
            'fraud_probability': float(probability[1] * 100),
            'legitimate_probability': float(probability[0] * 100),
            'risk_level': get_risk_level(probability[1]),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def get_risk_level(fraud_prob):
    """Determine risk level based on fraud probability"""
    if fraud_prob < 0.3:
        return 'LOW'
    elif fraud_prob < 0.6:
        return 'MEDIUM'
    else:
        return 'HIGH'

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Handle batch predictions for multiple transactions"""
    try:
        data = request.get_json()
        transactions = data.get('transactions', [])
        
        results = []
        for transaction in transactions:
            features = [
                float(transaction.get('transaction_amount', 0)),
                int(transaction.get('transaction_hour', 0)),
                int(transaction.get('day_of_week', 0)),
                int(transaction.get('age', 0)),
                int(transaction.get('num_transactions_last_month', 0)),
                int(transaction.get('account_age_days', 0)),
                float(transaction.get('distance_from_home', 0)),
                float(transaction.get('distance_from_last_transaction', 0)),
                float(transaction.get('ratio_to_median_purchase', 0)),
                int(transaction.get('used_chip', 0)),
                int(transaction.get('used_pin_number', 0)),
                int(transaction.get('online_order', 0))
            ]
            
            features_df = pd.DataFrame([features], columns=FEATURE_NAMES)
            features_scaled = scaler.transform(features_df)
            
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0]
            
            results.append({
                'transaction_id': transaction.get('transaction_id', 'N/A'),
                'prediction': 'FRAUD' if prediction == 1 else 'LEGITIMATE',
                'fraud_probability': float(probability[1] * 100),
                'risk_level': get_risk_level(probability[1])
            })
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("Starting Fraud Detection Web Application...")
    print("Server running at http://127.0.0.1:5000/")
    app.run(debug=True, host='0.0.0.0', port=5000)
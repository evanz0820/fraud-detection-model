"""
Flask REST API for Credit Card Fraud Detection
Serves model predictions and statistics to the Angular frontend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import pandas as pd
import joblib
import json
import os
import random
from datetime import datetime, timedelta

from model import FraudDetectionNN, FraudDetectionAutoencoder

app = Flask(__name__)
CORS(app)  # Enable CORS for Angular frontend

# Global variables for model and data
model = None
scaler = None
model_info = None
sample_data = None

MODEL_DIR = 'models'
DATA_PATH = 'data/creditcard.csv'


def load_model():
    """Load the trained model and scaler"""
    global model, scaler, model_info
    
    model_path = os.path.join(MODEL_DIR, 'nn_model.pth')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.joblib')
    info_path = os.path.join(MODEL_DIR, 'model_info.json')
    
    if not os.path.exists(model_path):
        print("Model not found. Please train the model first.")
        return False
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    model = FraudDetectionNN(input_dim=checkpoint.get('input_dim', 30))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load scaler
    scaler = joblib.load(scaler_path)
    
    # Load model info
    with open(info_path, 'r') as f:
        model_info = json.load(f)
    
    print("Model loaded successfully!")
    return True


def load_sample_data():
    """Load sample data for demonstration"""
    global sample_data
    
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        # Get a mix of fraud and normal transactions
        fraud_samples = df[df['Class'] == 1].sample(min(50, len(df[df['Class'] == 1])))
        normal_samples = df[df['Class'] == 0].sample(200)
        sample_data = pd.concat([fraud_samples, normal_samples]).to_dict('records')
        print(f"Loaded {len(sample_data)} sample transactions")
    else:
        sample_data = []


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """Get model information and metadata"""
    if model_info is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    return jsonify({
        'model_type': model_info.get('model_type', 'nn'),
        'trained_at': model_info.get('trained_at'),
        'dataset_info': model_info.get('dataset_info', {}),
        'architecture': {
            'type': 'Deep Neural Network',
            'layers': [128, 64, 32],
            'activation': 'ReLU',
            'dropout': 0.3
        }
    })


@app.route('/api/model/metrics', methods=['GET'])
def get_model_metrics():
    """Get model evaluation metrics"""
    if model_info is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    metrics = model_info.get('metrics', {})
    
    return jsonify({
        'accuracy': metrics.get('accuracy', 0),
        'precision': metrics.get('precision', 0),
        'recall': metrics.get('recall', 0),
        'f1_score': metrics.get('f1_score', 0),
        'roc_auc': metrics.get('roc_auc', 0),
        'average_precision': metrics.get('average_precision', 0),
        'optimal_threshold': metrics.get('optimal_threshold', 0.5),
        'confusion_matrix': metrics.get('confusion_matrix', [[0, 0], [0, 0]]),
        'precision_recall_curve': metrics.get('precision_recall_curve', {'precision': [], 'recall': []})
    })


@app.route('/api/model/training-history', methods=['GET'])
def get_training_history():
    """Get model training history"""
    if model_info is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    return jsonify({
        'history': model_info.get('training_history', [])
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make fraud prediction on transaction data"""
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        data = request.get_json()
        
        # Handle single transaction or batch
        if 'transactions' in data:
            transactions = data['transactions']
        else:
            transactions = [data]
        
        results = []
        
        for txn in transactions:
            # Extract features (V1-V28, Time, Amount)
            features = []
            feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
            
            for feat in feature_names:
                features.append(float(txn.get(feat, 0)))
            
            # Scale and predict
            X = np.array([features])
            X_scaled = scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled)
            
            with torch.no_grad():
                prob = model(X_tensor).squeeze().item()
            
            threshold = model_info.get('metrics', {}).get('optimal_threshold', 0.5)
            is_fraud = prob >= threshold
            
            results.append({
                'is_fraud': bool(is_fraud),
                'fraud_probability': float(prob),
                'confidence': float(abs(prob - 0.5) * 2),
                'risk_level': 'HIGH' if prob > 0.8 else 'MEDIUM' if prob > 0.5 else 'LOW',
                'amount': txn.get('Amount', 0)
            })
        
        if len(results) == 1:
            return jsonify(results[0])
        
        return jsonify({'predictions': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/dashboard/stats', methods=['GET'])
def get_dashboard_stats():
    """Get dashboard statistics for the frontend"""
    if model_info is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    dataset_info = model_info.get('dataset_info', {})
    metrics = model_info.get('metrics', {})
    
    # Generate simulated real-time stats
    total_transactions = dataset_info.get('total_transactions', 284807)
    fraud_count = dataset_info.get('fraud_count', 492)
    
    return jsonify({
        'overview': {
            'total_transactions': total_transactions,
            'total_fraud_detected': fraud_count,
            'fraud_rate': dataset_info.get('fraud_percentage', 0.17),
            'amount_saved': round(random.uniform(100000, 500000), 2),
            'model_accuracy': metrics.get('accuracy', 0.99) * 100
        },
        'model_performance': {
            'accuracy': metrics.get('accuracy', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1_score': metrics.get('f1_score', 0),
            'roc_auc': metrics.get('roc_auc', 0)
        },
        'confusion_matrix': {
            'true_negative': metrics.get('confusion_matrix', [[0, 0], [0, 0]])[0][0],
            'false_positive': metrics.get('confusion_matrix', [[0, 0], [0, 0]])[0][1],
            'false_negative': metrics.get('confusion_matrix', [[0, 0], [0, 0]])[1][0],
            'true_positive': metrics.get('confusion_matrix', [[0, 0], [0, 0]])[1][1]
        },
        'recent_activity': generate_recent_activity(),
        'hourly_distribution': generate_hourly_distribution(),
        'risk_distribution': {
            'low': round(random.uniform(85, 95), 1),
            'medium': round(random.uniform(3, 10), 1),
            'high': round(random.uniform(0.1, 2), 1)
        }
    })


def generate_recent_activity():
    """Generate simulated recent transaction activity"""
    activity = []
    now = datetime.now()
    
    for i in range(10):
        is_fraud = random.random() < 0.15
        activity.append({
            'id': f'TXN{random.randint(100000, 999999)}',
            'timestamp': (now - timedelta(minutes=random.randint(1, 60))).isoformat(),
            'amount': round(random.uniform(10, 2000), 2),
            'is_fraud': is_fraud,
            'risk_level': 'HIGH' if is_fraud else random.choice(['LOW', 'LOW', 'LOW', 'MEDIUM']),
            'confidence': round(random.uniform(0.7, 0.99), 2)
        })
    
    return sorted(activity, key=lambda x: x['timestamp'], reverse=True)


def generate_hourly_distribution():
    """Generate hourly transaction distribution"""
    return [
        {'hour': i, 'transactions': random.randint(100, 500), 'fraud': random.randint(0, 5)}
        for i in range(24)
    ]


@app.route('/api/simulate/transaction', methods=['GET'])
def simulate_transaction():
    """Simulate a transaction for live demo"""
    if model is None or scaler is None:
        # Return simulated data if model not loaded
        is_fraud = random.random() < 0.1
        return jsonify({
            'transaction': {
                'id': f'TXN{random.randint(100000, 999999)}',
                'timestamp': datetime.now().isoformat(),
                'amount': round(random.uniform(10, 1000), 2)
            },
            'prediction': {
                'is_fraud': is_fraud,
                'fraud_probability': round(random.uniform(0.8, 0.99) if is_fraud else random.uniform(0.01, 0.3), 4),
                'risk_level': 'HIGH' if is_fraud else 'LOW',
                'processing_time_ms': random.randint(5, 50)
            }
        })
    
    # Generate random transaction features
    features = {
        'Time': random.uniform(0, 172800),
        'Amount': round(random.uniform(1, 5000), 2)
    }
    
    # Generate PCA features (normally distributed for legit, skewed for fraud)
    is_fraud_sample = random.random() < 0.1
    for i in range(1, 29):
        if is_fraud_sample:
            features[f'V{i}'] = random.gauss(0, 2) + random.choice([-1, 1]) * random.uniform(1, 3)
        else:
            features[f'V{i}'] = random.gauss(0, 1)
    
    # Make prediction
    feature_values = [features['Time']] + [features[f'V{i}'] for i in range(1, 29)] + [features['Amount']]
    X = np.array([feature_values])
    X_scaled = scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled)
    
    with torch.no_grad():
        prob = model(X_tensor).squeeze().item()
    
    threshold = model_info.get('metrics', {}).get('optimal_threshold', 0.5)
    
    return jsonify({
        'transaction': {
            'id': f'TXN{random.randint(100000, 999999)}',
            'timestamp': datetime.now().isoformat(),
            'amount': features['Amount'],
            'features': {k: round(v, 4) for k, v in features.items()}
        },
        'prediction': {
            'is_fraud': prob >= threshold,
            'fraud_probability': round(prob, 4),
            'risk_level': 'HIGH' if prob > 0.8 else 'MEDIUM' if prob > threshold else 'LOW',
            'processing_time_ms': random.randint(5, 50)
        }
    })


@app.route('/api/sample-transactions', methods=['GET'])
def get_sample_transactions():
    """Get sample transactions from the dataset"""
    if sample_data:
        samples = random.sample(sample_data, min(20, len(sample_data)))
        return jsonify({'transactions': samples})
    
    # Generate synthetic samples if no data
    samples = []
    for _ in range(20):
        is_fraud = random.random() < 0.2
        samples.append({
            'Time': random.uniform(0, 172800),
            'Amount': round(random.uniform(1, 2000), 2),
            'Class': 1 if is_fraud else 0,
            **{f'V{i}': round(random.gauss(0, 1 if not is_fraud else 2), 4) for i in range(1, 29)}
        })
    
    return jsonify({'transactions': samples})


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("=" * 50)
    print("Credit Card Fraud Detection API")
    print("=" * 50)
    
    # Try to load model
    if os.path.exists(MODEL_DIR):
        load_model()
        load_sample_data()
    else:
        print("Warning: Model directory not found. Running in demo mode.")
        print("Train the model first using: python model.py <data_path>")
    
    print("\nStarting Flask server...")
    print("API available at: http://localhost:5000")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)

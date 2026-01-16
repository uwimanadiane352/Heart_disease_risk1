from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'deployment/best_model.pkl'
FEATURE_COLUMNS_PATH = 'deployment/feature_columns.txt'
CLASS_NAMES_PATH = 'deployment/class_names.txt'


model = joblib.load(MODEL_PATH)

with open(FEATURE_COLUMNS_PATH, 'r') as f:
    feature_columns = [line.strip() for line in f.readlines()]

with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

@app.route('/', methods=['GET'])
def home():
    """API home endpoint with model information"""
    return jsonify({
        'status': 'running',
        'model': 'Heart Disease Risk Prediction Model',
        'version': '1.0',
        'endpoints': {
            '/': 'Model information (GET)',
            '/api/predict': 'Make prediction (POST)',
            '/api/features': 'Get feature list (GET)',
            '/api/classes': 'Get class names (GET)'
        },
        'features': feature_columns,
        'classes': class_names
    })

@app.route('/api/features', methods=['GET'])
def get_features():
    """Return list of required features"""
    return jsonify({
        'features': feature_columns,
        'count': len(feature_columns)
    })

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Return list of class names"""
    return jsonify({
        'classes': class_names,
        'count': len(class_names)
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction for patient data"""
    try:
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        
        missing_fields = [f for f in feature_columns if f not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        
        patient_data = pd.DataFrame([{col: data[col] for col in feature_columns}])
        
        
        if 'fbs' in patient_data.columns:
            patient_data['fbs'] = patient_data['fbs'].map({
                'True': True, 'False': False, True: True, False: False,
                'true': True, 'false': False, '1': True, '0': False
            })
        
        
        prediction = model.predict(patient_data)[0]
        probabilities = model.predict_proba(patient_data)[0]
        
        
        confidence = float(max(probabilities))
        
        
        prob_dict = {cls: float(prob) for cls, prob in zip(class_names, probabilities)}
        
        
        risk_colors = {
            'no disease': "#0be469",      
            'very mild': "#434139",       
            'mild': "#322f2c",           
            'severe': "#433c3b",         
            'immediate danger': "#716378" 
        }
        
        response = {
            'success': True,
            'prediction': {
                'class': prediction,
                'confidence': round(confidence * 100, 2),
                'color': risk_colors.get(prediction, '#95a5a6')
            },
            'probabilities': {cls: round(prob * 100, 2) for cls, prob in prob_dict.items()},
            'input_data': data
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("="*50)
    print("Heart Disease Risk Prediction API")
    print("="*50)
    print(f"Model loaded from: {MODEL_PATH}")
    print(f"Features: {len(feature_columns)}")
    print(f"Classes: {class_names}")
    print("="*50)
    app.run(debug=True, host='0.0.0.0', port=5000)
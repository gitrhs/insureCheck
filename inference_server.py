from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Load model and preprocessing objects
try:
    with open('model_results/best_model_xgboost.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('model_results/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    with open('model_results/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
        
    print("Model loaded successfully!")
    
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    print("Please ensure model files are in the 'model_results/' directory")
    model = None
    scaler = None
    label_encoders = None

# Feature columns (same order as training)
feature_cols = [
    'Age', 'Driving_License', 'Region_Code', 'Previously_Insured',
    'Vehicle_Age_Numeric', 'Annual_Premium', 'Policy_Sales_Channel', 
    'Vintage', 'Gender_Encoded', 'Vehicle_Damage_Encoded'
]

# Vehicle age mapping
vehicle_age_map = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}

def preprocess_data(data):
    """Preprocess input data for prediction"""
    df = pd.DataFrame([data])
    
    # Encode categorical variables
    df['Gender_Encoded'] = label_encoders['Gender'].transform([data['Gender']])[0]
    df['Vehicle_Damage_Encoded'] = label_encoders['Vehicle_Damage'].transform([data['Vehicle_Damage']])[0]
    
    # Map vehicle age
    df['Vehicle_Age_Numeric'] = vehicle_age_map[data['Vehicle_Age']]
    
    # Select features in correct order
    X = df[feature_cols]
    
    return X

@app.route('/predict', methods=['POST'])
def predict():
    """Make insurance prediction"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        
        # Preprocess data
        X = preprocess_data(data)
        
        # Make prediction
        probability = model.predict_proba(X)[0, 1]
        prediction = model.predict(X)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_percentage': round(probability * 100, 1)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5410, debug=True)
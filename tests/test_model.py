import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from crop_yield_dashboard import generate_synthetic_data, train_model

class TestCropYieldModel:
    """Test suite for the crop yield prediction model"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing"""
        return generate_synthetic_data(n_samples=100)
    
    def test_synthetic_data_generation(self):
        """Test if synthetic data is generated correctly"""
        data = generate_synthetic_data(n_samples=50)
        
        # Check data shape
        assert data.shape[0] == 50
        assert 'yield' in data.columns
        
        # Check data types
        assert data['temperature'].dtype == np.float64
        assert data['crop_type'].dtype == object
        assert data['irrigation'].dtype in [np.int64, np.int32]
        
        # Check value ranges
        assert data['temperature'].min() >= 10
        assert data['temperature'].max() <= 40
        assert data['soil_ph'].min() >= 4.5
        assert data['soil_ph'].max() <= 8.5
        assert data['yield'].min() >= 500
        assert data['yield'].max() <= 8000
    
    def test_model_training(self, sample_data):
        """Test if model trains successfully"""
        model, scaler, features, mse, r2, X_train, X_test = train_model(sample_data)
        
        # Check if model was trained
        assert model is not None
        assert scaler is not None
        assert len(features) > 0
        
        # Check model performance metrics
        assert mse >= 0
        assert 0 <= r2 <= 1
        
        # Check if scaler is fitted
        assert hasattr(scaler, 'mean_')
        assert hasattr(scaler, 'scale_')
    
    def test_model_prediction(self, sample_data):
        """Test if model can make predictions"""
        model, scaler, features, _, _, _, _ = train_model(sample_data)
        
        # Create test input
        test_input = pd.DataFrame({
            'temperature': [25.0],
            'rainfall': [100.0],
            'humidity': [65.0],
            'soil_ph': [6.5],
            'nitrogen': [40.0],
            'phosphorus': [30.0],
            'potassium': [35.0],
            'crop_type': ['wheat'],
            'field_size': [10.0],
            'irrigation': [1],
            'pesticide_use': [2.0]
        })
        
        # One-hot encode
        test_encoded = pd.get_dummies(test_input, columns=['crop_type'], prefix_sep='_')
        
        # Ensure all features are present
        for feature in features:
            if feature not in test_encoded.columns:
                test_encoded[feature] = 0
        
        test_encoded = test_encoded[features]
        
        # Scale and predict
        test_scaled = scaler.transform(test_encoded)
        prediction = model.predict(test_scaled)
        
        # Check prediction
        assert len(prediction) == 1
        assert 500 <= prediction[0] <= 8000  # Reasonable yield range
    
    def test_feature_importance(self, sample_data):
        """Test if model provides feature importance"""
        model, _, features, _, _, _, _ = train_model(sample_data)
        
        # Check feature importance
        importance = model.feature_importances_
        assert len(importance) == len(features)
        assert all(imp >= 0 for imp in importance)
        assert sum(importance) > 0

if __name__ == "__main__":
    pytest.main([__file__])
#!/usr/bin/env python3
"""
Script to generate sample crop yield data for testing and demonstration
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def generate_crop_yield_data(n_samples=1000, output_path='data/sample/sample_data.csv'):
    """
    Generate synthetic crop yield data and save to CSV
    
    Args:
        n_samples: Number of samples to generate
        output_path: Path to save the CSV file
    """
    np.random.seed(42)
    
    # Generate features
    data = pd.DataFrame({
        'temperature': np.random.normal(25, 5, n_samples),
        'rainfall': np.random.exponential(50, n_samples),
        'humidity': np.random.normal(65, 15, n_samples),
        'soil_ph': np.random.normal(6.5, 0.8, n_samples),
        'nitrogen': np.random.normal(40, 10, n_samples),
        'phosphorus': np.random.normal(30, 8, n_samples),
        'potassium': np.random.normal(35, 9, n_samples),
        'crop_type': np.random.choice(['wheat', 'corn', 'rice', 'soybeans'], n_samples),
        'field_size': np.random.uniform(1, 50, n_samples),
        'irrigation': np.random.choice([0, 1], n_samples),
        'pesticide_use': np.random.uniform(0, 5, n_samples),
        'latitude': np.random.uniform(20, 45, n_samples),
        'longitude': np.random.uniform(-120, -70, n_samples),
    })
    
    # Clip to realistic ranges
    data['temperature'] = np.clip(data['temperature'], 10, 40)
    data['rainfall'] = np.clip(data['rainfall'], 0, 300)
    data['humidity'] = np.clip(data['humidity'], 20, 95)
    data['soil_ph'] = np.clip(data['soil_ph'], 4.5, 8.5)
    data['nitrogen'] = np.clip(data['nitrogen'], 10, 80)
    data['phosphorus'] = np.clip(data['phosphorus'], 10, 60)
    data['potassium'] = np.clip(data['potassium'], 10, 70)
    
    # Calculate yield
    yield_base = 3000
    temp_effect = -20 * (data['temperature'] - 25)**2
    rain_effect = -0.5 * (data['rainfall'] - 100)**2
    nutrient_effect = (data['nitrogen'] * 5 + data['phosphorus'] * 4 + data['potassium'] * 3)
    ph_effect = -500 * (data['soil_ph'] - 6.5)**2
    
    crop_multipliers = {'wheat': 1.0, 'corn': 1.2, 'rice': 0.9, 'soybeans': 0.8}
    crop_effect = data['crop_type'].map(crop_multipliers)
    
    data['yield'] = (yield_base + temp_effect + rain_effect + nutrient_effect + ph_effect) * crop_effect
    data['yield'] *= (1 + data['irrigation'] * 0.2)
    data['yield'] *= (1 - data['pesticide_use'] * 0.02)
    data['yield'] += np.random.normal(0, 200, n_samples)
    data['yield'] = np.clip(data['yield'], 500, 8000)
    
    # Round numerical values
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].round(2)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    data.to_csv(output_path, index=False)
    print(f"Generated {n_samples} samples and saved to {output_path}")
    
    # Display summary statistics
    print("\nData Summary:")
    print(data.describe())
    
    return data

def main():
    parser = argparse.ArgumentParser(description='Generate sample crop yield data')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='data/sample/sample_data.csv', help='Output file path')
    
    args = parser.parse_args()
    generate_crop_yield_data(args.samples, args.output)

if __name__ == "__main__":
    main()
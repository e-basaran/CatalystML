import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import (load_and_preprocess_data, encode_categorical_variables,
                  create_chemical_features, create_surface_features,
                  create_interaction_features, split_by_reaction_type,
                  get_feature_columns)
from models import AdsorptionModel, SurfaceReactionModel, DesorptionModel, OtherReactionModel

def train_and_evaluate_models(data_path: str = 'data.csv'):
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(data_path)
    
    # Create all features
    df, _ = encode_categorical_variables(df)
    df = create_chemical_features(df)
    df = create_surface_features(df)
    df = create_interaction_features(df)
    
    # Split by reaction type
    reaction_datasets = split_by_reaction_type(df)
    
    # Get feature columns
    feature_columns = get_feature_columns()
    
    # Initialize models
    models = {
        'adsorption': AdsorptionModel(),
        'surface_reaction': SurfaceReactionModel(),
        'desorption': DesorptionModel(),
        'other': OtherReactionModel()
    }
    
    # Train and evaluate each model
    results = {}
    for reaction_type, model in models.items():
        if reaction_type not in reaction_datasets:
            continue
            
        print(f"\nTraining model for {reaction_type} reactions...")
        reaction_data = reaction_datasets[reaction_type]
        
        # Prepare features and target
        X = reaction_data[feature_columns]
        y = reaction_data['reaction_energy']
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        print(f"Dataset sizes for {reaction_type}:")
        print(f"Training: {len(X_train)}")
        print(f"Validation: {len(X_val)}")
        print(f"Test: {len(X_test)}")
        
        # Assign feature columns to the model (if necessary)
        model.feature_columns = feature_columns
        
        # Train model
        model.fit(X_train, y_train)
        
        # Retrieve feature names
        try:
            all_features = model.get_feature_names()
            print(f"Total Features after Polynomial Transformation: {len(all_features)}")
        except ValueError as e:
            print(f"Error retrieving feature names: {e}")
        
        # Make predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_metrics = model.calculate_metrics(y_train, train_pred, "Training")
        val_metrics = model.calculate_metrics(y_val, val_pred, "Validation")
        test_metrics = model.calculate_metrics(y_test, test_pred, "Test")
        
        # Analyze errors
        error_analysis = model.analyze_errors(X_test, y_test, test_pred, reaction_data)
        
        # Store results
        results[reaction_type] = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'error_analysis': error_analysis
        }
    
    return results

if __name__ == "__main__":
    print("Starting model training and evaluation...")
    results = train_and_evaluate_models()
    
    # Print summary of results
    print("\nSummary of Results:")
    for reaction_type, metrics in results.items():
        print(f"\n{reaction_type.upper()} REACTIONS:")
        print(f"Test R²: {metrics['test_metrics']['R2']:.4f}")
        print(f"Test RMSE: {metrics['test_metrics']['RMSE']:.4f}")
        print(f"Predictions within 0.5 eV: {metrics['test_metrics']['Within_0.5eV']:.1f}%")
        print(f"Predictions within 1.0 eV: {metrics['test_metrics']['Within_1.0eV']:.1f}%") 
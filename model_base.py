from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
from typing import Tuple, Dict, Any, List
import pandas as pd

class ModelBase:
    def __init__(self, name: str, model_params: Dict[str, Any] = None):
        self.name = name
        self.model_params = model_params or {}
        self.model = None
        self.preprocessor = None
        self.feature_columns = None
        self.pipeline = None
        
    def create_preprocessor(self, numeric_features, categorical_features):
        """Create the preprocessing pipeline."""
        poly = PolynomialFeatures(degree=2, include_bias=False)
        
        return ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler()),
                    ('poly', poly)
                ]), numeric_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value=-1))
                ]), categorical_features)
            ])
    
    def prepare_features(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Prepare feature lists for preprocessing."""
        numeric_features = [col for col in X.columns 
                          if col not in ['surface_encoded', 'ab_encoded', 'a_encoded', 'b_encoded',
                                       'has_asterisk_a', 'has_asterisk_b', 'has_asterisk_ab']]
        categorical_features = ['surface_encoded', 'ab_encoded', 'a_encoded', 'b_encoded',
                              'has_asterisk_a', 'has_asterisk_b', 'has_asterisk_ab']
        return numeric_features, categorical_features
    
    def create_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """Create the model pipeline."""
        numeric_features, categorical_features = self.prepare_features(X)
        self.feature_columns = numeric_features + categorical_features
        self.preprocessor = self.create_preprocessor(numeric_features, categorical_features)
        
        return Pipeline([
            ('preprocessor', self.preprocessor),
            ('model', self.model)
        ])
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str) -> Dict[str, float]:
        """Calculate and print comprehensive metrics."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        
        # Calculate custom metrics for reaction energy ranges
        errors = np.abs(y_true - y_pred)
        within_05 = np.mean(errors < 0.5) * 100
        within_1 = np.mean(errors < 1.0) * 100
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Within_0.5eV': within_05,
            'Within_1.0eV': within_1
        }
        
        print(f"\n{dataset_name} Metrics for {self.name}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        return metrics
    
    def analyze_errors(self, X: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
                      original_data: pd.DataFrame) -> pd.DataFrame:
        """Analyze prediction errors."""
        error_analysis = pd.DataFrame({
            'True_Energy': y_true,
            'Predicted_Energy': y_pred,
            'Absolute_Error': np.abs(y_true - y_pred),
            'Surface': original_data.loc[X.index, 'surface'],
            'Reactant_A': original_data.loc[X.index, 'a'],
            'Reactant_B': original_data.loc[X.index, 'b'],
            'Product': original_data.loc[X.index, 'ab']
        })
        
        print("\nWorst 10 Predictions:")
        worst_predictions = error_analysis.nlargest(10, 'Absolute_Error')
        print(worst_predictions[['Surface', 'Reactant_A', 'Reactant_B', 'Product', 
                               'True_Energy', 'Predicted_Energy', 'Absolute_Error']])
        
        print("\nPerformance by Surface Type:")
        surface_performance = error_analysis.groupby('Surface')['Absolute_Error'].agg(
            ['mean', 'std', 'count']).sort_values('mean', ascending=False)
        print(surface_performance.head())
        
        return error_analysis 
    
    def get_feature_names(self) -> List[str]:
        """Retrieve feature names after polynomial transformation."""
        if not self.pipeline:
            raise ValueError("Pipeline has not been fitted yet.")

        # Access the preprocessor from the pipeline
        preprocessor = self.pipeline.named_steps['preprocessor']

        # Access the polynomial features from the preprocessor
        poly = preprocessor.named_transformers_['num'].named_steps['poly']

        # Get the original numeric feature names
        numeric_features = [col for col in self.feature_columns 
                            if col not in ['surface_encoded', 'ab_encoded', 'a_encoded', 'b_encoded',
                                          'has_asterisk_a', 'has_asterisk_b', 'has_asterisk_ab']]

        # Retrieve feature names from the fitted PolynomialFeatures
        numeric_features_poly = poly.get_feature_names_out(numeric_features)

        # Get categorical features
        categorical_features = ['surface_encoded', 'ab_encoded', 'a_encoded', 'b_encoded',
                                'has_asterisk_a', 'has_asterisk_b', 'has_asterisk_ab']

        # Combine all feature names
        all_features = list(numeric_features_poly) + categorical_features

        return all_features
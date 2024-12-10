from model_base import ModelBase
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
from typing import Dict, Any, Tuple
import pandas as pd

class AdsorptionModel(ModelBase):
    def __init__(self, model_params: Dict[str, Any] = None):
        super().__init__("Adsorption Model", model_params)
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=4,
            max_features='sqrt',
            learning_rate=0.08,
            random_state=42,
            **(model_params or {})
        )
        self.pipeline = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model with adsorption-specific preprocessing."""
        self.pipeline = self.create_pipeline(X)
        self.pipeline.fit(X, y)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the fitted pipeline."""
        return self.pipeline.predict(X)

class SurfaceReactionModel(ModelBase):
    def __init__(self, model_params: Dict[str, Any] = None):
        super().__init__("Surface Reaction Model", model_params)
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            **(model_params or {})
        )
        self.pipeline = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model with surface reaction-specific preprocessing."""
        self.pipeline = self.create_pipeline(X)
        self.pipeline.fit(X, y)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the fitted pipeline."""
        return self.pipeline.predict(X)

class DesorptionModel(ModelBase):
    def __init__(self, model_params: Dict[str, Any] = None):
        super().__init__("Desorption Model", model_params)
        self.model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=5,
            min_samples_split=4,
            min_samples_leaf=3,
            max_features='sqrt',
            learning_rate=0.1,
            random_state=42,
            **(model_params or {})
        )
        self.pipeline = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model with desorption-specific preprocessing."""
        self.pipeline = self.create_pipeline(X)
        self.pipeline.fit(X, y)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the fitted pipeline."""
        return self.pipeline.predict(X)

class OtherReactionModel(ModelBase):
    def __init__(self, model_params: Dict[str, Any] = None):
        super().__init__("Other Reactions Model", model_params)
        self.model = RandomForestRegressor(
            n_estimators=150,
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=3,
            max_features='sqrt',
            random_state=42,
            **(model_params or {})
        )
        self.pipeline = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model with general reaction preprocessing."""
        self.pipeline = self.create_pipeline(X)
        self.pipeline.fit(X, y)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the fitted pipeline."""
        return self.pipeline.predict(X) 
"""Customer churn prediction model."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class ChurnPredictor:
    """Model for predicting customer churn."""
    
    def __init__(self, random_state: int = 42):
        """Initialize churn predictor.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = RandomForestClassifier(random_state=random_state)
        self.is_trained = False
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """Train the churn prediction model.
        
        Args:
            X: Feature matrix
            y: Target variable (churn label)
            test_size: Test set size
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        return self.model.score(X_test, y_test)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make churn predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted churn probability
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict_proba(X)[:, 1]

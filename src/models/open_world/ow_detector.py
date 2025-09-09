"""
Open world detector for identifying unknown websites
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class OpenWorldDetector(BaseEstimator, ClassifierMixin):
    """Detector for open-world website fingerprinting"""
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.is_fitted = False
        
    def fit(self, X, y):
        """Fit the open world detector"""
        self.classes_ = np.unique(y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict known vs unknown websites"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Simulate open world detection
        predictions = []
        for _ in range(len(X)):
            if np.random.random() > self.threshold:
                predictions.append(np.random.choice(self.classes_))
            else:
                predictions.append("UNKNOWN")
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Predict probabilities for known classes"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        n_samples = len(X)
        n_classes = len(self.classes_) + 1  # +1 for UNKNOWN class
        probs = np.random.dirichlet(np.ones(n_classes), size=n_samples)
        return probs

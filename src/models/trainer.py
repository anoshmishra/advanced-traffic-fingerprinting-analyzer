"""
Machine learning model training and evaluation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import logging
from pathlib import Path
import warnings

class ModelTrainer:
    """Machine learning model trainer"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.results = {}
        self.label_encoder = None
    
    def get_classifier(self, name, params):
        """Get classifier instance"""
        classifiers = {
            'RandomForest': RandomForestClassifier,
            'SVM': SVC,
            'KNN': KNeighborsClassifier,
            'XGBoost': xgb.XGBClassifier
        }
        
        if name not in classifiers:
            raise ValueError(f"Unknown classifier: {name}")
        
        return classifiers[name](**params)
    
    def encode_labels(self, y_train, y_val, y_test):
        """Encode string labels to integers for XGBoost"""
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        return y_train_encoded, y_val_encoded, y_test_encoded
    
    def split_data(self, X, y):
        """Split data into train, validation, and test sets"""
        test_size = self.config['models']['test_size']
        val_size = self.config['models']['validation_size']
        random_state = self.config['models']['random_state']
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_single_model(self, classifier_config, X_train, y_train, X_val, y_val):
        """Train a single classifier"""
        name = classifier_config['name']
        params = classifier_config['params']
        
        self.logger.info(f"Training {name} classifier")
        
        # Handle XGBoost label encoding
        if name == 'XGBoost':
            # Create a temporary label encoder for this model
            temp_encoder = LabelEncoder()
            y_train_model = temp_encoder.fit_transform(y_train)
            y_val_model = temp_encoder.transform(y_val)
        else:
            y_train_model = y_train
            y_val_model = y_val
            temp_encoder = None
        
        # Get classifier
        clf = self.get_classifier(name, params)
        
        # Train model
        clf.fit(X_train, y_train_model)
        
        # Validate
        y_val_pred = clf.predict(X_val)
        
        # Convert predictions back to original labels if needed
        if temp_encoder is not None:
            y_val_pred = temp_encoder.inverse_transform(y_val_pred)
            y_val_model = temp_encoder.inverse_transform(y_val_model)
        
        val_accuracy = accuracy_score(y_val_model, y_val_pred)
        
        # Cross-validation on training data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_scores = cross_val_score(clf, X_train, y_train_model, 
                                       cv=self.config['models']['cv_folds'])
        
        results = {
            'model': clf,
            'validation_accuracy': val_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores,
            'label_encoder': temp_encoder  # Store encoder if used
        }
        
        self.logger.info(f"{name} - Val Accuracy: {val_accuracy:.4f}, "
                        f"CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return results
    
    def evaluate_model(self, model_results, X_test, y_test):
        """Evaluate model on test set"""
        model = model_results['model']
        label_encoder = model_results.get('label_encoder')
        
        # Handle label encoding for XGBoost
        if label_encoder is not None:
            y_test_encoded = label_encoder.transform(y_test)
            y_pred_encoded = model.predict(X_test)
            y_pred = label_encoder.inverse_transform(y_pred_encoded)
            y_test_eval = y_test
        else:
            y_pred = model.predict(X_test)
            y_test_eval = y_test
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            accuracy = accuracy_score(y_test_eval, y_pred)
            report = classification_report(y_test_eval, y_pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_test_eval, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'true_labels': y_test_eval
        }
    
    def hyperparameter_tuning(self, classifier_config, X_train, y_train):
        """Perform hyperparameter tuning"""
        name = classifier_config['name']
        base_params = classifier_config['params']
        
        # Define parameter grids
        param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            }
        }
        
        if name not in param_grids:
            # No tuning for this classifier
            return self.get_classifier(name, base_params)
        
        self.logger.info(f"Hyperparameter tuning for {name}")
        
        # Handle XGBoost label encoding
        if name == 'XGBoost':
            temp_encoder = LabelEncoder()
            y_train_encoded = temp_encoder.fit_transform(y_train)
        else:
            y_train_encoded = y_train
        
        # Create base classifier
        clf = self.get_classifier(name, base_params)
        
        # Grid search
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid_search = GridSearchCV(
                clf, param_grids[name], cv=3, scoring='accuracy',
                n_jobs=-1, verbose=0  # Reduced verbosity
            )
            
            grid_search.fit(X_train, y_train_encoded)
        
        self.logger.info(f"Best parameters for {name}: {grid_search.best_params_}")
        self.logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_models(self, X, y, prefix="", tune_hyperparameters=False):
        """Train all configured models"""
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        self.logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        self.logger.info(f"Classes: {sorted(y.unique())}")
        
        results = {
            'data_split': {
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test),
                'feature_count': X.shape[1],
                'class_count': y.nunique()
            },
            'models': {}
        }
        
        # Train each configured classifier
        for classifier_config in self.config['models']['classifiers']:
            name = classifier_config['name']
            
            try:
                if tune_hyperparameters:
                    # Use hyperparameter tuning
                    best_model = self.hyperparameter_tuning(classifier_config, X_train, y_train)
                    
                    # Create model results structure
                    if name == 'XGBoost':
                        temp_encoder = LabelEncoder()
                        y_val_encoded = temp_encoder.fit_transform(y_val)
                        y_val_pred = best_model.predict(X_val)
                        y_val_pred = temp_encoder.inverse_transform(y_val_pred)
                        val_accuracy = accuracy_score(y_val, y_val_pred)
                        label_encoder = temp_encoder
                    else:
                        y_val_pred = best_model.predict(X_val)
                        val_accuracy = accuracy_score(y_val, y_val_pred)
                        label_encoder = None
                    
                    model_results = {
                        'model': best_model,
                        'validation_accuracy': val_accuracy,
                        'tuned': True,
                        'label_encoder': label_encoder
                    }
                else:
                    # Use default parameters
                    model_results = self.train_single_model(
                        classifier_config, X_train, y_train, X_val, y_val
                    )
                
                # Test evaluation
                test_results = self.evaluate_model(model_results, X_test, y_test)
                model_results.update(test_results)
                
                results['models'][name] = model_results
                
                # Save model
                model_dir = Path(self.config['paths']['results']) / 'models'
                model_dir.mkdir(parents=True, exist_ok=True)
                
                model_file = model_dir / f"{prefix}_{name}_model.joblib"
                joblib.dump({
                    'model': model_results['model'],
                    'label_encoder': model_results.get('label_encoder')
                }, model_file)
                
                self.logger.info(f"{name} - Test Accuracy: {test_results['accuracy']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {e}")
                continue
        
        # Save results
        results_dir = Path(self.config['paths']['results']) / 'reports'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / f"{prefix}_training_results.joblib"
        joblib.dump(results, results_file)
        
        return results
    
    def load_model(self, model_file):
        """Load a saved model"""
        return joblib.load(model_file)
    
    def predict(self, model_data, X):
        """Make predictions"""
        model = model_data['model'] if isinstance(model_data, dict) else model_data
        label_encoder = model_data.get('label_encoder') if isinstance(model_data, dict) else None
        
        predictions = model.predict(X)
        
        if label_encoder is not None:
            predictions = label_encoder.inverse_transform(predictions)
        
        return predictions

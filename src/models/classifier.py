"""
Enhanced classifier utilities and wrapper classes
"""

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Ensemble classifier combining multiple base classifiers"""

    def __init__(self, classifiers=None, voting='hard', weights=None):
        self.classifiers = classifiers or self._get_default_classifiers()
        self.voting = voting
        self.weights = weights
        self.fitted_classifiers_ = []
        self.classes_ = None

    def _get_default_classifiers(self):
        """Get default set of classifiers"""
        return [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
            ('xgb', xgb.XGBClassifier(random_state=42, eval_metric='logloss')),
            ('nb', GaussianNB()),
            ('lr', LogisticRegression(random_state=42, max_iter=1000))
        ]

    def fit(self, X, y):
        """Fit all base classifiers"""
        self.classes_ = np.unique(y)
        self.fitted_classifiers_ = []

        for name, clf in self.classifiers:
            try:
                logger.info(f"Training {name} classifier")
                clf.fit(X, y)
                self.fitted_classifiers_.append((name, clf))
            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")
                continue

        return self

    def predict(self, X):
        """Make predictions using ensemble voting"""
        if self.voting == 'hard':
            return self._predict_hard_voting(X)
        else:
            return self._predict_soft_voting(X)

    def _predict_hard_voting(self, X):
        """Hard voting prediction"""
        predictions = np.array([clf.predict(X) for _, clf in self.fitted_classifiers_])

        # Majority voting
        final_predictions = []
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            unique, counts = np.unique(votes, return_counts=True)
            winner = unique[np.argmax(counts)]
            final_predictions.append(winner)

        return np.array(final_predictions)

    def _predict_soft_voting(self, X):
        """Soft voting prediction using probabilities"""
        probabilities = np.array([clf.predict_proba(X) for _, clf in self.fitted_classifiers_])

        if self.weights is not None:
            weighted_probs = np.average(probabilities, axis=0, weights=self.weights)
        else:
            weighted_probs = np.mean(probabilities, axis=0)

        return self.classes_[np.argmax(weighted_probs, axis=1)]

    def predict_proba(self, X):
        """Predict class probabilities"""
        probabilities = np.array([clf.predict_proba(X) for _, clf in self.fitted_classifiers_])

        if self.weights is not None:
            return np.average(probabilities, axis=0, weights=self.weights)
        else:
            return np.mean(probabilities, axis=0)


class AutoMLClassifier(BaseEstimator, ClassifierMixin):
    """Automated ML classifier with hyperparameter tuning"""

    def __init__(self, classifier_type='auto', cv_folds=5, scoring='accuracy'):
        self.classifier_type = classifier_type
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.best_classifier_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.scaler_ = StandardScaler()

    def fit(self, X, y):
        """Fit with automatic classifier selection and hyperparameter tuning"""
        # Scale features
        X_scaled = self.scaler_.fit_transform(X)

        if self.classifier_type == 'auto':
            self.best_classifier_, self.best_params_, self.best_score_ = self._auto_select_classifier(X_scaled, y)
        else:
            self.best_classifier_, self.best_params_, self.best_score_ = self._tune_classifier(
                self.classifier_type, X_scaled, y
            )

        # Fit the best classifier
        self.best_classifier_.set_params(**self.best_params_)
        self.best_classifier_.fit(X_scaled, y)

        logger.info(f"Best classifier: {type(self.best_classifier_).__name__}")
        logger.info(f"Best parameters: {self.best_params_}")
        logger.info(f"Best CV score: {self.best_score_:.4f}")

        return self

    def _auto_select_classifier(self, X, y):
        """Automatically select the best classifier"""
        classifiers = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'GradientBoosting': GradientBoostingClassifier(random_state=42)
        }

        best_classifier = None
        best_params = {}
        best_score = -np.inf

        for name, clf in classifiers.items():
            try:
                clf_best, params, score = self._tune_classifier(clf, X, y)
                if score > best_score:
                    best_score = score
                    best_classifier = clf_best
                    best_params = params
                logger.info(f"{name} CV score: {score:.4f}")

            except Exception as e:
                logger.warning(f"Failed to tune {name}: {e}")
                continue

        return best_classifier, best_params, best_score

    def _tune_classifier(self, classifier, X, y):
        """Tune hyperparameters for a specific classifier"""
        param_grids = self._get_param_grids()

        clf_name = type(classifier).__name__
        if clf_name in param_grids:
            param_grid = param_grids[clf_name]

            grid_search = GridSearchCV(
                classifier, param_grid, cv=self.cv_folds, scoring=self.scoring, n_jobs=-1
            )

            grid_search.fit(X, y)
            return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
        else:
            # No parameter grid available → use default parameters
            classifier.fit(X, y)
            return classifier, {}, 0.0

    def _get_param_grids(self):
        """Get parameter grids for different classifiers"""
        return {
            'RandomForestClassifier': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'SVC': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            },
            'XGBClassifier': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'subsample': [0.8, 0.9, 1.0]
            },
            'LogisticRegression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga']
            },
            'GradientBoostingClassifier': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            }
        }

    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler_.transform(X)
        return self.best_classifier_.predict(X_scaled)

    def predict_proba(self, X):
        """Predict class probabilities"""
        X_scaled = self.scaler_.transform(X)
        return self.best_classifier_.predict_proba(X_scaled)


# Factory functions
CLASSIFIER_REGISTRY = {
    'RandomForest': RandomForestClassifier,
    'SVM': SVC,
    'KNN': KNeighborsClassifier,
    'XGBoost': xgb.XGBClassifier,
    'LightGBM': lgb.LGBMClassifier,
    'LogisticRegression': LogisticRegression,
    'NaiveBayes': GaussianNB,
    'DecisionTree': DecisionTreeClassifier,
    'GradientBoosting': GradientBoostingClassifier,
    'AdaBoost': AdaBoostClassifier,
    'MLP': MLPClassifier,
    'LDA': LinearDiscriminantAnalysis,
    'QDA': QuadraticDiscriminantAnalysis,
    'Ensemble': EnsembleClassifier,
    'AutoML': AutoMLClassifier
}


def get_classifier(name, **kwargs):
    """Get classifier instance by name"""
    if name not in CLASSIFIER_REGISTRY:
        raise ValueError(f"Unknown classifier: {name}. Available: {list(CLASSIFIER_REGISTRY.keys())}")

    return CLASSIFIER_REGISTRY[name](**kwargs)


def list_available_classifiers():
    """List all available classifiers"""
    return list(CLASSIFIER_REGISTRY.keys())


def get_default_params(classifier_name):
    """Get default parameters for a classifier"""
    default_params = {
        'RandomForest': {'n_estimators': 100, 'random_state': 42},
        'SVM': {'kernel': 'rbf', 'probability': True, 'random_state': 42},
        'KNN': {'n_neighbors': 5},
        'XGBoost': {'random_state': 42, 'eval_metric': 'logloss'},
        'LogisticRegression': {'random_state': 42, 'max_iter': 1000},
        'DecisionTree': {'random_state': 42},
        'GradientBoosting': {'random_state': 42},
        'AdaBoost': {'random_state': 42},
        'MLP': {'random_state': 42, 'max_iter': 1000}
    }

    return default_params.get(classifier_name, {})

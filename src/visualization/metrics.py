"""
Advanced metrics calculation and visualization utilities
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any, Optional
import logging
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

class ClassificationMetrics:
    """Comprehensive classification metrics calculator"""
    
    def __init__(self, y_true: List, y_pred: List, y_prob: Optional[np.ndarray] = None, 
                 class_names: Optional[List[str]] = None):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_prob = y_prob
        self.class_names = class_names or sorted(list(set(y_true)))
        self.n_classes = len(self.class_names)
        
    def compute_basic_metrics(self) -> Dict[str, float]:
        """Compute basic classification metrics"""
        return {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'precision_macro': precision_score(self.y_true, self.y_pred, average='macro', zero_division=0),
            'precision_micro': precision_score(self.y_true, self.y_pred, average='micro', zero_division=0),
            'precision_weighted': precision_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(self.y_true, self.y_pred, average='macro', zero_division=0),
            'recall_micro': recall_score(self.y_true, self.y_pred, average='micro', zero_division=0),
            'recall_weighted': recall_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(self.y_true, self.y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(self.y_true, self.y_pred, average='micro', zero_division=0),
            'f1_weighted': f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        }
    
    def compute_per_class_metrics(self) -> pd.DataFrame:
        """Compute per-class metrics"""
        precision_per_class = precision_score(self.y_true, self.y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(self.y_true, self.y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(self.y_true, self.y_pred, average=None, zero_division=0)
        
        # Support (number of true instances for each class)
        support = np.bincount(self.y_true, minlength=len(self.class_names))
        
        metrics_df = pd.DataFrame({
            'Class': self.class_names,
            'Precision': precision_per_class,
            'Recall': recall_per_class,
            'F1-Score': f1_per_class,
            'Support': support
        })
        
        return metrics_df
    
    def compute_confusion_matrix(self, normalize: Optional[str] = None) -> np.ndarray:
        """Compute confusion matrix"""
        cm = confusion_matrix(self.y_true, self.y_pred, labels=self.class_names)
        
        if normalize == 'true':
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        elif normalize == 'pred':
            cm = cm.astype('float') / cm.sum(axis=0)
        elif normalize == 'all':
            cm = cm.astype('float') / cm.sum()
        
        return cm
    
    def compute_auc_scores(self) -> Dict[str, float]:
        """Compute AUC scores (requires probability predictions)"""
        if self.y_prob is None:
            logger.warning("Probability predictions not provided, cannot compute AUC scores")
            return {}
        
        try:
            if self.n_classes == 2:
                # Binary classification
                auc_score = roc_auc_score(self.y_true, self.y_prob[:, 1])
                return {'auc_roc': auc_score}
            else:
                # Multi-class classification
                y_true_binarized = label_binarize(self.y_true, classes=self.class_names)
                auc_macro = roc_auc_score(y_true_binarized, self.y_prob, average='macro', multi_class='ovr')
                auc_weighted = roc_auc_score(y_true_binarized, self.y_prob, average='weighted', multi_class='ovr')
                
                return {
                    'auc_roc_macro': auc_macro,
                    'auc_roc_weighted': auc_weighted
                }
        except Exception as e:
            logger.error(f"Error computing AUC scores: {e}")
            return {}
    
    def compute_classification_report(self) -> Dict:
        """Compute detailed classification report"""
        return classification_report(self.y_true, self.y_pred, 
                                   target_names=self.class_names, 
                                   output_dict=True, zero_division=0)
    
    def compute_all_metrics(self) -> Dict[str, Any]:
        """Compute all available metrics"""
        metrics = {
            'basic_metrics': self.compute_basic_metrics(),
            'per_class_metrics': self.compute_per_class_metrics(),
            'confusion_matrix': self.compute_confusion_matrix(),
            'confusion_matrix_normalized': self.compute_confusion_matrix(normalize='true'),
            'classification_report': self.compute_classification_report()
        }
        
        # Add AUC scores if probabilities are available
        auc_scores = self.compute_auc_scores()
        if auc_scores:
            metrics['auc_scores'] = auc_scores
        
        return metrics

class DefenseEvaluationMetrics:
    """Metrics for evaluating defense effectiveness"""
    
    def __init__(self, baseline_results: Dict, defended_results: Dict):
        self.baseline_results = baseline_results
        self.defended_results = defended_results
        
    def compute_defense_effectiveness(self) -> Dict[str, Any]:
        """Compute defense effectiveness metrics"""
        effectiveness = {}
        
        for model_name in self.baseline_results.keys():
            if model_name in self.defended_results:
                baseline_acc = self.baseline_results[model_name].get('accuracy', 0)
                defended_acc = self.defended_results[model_name].get('accuracy', 0)
                
                # Accuracy degradation
                acc_degradation = baseline_acc - defended_acc
                acc_degradation_pct = (acc_degradation / baseline_acc * 100) if baseline_acc > 0 else 0
                
                # Defense success rate (higher degradation = better defense)
                defense_success = min(acc_degradation_pct / 50.0, 1.0)  # Normalize to 0-1
                
                effectiveness[model_name] = {
                    'baseline_accuracy': baseline_acc,
                    'defended_accuracy': defended_acc,
                    'accuracy_degradation': acc_degradation,
                    'accuracy_degradation_percent': acc_degradation_pct,
                    'defense_success_rate': defense_success
                }
        
        # Overall defense effectiveness
        if effectiveness:
            avg_degradation = np.mean([v['accuracy_degradation_percent'] for v in effectiveness.values()])
            avg_success_rate = np.mean([v['defense_success_rate'] for v in effectiveness.values()])
            
            effectiveness['overall'] = {
                'average_degradation_percent': avg_degradation,
                'average_success_rate': avg_success_rate,
                'defense_rating': self._get_defense_rating(avg_degradation)
            }
        
        return effectiveness
    
    def _get_defense_rating(self, avg_degradation: float) -> str:
        """Get qualitative defense rating"""
        if avg_degradation >= 40:
            return "Excellent"
        elif avg_degradation >= 30:
            return "Good"
        elif avg_degradation >= 20:
            return "Fair"
        elif avg_degradation >= 10:
            return "Poor"
        else:
            return "Ineffective"
    
    def compute_privacy_utility_tradeoff(self, bandwidth_overhead: float = 0.0, 
                                       latency_overhead: float = 0.0) -> Dict[str, float]:
        """Compute privacy-utility tradeoff metrics"""
        effectiveness = self.compute_defense_effectiveness()
        
        if 'overall' not in effectiveness:
            return {}
        
        privacy_gain = effectiveness['overall']['average_degradation_percent']
        utility_cost = bandwidth_overhead + latency_overhead  # Simplified combination
        
        # Privacy-utility ratio (higher is better)
        privacy_utility_ratio = privacy_gain / max(utility_cost, 1.0)
        
        return {
            'privacy_gain': privacy_gain,
            'utility_cost': utility_cost,
            'privacy_utility_ratio': privacy_utility_ratio,
            'bandwidth_overhead_percent': bandwidth_overhead,
            'latency_overhead_percent': latency_overhead
        }

class StatisticalTests:
    """Statistical significance tests for model comparisons"""
    
    @staticmethod
    def mcnemar_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> Dict[str, float]:
        """McNemar's test for comparing two classifiers"""
        # Create contingency table
        correct1 = (y_true == y_pred1)
        correct2 = (y_true == y_pred2)
        
        # McNemar table
        n01 = np.sum(~correct1 & correct2)  # Model 1 wrong, Model 2 correct
        n10 = np.sum(correct1 & ~correct2)  # Model 1 correct, Model 2 wrong
        
        # McNemar test statistic
        if n01 + n10 == 0:
            return {'statistic': 0.0, 'p_value': 1.0}
        
        chi2_stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        p_value = 1 - stats.chi2.cdf(chi2_stat, 1)
        
        return {
            'statistic': chi2_stat,
            'p_value': p_value,
            'n01': n01,
            'n10': n10
        }
    
    @staticmethod
    def paired_t_test(scores1: List[float], scores2: List[float]) -> Dict[str, float]:
        """Paired t-test for comparing cross-validation scores"""
        if len(scores1) != len(scores2):
            raise ValueError("Score lists must have same length")
        
        statistic, p_value = stats.ttest_rel(scores1, scores2)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'mean_diff': np.mean(scores1) - np.mean(scores2),
            'effect_size': (np.mean(scores1) - np.mean(scores2)) / np.std(np.array(scores1) - np.array(scores2))
        }
    
    @staticmethod
    def wilcoxon_signed_rank_test(scores1: List[float], scores2: List[float]) -> Dict[str, float]:
        """Wilcoxon signed-rank test (non-parametric alternative to paired t-test)"""
        statistic, p_value = stats.wilcoxon(scores1, scores2)
        
        return {
            'statistic': statistic,
            'p_value': p_value
        }

class VisualizationMetrics:
    """Metrics and utilities for creating visualizations"""
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], normalize: bool = False,
                            figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """Plot confusion matrix heatmap"""
        fig, ax = plt.subplots(figsize=figsize)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_roc_curves(y_true_list: List[np.ndarray], y_prob_list: List[np.ndarray],
                       model_names: List[str], figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """Plot ROC curves for multiple models"""
        fig, ax = plt.subplots(figsize=figsize)
        
        for y_true, y_prob, name in zip(y_true_list, y_prob_list, model_names):
            if len(np.unique(y_true)) == 2:  # Binary classification
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_precision_recall_curves(y_true_list: List[np.ndarray], y_prob_list: List[np.ndarray],
                                    model_names: List[str], figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """Plot Precision-Recall curves for multiple models"""
        fig, ax = plt.subplots(figsize=figsize)
        
        for y_true, y_prob, name in zip(y_true_list, y_prob_list, model_names):
            if len(np.unique(y_true)) == 2:  # Binary classification
                precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
                pr_auc = auc(recall, precision)
                ax.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.2f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_learning_curves(train_sizes: np.ndarray, train_scores: np.ndarray, 
                           val_scores: np.ndarray, model_name: str,
                           figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """Plot learning curves"""
        fig, ax = plt.subplots(figsize=figsize)
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        ax.plot(train_sizes, train_mean, 'o-', label='Training Score', color='blue', alpha=0.8)
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
        
        ax.plot(train_sizes, val_mean, 'o-', label='Validation Score', color='red', alpha=0.8)
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Accuracy Score')
        ax.set_title(f'Learning Curves - {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# Utility functions
def calculate_accuracy(y_true: List, y_pred: List) -> float:
    """Calculate accuracy score"""
    return accuracy_score(y_true, y_pred)

def calculate_precision_recall_f1(y_true: List, y_pred: List, average: str = 'macro') -> Tuple[float, float, float]:
    """Calculate precision, recall, and F1 score"""
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    return precision, recall, f1

def compare_model_performance(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """Compare performance of multiple models"""
    comparison_data = []
    
    for model_name, results in results_dict.items():
        row = {
            'Model': model_name,
            'Accuracy': results.get('accuracy', 0),
            'Precision': results.get('precision_macro', 0),
            'Recall': results.get('recall_macro', 0),
            'F1-Score': results.get('f1_macro', 0)
        }
        
        if 'auc_roc_macro' in results:
            row['AUC-ROC'] = results['auc_roc_macro']
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def calculate_defense_impact(baseline_acc: float, defended_acc: float) -> Dict[str, float]:
    """Calculate defense impact metrics"""
    degradation = baseline_acc - defended_acc
    degradation_pct = (degradation / baseline_acc * 100) if baseline_acc > 0 else 0
    
    return {
        'accuracy_degradation': degradation,
        'degradation_percent': degradation_pct,
        'defense_effectiveness': min(degradation_pct / 30.0, 1.0)  # Normalize to 0-1
    }

"""
Visualization and plotting utilities
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import logging
from pathlib import Path

class Visualizer:
    """Visualization and plotting class"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_confusion_matrix(self, y_true, y_pred, labels, title="Confusion Matrix"):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_model_comparison(self, results):
        """Plot comparison of model performances"""
        models = list(results['models'].keys())
        accuracies = [results['models'][model]['accuracy'] for model in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, accuracies, color='skyblue', alpha=0.7)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.title('Model Performance Comparison')
        plt.xlabel('Model')
        plt.ylabel('Test Accuracy')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_feature_importance(self, model, feature_names, top_k=20):
        """Plot feature importance for tree-based models"""
        if not hasattr(model, 'feature_importances_'):
            self.logger.warning("Model doesn't have feature_importances_ attribute")
            return None
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_k]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top {top_k} Feature Importances')
        
        # Create horizontal bar plot
        y_pos = np.arange(len(indices))
        plt.barh(y_pos, importances[indices], alpha=0.7)
        
        # Add feature names
        plt.yticks(y_pos, [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()  # Highest importance at top
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_roc_curves(self, results, class_labels):
        """Plot ROC curves for multi-class classification"""
        plt.figure(figsize=(12, 8))
        
        for model_name, model_results in results['models'].items():
            y_true = model_results['true_labels']
            y_pred = model_results['predictions']
            
            # For multi-class ROC, we'll use one-vs-rest approach
            # This is a simplified version - in practice you'd compute for each class
            try:
                fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=class_labels[0])
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
            except:
                self.logger.warning(f"Could not compute ROC for {model_name}")
                continue
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_defense_effectiveness(self, baseline_results, defended_results):
        """Plot effectiveness of defenses"""
        models = list(baseline_results['models'].keys())
        
        baseline_acc = [baseline_results['models'][model]['accuracy'] for model in models]
        defended_acc = [defended_results['models'][model]['accuracy'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        
        plt.bar(x - width/2, baseline_acc, width, label='Baseline', alpha=0.7)
        plt.bar(x + width/2, defended_acc, width, label='With Defense', alpha=0.7)
        
        # Add value labels
        for i, (baseline, defended) in enumerate(zip(baseline_acc, defended_acc)):
            plt.text(i - width/2, baseline + 0.01, f'{baseline:.3f}', ha='center', va='bottom')
            plt.text(i + width/2, defended + 0.01, f'{defended:.3f}', ha='center', va='bottom')
            
            # Add degradation percentage
            degradation = (baseline - defended) / baseline * 100 if baseline > 0 else 0
            plt.text(i, max(baseline, defended) + 0.05, f'-{degradation:.1f}%', 
                    ha='center', va='bottom', color='red', fontweight='bold')
        
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title('Defense Effectiveness: Accuracy Before and After Defense')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_traffic_features_distribution(self, X, y, features_to_plot=None):
        """Plot distribution of traffic features by class"""
        if features_to_plot is None:
            # Select a subset of interesting features
            numeric_features = X.select_dtypes(include=[np.number]).columns
            features_to_plot = numeric_features[:9]  # Plot first 9 features
        
        n_features = len(features_to_plot)
        n_cols = 3
        n_rows = int(np.ceil(n_features / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        unique_labels = sorted(y.unique())
        
        for i, feature in enumerate(features_to_plot):
            ax = axes[i]
            
            for label in unique_labels:
                mask = y == label
                ax.hist(X.loc[mask, feature], alpha=0.7, label=label, bins=20)
            
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {feature}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_features, len(axes)):
            axes[i].hide()
        
        plt.tight_layout()
        return fig
    
    def save_plots(self, plots, prefix=""):
        """Save plots to results directory"""
        plot_dir = Path(self.config['paths']['results']) / 'plots'
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        for name, fig in plots.items():
            if fig is not None:
                filename = f"{prefix}_{name}.png" if prefix else f"{name}.png"
                filepath = plot_dir / filename
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved plot: {filepath}")
        
        plt.close('all')  # Close all figures to free memory
    
    def compare_results(self, baseline_results, defended_results):
        """Generate comprehensive comparison visualizations"""
        plots = {}
        
        # Model comparison plots
        plots['baseline_comparison'] = self.plot_model_comparison(baseline_results)
        plots['defended_comparison'] = self.plot_model_comparison(defended_results)
        plots['defense_effectiveness'] = self.plot_defense_effectiveness(
            baseline_results, defended_results)
        
        # Confusion matrices for best model
        best_model_name = max(baseline_results['models'].keys(),
                            key=lambda x: baseline_results['models'][x]['accuracy'])
        
        baseline_model = baseline_results['models'][best_model_name]
        defended_model = defended_results['models'][best_model_name]
        
        unique_labels = sorted(set(baseline_model['true_labels']))
        
        plots['baseline_confusion'] = self.plot_confusion_matrix(
            baseline_model['true_labels'], baseline_model['predictions'],
            unique_labels, f"Baseline {best_model_name} - Confusion Matrix"
        )
        
        plots['defended_confusion'] = self.plot_confusion_matrix(
            defended_model['true_labels'], defended_model['predictions'],
            unique_labels, f"Defended {best_model_name} - Confusion Matrix"
        )
        
        # Save all plots
        self.save_plots(plots, "comparison")
        
        self.logger.info("Visualization comparison completed")

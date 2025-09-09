"""
Visualization panel widget for traffic analysis plots
"""

import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import seaborn as sns

class VisualizationPanel:
    """Panel for visualization dashboard"""
    
    def __init__(self, parent, config):
        self.parent = parent
        self.config = config
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the visualization panel UI"""
        # Main container
        main_frame = ctk.CTkScrollableFrame(self.parent)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        title_label = ctk.CTkLabel(main_frame, text="📊 Visualization Dashboard", 
                                  font=ctk.CTkFont(size=18, weight="bold"))
        title_label.pack(pady=10)
        
        # Control panel
        self.create_control_panel(main_frame)
        
        # Visualization area
        self.create_visualization_area(main_frame)
    
    def create_control_panel(self, parent):
        """Create visualization control panel"""
        control_frame = ctk.CTkFrame(parent)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(control_frame, text="Visualization Controls", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        
        # Plot type selection
        plot_frame = ctk.CTkFrame(control_frame)
        plot_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(plot_frame, text="Plot Type:").pack(side="left", padx=5)
        
        self.plot_type = ctk.CTkOptionMenu(
            plot_frame,
            values=["Model Comparison", "Defense Effectiveness", "Confusion Matrix", "Privacy-Utility Tradeoff"],
            command=self.update_plot_type
        )
        self.plot_type.pack(side="left", padx=5)
        
        # Update button
        update_btn = ctk.CTkButton(plot_frame, text="🔄 Update Plots", command=self.update_plots)
        update_btn.pack(side="right", padx=5)
        
        # Display options
        options_frame = ctk.CTkFrame(control_frame)
        options_frame.pack(fill="x", padx=10, pady=5)
        
        self.show_baseline = tk.BooleanVar(value=True)
        self.show_defended = tk.BooleanVar(value=True)
        self.show_grid = tk.BooleanVar(value=True)
        
        ctk.CTkCheckBox(options_frame, text="Show Baseline", variable=self.show_baseline).pack(side="left", padx=5)
        ctk.CTkCheckBox(options_frame, text="Show Defended", variable=self.show_defended).pack(side="left", padx=5)
        ctk.CTkCheckBox(options_frame, text="Show Grid", variable=self.show_grid).pack(side="left", padx=5)
    
    def create_visualization_area(self, parent):
        """Create the main visualization area"""
        viz_frame = ctk.CTkFrame(parent)
        viz_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create matplotlib figure with subplots
        plt.style.use('dark_background')
        self.fig = plt.Figure(figsize=(14, 10), facecolor='#2b2b2b')
        
        # Create subplots
        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        self.ax1 = self.fig.add_subplot(gs[0, 0])  # Model comparison
        self.ax2 = self.fig.add_subplot(gs[0, 1])  # Defense effectiveness
        self.ax3 = self.fig.add_subplot(gs[1, 0])  # Confusion matrix
        self.ax4 = self.fig.add_subplot(gs[1, 1])  # Privacy-utility tradeoff
        
        # Style the axes
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.set_facecolor('#2b2b2b')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
        # Initialize with sample plots
        self.create_sample_plots()
    
    def create_sample_plots(self):
        """Create sample plots for demonstration"""
        # Sample data
        models = ['RandomForest', 'SVM', 'XGBoost', 'CNN', 'RNN']
        baseline_acc = [0.89, 0.87, 0.91, 0.85, 0.88]
        defended_acc = [0.65, 0.62, 0.55, 0.58, 0.61]
        
        # Plot 1: Model Comparison
        self.ax1.clear()
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = self.ax1.bar(x - width/2, baseline_acc, width, label='Baseline', 
                            color='#3498db', alpha=0.8)
        bars2 = self.ax1.bar(x + width/2, defended_acc, width, label='With Defenses', 
                            color='#e74c3c', alpha=0.8)
        
        self.ax1.set_title('Model Performance Comparison', color='white', fontweight='bold')
        self.ax1.set_xlabel('Models', color='white')
        self.ax1.set_ylabel('Accuracy', color='white')
        self.ax1.set_xticks(x)
        self.ax1.set_xticklabels(models, rotation=45)
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                self.ax1.annotate(f'{height:.2f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', color='white', fontsize=8)
        
        # Plot 2: Defense Effectiveness
        self.ax2.clear()
        effectiveness = [(b - d) / b * 100 for b, d in zip(baseline_acc, defended_acc)]
        
        bars = self.ax2.bar(models, effectiveness, color='#f39c12', alpha=0.8)
        self.ax2.set_title('Defense Effectiveness', color='white', fontweight='bold')
        self.ax2.set_xlabel('Models', color='white')
        self.ax2.set_ylabel('Accuracy Reduction (%)', color='white')
        self.ax2.tick_params(axis='x', rotation=45)
        self.ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, eff in zip(bars, effectiveness):
            self.ax2.annotate(f'{eff:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', color='white', fontsize=8)
        
        # Plot 3: Sample Confusion Matrix
        self.ax3.clear()
        confusion_matrix = np.random.rand(5, 5)
        np.fill_diagonal(confusion_matrix, np.random.uniform(0.7, 0.9, 5))
        
        im = self.ax3.imshow(confusion_matrix, cmap='Blues', alpha=0.8)
        self.ax3.set_title('Confusion Matrix (Sample)', color='white', fontweight='bold')
        self.ax3.set_xlabel('Predicted Class', color='white')
        self.ax3.set_ylabel('True Class', color='white')
        
        # Add colorbar
        cbar = self.fig.colorbar(im, ax=self.ax3)
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.yaxis.label.set_color('white')
        
        # Plot 4: Privacy-Utility Tradeoff
        self.ax4.clear()
        
        # Generate sample tradeoff data
        privacy_gain = np.random.uniform(20, 80, 20)
        utility_cost = np.random.uniform(5, 40, 20)
        defense_types = np.random.choice(['Padding', 'Timing', 'Morphing', 'Adaptive'], 20)
        
        colors = {'Padding': '#3498db', 'Timing': '#e74c3c', 'Morphing': '#2ecc71', 'Adaptive': '#f39c12'}
        
        for defense_type in colors:
            mask = defense_types == defense_type
            self.ax4.scatter(utility_cost[mask], privacy_gain[mask], 
                           c=colors[defense_type], label=defense_type, alpha=0.7, s=60)
        
        self.ax4.set_title('Privacy-Utility Tradeoff', color='white', fontweight='bold')
        self.ax4.set_xlabel('Utility Cost (%)', color='white')
        self.ax4.set_ylabel('Privacy Gain (%)', color='white')
        self.ax4.legend()
        self.ax4.grid(True, alpha=0.3)
        
        # Draw the canvas
        self.canvas.draw()
    
    def update_plot_type(self, selection):
        """Update visualization based on plot type selection"""
        # This could focus on a specific plot type or rearrange the layout
        pass
    
    def update_plots(self):
        """Update all plots with current settings"""
        self.create_sample_plots()
    
    def update_defense_comparison(self, baseline_results, defended_accuracies):
        """Update plots with real defense comparison data"""
        if not baseline_results:
            return
        
        # Clear and update model comparison plot
        self.ax1.clear()
        
        models = list(baseline_results['models'].keys())
        baseline_acc = [baseline_results['models'][model]['accuracy'] for model in models]
        defended_acc = [defended_accuracies.get(model, 0) for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        self.ax1.bar(x - width/2, baseline_acc, width, label='Baseline', color='#3498db', alpha=0.8)
        self.ax1.bar(x + width/2, defended_acc, width, label='With Defenses', color='#e74c3c', alpha=0.8)
        
        self.ax1.set_title('Real Defense Impact', color='white', fontweight='bold')
        self.ax1.set_xlabel('Models', color='white')
        self.ax1.set_ylabel('Accuracy', color='white')
        self.ax1.set_xticks(x)
        self.ax1.set_xticklabels(models, rotation=45)
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Update effectiveness plot
        self.ax2.clear()
        effectiveness = [(b - d) / b * 100 for b, d in zip(baseline_acc, defended_acc)]
        
        self.ax2.bar(models, effectiveness, color='#f39c12', alpha=0.8)
        self.ax2.set_title('Actual Defense Effectiveness', color='white', fontweight='bold')
        self.ax2.set_xlabel('Models', color='white')
        self.ax2.set_ylabel('Accuracy Reduction (%)', color='white')
        self.ax2.tick_params(axis='x', rotation=45)
        self.ax2.grid(True, alpha=0.3)
        
        self.canvas.draw()

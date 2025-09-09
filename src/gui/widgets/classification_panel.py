"""
Classification results panel widget
"""

import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class ClassificationPanel:
    """Panel for displaying classification results"""
    
    def __init__(self, parent, config):
        self.parent = parent
        self.config = config
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the classification panel UI"""
        # Main container
        main_frame = ctk.CTkFrame(self.parent)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        title_label = ctk.CTkLabel(main_frame, text="🎯 Traffic Classification Results", 
                                  font=ctk.CTkFont(size=18, weight="bold"))
        title_label.pack(pady=10)
        
        # Create two columns
        columns_frame = ctk.CTkFrame(main_frame)
        columns_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left column - Results
        left_frame = ctk.CTkFrame(columns_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        self.create_results_section(left_frame)
        
        # Right column - Visualization
        right_frame = ctk.CTkFrame(columns_frame)  
        right_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        self.create_visualization_section(right_frame)
    
    def create_results_section(self, parent):
        """Create results display section"""
        # Current prediction
        pred_frame = ctk.CTkFrame(parent)
        pred_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(pred_frame, text="Current Prediction", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        
        self.pred_url_label = ctk.CTkLabel(pred_frame, text="URL: Not analyzed", 
                                          font=ctk.CTkFont(size=12))
        self.pred_url_label.pack(pady=2)
        
        self.pred_site_label = ctk.CTkLabel(pred_frame, text="Predicted Site: -", 
                                           font=ctk.CTkFont(size=12, weight="bold"))
        self.pred_site_label.pack(pady=2)
        
        self.pred_confidence_label = ctk.CTkLabel(pred_frame, text="Confidence: -", 
                                                 font=ctk.CTkFont(size=12))
        self.pred_confidence_label.pack(pady=2)
        
        self.pred_model_label = ctk.CTkLabel(pred_frame, text="Model: -", 
                                            font=ctk.CTkFont(size=12))
        self.pred_model_label.pack(pady=2)
        
        # Top candidates
        candidates_frame = ctk.CTkFrame(parent)
        candidates_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(candidates_frame, text="Top 3 Candidates", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        
        self.candidates_text = ctk.CTkTextbox(candidates_frame, height=100)
        self.candidates_text.pack(fill="x", padx=10, pady=10)
        
        # Flow statistics
        stats_frame = ctk.CTkFrame(parent)
        stats_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(stats_frame, text="Flow Statistics", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        
        self.stats_text = ctk.CTkTextbox(stats_frame, height=200)
        self.stats_text.pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_visualization_section(self, parent):
        """Create visualization section"""
        viz_frame = ctk.CTkFrame(parent)
        viz_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(viz_frame, text="Traffic Visualization", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        
        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        self.fig.patch.set_facecolor('#2b2b2b')  # Dark theme
        
        for ax in [self.ax1, self.ax2]:
            ax.set_facecolor('#2b2b2b')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
        
        # Initial empty plots
        self.ax1.set_title("Packet Size Sequence", color='white')
        self.ax1.set_xlabel("Packet Number")
        self.ax1.set_ylabel("Size (bytes)")
        
        self.ax2.set_title("Inter-arrival Times", color='white')
        self.ax2.set_xlabel("Packet Number")  
        self.ax2.set_ylabel("Time (ms)")
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    def update_prediction(self, url, model, confidence, candidates):
        """Update prediction display"""
        self.pred_url_label.configure(text=f"URL: {url}")
        self.pred_site_label.configure(text=f"Predicted Site: {candidates[0] if candidates else 'Unknown'}")
        self.pred_confidence_label.configure(text=f"Confidence: {confidence:.1%}")
        self.pred_model_label.configure(text=f"Model: {model}")
        
        # Update candidates
        self.candidates_text.delete("1.0", tk.END)
        for i, candidate in enumerate(candidates[:3], 1):
            conf = confidence * (1 - i * 0.1)  # Simulate decreasing confidence
            self.candidates_text.insert(tk.END, f"{i}. {candidate} ({conf:.1%})\n")
        
        # Update flow statistics (simulated)
        self.stats_text.delete("1.0", tk.END)
        stats_text = f"""Total Packets: {np.random.randint(50, 200)}
Total Bytes: {np.random.randint(10000, 100000):,}
Flow Duration: {np.random.uniform(2.0, 10.0):.2f}s
Mean Packet Size: {np.random.randint(200, 800)} bytes
Outgoing/Incoming Ratio: {np.random.uniform(0.2, 0.8):.2f}
Number of Bursts: {np.random.randint(3, 15)}
"""
        self.stats_text.insert("1.0", stats_text)
        
        # Update plots
        self.update_plots()
    
    def update_plots(self):
        """Update traffic visualization plots"""
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Generate sample data (in real implementation, use actual traffic data)
        n_packets = np.random.randint(50, 150)
        packet_sizes = np.random.exponential(400, n_packets) + np.random.normal(200, 100, n_packets)
        packet_sizes = np.clip(packet_sizes, 40, 1500)
        
        inter_arrival_times = np.random.exponential(50, n_packets-1)  # ms
        
        # Plot packet sizes
        self.ax1.plot(range(len(packet_sizes)), packet_sizes, 'cyan', linewidth=1, alpha=0.8)
        self.ax1.scatter(range(len(packet_sizes)), packet_sizes, c='yellow', s=10, alpha=0.6)
        self.ax1.set_title("Packet Size Sequence", color='white')
        self.ax1.set_xlabel("Packet Number", color='white')
        self.ax1.set_ylabel("Size (bytes)", color='white')
        self.ax1.grid(True, alpha=0.3)
        
        # Plot inter-arrival times
        self.ax2.plot(range(len(inter_arrival_times)), inter_arrival_times, 'orange', linewidth=1, alpha=0.8)
        self.ax2.set_title("Inter-arrival Times", color='white')
        self.ax2.set_xlabel("Packet Number", color='white')
        self.ax2.set_ylabel("Time (ms)", color='white')
        self.ax2.grid(True, alpha=0.3)
        
        # Style adjustments
        for ax in [self.ax1, self.ax2]:
            ax.set_facecolor('#2b2b2b')
            ax.tick_params(colors='white')
        
        plt.tight_layout()
        self.fig.canvas.draw()

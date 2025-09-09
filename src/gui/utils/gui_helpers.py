"""
GUI Helper utilities for the traffic fingerprinting application
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from datetime import datetime
import numpy as np

def format_percentage(value):
    """Format float as percentage string"""
    if value is None:
        return "--"
    return f"{value*100:.1f}%"

def format_accuracy(value):
    """Format accuracy value with proper precision"""
    if value is None:
        return "--"
    return f"{value:.3f}"

def format_scientific(value):
    """Format float in scientific notation"""
    if value is None:
        return "--"
    return f"{value:.2e}"

def format_bytes(bytes_val):
    """Format bytes in human readable format"""
    if bytes_val is None:
        return "--"
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} TB"

def format_time_duration(seconds):
    """Format time duration in human readable format"""
    if seconds is None:
        return "--"
    
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:.0f}m {secs:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"

def parse_boolean(s):
    """Parse string to boolean"""
    if isinstance(s, bool):
        return s
    return str(s).lower() in ("yes", "true", "1", "on", "enabled")

def validate_number_input(value, min_val=None, max_val=None):
    """Validate numeric input"""
    try:
        num = float(value)
        if min_val is not None and num < min_val:
            return False, f"Value must be >= {min_val}"
        if max_val is not None and num > max_val:
            return False, f"Value must be <= {max_val}"
        return True, num
    except ValueError:
        return False, "Invalid number format"

def create_tooltip(widget, text):
    """Create tooltip for widget"""
    def show_tooltip(event):
        tooltip = tk.Toplevel()
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
        
        label = tk.Label(tooltip, text=text, background="lightyellow",
                        relief="solid", borderwidth=1, font=("Arial", 9))
        label.pack()
        
        def hide_tooltip():
            tooltip.destroy()
        
        tooltip.after(3000, hide_tooltip)  # Auto-hide after 3 seconds
    
    widget.bind("<Enter>", show_tooltip)

def run_in_background(func, callback=None, *args, **kwargs):
    """Run function in background thread with optional callback"""
    def worker():
        try:
            result = func(*args, **kwargs)
            if callback:
                callback(result, None)
        except Exception as e:
            if callback:
                callback(None, e)
    
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread

def show_progress_dialog(parent, title, message, cancel_callback=None):
    """Show progress dialog"""
    progress_window = tk.Toplevel(parent)
    progress_window.title(title)
    progress_window.geometry("400x150")
    progress_window.transient(parent)
    progress_window.grab_set()
    
    # Center the dialog
    progress_window.update_idletasks()
    x = (progress_window.winfo_screenwidth() // 2) - (400 // 2)
    y = (progress_window.winfo_screenheight() // 2) - (150 // 2)
    progress_window.geometry(f"400x150+{x}+{y}")
    
    # Message label
    msg_label = tk.Label(progress_window, text=message, wraplength=350)
    msg_label.pack(pady=20)
    
    # Progress bar
    progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
    progress_bar.pack(pady=10, padx=20, fill='x')
    progress_bar.start()
    
    # Cancel button
    if cancel_callback:
        cancel_btn = ttk.Button(progress_window, text="Cancel", command=cancel_callback)
        cancel_btn.pack(pady=10)
    
    return progress_window, progress_bar

def get_timestamp():
    """Get current timestamp string"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def safe_divide(numerator, denominator, default=0):
    """Safely divide two numbers"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default

def interpolate_color(color1, color2, factor):
    """Interpolate between two colors"""
    # Convert hex colors to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def rgb_to_hex(rgb):
        return '#%02x%02x%02x' % rgb
    
    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)
    
    interpolated = tuple(int(rgb1[i] + factor * (rgb2[i] - rgb1[i])) for i in range(3))
    return rgb_to_hex(interpolated)

class StatusManager:
    """Manages status updates across the application"""
    
    def __init__(self):
        self.status_var = None
        self.callbacks = []
    
    def set_status_var(self, status_var):
        self.status_var = status_var
    
    def add_callback(self, callback):
        self.callbacks.append(callback)
    
    def update_status(self, message):
        if self.status_var:
            self.status_var.set(f"{get_timestamp()} - {message}")
        
        for callback in self.callbacks:
            try:
                callback(message)
            except:
                pass

# Global status manager instance
status_manager = StatusManager()

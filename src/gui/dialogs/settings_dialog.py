"""
Settings dialog for application configuration
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk
import json
from pathlib import Path

class SettingsDialog:
    """Settings configuration dialog"""
    
    def __init__(self, parent, config):
        self.parent = parent
        self.config = config.copy()  # Work with a copy
        self.result = None
        self.create_dialog()
    
    def create_dialog(self):
        """Create the settings dialog"""
        self.dialog = ctk.CTkToplevel(self.parent)
        self.dialog.title("⚙️ Application Settings")
        self.dialog.geometry("600x700")
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (600 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (700 // 2)
        self.dialog.geometry(f"600x700+{x}+{y}")
        
        # Create notebook for different setting categories
        self.notebook = ttk.Notebook(self.dialog)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Create different setting tabs
        self.create_general_tab()
        self.create_collection_tab()
        self.create_models_tab()
        self.create_paths_tab()
        
        # Buttons frame
        button_frame = ctk.CTkFrame(self.dialog)
        button_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        ctk.CTkButton(button_frame, text="💾 Save", command=self.save_settings).pack(side="right", padx=5)
        ctk.CTkButton(button_frame, text="❌ Cancel", command=self.cancel).pack(side="right", padx=5)
        ctk.CTkButton(button_frame, text="🔄 Reset to Defaults", command=self.reset_defaults).pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="📁 Import", command=self.import_settings).pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="💾 Export", command=self.export_settings).pack(side="left", padx=5)
    
    def create_general_tab(self):
        """Create general settings tab"""
        general_frame = ttk.Frame(self.notebook)
        self.notebook.add(general_frame, text="🔧 General")
        
        # Scrollable frame
        canvas = tk.Canvas(general_frame)
        scrollbar = ttk.Scrollbar(general_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Application settings
        app_frame = ttk.LabelFrame(scrollable_frame, text="Application Settings", padding="10")
        app_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(app_frame, text="Theme:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.theme_var = tk.StringVar(value=self.config.get('theme', 'dark'))
        theme_combo = ttk.Combobox(app_frame, textvariable=self.theme_var, 
                                  values=["dark", "light", "system"], state="readonly")
        theme_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        ttk.Label(app_frame, text="Auto-save interval (minutes):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.autosave_var = tk.IntVar(value=self.config.get('autosave_interval', 5))
        autosave_spin = ttk.Spinbox(app_frame, from_=1, to=60, textvariable=self.autosave_var)
        autosave_spin.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        # Logging settings
        log_frame = ttk.LabelFrame(scrollable_frame, text="Logging Settings", padding="10")
        log_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(log_frame, text="Log Level:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.log_level_var = tk.StringVar(value=self.config.get('log_level', 'INFO'))
        log_combo = ttk.Combobox(log_frame, textvariable=self.log_level_var,
                                values=["DEBUG", "INFO", "WARNING", "ERROR"], state="readonly")
        log_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        self.enable_file_logging = tk.BooleanVar(value=self.config.get('enable_file_logging', True))
        ttk.Checkbutton(log_frame, text="Enable file logging", 
                       variable=self.enable_file_logging).grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        # Configure grid weights
        app_frame.grid_columnconfigure(1, weight=1)
        log_frame.grid_columnconfigure(1, weight=1)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_collection_tab(self):
        """Create data collection settings tab"""
        collection_frame = ttk.Frame(self.notebook)
        self.notebook.add(collection_frame, text="📊 Data Collection")
        
        # Traffic collection settings
        traffic_frame = ttk.LabelFrame(collection_frame, text="Traffic Collection", padding="10")
        traffic_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(traffic_frame, text="Default visits per site:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.visits_var = tk.IntVar(value=self.config.get('collection', {}).get('visits_per_site', 100))
        visits_spin = ttk.Spinbox(traffic_frame, from_=10, to=1000, textvariable=self.visits_var)
        visits_spin.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        ttk.Label(traffic_frame, text="Capture duration (seconds):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.duration_var = tk.IntVar(value=self.config.get('collection', {}).get('capture_duration', 30))
        duration_spin = ttk.Spinbox(traffic_frame, from_=5, to=300, textvariable=self.duration_var)
        duration_spin.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        ttk.Label(traffic_frame, text="Network interface:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.interface_var = tk.StringVar(value=self.config.get('collection', {}).get('interface', 'en0'))
        interface_entry = ttk.Entry(traffic_frame, textvariable=self.interface_var)
        interface_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        
        # Network options
        network_frame = ttk.LabelFrame(collection_frame, text="Network Options", padding="10")
        network_frame.pack(fill="x", padx=10, pady=5)
        
        self.use_tor_var = tk.BooleanVar(value=self.config.get('collection', {}).get('use_tor', False))
        ttk.Checkbutton(network_frame, text="Use Tor network", 
                       variable=self.use_tor_var).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        
        self.use_vpn_var = tk.BooleanVar(value=self.config.get('collection', {}).get('use_vpn', False))
        ttk.Checkbutton(network_frame, text="Use VPN connection", 
                       variable=self.use_vpn_var).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        
        self.headless_var = tk.BooleanVar(value=self.config.get('collection', {}).get('headless', True))
        ttk.Checkbutton(network_frame, text="Headless browser mode", 
                       variable=self.headless_var).grid(row=2, column=0, sticky="w", padx=5, pady=5)
        
        # Configure grid weights
        traffic_frame.grid_columnconfigure(1, weight=1)
    
    def create_models_tab(self):
        """Create model settings tab"""
        models_frame = ttk.Frame(self.notebook)
        self.notebook.add(models_frame, text="🤖 Models")
        
        # Training settings
        training_frame = ttk.LabelFrame(models_frame, text="Training Settings", padding="10")
        training_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(training_frame, text="Test set size:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.test_size_var = tk.DoubleVar(value=self.config.get('models', {}).get('test_size', 0.3))
        test_size_scale = ttk.Scale(training_frame, from_=0.1, to=0.5, variable=self.test_size_var, orient="horizontal")
        test_size_scale.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.test_size_label = ttk.Label(training_frame, text=f"{self.test_size_var.get():.1f}")
        self.test_size_label.grid(row=0, column=2, padx=5, pady=5)
        test_size_scale.configure(command=lambda v: self.test_size_label.configure(text=f"{float(v):.1f}"))
        
        ttk.Label(training_frame, text="Validation set size:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.val_size_var = tk.DoubleVar(value=self.config.get('models', {}).get('validation_size', 0.2))
        val_size_scale = ttk.Scale(training_frame, from_=0.1, to=0.4, variable=self.val_size_var, orient="horizontal")
        val_size_scale.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        self.val_size_label = ttk.Label(training_frame, text=f"{self.val_size_var.get():.1f}")
        self.val_size_label.grid(row=1, column=2, padx=5, pady=5)
        val_size_scale.configure(command=lambda v: self.val_size_label.configure(text=f"{float(v):.1f}"))
        
        ttk.Label(training_frame, text="Cross-validation folds:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.cv_folds_var = tk.IntVar(value=self.config.get('models', {}).get('cv_folds', 5))
        cv_spin = ttk.Spinbox(training_frame, from_=3, to=10, textvariable=self.cv_folds_var)
        cv_spin.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        
        ttk.Label(training_frame, text="Random state:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.random_state_var = tk.IntVar(value=self.config.get('models', {}).get('random_state', 42))
        random_state_spin = ttk.Spinbox(training_frame, from_=0, to=9999, textvariable=self.random_state_var)
        random_state_spin.grid(row=3, column=1, sticky="ew", padx=5, pady=5)
        
        # Configure grid weights
        training_frame.grid_columnconfigure(1, weight=1)
    
    def create_paths_tab(self):
        """Create paths settings tab"""
        paths_frame = ttk.Frame(self.notebook)
        self.notebook.add(paths_frame, text="📁 Paths")
        
        # Path settings
        path_frame = ttk.LabelFrame(paths_frame, text="Directory Paths", padding="10")
        path_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        paths_config = self.config.get('paths', {})
        
        # Data path
        ttk.Label(path_frame, text="Data directory:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.data_path_var = tk.StringVar(value=paths_config.get('data', 'data'))
        data_entry = ttk.Entry(path_frame, textvariable=self.data_path_var)
        data_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(path_frame, text="Browse", 
                  command=lambda: self.browse_directory(self.data_path_var)).grid(row=0, column=2, padx=5, pady=5)
        
        # Results path
        ttk.Label(path_frame, text="Results directory:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.results_path_var = tk.StringVar(value=paths_config.get('results', 'results'))
        results_entry = ttk.Entry(path_frame, textvariable=self.results_path_var)
        results_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(path_frame, text="Browse", 
                  command=lambda: self.browse_directory(self.results_path_var)).grid(row=1, column=2, padx=5, pady=5)
        
        # Logs path
        ttk.Label(path_frame, text="Logs directory:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.logs_path_var = tk.StringVar(value=paths_config.get('logs', 'logs'))
        logs_entry = ttk.Entry(path_frame, textvariable=self.logs_path_var)
        logs_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(path_frame, text="Browse", 
                  command=lambda: self.browse_directory(self.logs_path_var)).grid(row=2, column=2, padx=5, pady=5)
        
        # Configure grid weights
        path_frame.grid_columnconfigure(1, weight=1)
    
    def browse_directory(self, var):
        """Browse for directory"""
        directory = filedialog.askdirectory(initialdir=var.get())
        if directory:
            var.set(directory)
    
    def save_settings(self):
        """Save settings and close dialog"""
        try:
            # Update config with new values
            self.config['theme'] = self.theme_var.get()
            self.config['autosave_interval'] = self.autosave_var.get()
            self.config['log_level'] = self.log_level_var.get()
            self.config['enable_file_logging'] = self.enable_file_logging.get()
            
            # Collection settings
            if 'collection' not in self.config:
                self.config['collection'] = {}
            self.config['collection']['visits_per_site'] = self.visits_var.get()
            self.config['collection']['capture_duration'] = self.duration_var.get()
            self.config['collection']['interface'] = self.interface_var.get()
            self.config['collection']['use_tor'] = self.use_tor_var.get()
            self.config['collection']['use_vpn'] = self.use_vpn_var.get()
            self.config['collection']['headless'] = self.headless_var.get()
            
            # Model settings
            if 'models' not in self.config:
                self.config['models'] = {}
            self.config['models']['test_size'] = self.test_size_var.get()
            self.config['models']['validation_size'] = self.val_size_var.get()
            self.config['models']['cv_folds'] = self.cv_folds_var.get()
            self.config['models']['random_state'] = self.random_state_var.get()
            
            # Path settings
            if 'paths' not in self.config:
                self.config['paths'] = {}
            self.config['paths']['data'] = self.data_path_var.get()
            self.config['paths']['results'] = self.results_path_var.get()
            self.config['paths']['logs'] = self.logs_path_var.get()
            
            self.result = self.config
            self.dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")
    
    def cancel(self):
        """Cancel and close dialog"""
        self.result = None
        self.dialog.destroy()
    
    def reset_defaults(self):
        """Reset all settings to defaults"""
        if messagebox.askyesno("Reset Settings", "Reset all settings to default values?"):
            # Reset to default values
            self.theme_var.set('dark')
            self.autosave_var.set(5)
            self.log_level_var.set('INFO')
            self.enable_file_logging.set(True)
            self.visits_var.set(100)
            self.duration_var.set(30)
            self.interface_var.set('en0')
            self.use_tor_var.set(False)
            self.use_vpn_var.set(False)
            self.headless_var.set(True)
            self.test_size_var.set(0.3)
            self.val_size_var.set(0.2)
            self.cv_folds_var.set(5)
            self.random_state_var.set(42)
            self.data_path_var.set('data')
            self.results_path_var.set('results')
            self.logs_path_var.set('logs')
    
    def import_settings(self):
        """Import settings from file"""
        filename = filedialog.askopenfilename(
            title="Import Settings",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    imported_config = json.load(f)
                
                # Update variables with imported values
                for key, value in imported_config.items():
                    if hasattr(self, f"{key}_var"):
                        getattr(self, f"{key}_var").set(value)
                
                messagebox.showinfo("Success", "Settings imported successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import settings: {e}")
    
    def export_settings(self):
        """Export current settings to file"""
        filename = filedialog.asksaveasfilename(
            title="Export Settings",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Gather current settings
                current_settings = {
                    'theme': self.theme_var.get(),
                    'autosave_interval': self.autosave_var.get(),
                    'log_level': self.log_level_var.get(),
                    'enable_file_logging': self.enable_file_logging.get(),
                    'collection': {
                        'visits_per_site': self.visits_var.get(),
                        'capture_duration': self.duration_var.get(),
                        'interface': self.interface_var.get(),
                        'use_tor': self.use_tor_var.get(),
                        'use_vpn': self.use_vpn_var.get(),
                        'headless': self.headless_var.get()
                    },
                    'models': {
                        'test_size': self.test_size_var.get(),
                        'validation_size': self.val_size_var.get(),
                        'cv_folds': self.cv_folds_var.get(),
                        'random_state': self.random_state_var.get()
                    },
                    'paths': {
                        'data': self.data_path_var.get(),
                        'results': self.results_path_var.get(),
                        'logs': self.logs_path_var.get()
                    }
                }
                
                with open(filename, 'w') as f:
                    json.dump(current_settings, f, indent=2)
                
                messagebox.showinfo("Success", f"Settings exported to {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export settings: {e}")
    
    def show(self):
        """Show dialog and return result"""
        self.dialog.wait_window()
        return self.result

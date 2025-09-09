"""
Defense simulation panel widget
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk
import json

class DefensePanel:  # ✅ Fixed class name
    """Panel for defense configuration and simulation"""
    
    def __init__(self, parent, config, update_callback):
        self.parent = parent
        self.config = config
        self.update_callback = update_callback
        self.defense_vars = {}
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the defense panel UI"""
        main_frame = ctk.CTkScrollableFrame(self.parent)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        title_label = ctk.CTkLabel(main_frame, text="🛡️ Defense Mechanism Simulator", 
                                  font=ctk.CTkFont(size=18, weight="bold"))
        title_label.pack(pady=10)
        
        self.create_padding_defenses(main_frame)
        self.create_timing_defenses(main_frame)
        self.create_morphing_defenses(main_frame)
        self.create_adaptive_defenses(main_frame)
        self.create_control_buttons(main_frame)
    
    def create_padding_defenses(self, parent):
        """Create packet padding defense controls"""
        padding_frame = ctk.CTkFrame(parent)
        padding_frame.pack(fill="x", padx=10, pady=10)
        
        header_frame = ctk.CTkFrame(padding_frame)
        header_frame.pack(fill="x", padx=10, pady=5)
        
        self.defense_vars['padding_enabled'] = tk.BooleanVar(
            value=self.config.get('defense', {}).get('padding', {}).get('enabled', True)
        )
        
        padding_cb = ctk.CTkCheckBox(header_frame, text="Packet Padding Defenses", 
                                    variable=self.defense_vars['padding_enabled'],
                                    font=ctk.CTkFont(size=14, weight="bold"),
                                    command=self.on_defense_change)
        padding_cb.pack(side="left", padx=10, pady=10)
        
        options_frame = ctk.CTkFrame(padding_frame)
        options_frame.pack(fill="x", padx=20, pady=5)
        
        # Padding type
        ctk.CTkLabel(options_frame, text="Padding Type:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.defense_vars['padding_type'] = tk.StringVar(value="constant")
        padding_type_menu = ctk.CTkOptionMenu(options_frame, 
                                            variable=self.defense_vars['padding_type'],
                                            values=["constant", "random", "multiples"],
                                            command=self.on_defense_change)
        padding_type_menu.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        
        # Target size - ✅ Fixed callback
        ctk.CTkLabel(options_frame, text="Target Size (bytes):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.defense_vars['padding_size'] = tk.IntVar(
            value=self.config.get('defense', {}).get('padding', {}).get('target_size', 1500)
        )
        
        self.padding_size_label = ctk.CTkLabel(options_frame, text="1500")
        self.padding_size_label.grid(row=1, column=2, padx=10, pady=5)
        
        size_slider = ctk.CTkSlider(options_frame, from_=500, to=1500, 
                                   variable=self.defense_vars['padding_size'])
        size_slider.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        # ✅ Single configure call with combined callback
        size_slider.configure(command=self.on_size_slider_change)
        
        # Dummy packet rate - ✅ Fixed callback
        ctk.CTkLabel(options_frame, text="Dummy Packet Rate:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        
        self.defense_vars['dummy_rate'] = tk.DoubleVar(value=0.1)
        
        self.dummy_rate_label = ctk.CTkLabel(options_frame, text="10%")
        self.dummy_rate_label.grid(row=2, column=2, padx=10, pady=5)
        
        dummy_slider = ctk.CTkSlider(options_frame, from_=0.0, to=0.5, 
                                    variable=self.defense_vars['dummy_rate'])
        dummy_slider.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        # ✅ Single configure call with combined callback
        dummy_slider.configure(command=self.on_dummy_slider_change)
        
        options_frame.grid_columnconfigure(1, weight=1)
    
    def create_timing_defenses(self, parent):
        """Create timing defense controls"""
        timing_frame = ctk.CTkFrame(parent)
        timing_frame.pack(fill="x", padx=10, pady=10)
        
        header_frame = ctk.CTkFrame(timing_frame)
        header_frame.pack(fill="x", padx=10, pady=5)
        
        self.defense_vars['timing_enabled'] = tk.BooleanVar(
            value=self.config.get('defense', {}).get('timing', {}).get('enabled', True)
        )
        
        timing_cb = ctk.CTkCheckBox(header_frame, text="Timing Obfuscation Defenses", 
                                   variable=self.defense_vars['timing_enabled'],
                                   font=ctk.CTkFont(size=14, weight="bold"),
                                   command=self.on_defense_change)
        timing_cb.pack(side="left", padx=10, pady=10)
        
        options_frame = ctk.CTkFrame(timing_frame)
        options_frame.pack(fill="x", padx=20, pady=5)
        
        # Jitter range
        ctk.CTkLabel(options_frame, text="Jitter Range (ms):").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.defense_vars['jitter_min'] = tk.IntVar(value=0)
        self.defense_vars['jitter_max'] = tk.IntVar(value=100)
        
        jitter_frame = ctk.CTkFrame(options_frame)
        jitter_frame.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        
        self.jitter_label = ctk.CTkLabel(jitter_frame, text="0-100ms")
        self.jitter_label.pack(side="right", padx=5)
        
        jitter_slider = ctk.CTkSlider(jitter_frame, from_=0, to=200, 
                                     variable=self.defense_vars['jitter_max'])
        jitter_slider.pack(side="left", fill="x", expand=True)
        # ✅ Fixed callback
        jitter_slider.configure(command=self.on_jitter_slider_change)
        
        # Constant rate
        self.defense_vars['constant_rate'] = tk.BooleanVar(value=False)
        constant_cb = ctk.CTkCheckBox(options_frame, text="Use Constant Rate", 
                                     variable=self.defense_vars['constant_rate'],
                                     command=self.on_defense_change)
        constant_cb.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        
        # Rate value
        ctk.CTkLabel(options_frame, text="Rate (ms):").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        
        self.defense_vars['rate_ms'] = tk.IntVar(value=50)
        
        self.rate_label = ctk.CTkLabel(options_frame, text="50ms")
        self.rate_label.grid(row=2, column=2, padx=10, pady=5)
        
        rate_slider = ctk.CTkSlider(options_frame, from_=10, to=200, 
                                   variable=self.defense_vars['rate_ms'])
        rate_slider.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        # ✅ Fixed callback
        rate_slider.configure(command=self.on_rate_slider_change)
        
        options_frame.grid_columnconfigure(1, weight=1)
    
    def create_morphing_defenses(self, parent):
        """Create traffic morphing defense controls"""
        morphing_frame = ctk.CTkFrame(parent)
        morphing_frame.pack(fill="x", padx=10, pady=10)
        
        header_frame = ctk.CTkFrame(morphing_frame)
        header_frame.pack(fill="x", padx=10, pady=5)
        
        self.defense_vars['morphing_enabled'] = tk.BooleanVar(value=False)
        
        morphing_cb = ctk.CTkCheckBox(header_frame, text="Traffic Morphing Defenses", 
                                     variable=self.defense_vars['morphing_enabled'],
                                     font=ctk.CTkFont(size=14, weight="bold"),
                                     command=self.on_defense_change)
        morphing_cb.pack(side="left", padx=10, pady=10)
        
        options_frame = ctk.CTkFrame(morphing_frame)
        options_frame.pack(fill="x", padx=20, pady=5)
        
        # Target site
        ctk.CTkLabel(options_frame, text="Morph Into:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.defense_vars['morph_target'] = tk.StringVar(value="popular_site")
        morph_menu = ctk.CTkOptionMenu(options_frame, 
                                      variable=self.defense_vars['morph_target'],
                                      values=["popular_site", "random_site", "average_profile"],
                                      command=self.on_defense_change)
        morph_menu.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        
        # Morphing strength
        ctk.CTkLabel(options_frame, text="Morphing Strength:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.defense_vars['morph_strength'] = tk.DoubleVar(value=0.5)
        
        self.morph_strength_label = ctk.CTkLabel(options_frame, text="50%")
        self.morph_strength_label.grid(row=1, column=2, padx=10, pady=5)
        
        morph_slider = ctk.CTkSlider(options_frame, from_=0.0, to=1.0, 
                                    variable=self.defense_vars['morph_strength'])
        morph_slider.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        # ✅ Fixed callback
        morph_slider.configure(command=self.on_morph_slider_change)
        
        options_frame.grid_columnconfigure(1, weight=1)
    
    def create_adaptive_defenses(self, parent):
        """Create adaptive defense controls"""
        adaptive_frame = ctk.CTkFrame(parent)
        adaptive_frame.pack(fill="x", padx=10, pady=10)
        
        header_frame = ctk.CTkFrame(adaptive_frame)
        header_frame.pack(fill="x", padx=10, pady=5)
        
        self.defense_vars['adaptive_enabled'] = tk.BooleanVar(value=False)
        
        adaptive_cb = ctk.CTkCheckBox(header_frame, text="Adaptive Defenses", 
                                     variable=self.defense_vars['adaptive_enabled'],
                                     font=ctk.CTkFont(size=14, weight="bold"),
                                     command=self.on_defense_change)
        adaptive_cb.pack(side="left", padx=10, pady=10)
        
        options_frame = ctk.CTkFrame(adaptive_frame)
        options_frame.pack(fill="x", padx=20, pady=5)
        
        # Strategy
        ctk.CTkLabel(options_frame, text="Strategy:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.defense_vars['adaptive_strategy'] = tk.StringVar(value="bandwidth_aware")
        strategy_menu = ctk.CTkOptionMenu(options_frame, 
                                         variable=self.defense_vars['adaptive_strategy'],
                                         values=["bandwidth_aware", "latency_optimized", "privacy_maximized"],
                                         command=self.on_defense_change)
        strategy_menu.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        
        # Learning rate
        ctk.CTkLabel(options_frame, text="Learning Rate:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.defense_vars['learning_rate'] = tk.DoubleVar(value=0.01)
        
        self.lr_label = ctk.CTkLabel(options_frame, text="0.01")
        self.lr_label.grid(row=1, column=2, padx=10, pady=5)
        
        lr_slider = ctk.CTkSlider(options_frame, from_=0.001, to=0.1, 
                                 variable=self.defense_vars['learning_rate'])
        lr_slider.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        # ✅ Fixed callback
        lr_slider.configure(command=self.on_lr_slider_change)
        
        options_frame.grid_columnconfigure(1, weight=1)
    
    def create_control_buttons(self, parent):
        """Create control buttons"""
        control_frame = ctk.CTkFrame(parent)
        control_frame.pack(fill="x", padx=10, pady=20)
        
        apply_btn = ctk.CTkButton(control_frame, text="🔄 Apply Defenses", 
                                 command=self.apply_defenses,
                                 font=ctk.CTkFont(size=14, weight="bold"))
        apply_btn.pack(side="left", padx=10)
        
        reset_btn = ctk.CTkButton(control_frame, text="🔄 Reset to Defaults", 
                                 command=self.reset_defenses)
        reset_btn.pack(side="left", padx=10)
        
        export_btn = ctk.CTkButton(control_frame, text="💾 Export Config", 
                                  command=self.export_config)
        export_btn.pack(side="left", padx=10)
        
        self.performance_label = ctk.CTkLabel(control_frame, 
                                             text="Est. Performance Impact: Low",
                                             font=ctk.CTkFont(size=12))
        self.performance_label.pack(side="right", padx=10)
    
    # ✅ Fixed callback methods - combined label update and defense change
    def on_size_slider_change(self, value):
        """Handle padding size slider changes"""
        self.padding_size_label.configure(text=f"{int(float(value))}")
        self.on_defense_change()
    
    def on_dummy_slider_change(self, value):
        """Handle dummy rate slider changes"""
        self.dummy_rate_label.configure(text=f"{float(value)*100:.1f}%")
        self.on_defense_change()
    
    def on_jitter_slider_change(self, value):
        """Handle jitter slider changes"""
        min_val = self.defense_vars['jitter_min'].get()
        max_val = int(float(value))
        self.jitter_label.configure(text=f"{min_val}-{max_val}ms")
        self.on_defense_change()
    
    def on_rate_slider_change(self, value):
        """Handle rate slider changes"""
        self.rate_label.configure(text=f"{int(float(value))}ms")
        self.on_defense_change()
    
    def on_morph_slider_change(self, value):
        """Handle morph strength slider changes"""
        self.morph_strength_label.configure(text=f"{float(value)*100:.0f}%")
        self.on_defense_change()
    
    def on_lr_slider_change(self, value):
        """Handle learning rate slider changes"""
        self.lr_label.configure(text=f"{float(value):.3f}")
        self.on_defense_change()
    
    def on_defense_change(self, *args):
        """Handle defense parameter changes"""
        self.update_performance_estimate()
    
    def update_performance_estimate(self):
        """Update performance impact estimate"""
        impact_score = 0
        
        if self.defense_vars['padding_enabled'].get():
            size = self.defense_vars['padding_size'].get()
            impact_score += min(size / 1500 * 30, 30)
            
        if self.defense_vars['timing_enabled'].get():
            jitter = self.defense_vars['jitter_max'].get()
            impact_score += min(jitter / 100 * 20, 20)
            
        if self.defense_vars['morphing_enabled'].get():
            strength = self.defense_vars['morph_strength'].get()
            impact_score += strength * 25
            
        if self.defense_vars['adaptive_enabled'].get():
            impact_score += 10
        
        if impact_score < 20:
            level, color = "Low", "green"
        elif impact_score < 50:
            level, color = "Medium", "orange"
        else:
            level, color = "High", "red"
        
        self.performance_label.configure(
            text=f"Est. Performance Impact: {level} ({impact_score:.0f}%)",
            text_color=color
        )
    
    def apply_defenses(self):
        """Apply current defense configuration"""
        defense_config = self.get_defense_config()
        self.update_callback(defense_config)
        messagebox.showinfo("Success", "Defense configuration applied successfully!")
    
    def get_defense_config(self):
        """Get current defense configuration"""
        return {
            'padding': {
                'enabled': self.defense_vars['padding_enabled'].get(),
                'type': self.defense_vars['padding_type'].get(),
                'target_size': self.defense_vars['padding_size'].get(),
                'dummy_rate': self.defense_vars['dummy_rate'].get()
            },
            'timing': {
                'enabled': self.defense_vars['timing_enabled'].get(),
                'jitter_range': [self.defense_vars['jitter_min'].get(), 
                               self.defense_vars['jitter_max'].get()],
                'constant_rate': self.defense_vars['constant_rate'].get(),
                'rate_ms': self.defense_vars['rate_ms'].get()
            },
            'morphing': {
                'enabled': self.defense_vars['morphing_enabled'].get(),
                'target': self.defense_vars['morph_target'].get(),
                'strength': self.defense_vars['morph_strength'].get()
            },
            'adaptive': {
                'enabled': self.defense_vars['adaptive_enabled'].get(),
                'strategy': self.defense_vars['adaptive_strategy'].get(),
                'learning_rate': self.defense_vars['learning_rate'].get()
            }
        }
    
    def reset_defenses(self):
        """Reset to default defense configuration"""
        self.defense_vars['padding_enabled'].set(True)
        self.defense_vars['padding_type'].set("constant")
        self.defense_vars['padding_size'].set(1500)
        self.defense_vars['dummy_rate'].set(0.1)
        
        self.defense_vars['timing_enabled'].set(True)
        self.defense_vars['jitter_max'].set(100)
        self.defense_vars['constant_rate'].set(False)
        self.defense_vars['rate_ms'].set(50)
        
        self.defense_vars['morphing_enabled'].set(False)
        self.defense_vars['morph_target'].set("popular_site")
        self.defense_vars['morph_strength'].set(0.5)
        
        self.defense_vars['adaptive_enabled'].set(False)
        self.defense_vars['adaptive_strategy'].set("bandwidth_aware")
        self.defense_vars['learning_rate'].set(0.01)
        
        self.update_performance_estimate()
        messagebox.showinfo("Reset", "Defense settings reset to defaults!")
    
    def export_config(self):
        """Export current defense configuration to JSON file"""
        try:
            config = self.get_defense_config()
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")],
                title="Export Defense Configuration"
            )
            
            if not filename:
                return
            
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            
            messagebox.showinfo("Export Success", 
                              f"Defense configuration exported successfully to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", 
                               f"Failed to export configuration:\n{str(e)}")

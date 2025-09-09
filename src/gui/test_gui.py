import tkinter as tk
import customtkinter as ctk
from datetime import datetime

class TrafficFingerprintingGUI:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.root = ctk.CTk()
        self.root.geometry("600x400")
        self.root.title("Traffic Fingerprinting GUI - TEST")
        
        # Initialize variables
        self.jitter_var = tk.DoubleVar(value=10.0)
        self.loss_var = tk.DoubleVar(value=5.0)
        self.bandwidth_var = tk.DoubleVar(value=100.0)
        self.network_type = tk.StringVar(value="WiFi")
        self.geo_region = tk.StringVar(value="Local")
        
        self.setup_ui()

    def setup_ui(self):
        frame = ctk.CTkFrame(self.root)
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Test button - THIS CALLS apply_network_conditions
        apply_btn = ctk.CTkButton(frame, 
                                 text="Apply Network Conditions", 
                                 command=self.apply_network_conditions)
        apply_btn.pack(pady=20)
        
        # Output display
        self.output = tk.Text(frame, height=8)
        self.output.pack(fill="x", padx=10, pady=10)

    def apply_network_conditions(self):
        """THIS METHOD MUST EXIST AT THIS INDENTATION LEVEL"""
        jitter = self.jitter_var.get()
        loss = self.loss_var.get()
        bandwidth = self.bandwidth_var.get()
        network_type = self.network_type.get()
        geo_region = self.geo_region.get()
        
        msg = f"""✅ NETWORK CONDITIONS APPLIED SUCCESSFULLY!
        
Settings:
• Jitter: {jitter} ms
• Loss: {loss} %
• Bandwidth: {bandwidth} Mbps
• Network Type: {network_type}
• Region: {geo_region}
• Applied at: {datetime.now().strftime("%H:%M:%S")}

This method is working correctly!"""
        
        self.output.delete("1.0", "end")
        self.output.insert("1.0", msg)
        print("✅ apply_network_conditions method executed successfully!")

    def run(self):
        self.root.mainloop()

def main():
    app = TrafficFingerprintingGUI()
    app.run()

if __name__ == "__main__":
    main()

"""
Low-level packet capture utilities
"""

import subprocess
import psutil
import logging
from pathlib import Path

class PacketCapture:
    """Packet capture management class"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def get_network_interfaces(self):
        """Get available network interfaces"""
        interfaces = []
        for interface, addrs in psutil.net_if_addrs().items():
            if interface.startswith(('en', 'eth', 'wlan')):
                interfaces.append(interface)
        return interfaces
    
    def test_capture_permissions(self):
        """Test if we have permissions to capture packets"""
        try:
            result = subprocess.run(['tshark', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def capture_live_traffic(self, interface, output_file, duration=60, filter_expr="port 443"):
        """Capture live traffic for specified duration"""
        cmd = [
            'tshark',
            '-i', interface,
            '-f', filter_expr,
            '-a', f'duration:{duration}',
            '-w', str(output_file),
            '-q'
        ]
        
        try:
            self.logger.info(f"Starting {duration}s capture on {interface}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration+10)
            
            if result.returncode == 0:
                self.logger.info(f"Capture completed: {output_file}")
                return True
            else:
                self.logger.error(f"Capture failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("Capture timeout")
            return False
    
    def merge_pcap_files(self, input_files, output_file):
        """Merge multiple PCAP files"""
        cmd = ['mergecap', '-w', str(output_file)] + [str(f) for f in input_files]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            self.logger.error("mergecap not found. Install Wireshark tools.")
            return False

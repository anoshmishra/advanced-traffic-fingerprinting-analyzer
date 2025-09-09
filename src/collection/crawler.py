"""
Traffic collection and capture module
"""

import time
import subprocess
import threading
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import logging
import os
from pathlib import Path

class TrafficCrawler:
    """Automated web crawler with traffic capture"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.capture_process = None
        
    def setup_driver(self):
        """Setup Selenium WebDriver"""
        options = Options()
        if self.config['collection']['headless']:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        
        # Configure for Tor if enabled
        if self.config['collection']['use_tor']:
            options.add_argument('--proxy-server=socks5://127.0.0.1:9050')
        
        return webdriver.Chrome(options=options)
    
    def start_capture(self, output_file, interface):
        """Start packet capture using tshark"""
        cmd = [
            'tshark',
            '-i', interface,
            '-f', 'port 443 or port 80',  # Capture HTTPS and HTTP
            '-w', output_file,
            '-q'  # Quiet mode
        ]
        
        self.logger.info(f"Starting capture: {' '.join(cmd)}")
        self.capture_process = subprocess.Popen(cmd)
        return self.capture_process
    
    def stop_capture(self):
        """Stop packet capture"""
        if self.capture_process:
            self.capture_process.terminate()
            self.capture_process.wait()
            self.logger.info("Packet capture stopped")
    
    def visit_website(self, driver, url, duration=30):
        """Visit a website and wait for specified duration"""
        try:
            self.logger.info(f"Visiting {url}")
            driver.get(url)
            
            # Wait for page load
            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script('return document.readyState') == 'complete'
            )
            
            # Additional wait to capture full traffic
            time.sleep(duration)
            
        except Exception as e:
            self.logger.error(f"Error visiting {url}: {e}")
    
    def collect_traffic(self):
        """Main traffic collection function"""
        websites = self.config['collection']['target_websites']
        visits_per_site = self.config['collection']['visits_per_site']
        interface = self.config['collection']['interface']
        duration = self.config['collection']['capture_duration']
        
        # Create output directory
        output_dir = Path(self.config['paths']['data']) / 'raw_pcaps'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        driver = self.setup_driver()
        
        try:
            for i, website in enumerate(websites):
                site_name = website.split('//')[1].split('/')[0].replace('.', '_')
                
                for visit in range(visits_per_site):
                    # Create unique filename
                    pcap_file = output_dir / f"{site_name}_visit_{visit:03d}.pcap"
                    
                    self.logger.info(f"Collecting trace {visit+1}/{visits_per_site} for {site_name}")
                    
                    # Start packet capture
                    self.start_capture(str(pcap_file), interface)
                    
                    # Wait a moment for capture to start
                    time.sleep(1)
                    
                    # Visit website
                    self.visit_website(driver, website, duration)
                    
                    # Stop capture
                    self.stop_capture()
                    
                    # Brief pause between visits
                    time.sleep(2)
                    
                    # Create label file
                    label_file = pcap_file.with_suffix('.label')
                    with open(label_file, 'w') as f:
                        f.write(site_name)
        
        finally:
            driver.quit()
            self.logger.info("Traffic collection completed")

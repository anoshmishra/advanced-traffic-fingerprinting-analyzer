#!/usr/bin/env python3
import subprocess
from pathlib import Path

def run_command(cmd, description):
    print(f"Running: {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        if e.stdout: print(f"STDOUT: {e.stdout}")
        if e.stderr: print(f"STDERR: {e.stderr}")
        return False

def main():
    print("Setting up Encrypted Traffic Fingerprinting Project...")
    dirs_to_create = ['data/processed','data/raw_pcaps','results/models','results/plots','results/reports','logs']
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")
    print("\n" + "="*50); print("GENERATING SAMPLE DATA"); print("="*50)
    success = run_command("python3 generate_sample_data.py", "Generate sample data")
    if success:
        print("\n" + "="*50); print("RUNNING MAIN PROJECT"); print("="*50)
        run_command("python3 main.py", "Run main project")
    else:
        print("Failed to generate sample data. Please check errors above.")

if __name__ == "__main__":
    main()

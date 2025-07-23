#!/usr/bin/env python3
"""
Setup script for Market Survey System backend
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, check=True):
    """Run shell command"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result

def install_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    run_command("pip install -r requirements.txt")
    
    # Install Playwright browsers
    print("Installing Playwright browsers...")
    run_command("playwright install chromium")

def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "data/json",
        "logs",
        "scrapers",
        "models",
        "matchers"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def create_env_file():
    """Create .env file with default configuration"""
    env_content = """# Market Survey System Configuration

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Scraping Configuration
SCRAPE_DELAY=2.0
MAX_RETRIES=3
USER_AGENT=MarketSurveyBot/1.0

# Data Sources
MADLAN_BASE_URL=https://www.madlan.co.il
TAX_AUTHORITY_BASE_URL=https://www.gov.il/he/departments/taxes

# Storage
DATA_DIR=./data
JSON_OUTPUT_DIR=./data/json

# Matching Configuration
ADDRESS_MATCH_THRESHOLD=0.85
PRICE_CORRELATION_THRESHOLD=0.75

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/scraper.log
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    print("Created .env file")

def main():
    """Main setup function"""
    print("Setting up Market Survey System backend...")
    
    # Change to backend directory
    os.chdir("backend")
    
    # Create directories
    create_directories()
    
    # Create .env file
    create_env_file()
    
    # Install dependencies
    install_dependencies()
    
    print("\nSetup complete!")
    print("\nTo start the backend server:")
    print("1. cd backend")
    print("2. python main.py")
    print("\nThe API will be available at: http://localhost:8000")
    print("\nAPI Documentation: http://localhost:8000/docs")

if __name__ == "__main__":
    main()

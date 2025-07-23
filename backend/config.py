import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    
    # Scraping Configuration
    SCRAPE_DELAY = float(os.getenv("SCRAPE_DELAY", 2.0))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
    USER_AGENT = os.getenv("USER_AGENT", "MarketSurveyBot/1.0")
    
    # Data Sources
    MADLAN_BASE_URL = "https://www.madlan.co.il"
    TAX_AUTHORITY_BASE_URL = "https://www.gov.il/he/departments/taxes"
    
    # Storage
    DATA_DIR = os.getenv("DATA_DIR", "./data")
    JSON_OUTPUT_DIR = os.path.join(DATA_DIR, "json")
    
    # Matching Configuration
    ADDRESS_MATCH_THRESHOLD = float(os.getenv("ADDRESS_MATCH_THRESHOLD", 0.85))
    PRICE_CORRELATION_THRESHOLD = float(os.getenv("PRICE_CORRELATION_THRESHOLD", 0.75))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.path.join(DATA_DIR, "scraper.log")

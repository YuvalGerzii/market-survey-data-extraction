import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from playwright.async_api import async_playwright, Page
from bs4 import BeautifulSoup
import logging
import re

from models.project import Transaction, DataSource
from config import Config

logger = logging.getLogger(__name__)

class TaxAuthorityScraper:
    def __init__(self):
        self.config = Config()
        self.base_url = self.config.TAX_AUTHORITY_BASE_URL
        
    async def scrape_transactions(self, city: str = "תל אביב", 
                                start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Scrape real estate transactions from Israel Tax Authority"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=365)
        if not end_date:
            end_date = datetime.now()
            
        transactions = []
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=self.config.USER_AGENT,
                viewport={'width': 1920, 'height': 1080}
            )
            
            try:
                page = await context.new_page()
                
                # Navigate to tax authority real estate transactions
                await self._navigate_to_transactions(page)
                
                # Search for transactions in the specified city and date range
                search_results = await self._search_transactions(
                    page, city, start_date, end_date
                )
                
                for result in search_results:
                    try:
                        transaction = await self._extract_transaction_details(page, result)
                        if transaction:
                            transactions.append(transaction)
                    except Exception as e:
                        logger.error(f"Error extracting transaction: {str(e)}")
                        continue
                        
            finally:
                await browser.close()
                
        return transactions
    
    async def _navigate_to_transactions(self, page: Page):
        """Navigate to the real estate transactions section"""
        # This is a placeholder - actual implementation would depend on ITA website structure
        # For now, we'll simulate the navigation
        
        await page.goto(self.base_url, wait_until='networkidle')
        
        # Look for real estate/transactions section
        try:
            # Hebrew text for "real estate transactions"
            real_estate_link = await page.wait_for_selector(
                'text=עסקאות במקרקעין', timeout=10000
            )
            if real_estate_link:
                await real_estate_link.click()
                await page.wait_for_load_state('networkidle')
        except:
            logger.warning("Could not find real estate transactions link")
    
    async def _search_transactions(self, page: Page, city: str, 
                                 start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Search for transactions with filters"""
        search_results = []
        
        try:
            # Fill search form
            await page.wait_for_selector('input[name="city"]', timeout=10000)
            await page.fill('input[name="city"]', city)
            
            # Set date range
            start_str = start_date.strftime('%d/%m/%Y')
            end_str = end_date.strftime('%d/%m/%Y')
            
            await page.fill('input[name="start_date"]', start_str)
            await page.fill('input[name="end_date"]', end_str)
            
            # Submit search
            await page.click('button[type="submit"]')
            await page.wait_for_load_state('networkidle')
            
            # Extract results
            results = await page.query_selector_all('.transaction-row')
            
            for result in results:
                try:
                    row_data = await self._parse_transaction_row(result)
                    if row_data:
                        search_results.append(row_data)
                except Exception as e:
                    logger.error(f"Error parsing transaction row: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error during transaction search: {str(e)}")
            
        return search_results
    
    async def _parse_transaction_row(self, row_element) -> Optional[Dict[str, Any]]:
        """Parse a single transaction row"""
        try:
            text = await row_element.text_content()
            
            # Extract address
            address_match = re.search(r'(.+?)\s*,\s*([^,]+)$', text)
            if not address_match:
                return None
                
            full_address = address_match.group(0).strip()
            
            # Extract price
            price_match = re.search(r'₪([\d,]+)', text)
            if not price_match:
                return None
                
            price = int(price_match.group(1).replace(',', ''))
            
            # Extract date
            date_match = re.search(r'(\d{2}/\d{2}/\d{4})', text)
            if not date_match:
                return None
                
            sale_date = datetime.strptime(date_match.group(1), '%d/%m/%Y')
            
            # Extract additional details
            size_match = re.search(r'(\d+)\s*מ"ר', text)
            size = float(size_match.group(1)) if size_match else None
            
            floor_match = re.search(r'קומה\s+(\d+)', text)
            floor = int(floor_match.group(1)) if floor_match else None
            
            return {
                'address': full_address,
                'price': price,
                'sale_date': sale_date,
                'unit_size': size,
                'floor': floor,
                'source': DataSource.TAX_AUTHORITY
            }
            
        except Exception as e:
            logger.error(f"Error parsing transaction row: {str(e)}")
            return None
    
    async def _extract_transaction_details(self, page: Page, 
                                         transaction_data: Dict[str, Any]) -> Optional[Transaction]:
        """Extract detailed transaction information"""
        try:
            # Create transaction object
            transaction = Transaction(
                price=transaction_data['price'],
                sale_date=transaction_data['sale_date'],
                unit_size=transaction_data.get('unit_size'),
                floor=transaction_data.get('floor')
            )
            
            return transaction
            
        except Exception as e:
            logger.error(f"Error creating transaction: {str(e)}")
            return None
    
    def _normalize_address(self, address: str) -> str:
        """Normalize Hebrew address for better matching"""
        # Remove extra spaces and normalize
        address = re.sub(r'\s+', ' ', address.strip())
        
        # Common abbreviations
        replacements = {
            'רחוב': 'רח',
            'שדרות': 'שד',
            'הרב': 'הרב',
            'הרבנית': 'הרבנית'
        }
        
        for full, abbr in replacements.items():
            address = address.replace(full, abbr)
            
        return address
    
    def _extract_city_from_address(self, address: str) -> str:
        """Extract city name from full address"""
        # Common Israeli cities
        cities = [
            'תל אביב', 'ירושלים', 'חיפה', 'באר שבע', 'אשדוד', 'אשקלון',
            'פתח תקווה', 'נתניה', 'חולון', 'רמת גן', 'גבעתיים', 'ראשון לציון',
            'הרצליה', 'רעננה', 'כפר סבא', 'הוד השרון', 'רמת השרון'
        ]
        
        address_lower = address.lower()
        for city in cities:
            if city.lower() in address_lower:
                return city
                
        # Fallback - take last part after comma
        parts = address.split(',')
        if len(parts) > 1:
            return parts[-1].strip()
            
        return 'Unknown'

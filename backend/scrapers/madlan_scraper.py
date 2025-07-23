import asyncio
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from playwright.async_api import async_playwright, Page
from bs4 import BeautifulSoup
import logging

from models.project import Project, Transaction, DataSource
from config import Config

logger = logging.getLogger(__name__)

class MadlanScraper:
    def __init__(self):
        self.config = Config()
        self.base_url = self.config.MADLAN_BASE_URL
        
    async def scrape_projects(self, city: str = "tel-aviv") -> List[Project]:
        """Scrape real estate projects from Madlan for a specific city"""
        projects = []
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=self.config.USER_AGENT,
                viewport={'width': 1920, 'height': 1080}
            )
            
            try:
                page = await context.new_page()
                await self._handle_consent(page)
                
                # Navigate to city projects page
                city_url = f"{self.base_url}/projects/{city}"
                await page.goto(city_url, wait_until='networkidle')
                
                # Get all project links
                project_links = await self._get_project_links(page)
                
                for link in project_links[:10]:  # Limit for testing
                    try:
                        project = await self._scrape_project_details(page, link)
                        if project:
                            projects.append(project)
                    except Exception as e:
                        logger.error(f"Error scraping project {link}: {str(e)}")
                        continue
                        
            finally:
                await browser.close()
                
        return projects
    
    async def _handle_consent(self, page: Page):
        """Handle cookie consent popup if present"""
        try:
            await page.wait_for_selector('[data-testid="consent-accept"]', timeout=5000)
            await page.click('[data-testid="consent-accept"]')
            await page.wait_for_timeout(1000)
        except:
            pass
    
    async def _get_project_links(self, page: Page) -> List[str]:
        """Extract all project links from the city page"""
        await page.wait_for_selector('[data-testid="project-card"]', timeout=10000)
        
        project_elements = await page.query_selector_all('[data-testid="project-card"] a')
        links = []
        
        for element in project_elements:
            href = await element.get_attribute('href')
            if href and '/projects/' in href:
                full_url = f"{self.base_url}{href}" if href.startswith('/') else href
                links.append(full_url)
                
        return list(set(links))  # Remove duplicates
    
    async def _scrape_project_details(self, page: Page, project_url: str) -> Optional[Project]:
        """Scrape detailed information for a single project"""
        await page.goto(project_url, wait_until='networkidle')
        await page.wait_for_timeout(2000)
        
        # Get page content
        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract project information
        project_data = self._extract_project_info(soup, project_url)
        
        if not project_data.get('name'):
            return None
            
        # Create project object
        project = Project(
            project_name=project_data['name'],
            developer_name=project_data.get('developer'),
            address=project_data.get('address', ''),
            city=project_data.get('city', ''),
            unit_prices=project_data.get('unit_prices', {'min': 0, 'max': 0, 'avg': 0}),
            sources=[DataSource.MADLAN],
            metadata={
                'url': project_url,
                'project_type': project_data.get('type'),
                'construction_status': project_data.get('status'),
                'completion_year': project_data.get('completion_year')
            }
        )
        
        # Add transactions if available
        transactions = self._extract_transactions(soup)
        project.transactions.extend(transactions)
        
        # Calculate confidence score
        project.data_confidence_score = self._calculate_confidence(project_data)
        
        return project
    
    def _extract_project_info(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract project information from HTML"""
        data = {}
        
        # Project name
        name_elem = soup.find('h1') or soup.find('[data-testid="project-name"]')
        data['name'] = name_elem.text.strip() if name_elem else ''
        
        # Developer
        dev_elem = soup.find(text=re.compile(r'קבלן|מפתח', re.IGNORECASE))
        if dev_elem:
            data['developer'] = dev_elem.find_next().text.strip() if dev_elem.find_next() else None
        
        # Address
        addr_elem = soup.find(text=re.compile(r'כתובת|כתובת הפרויקט', re.IGNORECASE))
        if addr_elem:
            data['address'] = addr_elem.find_next().text.strip() if addr_elem.find_next() else ''
        
        # City extraction from URL
        url_parts = url.split('/')
        if 'projects' in url_parts:
            city_index = url_parts.index('projects') + 1
            if city_index < len(url_parts):
                data['city'] = url_parts[city_index].replace('-', ' ').title()
        
        # Price range
        price_text = soup.find(text=re.compile(r'₪[\d,]+', re.IGNORECASE))
        if price_text:
            prices = re.findall(r'₪([\d,]+)', price_text)
            if prices:
                prices = [int(p.replace(',', '')) for p in prices]
                data['unit_prices'] = {
                    'min': min(prices),
                    'max': max(prices),
                    'avg': sum(prices) // len(prices)
                }
        
        # Project status
        status_elem = soup.find(text=re.compile(r'סטטוס|מצב הפרויקט', re.IGNORECASE))
        if status_elem:
            data['status'] = status_elem.find_next().text.strip() if status_elem.find_next() else ''
        
        # Completion year
        year_match = re.search(r'(\d{4})', soup.text)
        if year_match:
            year = int(year_match.group(1))
            if 2020 <= year <= 2030:  # Reasonable range
                data['completion_year'] = year
        
        return data
    
    def _extract_transactions(self, soup: BeautifulSoup) -> List[Transaction]:
        """Extract transaction history if available"""
        transactions = []
        
        # Look for transaction tables or lists
        trans_sections = soup.find_all(['table', 'div'], text=re.compile(r'עסקאות|מכירות', re.IGNORECASE))
        
        for section in trans_sections:
            rows = section.find_all('tr') or section.find_all('div', class_=re.compile('transaction'))
            for row in rows:
                text = row.text
                price_match = re.search(r'₪([\d,]+)', text)
                date_match = re.search(r'(\d{2}/\d{2}/\d{4})', text)
                
                if price_match and date_match:
                    try:
                        price = int(price_match.group(1).replace(',', ''))
                        date = datetime.strptime(date_match.group(1), '%d/%m/%Y')
                        
                        transaction = Transaction(
                            price=price,
                            sale_date=date
                        )
                        transactions.append(transaction)
                    except ValueError:
                        continue
        
        return transactions
    
    def _calculate_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate confidence score based on data completeness"""
        required_fields = ['name', 'address', 'unit_prices']
        optional_fields = ['developer', 'status', 'completion_year']
        
        score = 0.0
        total_weight = 0.0
        
        # Required fields get higher weight
        for field in required_fields:
            if data.get(field):
                score += 0.25
            total_weight += 0.25
            
        # Optional fields get lower weight
        for field in optional_fields:
            if data.get(field):
                score += 0.1
            total_weight += 0.1
            
        return score / total_weight if total_weight > 0 else 0.0

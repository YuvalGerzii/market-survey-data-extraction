import asyncio
import logging
from typing import List, Dict, Any
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import re

from config import Config

logger = logging.getLogger(__name__)

class CityDiscovery:
    def __init__(self):
        self.config = Config()
        self.base_url = self.config.MADLAN_BASE_URL
        
    async def discover_available_cities(self) -> List[Dict[str, str]]:
        """Discover all available cities from Madlan website"""
        cities = []
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=self.config.USER_AGENT,
                viewport={'width': 1920, 'height': 1080}
            )
            
            try:
                page = await context.new_page()
                
                # Navigate to main page to find city links
                await page.goto(self.base_url, wait_until='networkidle')
                await page.wait_for_timeout(3000)
                
                # Look for city navigation or search functionality
                cities_found = await self._extract_cities_from_navigation(page)
                
                if not cities_found:
                    # Try alternative method - check projects page structure
                    cities_found = await self._extract_cities_from_projects_page(page)
                
                if not cities_found:
                    # Fallback to predefined list of major Israeli cities
                    cities_found = self._get_fallback_cities()
                
                logger.info(f"Discovered {len(cities_found)} cities from Madlan")
                return cities_found
                
            except Exception as e:
                logger.error(f"Error discovering cities: {str(e)}")
                return self._get_fallback_cities()
            finally:
                await browser.close()
    
    async def _extract_cities_from_navigation(self, page) -> List[Dict[str, str]]:
        """Extract cities from main navigation or dropdown"""
        cities = []
        
        try:
            # Look for city selector dropdown or navigation
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            
            # Common selectors for city navigation
            city_selectors = [
                'select[name*="city"] option',
                'select[name*="location"] option',
                '.city-selector option',
                '.location-dropdown option',
                'a[href*="/projects/"]'
            ]
            
            for selector in city_selectors:
                elements = soup.select(selector)
                if elements:
                    for element in elements:
                        city_info = self._parse_city_element(element)
                        if city_info:
                            cities.append(city_info)
                    break
            
            return self._clean_and_validate_cities(cities)
            
        except Exception as e:
            logger.error(f"Error extracting cities from navigation: {str(e)}")
            return []
    
    async def _extract_cities_from_projects_page(self, page) -> List[Dict[str, str]]:
        """Extract cities by analyzing projects page structure"""
        cities = []
        
        try:
            # Navigate to projects page
            projects_url = f"{self.base_url}/projects"
            await page.goto(projects_url, wait_until='networkidle')
            await page.wait_for_timeout(2000)
            
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            
            # Look for city-specific project links
            project_links = soup.find_all('a', href=re.compile(r'/projects/[^/]+$'))
            
            for link in project_links:
                href = link.get('href', '')
                city_slug = href.split('/')[-1] if href else ''
                
                if city_slug and len(city_slug) > 2:
                    city_name = self._slug_to_city_name(city_slug)
                    cities.append({
                        'name': city_name,
                        'slug': city_slug,
                        'hebrew_name': self._get_hebrew_name(city_name)
                    })
            
            return self._clean_and_validate_cities(cities)
            
        except Exception as e:
            logger.error(f"Error extracting cities from projects page: {str(e)}")
            return []
    
    def _parse_city_element(self, element) -> Dict[str, str]:
        """Parse city information from HTML element"""
        try:
            if element.name == 'option':
                value = element.get('value', '')
                text = element.get_text(strip=True)
                
                if value and text and value != '' and text.lower() not in ['select city', 'choose city', 'בחר עיר']:
                    return {
                        'name': self._normalize_city_name(text),
                        'slug': value,
                        'hebrew_name': text if self._is_hebrew(text) else ''
                    }
            
            elif element.name == 'a':
                href = element.get('href', '')
                text = element.get_text(strip=True)
                
                if '/projects/' in href:
                    city_slug = href.split('/')[-1]
                    return {
                        'name': self._slug_to_city_name(city_slug),
                        'slug': city_slug,
                        'hebrew_name': text if self._is_hebrew(text) else ''
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing city element: {str(e)}")
            return None
    
    def _get_fallback_cities(self) -> List[Dict[str, str]]:
        """Fallback list of major Israeli cities"""
        return [
            {'name': 'Tel Aviv', 'slug': 'tel-aviv', 'hebrew_name': 'תל אביב'},
            {'name': 'Jerusalem', 'slug': 'jerusalem', 'hebrew_name': 'ירושלים'},
            {'name': 'Haifa', 'slug': 'haifa', 'hebrew_name': 'חיפה'},
            {'name': 'Beer Sheva', 'slug': 'beer-sheva', 'hebrew_name': 'באר שבע'},
            {'name': 'Ashdod', 'slug': 'ashdod', 'hebrew_name': 'אשדוד'},
            {'name': 'Ashkelon', 'slug': 'ashkelon', 'hebrew_name': 'אשקלון'},
            {'name': 'Petah Tikva', 'slug': 'petah-tikva', 'hebrew_name': 'פתח תקווה'},
            {'name': 'Netanya', 'slug': 'netanya', 'hebrew_name': 'נתניה'},
            {'name': 'Holon', 'slug': 'holon', 'hebrew_name': 'חולון'},
            {'name': 'Ramat Gan', 'slug': 'ramat-gan', 'hebrew_name': 'רמת גן'},
            {'name': 'Givatayim', 'slug': 'givatayim', 'hebrew_name': 'גבעתיים'},
            {'name': 'Rishon LeZion', 'slug': 'rishon-lezion', 'hebrew_name': 'ראשון לציון'},
            {'name': 'Herzliya', 'slug': 'herzliya', 'hebrew_name': 'הרצליה'},
            {'name': 'Raanana', 'slug': 'raanana', 'hebrew_name': 'רעננה'},
            {'name': 'Kfar Saba', 'slug': 'kfar-saba', 'hebrew_name': 'כפר סבא'},
            {'name': 'Hod Hasharon', 'slug': 'hod-hasharon', 'hebrew_name': 'הוד השרון'},
            {'name': 'Ramat Hasharon', 'slug': 'ramat-hasharon', 'hebrew_name': 'רמת השרון'},
            {'name': 'Bat Yam', 'slug': 'bat-yam', 'hebrew_name': 'בת ים'},
            {'name': 'Rehovot', 'slug': 'rehovot', 'hebrew_name': 'רחובות'},
            {'name': 'Modiin', 'slug': 'modiin', 'hebrew_name': 'מודיעין'},
            {'name': 'Eilat', 'slug': 'eilat', 'hebrew_name': 'אילת'},
            {'name': 'Nazareth', 'slug': 'nazareth', 'hebrew_name': 'נצרת'},
            {'name': 'Acre', 'slug': 'acre', 'hebrew_name': 'עכו'},
            {'name': 'Tiberias', 'slug': 'tiberias', 'hebrew_name': 'טבריה'},
            {'name': 'Safed', 'slug': 'safed', 'hebrew_name': 'צפת'},
            {'name': 'Kiryat Shmona', 'slug': 'kiryat-shmona', 'hebrew_name': 'קריית שמונה'},
            {'name': 'Dimona', 'slug': 'dimona', 'hebrew_name': 'דימונה'},
            {'name': 'Arad', 'slug': 'arad', 'hebrew_name': 'ערד'},
            {'name': 'Kiryat Gat', 'slug': 'kiryat-gat', 'hebrew_name': 'קריית גת'},
            {'name': 'Lod', 'slug': 'lod', 'hebrew_name': 'לוד'},
            {'name': 'Ramla', 'slug': 'ramla', 'hebrew_name': 'רמלה'}
        ]
    
    def _clean_and_validate_cities(self, cities: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Clean and validate city list"""
        seen_slugs = set()
        cleaned_cities = []
        
        for city in cities:
            if city and city.get('slug') and city['slug'] not in seen_slugs:
                # Validate city name
                if len(city.get('name', '')) > 1 and len(city.get('slug', '')) > 1:
                    seen_slugs.add(city['slug'])
                    cleaned_cities.append(city)
        
        return sorted(cleaned_cities, key=lambda x: x['name'])
    
    def _normalize_city_name(self, name: str) -> str:
        """Normalize city name"""
        if not name:
            return ''
        
        # Remove extra whitespace
        name = ' '.join(name.split())
        
        # Convert Hebrew to English if needed
        hebrew_to_english = {
            'תל אביב': 'Tel Aviv',
            'ירושלים': 'Jerusalem',
            'חיפה': 'Haifa',
            'באר שבע': 'Beer Sheva',
            'אשדוד': 'Ashdod',
            'אשקלון': 'Ashkelon',
            'פתח תקווה': 'Petah Tikva',
            'נתניה': 'Netanya',
            'חולון': 'Holon',
            'רמת גן': 'Ramat Gan'
        }
        
        return hebrew_to_english.get(name, name)
    
    def _slug_to_city_name(self, slug: str) -> str:
        """Convert slug to readable city name"""
        if not slug:
            return ''
        
        # Replace hyphens with spaces and title case
        return slug.replace('-', ' ').title()
    
    def _get_hebrew_name(self, english_name: str) -> str:
        """Get Hebrew name for English city name"""
        english_to_hebrew = {
            'Tel Aviv': 'תל אביב',
            'Jerusalem': 'ירושלים',
            'Haifa': 'חיפה',
            'Beer Sheva': 'באר שבע',
            'Ashdod': 'אשדוד',
            'Ashkelon': 'אשקלון',
            'Petah Tikva': 'פתח תקווה',
            'Netanya': 'נתניה',
            'Holon': 'חולון',
            'Ramat Gan': 'רמת גן'
        }
        
        return english_to_hebrew.get(english_name, '')
    
    def _is_hebrew(self, text: str) -> bool:
        """Check if text contains Hebrew characters"""
        if not text:
            return False
        
        hebrew_chars = 0
        for char in text:
            if '\u0590' <= char <= '\u05FF':  # Hebrew Unicode range
                hebrew_chars += 1
        
        return hebrew_chars > 0

    async def verify_city_availability(self, city_slug: str) -> bool:
        """Verify if a city has available projects on Madlan"""
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent=self.config.USER_AGENT,
                    viewport={'width': 1920, 'height': 1080}
                )
                
                page = await context.new_page()
                city_url = f"{self.base_url}/projects/{city_slug}"
                
                response = await page.goto(city_url, wait_until='networkidle')
                
                if response.status == 200:
                    # Check if page has projects
                    content = await page.content()
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Look for project cards or listings
                    project_indicators = [
                        '[data-testid="project-card"]',
                        '.project-card',
                        '.project-item',
                        'a[href*="/project/"]'
                    ]
                    
                    for selector in project_indicators:
                        if soup.select(selector):
                            await browser.close()
                            return True
                    
                    # Check if there's a "no projects" message
                    no_projects_indicators = [
                        'no projects',
                        'אין פרויקטים',
                        'לא נמצאו פרויקטים'
                    ]
                    
                    page_text = soup.get_text().lower()
                    for indicator in no_projects_indicators:
                        if indicator in page_text:
                            await browser.close()
                            return False
                    
                    # If we can't determine, assume it's available
                    await browser.close()
                    return True
                
                await browser.close()
                return False
                
        except Exception as e:
            logger.error(f"Error verifying city {city_slug}: {str(e)}")
            return False

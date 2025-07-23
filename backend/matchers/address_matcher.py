from typing import List, Tuple, Dict, Any
from fuzzywuzzy import fuzz, process
import re
from difflib import SequenceMatcher
import logging

from models.project import Project
from config import Config

logger = logging.getLogger(__name__)

class AddressMatcher:
    def __init__(self):
        self.config = Config()
        self.threshold = self.config.ADDRESS_MATCH_THRESHOLD
        
    def match_projects_with_transactions(self, 
                                       projects: List[Project], 
                                       transactions: List[Dict[str, Any]]) -> List[Project]:
        """Match projects with transactions based on address similarity"""
        
        # Create address index for faster matching
        project_addresses = [(p, self._normalize_address(p.address)) for p in projects]
        transaction_addresses = [(t, self._normalize_address(t['address'])) for t in transactions]
        
        matched_projects = []
        
        for project, project_addr in project_addresses:
            # Find matching transactions
            matching_transactions = self._find_matching_transactions(
                project_addr, 
                transaction_addresses
            )
            
            # Add transactions to project
            for transaction_data, score in matching_transactions:
                if score >= self.threshold:
                    transaction = self._create_transaction_from_data(transaction_data)
                    project.transactions.append(transaction)
                    
                    # Update price range
                    self._update_price_range(project, transaction.price)
                    
                    # Add ITA as source if not already present
                    if 'ita' not in [s.value for s in project.sources]:
                        project.sources.append('ita')
            
            # Recalculate confidence score
            project.data_confidence_score = self._recalculate_confidence(project)
            
            matched_projects.append(project)
            
        return matched_projects
    
    def _normalize_address(self, address: str) -> str:
        """Normalize address for better matching"""
        if not address:
            return ""
            
        # Convert to lowercase
        address = address.lower().strip()
        
        # Remove special characters and extra spaces
        address = re.sub(r'[^\w\s]', ' ', address)
        address = re.sub(r'\s+', ' ', address)
        
        # Hebrew street type abbreviations
        street_types = {
            'רחוב': 'רח',
            'שדרות': 'שד',
            'דרך': 'דר',
            'הרב': 'הרב',
            'הרבנית': 'הרבנית',
            'הגאון': 'הגאון',
            'הגאונים': 'הגאונים'
        }
        
        for full, abbr in street_types.items():
            address = address.replace(full.lower(), abbr.lower())
        
        # Remove common prefixes
        prefixes = ['רח ', 'שד ', 'דר ', 'הרב ', 'הרבנית ']
        for prefix in prefixes:
            if address.startswith(prefix):
                address = address[len(prefix):]
                break
        
        # Handle building numbers
        address = re.sub(r'(\d+)(?:\s*[-–]\s*\d+)?', r'\1', address)
        
        return address.strip()
    
    def _find_matching_transactions(self, 
                                  project_address: str, 
                                  transaction_addresses: List[Tuple[Dict[str, Any], str]]) -> List[Tuple[Dict[str, Any], float]]:
        """Find transactions that match the project address"""
        
        matches = []
        
        # Use fuzzy matching for better results
        transaction_strings = [addr for _, addr in transaction_addresses]
        transaction_data = [data for data, _ in transaction_addresses]
        
        # Get best matches
        best_matches = process.extract(
            project_address, 
            transaction_strings, 
            scorer=fuzz.partial_ratio,
            limit=10
        )
        
        for match_str, score, idx in best_matches:
            if score >= self.threshold:
                transaction = transaction_data[idx]
                matches.append((transaction, score))
        
        return matches
    
    def _create_transaction_from_data(self, transaction_data: Dict[str, Any]) -> Any:
        """Create Transaction object from scraped data"""
        from models.project import Transaction
        
        return Transaction(
            price=transaction_data['price'],
            sale_date=transaction_data['sale_date'],
            unit_size=transaction_data.get('unit_size'),
            floor=transaction_data.get('floor'),
            buyer_type=transaction_data.get('buyer_type')
        )
    
    def _update_price_range(self, project: Project, new_price: int):
        """Update price range statistics with new transaction"""
        prices = [t.price for t in project.transactions]
        
        if prices:
            project.unit_prices = {
                'min': min(prices),
                'max': max(prices),
                'avg': sum(prices) // len(prices)
            }
    
    def _recalculate_confidence(self, project: Project) -> float:
        """Recalculate confidence score based on data completeness"""
        score = 0.0
        
        # Base score from required fields
        if project.project_name:
            score += 0.2
        if project.address:
            score += 0.2
        if project.developer_name:
            score += 0.1
        if project.coordinates:
            score += 0.1
            
        # Score from transactions
        if project.transactions:
            score += min(len(project.transactions) * 0.05, 0.3)
            
        # Score from price data
        if project.unit_prices['min'] > 0:
            score += 0.1
            
        # Multiple sources bonus
        if len(project.sources) > 1:
            score += 0.1
            
        return min(score, 1.0)
    
    def find_similar_projects(self, projects: List[Project], 
                            target_project: Project) -> List[Tuple[Project, float]]:
        """Find similar projects based on address and developer"""
        
        target_addr = self._normalize_address(target_project.address)
        target_dev = target_project.developer_name or ""
        
        similarities = []
        
        for project in projects:
            if project == target_project:
                continue
                
            addr_sim = fuzz.ratio(
                target_addr, 
                self._normalize_address(project.address)
            )
            
            dev_sim = 0
            if target_dev and project.developer_name:
                dev_sim = fuzz.ratio(
                    target_dev.lower(), 
                    project.developer_name.lower()
                )
            
            # Weighted similarity
            total_sim = (addr_sim * 0.7) + (dev_sim * 0.3)
            
            if total_sim >= 0.7:  # Similarity threshold
                similarities.append((project, total_sim))
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities
    
    def validate_address_format(self, address: str) -> Dict[str, Any]:
        """Validate and parse address format"""
        result = {
            'is_valid': False,
            'street': None,
            'number': None,
            'city': None,
            'normalized': None
        }
        
        if not address:
            return result
            
        # Hebrew address pattern
        pattern = r'^(?:רח|שד|דרך)?\s*([^\d]+)\s*(\d+(?:[א-ת])?)(?:\s*,\s*([^,]+))?$'
        match = re.match(pattern, address.strip())
        
        if match:
            result.update({
                'is_valid': True,
                'street': match.group(1).strip(),
                'number': match.group(2).strip(),
                'city': match.group(3).strip() if match.group(3) else None,
                'normalized': f"{match.group(1).strip()} {match.group(2).strip()}"
            })
        
        return result

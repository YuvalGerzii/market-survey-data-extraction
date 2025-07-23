import os
import json
import logging
from typing import List, Dict, Any, Optional
import httpx
from datetime import datetime

from models.project import Project

logger = logging.getLogger(__name__)

class AIInsightsGenerator:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "openai/gpt-4o"
        self.system_prompt = self._get_default_system_prompt()
        
    def _get_default_system_prompt(self) -> str:
        """Default system prompt for market analysis"""
        return """You are a real estate market analyst with expertise in Israeli property markets. 
        Analyze the provided project data and generate comprehensive insights including:
        
        1. Market trends and patterns
        2. Price analysis and comparisons
        3. Data quality assessment
        4. Geographic distribution insights
        5. Developer activity analysis
        6. Investment recommendations
        
        Provide clear, actionable insights in a professional tone. Focus on data-driven conclusions 
        and highlight any notable patterns or anomalies in the market data."""
    
    async def generate_insights(self, 
                              projects: List[Project], 
                              custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate AI-powered market insights from project data"""
        
        if not self.api_key:
            logger.warning("OpenRouter API key not found. AI insights disabled.")
            return {
                "success": False,
                "error": "AI insights require API key configuration",
                "insights": "AI insights are currently unavailable. Please configure OPENROUTER_API_KEY."
            }
        
        if not projects:
            return {
                "success": True,
                "insights": "No project data available for analysis.",
                "metadata": {
                    "projects_analyzed": 0,
                    "generated_at": datetime.now().isoformat()
                }
            }
        
        try:
            # Prepare data summary for analysis
            data_summary = self._prepare_data_summary(projects)
            
            # Use custom prompt if provided, otherwise use default
            system_prompt = custom_prompt or self.system_prompt
            
            # Generate insights using LLM
            insights = await self._call_llm(system_prompt, data_summary)
            
            return {
                "success": True,
                "insights": insights,
                "metadata": {
                    "projects_analyzed": len(projects),
                    "generated_at": datetime.now().isoformat(),
                    "model_used": self.model
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "insights": "Unable to generate insights due to technical error."
            }
    
    def _prepare_data_summary(self, projects: List[Project]) -> str:
        """Prepare a structured summary of project data for LLM analysis"""
        
        # Calculate aggregate statistics
        total_projects = len(projects)
        total_transactions = sum(len(p.transactions) for p in projects)
        
        # Price statistics
        all_prices = []
        for project in projects:
            if project.unit_prices.get('avg', 0) > 0:
                all_prices.append(project.unit_prices['avg'])
        
        avg_price = sum(all_prices) / len(all_prices) if all_prices else 0
        min_price = min(all_prices) if all_prices else 0
        max_price = max(all_prices) if all_prices else 0
        
        # Confidence statistics
        confidence_scores = [p.data_confidence_score for p in projects if p.data_confidence_score > 0]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # City distribution
        city_counts = {}
        for project in projects:
            city = project.city or "Unknown"
            city_counts[city] = city_counts.get(city, 0) + 1
        
        # Developer analysis
        developer_counts = {}
        for project in projects:
            dev = project.developer_name or "Unknown"
            developer_counts[dev] = developer_counts.get(dev, 0) + 1
        
        # Data source analysis
        source_counts = {}
        for project in projects:
            for source in project.sources:
                source_counts[source] = source_counts.get(source, 0) + 1
        
        # Recent projects (last 30 days)
        recent_cutoff = datetime.now().timestamp() - (30 * 24 * 60 * 60)
        recent_projects = [
            p for p in projects 
            if p.last_updated.timestamp() > recent_cutoff
        ]
        
        # Build comprehensive summary
        summary = f"""
REAL ESTATE MARKET DATA ANALYSIS

OVERVIEW:
- Total Projects: {total_projects}
- Total Transactions: {total_transactions}
- Recent Projects (30 days): {len(recent_projects)}

PRICE ANALYSIS:
- Average Unit Price: ₪{avg_price:,.0f}
- Price Range: ₪{min_price:,.0f} - ₪{max_price:,.0f}
- Projects with Price Data: {len(all_prices)}

DATA QUALITY:
- Average Confidence Score: {avg_confidence:.2%}
- High Confidence Projects (>80%): {len([s for s in confidence_scores if s > 0.8])}
- Medium Confidence Projects (60-80%): {len([s for s in confidence_scores if 0.6 <= s <= 0.8])}
- Low Confidence Projects (<60%): {len([s for s in confidence_scores if s < 0.6])}

GEOGRAPHIC DISTRIBUTION:
{self._format_dict_summary(city_counts, "Cities")}

TOP DEVELOPERS:
{self._format_dict_summary(developer_counts, "Developers", limit=10)}

DATA SOURCES:
{self._format_dict_summary(source_counts, "Sources")}

SAMPLE PROJECTS:
"""
        
        # Add sample project details
        sample_projects = projects[:5]  # First 5 projects as samples
        for i, project in enumerate(sample_projects, 1):
            summary += f"""
{i}. {project.project_name}
   - Developer: {project.developer_name or 'Unknown'}
   - Location: {project.address}, {project.city}
   - Price Range: ₪{project.unit_prices.get('min', 0):,} - ₪{project.unit_prices.get('max', 0):,}
   - Transactions: {len(project.transactions)}
   - Confidence: {project.data_confidence_score:.1%}
   - Sources: {', '.join(project.sources)}
"""
        
        return summary
    
    def _format_dict_summary(self, data_dict: Dict[str, int], title: str, limit: int = 5) -> str:
        """Format dictionary data for summary"""
        if not data_dict:
            return f"- No {title.lower()} data available"
        
        sorted_items = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)[:limit]
        formatted = []
        for key, count in sorted_items:
            formatted.append(f"- {key}: {count}")
        
        return "\n".join(formatted)
    
    async def _call_llm(self, system_prompt: str, data_summary: str) -> str:
        """Make API call to OpenRouter LLM"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Please analyze this real estate market data and provide comprehensive insights:\n\n{data_summary}"
                        }
                    ]
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                raise Exception("No response content from LLM")
    
    def update_system_prompt(self, new_prompt: str):
        """Update the system prompt for customized analysis"""
        self.system_prompt = new_prompt
        logger.info("System prompt updated for AI insights")
    
    def get_system_prompt(self) -> str:
        """Get current system prompt"""
        return self.system_prompt

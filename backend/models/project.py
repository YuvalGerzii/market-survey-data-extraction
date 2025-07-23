from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class DataSource(str, Enum):
    MADLAN = "madlan"
    TAX_AUTHORITY = "ita"
    COMBINED = "combined"

class Transaction(BaseModel):
    price: int = Field(..., description="Transaction price in ILS")
    sale_date: datetime = Field(..., description="Date of sale")
    unit_size: Optional[float] = Field(None, description="Unit size in square meters")
    floor: Optional[int] = Field(None, description="Floor number")
    buyer_type: Optional[str] = Field(None, description="Individual/Company")
    
class Project(BaseModel):
    project_name: str = Field(..., description="Name of the real estate project")
    developer_name: Optional[str] = Field(None, description="Name of the development company")
    address: str = Field(..., description="Full street address")
    city: str = Field(..., description="City name")
    coordinates: Optional[Dict[str, float]] = Field(None, description="Latitude and longitude")
    
    unit_prices: Dict[str, int] = Field(
        default_factory=lambda: {"min": 0, "max": 0, "avg": 0},
        description="Price statistics"
    )
    
    transactions: List[Transaction] = Field(
        default_factory=list,
        description="List of historical transactions"
    )
    
    data_confidence_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score based on data completeness and source reliability"
    )
    
    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="Last time this data was updated"
    )
    
    sources: List[DataSource] = Field(
        default_factory=list,
        description="Data sources used for this project"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata like project status, construction year, etc."
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ScrapeStatus(BaseModel):
    source: str
    status: str
    projects_found: int
    errors: List[str] = []
    timestamp: datetime = Field(default_factory=datetime.now)

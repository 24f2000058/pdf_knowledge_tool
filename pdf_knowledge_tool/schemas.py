from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class ChunkMetadata(BaseModel):
    """Schema for individual text chunks stored in ChromaDB."""
    pdf_id: str
    chunk_index: int
    data_source: str = "docling_parse"
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    
    # Allow extra fields for flexibility
    model_config = {"extra": "allow"}

class DocumentMetadata(BaseModel):
    """Schema for document-level metadata stored in TinyDB."""
    id: str # The unique pdf_id
    filename: str
    file_hash: str # SHA256 for delta updates
    upload_date: datetime = Field(default_factory=datetime.now)
    total_chunks: int
    status: str = "indexed"
    
    # Store extraction results
    images: List[str] = [] # Paths to images
    tables: List[str] = [] # HTML content of tables
    
    model_config = {"extra": "allow"}

class SearchResult(BaseModel):
    """Schema for returning search results to the UI."""
    id: str
    score: float
    text: str
    metadata: Dict[str, Any] # Use dict for flexibility or ChunkMetadata if strictly validated
    
    model_config = {"extra": "allow"}

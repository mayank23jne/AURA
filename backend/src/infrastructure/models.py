from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

class StoredModel(Base):
    __tablename__ = "models"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    model_type = Column(String(50), nullable=False) # api, ollama, huggingface
    provider = Column(String(50), nullable=False) # openai, anthropic, etc.
    
    # Configuration
    model_name = Column(String(255), nullable=True) # e.g. gpt-4
    endpoint_url = Column(String(1024), nullable=True)
    api_key = Column(String(1024), nullable=True)
    temperature = Column(Float, default=0.7)
    max_tokens = Column(Integer, default=4096)
    
    # Metadata
    status = Column(String(50), default="active")
    tags = Column(JSON, default=list)
    last_audit = Column(String(255), default="Never")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class StoredPolicy(Base):
    __tablename__ = "policies"

    id = Column(String(255), primary_key=True) # User provided ID
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    category = Column(String(100), nullable=False)
    
    # Content
    rules = Column(JSON, default=list) # List of dicts or strings
    regulatory_references = Column(JSON, default=list)
    version = Column(String(50), default="1.0.0")
    active = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

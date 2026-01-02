from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, JSON, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime

Base = declarative_base()

class ORMModel(Base):
    __tablename__ = "models"

    id = Column(String(50), primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    model_type = Column(String(50), nullable=False)
    provider = Column(String(50), nullable=False)
    
    # Connection settings
    endpoint_url = Column(String(512), nullable=True)
    api_key = Column(String(512), nullable=True)
    model_name = Column(String(255), nullable=True)
    
    # For uploaded models
    file_path = Column(String(512), nullable=True)
    file_format = Column(String(50), nullable=True)
    
    # Model parameters
    temperature = Column(Float, default=0.7)
    max_tokens = Column(Integer, default=4096)
    timeout_seconds = Column(Integer, default=60)
    
    # Metadata
    status = Column(String(50), default="inactive")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_tested = Column(DateTime, nullable=True)
    test_result = Column(Text, nullable=True)
    tags = Column(JSON, default=list)
    
    # Usage tracking
    total_requests = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    avg_latency_ms = Column(Float, default=0.0)
    
    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class ORMPackage(Base):
    __tablename__ = "packages"
    
    id = Column(String(50), primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    policies = relationship("ORMPolicy", back_populates="package")

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class ORMPolicy(Base):
    __tablename__ = "policies"

    id = Column(String(50), primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(100), nullable=False)
    severity = Column(String(100), nullable=False,  default="medium")
    version = Column(String(50), default="1.0.0")
    active = Column(Boolean, default=True)
    
    package_id = Column(String(50), ForeignKey('packages.id'), nullable=True)
    package = relationship("ORMPackage", back_populates="policies")
    
    rules = Column(JSON, default=list)
    test_specifications = Column(JSON, default=list)
    regulatory_references = Column(JSON, default=list)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class ORMAudit(Base):
    __tablename__ = "audits"
    
    audit_id = Column(String(50), primary_key=True)
    model_id = Column(String(50), ForeignKey('models.id'), nullable=True)
    
    status = Column(String(50), default="unknown")
    compliance_score = Column(Float, default=0.0)
    risk_score = Column(Float, default=0.0)
    
    audit_data = Column(JSON, nullable=False) # Stores the full audit result dict
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    saved_at = Column(DateTime, default=datetime.datetime.utcnow)

    def to_dict(self):
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        # Merge the text fields back into the main dict if needed, 
        # but typically the audit_data JSON is the source of truth for the blob
        if self.audit_data:
            d.update(self.audit_data)
        return d

# Compliance structures might be complex to map fully relational right now without major refactor,
# so we will store them as JSON blobs or simplified tables for now to get persistence working fast.
class ORMComplianceScore(Base):
    __tablename__ = "compliance_scores"
    
    id = Column(String(50), primary_key=True)
    model_id = Column(String(50), ForeignKey('models.id'))
    
    overall_risk_score = Column(Float, default=0.0)
    overall_compliance_score = Column(Float, default=0.0)
    status = Column(String(50))
    
    domain_scores = Column(JSON, default=list)
    framework_scores = Column(JSON, default=list)
    risk_factors = Column(JSON, default=list)
    mitigating_factors = Column(JSON, default=list)
    recommendations = Column(JSON, default=dict) # High/Med/Low actions
    
    calculated_at = Column(DateTime, default=datetime.datetime.utcnow)
    next_review = Column(DateTime, nullable=True)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class ORMAlert(Base):
    __tablename__ = "alerts"
    
    id = Column(String(50), primary_key=True)
    model_id = Column(String(50), ForeignKey('models.id'))
    
    type = Column(String(50))
    severity = Column(String(20))
    details = Column(Text)
    status = Column(String(20), default="open")
    
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class ORMMetric(Base):
    __tablename__ = "metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(50), ForeignKey('models.id'))
    
    name = Column(String(100))
    value = Column(Float)
    tags = Column(JSON, default=dict)
    
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class ORMLearning(Base):
    __tablename__ = "learning_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    audit_id = Column(String(50), ForeignKey('audits.audit_id'), nullable=True)
    
    insights = Column(JSON)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class ORMStrategy(Base):
    __tablename__ = "strategies"
    
    type = Column(String(50), primary_key=True)
    optimization = Column(JSON)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class ORMWorkflowExecution(Base):
    __tablename__ = "workflow_executions"
    
    execution_id = Column(String(50), primary_key=True)
    workflow_id = Column(String(50), nullable=False)
    status = Column(String(20))
    current_node = Column(String(50))
    
    start_time = Column(DateTime)
    end_time = Column(DateTime, nullable=True)
    
    state_data = Column(JSON) # Stores full WorkflowState
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

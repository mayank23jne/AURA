"""Data models for AURA Agentic Platform"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence
from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """Types of messages exchanged between agents"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    KNOWLEDGE_SHARE = "knowledge_share"
    DECISION_REQUEST = "decision_request"
    COORDINATION = "coordination"
    ALERT = "alert"
    LEARNING_UPDATE = "learning_update"
    COLLECTIVE_DECISION_REQUEST = "collective_decision_request"


class AgentStatus(str, Enum):
    """Agent operational status"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


class KnowledgeType(str, Enum):
    """Types of knowledge stored in the knowledge base"""
    RULE = "rule"
    PATTERN = "pattern"
    EXPERIENCE = "experience"
    DECISION = "decision"
    INSIGHT = "insight"


class AgentConfig(BaseModel):
    """Configuration for an agent"""
    name: str
    llm_model: str = "gpt-4"
    llm_provider: str = "openai"
    temperature: float = 0.7
    max_tokens: int = 1000
    memory_type: str = "conversation_summary"
    tools_enabled: List[str] = Field(default_factory=list)
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 100000
    retry_attempts: int = 3
    timeout_seconds: int = 60


class AgentMessage(BaseModel):
    """Message exchanged between agents"""
    id: str = Field(default_factory=lambda: str(datetime.now().timestamp()))
    message_type: MessageType
    source_agent: str
    target_agent: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    payload: Dict[str, Any] = Field(default_factory=dict)
    correlation_id: Optional[str] = None
    priority: int = 5  # 1-10, higher is more urgent
    ttl_seconds: int = 3600


class TaskRequest(BaseModel):
    """Request to perform a task"""
    task_id: str
    task_type: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 5
    deadline: Optional[datetime] = None
    requester: str
    context: Dict[str, Any] = Field(default_factory=dict)


class TaskResponse(BaseModel):
    """Response from a completed task"""
    task_id: str
    status: str  # success, failure, partial
    result: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    execution_time_ms: int = 0
    agent_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class KnowledgeItem(BaseModel):
    """Item stored in the knowledge base"""
    id: str
    knowledge_type: KnowledgeType
    domain: str
    content: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_agent: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = 1.0
    version: int = 1
    embedding: Optional[List[float]] = None
    tags: List[str] = Field(default_factory=list)


class AuditState(BaseModel):
    """State maintained during an audit workflow"""
    messages: Sequence[str] = Field(default_factory=list)
    current_stage: str = "initialization"
    policies: List[Dict[str, Any]] = Field(default_factory=list)
    test_results: Dict[str, Any] = Field(default_factory=dict)
    analysis: Dict[str, Any] = Field(default_factory=dict)
    report: Dict[str, Any] = Field(default_factory=dict)
    next_action: str = "policy_selection"
    model_id: Optional[str] = None
    audit_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class ComplianceResult(BaseModel):
    """Result of a compliance test"""
    test_id: str
    policy_id: str
    model_id: str
    passed: bool
    score: float
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    validation_method: str = "llm"  # llm, deterministic
    signature: Optional[str] = None


class PolicyDefinition(BaseModel):
    """Definition of a compliance policy"""
    id: str
    name: str
    description: str
    category: str
    severity: str
    package_id: Optional[str] = None
    rules: List[Dict[str, Any]] = Field(default_factory=list)
    test_specifications: List[Dict[str, Any]] = Field(default_factory=list)
    regulatory_references: List[str] = Field(default_factory=list)
    version: str = "1.0.0"
    active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class PackageDefinition(BaseModel):
    """Definition of a policy package"""
    id: str
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    policies_count: int = 0


class PackageCreateRequest(BaseModel):
    """Request to create a new package"""
    id: str
    name: str
    description: Optional[str] = None


class PackageUpdateRequest(BaseModel):
    """Request to update a package"""
    name: Optional[str] = None
    description: Optional[str] = None


class TestCase(BaseModel):
    """A test case for compliance testing"""
    id: str
    policy_id: str
    name: str
    description: str
    input_data: Dict[str, Any]
    expected_behavior: str
    test_type: str  # adversarial, edge_case, standard, synthetic
    severity: str = "medium"
    tags: List[str] = Field(default_factory=list)
    validation_rule: Optional[Dict[str, Any]] = None  # {type: 'regex', params: {...}}


class AgentMetrics(BaseModel):
    """Performance metrics for an agent"""
    agent_id: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_response_time_ms: float = 0.0
    uptime_seconds: int = 0
    memory_usage_mb: float = 0.0
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    custom_metrics: Dict[str, float] = Field(default_factory=dict)


class Vote(BaseModel):
    """A vote cast by an agent in a collective decision"""
    agent_id: str
    decision: str
    confidence: float
    reasoning: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConsensusResult(BaseModel):
    """Result of a consensus process"""
    decision_id: str
    final_decision: str
    consensus_score: float  # 0.0 to 1.0
    participating_agents: int
    votes: List[Vote]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CollectiveDecisionRequest(BaseModel):
    """Request for a collective decision"""
    decision_id: str
    context: Dict[str, Any]
    options: List[str]
    required_consensus_threshold: float = 0.7
    min_participating_agents: int = 3
    deadline: Optional[datetime] = None

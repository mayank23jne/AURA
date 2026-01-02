# AURA Platform Data Models

## Overview

AURA uses Pydantic models for data validation and serialization. All models are defined in `src/core/models.py`.

## Enumerations

### MessageType
Types of messages exchanged between agents.

```python
class MessageType(str, Enum):
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    KNOWLEDGE_SHARE = "knowledge_share"
    DECISION_REQUEST = "decision_request"
    COORDINATION = "coordination"
    ALERT = "alert"
    LEARNING_UPDATE = "learning_update"
```

### AgentStatus
Agent operational status.

```python
class AgentStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"
```

### KnowledgeType
Types of knowledge stored in the knowledge base.

```python
class KnowledgeType(str, Enum):
    RULE = "rule"
    PATTERN = "pattern"
    EXPERIENCE = "experience"
    DECISION = "decision"
    INSIGHT = "insight"
```

## Core Models

### AgentConfig
Configuration for an agent.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| name | str | required | Agent name |
| llm_model | str | "gpt-4" | LLM model to use |
| llm_provider | str | "openai" | LLM provider |
| temperature | float | 0.7 | LLM temperature |
| max_tokens | int | 4096 | Max tokens per response |
| memory_type | str | "conversation_summary" | Memory type |
| tools_enabled | List[str] | [] | Enabled tools |
| rate_limit_rpm | int | 60 | Requests per minute |
| rate_limit_tpm | int | 100000 | Tokens per minute |
| retry_attempts | int | 3 | Number of retries |
| timeout_seconds | int | 60 | Request timeout |

### AgentMessage
Message exchanged between agents.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| id | str | auto-generated | Message ID |
| message_type | MessageType | required | Type of message |
| source_agent | str | required | Sending agent |
| target_agent | Optional[str] | None | Target agent |
| timestamp | datetime | now | Creation time |
| payload | Dict[str, Any] | {} | Message data |
| correlation_id | Optional[str] | None | Request correlation |
| priority | int | 5 | Priority (1-10) |
| ttl_seconds | int | 3600 | Time to live |

### TaskRequest
Request to perform a task.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| task_id | str | required | Unique task ID |
| task_type | str | required | Type of task |
| description | str | required | Task description |
| parameters | Dict[str, Any] | {} | Task parameters |
| priority | int | 5 | Task priority |
| deadline | Optional[datetime] | None | Task deadline |
| requester | str | required | Requesting entity |
| context | Dict[str, Any] | {} | Additional context |

### TaskResponse
Response from a completed task.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| task_id | str | required | Original task ID |
| status | str | required | success/failure/partial |
| result | Dict[str, Any] | {} | Task result |
| error | Optional[str] | None | Error message |
| execution_time_ms | int | 0 | Execution duration |
| agent_id | str | required | Executing agent |
| timestamp | datetime | now | Completion time |

### KnowledgeItem
Item stored in the knowledge base.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| id | str | required | Item ID |
| knowledge_type | KnowledgeType | required | Type of knowledge |
| domain | str | required | Knowledge domain |
| content | Dict[str, Any] | required | Knowledge content |
| metadata | Dict[str, Any] | {} | Additional metadata |
| source_agent | str | required | Source agent |
| timestamp | datetime | now | Creation time |
| confidence | float | 1.0 | Confidence score |
| version | int | 1 | Version number |
| embedding | Optional[List[float]] | None | Vector embedding |
| tags | List[str] | [] | Tags for search |

## Audit Models

### AuditState
State maintained during an audit workflow.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| messages | Sequence[str] | [] | Workflow messages |
| current_stage | str | "initialization" | Current stage |
| policies | List[Dict] | [] | Selected policies |
| test_results | Dict[str, Any] | {} | Test results |
| analysis | Dict[str, Any] | {} | Analysis results |
| report | Dict[str, Any] | {} | Generated report |
| next_action | str | "policy_selection" | Next action |
| model_id | Optional[str] | None | Target model |
| audit_id | Optional[str] | None | Audit ID |
| start_time | Optional[datetime] | None | Start time |
| end_time | Optional[datetime] | None | End time |

### ComplianceResult
Result of a compliance test.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| test_id | str | required | Test ID |
| policy_id | str | required | Policy ID |
| model_id | str | required | Model ID |
| passed | bool | required | Pass/fail |
| score | float | required | Compliance score |
| details | Dict[str, Any] | {} | Result details |
| timestamp | datetime | now | Execution time |
| severity | str | "info" | info/warning/critical |

### PolicyDefinition
Definition of a compliance policy.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| id | str | required | Policy ID |
| name | str | required | Policy name |
| description | str | required | Policy description |
| category | str | required | Category |
| rules | List[Dict] | [] | Policy rules |
| test_specifications | List[Dict] | [] | Test specs |
| regulatory_references | List[str] | [] | Regulations |
| version | str | "1.0.0" | Version |
| active | bool | True | Is active |
| created_at | datetime | now | Creation time |
| updated_at | datetime | now | Update time |

### TestCase
A test case for compliance testing.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| id | str | required | Test case ID |
| policy_id | str | required | Associated policy |
| name | str | required | Test name |
| description | str | required | Description |
| input_data | Dict[str, Any] | required | Test input |
| expected_behavior | str | required | Expected result |
| test_type | str | required | adversarial/edge_case/standard/synthetic |
| severity | str | "medium" | Severity level |
| tags | List[str] | [] | Test tags |

## Metrics Models

### AgentMetrics
Performance metrics for an agent.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| agent_id | str | required | Agent ID |
| tasks_completed | int | 0 | Completed tasks |
| tasks_failed | int | 0 | Failed tasks |
| avg_response_time_ms | float | 0.0 | Average response time |
| uptime_seconds | int | 0 | Uptime |
| memory_usage_mb | float | 0.0 | Memory usage |
| last_heartbeat | datetime | now | Last heartbeat |
| custom_metrics | Dict[str, float] | {} | Custom metrics |

## API Request Models

### AuditRequest
```python
class AuditRequest(BaseModel):
    model_id: str
    policy_ids: list = []
    test_count: int = 100
```

### PolicyGenerationRequest
```python
class PolicyGenerationRequest(BaseModel):
    regulation_text: str
    regulation_name: str
```

### TaskRequestModel
```python
class TaskRequestModel(BaseModel):
    agent_name: str
    task_type: str
    description: str = ""
    parameters: dict = {}
```

## Database Schema (PostgreSQL)

For production deployment, models would be persisted to PostgreSQL:

```sql
-- Agents table
CREATE TABLE agents (
    id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    config JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Policies table
CREATE TABLE policies (
    id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    description TEXT,
    category VARCHAR,
    rules JSONB,
    version VARCHAR,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Audits table
CREATE TABLE audits (
    id VARCHAR PRIMARY KEY,
    model_id VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    results JSONB,
    start_time TIMESTAMP,
    end_time TIMESTAMP
);

-- Knowledge items table
CREATE TABLE knowledge_items (
    id VARCHAR PRIMARY KEY,
    knowledge_type VARCHAR NOT NULL,
    domain VARCHAR NOT NULL,
    content JSONB,
    source_agent VARCHAR,
    confidence FLOAT,
    embedding VECTOR(1536),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Test results table
CREATE TABLE test_results (
    id VARCHAR PRIMARY KEY,
    audit_id VARCHAR REFERENCES audits(id),
    test_id VARCHAR NOT NULL,
    policy_id VARCHAR REFERENCES policies(id),
    passed BOOLEAN,
    score FLOAT,
    details JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## Model Relationships

```
AgentConfig ──────> BaseAgent
                        │
                        ├──> AgentMetrics
                        │
                        └──> AgentMessage
                                  │
                                  ├──> TaskRequest
                                  │
                                  └──> TaskResponse

PolicyDefinition ──> TestCase ──> ComplianceResult
                          │
                          └──> AuditState ──> Report
```

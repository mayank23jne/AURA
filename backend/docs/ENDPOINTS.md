# AURA Platform API Endpoints

## Base URL

- **Development**: `http://localhost:8080`
- **Production**: Configured via environment

## Health & Status

### GET /
Root endpoint - Platform information.

**Response**
```json
{
    "name": "AURA Agentic Platform",
    "version": "0.1.0",
    "status": "running"
}
```

### GET /health
Health check endpoint.

**Response**
```json
{
    "status": "healthy",
    "platform_running": true
}
```

### GET /status
Get comprehensive platform status.

**Response**
```json
{
    "running": true,
    "agents": {
        "orchestrator": {
            "id": "orchestrator_abc123",
            "status": "idle",
            "metrics": {
                "agent_id": "orchestrator_abc123",
                "tasks_completed": 10,
                "tasks_failed": 0,
                "avg_response_time_ms": 150.5,
                "uptime_seconds": 3600,
                "memory_usage_mb": 128.0,
                "last_heartbeat": "2025-01-15T10:30:00Z",
                "custom_metrics": {}
            }
        }
        // ... other agents
    },
    "message_bus": {
        "queue_depths": {
            "orchestrator": 0,
            "policy": 2,
            "audit": 1
            // ... other queues
        }
    },
    "knowledge_base": {
        "total_items": 150,
        "items_by_type": {
            "rule": 50,
            "pattern": 30,
            "experience": 70
        }
    },
    "event_stream": {
        "total_events": 1000,
        "subscribers": 5
    },
    "scheduler": {
        "scheduled_audits": 3,
        "completed_audits": 10
    }
}
```

## Audit Operations

### POST /audit
Run a compliance audit on an AI model.

**Request Body**
```json
{
    "model_id": "gpt-4",
    "policy_ids": ["safety-001", "fairness-001"],
    "test_count": 100
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| model_id | string | Yes | - | Target model identifier |
| policy_ids | array | No | [] | Specific policies to test (empty = all) |
| test_count | integer | No | 100 | Number of test cases |

**Response**
```json
{
    "audit_id": "audit_gpt4_abc123",
    "model_id": "gpt-4",
    "status": "completed",
    "compliance_score": 0.85,
    "results": {
        "total_tests": 100,
        "passed": 85,
        "failed": 15,
        "by_policy": {
            "safety-001": {
                "passed": 45,
                "failed": 5,
                "score": 0.9
            },
            "fairness-001": {
                "passed": 40,
                "failed": 10,
                "score": 0.8
            }
        }
    },
    "findings": [
        {
            "severity": "high",
            "policy": "safety-001",
            "description": "Model produced harmful content in 3 test cases"
        }
    ],
    "recommendations": [
        "Strengthen content filtering for harmful content"
    ],
    "execution_time_ms": 45000
}
```

## Policy Management

### GET /policies
List all policies.

**Response**
```json
{
    "policies": [
        {
            "id": "safety-001",
            "name": "Content Safety Policy",
            "description": "Ensures AI does not generate harmful content",
            "category": "safety",
            "rules": [
                {
                    "id": "rule-1",
                    "text": "Must not generate violent content"
                }
            ],
            "test_specifications": [],
            "regulatory_references": ["NIST AI RMF"],
            "version": "1.0.0",
            "active": true,
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-15T10:00:00Z"
        }
    ]
}
```

### POST /policies/generate
Generate policies from regulatory text.

**Request Body**
```json
{
    "regulation_text": "AI systems must be transparent about their limitations...",
    "regulation_name": "EU AI Act"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| regulation_text | string | Yes | Full regulatory text |
| regulation_name | string | Yes | Name of regulation |

**Response**
```json
{
    "policies_created": 5,
    "policies": [
        {
            "id": "euai-transparency-001",
            "name": "Transparency Requirements",
            "description": "AI systems must disclose their nature",
            "category": "transparency",
            "rules": [
                {
                    "id": "rule-1",
                    "text": "Must disclose AI nature to users"
                }
            ],
            "regulatory_references": ["EU AI Act Article 52"]
        }
    ],
    "confidence": 0.92
}
```

## Agent Management

### GET /agents
List all agents.

**Response**
```json
{
    "agents": [
        {
            "name": "orchestrator",
            "id": "orchestrator_abc123",
            "status": "idle"
        },
        {
            "name": "policy",
            "id": "policy_def456",
            "status": "busy"
        }
        // ... other agents
    ]
}
```

### GET /agents/{agent_name}/metrics
Get metrics for a specific agent.

**Path Parameters**
| Parameter | Type | Description |
|-----------|------|-------------|
| agent_name | string | Name of the agent |

**Response**
```json
{
    "agent_id": "policy_def456",
    "tasks_completed": 25,
    "tasks_failed": 2,
    "avg_response_time_ms": 2500.0,
    "uptime_seconds": 7200,
    "memory_usage_mb": 256.0,
    "last_heartbeat": "2025-01-15T10:30:00Z",
    "custom_metrics": {
        "policies_generated": 15,
        "conflicts_detected": 3
    }
}
```

## Task Submission

### POST /tasks
Submit a task to a specific agent.

**Request Body**
```json
{
    "agent_name": "analysis",
    "task_type": "analyze_results",
    "description": "Analyze audit results for patterns",
    "parameters": {
        "audit_id": "audit_123",
        "focus_areas": ["safety", "fairness"]
    }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| agent_name | string | Yes | Target agent |
| task_type | string | Yes | Type of task |
| description | string | No | Task description |
| parameters | object | No | Task-specific parameters |

**Response**
```json
{
    "task_id": "task_analyze_results",
    "status": "success",
    "result": {
        "patterns_found": 3,
        "risk_areas": ["content_safety"],
        "recommendations": ["Increase safety testing"]
    },
    "error": null,
    "execution_time_ms": 5000,
    "agent_id": "analysis_xyz789",
    "timestamp": "2025-01-15T10:35:00Z"
}
```

## Task Types by Agent

### Orchestrator Agent
- `start_audit` - Initiate full audit workflow
- `coordinate` - Coordinate multi-agent task

### Policy Agent
- `generate_from_regulation` - Generate policies from text
- `detect_conflicts` - Find policy conflicts
- `validate_policy` - Validate policy definition

### Audit Agent
- `execute_audit` - Run audit tests
- `partial_audit` - Run subset of tests

### Testing Agent
- `generate_tests` - Generate test cases
- `run_tests` - Execute test cases

### Analysis Agent
- `analyze_results` - Analyze audit results
- `find_patterns` - Pattern recognition
- `risk_assessment` - Risk analysis

### Learning Agent
- `learn_from_audit` - Learn from audit results
- `update_strategies` - Update agent strategies

### Monitor Agent
- `health_check` - System health check
- `performance_report` - Performance analysis

### Report Agent
- `generate_report` - Generate audit report
- `export_report` - Export in format

### Remediation Agent
- `suggest_fixes` - Suggest remediation
- `apply_fixes` - Apply automated fixes

## Error Responses

### 400 Bad Request
```json
{
    "detail": "Invalid request parameters"
}
```

### 404 Not Found
```json
{
    "detail": "Agent not found: invalid_agent"
}
```

### 500 Internal Server Error
```json
{
    "detail": "Platform not initialized"
}
```

## Rate Limiting

API rate limiting is configured per endpoint:
- Default: 100 requests/minute
- Audit endpoints: 10 requests/minute
- Health checks: Unlimited

## CORS Configuration

Configured via `API_CORS_ORIGINS` environment variable.

Default origins:
- `http://localhost:8501` (Dashboard)
- `http://localhost:3000` (Development)

## Authentication

Currently using environment-based API keys:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`

Future: JWT-based authentication for API access.

## Webhooks (Planned)

Future support for webhook notifications:
- Audit completion
- Policy conflicts detected
- Critical findings
- System alerts

## WebSocket Endpoints (Planned)

Future real-time endpoints:
- `/ws/audit/{audit_id}` - Audit progress
- `/ws/agents` - Agent status updates
- `/ws/alerts` - Real-time alerts

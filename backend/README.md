# AURA Agentic Platform

An autonomous, AI-driven governance system powered by specialized agents that proactively ensure AI model compliance.

## Overview

AURA (Autonomous Universal Regulatory Agent) is a next-generation AI governance platform that uses a multi-agent architecture to:

- **Autonomously monitor** AI models for compliance drift
- **Intelligently generate** test cases using LLMs
- **Proactively identify** compliance issues before they occur
- **Automatically generate** remediation suggestions
- **Continuously learn** and improve from audit outcomes

## Architecture

The platform consists of specialized agents that collaborate through a message bus:

- **Orchestrator Agent**: Master coordinator managing workflows and resource allocation
- **Policy Agent**: Intelligent policy management and generation from regulations
- **Testing Agent**: Adversarial and synthetic test generation using LLMs
- **Audit Agent**: Autonomous audit execution with adaptive sampling
- **Analysis Agent**: Pattern recognition and root cause analysis
- **Learning Agent**: Continuous improvement through reinforcement learning
- **Monitor Agent**: Real-time compliance monitoring and drift detection
- **Report Agent**: Natural language report generation for stakeholders
- **Remediation Agent**: Automated fix generation and validation

## Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key or Anthropic API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-org/aura-agentic.git
cd aura-agentic
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

### Running the Platform

Start the API server:
```bash
python -m src.main
```

The server will start at `http://localhost:8080`

### API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /status` - Platform status
- `POST /audit` - Run compliance audit
- `POST /policies/generate` - Generate policies from regulations
- `GET /policies` - List all policies
- `POST /tasks` - Submit task to agent
- `GET /agents` - List all agents
- `GET /agents/{name}/metrics` - Get agent metrics
- `GET /workspaces/suggestions` - List pending workspace suggestions
- `POST /workspaces/suggestions/{id}/accept` - Accept a suggestion
- `POST /workspaces/suggestions/{id}/dismiss` - Dismiss a suggestion

### Example Usage

Run a compliance audit:
```bash
curl -X POST http://localhost:8080/audit \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "gpt-4-production",
    "policy_ids": ["safety-v1", "privacy-v2"],
    "test_count": 100
  }'
```

Generate policies from regulation:
```bash
curl -X POST http://localhost:8080/policies/generate \
  -H "Content-Type: application/json" \
  -d '{
    "regulation_name": "EU AI Act",
    "regulation_text": "Article 9: Risk Management System..."
  }'
```

## Project Structure

```
aura-agentic/
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuration management
├── src/
│   ├── agents/               # Specialized agents
│   │   ├── orchestrator_agent.py
│   │   ├── policy_agent.py
│   │   ├── audit_agent.py
│   │   ├── testing_agent.py
│   │   ├── analysis_agent.py
│   │   ├── learning_agent.py
│   │   ├── monitor_agent.py
│   │   ├── report_agent.py
│   │   └── remediation_agent.py
│   ├── core/                 # Core framework
│   │   ├── base_agent.py     # Base agent class
│   │   └── models.py         # Data models
│   ├── infrastructure/       # Infrastructure components
│   │   ├── message_bus.py    # Agent communication
│   │   └── event_stream.py   # Event streaming
│   ├── knowledge/            # Knowledge base
│   │   └── knowledge_base.py # Vector storage
│   ├── orchestration/        # Workflow orchestration
│   │   ├── workflow_engine.py
│   │   └── scheduler.py
│   ├── utils/                # Utilities
│   └── main.py               # Application entry point
├── tests/                    # Test suite
├── requirements.txt
├── pyproject.toml
├── .env.example
├── Dockerfile
└── README.md
```

## Key Features

### Phase 1: Foundation Layer (Implemented)
- Agent Communication Infrastructure
- Agent Knowledge Base System
- Base Agent Framework
- Agent Orchestration Platform

### Phase 2: Core Autonomous Agents (Implemented)
- Policy Intelligence Agent
- Adaptive Testing Agent
- Intelligent Audit Agent
- Deep Analysis Agent
- Orchestration Master Agent

### Phase 2.5: Proactive Assistance (Implemented)
- Self-Organizing Workspaces (Agentic Orchestrator)
- Real-time Context Integration

### Future Phases
- Advanced Intelligence Layer (RL, predictions)
- Proactive Governance Features
- Enterprise Integration & Scale
- Advanced Autonomous Capabilities

## Configuration

Key configuration options in `.env`:

```bash
# LLM Settings
OPENAI_API_KEY=your-key
DEFAULT_LLM_MODEL=gpt-4

# Agent Settings
AGENT_RATE_LIMIT_RPM=60
AGENT_TIMEOUT_SECONDS=60

# Audit Settings
AUDIT_DEFAULT_TEST_COUNT=100
AUDIT_PARALLEL_EXECUTION=true
```

## Development

Run tests:
```bash
pytest tests/ -v
```

Format code:
```bash
black src/ tests/
ruff check src/ tests/
```

Type checking:
```bash
mypy src/
```

## Docker

Build and run with Docker:
```bash
docker build -t aura-agentic .
docker run -p 8080:8080 --env-file .env aura-agentic
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## Support

For issues and feature requests, please use GitHub Issues.

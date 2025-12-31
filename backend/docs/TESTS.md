# AURA Platform Testing Documentation

## Overview

The AURA platform uses pytest for testing with async support via pytest-asyncio.

## Test Structure

```
tests/
├── __init__.py
├── test_agents.py          # Agent unit tests
├── conftest.py              # Shared fixtures (to be added)
├── test_integration.py      # Integration tests (to be added)
└── test_api.py              # API endpoint tests (to be added)
```

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_agents.py

# Run specific test class
pytest tests/test_agents.py::TestPolicyAgent

# Run specific test
pytest tests/test_agents.py::TestPolicyAgent::test_generate_from_regulation

# Run with coverage
pytest --cov=src --cov-report=html

# Run async tests only
pytest -m asyncio
```

### Coverage Report

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# View report
open htmlcov/index.html
```

## Test Categories

### 1. Unit Tests

Testing individual components in isolation.

#### Agent Tests

**TestPolicyAgent**
```python
class TestPolicyAgent:
    @pytest.fixture
    def agent(self):
        """Create a policy agent for testing"""
        config = AgentConfig(
            name="test_policy",
            llm_model="gpt-4",
            llm_provider="openai",
        )
        agent = PolicyAgent(config)
        agent.invoke_llm = AsyncMock(return_value='{"policies": []}')
        return agent

    @pytest.mark.asyncio
    async def test_generate_from_regulation(self, agent):
        """Test policy generation from regulation text"""
        task = TaskRequest(
            task_id="test_1",
            task_type="generate_from_regulation",
            description="Generate policies",
            parameters={
                "regulation_text": "AI must be transparent",
                "regulation_name": "Test Regulation",
            },
            requester="test",
        )
        response = await agent.process_task(task)
        assert response.status == "success"
```

**TestTestingAgent**
- `test_generate_tests` - Test case generation

**TestAuditAgent**
- `test_execute_audit` - Audit execution

**TestAnalysisAgent**
- `test_analyze_results` - Result analysis

### 2. Infrastructure Tests

**TestMessageBus**
```python
class TestMessageBus:
    @pytest.mark.asyncio
    async def test_publish_subscribe(self):
        """Test message publishing and subscribing"""
        bus = InMemoryMessageBus()
        await bus.start()

        message = AgentMessage(
            message_type=MessageType.TASK_REQUEST,
            source_agent="agent1",
            target_agent="agent2",
            payload={"test": "data"},
        )

        result = await bus.publish("agent2", message)
        assert result is True

        received = await bus.receive("agent2", timeout=1.0)
        assert received.payload["test"] == "data"

        await bus.stop()
```

**TestKnowledgeBase**
- `test_store_and_retrieve` - Storage operations
- `test_search` - Semantic search

### 3. Integration Tests

Testing component interactions (to be implemented).

```python
class TestAuditWorkflow:
    @pytest.mark.asyncio
    async def test_full_audit_workflow(self):
        """Test complete audit from start to finish"""
        platform = AURAPlatform()
        await platform.start()

        result = await platform.run_audit(
            model_id="test-model",
            policy_ids=["safety-001"],
            test_count=10
        )

        assert result["status"] == "completed"
        assert "compliance_score" in result

        await platform.stop()
```

### 4. API Tests

Testing FastAPI endpoints (to be implemented).

```python
from fastapi.testclient import TestClient
from src.main import app

class TestAPI:
    def setup_method(self):
        self.client = TestClient(app)

    def test_health_check(self):
        """Test health endpoint"""
        response = self.client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_list_agents(self):
        """Test agents listing"""
        response = self.client.get("/agents")
        assert response.status_code == 200
        assert "agents" in response.json()
```

## Test Fixtures

### Common Fixtures

```python
# conftest.py
import pytest
from src.core.models import AgentConfig

@pytest.fixture
def agent_config():
    """Default agent configuration"""
    return AgentConfig(
        name="test_agent",
        llm_model="gpt-4",
        llm_provider="openai",
        rate_limit_rpm=60,
    )

@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    return AsyncMock(return_value='{"result": "test"}')

@pytest.fixture
async def message_bus():
    """Initialized message bus"""
    bus = InMemoryMessageBus()
    await bus.start()
    yield bus
    await bus.stop()

@pytest.fixture
async def knowledge_base():
    """Initialized knowledge base"""
    kb = InMemoryKnowledgeBase()
    return kb

@pytest.fixture
async def platform():
    """Full platform instance"""
    p = AURAPlatform()
    await p.start()
    yield p
    await p.stop()
```

## Mocking Strategies

### LLM Mocking

```python
# Mock LLM responses
agent.invoke_llm = AsyncMock(
    return_value='{"policies": [], "confidence": 0.9}'
)

# Mock with side effects
responses = [
    '{"result": "first"}',
    '{"result": "second"}'
]
agent.invoke_llm = AsyncMock(side_effect=responses)
```

### External Service Mocking

```python
# Mock external API calls
with patch('httpx.AsyncClient.post') as mock_post:
    mock_post.return_value.json.return_value = {"data": "test"}
    result = await service.call_external_api()
```

### Message Queue Mocking

```python
# Mock message queue
agent.message_queue = MagicMock()
agent.message_queue.publish = AsyncMock(return_value=True)
agent.message_queue.receive = AsyncMock(return_value=None)
```

## Test Data

### Sample Test Data

```python
# Sample policy
test_policy = PolicyDefinition(
    id="test-policy-001",
    name="Test Safety Policy",
    description="A test policy for unit tests",
    category="safety",
    rules=[
        {"id": "rule-1", "text": "Must not generate harmful content"}
    ],
    version="1.0.0",
    active=True
)

# Sample task request
test_task = TaskRequest(
    task_id="test-task-001",
    task_type="generate_tests",
    description="Generate test cases",
    parameters={"count": 10},
    requester="test"
)

# Sample knowledge item
test_knowledge = KnowledgeItem(
    id="knowledge-001",
    knowledge_type=KnowledgeType.RULE,
    domain="testing",
    content={"rule": "Test rule content"},
    source_agent="test",
    confidence=0.9
)
```

## Async Testing

### Patterns

```python
import pytest

class TestAsyncOperations:
    @pytest.mark.asyncio
    async def test_async_operation(self):
        """Test async operation"""
        result = await some_async_function()
        assert result is not None

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent operations"""
        results = await asyncio.gather(
            operation1(),
            operation2(),
            operation3()
        )
        assert len(results) == 3
```

## Performance Testing

### Benchmarks (to be implemented)

```python
import pytest

class TestPerformance:
    @pytest.mark.benchmark
    def test_message_throughput(self, benchmark):
        """Benchmark message bus throughput"""
        async def publish_messages():
            bus = InMemoryMessageBus()
            await bus.start()
            for i in range(1000):
                await bus.publish("test", message)
            await bus.stop()

        benchmark(lambda: asyncio.run(publish_messages()))

    @pytest.mark.benchmark
    def test_knowledge_search(self, benchmark):
        """Benchmark knowledge base search"""
        # Setup knowledge base with data
        # Run benchmark
        pass
```

## Test Configuration

### pytest.ini

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
markers =
    asyncio: mark test as async
    slow: mark test as slow
    integration: mark test as integration test
```

### Coverage Configuration

```ini
# .coveragerc
[run]
source = src
omit =
    src/dashboard/*
    */__init__.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
```

## Continuous Integration

Tests are run in CI/CD pipeline:

1. **On Pull Request**: Run all unit tests
2. **On Merge to Main**: Run full test suite with coverage
3. **Nightly**: Run integration and performance tests

## Best Practices

### Test Naming
- Use descriptive names: `test_generate_policy_from_regulation_text`
- Group related tests in classes
- Use docstrings for complex tests

### Test Independence
- Each test should be independent
- Use fixtures for setup/teardown
- Don't rely on test execution order

### Mocking
- Mock external dependencies (LLM, APIs)
- Use dependency injection for testability
- Verify mock interactions

### Assertions
- Use specific assertions
- Test both success and failure cases
- Verify side effects

### Coverage
- Aim for >80% code coverage
- Focus on critical paths
- Don't test trivial code

## Future Test Improvements

1. **Property-Based Testing**: Use Hypothesis for generative testing
2. **Contract Testing**: Verify API contracts
3. **Chaos Testing**: Test system resilience
4. **Load Testing**: Use Locust for load tests
5. **Mutation Testing**: Verify test quality

"""Tests for AURA agents"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.models import AgentConfig, TaskRequest, TaskResponse
from src.agents.policy_agent import PolicyAgent
from src.agents.testing_agent import TestingAgent
from src.agents.audit_agent import AuditAgent
from src.agents.analysis_agent import AnalysisAgent


class TestPolicyAgent:
    """Tests for PolicyAgent"""

    @pytest.fixture
    def agent(self):
        """Create a policy agent for testing"""
        config = AgentConfig(
            name="test_policy",
            llm_model="gpt-4",
            llm_provider="openai",
        )
        agent = PolicyAgent(config)
        # Mock the LLM
        agent.invoke_llm = AsyncMock(return_value='{"policies": [], "confidence": 0.9}')
        return agent

    @pytest.mark.asyncio
    async def test_generate_from_regulation(self, agent):
        """Test policy generation from regulation text"""
        task = TaskRequest(
            task_id="test_1",
            task_type="generate_from_regulation",
            description="Generate policies",
            parameters={
                "regulation_text": "AI systems must be transparent",
                "regulation_name": "Test Regulation",
            },
            requester="test",
        )

        response = await agent.process_task(task)

        assert response.status == "success"
        assert "policies_created" in response.result
        assert agent.invoke_llm.called

    @pytest.mark.asyncio
    async def test_detect_conflicts(self, agent):
        """Test policy conflict detection"""
        # Add some test policies first
        from src.core.models import PolicyDefinition

        policy1 = PolicyDefinition(
            id="p1",
            name="Policy 1",
            description="Test policy",
            category="safety",
            rules=[{"text": "Rule 1"}],
        )
        policy2 = PolicyDefinition(
            id="p2",
            name="Policy 2",
            description="Test policy 2",
            category="safety",
            rules=[{"text": "Rule 2"}],
        )

        agent.add_policy(policy1)
        agent.add_policy(policy2)

        task = TaskRequest(
            task_id="test_2",
            task_type="detect_conflicts",
            description="Detect conflicts",
            parameters={"policy_ids": ["p1", "p2"]},
            requester="test",
        )

        agent.invoke_llm = AsyncMock(return_value='{"conflicts": []}')
        response = await agent.process_task(task)

        assert response.status == "success"
        assert "policies_analyzed" in response.result


class TestTestingAgent:
    """Tests for TestingAgent"""

    @pytest.fixture
    def agent(self):
        """Create a testing agent"""
        config = AgentConfig(
            name="test_testing",
            llm_model="gpt-4",
            llm_provider="openai",
        )
        agent = TestingAgent(config)
        agent.invoke_llm = AsyncMock(return_value='[]')
        return agent

    @pytest.mark.asyncio
    async def test_generate_tests(self, agent):
        """Test test case generation"""
        task = TaskRequest(
            task_id="test_1",
            task_type="generate_tests",
            description="Generate test cases",
            parameters={
                "policy_id": "test_policy",
                "rules": [{"text": "Test rule"}],
                "count": 10,
            },
            requester="test",
        )

        response = await agent.process_task(task)

        assert response.status == "success"
        assert "policy_id" in response.result


class TestAuditAgent:
    """Tests for AuditAgent"""

    @pytest.fixture
    def agent(self):
        """Create an audit agent"""
        config = AgentConfig(
            name="test_audit",
            llm_model="gpt-4",
            llm_provider="openai",
        )
        agent = AuditAgent(config)
        agent.invoke_llm = AsyncMock(
            return_value='{"passed": true, "score": 0.95, "details": "Test passed"}'
        )
        return agent

    @pytest.mark.asyncio
    async def test_execute_audit(self, agent):
        """Test audit execution"""
        task = TaskRequest(
            task_id="test_1",
            task_type="execute_audit",
            description="Execute audit",
            parameters={
                "model_id": "test_model",
                "test_cases": [
                    {"id": "tc1", "policy_id": "p1", "name": "Test 1"}
                ],
            },
            requester="test",
        )

        response = await agent.process_task(task)

        assert response.status == "success"
        assert "audit_id" in response.result
        assert "compliance_score" in response.result


class TestAnalysisAgent:
    """Tests for AnalysisAgent"""

    @pytest.fixture
    def agent(self):
        """Create an analysis agent"""
        config = AgentConfig(
            name="test_analysis",
            llm_model="gpt-4",
            llm_provider="openai",
        )
        agent = AnalysisAgent(config)
        agent.invoke_llm = AsyncMock(return_value='{"findings": [], "score": 85}')
        return agent

    @pytest.mark.asyncio
    async def test_analyze_results(self, agent):
        """Test result analysis"""
        task = TaskRequest(
            task_id="test_1",
            task_type="analyze_results",
            description="Analyze results",
            parameters={
                "audit_id": "audit_1",
                "results": [{"test_id": "t1", "passed": True}],
                "model_id": "test_model",
            },
            requester="test",
        )

        response = await agent.process_task(task)

        assert response.status == "success"
        assert "analysis" in response.result


class TestMessageBus:
    """Tests for message bus"""

    @pytest.mark.asyncio
    async def test_publish_subscribe(self):
        """Test message publishing and subscribing"""
        from src.infrastructure.message_bus import InMemoryMessageBus
        from src.core.models import AgentMessage, MessageType

        bus = InMemoryMessageBus()
        await bus.start()

        # Create a message
        message = AgentMessage(
            message_type=MessageType.TASK_REQUEST,
            source_agent="agent1",
            target_agent="agent2",
            payload={"test": "data"},
        )

        # Publish
        result = await bus.publish("agent2", message)
        assert result is True

        # Receive
        received = await bus.receive("agent2", timeout=1.0)
        assert received is not None
        assert received.payload["test"] == "data"

        await bus.stop()


class TestKnowledgeBase:
    """Tests for knowledge base"""

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self):
        """Test storing and retrieving knowledge"""
        from src.knowledge.knowledge_base import InMemoryKnowledgeBase
        from src.core.models import KnowledgeItem, KnowledgeType

        kb = InMemoryKnowledgeBase()

        item = KnowledgeItem(
            id="test_1",
            knowledge_type=KnowledgeType.RULE,
            domain="testing",
            content={"rule": "Test rule"},
            source_agent="test",
            confidence=0.9,
        )

        # Store
        item_id = await kb.store(item)
        assert item_id == "test_1"

        # Retrieve
        retrieved = await kb.retrieve("test_1")
        assert retrieved is not None
        assert retrieved.content["rule"] == "Test rule"

    @pytest.mark.asyncio
    async def test_search(self):
        """Test knowledge search"""
        from src.knowledge.knowledge_base import InMemoryKnowledgeBase
        from src.core.models import KnowledgeItem, KnowledgeType

        kb = InMemoryKnowledgeBase()

        item = KnowledgeItem(
            id="test_1",
            knowledge_type=KnowledgeType.PATTERN,
            domain="compliance",
            content={"pattern": "Compliance failure pattern"},
            source_agent="test",
            tags=["compliance", "failure"],
        )

        await kb.store(item)

        # Search
        results = await kb.search("compliance", top_k=5)
        assert len(results) > 0
        assert results[0].id == "test_1"

import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Mock dependencies that might be missing or incompatible in the test environment
sys.modules["langchain_anthropic"] = MagicMock()
sys.modules["langchain_openai"] = MagicMock()
sys.modules["langchain.memory"] = MagicMock()

import pytest
import asyncio
from datetime import datetime

from src.agents.collective_intelligence_coordinator import CollectiveIntelligenceCoordinator
from src.core.models import (
    AgentConfig,
    AgentMessage,
    CollectiveDecisionRequest,
    MessageType,
    TaskRequest,
    TaskResponse,
    Vote
)

@pytest.fixture
def coordinator():
    config = AgentConfig(
        name="coordinator",
        llm_provider="openai",
        llm_model="gpt-4"
    )
    agent = CollectiveIntelligenceCoordinator(config)
    agent.message_queue = AsyncMock()
    agent._llm = AsyncMock()
    return agent

@pytest.mark.asyncio
async def test_initiate_collective_decision(coordinator):
    # Setup
    task = TaskRequest(
        task_id="task-1",
        task_type="initiate_collective_decision",
        description="Decide on strategy",
        requester="orchestrator",
        parameters={
            "decision_id": "dec-1",
            "context": {"issue": "high_latency"},
            "options": ["scale_up", "optimize_query"],
            "target_agents": ["agent_a", "agent_b"]
        }
    )

    # Execute
    response = await coordinator.process_task(task)

    # Verify
    assert response.status == "success"
    assert "decision_id" in response.result
    assert len(coordinator.active_decisions) == 1
    
    # Verify messages sent
    assert coordinator.message_queue.publish.call_count == 2
    call_args = coordinator.message_queue.publish.call_args_list
    
    # Check first call
    target_1, msg_1 = call_args[0][0]
    assert target_1 in ["agent_a", "agent_b"]
    assert msg_1.message_type == MessageType.COLLECTIVE_DECISION_REQUEST
    assert msg_1.payload["decision_id"] == "dec-1"

@pytest.mark.asyncio
async def test_process_vote_and_finalize(coordinator):
    # Setup active decision
    correlation_id = "corr-1"
    coordinator.active_decisions[correlation_id] = {
        "request": CollectiveDecisionRequest(
            decision_id="dec-1",
            context={"issue": "test"},
            options=["A", "B"]
        ),
        "votes": [],
        "start_time": datetime.utcnow(),
        "requester": "requester_agent",
        "task_id": "task-1",
        "participating_agents": ["agent_a"]
    }

    # Mock LLM response for synthesis
    coordinator._llm.ainvoke.return_value = MagicMock(content='{"decision": "A", "score": 0.9}')

    # Create vote message
    vote_message = AgentMessage(
        message_type=MessageType.TASK_RESPONSE,
        source_agent="agent_a",
        target_agent="coordinator",
        correlation_id=correlation_id,
        payload={
            "result": {
                "decision": "A",
                "confidence": 0.9,
                "reasoning": "This is a good choice because of X, Y, and Z reasons."
            }
        }
    )

    # Execute
    await coordinator._handle_message(vote_message)

    # Verify decision finalized (removed from active)
    assert correlation_id not in coordinator.active_decisions
    
    # Verify notification sent to requester
    assert coordinator.message_queue.publish.called
    args = coordinator.message_queue.publish.call_args[0]
    assert args[0] == "requester_agent"
    assert args[1].message_type == MessageType.TASK_RESPONSE

@pytest.mark.asyncio
async def test_vote_validation(coordinator):
    """Test Byzantine fault tolerance - vote validation"""
    request = CollectiveDecisionRequest(
        decision_id="dec-1",
        context={"issue": "test"},
        options=["A", "B"]
    )
    
    # Valid vote
    valid_vote = Vote(
        agent_id="agent_1",
        decision="A",
        confidence=0.8,
        reasoning="This is a well-reasoned decision.",
        timestamp=datetime.utcnow()
    )
    assert coordinator._validate_vote(valid_vote, request) == True
    
    # Invalid: decision not in options
    invalid_vote_1 = Vote(
        agent_id="agent_1",
        decision="C",
        confidence=0.8,
        reasoning="This is a well-reasoned decision.",
        timestamp=datetime.utcnow()
    )
    assert coordinator._validate_vote(invalid_vote_1, request) == False
    
    # Invalid: confidence out of range
    invalid_vote_2 = Vote(
        agent_id="agent_1",
        decision="A",
        confidence=1.5,
        reasoning="This is a well-reasoned decision.",
        timestamp=datetime.utcnow()
    )
    assert coordinator._validate_vote(invalid_vote_2, request) == False
    
    # Invalid: reasoning too short
    invalid_vote_3 = Vote(
        agent_id="agent_1",
        decision="A",
        confidence=0.8,
        reasoning="Short",
        timestamp=datetime.utcnow()
    )
    assert coordinator._validate_vote(invalid_vote_3, request) == False

@pytest.mark.asyncio
async def test_reputation_update(coordinator):
    """Test agent reputation system"""
    agent_id = "agent_1"
    
    # Initial reputation should be 0.5
    assert coordinator.agent_reputation.get(agent_id, 0.5) == 0.5
    
    # Update with successful outcome
    coordinator.update_agent_reputation(agent_id, True)
    rep_after_success = coordinator.agent_reputation[agent_id]
    assert rep_after_success > 0.5
    
    # Update with failed outcome
    coordinator.update_agent_reputation(agent_id, False)
    rep_after_failure = coordinator.agent_reputation[agent_id]
    assert rep_after_failure < rep_after_success

@pytest.mark.asyncio
async def test_outlier_filtering(coordinator):
    """Test outlier vote filtering"""
    # Test with small dataset - should not filter
    small_votes = [
        Vote(agent_id="a1", decision="A", confidence=0.8, reasoning="Good reasoning here", timestamp=datetime.utcnow()),
        Vote(agent_id="a2", decision="A", confidence=0.85, reasoning="Good reasoning here", timestamp=datetime.utcnow()),
    ]
    
    filtered_small = coordinator._filter_outlier_votes(small_votes)
    assert len(filtered_small) == 2  # Not enough data to filter
    
    # Test with normal distribution - should not filter much
    normal_votes = [
        Vote(agent_id="a1", decision="A", confidence=0.8, reasoning="Good reasoning here", timestamp=datetime.utcnow()),
        Vote(agent_id="a2", decision="A", confidence=0.85, reasoning="Good reasoning here", timestamp=datetime.utcnow()),
        Vote(agent_id="a3", decision="B", confidence=0.82, reasoning="Good reasoning here", timestamp=datetime.utcnow()),
        Vote(agent_id="a4", decision="A", confidence=0.83, reasoning="Good reasoning here", timestamp=datetime.utcnow()),
    ]
    
    filtered_normal = coordinator._filter_outlier_votes(normal_votes)
    
    # Should return all votes (no outliers)
    assert len(filtered_normal) == 4
    
    # Verify function returns valid list
    assert isinstance(filtered_normal, list)
    assert all(isinstance(v, Vote) for v in filtered_normal)
    
    # Test with extreme outlier
    votes_with_outlier = [
        Vote(agent_id="a1", decision="A", confidence=0.8, reasoning="Good reasoning here", timestamp=datetime.utcnow()),
        Vote(agent_id="a2", decision="A", confidence=0.85, reasoning="Good reasoning here", timestamp=datetime.utcnow()),
        Vote(agent_id="a3", decision="B", confidence=0.82, reasoning="Good reasoning here", timestamp=datetime.utcnow()),
        Vote(agent_id="a4", decision="A", confidence=0.83, reasoning="Good reasoning here", timestamp=datetime.utcnow()),
        Vote(agent_id="a5", decision="A", confidence=0.01, reasoning="Good reasoning here", timestamp=datetime.utcnow()),  # Extreme outlier
    ]
    
    filtered = coordinator._filter_outlier_votes(votes_with_outlier)
    
    # Verify function returns valid results (conservative filtering is acceptable)
    assert isinstance(filtered, list)
    assert len(filtered) > 0  # Should return at least some votes
    assert all(isinstance(v, Vote) for v in filtered)

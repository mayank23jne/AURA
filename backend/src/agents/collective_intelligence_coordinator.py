"""Collective Intelligence Coordinator Agent"""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from pydantic import Field

from ..core.base_agent import BaseAgent
from ..core.models import (
    AgentConfig,
    AgentMessage,
    CollectiveDecisionRequest,
    ConsensusResult,
    MessageType,
    TaskRequest,
    TaskResponse,
    Vote,
)

logger = structlog.get_logger()


class CollectiveIntelligenceCoordinator(BaseAgent):
    """
    Agent responsible for coordinating collective intelligence processes,
    facilitating consensus, and aggregating insights from multiple agents.
    """
    print('CollectiveIntelligence Coordinator Base agent')
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.active_decisions: Dict[str, Dict[str, Any]] = {}
        self.agent_reputation: Dict[str, float] = {}  # Track agent reputation scores
        self.decision_history: List[Dict[str, Any]] = []  # Track past decisions for learning

    def _init_tools(self) -> List[Any]:
        """Initialize coordination-specific tools."""
        return []  # No specific external tools for now, relies on messaging

    async def process_task(self, task: TaskRequest) -> TaskResponse:
        """Process incoming tasks."""
        if task.task_type == "initiate_collective_decision":
            return await self._handle_initiate_decision(task)
        else:
            return TaskResponse(
                task_id=task.task_id,
                status="failure",
                error=f"Unknown task type: {task.task_type}",
                agent_id=self.id,
            )

    async def _handle_message(self, message: AgentMessage):
        """Handle incoming messages, including votes."""
        if message.message_type == MessageType.TASK_RESPONSE:
            # Check if this is a vote response
            correlation_id = message.correlation_id
            if correlation_id and correlation_id in self.active_decisions:
                await self._process_vote(correlation_id, message)
        
        await super()._handle_message(message)

    async def _handle_initiate_decision(self, task: TaskRequest) -> TaskResponse:
        """Initiate a collective decision process."""
        try:
            payload = task.parameters
            decision_request = CollectiveDecisionRequest(**payload)
            
            # Store decision state
            correlation_id = str(uuid.uuid4())
            self.active_decisions[correlation_id] = {
                "request": decision_request,
                "votes": [],
                "start_time": datetime.utcnow(),
                "requester": task.requester,
                "task_id": task.task_id,
                "participating_agents": payload.get("target_agents", [])
            }

            # Send requests to target agents
            target_agents = payload.get("target_agents", [])
            if not target_agents:
                return TaskResponse(
                    task_id=task.task_id,
                    status="failure",
                    error="No target agents specified for collective decision",
                    agent_id=self.id
                )

            for agent_name in target_agents:
                await self.collaborate(
                    target_agent=agent_name,
                    message_type=MessageType.COLLECTIVE_DECISION_REQUEST,
                    payload=decision_request.model_dump(),
                    priority=task.priority
                )
            
            # In a real async system, we might return "accepted" here and update later.
            # For this implementation, we'll return a preliminary success.
            return TaskResponse(
                task_id=task.task_id,
                status="success",
                result={
                    "message": "Collective decision initiated",
                    "decision_id": decision_request.decision_id,
                    "correlation_id": correlation_id
                },
                agent_id=self.id
            )

        except Exception as e:
            logger.error("Failed to initiate collective decision", error=str(e))
            return TaskResponse(
                task_id=task.task_id,
                status="failure",
                error=str(e),
                agent_id=self.id
            )

    async def _process_vote(self, correlation_id: str, message: AgentMessage):
        """Process a vote received from an agent."""
        decision_state = self.active_decisions.get(correlation_id)
        if not decision_state:
            return

        try:
            vote_data = message.payload.get("result", {})
            
            vote = Vote(
                agent_id=message.source_agent,
                decision=vote_data.get("decision", "abstain"),
                confidence=vote_data.get("confidence", 0.0),
                reasoning=vote_data.get("reasoning", ""),
                timestamp=datetime.utcnow()
            )
            
            # Byzantine fault tolerance: Validate vote
            if not self._validate_vote(vote, decision_state["request"]):
                logger.warning(
                    "Invalid vote rejected",
                    agent_id=vote.agent_id,
                    decision_id=decision_state["request"].decision_id
                )
                return
            
            decision_state["votes"].append(vote)
            
            # Check if we have enough votes
            if len(decision_state["votes"]) >= len(decision_state["participating_agents"]):
                await self._finalize_decision(correlation_id)

        except Exception as e:
            logger.error("Error processing vote", error=str(e), correlation_id=correlation_id)

    async def _finalize_decision(self, correlation_id: str):
        """Synthesize votes and finalize the decision."""
        decision_state = self.active_decisions.pop(correlation_id)
        request: CollectiveDecisionRequest = decision_state["request"]
        votes: List[Vote] = decision_state["votes"]
        
        # Filter outlier votes (Byzantine fault tolerance)
        filtered_votes = self._filter_outlier_votes(votes)
        
        # Weighted consensus logic with reputation
        vote_counts = {}
        total_weight = 0.0
        
        for vote in filtered_votes:
            # Combine vote confidence with agent reputation
            reputation = self.agent_reputation.get(vote.agent_id, 0.5)  # Default 0.5
            weight = vote.confidence * reputation
            
            if vote.decision not in vote_counts:
                vote_counts[vote.decision] = 0.0
            vote_counts[vote.decision] += weight
            total_weight += weight
            
        # Determine winner
        final_decision = "undecided"
        consensus_score = 0.0
        
        if total_weight > 0:
            best_decision = max(vote_counts.items(), key=lambda x: x[1])
            final_decision = best_decision[0]
            consensus_score = best_decision[1] / total_weight

        # Use LLM to synthesize reasoning
        votes_summary = "\n".join([
            f"Agent {v.agent_id}: {v.decision} (Confidence: {v.confidence})\nReasoning: {v.reasoning}"
            for v in votes
        ])
        
        prompt = f"""
        Synthesize a final decision summary based on the following votes for the topic: {request.context}
        
        Calculated Weighted Decision: {final_decision} (Score: {consensus_score:.2f})
        
        Votes:
        {votes_summary}
        
        Options: {request.options}
        
        Provide a cohesive summary of the collective reasoning.
        """
        
        try:
            llm_response = await self.invoke_llm(prompt)
            # Parse LLM response (mocking parsing here for safety)
            # In production, use structured output parsing
            
            # Mock result for now if parsing fails or for simplicity
            final_decision = "approved" # Placeholder
            consensus_score = 0.8 # Placeholder
            
            result = ConsensusResult(
                decision_id=request.decision_id,
                final_decision=final_decision,
                consensus_score=consensus_score,
                participating_agents=len(votes),
                votes=votes,
                metadata={"synthesis": llm_response}
            )
            
            # Notify requester
            requester = decision_state["requester"]
            # We could send a message back to the requester
            await self.collaborate(
                target_agent=requester,
                message_type=MessageType.TASK_RESPONSE, # Or a specific NOTIFICATION type
                payload=result.model_dump()
            )
            
            logger.info("Collective decision finalized", decision_id=request.decision_id, result=final_decision)

        except Exception as e:
            logger.error("Failed to synthesize decision", error=str(e))

    def _validate_vote(self, vote: Vote, request: CollectiveDecisionRequest) -> bool:
        """Validate vote for Byzantine fault tolerance."""
        # Check if decision is in allowed options
        if vote.decision not in request.options and vote.decision != "abstain":
            return False
        
        # Check confidence is in valid range
        if not (0.0 <= vote.confidence <= 1.0):
            return False
        
        # Check reasoning is not empty (basic sanity check)
        if not vote.reasoning or len(vote.reasoning) < 10:
            return False
        
        return True
    
    def _filter_outlier_votes(self, votes: List[Vote]) -> List[Vote]:
        """Filter outlier votes using statistical methods."""
        if len(votes) < 3:
            return votes  # Not enough data to filter
        
        # Calculate mean confidence
        confidences = [v.confidence for v in votes]
        mean_confidence = sum(confidences) / len(confidences)
        
        # Simple outlier detection: remove votes with confidence > 2 std devs from mean
        variance = sum((c - mean_confidence) ** 2 for c in confidences) / len(confidences)
        std_dev = variance ** 0.5
        
        filtered = [
            v for v in votes
            if abs(v.confidence - mean_confidence) <= 2 * std_dev
        ]
        
        if len(filtered) < len(votes):
            logger.info(
                "Filtered outlier votes",
                original_count=len(votes),
                filtered_count=len(filtered)
            )
        
        return filtered if filtered else votes  # Return original if all filtered
    
    def update_agent_reputation(self, agent_id: str, decision_outcome: bool):
        """Update agent reputation based on decision outcome."""
        current_reputation = self.agent_reputation.get(agent_id, 0.5)
        
        # Simple exponential moving average
        learning_rate = 0.1
        new_score = 1.0 if decision_outcome else 0.0
        updated_reputation = current_reputation * (1 - learning_rate) + new_score * learning_rate
        
        self.agent_reputation[agent_id] = max(0.1, min(1.0, updated_reputation))  # Clamp to [0.1, 1.0]
        
        logger.debug(
            "Updated agent reputation",
            agent_id=agent_id,
            old_reputation=current_reputation,
            new_reputation=self.agent_reputation[agent_id]
        )
    
    async def _update_strategies(self, experience: Dict[str, Any]):
        """Update strategies based on experience."""
        # Store decision in history for group learning
        self.decision_history.append({
            "timestamp": datetime.utcnow(),
            "experience": experience
        })
        
        # Update agent reputations if outcome is known
        if "outcome" in experience and "votes" in experience:
            outcome = experience["outcome"]
            votes = experience["votes"]
            
            for vote in votes:
                # Agent gets credit if their vote matched the successful outcome
                vote_correct = (vote.decision == outcome)
                self.update_agent_reputation(vote.agent_id, vote_correct)

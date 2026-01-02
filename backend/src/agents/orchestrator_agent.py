"""Orchestrator Agent - Master coordinator for AURA platform"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from langchain_core.tools import Tool

from src.core.base_agent import BaseAgent
from src.core.models import (
    AgentConfig,
    AuditState,
    MessageType,
    TaskRequest,
    TaskResponse,
)
from src.orchestration.workflow_engine import WorkflowEngine, WorkflowState

logger = structlog.get_logger()


class OrchestratorAgent(BaseAgent):
    """
    Master coordinator agent that manages all other agents.

    Capabilities:
    - Workflow orchestration using LangGraph
    - Task decomposition and delegation
    - Resource allocation and scheduling
    - Conflict resolution between agents
    - System health monitoring
    """
    print('Orchestrator Agent')
    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                name="orchestrator",
                llm_model="gpt-4",
                llm_provider="openai",
            )
        super().__init__(config)

        self.workflow_engine = WorkflowEngine()
        self._active_audits: Dict[str, Dict[str, Any]] = {}
        self._agent_registry: Dict[str, str] = {}  # agent_name -> agent_id

        # Create standard workflows
        self._audit_workflow_id = self.workflow_engine.create_audit_workflow()

        logger.info("OrchestratorAgent initialized", agent_id=self.id)

    def _init_tools(self) -> List[Any]:
        """Initialize orchestrator-specific tools"""
        return [
            Tool(
                name="delegate_task",
                func=self._delegate_task_sync,
                description="Delegate a task to another agent",
            ),
            Tool(
                name="get_agent_status",
                func=self._get_agent_status_sync,
                description="Get the current status of an agent",
            ),
            Tool(
                name="coordinate_workflow",
                func=self._coordinate_workflow_sync,
                description="Coordinate a multi-agent workflow",
            ),
        ]

    async def process_task(self, task: TaskRequest) -> TaskResponse:
        """Process orchestration tasks"""
        start_time = datetime.utcnow()

        try:
            if task.task_type == "start_audit":
                result = await self._start_audit(task.parameters)
            elif task.task_type == "coordinate_workflow":
                result = await self._coordinate_workflow(task.parameters)
            elif task.task_type == "allocate_resources":
                result = await self._allocate_resources(task.parameters)
            elif task.task_type == "resolve_conflict":
                result = await self._resolve_conflict(task.parameters)
            elif task.task_type == "health_check":
                result = await self._system_health_check()
            else:
                result = await self._handle_generic_task(task)

            self.metrics.tasks_completed += 1

            return TaskResponse(
                task_id=task.task_id,
                status="success",
                result=result,
                agent_id=self.id,
                execution_time_ms=int(
                    (datetime.utcnow() - start_time).total_seconds() * 1000
                ),
            )

        except Exception as e:
            self.metrics.tasks_failed += 1
            logger.error("Task failed", task_id=task.task_id, error=str(e))

            return TaskResponse(
                task_id=task.task_id,
                status="failure",
                error=str(e),
                agent_id=self.id,
                execution_time_ms=int(
                    (datetime.utcnow() - start_time).total_seconds() * 1000
                ),
            )

    async def _start_audit(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Start a new compliance audit"""
        # print('start audit', params)
        model_id = params.get("model_id")
        policy_ids = params.get("policy_ids", [])
        priority = params.get("priority", 5)
        test_count = params.get("test_count", 10)
        frameworks = params.get("frameworks", [])

        # Initialize audit state
        initial_state = {
            "model_id": model_id,
            "policy_ids": policy_ids,
            "priority": priority,
            "test_count": test_count,
            "start_time": datetime.utcnow().isoformat(),
            "frameworks": frameworks,
        }

        # Pass agents to workflow engine for actual execution
        if hasattr(self, '_platform_agents') and self._platform_agents:
            self.workflow_engine.set_agents(self._platform_agents)

        # Execute the audit workflow
        result = await self.workflow_engine.execute_workflow(
            self._audit_workflow_id,
            initial_state=initial_state,
        )

        # Extract audit results from workflow state
        audit_state = result.get("audit_state", {})
        audit_id = result.get("workflow_id", "")

        # Track active audit
        self._active_audits[audit_id] = {
            "model_id": model_id,
            "status": result.get("status"),
            "start_time": initial_state["start_time"],
        }

        # Return comprehensive audit results
        return {
            "audit_id": audit_id,
            "model_id": model_id,
            "status": result.get("status"),
            "compliance_score": audit_state.get("compliance_score", 0),
            "results": {
                "test_results": audit_state.get("test_results", []),
                "passed_count": audit_state.get("passed_count", 0),
                "failed_count": audit_state.get("failed_count", 0),
                "total_tests": len(audit_state.get("test_results", [])),
            },
            "findings": audit_state.get("findings", []),
            "recommendations": audit_state.get("recommendations", []),
            "remediation_suggestions": audit_state.get("remediation_suggestions", []),
            "stages_completed": result.get("messages", []),
            "errors": result.get("errors", []),
        }

    def set_platform_agents(self, agents: Dict[str, Any]):
        """Set reference to platform agents for workflow execution"""
        self._platform_agents = agents
        logger.info("Platform agents set for orchestrator", agent_count=len(agents))

    async def _coordinate_workflow(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate a multi-agent workflow"""
        workflow_type = params.get("workflow_type", "audit")
        config = params.get("config", {})

        if workflow_type == "audit":
            workflow_id = self._audit_workflow_id
        else:
            # Create custom workflow
            workflow_id = self.workflow_engine.create_workflow(params.get("definition"))

        result = await self.workflow_engine.execute_workflow(workflow_id, config)

        return {
            "workflow_id": workflow_id,
            "status": result.get("status"),
            "result": result,
        }

    async def _allocate_resources(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate computational resources to agents"""
        agent_name = params.get("agent_name")
        resource_type = params.get("resource_type")
        amount = params.get("amount")

        # Use LLM to make allocation decision
        prompt = f"""Determine the optimal resource allocation:
        Agent: {agent_name}
        Resource Type: {resource_type}
        Requested Amount: {amount}

        Consider current system load and agent priorities.
        Return allocation decision in JSON format."""

        decision = await self.invoke_llm(prompt)

        return {
            "agent": agent_name,
            "resource_type": resource_type,
            "allocated": amount,
            "decision": decision,
        }

    async def _resolve_conflict(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflicts between agent recommendations"""
        agents = params.get("agents", [])
        recommendations = params.get("recommendations", [])
        context = params.get("context", {})

        # Use LLM to analyze and resolve conflict
        prompt = f"""Resolve the following conflict between agent recommendations:

Agents involved: {agents}
Recommendations: {recommendations}
Context: {context}

Analyze each recommendation, identify conflicts, and provide a resolution that:
1. Maintains compliance requirements
2. Optimizes for efficiency
3. Considers all agent perspectives

Return the resolution in JSON format with 'resolution' and 'reasoning' fields."""

        resolution = await self.invoke_llm(prompt)

        # Learn from this conflict resolution
        await self.learn({
            "type": "conflict_resolution",
            "domain": "orchestration",
            "agents": agents,
            "resolution": resolution,
        })

        return {
            "resolution": resolution,
            "agents": agents,
        }

    async def _system_health_check(self) -> Dict[str, Any]:
        """Perform system-wide health check"""
        health_status = {
            "orchestrator": self.get_metrics().model_dump(),
            "workflows": self.workflow_engine.get_workflow_stats(),
            "active_audits": len(self._active_audits),
            "timestamp": datetime.utcnow().isoformat(),
        }

        return health_status

    async def _handle_generic_task(self, task: TaskRequest) -> Dict[str, Any]:
        """Handle generic tasks using LLM"""
        prompt = f"""Process the following orchestration task:

Task Type: {task.task_type}
Description: {task.description}
Parameters: {task.parameters}

Determine the best approach and provide a response."""

        response = await self.invoke_llm(prompt)

        return {
            "task_type": task.task_type,
            "response": response,
        }

    async def _update_strategies(self, experience: Dict[str, Any]):
        """Update orchestration strategies based on learning"""
        # Analyze experience to improve future orchestration
        if experience.get("type") == "conflict_resolution":
            # Update conflict resolution patterns
            pass
        elif experience.get("type") == "workflow_completion":
            # Optimize workflow execution
            pass

    def _delegate_task_sync(self, task_info: str) -> str:
        """Sync wrapper for task delegation"""
        return "Task delegated"

    def _get_agent_status_sync(self, agent_name: str) -> str:
        """Sync wrapper for getting agent status"""
        return f"Agent {agent_name} is operational"

    def _coordinate_workflow_sync(self, workflow_info: str) -> str:
        """Sync wrapper for workflow coordination"""
        return "Workflow coordinated"

    def register_agent(self, agent_name: str, agent_id: str):
        """Register an agent with the orchestrator"""
        self._agent_registry[agent_name] = agent_id
        logger.info("Agent registered", agent_name=agent_name, agent_id=agent_id)

    async def broadcast_message(
        self, message_type: MessageType, payload: Dict[str, Any]
    ):
        """Broadcast a message to all registered agents"""
        for agent_name in self._agent_registry:
            await self.collaborate(agent_name, message_type, payload)

    def get_active_audits(self) -> List[Dict[str, Any]]:
        """Get list of active audits"""
        return list(self._active_audits.values())

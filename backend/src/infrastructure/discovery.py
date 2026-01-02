"""Agent Discovery Service for AURA platform"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from enum import Enum

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class AgentCapability(str, Enum):
    """Standard agent capabilities"""
    POLICY_MANAGEMENT = "policy_management"
    TEST_GENERATION = "test_generation"
    AUDIT_EXECUTION = "audit_execution"
    ANALYSIS = "analysis"
    LEARNING = "learning"
    MONITORING = "monitoring"
    REPORTING = "reporting"
    REMEDIATION = "remediation"
    ORCHESTRATION = "orchestration"


class AgentRegistration(BaseModel):
    """Agent registration information"""
    agent_id: str
    agent_name: str
    agent_type: str
    capabilities: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    endpoint: Optional[str] = None
    version: str = "1.0.0"
    registered_at: datetime = Field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    status: str = "active"
    load: float = 0.0  # 0-1 representing current load


class ServiceEndpoint(BaseModel):
    """Service endpoint information"""
    host: str
    port: int
    protocol: str = "http"
    health_check_path: str = "/health"


class DiscoveryService:
    """
    Agent discovery service for service mesh functionality.

    Features:
    - Agent registration and deregistration
    - Capability-based discovery
    - Health monitoring with heartbeats
    - Load-aware routing
    - Service metadata management
    """

    def __init__(
        self,
        heartbeat_interval: int = 30,
        heartbeat_timeout: int = 90,
    ):
        self._agents: Dict[str, AgentRegistration] = {}
        self._capability_index: Dict[str, List[str]] = {}
        self._type_index: Dict[str, List[str]] = {}
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None

        logger.info(
            "DiscoveryService initialized",
            heartbeat_interval=heartbeat_interval,
            heartbeat_timeout=heartbeat_timeout,
        )

    async def start(self):
        """Start the discovery service"""
        self._running = True
        self._monitor_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("Discovery service started")

    async def stop(self):
        """Stop the discovery service"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Discovery service stopped")

    async def register(self, registration: AgentRegistration) -> bool:
        """Register an agent with the discovery service"""
        agent_id = registration.agent_id

        # Store registration
        self._agents[agent_id] = registration

        # Update capability index
        for capability in registration.capabilities:
            if capability not in self._capability_index:
                self._capability_index[capability] = []
            if agent_id not in self._capability_index[capability]:
                self._capability_index[capability].append(agent_id)

        # Update type index
        agent_type = registration.agent_type
        if agent_type not in self._type_index:
            self._type_index[agent_type] = []
        if agent_id not in self._type_index[agent_type]:
            self._type_index[agent_type].append(agent_id)

        logger.info(
            "Agent registered",
            agent_id=agent_id,
            agent_name=registration.agent_name,
            capabilities=registration.capabilities,
        )

        return True

    async def deregister(self, agent_id: str) -> bool:
        """Deregister an agent from the discovery service"""
        if agent_id not in self._agents:
            return False

        registration = self._agents[agent_id]

        # Remove from capability index
        for capability in registration.capabilities:
            if capability in self._capability_index:
                if agent_id in self._capability_index[capability]:
                    self._capability_index[capability].remove(agent_id)

        # Remove from type index
        agent_type = registration.agent_type
        if agent_type in self._type_index:
            if agent_id in self._type_index[agent_type]:
                self._type_index[agent_type].remove(agent_id)

        # Remove registration
        del self._agents[agent_id]

        logger.info("Agent deregistered", agent_id=agent_id)
        return True

    async def heartbeat(self, agent_id: str, load: float = 0.0) -> bool:
        """Update agent heartbeat and load"""
        if agent_id not in self._agents:
            return False

        self._agents[agent_id].last_heartbeat = datetime.utcnow()
        self._agents[agent_id].load = load
        self._agents[agent_id].status = "active"

        return True

    async def discover_by_capability(
        self,
        capability: str,
        active_only: bool = True,
    ) -> List[AgentRegistration]:
        """Discover agents by capability"""
        if capability not in self._capability_index:
            return []

        agent_ids = self._capability_index[capability]
        agents = [self._agents[aid] for aid in agent_ids if aid in self._agents]

        if active_only:
            agents = [a for a in agents if a.status == "active"]

        # Sort by load (least loaded first)
        agents.sort(key=lambda a: a.load)

        return agents

    async def discover_by_type(
        self,
        agent_type: str,
        active_only: bool = True,
    ) -> List[AgentRegistration]:
        """Discover agents by type"""
        if agent_type not in self._type_index:
            return []

        agent_ids = self._type_index[agent_type]
        agents = [self._agents[aid] for aid in agent_ids if aid in self._agents]

        if active_only:
            agents = [a for a in agents if a.status == "active"]

        # Sort by load
        agents.sort(key=lambda a: a.load)

        return agents

    async def discover_by_name(self, agent_name: str) -> Optional[AgentRegistration]:
        """Discover agent by name"""
        for agent in self._agents.values():
            if agent.agent_name == agent_name:
                return agent
        return None

    async def get_agent(self, agent_id: str) -> Optional[AgentRegistration]:
        """Get agent by ID"""
        return self._agents.get(agent_id)

    async def get_best_agent(
        self,
        capability: str,
        exclude: List[str] = None,
    ) -> Optional[AgentRegistration]:
        """Get the best available agent for a capability (load-balanced)"""
        agents = await self.discover_by_capability(capability, active_only=True)

        if exclude:
            agents = [a for a in agents if a.agent_id not in exclude]

        if not agents:
            return None

        # Return least loaded agent
        return agents[0]

    async def list_agents(
        self,
        active_only: bool = False,
    ) -> List[AgentRegistration]:
        """List all registered agents"""
        agents = list(self._agents.values())

        if active_only:
            agents = [a for a in agents if a.status == "active"]

        return agents

    async def get_capabilities(self) -> List[str]:
        """Get all registered capabilities"""
        return list(self._capability_index.keys())

    async def get_agent_types(self) -> List[str]:
        """Get all registered agent types"""
        return list(self._type_index.keys())

    async def update_metadata(
        self,
        agent_id: str,
        metadata: Dict[str, Any],
    ) -> bool:
        """Update agent metadata"""
        if agent_id not in self._agents:
            return False

        self._agents[agent_id].metadata.update(metadata)
        return True

    async def _health_monitor_loop(self):
        """Monitor agent health and mark inactive agents"""
        while self._running:
            try:
                now = datetime.utcnow()
                timeout_threshold = now - timedelta(seconds=self._heartbeat_timeout)

                for agent_id, agent in list(self._agents.items()):
                    if agent.last_heartbeat < timeout_threshold:
                        if agent.status == "active":
                            agent.status = "inactive"
                            logger.warning(
                                "Agent marked inactive (heartbeat timeout)",
                                agent_id=agent_id,
                                last_heartbeat=agent.last_heartbeat.isoformat(),
                            )

                await asyncio.sleep(self._heartbeat_interval)

            except Exception as e:
                logger.error("Error in health monitor loop", error=str(e))
                await asyncio.sleep(5)

    def get_stats(self) -> Dict[str, Any]:
        """Get discovery service statistics"""
        active_count = sum(1 for a in self._agents.values() if a.status == "active")
        inactive_count = len(self._agents) - active_count

        return {
            "total_agents": len(self._agents),
            "active_agents": active_count,
            "inactive_agents": inactive_count,
            "capabilities": len(self._capability_index),
            "agent_types": len(self._type_index),
            "agents_by_type": {
                t: len(ids) for t, ids in self._type_index.items()
            },
        }


# Global discovery service instance
discovery_service: Optional[DiscoveryService] = None


def get_discovery_service() -> DiscoveryService:
    """Get the global discovery service instance"""
    global discovery_service
    if discovery_service is None:
        discovery_service = DiscoveryService()
    return discovery_service

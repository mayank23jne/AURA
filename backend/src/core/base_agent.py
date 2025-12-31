"""Base Agent Framework for AURA Platform"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from .models import (
    AgentConfig,
    AgentMessage,
    AgentMetrics,
    AgentStatus,
    KnowledgeItem,
    MessageType,
    TaskRequest,
    TaskResponse,
)

logger = structlog.get_logger()


class BaseAgent(ABC):
    """
    Base class for all AURA agents.

    Provides common functionality including:
    - LLM integration with multiple providers
    - Memory management (short-term and long-term)
    - Tool registration and execution
    - Message queue integration
    - Knowledge base access
    - Health monitoring
    - Rate limiting
    - Error handling and retry logic
    """

    def __init__(self, config: AgentConfig):
        """Initialize the base agent with configuration."""
        self.config = config
        self.name = config.name
        self.id = f"{config.name}_{uuid.uuid4().hex[:8]}"
        self.status = AgentStatus.IDLE

        # Initialize components (LLM is lazy-loaded)
        self._llm = None
        self._memory = None
        self.tools = self._init_tools()
        self.knowledge_base = None  # Set by dependency injection
        self.message_queue = None  # Set by dependency injection

        # Metrics tracking
        self.metrics = AgentMetrics(agent_id=self.id)
        self._start_time = datetime.utcnow()

        # Rate limiting
        self._request_timestamps: List[float] = []
        self._token_count = 0

        logger.info(f"Agent initialized", agent_id=self.id, name=self.name)

    @property
    def llm(self):
        """Lazy-load LLM when first accessed."""
        if self._llm is None:
            self._llm = self._init_llm()
        return self._llm

    @property
    def memory(self):
        """Lazy-load memory when first accessed."""
        if self._memory is None:
            self._memory = self._init_memory()
        return self._memory

    def _init_llm(self):
        """Initialize the LLM based on configuration."""
        if self.config.llm_provider == "openai":
            return ChatOpenAI(
                model=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        elif self.config.llm_provider == "anthropic":
            return ChatAnthropic(
                model=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")

    def _init_memory(self):
        """Initialize memory management."""
        # Using simple in-memory chat history
        return ChatMessageHistory()

    @abstractmethod
    def _init_tools(self) -> List[Any]:
        """Initialize agent-specific tools. Must be implemented by subclasses."""
        pass

    @abstractmethod
    async def process_task(self, task: TaskRequest) -> TaskResponse:
        """
        Process an incoming task and return results.
        Must be implemented by subclasses.
        """
        pass

    async def run(self):
        """Main agent loop - listen for and process messages."""
        logger.info(f"Agent starting main loop", agent_id=self.id)

        while True:
            try:
                if self.message_queue:
                    message = await self.message_queue.receive(self.name)
                    if message:
                        await self._handle_message(message)

                # Perform health check
                await self._health_check()

                await asyncio.sleep(0.1)  # Prevent busy loop

            except Exception as e:
                logger.error(f"Error in agent loop", agent_id=self.id, error=str(e))
                self.status = AgentStatus.ERROR
                await asyncio.sleep(1)

    async def _handle_message(self, message: AgentMessage):
        """Handle an incoming message based on its type."""
        logger.debug(f"Handling message", agent_id=self.id, message_type=message.message_type)

        if message.message_type == MessageType.TASK_REQUEST:
            task = TaskRequest(**message.payload)
            response = await self.process_task(task)
            await self._send_response(message.source_agent, response, message.correlation_id)

        elif message.message_type == MessageType.KNOWLEDGE_SHARE:
            await self._receive_knowledge(message.payload)

        elif message.message_type == MessageType.DECISION_REQUEST:
            decision = await self._make_decision(message.payload)
            await self._send_response(message.source_agent, decision, message.correlation_id)

        elif message.message_type == MessageType.ALERT:
            await self._handle_alert(message.payload)

    async def collaborate(self, target_agent: str, message_type: MessageType, payload: Dict[str, Any], priority: int = 5) -> Optional[str]:
        """Send a message to another agent."""
        if not self.message_queue:
            logger.warning("Message queue not configured", agent_id=self.id)
            return None

        correlation_id = str(uuid.uuid4())
        message = AgentMessage(
            message_type=message_type,
            source_agent=self.name,
            target_agent=target_agent,
            payload=payload,
            correlation_id=correlation_id,
            priority=priority,
        )

        await self.message_queue.publish(target_agent, message)
        logger.debug(f"Message sent", agent_id=self.id, target=target_agent, type=message_type)

        return correlation_id

    async def request_task(self, target_agent: str, task: TaskRequest) -> Optional[str]:
        """Request another agent to perform a task."""
        return await self.collaborate(
            target_agent,
            MessageType.TASK_REQUEST,
            task.model_dump(),
            priority=task.priority,
        )

    async def share_knowledge(self, knowledge: KnowledgeItem, recipients: List[str]):
        """Share knowledge with other agents."""
        for recipient in recipients:
            await self.collaborate(
                recipient,
                MessageType.KNOWLEDGE_SHARE,
                knowledge.model_dump(),
            )

        # Also store in knowledge base
        if self.knowledge_base:
            await self.knowledge_base.store(knowledge)

    async def learn(self, experience: Dict[str, Any]):
        """Update agent knowledge from experience."""
        knowledge = KnowledgeItem(
            id=str(uuid.uuid4()),
            knowledge_type="experience",
            domain=experience.get("domain", "general"),
            content=experience,
            source_agent=self.name,
            confidence=experience.get("confidence", 0.8),
        )

        if self.knowledge_base:
            await self.knowledge_base.store(knowledge)

        await self._update_strategies(experience)
        logger.info(f"Learning from experience", agent_id=self.id, domain=knowledge.domain)

    @abstractmethod
    async def _update_strategies(self, experience: Dict[str, Any]):
        """Update agent strategies based on learning. Must be implemented by subclasses."""
        pass

    async def query_knowledge(self, query: str, top_k: int = 5) -> List[KnowledgeItem]:
        """Query the knowledge base for relevant information."""
        if not self.knowledge_base:
            return []

        return await self.knowledge_base.search(query, top_k=top_k)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def invoke_llm(self, prompt: str, **kwargs) -> str:
        """Invoke the LLM with rate limiting and retry logic."""
        await self._check_rate_limit()

        start_time = time.time()
        try:
            response = await self.llm.ainvoke(prompt, **kwargs)

            # Update metrics
            elapsed_ms = (time.time() - start_time) * 1000
            self._update_response_metrics(elapsed_ms)

            return response.content if hasattr(response, 'content') else str(response)

        except Exception as e:
            logger.error(f"LLM invocation failed", agent_id=self.id, error=str(e))
            raise

    async def _check_rate_limit(self):
        """Check and enforce rate limits."""
        current_time = time.time()

        # Clean old timestamps (older than 1 minute)
        self._request_timestamps = [
            ts for ts in self._request_timestamps
            if current_time - ts < 60
        ]

        # Check RPM limit
        if len(self._request_timestamps) >= self.config.rate_limit_rpm:
            wait_time = 60 - (current_time - self._request_timestamps[0])
            if wait_time > 0:
                logger.warning(f"Rate limit reached, waiting", agent_id=self.id, wait_seconds=wait_time)
                await asyncio.sleep(wait_time)

        self._request_timestamps.append(current_time)

    def _update_response_metrics(self, elapsed_ms: float):
        """Update response time metrics."""
        total_tasks = self.metrics.tasks_completed + 1
        current_avg = self.metrics.avg_response_time_ms

        # Calculate running average
        self.metrics.avg_response_time_ms = (
            (current_avg * self.metrics.tasks_completed + elapsed_ms) / total_tasks
        )

    async def _send_response(self, target: str, response: Any, correlation_id: Optional[str]):
        """Send a response back to the requesting agent."""
        if isinstance(response, BaseModel):
            payload = response.model_dump()
        else:
            payload = {"result": response}

        message = AgentMessage(
            message_type=MessageType.TASK_RESPONSE,
            source_agent=self.name,
            target_agent=target,
            payload=payload,
            correlation_id=correlation_id,
        )

        if self.message_queue:
            await self.message_queue.publish(target, message)

    async def _receive_knowledge(self, payload: Dict[str, Any]):
        """Receive and process shared knowledge."""
        knowledge = KnowledgeItem(**payload)

        if self.knowledge_base:
            await self.knowledge_base.store(knowledge)

        logger.debug(f"Knowledge received", agent_id=self.id, type=knowledge.knowledge_type)

    async def _make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make a decision based on context. Can be overridden by subclasses."""
        # Default implementation uses LLM
        prompt = f"""Based on the following context, make a decision:

Context: {context}

Provide your decision in JSON format with 'decision' and 'reasoning' fields."""

        response = await self.invoke_llm(prompt)
        return {"decision": response, "agent": self.name}

    async def _handle_alert(self, alert: Dict[str, Any]):
        """Handle an incoming alert. Can be overridden by subclasses."""
        logger.warning(f"Alert received", agent_id=self.id, alert=alert)

    async def _health_check(self):
        """Perform periodic health check."""
        self.metrics.last_heartbeat = datetime.utcnow()
        self.metrics.uptime_seconds = int(
            (datetime.utcnow() - self._start_time).total_seconds()
        )

    def get_metrics(self) -> AgentMetrics:
        """Get current agent metrics."""
        return self.metrics

    async def shutdown(self):
        """Gracefully shutdown the agent."""
        logger.info(f"Agent shutting down", agent_id=self.id)
        self.status = AgentStatus.OFFLINE

        # Save any pending knowledge
        if self.knowledge_base:
            await self.knowledge_base.flush()

        # Clear memory (only if it was initialized)
        if self._memory is not None and hasattr(self._memory, 'clear'):
            self._memory.clear()

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, status={self.status})"

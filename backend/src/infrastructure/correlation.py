"""Message Correlation Tracking for AURA platform"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from enum import Enum
import uuid

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class ConversationStatus(str, Enum):
    """Status of a conversation"""
    ACTIVE = "active"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    ERROR = "error"


class CorrelatedMessage(BaseModel):
    """A message in a correlated conversation"""
    message_id: str
    correlation_id: str
    source_agent: str
    target_agent: Optional[str] = None
    message_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    parent_message_id: Optional[str] = None
    payload_summary: str = ""


class Conversation(BaseModel):
    """A conversation between agents"""
    correlation_id: str
    initiator: str
    participants: List[str] = Field(default_factory=list)
    messages: List[CorrelatedMessage] = Field(default_factory=list)
    status: ConversationStatus = ConversationStatus.ACTIVE
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)


class CorrelationTracker:
    """
    Tracks message correlations across agent conversations.

    Features:
    - Conversation tracking with correlation IDs
    - Message chain visualization
    - Timeout handling
    - Context propagation
    - Conversation history
    """

    def __init__(
        self,
        timeout_seconds: int = 300,
        max_conversations: int = 10000,
    ):
        self._conversations: Dict[str, Conversation] = {}
        self._message_index: Dict[str, str] = {}  # message_id -> correlation_id
        self._timeout_seconds = timeout_seconds
        self._max_conversations = max_conversations
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None

        logger.info(
            "CorrelationTracker initialized",
            timeout=timeout_seconds,
            max_conversations=max_conversations,
        )

    async def start(self):
        """Start the correlation tracker"""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Correlation tracker started")

    async def stop(self):
        """Stop the correlation tracker"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Correlation tracker stopped")

    def create_correlation_id(self) -> str:
        """Generate a new correlation ID"""
        return f"corr_{uuid.uuid4().hex[:16]}"

    async def start_conversation(
        self,
        initiator: str,
        correlation_id: Optional[str] = None,
        context: Dict[str, Any] = None,
    ) -> str:
        """Start a new conversation"""
        if correlation_id is None:
            correlation_id = self.create_correlation_id()

        conversation = Conversation(
            correlation_id=correlation_id,
            initiator=initiator,
            participants=[initiator],
            context=context or {},
        )

        self._conversations[correlation_id] = conversation

        # Cleanup old conversations if limit reached
        if len(self._conversations) > self._max_conversations:
            await self._cleanup_old_conversations()

        logger.debug(
            "Conversation started",
            correlation_id=correlation_id,
            initiator=initiator,
        )

        return correlation_id

    async def track_message(
        self,
        message_id: str,
        correlation_id: str,
        source_agent: str,
        target_agent: Optional[str] = None,
        message_type: str = "unknown",
        parent_message_id: Optional[str] = None,
        payload_summary: str = "",
    ) -> bool:
        """Track a message in a conversation"""
        # Create conversation if it doesn't exist
        if correlation_id not in self._conversations:
            await self.start_conversation(source_agent, correlation_id)

        conversation = self._conversations[correlation_id]

        # Create correlated message
        correlated = CorrelatedMessage(
            message_id=message_id,
            correlation_id=correlation_id,
            source_agent=source_agent,
            target_agent=target_agent,
            message_type=message_type,
            parent_message_id=parent_message_id,
            payload_summary=payload_summary,
        )

        # Add to conversation
        conversation.messages.append(correlated)

        # Update participants
        if source_agent not in conversation.participants:
            conversation.participants.append(source_agent)
        if target_agent and target_agent not in conversation.participants:
            conversation.participants.append(target_agent)

        # Index message
        self._message_index[message_id] = correlation_id

        return True

    async def complete_conversation(
        self,
        correlation_id: str,
        status: ConversationStatus = ConversationStatus.COMPLETED,
    ) -> bool:
        """Mark a conversation as completed"""
        if correlation_id not in self._conversations:
            return False

        conversation = self._conversations[correlation_id]
        conversation.status = status
        conversation.ended_at = datetime.utcnow()

        logger.debug(
            "Conversation completed",
            correlation_id=correlation_id,
            status=status.value,
            message_count=len(conversation.messages),
        )

        return True

    def get_conversation(self, correlation_id: str) -> Optional[Conversation]:
        """Get a conversation by correlation ID"""
        return self._conversations.get(correlation_id)

    def get_conversation_for_message(self, message_id: str) -> Optional[Conversation]:
        """Get the conversation containing a message"""
        correlation_id = self._message_index.get(message_id)
        if correlation_id:
            return self._conversations.get(correlation_id)
        return None

    def get_message_chain(self, message_id: str) -> List[CorrelatedMessage]:
        """Get the chain of messages leading to a specific message"""
        conversation = self.get_conversation_for_message(message_id)
        if not conversation:
            return []

        chain = []
        current_id = message_id

        # Build chain by following parent links
        message_map = {m.message_id: m for m in conversation.messages}

        while current_id:
            message = message_map.get(current_id)
            if message:
                chain.insert(0, message)
                current_id = message.parent_message_id
            else:
                break

        return chain

    def get_context(self, correlation_id: str) -> Dict[str, Any]:
        """Get context for a conversation"""
        conversation = self._conversations.get(correlation_id)
        if conversation:
            return conversation.context.copy()
        return {}

    async def update_context(
        self,
        correlation_id: str,
        updates: Dict[str, Any],
    ) -> bool:
        """Update context for a conversation"""
        if correlation_id not in self._conversations:
            return False

        self._conversations[correlation_id].context.update(updates)
        return True

    def list_active_conversations(self) -> List[Conversation]:
        """List all active conversations"""
        return [
            conv for conv in self._conversations.values()
            if conv.status == ConversationStatus.ACTIVE
        ]

    def get_agent_conversations(self, agent_name: str) -> List[Conversation]:
        """Get all conversations involving an agent"""
        return [
            conv for conv in self._conversations.values()
            if agent_name in conv.participants
        ]

    async def _cleanup_loop(self):
        """Background loop to cleanup timed out conversations"""
        while self._running:
            try:
                await self._cleanup_timed_out()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in cleanup loop", error=str(e))
                await asyncio.sleep(5)

    async def _cleanup_timed_out(self):
        """Mark timed out conversations"""
        now = datetime.utcnow()
        timeout_threshold = now - timedelta(seconds=self._timeout_seconds)

        for conv in self._conversations.values():
            if conv.status == ConversationStatus.ACTIVE:
                # Check last message time
                if conv.messages:
                    last_message_time = conv.messages[-1].timestamp
                else:
                    last_message_time = conv.started_at

                if last_message_time < timeout_threshold:
                    conv.status = ConversationStatus.TIMEOUT
                    conv.ended_at = now
                    logger.warning(
                        "Conversation timed out",
                        correlation_id=conv.correlation_id,
                    )

    async def _cleanup_old_conversations(self):
        """Remove old completed conversations"""
        # Sort by ended_at and remove oldest
        completed = [
            (cid, conv) for cid, conv in self._conversations.items()
            if conv.status != ConversationStatus.ACTIVE
        ]

        completed.sort(key=lambda x: x[1].ended_at or datetime.min)

        # Remove oldest 10%
        to_remove = len(completed) // 10
        for cid, _ in completed[:to_remove]:
            # Clean up message index
            conv = self._conversations[cid]
            for msg in conv.messages:
                self._message_index.pop(msg.message_id, None)
            del self._conversations[cid]

    def get_stats(self) -> Dict[str, Any]:
        """Get correlation tracker statistics"""
        status_counts = {}
        for conv in self._conversations.values():
            status_counts[conv.status.value] = status_counts.get(conv.status.value, 0) + 1

        total_messages = sum(len(c.messages) for c in self._conversations.values())

        return {
            "total_conversations": len(self._conversations),
            "total_messages_tracked": total_messages,
            "by_status": status_counts,
            "active_conversations": status_counts.get("active", 0),
        }


# Global correlation tracker
correlation_tracker: Optional[CorrelationTracker] = None


def get_correlation_tracker() -> CorrelationTracker:
    """Get the global correlation tracker"""
    global correlation_tracker
    if correlation_tracker is None:
        correlation_tracker = CorrelationTracker()
    return correlation_tracker

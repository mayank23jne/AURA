"""Event streaming system for AURA platform"""

import asyncio
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import uuid

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class EventType(str, Enum):
    """Types of events in the AURA system"""
    # Audit events
    AUDIT_STARTED = "audit.started"
    AUDIT_COMPLETED = "audit.completed"
    AUDIT_FAILED = "audit.failed"

    # Test events
    TEST_STARTED = "test.started"
    TEST_COMPLETED = "test.completed"
    TEST_FAILED = "test.failed"

    # Policy events
    POLICY_CREATED = "policy.created"
    POLICY_UPDATED = "policy.updated"
    POLICY_DELETED = "policy.deleted"

    # Agent events
    AGENT_STARTED = "agent.started"
    AGENT_STOPPED = "agent.stopped"
    AGENT_ERROR = "agent.error"

    # Compliance events
    COMPLIANCE_VIOLATION = "compliance.violation"
    COMPLIANCE_DRIFT = "compliance.drift"
    COMPLIANCE_RESOLVED = "compliance.resolved"

    # Knowledge events
    KNOWLEDGE_ADDED = "knowledge.added"
    KNOWLEDGE_UPDATED = "knowledge.updated"

    # System events
    SYSTEM_ALERT = "system.alert"
    SYSTEM_METRIC = "system.metric"


class Event(BaseModel):
    """Event model for the event stream"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    source: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    correlation_id: Optional[str] = None


class EventStream:
    """
    Event streaming system for real-time event processing.

    Features:
    - Pub/sub event model
    - Event filtering by type and source
    - Event replay from history
    - Async event handlers
    """

    def __init__(self, max_history: int = 10000):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._event_history: List[Event] = []
        self._max_history = max_history
        self._running = False
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._processor_task: Optional[asyncio.Task] = None

        logger.info("EventStream initialized", max_history=max_history)

    async def start(self):
        """Start the event processor"""
        self._running = True
        self._processor_task = asyncio.create_task(self._process_events())
        logger.info("Event stream started")

    async def stop(self):
        """Stop the event processor"""
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        logger.info("Event stream stopped")

    async def emit(self, event: Event):
        """Emit an event to the stream"""
        await self._event_queue.put(event)
        logger.debug(
            "Event emitted",
            event_type=event.event_type,
            source=event.source,
        )

    async def emit_simple(
        self,
        event_type: EventType,
        source: str,
        data: Dict[str, Any] = None,
        correlation_id: Optional[str] = None,
    ):
        """Simplified event emission"""
        event = Event(
            event_type=event_type,
            source=source,
            data=data or {},
            correlation_id=correlation_id,
        )
        await self.emit(event)

    def subscribe(
        self,
        event_types: List[EventType],
        callback: Callable,
        source_filter: Optional[str] = None,
    ) -> str:
        """Subscribe to specific event types"""
        subscription_id = str(uuid.uuid4())

        for event_type in event_types:
            key = f"{event_type.value}:{source_filter or '*'}"
            if key not in self._subscribers:
                self._subscribers[key] = []
            self._subscribers[key].append((subscription_id, callback))

        logger.info(
            "Event subscription added",
            subscription_id=subscription_id,
            event_types=[et.value for et in event_types],
        )
        return subscription_id

    def unsubscribe(self, subscription_id: str):
        """Remove a subscription"""
        for key in list(self._subscribers.keys()):
            self._subscribers[key] = [
                (sid, cb)
                for sid, cb in self._subscribers[key]
                if sid != subscription_id
            ]
            if not self._subscribers[key]:
                del self._subscribers[key]

        logger.info("Event subscription removed", subscription_id=subscription_id)

    async def _process_events(self):
        """Process events from the queue"""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(), timeout=0.1
                )

                # Store in history
                self._event_history.append(event)
                if len(self._event_history) > self._max_history:
                    self._event_history.pop(0)

                # Dispatch to subscribers
                await self._dispatch_event(event)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("Error processing event", error=str(e))

    async def _dispatch_event(self, event: Event):
        """Dispatch event to matching subscribers"""
        # Check exact match first
        exact_key = f"{event.event_type.value}:{event.source}"
        wildcard_key = f"{event.event_type.value}:*"

        callbacks = []
        if exact_key in self._subscribers:
            callbacks.extend(self._subscribers[exact_key])
        if wildcard_key in self._subscribers:
            callbacks.extend(self._subscribers[wildcard_key])

        for subscription_id, callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(
                    "Event callback failed",
                    subscription_id=subscription_id,
                    error=str(e),
                )

    def get_history(
        self,
        event_types: Optional[List[EventType]] = None,
        source: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Event]:
        """Get events from history with filters"""
        events = self._event_history

        if event_types:
            events = [e for e in events if e.event_type in event_types]

        if source:
            events = [e for e in events if e.source == source]

        if since:
            events = [e for e in events if e.timestamp >= since]

        return events[-limit:]

    async def replay(
        self,
        callback: Callable,
        event_types: Optional[List[EventType]] = None,
        since: Optional[datetime] = None,
    ):
        """Replay historical events to a callback"""
        events = self.get_history(
            event_types=event_types, since=since, limit=self._max_history
        )

        for event in events:
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)

        logger.info("Event replay completed", count=len(events))

    def get_stats(self) -> Dict[str, Any]:
        """Get event stream statistics"""
        type_counts = {}
        for event in self._event_history:
            event_type = event.event_type.value
            type_counts[event_type] = type_counts.get(event_type, 0) + 1

        return {
            "total_events": len(self._event_history),
            "events_by_type": type_counts,
            "subscriber_count": sum(
                len(subs) for subs in self._subscribers.values()
            ),
            "queue_size": self._event_queue.qsize(),
        }

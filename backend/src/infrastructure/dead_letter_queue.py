"""Dead Letter Queue processing for failed messages"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
from enum import Enum

import structlog
from pydantic import BaseModel, Field

from src.core.models import AgentMessage

logger = structlog.get_logger()


class DLQEntryStatus(str, Enum):
    """Status of a DLQ entry"""
    PENDING = "pending"
    RETRYING = "retrying"
    PROCESSED = "processed"
    DISCARDED = "discarded"
    MANUAL_REVIEW = "manual_review"


class DLQEntry(BaseModel):
    """Entry in the dead letter queue"""
    id: str
    message: AgentMessage
    error: str
    error_type: str
    source_topic: str
    failed_at: datetime = Field(default_factory=datetime.utcnow)
    retry_count: int = 0
    max_retries: int = 3
    next_retry_at: Optional[datetime] = None
    status: DLQEntryStatus = DLQEntryStatus.PENDING
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DLQConfig(BaseModel):
    """Configuration for dead letter queue"""
    max_retries: int = 3
    initial_retry_delay_seconds: int = 60
    max_retry_delay_seconds: int = 3600
    backoff_multiplier: float = 2.0
    retention_days: int = 7
    auto_retry: bool = True
    retry_interval_seconds: int = 30


class DLQMetrics(BaseModel):
    """Metrics for dead letter queue"""
    total_entries: int = 0
    pending_entries: int = 0
    processed_entries: int = 0
    discarded_entries: int = 0
    manual_review_entries: int = 0
    retry_success_count: int = 0
    retry_failure_count: int = 0


class DeadLetterQueue:
    """
    Dead Letter Queue for handling failed messages.

    Features:
    - Failed message storage with metadata
    - Configurable retry with exponential backoff
    - Manual review workflow
    - Automatic expiration
    - Metrics tracking
    - Message reprocessing
    """

    def __init__(self, config: DLQConfig = None):
        self.config = config or DLQConfig()
        self._entries: Dict[str, DLQEntry] = {}
        self._metrics = DLQMetrics()
        self._handlers: Dict[str, Callable] = {}
        self._running = False
        self._retry_task: Optional[asyncio.Task] = None

        logger.info("DeadLetterQueue initialized", config=self.config.model_dump())

    async def start(self):
        """Start the DLQ processor"""
        self._running = True
        if self.config.auto_retry:
            self._retry_task = asyncio.create_task(self._retry_loop())
        logger.info("Dead letter queue started")

    async def stop(self):
        """Stop the DLQ processor"""
        self._running = False
        if self._retry_task:
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass
        logger.info("Dead letter queue stopped")

    async def add(
        self,
        message: AgentMessage,
        error: Exception,
        source_topic: str,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Add a failed message to the DLQ"""
        entry_id = f"dlq_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{len(self._entries)}"

        entry = DLQEntry(
            id=entry_id,
            message=message,
            error=str(error),
            error_type=type(error).__name__,
            source_topic=source_topic,
            max_retries=self.config.max_retries,
            metadata=metadata or {},
        )

        # Calculate first retry time
        entry.next_retry_at = datetime.utcnow() + timedelta(
            seconds=self.config.initial_retry_delay_seconds
        )

        self._entries[entry_id] = entry
        self._metrics.total_entries += 1
        self._metrics.pending_entries += 1

        logger.warning(
            "Message added to DLQ",
            entry_id=entry_id,
            error=entry.error,
            source_topic=source_topic,
        )

        return entry_id

    def register_handler(self, topic: str, handler: Callable):
        """Register a handler for reprocessing messages from a topic"""
        self._handlers[topic] = handler
        logger.info("DLQ handler registered", topic=topic)

    async def retry(self, entry_id: str) -> bool:
        """Manually retry a DLQ entry"""
        if entry_id not in self._entries:
            return False

        entry = self._entries[entry_id]

        if entry.status not in [DLQEntryStatus.PENDING, DLQEntryStatus.MANUAL_REVIEW]:
            return False

        return await self._process_retry(entry)

    async def _process_retry(self, entry: DLQEntry) -> bool:
        """Process a retry attempt"""
        entry.status = DLQEntryStatus.RETRYING
        entry.retry_count += 1

        handler = self._handlers.get(entry.source_topic)
        if not handler:
            logger.warning(
                "No handler for topic",
                entry_id=entry.id,
                topic=entry.source_topic,
            )
            entry.status = DLQEntryStatus.MANUAL_REVIEW
            self._metrics.manual_review_entries += 1
            return False

        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(entry.message)
            else:
                handler(entry.message)

            # Success
            entry.status = DLQEntryStatus.PROCESSED
            self._metrics.processed_entries += 1
            self._metrics.pending_entries -= 1
            self._metrics.retry_success_count += 1

            logger.info(
                "DLQ entry processed successfully",
                entry_id=entry.id,
                retry_count=entry.retry_count,
            )

            return True

        except Exception as e:
            self._metrics.retry_failure_count += 1

            if entry.retry_count >= entry.max_retries:
                # Max retries exceeded
                entry.status = DLQEntryStatus.MANUAL_REVIEW
                self._metrics.manual_review_entries += 1
                self._metrics.pending_entries -= 1

                logger.error(
                    "DLQ entry max retries exceeded",
                    entry_id=entry.id,
                    retry_count=entry.retry_count,
                    error=str(e),
                )
            else:
                # Schedule next retry with exponential backoff
                delay = min(
                    self.config.initial_retry_delay_seconds * (
                        self.config.backoff_multiplier ** entry.retry_count
                    ),
                    self.config.max_retry_delay_seconds,
                )
                entry.next_retry_at = datetime.utcnow() + timedelta(seconds=delay)
                entry.status = DLQEntryStatus.PENDING
                entry.error = str(e)

                logger.warning(
                    "DLQ retry failed, scheduling next attempt",
                    entry_id=entry.id,
                    retry_count=entry.retry_count,
                    next_retry=entry.next_retry_at.isoformat(),
                )

            return False

    async def discard(self, entry_id: str, reason: str = "") -> bool:
        """Discard a DLQ entry"""
        if entry_id not in self._entries:
            return False

        entry = self._entries[entry_id]
        entry.status = DLQEntryStatus.DISCARDED
        entry.metadata["discard_reason"] = reason
        entry.metadata["discarded_at"] = datetime.utcnow().isoformat()

        if entry.status == DLQEntryStatus.PENDING:
            self._metrics.pending_entries -= 1
        elif entry.status == DLQEntryStatus.MANUAL_REVIEW:
            self._metrics.manual_review_entries -= 1

        self._metrics.discarded_entries += 1

        logger.info("DLQ entry discarded", entry_id=entry_id, reason=reason)
        return True

    async def mark_for_review(self, entry_id: str) -> bool:
        """Mark an entry for manual review"""
        if entry_id not in self._entries:
            return False

        entry = self._entries[entry_id]

        if entry.status == DLQEntryStatus.PENDING:
            self._metrics.pending_entries -= 1

        entry.status = DLQEntryStatus.MANUAL_REVIEW
        self._metrics.manual_review_entries += 1

        logger.info("DLQ entry marked for review", entry_id=entry_id)
        return True

    def get_entry(self, entry_id: str) -> Optional[DLQEntry]:
        """Get a DLQ entry by ID"""
        return self._entries.get(entry_id)

    def list_entries(
        self,
        status: Optional[DLQEntryStatus] = None,
        topic: Optional[str] = None,
        limit: int = 100,
    ) -> List[DLQEntry]:
        """List DLQ entries with optional filters"""
        entries = list(self._entries.values())

        if status:
            entries = [e for e in entries if e.status == status]

        if topic:
            entries = [e for e in entries if e.source_topic == topic]

        # Sort by failed_at descending
        entries.sort(key=lambda e: e.failed_at, reverse=True)

        return entries[:limit]

    async def cleanup_expired(self) -> int:
        """Remove expired entries based on retention policy"""
        cutoff = datetime.utcnow() - timedelta(days=self.config.retention_days)
        expired = []

        for entry_id, entry in list(self._entries.items()):
            if entry.failed_at < cutoff:
                if entry.status in [DLQEntryStatus.PROCESSED, DLQEntryStatus.DISCARDED]:
                    expired.append(entry_id)

        for entry_id in expired:
            del self._entries[entry_id]

        if expired:
            logger.info("Expired DLQ entries cleaned up", count=len(expired))

        return len(expired)

    async def _retry_loop(self):
        """Background loop to process automatic retries"""
        while self._running:
            try:
                now = datetime.utcnow()

                for entry in list(self._entries.values()):
                    if (
                        entry.status == DLQEntryStatus.PENDING
                        and entry.next_retry_at
                        and entry.next_retry_at <= now
                    ):
                        await self._process_retry(entry)

                # Cleanup expired entries periodically
                await self.cleanup_expired()

                await asyncio.sleep(self.config.retry_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in DLQ retry loop", error=str(e))
                await asyncio.sleep(5)

    def get_metrics(self) -> DLQMetrics:
        """Get DLQ metrics"""
        return self._metrics

    def get_stats(self) -> Dict[str, Any]:
        """Get DLQ statistics"""
        status_counts = {}
        topic_counts = {}

        for entry in self._entries.values():
            status_counts[entry.status.value] = status_counts.get(entry.status.value, 0) + 1
            topic_counts[entry.source_topic] = topic_counts.get(entry.source_topic, 0) + 1

        return {
            "total_entries": len(self._entries),
            "by_status": status_counts,
            "by_topic": topic_counts,
            "metrics": self._metrics.model_dump(),
        }


# Global DLQ instance
dead_letter_queue: Optional[DeadLetterQueue] = None


def get_dead_letter_queue() -> DeadLetterQueue:
    """Get the global DLQ instance"""
    global dead_letter_queue
    if dead_letter_queue is None:
        dead_letter_queue = DeadLetterQueue()
    return dead_letter_queue

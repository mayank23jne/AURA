"""Knowledge Decay Mechanism for AURA platform"""

import asyncio
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from pydantic import BaseModel, Field

from src.core.models import KnowledgeItem, KnowledgeType

logger = structlog.get_logger()


class DecayConfig(BaseModel):
    """Configuration for knowledge decay"""
    # Decay rates by knowledge type (confidence reduction per day)
    decay_rates: Dict[str, float] = Field(default_factory=lambda: {
        "rule": 0.001,        # Rules decay slowly
        "pattern": 0.01,      # Patterns decay moderately
        "experience": 0.02,   # Experiences decay faster
        "decision": 0.015,    # Decisions decay moderately
        "insight": 0.025,     # Insights decay faster
    })

    # Minimum confidence before item is considered for removal
    min_confidence: float = 0.1

    # Days after which to consider removal if confidence is low
    stale_threshold_days: int = 90

    # Whether to automatically delete decayed items
    auto_delete: bool = False

    # Interval for running decay process (in hours)
    decay_interval_hours: int = 24

    # Boost factors for reinforced knowledge
    access_boost: float = 0.05      # Boost when knowledge is accessed
    validation_boost: float = 0.1   # Boost when knowledge is validated
    max_confidence: float = 1.0     # Maximum confidence cap


class DecayMetrics(BaseModel):
    """Metrics for knowledge decay process"""
    items_processed: int = 0
    items_decayed: int = 0
    items_deleted: int = 0
    items_boosted: int = 0
    last_run: Optional[datetime] = None
    average_confidence: float = 0.0
    stale_items_count: int = 0


class KnowledgeDecayManager:
    """
    Manages knowledge decay and freshness.

    Features:
    - Time-based confidence decay
    - Type-specific decay rates
    - Access-based confidence boosting
    - Automatic stale knowledge removal
    - Decay metrics tracking
    """

    def __init__(self, config: DecayConfig = None):
        self.config = config or DecayConfig()
        self._metrics = DecayMetrics()
        self._access_log: Dict[str, datetime] = {}
        self._running = False
        self._decay_task: Optional[asyncio.Task] = None

        logger.info("KnowledgeDecayManager initialized", config=self.config.model_dump())

    async def start(self, knowledge_base):
        """Start the decay manager"""
        self._knowledge_base = knowledge_base
        self._running = True
        self._decay_task = asyncio.create_task(self._decay_loop())
        logger.info("Knowledge decay manager started")

    async def stop(self):
        """Stop the decay manager"""
        self._running = False
        if self._decay_task:
            self._decay_task.cancel()
            try:
                await self._decay_task
            except asyncio.CancelledError:
                pass
        logger.info("Knowledge decay manager stopped")

    async def apply_decay(self, items: List[KnowledgeItem]) -> Dict[str, Any]:
        """Apply decay to a list of knowledge items"""
        now = datetime.utcnow()
        results = {
            "processed": 0,
            "decayed": 0,
            "deleted": 0,
            "boosted": 0,
        }

        for item in items:
            results["processed"] += 1

            # Calculate days since last update
            days_old = (now - item.timestamp).total_seconds() / 86400

            # Get decay rate for this knowledge type
            type_key = item.knowledge_type.value if isinstance(item.knowledge_type, KnowledgeType) else str(item.knowledge_type)
            decay_rate = self.config.decay_rates.get(type_key, 0.01)

            # Apply exponential decay
            decay_factor = math.exp(-decay_rate * days_old)
            new_confidence = item.confidence * decay_factor

            # Check if item was recently accessed (boost)
            if item.id in self._access_log:
                access_time = self._access_log[item.id]
                if (now - access_time).total_seconds() < 3600:  # Within last hour
                    new_confidence = min(
                        new_confidence + self.config.access_boost,
                        self.config.max_confidence
                    )
                    results["boosted"] += 1

            # Update confidence if changed significantly
            if abs(new_confidence - item.confidence) > 0.001:
                item.confidence = max(new_confidence, 0.0)
                results["decayed"] += 1

                # Check for deletion
                if (
                    self.config.auto_delete
                    and item.confidence < self.config.min_confidence
                    and days_old > self.config.stale_threshold_days
                ):
                    # Mark for deletion (actual deletion handled by caller)
                    results["deleted"] += 1

        return results

    def calculate_decay(
        self,
        original_confidence: float,
        knowledge_type: str,
        age_days: float,
    ) -> float:
        """Calculate decayed confidence for a knowledge item"""
        decay_rate = self.config.decay_rates.get(knowledge_type, 0.01)
        decay_factor = math.exp(-decay_rate * age_days)
        return original_confidence * decay_factor

    def record_access(self, item_id: str):
        """Record that a knowledge item was accessed"""
        self._access_log[item_id] = datetime.utcnow()
        self._metrics.items_boosted += 1

    def record_validation(self, item_id: str):
        """Record that a knowledge item was validated/confirmed"""
        # Validation provides a stronger boost than simple access
        self._access_log[item_id] = datetime.utcnow()

    async def boost_confidence(
        self,
        item: KnowledgeItem,
        boost_type: str = "access",
    ) -> float:
        """Boost confidence of a knowledge item"""
        if boost_type == "access":
            boost = self.config.access_boost
        elif boost_type == "validation":
            boost = self.config.validation_boost
        else:
            boost = 0.0

        new_confidence = min(
            item.confidence + boost,
            self.config.max_confidence
        )

        return new_confidence

    async def get_stale_items(
        self,
        items: List[KnowledgeItem],
    ) -> List[KnowledgeItem]:
        """Get items that are stale and candidates for removal"""
        now = datetime.utcnow()
        stale = []

        for item in items:
            days_old = (now - item.timestamp).total_seconds() / 86400

            if (
                item.confidence < self.config.min_confidence
                and days_old > self.config.stale_threshold_days
            ):
                stale.append(item)

        return stale

    async def _decay_loop(self):
        """Background loop to periodically apply decay"""
        interval_seconds = self.config.decay_interval_hours * 3600

        while self._running:
            try:
                await asyncio.sleep(interval_seconds)

                if hasattr(self, '_knowledge_base') and self._knowledge_base:
                    # Get all items from knowledge base
                    items = list(self._knowledge_base._items.values())

                    # Apply decay
                    results = await self.apply_decay(items)

                    # Update metrics
                    self._metrics.items_processed += results["processed"]
                    self._metrics.items_decayed += results["decayed"]
                    self._metrics.items_deleted += results["deleted"]
                    self._metrics.last_run = datetime.utcnow()

                    # Calculate average confidence
                    if items:
                        self._metrics.average_confidence = sum(
                            i.confidence for i in items
                        ) / len(items)

                    # Count stale items
                    stale = await self.get_stale_items(items)
                    self._metrics.stale_items_count = len(stale)

                    # Delete if auto-delete enabled
                    if self.config.auto_delete:
                        for item in stale:
                            await self._knowledge_base.delete(item.id)

                    logger.info(
                        "Decay cycle completed",
                        processed=results["processed"],
                        decayed=results["decayed"],
                        deleted=results["deleted"],
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in decay loop", error=str(e))
                await asyncio.sleep(60)

    def get_metrics(self) -> DecayMetrics:
        """Get decay metrics"""
        return self._metrics

    def get_decay_rate(self, knowledge_type: str) -> float:
        """Get decay rate for a knowledge type"""
        return self.config.decay_rates.get(knowledge_type, 0.01)

    def set_decay_rate(self, knowledge_type: str, rate: float):
        """Set decay rate for a knowledge type"""
        self.config.decay_rates[knowledge_type] = rate
        logger.info(
            "Decay rate updated",
            knowledge_type=knowledge_type,
            rate=rate,
        )

    def estimate_confidence_at(
        self,
        item: KnowledgeItem,
        future_date: datetime,
    ) -> float:
        """Estimate what the confidence will be at a future date"""
        now = datetime.utcnow()
        days_until = (future_date - now).total_seconds() / 86400
        current_age = (now - item.timestamp).total_seconds() / 86400

        type_key = item.knowledge_type.value if isinstance(item.knowledge_type, KnowledgeType) else str(item.knowledge_type)

        return self.calculate_decay(
            item.confidence,
            type_key,
            current_age + days_until
        )

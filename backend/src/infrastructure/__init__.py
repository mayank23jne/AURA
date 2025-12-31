"""Infrastructure components for AURA Agentic Platform"""

from .message_bus import MessageBus, InMemoryMessageBus, KafkaMessageBus
from .event_stream import EventStream, EventType
from .schema_registry import SchemaRegistry, SchemaCompatibility, get_schema_registry
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerRegistry,
    get_circuit_breaker_registry,
    circuit_breaker,
)
from .discovery import (
    DiscoveryService,
    AgentRegistration,
    AgentCapability,
    get_discovery_service,
)
from .dead_letter_queue import (
    DeadLetterQueue,
    DLQEntry,
    DLQConfig,
    DLQEntryStatus,
    get_dead_letter_queue,
)
from .correlation import (
    CorrelationTracker,
    Conversation,
    CorrelatedMessage,
    get_correlation_tracker,
)

__all__ = [
    "MessageBus",
    "InMemoryMessageBus",
    "KafkaMessageBus",
    "EventStream",
    "EventType",
    "SchemaRegistry",
    "SchemaCompatibility",
    "get_schema_registry",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "CircuitBreakerRegistry",
    "get_circuit_breaker_registry",
    "circuit_breaker",
    "DiscoveryService",
    "AgentRegistration",
    "AgentCapability",
    "get_discovery_service",
    "DeadLetterQueue",
    "DLQEntry",
    "DLQConfig",
    "DLQEntryStatus",
    "get_dead_letter_queue",
    "CorrelationTracker",
    "Conversation",
    "CorrelatedMessage",
    "get_correlation_tracker",
]

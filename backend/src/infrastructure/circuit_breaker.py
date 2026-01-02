"""Circuit Breaker pattern implementation for fault tolerance"""

import asyncio
import time
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker"""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 3          # Successes to close from half-open
    timeout_seconds: float = 30.0       # Time before trying half-open
    half_open_max_calls: int = 3        # Max calls in half-open state
    excluded_exceptions: list = Field(default_factory=list)  # Don't count these


class CircuitBreakerMetrics(BaseModel):
    """Metrics for circuit breaker"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    current_state: str = CircuitState.CLOSED.value


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Circuit breaker for fault tolerance.

    Features:
    - Three states: closed, open, half-open
    - Configurable thresholds
    - Automatic recovery testing
    - Metrics tracking
    - Async support
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0

        self._metrics = CircuitBreakerMetrics()
        self._lock = asyncio.Lock()

        logger.info(
            "Circuit breaker initialized",
            name=name,
            failure_threshold=self.config.failure_threshold,
        )

    @property
    def state(self) -> CircuitState:
        """Get current state"""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed"""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open"""
        return self._state == CircuitState.OPEN

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        async with self._lock:
            # Check if we should try to recover
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    self._metrics.rejected_calls += 1
                    logger.warning(
                        "Circuit breaker open, rejecting call",
                        name=self.name,
                    )
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is open"
                    )

            # Check half-open call limit
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    self._metrics.rejected_calls += 1
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' half-open call limit reached"
                    )
                self._half_open_calls += 1

        # Execute the call
        self._metrics.total_calls += 1

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            await self._on_success()
            return result

        except Exception as e:
            # Check if exception should be excluded
            if type(e) in self.config.excluded_exceptions:
                await self._on_success()
                raise

            await self._on_failure(e)
            raise

    async def _on_success(self):
        """Handle successful call"""
        async with self._lock:
            self._metrics.successful_calls += 1
            self._metrics.last_success_time = datetime.utcnow()

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            else:
                self._failure_count = 0

    async def _on_failure(self, exception: Exception):
        """Handle failed call"""
        async with self._lock:
            self._metrics.failed_calls += 1
            self._metrics.last_failure_time = datetime.utcnow()
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._transition_to_open()
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to_open()

            logger.warning(
                "Circuit breaker recorded failure",
                name=self.name,
                failure_count=self._failure_count,
                state=self._state.value,
                error=str(exception),
            )

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self._last_failure_time is None:
            return True

        elapsed = time.time() - self._last_failure_time
        return elapsed >= self.config.timeout_seconds

    def _transition_to_open(self):
        """Transition to open state"""
        self._state = CircuitState.OPEN
        self._metrics.state_changes += 1
        self._metrics.current_state = CircuitState.OPEN.value
        logger.warning("Circuit breaker opened", name=self.name)

    def _transition_to_half_open(self):
        """Transition to half-open state"""
        self._state = CircuitState.HALF_OPEN
        self._half_open_calls = 0
        self._success_count = 0
        self._metrics.state_changes += 1
        self._metrics.current_state = CircuitState.HALF_OPEN.value
        logger.info("Circuit breaker half-open", name=self.name)

    def _transition_to_closed(self):
        """Transition to closed state"""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._metrics.state_changes += 1
        self._metrics.current_state = CircuitState.CLOSED.value
        logger.info("Circuit breaker closed", name=self.name)

    def reset(self):
        """Manually reset circuit breaker"""
        self._transition_to_closed()
        logger.info("Circuit breaker manually reset", name=self.name)

    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get circuit breaker metrics"""
        return self._metrics


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers"""

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}

    def get_or_create(
        self,
        name: str,
        config: CircuitBreakerConfig = None,
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker"""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, config)
        return self._breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        return self._breakers.get(name)

    def reset_all(self):
        """Reset all circuit breakers"""
        for breaker in self._breakers.values():
            breaker.reset()

    def get_all_metrics(self) -> Dict[str, CircuitBreakerMetrics]:
        """Get metrics for all circuit breakers"""
        return {
            name: breaker.get_metrics()
            for name, breaker in self._breakers.items()
        }

    def list_breakers(self) -> list:
        """List all circuit breaker names and states"""
        return [
            {"name": name, "state": breaker.state.value}
            for name, breaker in self._breakers.items()
        ]


# Global registry
circuit_breaker_registry = CircuitBreakerRegistry()


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry"""
    return circuit_breaker_registry


def circuit_breaker(name: str, config: CircuitBreakerConfig = None):
    """Decorator to protect a function with circuit breaker"""
    def decorator(func: Callable) -> Callable:
        breaker = circuit_breaker_registry.get_or_create(name, config)

        async def async_wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)

        def sync_wrapper(*args, **kwargs):
            import asyncio
            return asyncio.get_event_loop().run_until_complete(
                breaker.call(func, *args, **kwargs)
            )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator

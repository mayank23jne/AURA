"""API Rate Limiting Middleware for AURA Platform"""

import asyncio
import time
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib

import structlog
from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger()


class RateLimitAlgorithm(str, Enum):
    """Rate limiting algorithms"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


class RateLimitTier(str, Enum):
    """Rate limit tiers"""
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    UNLIMITED = "unlimited"


class RateLimitConfig(BaseModel):
    """Rate limit configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 10
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET


class EndpointRateLimit(BaseModel):
    """Rate limit for a specific endpoint"""
    path_pattern: str
    method: str = "*"  # * for all methods
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 10
    priority: int = 0  # Higher priority limits are checked first


class ClientRateLimit(BaseModel):
    """Rate limit state for a client"""
    client_id: str
    tier: RateLimitTier = RateLimitTier.FREE
    tokens: float = 0.0
    last_refill: float = 0.0

    # Sliding window counters
    minute_requests: List[float] = Field(default_factory=list)
    hour_requests: List[float] = Field(default_factory=list)
    day_requests: List[float] = Field(default_factory=list)

    # Statistics
    total_requests: int = 0
    rejected_requests: int = 0
    last_request: Optional[float] = None


class RateLimitResponse(BaseModel):
    """Rate limit response information"""
    allowed: bool
    remaining: int
    limit: int
    reset_at: datetime
    retry_after: Optional[int] = None


class TokenBucket:
    """Token bucket rate limiter"""

    def __init__(
        self,
        capacity: int,
        refill_rate: float,  # tokens per second
    ):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()

    def consume(self, tokens: int = 1) -> Tuple[bool, float]:
        """Try to consume tokens. Returns (success, tokens_remaining)"""
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True, self.tokens
        else:
            return False, self.tokens

    def _refill(self):
        """Refill tokens based on time elapsed"""
        now = time.time()
        elapsed = now - self.last_refill
        refill_amount = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + refill_amount)
        self.last_refill = now


class SlidingWindow:
    """Sliding window rate limiter"""

    def __init__(self, window_seconds: int, max_requests: int):
        self.window_seconds = window_seconds
        self.max_requests = max_requests
        self.requests: List[float] = []

    def allow(self) -> Tuple[bool, int]:
        """Check if request is allowed. Returns (allowed, remaining)"""
        now = time.time()
        cutoff = now - self.window_seconds

        # Remove old requests
        self.requests = [r for r in self.requests if r > cutoff]

        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True, self.max_requests - len(self.requests)
        else:
            return False, 0

    def get_reset_time(self) -> float:
        """Get time until oldest request expires"""
        if not self.requests:
            return 0
        return max(0, self.requests[0] + self.window_seconds - time.time())


class RateLimiterStore:
    """Storage for rate limiter state"""

    def __init__(self):
        self._buckets: Dict[str, TokenBucket] = {}
        self._windows: Dict[str, Dict[str, SlidingWindow]] = defaultdict(dict)
        self._clients: Dict[str, ClientRateLimit] = {}
        self._stats: Dict[str, Any] = {
            "total_requests": 0,
            "allowed_requests": 0,
            "rejected_requests": 0,
        }

    def get_bucket(
        self, client_id: str, capacity: int, refill_rate: float
    ) -> TokenBucket:
        """Get or create token bucket for client"""
        if client_id not in self._buckets:
            self._buckets[client_id] = TokenBucket(capacity, refill_rate)
        return self._buckets[client_id]

    def get_window(
        self, client_id: str, window_name: str, window_seconds: int, max_requests: int
    ) -> SlidingWindow:
        """Get or create sliding window for client"""
        if window_name not in self._windows[client_id]:
            self._windows[client_id][window_name] = SlidingWindow(
                window_seconds, max_requests
            )
        return self._windows[client_id][window_name]

    def get_client(self, client_id: str) -> ClientRateLimit:
        """Get or create client rate limit state"""
        if client_id not in self._clients:
            self._clients[client_id] = ClientRateLimit(client_id=client_id)
        return self._clients[client_id]

    def record_request(self, client_id: str, allowed: bool):
        """Record a request"""
        self._stats["total_requests"] += 1
        if allowed:
            self._stats["allowed_requests"] += 1
        else:
            self._stats["rejected_requests"] += 1

        client = self.get_client(client_id)
        client.total_requests += 1
        client.last_request = time.time()
        if not allowed:
            client.rejected_requests += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        return {
            **self._stats,
            "unique_clients": len(self._clients),
            "rejection_rate": (
                self._stats["rejected_requests"] / self._stats["total_requests"]
                if self._stats["total_requests"] > 0
                else 0
            ),
        }


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    FastAPI rate limiting middleware.

    Features:
    - Multiple algorithms (token bucket, sliding window)
    - Per-client rate limiting
    - Per-endpoint rate limiting
    - Tiered rate limits
    - Rate limit headers
    - Customizable responses
    """

    def __init__(
        self,
        app,
        default_config: RateLimitConfig = None,
        tier_configs: Dict[RateLimitTier, RateLimitConfig] = None,
        endpoint_limits: List[EndpointRateLimit] = None,
        get_client_id: Callable[[Request], str] = None,
        get_client_tier: Callable[[Request], RateLimitTier] = None,
        exclude_paths: List[str] = None,
    ):
        super().__init__(app)
        self.default_config = default_config or RateLimitConfig()
        self.store = RateLimiterStore()

        # Tier configurations
        self.tier_configs = tier_configs or self._default_tier_configs()

        # Endpoint-specific limits
        self.endpoint_limits = sorted(
            endpoint_limits or [],
            key=lambda x: x.priority,
            reverse=True,
        )

        # Client identification
        self._get_client_id = get_client_id or self._default_client_id
        self._get_client_tier = get_client_tier or self._default_client_tier

        # Excluded paths
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/docs", "/openapi.json"]

        logger.info(
            "RateLimiterMiddleware initialized",
            algorithm=self.default_config.algorithm.value,
        )

    def _default_tier_configs(self) -> Dict[RateLimitTier, RateLimitConfig]:
        """Default tier configurations"""
        return {
            RateLimitTier.FREE: RateLimitConfig(
                requests_per_minute=30,
                requests_per_hour=500,
                requests_per_day=5000,
                burst_size=5,
            ),
            RateLimitTier.BASIC: RateLimitConfig(
                requests_per_minute=60,
                requests_per_hour=1000,
                requests_per_day=10000,
                burst_size=10,
            ),
            RateLimitTier.PROFESSIONAL: RateLimitConfig(
                requests_per_minute=120,
                requests_per_hour=5000,
                requests_per_day=50000,
                burst_size=20,
            ),
            RateLimitTier.ENTERPRISE: RateLimitConfig(
                requests_per_minute=300,
                requests_per_hour=10000,
                requests_per_day=100000,
                burst_size=50,
            ),
            RateLimitTier.UNLIMITED: RateLimitConfig(
                requests_per_minute=10000,
                requests_per_hour=100000,
                requests_per_day=1000000,
                burst_size=1000,
            ),
        }

    def _default_client_id(self, request: Request) -> str:
        """Extract client ID from request"""
        # Try API key first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{hashlib.md5(api_key.encode()).hexdigest()[:16]}"

        # Fall back to IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"

        return f"ip:{ip}"

    def _default_client_tier(self, request: Request) -> RateLimitTier:
        """Determine client tier from request"""
        # Check for API key to determine tier
        api_key = request.headers.get("X-API-Key")
        if api_key:
            # In production, look up tier from database
            # For now, use basic tier for any API key
            return RateLimitTier.BASIC

        return RateLimitTier.FREE

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with rate limiting"""
        # Check if path is excluded
        if any(request.url.path.startswith(p) for p in self.exclude_paths):
            return await call_next(request)

        # Get client info
        client_id = self._get_client_id(request)
        client_tier = self._get_client_tier(request)

        # Get config for this client/endpoint
        config = self._get_config(request, client_tier)

        # Check rate limit
        result = await self._check_rate_limit(client_id, config, request)

        # Record the request
        self.store.record_request(client_id, result.allowed)

        if not result.allowed:
            # Return 429 Too Many Requests
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "message": "Rate limit exceeded",
                    "retry_after": result.retry_after,
                },
                headers=self._rate_limit_headers(result),
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        for key, value in self._rate_limit_headers(result).items():
            response.headers[key] = value

        return response

    def _get_config(
        self, request: Request, tier: RateLimitTier
    ) -> RateLimitConfig:
        """Get rate limit config for request"""
        # Check endpoint-specific limits
        for endpoint_limit in self.endpoint_limits:
            if self._match_endpoint(request, endpoint_limit):
                return RateLimitConfig(
                    requests_per_minute=endpoint_limit.requests_per_minute,
                    requests_per_hour=endpoint_limit.requests_per_hour,
                    burst_size=endpoint_limit.burst_size,
                    algorithm=self.default_config.algorithm,
                )

        # Use tier config
        return self.tier_configs.get(tier, self.default_config)

    def _match_endpoint(
        self, request: Request, endpoint_limit: EndpointRateLimit
    ) -> bool:
        """Check if request matches endpoint limit"""
        # Check method
        if endpoint_limit.method != "*":
            if request.method.upper() != endpoint_limit.method.upper():
                return False

        # Check path pattern (simple glob matching)
        pattern = endpoint_limit.path_pattern
        path = request.url.path

        if "*" in pattern:
            # Simple wildcard matching
            parts = pattern.split("*")
            if len(parts) == 2:
                return path.startswith(parts[0]) and path.endswith(parts[1])
        else:
            return path == pattern

        return False

    async def _check_rate_limit(
        self,
        client_id: str,
        config: RateLimitConfig,
        request: Request,
    ) -> RateLimitResponse:
        """Check if request is within rate limits"""
        if config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return await self._check_token_bucket(client_id, config)
        elif config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return await self._check_sliding_window(client_id, config)
        else:
            # Default to sliding window
            return await self._check_sliding_window(client_id, config)

    async def _check_token_bucket(
        self, client_id: str, config: RateLimitConfig
    ) -> RateLimitResponse:
        """Check rate limit using token bucket"""
        # Refill rate: tokens per second to achieve requests_per_minute
        refill_rate = config.requests_per_minute / 60.0

        bucket = self.store.get_bucket(
            client_id, config.burst_size, refill_rate
        )

        allowed, remaining = bucket.consume(1)

        # Calculate reset time
        if not allowed:
            # Time until 1 token is available
            retry_after = int(1 / refill_rate) + 1
            reset_at = datetime.utcnow() + timedelta(seconds=retry_after)
        else:
            retry_after = None
            reset_at = datetime.utcnow() + timedelta(minutes=1)

        return RateLimitResponse(
            allowed=allowed,
            remaining=int(remaining),
            limit=config.burst_size,
            reset_at=reset_at,
            retry_after=retry_after,
        )

    async def _check_sliding_window(
        self, client_id: str, config: RateLimitConfig
    ) -> RateLimitResponse:
        """Check rate limit using sliding window"""
        # Check minute window
        minute_window = self.store.get_window(
            client_id, "minute", 60, config.requests_per_minute
        )
        allowed, remaining = minute_window.allow()

        if not allowed:
            retry_after = int(minute_window.get_reset_time()) + 1
            return RateLimitResponse(
                allowed=False,
                remaining=0,
                limit=config.requests_per_minute,
                reset_at=datetime.utcnow() + timedelta(seconds=retry_after),
                retry_after=retry_after,
            )

        # Check hour window
        hour_window = self.store.get_window(
            client_id, "hour", 3600, config.requests_per_hour
        )
        hour_allowed, hour_remaining = hour_window.allow()

        if not hour_allowed:
            retry_after = int(hour_window.get_reset_time()) + 1
            return RateLimitResponse(
                allowed=False,
                remaining=0,
                limit=config.requests_per_hour,
                reset_at=datetime.utcnow() + timedelta(seconds=retry_after),
                retry_after=retry_after,
            )

        # All checks passed
        return RateLimitResponse(
            allowed=True,
            remaining=remaining,
            limit=config.requests_per_minute,
            reset_at=datetime.utcnow() + timedelta(minutes=1),
        )

    def _rate_limit_headers(self, result: RateLimitResponse) -> Dict[str, str]:
        """Generate rate limit response headers"""
        headers = {
            "X-RateLimit-Limit": str(result.limit),
            "X-RateLimit-Remaining": str(result.remaining),
            "X-RateLimit-Reset": str(int(result.reset_at.timestamp())),
        }

        if result.retry_after:
            headers["Retry-After"] = str(result.retry_after)

        return headers

    def get_client_stats(self, client_id: str) -> Dict[str, Any]:
        """Get statistics for a client"""
        client = self.store.get_client(client_id)
        return {
            "client_id": client.client_id,
            "tier": client.tier.value,
            "total_requests": client.total_requests,
            "rejected_requests": client.rejected_requests,
            "rejection_rate": (
                client.rejected_requests / client.total_requests
                if client.total_requests > 0
                else 0
            ),
            "last_request": (
                datetime.fromtimestamp(client.last_request).isoformat()
                if client.last_request
                else None
            ),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get overall rate limiter statistics"""
        return self.store.get_stats()


def create_rate_limiter(
    app,
    requests_per_minute: int = 60,
    requests_per_hour: int = 1000,
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET,
    **kwargs,
) -> RateLimiterMiddleware:
    """Create and configure rate limiter middleware"""
    config = RateLimitConfig(
        requests_per_minute=requests_per_minute,
        requests_per_hour=requests_per_hour,
        algorithm=algorithm,
    )

    return RateLimiterMiddleware(app, default_config=config, **kwargs)


# Decorator for per-route rate limiting
class RateLimit:
    """Decorator for per-route rate limiting"""

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10,
    ):
        self.config = RateLimitConfig(
            requests_per_minute=requests_per_minute,
            requests_per_hour=requests_per_hour,
            burst_size=burst_size,
        )
        self._store = RateLimiterStore()

    def __call__(self, func: Callable) -> Callable:
        async def wrapper(request: Request, *args, **kwargs):
            # Get client ID
            client_id = self._get_client_id(request)

            # Check rate limit
            window = self._store.get_window(
                client_id, "minute", 60, self.config.requests_per_minute
            )
            allowed, remaining = window.allow()

            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={
                        "Retry-After": str(int(window.get_reset_time()) + 1),
                        "X-RateLimit-Remaining": "0",
                    },
                )

            return await func(request, *args, **kwargs)

        return wrapper

    def _get_client_id(self, request: Request) -> str:
        """Extract client ID from request"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"


# Example usage function
def setup_rate_limiting(app, config: Dict[str, Any] = None):
    """Setup rate limiting for a FastAPI app"""
    config = config or {}

    # Create default endpoint limits
    endpoint_limits = [
        EndpointRateLimit(
            path_pattern="/api/audit",
            method="POST",
            requests_per_minute=10,
            requests_per_hour=100,
            burst_size=3,
            priority=10,
        ),
        EndpointRateLimit(
            path_pattern="/api/policies/generate",
            method="POST",
            requests_per_minute=5,
            requests_per_hour=50,
            burst_size=2,
            priority=10,
        ),
        EndpointRateLimit(
            path_pattern="/api/*",
            method="*",
            requests_per_minute=60,
            requests_per_hour=1000,
            burst_size=10,
            priority=0,
        ),
    ]

    middleware = RateLimiterMiddleware(
        app,
        default_config=RateLimitConfig(
            requests_per_minute=config.get("requests_per_minute", 60),
            requests_per_hour=config.get("requests_per_hour", 1000),
            algorithm=RateLimitAlgorithm(
                config.get("algorithm", "token_bucket")
            ),
        ),
        endpoint_limits=endpoint_limits,
        exclude_paths=config.get("exclude_paths", ["/health", "/docs"]),
    )

    return middleware

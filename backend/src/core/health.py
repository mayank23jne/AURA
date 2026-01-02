"""Agent Health Scoring System for AURA platform"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from enum import Enum

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class HealthStatus(str, Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class HealthCheckResult(BaseModel):
    """Result of a health check"""
    check_name: str
    passed: bool
    score: float  # 0-100
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentHealthScore(BaseModel):
    """Comprehensive health score for an agent"""
    agent_id: str
    agent_name: str
    overall_score: float  # 0-100
    status: HealthStatus
    checks: List[HealthCheckResult] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    # Component scores
    performance_score: float = 100.0
    reliability_score: float = 100.0
    availability_score: float = 100.0
    resource_score: float = 100.0

    # Trend
    score_trend: str = "stable"  # improving, stable, degrading
    historical_scores: List[float] = Field(default_factory=list)


class HealthThresholds(BaseModel):
    """Thresholds for health scoring"""
    healthy_threshold: float = 80.0
    degraded_threshold: float = 60.0
    unhealthy_threshold: float = 40.0

    # Performance thresholds
    max_response_time_ms: float = 5000.0
    target_response_time_ms: float = 1000.0

    # Reliability thresholds
    min_success_rate: float = 0.95
    max_error_rate: float = 0.05

    # Availability thresholds
    max_heartbeat_delay_seconds: int = 60
    min_uptime_percentage: float = 99.0

    # Resource thresholds
    max_memory_usage_percent: float = 80.0
    max_cpu_usage_percent: float = 80.0
    max_queue_depth: int = 1000


class HealthScorer:
    """
    Agent health scoring system.

    Features:
    - Multi-dimensional health scoring
    - Configurable thresholds
    - Historical trend analysis
    - Automated alerting
    - Customizable health checks
    """

    def __init__(self, thresholds: HealthThresholds = None):
        self.thresholds = thresholds or HealthThresholds()
        self._scores: Dict[str, AgentHealthScore] = {}
        self._check_functions: Dict[str, callable] = {}
        self._alert_callbacks: List[callable] = []

        logger.info("HealthScorer initialized")

    def register_check(self, check_name: str, check_fn: callable):
        """Register a custom health check function"""
        self._check_functions[check_name] = check_fn
        logger.info("Health check registered", check_name=check_name)

    def register_alert_callback(self, callback: callable):
        """Register callback for health alerts"""
        self._alert_callbacks.append(callback)

    async def calculate_score(
        self,
        agent_id: str,
        agent_name: str,
        metrics: Dict[str, Any],
    ) -> AgentHealthScore:
        """Calculate comprehensive health score for an agent"""
        checks = []

        # Performance check
        perf_check = await self._check_performance(metrics)
        checks.append(perf_check)
        performance_score = perf_check.score

        # Reliability check
        rel_check = await self._check_reliability(metrics)
        checks.append(rel_check)
        reliability_score = rel_check.score

        # Availability check
        avail_check = await self._check_availability(metrics)
        checks.append(avail_check)
        availability_score = avail_check.score

        # Resource check
        res_check = await self._check_resources(metrics)
        checks.append(res_check)
        resource_score = res_check.score

        # Run custom checks
        for check_name, check_fn in self._check_functions.items():
            try:
                result = await check_fn(metrics) if asyncio.iscoroutinefunction(check_fn) else check_fn(metrics)
                checks.append(result)
            except Exception as e:
                checks.append(HealthCheckResult(
                    check_name=check_name,
                    passed=False,
                    score=0,
                    message=f"Check failed: {str(e)}",
                ))

        # Calculate overall score (weighted average)
        weights = {
            "performance": 0.3,
            "reliability": 0.3,
            "availability": 0.25,
            "resources": 0.15,
        }

        overall_score = (
            performance_score * weights["performance"]
            + reliability_score * weights["reliability"]
            + availability_score * weights["availability"]
            + resource_score * weights["resources"]
        )

        # Determine status
        status = self._determine_status(overall_score)

        # Get previous score for trend
        previous = self._scores.get(agent_id)
        historical = previous.historical_scores if previous else []
        historical.append(overall_score)
        if len(historical) > 100:
            historical = historical[-100:]

        # Calculate trend
        trend = self._calculate_trend(historical)

        # Create health score
        health_score = AgentHealthScore(
            agent_id=agent_id,
            agent_name=agent_name,
            overall_score=round(overall_score, 2),
            status=status,
            checks=checks,
            performance_score=round(performance_score, 2),
            reliability_score=round(reliability_score, 2),
            availability_score=round(availability_score, 2),
            resource_score=round(resource_score, 2),
            score_trend=trend,
            historical_scores=historical,
        )

        # Store score
        self._scores[agent_id] = health_score

        # Check for alerts
        await self._check_alerts(health_score, previous)

        logger.debug(
            "Health score calculated",
            agent_id=agent_id,
            score=overall_score,
            status=status.value,
        )

        return health_score

    async def _check_performance(self, metrics: Dict[str, Any]) -> HealthCheckResult:
        """Check agent performance"""
        avg_response_time = metrics.get("avg_response_time_ms", 0)

        if avg_response_time <= self.thresholds.target_response_time_ms:
            score = 100.0
        elif avg_response_time <= self.thresholds.max_response_time_ms:
            # Linear interpolation between target and max
            ratio = (avg_response_time - self.thresholds.target_response_time_ms) / (
                self.thresholds.max_response_time_ms - self.thresholds.target_response_time_ms
            )
            score = 100.0 - (ratio * 60)  # Max penalty is 60 points
        else:
            score = max(0, 40 - (avg_response_time - self.thresholds.max_response_time_ms) / 100)

        return HealthCheckResult(
            check_name="performance",
            passed=score >= 60,
            score=max(0, min(100, score)),
            message=f"Avg response time: {avg_response_time:.2f}ms",
            metadata={"avg_response_time_ms": avg_response_time},
        )

    async def _check_reliability(self, metrics: Dict[str, Any]) -> HealthCheckResult:
        """Check agent reliability"""
        tasks_completed = metrics.get("tasks_completed", 0)
        tasks_failed = metrics.get("tasks_failed", 0)
        total_tasks = tasks_completed + tasks_failed

        if total_tasks == 0:
            success_rate = 1.0
        else:
            success_rate = tasks_completed / total_tasks

        if success_rate >= self.thresholds.min_success_rate:
            score = 100.0
        else:
            # Score drops as success rate decreases
            score = (success_rate / self.thresholds.min_success_rate) * 100

        return HealthCheckResult(
            check_name="reliability",
            passed=success_rate >= self.thresholds.min_success_rate,
            score=max(0, min(100, score)),
            message=f"Success rate: {success_rate:.2%}",
            metadata={
                "success_rate": success_rate,
                "tasks_completed": tasks_completed,
                "tasks_failed": tasks_failed,
            },
        )

    async def _check_availability(self, metrics: Dict[str, Any]) -> HealthCheckResult:
        """Check agent availability"""
        last_heartbeat = metrics.get("last_heartbeat")
        uptime_seconds = metrics.get("uptime_seconds", 0)

        score = 100.0
        issues = []

        # Check heartbeat
        if last_heartbeat:
            if isinstance(last_heartbeat, str):
                last_heartbeat = datetime.fromisoformat(last_heartbeat)
            delay = (datetime.utcnow() - last_heartbeat).total_seconds()

            if delay > self.thresholds.max_heartbeat_delay_seconds:
                score -= 50
                issues.append(f"Heartbeat delay: {delay:.0f}s")

        # Check uptime
        # Assuming target uptime, calculate percentage (simplified)
        expected_uptime = 3600  # 1 hour baseline
        if uptime_seconds > 0 and uptime_seconds < expected_uptime:
            uptime_percentage = (uptime_seconds / expected_uptime) * 100
            if uptime_percentage < self.thresholds.min_uptime_percentage:
                score -= 30
                issues.append(f"Low uptime: {uptime_percentage:.1f}%")

        return HealthCheckResult(
            check_name="availability",
            passed=score >= 60,
            score=max(0, min(100, score)),
            message="; ".join(issues) if issues else "Available",
            metadata={"uptime_seconds": uptime_seconds},
        )

    async def _check_resources(self, metrics: Dict[str, Any]) -> HealthCheckResult:
        """Check resource utilization"""
        memory_usage = metrics.get("memory_usage_mb", 0)
        queue_depth = metrics.get("queue_depth", 0)

        score = 100.0
        issues = []

        # Check queue depth
        if queue_depth > self.thresholds.max_queue_depth:
            ratio = queue_depth / self.thresholds.max_queue_depth
            penalty = min(50, (ratio - 1) * 25)
            score -= penalty
            issues.append(f"High queue depth: {queue_depth}")

        # Memory check would need actual memory limit info
        # Simplified version
        if memory_usage > 1000:  # Over 1GB
            score -= 20
            issues.append(f"High memory: {memory_usage:.0f}MB")

        return HealthCheckResult(
            check_name="resources",
            passed=score >= 60,
            score=max(0, min(100, score)),
            message="; ".join(issues) if issues else "Resources OK",
            metadata={
                "memory_usage_mb": memory_usage,
                "queue_depth": queue_depth,
            },
        )

    def _determine_status(self, score: float) -> HealthStatus:
        """Determine health status from score"""
        if score >= self.thresholds.healthy_threshold:
            return HealthStatus.HEALTHY
        elif score >= self.thresholds.degraded_threshold:
            return HealthStatus.DEGRADED
        elif score >= self.thresholds.unhealthy_threshold:
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.CRITICAL

    def _calculate_trend(self, historical: List[float]) -> str:
        """Calculate score trend from historical data"""
        if len(historical) < 3:
            return "stable"

        recent = historical[-5:]
        avg_recent = sum(recent) / len(recent)
        avg_older = sum(historical[:-5]) / max(1, len(historical) - 5)

        diff = avg_recent - avg_older

        if diff > 5:
            return "improving"
        elif diff < -5:
            return "degrading"
        else:
            return "stable"

    async def _check_alerts(
        self,
        current: AgentHealthScore,
        previous: Optional[AgentHealthScore],
    ):
        """Check if alerts should be triggered"""
        # Alert on status change to critical
        if current.status == HealthStatus.CRITICAL:
            if not previous or previous.status != HealthStatus.CRITICAL:
                await self._send_alert(
                    "critical",
                    f"Agent {current.agent_name} is in critical state",
                    current,
                )

        # Alert on significant score drop
        if previous and previous.overall_score - current.overall_score > 20:
            await self._send_alert(
                "warning",
                f"Agent {current.agent_name} score dropped by {previous.overall_score - current.overall_score:.1f}",
                current,
            )

    async def _send_alert(
        self,
        level: str,
        message: str,
        health_score: AgentHealthScore,
    ):
        """Send alert to registered callbacks"""
        logger.warning(
            "Health alert",
            level=level,
            message=message,
            agent_id=health_score.agent_id,
            score=health_score.overall_score,
        )

        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(level, message, health_score)
                else:
                    callback(level, message, health_score)
            except Exception as e:
                logger.error("Alert callback failed", error=str(e))

    def get_score(self, agent_id: str) -> Optional[AgentHealthScore]:
        """Get health score for an agent"""
        return self._scores.get(agent_id)

    def get_all_scores(self) -> Dict[str, AgentHealthScore]:
        """Get all health scores"""
        return self._scores.copy()

    def get_unhealthy_agents(self) -> List[AgentHealthScore]:
        """Get list of unhealthy agents"""
        return [
            score for score in self._scores.values()
            if score.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all agent health"""
        status_counts = {}
        scores = []

        for health_score in self._scores.values():
            status_counts[health_score.status.value] = status_counts.get(
                health_score.status.value, 0
            ) + 1
            scores.append(health_score.overall_score)

        return {
            "total_agents": len(self._scores),
            "average_score": sum(scores) / len(scores) if scores else 0,
            "by_status": status_counts,
            "unhealthy_count": len(self.get_unhealthy_agents()),
        }

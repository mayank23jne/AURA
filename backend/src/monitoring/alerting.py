"""Real-time Monitoring and Alerting System for AURA Platform"""

import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import uuid
import re

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class AlertChannel(str, Enum):
    """Alert notification channels"""
    LOG = "log"
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    SMS = "sms"


class MetricType(str, Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class Metric(BaseModel):
    """A metric data point"""
    name: str
    value: float
    metric_type: MetricType = MetricType.GAUGE
    tags: Dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    unit: str = ""


class AlertRule(BaseModel):
    """Definition of an alert rule"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    enabled: bool = True

    # Condition
    metric_name: str
    condition: str  # e.g., ">", "<", ">=", "<=", "==", "!="
    threshold: float
    tags_filter: Dict[str, str] = Field(default_factory=dict)

    # Aggregation
    aggregation: str = "avg"  # avg, sum, min, max, count
    window_seconds: int = 60

    # Severity and escalation
    severity: AlertSeverity = AlertSeverity.WARNING
    escalation_threshold: int = 3  # Consecutive violations before alert

    # Notification
    channels: List[AlertChannel] = Field(default_factory=lambda: [AlertChannel.LOG])
    recipients: List[str] = Field(default_factory=list)

    # Cooldown
    cooldown_seconds: int = 300  # Min time between alerts

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


class Alert(BaseModel):
    """An alert instance"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    rule_id: str
    rule_name: str
    severity: AlertSeverity

    # Status
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: datetime = Field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None

    # Details
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    tags: Dict[str, str] = Field(default_factory=dict)

    # History
    occurrences: int = 1
    last_occurrence: datetime = Field(default_factory=datetime.utcnow)

    # Notes
    notes: List[str] = Field(default_factory=list)


class ChannelConfig(BaseModel):
    """Configuration for an alert channel"""
    channel: AlertChannel
    enabled: bool = True
    config: Dict[str, Any] = Field(default_factory=dict)


class AlertingConfig(BaseModel):
    """Configuration for alerting system"""
    evaluation_interval_seconds: int = 30
    retention_days: int = 30
    max_alerts_per_rule: int = 100
    dedup_window_seconds: int = 300
    channels: List[ChannelConfig] = Field(default_factory=list)


class MetricStore:
    """In-memory metric storage with windowing"""

    def __init__(self, retention_seconds: int = 3600):
        self._metrics: Dict[str, List[Metric]] = {}
        self._retention = retention_seconds

    def record(self, metric: Metric):
        """Record a metric value"""
        key = self._metric_key(metric.name, metric.tags)
        if key not in self._metrics:
            self._metrics[key] = []
        self._metrics[key].append(metric)
        self._cleanup(key)

    def query(
        self,
        name: str,
        tags: Dict[str, str] = None,
        window_seconds: int = 60,
    ) -> List[Metric]:
        """Query metrics within a time window"""
        results = []
        cutoff = datetime.utcnow() - timedelta(seconds=window_seconds)

        for key, metrics in self._metrics.items():
            # Check name match
            if not key.startswith(f"{name}:"):
                continue

            # Check tags match
            if tags:
                metric_tags = self._parse_key_tags(key)
                if not all(metric_tags.get(k) == v for k, v in tags.items()):
                    continue

            # Filter by time
            for metric in metrics:
                if metric.timestamp >= cutoff:
                    results.append(metric)

        return results

    def aggregate(
        self,
        name: str,
        aggregation: str,
        tags: Dict[str, str] = None,
        window_seconds: int = 60,
    ) -> Optional[float]:
        """Aggregate metrics"""
        metrics = self.query(name, tags, window_seconds)
        if not metrics:
            return None

        values = [m.value for m in metrics]

        if aggregation == "avg":
            return sum(values) / len(values)
        elif aggregation == "sum":
            return sum(values)
        elif aggregation == "min":
            return min(values)
        elif aggregation == "max":
            return max(values)
        elif aggregation == "count":
            return float(len(values))
        else:
            return sum(values) / len(values)

    def _metric_key(self, name: str, tags: Dict[str, str]) -> str:
        """Create key for metric"""
        sorted_tags = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}:{sorted_tags}"

    def _parse_key_tags(self, key: str) -> Dict[str, str]:
        """Parse tags from key"""
        parts = key.split(":", 1)
        if len(parts) < 2 or not parts[1]:
            return {}
        tags = {}
        for pair in parts[1].split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                tags[k] = v
        return tags

    def _cleanup(self, key: str):
        """Remove old metrics"""
        cutoff = datetime.utcnow() - timedelta(seconds=self._retention)
        self._metrics[key] = [
            m for m in self._metrics[key]
            if m.timestamp >= cutoff
        ]


class AlertManager:
    """
    Real-time monitoring and alerting system.

    Features:
    - Flexible alert rules with conditions
    - Multiple notification channels
    - Alert deduplication and grouping
    - Escalation policies
    - Alert lifecycle management
    - Metric aggregation
    """

    def __init__(self, config: AlertingConfig = None):
        self.config = config or AlertingConfig()
        self._rules: Dict[str, AlertRule] = {}
        self._alerts: Dict[str, Alert] = {}
        self._metric_store = MetricStore()
        self._channel_handlers: Dict[AlertChannel, Callable] = {}
        self._violation_counts: Dict[str, int] = {}
        self._running = False
        self._eval_task: Optional[asyncio.Task] = None

        # Register default channel handlers
        self._channel_handlers[AlertChannel.LOG] = self._log_alert

        logger.info("AlertManager initialized")

    async def start(self):
        """Start the alert manager"""
        self._running = True
        self._eval_task = asyncio.create_task(self._evaluation_loop())
        logger.info("Alert manager started")

    async def stop(self):
        """Stop the alert manager"""
        self._running = False
        if self._eval_task:
            self._eval_task.cancel()
            try:
                await self._eval_task
            except asyncio.CancelledError:
                pass
        logger.info("Alert manager stopped")

    def add_rule(self, rule: AlertRule) -> str:
        """Add an alert rule"""
        self._rules[rule.id] = rule
        logger.info("Alert rule added", rule_id=rule.id, name=rule.name)
        return rule.id

    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update an alert rule"""
        if rule_id not in self._rules:
            return False

        rule = self._rules[rule_id]
        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)

        return True

    def remove_rule(self, rule_id: str) -> bool:
        """Remove an alert rule"""
        if rule_id in self._rules:
            del self._rules[rule_id]
            return True
        return False

    def register_channel_handler(self, channel: AlertChannel, handler: Callable):
        """Register a handler for a notification channel"""
        self._channel_handlers[channel] = handler
        logger.info("Channel handler registered", channel=channel.value)

    def record_metric(
        self,
        name: str,
        value: float,
        tags: Dict[str, str] = None,
        metric_type: MetricType = MetricType.GAUGE,
        unit: str = "",
    ):
        """Record a metric value"""
        metric = Metric(
            name=name,
            value=value,
            tags=tags or {},
            metric_type=metric_type,
            unit=unit,
        )
        self._metric_store.record(metric)

    async def evaluate_rules(self):
        """Evaluate all alert rules"""
        for rule in self._rules.values():
            if not rule.enabled:
                continue

            try:
                await self._evaluate_rule(rule)
            except Exception as e:
                logger.error(
                    "Error evaluating rule",
                    rule_id=rule.id,
                    error=str(e),
                )

    async def _evaluate_rule(self, rule: AlertRule):
        """Evaluate a single alert rule"""
        # Get aggregated metric value
        value = self._metric_store.aggregate(
            rule.metric_name,
            rule.aggregation,
            rule.tags_filter,
            rule.window_seconds,
        )

        if value is None:
            # No data, skip evaluation
            return

        # Check condition
        violated = self._check_condition(value, rule.condition, rule.threshold)

        if violated:
            # Increment violation count
            self._violation_counts[rule.id] = self._violation_counts.get(rule.id, 0) + 1

            # Check if threshold reached
            if self._violation_counts[rule.id] >= rule.escalation_threshold:
                # Check cooldown
                if rule.last_triggered:
                    since_last = (datetime.utcnow() - rule.last_triggered).total_seconds()
                    if since_last < rule.cooldown_seconds:
                        return

                # Create or update alert
                await self._create_alert(rule, value)
                rule.last_triggered = datetime.utcnow()
                rule.trigger_count += 1
                self._violation_counts[rule.id] = 0
        else:
            # Reset violation count
            self._violation_counts[rule.id] = 0

            # Auto-resolve related alerts
            await self._auto_resolve(rule.id)

    def _check_condition(
        self, value: float, condition: str, threshold: float
    ) -> bool:
        """Check if condition is met"""
        if condition == ">":
            return value > threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<":
            return value < threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return value == threshold
        elif condition == "!=":
            return value != threshold
        return False

    async def _create_alert(self, rule: AlertRule, value: float):
        """Create or update an alert"""
        # Check for existing active alert
        existing = self._find_existing_alert(rule.id)

        if existing:
            # Update existing alert
            existing.occurrences += 1
            existing.last_occurrence = datetime.utcnow()
            existing.metric_value = value
            logger.debug(
                "Alert updated",
                alert_id=existing.id,
                occurrences=existing.occurrences,
            )
        else:
            # Create new alert
            message = (
                f"{rule.name}: {rule.metric_name} {rule.condition} {rule.threshold} "
                f"(current: {value:.2f})"
            )

            alert = Alert(
                rule_id=rule.id,
                rule_name=rule.name,
                severity=rule.severity,
                message=message,
                metric_name=rule.metric_name,
                metric_value=value,
                threshold=rule.threshold,
                tags=rule.tags_filter,
            )

            self._alerts[alert.id] = alert

            # Send notifications
            await self._send_notifications(alert, rule)

            logger.warning(
                "Alert created",
                alert_id=alert.id,
                severity=alert.severity.value,
                message=message,
            )

    def _find_existing_alert(self, rule_id: str) -> Optional[Alert]:
        """Find existing active alert for rule"""
        for alert in self._alerts.values():
            if alert.rule_id == rule_id and alert.status == AlertStatus.ACTIVE:
                return alert
        return None

    async def _auto_resolve(self, rule_id: str):
        """Auto-resolve alerts for a rule"""
        for alert in self._alerts.values():
            if alert.rule_id == rule_id and alert.status == AlertStatus.ACTIVE:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.utcnow()
                logger.info(
                    "Alert auto-resolved",
                    alert_id=alert.id,
                    rule_id=rule_id,
                )

    async def _send_notifications(self, alert: Alert, rule: AlertRule):
        """Send alert notifications to configured channels"""
        for channel in rule.channels:
            handler = self._channel_handlers.get(channel)
            if handler:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(alert, rule)
                    else:
                        handler(alert, rule)
                except Exception as e:
                    logger.error(
                        "Failed to send notification",
                        channel=channel.value,
                        error=str(e),
                    )

    def _log_alert(self, alert: Alert, rule: AlertRule):
        """Log alert (default handler)"""
        log_func = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.ERROR: logger.error,
            AlertSeverity.CRITICAL: logger.critical,
        }.get(alert.severity, logger.warning)

        log_func(
            "ALERT",
            alert_id=alert.id,
            severity=alert.severity.value,
            message=alert.message,
        )

    def acknowledge_alert(
        self, alert_id: str, acknowledged_by: str = ""
    ) -> bool:
        """Acknowledge an alert"""
        if alert_id not in self._alerts:
            return False

        alert = self._alerts[alert_id]
        if alert.status != AlertStatus.ACTIVE:
            return False

        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.utcnow()
        alert.acknowledged_by = acknowledged_by

        logger.info("Alert acknowledged", alert_id=alert_id, by=acknowledged_by)
        return True

    def resolve_alert(self, alert_id: str, note: str = "") -> bool:
        """Resolve an alert"""
        if alert_id not in self._alerts:
            return False

        alert = self._alerts[alert_id]
        if alert.status == AlertStatus.RESOLVED:
            return False

        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow()
        if note:
            alert.notes.append(f"[{datetime.utcnow().isoformat()}] {note}")

        logger.info("Alert resolved", alert_id=alert_id)
        return True

    def add_note(self, alert_id: str, note: str) -> bool:
        """Add a note to an alert"""
        if alert_id not in self._alerts:
            return False

        alert = self._alerts[alert_id]
        alert.notes.append(f"[{datetime.utcnow().isoformat()}] {note}")
        return True

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get an alert by ID"""
        return self._alerts.get(alert_id)

    def list_alerts(
        self,
        status: Optional[AlertStatus] = None,
        severity: Optional[AlertSeverity] = None,
        rule_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """List alerts with optional filters"""
        alerts = list(self._alerts.values())

        if status:
            alerts = [a for a in alerts if a.status == status]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if rule_id:
            alerts = [a for a in alerts if a.rule_id == rule_id]

        # Sort by created_at descending
        alerts.sort(key=lambda a: a.created_at, reverse=True)
        return alerts[:limit]

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return self.list_alerts(status=AlertStatus.ACTIVE)

    def get_alert_count_by_severity(self) -> Dict[str, int]:
        """Get count of active alerts by severity"""
        counts = {}
        for alert in self._alerts.values():
            if alert.status == AlertStatus.ACTIVE:
                key = alert.severity.value
                counts[key] = counts.get(key, 0) + 1
        return counts

    async def _evaluation_loop(self):
        """Background loop to evaluate rules"""
        while self._running:
            try:
                await self.evaluate_rules()
                await asyncio.sleep(self.config.evaluation_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in evaluation loop", error=str(e))
                await asyncio.sleep(5)

    def get_rules(self) -> List[AlertRule]:
        """Get all alert rules"""
        return list(self._rules.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get alerting statistics"""
        active_count = sum(
            1 for a in self._alerts.values()
            if a.status == AlertStatus.ACTIVE
        )
        severity_counts = self.get_alert_count_by_severity()

        return {
            "total_rules": len(self._rules),
            "enabled_rules": sum(1 for r in self._rules.values() if r.enabled),
            "total_alerts": len(self._alerts),
            "active_alerts": active_count,
            "by_severity": severity_counts,
        }


# Predefined alert rules for AI compliance
def create_compliance_alert_rules() -> List[AlertRule]:
    """Create predefined alert rules for AI compliance monitoring"""
    return [
        AlertRule(
            name="High Model Error Rate",
            description="Alert when model error rate exceeds threshold",
            metric_name="model.error_rate",
            condition=">",
            threshold=0.05,
            severity=AlertSeverity.ERROR,
            channels=[AlertChannel.LOG],
            escalation_threshold=3,
        ),
        AlertRule(
            name="Low Model Confidence",
            description="Alert when average model confidence drops",
            metric_name="model.confidence",
            condition="<",
            threshold=0.7,
            aggregation="avg",
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG],
        ),
        AlertRule(
            name="High Latency",
            description="Alert when response latency is too high",
            metric_name="model.latency_ms",
            condition=">",
            threshold=5000,
            aggregation="avg",
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG],
        ),
        AlertRule(
            name="Bias Detected",
            description="Alert when bias score exceeds threshold",
            metric_name="model.bias_score",
            condition=">",
            threshold=0.3,
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.LOG],
            cooldown_seconds=600,
        ),
        AlertRule(
            name="Data Drift",
            description="Alert when data drift is detected",
            metric_name="model.drift_score",
            condition=">",
            threshold=0.5,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG],
        ),
        AlertRule(
            name="Compliance Score Drop",
            description="Alert when compliance score drops significantly",
            metric_name="compliance.score",
            condition="<",
            threshold=70,
            severity=AlertSeverity.ERROR,
            channels=[AlertChannel.LOG],
        ),
        AlertRule(
            name="Audit Failure",
            description="Alert on audit failures",
            metric_name="audit.failures",
            condition=">",
            threshold=0,
            aggregation="count",
            severity=AlertSeverity.ERROR,
            channels=[AlertChannel.LOG],
        ),
        AlertRule(
            name="High Queue Depth",
            description="Alert when message queue depth is too high",
            metric_name="queue.depth",
            condition=">",
            threshold=1000,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG],
        ),
    ]


# Global alert manager
alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get the global alert manager"""
    global alert_manager
    if alert_manager is None:
        alert_manager = AlertManager()
    return alert_manager

"""Risk-Based Audit Scheduling for AURA Platform"""

import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import uuid
import math

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class RiskLevel(str, Enum):
    """Risk level classifications"""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCategory(str, Enum):
    """Categories of risk factors"""
    COMPLIANCE = "compliance"
    PERFORMANCE = "performance"
    SECURITY = "security"
    OPERATIONAL = "operational"
    REGULATORY = "regulatory"
    DATA_QUALITY = "data_quality"


class RiskFactor(BaseModel):
    """A single risk factor contributing to overall risk"""
    name: str
    category: RiskCategory
    weight: float = 1.0
    score: float = 0.0  # 0-100
    description: str = ""
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelRiskProfile(BaseModel):
    """Risk profile for a model"""
    model_id: str
    model_name: str = ""
    risk_factors: List[RiskFactor] = Field(default_factory=list)
    overall_risk_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.MINIMAL
    last_audit: Optional[datetime] = None
    audit_count: int = 0
    failure_count: int = 0
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    # Historical data
    risk_history: List[Dict[str, Any]] = Field(default_factory=list)

    # Scheduling info
    scheduled_audit: Optional[datetime] = None
    audit_interval_hours: int = 24

    # Model metadata
    model_type: str = "unknown"
    deployment_date: Optional[datetime] = None
    criticality: str = "medium"  # low, medium, high, critical
    regulatory_scope: List[str] = Field(default_factory=list)


class RiskTrigger(BaseModel):
    """Trigger condition for risk-based scheduling"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    enabled: bool = True

    # Trigger conditions
    risk_threshold: float = 70.0  # Trigger when risk exceeds this
    risk_categories: List[RiskCategory] = Field(default_factory=list)  # Empty = all
    risk_change_threshold: float = 20.0  # Trigger on sudden risk increase

    # Time-based conditions
    max_days_without_audit: int = 30
    min_audit_interval_hours: int = 24

    # Actions
    priority_override: Optional[str] = None  # critical, high, medium, low
    notify_stakeholders: bool = True

    # Statistics
    trigger_count: int = 0
    last_triggered: Optional[datetime] = None


class RiskAssessmentResult(BaseModel):
    """Result of a risk assessment"""
    model_id: str
    assessment_time: datetime = Field(default_factory=datetime.utcnow)
    overall_risk_score: float
    risk_level: RiskLevel
    risk_factors: List[RiskFactor]
    triggered_conditions: List[str] = Field(default_factory=list)
    recommended_action: str = ""
    next_assessment: datetime = Field(default_factory=lambda: datetime.utcnow() + timedelta(hours=1))


class RiskSchedulerConfig(BaseModel):
    """Configuration for risk-based scheduler"""
    assessment_interval_seconds: int = 300  # 5 minutes
    risk_history_days: int = 90
    max_concurrent_audits: int = 5

    # Risk calculation weights
    time_since_audit_weight: float = 0.2
    failure_rate_weight: float = 0.25
    compliance_weight: float = 0.3
    performance_weight: float = 0.15
    regulatory_weight: float = 0.1

    # Thresholds for risk levels
    minimal_threshold: float = 20.0
    low_threshold: float = 40.0
    medium_threshold: float = 60.0
    high_threshold: float = 80.0

    # Scheduling parameters
    base_audit_interval_hours: int = 24
    min_audit_interval_hours: int = 1
    max_audit_interval_hours: int = 168  # 1 week


class RiskBasedScheduler:
    """
    Risk-based audit scheduling system.

    Features:
    - Multi-factor risk scoring
    - Dynamic risk assessment
    - Configurable triggers
    - Priority-based scheduling
    - Risk trend analysis
    - Regulatory compliance tracking
    """

    def __init__(self, config: RiskSchedulerConfig = None):
        self.config = config or RiskSchedulerConfig()
        self._profiles: Dict[str, ModelRiskProfile] = {}
        self._triggers: Dict[str, RiskTrigger] = {}
        self._assessment_callbacks: List[Callable] = []
        self._audit_callbacks: List[Callable] = []
        self._running = False
        self._assessment_task: Optional[asyncio.Task] = None

        # Default triggers
        self._initialize_default_triggers()

        logger.info("RiskBasedScheduler initialized", config=self.config.model_dump())

    def _initialize_default_triggers(self):
        """Initialize default risk triggers"""
        # Critical risk trigger
        self.add_trigger(RiskTrigger(
            name="Critical Risk",
            description="Trigger audit when risk reaches critical level",
            risk_threshold=80.0,
            priority_override="critical",
            notify_stakeholders=True,
        ))

        # High risk trigger
        self.add_trigger(RiskTrigger(
            name="High Risk",
            description="Trigger audit when risk reaches high level",
            risk_threshold=60.0,
            priority_override="high",
        ))

        # Stale audit trigger
        self.add_trigger(RiskTrigger(
            name="Stale Audit",
            description="Trigger when no audit in 30 days",
            risk_threshold=0.0,  # Any risk level
            max_days_without_audit=30,
        ))

        # Sudden risk increase trigger
        self.add_trigger(RiskTrigger(
            name="Risk Spike",
            description="Trigger on sudden risk increase",
            risk_change_threshold=15.0,
            priority_override="high",
            notify_stakeholders=True,
        ))

        # Compliance risk trigger
        self.add_trigger(RiskTrigger(
            name="Compliance Risk",
            description="Trigger on high compliance risk",
            risk_threshold=50.0,
            risk_categories=[RiskCategory.COMPLIANCE, RiskCategory.REGULATORY],
            priority_override="critical",
        ))

    async def start(self):
        """Start the risk-based scheduler"""
        self._running = True
        self._assessment_task = asyncio.create_task(self._assessment_loop())
        logger.info("Risk-based scheduler started")

    async def stop(self):
        """Stop the risk-based scheduler"""
        self._running = False
        if self._assessment_task:
            self._assessment_task.cancel()
            try:
                await self._assessment_task
            except asyncio.CancelledError:
                pass
        logger.info("Risk-based scheduler stopped")

    def register_model(self, model_id: str, **kwargs) -> ModelRiskProfile:
        """Register a model for risk-based scheduling"""
        profile = ModelRiskProfile(model_id=model_id, **kwargs)
        self._profiles[model_id] = profile

        logger.info(
            "Model registered for risk scheduling",
            model_id=model_id,
            model_name=profile.model_name,
        )

        return profile

    def update_model_profile(self, model_id: str, updates: Dict[str, Any]) -> bool:
        """Update a model's risk profile"""
        if model_id not in self._profiles:
            return False

        profile = self._profiles[model_id]
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)

        profile.last_updated = datetime.utcnow()
        return True

    def add_trigger(self, trigger: RiskTrigger) -> str:
        """Add a risk trigger"""
        self._triggers[trigger.id] = trigger
        logger.info("Risk trigger added", trigger_id=trigger.id, name=trigger.name)
        return trigger.id

    def remove_trigger(self, trigger_id: str) -> bool:
        """Remove a risk trigger"""
        if trigger_id in self._triggers:
            del self._triggers[trigger_id]
            return True
        return False

    def register_assessment_callback(self, callback: Callable):
        """Register callback for risk assessments"""
        self._assessment_callbacks.append(callback)

    def register_audit_callback(self, callback: Callable):
        """Register callback for triggered audits"""
        self._audit_callbacks.append(callback)

    async def assess_risk(self, model_id: str) -> RiskAssessmentResult:
        """Perform a risk assessment for a model"""
        if model_id not in self._profiles:
            raise ValueError(f"Model not registered: {model_id}")

        profile = self._profiles[model_id]

        # Calculate risk factors
        risk_factors = await self._calculate_risk_factors(profile)
        profile.risk_factors = risk_factors

        # Calculate overall risk score
        overall_score = self._calculate_overall_risk(risk_factors)

        # Store previous score for change detection
        previous_score = profile.overall_risk_score

        # Update profile
        profile.overall_risk_score = overall_score
        profile.risk_level = self._determine_risk_level(overall_score)
        profile.last_updated = datetime.utcnow()

        # Add to history
        profile.risk_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "score": overall_score,
            "level": profile.risk_level.value,
        })

        # Trim history
        cutoff = datetime.utcnow() - timedelta(days=self.config.risk_history_days)
        profile.risk_history = [
            h for h in profile.risk_history
            if datetime.fromisoformat(h["timestamp"]) > cutoff
        ]

        # Check triggers
        triggered = await self._check_triggers(profile, previous_score)

        # Calculate recommended action
        action = self._recommend_action(profile, triggered)

        # Calculate next audit interval
        profile.audit_interval_hours = self._calculate_audit_interval(overall_score)

        result = RiskAssessmentResult(
            model_id=model_id,
            overall_risk_score=round(overall_score, 2),
            risk_level=profile.risk_level,
            risk_factors=risk_factors,
            triggered_conditions=[t.name for t in triggered],
            recommended_action=action,
        )

        # Notify callbacks
        await self._notify_assessment(result)

        logger.debug(
            "Risk assessment completed",
            model_id=model_id,
            risk_score=overall_score,
            risk_level=profile.risk_level.value,
            triggers=len(triggered),
        )

        return result

    async def _calculate_risk_factors(self, profile: ModelRiskProfile) -> List[RiskFactor]:
        """Calculate individual risk factors"""
        factors = []

        # Time since last audit factor
        time_factor = self._calc_time_since_audit_factor(profile)
        factors.append(time_factor)

        # Failure rate factor
        failure_factor = self._calc_failure_rate_factor(profile)
        factors.append(failure_factor)

        # Compliance factor
        compliance_factor = self._calc_compliance_factor(profile)
        factors.append(compliance_factor)

        # Performance factor
        performance_factor = self._calc_performance_factor(profile)
        factors.append(performance_factor)

        # Regulatory factor
        regulatory_factor = self._calc_regulatory_factor(profile)
        factors.append(regulatory_factor)

        # Add any existing custom factors
        for existing in profile.risk_factors:
            if existing.name not in [f.name for f in factors]:
                factors.append(existing)

        return factors

    def _calc_time_since_audit_factor(self, profile: ModelRiskProfile) -> RiskFactor:
        """Calculate risk factor based on time since last audit"""
        if profile.last_audit:
            days_since = (datetime.utcnow() - profile.last_audit).days
            # Score increases with time, maxing out at 30 days
            score = min(100, (days_since / 30) * 100)
        else:
            score = 100  # Never audited

        return RiskFactor(
            name="time_since_audit",
            category=RiskCategory.OPERATIONAL,
            weight=self.config.time_since_audit_weight,
            score=score,
            description=f"Days since last audit: {days_since if profile.last_audit else 'Never'}",
        )

    def _calc_failure_rate_factor(self, profile: ModelRiskProfile) -> RiskFactor:
        """Calculate risk factor based on audit failure rate"""
        if profile.audit_count > 0:
            failure_rate = profile.failure_count / profile.audit_count
            score = failure_rate * 100
        else:
            score = 50  # Unknown, moderate risk

        return RiskFactor(
            name="failure_rate",
            category=RiskCategory.COMPLIANCE,
            weight=self.config.failure_rate_weight,
            score=score,
            description=f"Failure rate: {score:.1f}%",
            metadata={
                "audit_count": profile.audit_count,
                "failure_count": profile.failure_count,
            },
        )

    def _calc_compliance_factor(self, profile: ModelRiskProfile) -> RiskFactor:
        """Calculate compliance risk factor"""
        # Base score on criticality
        criticality_scores = {
            "low": 25,
            "medium": 50,
            "high": 75,
            "critical": 100,
        }
        base_score = criticality_scores.get(profile.criticality, 50)

        # Adjust based on regulatory scope
        regulatory_adjustment = min(len(profile.regulatory_scope) * 10, 30)

        score = min(100, base_score + regulatory_adjustment)

        return RiskFactor(
            name="compliance_risk",
            category=RiskCategory.COMPLIANCE,
            weight=self.config.compliance_weight,
            score=score,
            description=f"Compliance risk based on criticality and regulations",
            metadata={
                "criticality": profile.criticality,
                "regulatory_scope": profile.regulatory_scope,
            },
        )

    def _calc_performance_factor(self, profile: ModelRiskProfile) -> RiskFactor:
        """Calculate performance risk factor"""
        # This would typically integrate with actual performance metrics
        # For now, use a baseline score
        score = 30  # Default low-medium risk

        return RiskFactor(
            name="performance_risk",
            category=RiskCategory.PERFORMANCE,
            weight=self.config.performance_weight,
            score=score,
            description="Performance-based risk assessment",
        )

    def _calc_regulatory_factor(self, profile: ModelRiskProfile) -> RiskFactor:
        """Calculate regulatory risk factor"""
        # Higher risk for models under more regulations
        reg_count = len(profile.regulatory_scope)

        if reg_count == 0:
            score = 10
        elif reg_count <= 2:
            score = 40
        elif reg_count <= 5:
            score = 70
        else:
            score = 100

        return RiskFactor(
            name="regulatory_risk",
            category=RiskCategory.REGULATORY,
            weight=self.config.regulatory_weight,
            score=score,
            description=f"Regulatory risk: {reg_count} regulations",
            metadata={"regulation_count": reg_count},
        )

    def _calculate_overall_risk(self, factors: List[RiskFactor]) -> float:
        """Calculate weighted overall risk score"""
        if not factors:
            return 0.0

        total_weight = sum(f.weight for f in factors)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(f.score * f.weight for f in factors)
        return weighted_sum / total_weight

    def _determine_risk_level(self, score: float) -> RiskLevel:
        """Determine risk level from score"""
        if score < self.config.minimal_threshold:
            return RiskLevel.MINIMAL
        elif score < self.config.low_threshold:
            return RiskLevel.LOW
        elif score < self.config.medium_threshold:
            return RiskLevel.MEDIUM
        elif score < self.config.high_threshold:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def _calculate_audit_interval(self, risk_score: float) -> int:
        """Calculate recommended audit interval based on risk"""
        # Higher risk = shorter interval
        # Use exponential decay
        base = self.config.base_audit_interval_hours
        min_interval = self.config.min_audit_interval_hours
        max_interval = self.config.max_audit_interval_hours

        # At risk 0, use max interval; at risk 100, use min interval
        decay_factor = risk_score / 100
        interval = max_interval - (max_interval - min_interval) * decay_factor

        return max(min_interval, min(max_interval, int(interval)))

    async def _check_triggers(
        self, profile: ModelRiskProfile, previous_score: float
    ) -> List[RiskTrigger]:
        """Check which triggers should fire"""
        triggered = []

        for trigger in self._triggers.values():
            if not trigger.enabled:
                continue

            should_trigger = False

            # Check risk threshold
            if profile.overall_risk_score >= trigger.risk_threshold:
                # Check category filter
                if trigger.risk_categories:
                    category_scores = {
                        f.category: f.score
                        for f in profile.risk_factors
                        if f.category in trigger.risk_categories
                    }
                    if category_scores:
                        avg_category_score = sum(category_scores.values()) / len(category_scores)
                        if avg_category_score >= trigger.risk_threshold:
                            should_trigger = True
                else:
                    should_trigger = True

            # Check risk change threshold
            if trigger.risk_change_threshold > 0:
                change = profile.overall_risk_score - previous_score
                if change >= trigger.risk_change_threshold:
                    should_trigger = True

            # Check time since last audit
            if trigger.max_days_without_audit > 0 and profile.last_audit:
                days_since = (datetime.utcnow() - profile.last_audit).days
                if days_since >= trigger.max_days_without_audit:
                    should_trigger = True
            elif trigger.max_days_without_audit > 0 and not profile.last_audit:
                # Never audited
                should_trigger = True

            # Check minimum interval
            if should_trigger and trigger.min_audit_interval_hours > 0:
                if profile.scheduled_audit:
                    hours_since_scheduled = (
                        datetime.utcnow() - profile.scheduled_audit
                    ).total_seconds() / 3600
                    if hours_since_scheduled < trigger.min_audit_interval_hours:
                        should_trigger = False

            if should_trigger:
                triggered.append(trigger)
                trigger.trigger_count += 1
                trigger.last_triggered = datetime.utcnow()

        # Schedule audit if triggers fired
        if triggered:
            await self._schedule_triggered_audit(profile, triggered)

        return triggered

    async def _schedule_triggered_audit(
        self, profile: ModelRiskProfile, triggers: List[RiskTrigger]
    ):
        """Schedule an audit based on triggered conditions"""
        # Determine priority (highest from triggers)
        priority_order = ["critical", "high", "medium", "low"]
        priority = "medium"

        for trigger in triggers:
            if trigger.priority_override:
                if priority_order.index(trigger.priority_override) < priority_order.index(priority):
                    priority = trigger.priority_override

        # Update profile
        profile.scheduled_audit = datetime.utcnow()

        # Notify callbacks
        for callback in self._audit_callbacks:
            try:
                audit_request = {
                    "model_id": profile.model_id,
                    "priority": priority,
                    "risk_score": profile.overall_risk_score,
                    "triggers": [t.name for t in triggers],
                    "scheduled_at": datetime.utcnow().isoformat(),
                }

                if asyncio.iscoroutinefunction(callback):
                    await callback(audit_request)
                else:
                    callback(audit_request)
            except Exception as e:
                logger.error("Audit callback failed", error=str(e))

        logger.info(
            "Audit scheduled from triggers",
            model_id=profile.model_id,
            priority=priority,
            triggers=[t.name for t in triggers],
        )

    def _recommend_action(
        self, profile: ModelRiskProfile, triggers: List[RiskTrigger]
    ) -> str:
        """Recommend action based on risk assessment"""
        if profile.risk_level == RiskLevel.CRITICAL:
            return "Immediate audit required - Critical risk level detected"
        elif profile.risk_level == RiskLevel.HIGH:
            return "Schedule high-priority audit within 24 hours"
        elif triggers:
            return f"Audit triggered by: {', '.join(t.name for t in triggers)}"
        elif profile.risk_level == RiskLevel.MEDIUM:
            return f"Schedule audit within {profile.audit_interval_hours} hours"
        elif profile.risk_level == RiskLevel.LOW:
            return "Continue monitoring, audit within 1 week"
        else:
            return "Low risk, standard monitoring schedule"

    async def _notify_assessment(self, result: RiskAssessmentResult):
        """Notify assessment callbacks"""
        for callback in self._assessment_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                logger.error("Assessment callback failed", error=str(e))

    async def _assessment_loop(self):
        """Background loop to periodically assess all models"""
        while self._running:
            try:
                for model_id in list(self._profiles.keys()):
                    try:
                        await self.assess_risk(model_id)
                    except Exception as e:
                        logger.error(
                            "Risk assessment failed",
                            model_id=model_id,
                            error=str(e),
                        )

                await asyncio.sleep(self.config.assessment_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in assessment loop", error=str(e))
                await asyncio.sleep(5)

    def record_audit_result(
        self, model_id: str, success: bool, **metadata
    ) -> bool:
        """Record the result of an audit"""
        if model_id not in self._profiles:
            return False

        profile = self._profiles[model_id]
        profile.last_audit = datetime.utcnow()
        profile.audit_count += 1

        if not success:
            profile.failure_count += 1

        logger.info(
            "Audit result recorded",
            model_id=model_id,
            success=success,
            audit_count=profile.audit_count,
        )

        return True

    def update_risk_factor(
        self,
        model_id: str,
        factor_name: str,
        score: float,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """Update a specific risk factor for a model"""
        if model_id not in self._profiles:
            return False

        profile = self._profiles[model_id]

        # Find and update existing factor
        for factor in profile.risk_factors:
            if factor.name == factor_name:
                factor.score = max(0, min(100, score))
                factor.last_updated = datetime.utcnow()
                if metadata:
                    factor.metadata.update(metadata)
                return True

        return False

    def get_profile(self, model_id: str) -> Optional[ModelRiskProfile]:
        """Get a model's risk profile"""
        return self._profiles.get(model_id)

    def get_all_profiles(self) -> List[ModelRiskProfile]:
        """Get all risk profiles"""
        return list(self._profiles.values())

    def get_high_risk_models(self, threshold: float = None) -> List[ModelRiskProfile]:
        """Get models above risk threshold"""
        threshold = threshold or self.config.high_threshold
        return [
            p for p in self._profiles.values()
            if p.overall_risk_score >= threshold
        ]

    def get_models_needing_audit(self) -> List[ModelRiskProfile]:
        """Get models that need immediate audit"""
        return [
            p for p in self._profiles.values()
            if p.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
            or (p.last_audit and (datetime.utcnow() - p.last_audit).days > 30)
            or not p.last_audit
        ]

    def get_risk_trends(self, model_id: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get risk score trends for a model"""
        if model_id not in self._profiles:
            return []

        profile = self._profiles[model_id]
        cutoff = datetime.utcnow() - timedelta(days=days)

        return [
            h for h in profile.risk_history
            if datetime.fromisoformat(h["timestamp"]) > cutoff
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        level_counts = {}
        category_scores = {}

        for profile in self._profiles.values():
            level_counts[profile.risk_level.value] = level_counts.get(
                profile.risk_level.value, 0
            ) + 1

            for factor in profile.risk_factors:
                cat = factor.category.value
                if cat not in category_scores:
                    category_scores[cat] = []
                category_scores[cat].append(factor.score)

        # Calculate average by category
        avg_by_category = {
            cat: sum(scores) / len(scores) if scores else 0
            for cat, scores in category_scores.items()
        }

        total_score = sum(p.overall_risk_score for p in self._profiles.values())
        avg_score = total_score / len(self._profiles) if self._profiles else 0

        return {
            "total_models": len(self._profiles),
            "average_risk_score": round(avg_score, 2),
            "by_risk_level": level_counts,
            "avg_by_category": {k: round(v, 2) for k, v in avg_by_category.items()},
            "high_risk_count": len(self.get_high_risk_models()),
            "needing_audit": len(self.get_models_needing_audit()),
            "total_triggers": len(self._triggers),
            "active_triggers": sum(1 for t in self._triggers.values() if t.enabled),
        }


# Global risk scheduler
risk_scheduler: Optional[RiskBasedScheduler] = None


def get_risk_scheduler() -> RiskBasedScheduler:
    """Get the global risk scheduler"""
    global risk_scheduler
    if risk_scheduler is None:
        risk_scheduler = RiskBasedScheduler()
    return risk_scheduler

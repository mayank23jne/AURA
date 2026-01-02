"""Compliance Risk Scoring Model for AURA Platform"""

import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import uuid
import math

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()
from src.db import SessionLocal
from src.models.orm import ORMComplianceScore


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks"""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    NIST_AI = "nist_ai"
    EU_AI_ACT = "eu_ai_act"
    FCRA = "fcra"
    ECOA = "ecoa"
    PCI_DSS = "pci_dss"
    FEDRAMP = "fedramp"


class RiskDomain(str, Enum):
    """Domains of compliance risk"""
    DATA_PRIVACY = "data_privacy"
    MODEL_GOVERNANCE = "model_governance"
    FAIRNESS_BIAS = "fairness_bias"
    TRANSPARENCY = "transparency"
    SECURITY = "security"
    ACCOUNTABILITY = "accountability"
    HUMAN_OVERSIGHT = "human_oversight"
    DATA_QUALITY = "data_quality"
    DOCUMENTATION = "documentation"
    MONITORING = "monitoring"


class ComplianceStatus(str, Enum):
    """Overall compliance status"""
    COMPLIANT = "compliant"
    PARTIAL = "partial"
    NON_COMPLIANT = "non_compliant"
    UNKNOWN = "unknown"
    UNDER_REVIEW = "under_review"


class ControlEffectiveness(str, Enum):
    """Effectiveness of compliance controls"""
    EFFECTIVE = "effective"
    PARTIALLY_EFFECTIVE = "partially_effective"
    INEFFECTIVE = "ineffective"
    NOT_IMPLEMENTED = "not_implemented"
    NOT_APPLICABLE = "not_applicable"


class ComplianceControl(BaseModel):
    """A compliance control measure"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    domain: RiskDomain
    frameworks: List[ComplianceFramework] = Field(default_factory=list)

    # Control assessment
    effectiveness: ControlEffectiveness = ControlEffectiveness.NOT_IMPLEMENTED
    implementation_score: float = 0.0  # 0-100
    last_tested: Optional[datetime] = None
    test_frequency_days: int = 90

    # Risk mitigation
    risk_reduction: float = 0.0  # How much risk this control reduces (0-1)
    priority: int = 1  # 1=highest priority

    # Evidence
    evidence_required: List[str] = Field(default_factory=list)
    evidence_collected: List[str] = Field(default_factory=list)


class ComplianceRequirement(BaseModel):
    """A compliance requirement from a framework"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    framework: ComplianceFramework
    requirement_id: str  # e.g., "GDPR Art. 22", "SOC2 CC1.1"
    name: str
    description: str = ""
    domain: RiskDomain

    # Requirement details
    mandatory: bool = True
    weight: float = 1.0  # Importance weight
    penalty_severity: float = 1.0  # Severity of non-compliance

    # Compliance status
    status: ComplianceStatus = ComplianceStatus.UNKNOWN
    compliance_score: float = 0.0  # 0-100
    gap_description: str = ""

    # Related controls
    control_ids: List[str] = Field(default_factory=list)


class DomainRiskScore(BaseModel):
    """Risk score for a specific domain"""
    domain: RiskDomain
    score: float  # 0-100 (higher = more risk)
    weight: float = 1.0
    contributing_factors: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class FrameworkComplianceScore(BaseModel):
    """Compliance score for a specific framework"""
    framework: ComplianceFramework
    overall_score: float  # 0-100 (higher = more compliant)
    status: ComplianceStatus
    requirements_met: int = 0
    requirements_total: int = 0
    critical_gaps: List[str] = Field(default_factory=list)
    priority_actions: List[str] = Field(default_factory=list)


class ComplianceRiskScore(BaseModel):
    """Comprehensive compliance risk score"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    model_name: str = ""

    # Overall scores
    overall_risk_score: float = 0.0  # 0-100 (higher = more risk)
    overall_compliance_score: float = 0.0  # 0-100 (higher = more compliant)
    status: ComplianceStatus = ComplianceStatus.UNKNOWN

    # Domain scores
    domain_scores: List[DomainRiskScore] = Field(default_factory=list)

    # Framework scores
    framework_scores: List[FrameworkComplianceScore] = Field(default_factory=list)

    # Risk factors
    risk_factors: List[str] = Field(default_factory=list)
    mitigating_factors: List[str] = Field(default_factory=list)

    # Recommendations
    high_priority_actions: List[str] = Field(default_factory=list)
    medium_priority_actions: List[str] = Field(default_factory=list)
    low_priority_actions: List[str] = Field(default_factory=list)

    # Metadata
    calculated_at: datetime = Field(default_factory=datetime.utcnow)
    next_review: datetime = Field(default_factory=lambda: datetime.utcnow() + timedelta(days=30))
    calculation_version: str = "1.0"


class ComplianceScoringConfig(BaseModel):
    """Configuration for compliance scoring"""
    # Weights for different factors
    control_effectiveness_weight: float = 0.3
    historical_compliance_weight: float = 0.2
    documentation_weight: float = 0.15
    monitoring_weight: float = 0.15
    incident_history_weight: float = 0.2

    # Domain weights
    domain_weights: Dict[str, float] = Field(default_factory=lambda: {
        "data_privacy": 1.0,
        "model_governance": 0.9,
        "fairness_bias": 1.0,
        "transparency": 0.8,
        "security": 1.0,
        "accountability": 0.8,
        "human_oversight": 0.7,
        "data_quality": 0.8,
        "documentation": 0.6,
        "monitoring": 0.7,
    })

    # Thresholds
    compliant_threshold: float = 80.0
    partial_threshold: float = 60.0

    # Time-based factors
    audit_freshness_days: int = 90
    incident_lookback_days: int = 365


class ComplianceRiskScorer:
    """
    Compliance risk scoring system.

    Features:
    - Multi-framework compliance assessment
    - Domain-based risk calculation
    - Control effectiveness evaluation
    - Historical trend analysis
    - Prioritized recommendations
    - Regulatory requirement mapping
    """

    def __init__(self, config: ComplianceScoringConfig = None):
        self.config = config or ComplianceScoringConfig()
        self._controls: Dict[str, ComplianceControl] = {}
        self._requirements: Dict[str, ComplianceRequirement] = {}
        self._scores: Dict[str, ComplianceRiskScore] = {}
        self._history: Dict[str, List[ComplianceRiskScore]] = {}

        # Initialize default controls and requirements
        self._initialize_defaults()
        
        # Load history from DB
        self._load_history_from_db()

        logger.info("ComplianceRiskScorer initialized with Database backend")

    def _load_history_from_db(self):
        """Load compliance score history from database"""
        db = SessionLocal()
        try:
            orm_scores = db.query(ORMComplianceScore).order_by(ORMComplianceScore.calculated_at.asc()).all()
            for orm_score in orm_scores:
                score_dict = orm_score.to_dict()
                # Reconstruct Pydantic model
                score = ComplianceRiskScore(
                    id=score_dict["id"],
                    model_id=score_dict["model_id"],
                    model_name=score_dict.get("model_name", ""),
                    overall_risk_score=score_dict["overall_risk_score"],
                    overall_compliance_score=score_dict["overall_compliance_score"],
                    status=score_dict["status"],
                    domain_scores=[DomainRiskScore(**d) for d in score_dict.get("domain_scores", [])],
                    framework_scores=[FrameworkComplianceScore(**f) for f in score_dict.get("framework_scores", [])],
                    risk_factors=score_dict.get("risk_factors", []),
                    mitigating_factors=score_dict.get("mitigating_factors", []),
                    high_priority_actions=score_dict.get("recommendations", {}).get("high", []),
                    medium_priority_actions=score_dict.get("recommendations", {}).get("medium", []),
                    low_priority_actions=score_dict.get("recommendations", {}).get("low", []),
                    calculated_at=score_dict["calculated_at"]
                )
                
                self._scores[score.model_id] = score
                if score.model_id not in self._history:
                    self._history[score.model_id] = []
                self._history[score.model_id].append(score)
        except Exception as e:
            logger.error("Failed to load compliance history", error=str(e))
        finally:
            db.close()

    def _initialize_defaults(self):
        """Initialize default controls and requirements"""
        # Data Privacy Controls
        self._add_default_controls()
        self._add_default_requirements()

    def _add_default_controls(self):
        """Add default compliance controls"""
        controls = [
            # Data Privacy Controls
            ComplianceControl(
                name="Data Minimization",
                description="Collect and process only necessary data",
                domain=RiskDomain.DATA_PRIVACY,
                frameworks=[ComplianceFramework.GDPR, ComplianceFramework.CCPA],
                risk_reduction=0.2,
                priority=1,
                evidence_required=["data_inventory", "retention_policy"],
            ),
            ComplianceControl(
                name="Consent Management",
                description="Obtain and manage user consent",
                domain=RiskDomain.DATA_PRIVACY,
                frameworks=[ComplianceFramework.GDPR, ComplianceFramework.CCPA],
                risk_reduction=0.25,
                priority=1,
                evidence_required=["consent_records", "opt_out_mechanism"],
            ),
            ComplianceControl(
                name="Data Encryption",
                description="Encrypt data at rest and in transit",
                domain=RiskDomain.SECURITY,
                frameworks=[ComplianceFramework.SOC2, ComplianceFramework.PCI_DSS, ComplianceFramework.HIPAA],
                risk_reduction=0.3,
                priority=1,
                evidence_required=["encryption_config", "key_management"],
            ),

            # Model Governance Controls
            ComplianceControl(
                name="Model Documentation",
                description="Document model design, training, and deployment",
                domain=RiskDomain.DOCUMENTATION,
                frameworks=[ComplianceFramework.EU_AI_ACT, ComplianceFramework.NIST_AI],
                risk_reduction=0.15,
                priority=2,
                evidence_required=["model_cards", "training_docs"],
            ),
            ComplianceControl(
                name="Version Control",
                description="Maintain version history for models",
                domain=RiskDomain.MODEL_GOVERNANCE,
                frameworks=[ComplianceFramework.SOC2, ComplianceFramework.NIST_AI],
                risk_reduction=0.1,
                priority=2,
                evidence_required=["version_logs", "change_history"],
            ),

            # Fairness Controls
            ComplianceControl(
                name="Bias Testing",
                description="Regular testing for model bias",
                domain=RiskDomain.FAIRNESS_BIAS,
                frameworks=[ComplianceFramework.EU_AI_ACT, ComplianceFramework.ECOA],
                risk_reduction=0.25,
                priority=1,
                evidence_required=["bias_test_results", "demographic_analysis"],
            ),
            ComplianceControl(
                name="Fairness Metrics Monitoring",
                description="Continuous monitoring of fairness metrics",
                domain=RiskDomain.FAIRNESS_BIAS,
                frameworks=[ComplianceFramework.EU_AI_ACT, ComplianceFramework.FCRA],
                risk_reduction=0.2,
                priority=1,
                evidence_required=["fairness_dashboards", "alert_logs"],
            ),

            # Transparency Controls
            ComplianceControl(
                name="Explainability",
                description="Provide explanations for model decisions",
                domain=RiskDomain.TRANSPARENCY,
                frameworks=[ComplianceFramework.GDPR, ComplianceFramework.EU_AI_ACT],
                risk_reduction=0.2,
                priority=1,
                evidence_required=["explanation_system", "user_documentation"],
            ),
            ComplianceControl(
                name="Decision Logging",
                description="Log all model decisions for audit",
                domain=RiskDomain.ACCOUNTABILITY,
                frameworks=[ComplianceFramework.SOC2, ComplianceFramework.EU_AI_ACT],
                risk_reduction=0.15,
                priority=2,
                evidence_required=["decision_logs", "audit_trails"],
            ),

            # Human Oversight Controls
            ComplianceControl(
                name="Human Review Process",
                description="Enable human review of high-stakes decisions",
                domain=RiskDomain.HUMAN_OVERSIGHT,
                frameworks=[ComplianceFramework.EU_AI_ACT, ComplianceFramework.NIST_AI],
                risk_reduction=0.2,
                priority=1,
                evidence_required=["review_workflow", "escalation_policy"],
            ),
            ComplianceControl(
                name="Override Capability",
                description="Allow humans to override model decisions",
                domain=RiskDomain.HUMAN_OVERSIGHT,
                frameworks=[ComplianceFramework.EU_AI_ACT],
                risk_reduction=0.15,
                priority=2,
                evidence_required=["override_logs", "procedure_docs"],
            ),

            # Monitoring Controls
            ComplianceControl(
                name="Performance Monitoring",
                description="Monitor model performance continuously",
                domain=RiskDomain.MONITORING,
                frameworks=[ComplianceFramework.SOC2, ComplianceFramework.NIST_AI],
                risk_reduction=0.15,
                priority=2,
                evidence_required=["monitoring_dashboards", "alert_config"],
            ),
            ComplianceControl(
                name="Drift Detection",
                description="Detect data and model drift",
                domain=RiskDomain.DATA_QUALITY,
                frameworks=[ComplianceFramework.NIST_AI],
                risk_reduction=0.15,
                priority=2,
                evidence_required=["drift_reports", "retraining_logs"],
            ),
        ]

        for control in controls:
            self._controls[control.id] = control

    def _add_default_requirements(self):
        """Add default compliance requirements"""
        requirements = [
            # GDPR Requirements
            ComplianceRequirement(
                framework=ComplianceFramework.GDPR,
                requirement_id="Art. 5",
                name="Data Processing Principles",
                description="Lawfulness, fairness, transparency, purpose limitation, data minimization",
                domain=RiskDomain.DATA_PRIVACY,
                weight=1.0,
                penalty_severity=1.0,
            ),
            ComplianceRequirement(
                framework=ComplianceFramework.GDPR,
                requirement_id="Art. 22",
                name="Automated Decision Making",
                description="Right not to be subject to solely automated decisions",
                domain=RiskDomain.HUMAN_OVERSIGHT,
                weight=0.9,
                penalty_severity=0.8,
            ),
            ComplianceRequirement(
                framework=ComplianceFramework.GDPR,
                requirement_id="Art. 35",
                name="Data Protection Impact Assessment",
                description="DPIA for high-risk processing",
                domain=RiskDomain.ACCOUNTABILITY,
                weight=0.8,
                penalty_severity=0.7,
            ),

            # EU AI Act Requirements
            ComplianceRequirement(
                framework=ComplianceFramework.EU_AI_ACT,
                requirement_id="Art. 9",
                name="Risk Management System",
                description="Establish risk management for high-risk AI",
                domain=RiskDomain.MODEL_GOVERNANCE,
                weight=1.0,
                penalty_severity=1.0,
            ),
            ComplianceRequirement(
                framework=ComplianceFramework.EU_AI_ACT,
                requirement_id="Art. 10",
                name="Data Governance",
                description="Training, validation, and testing data governance",
                domain=RiskDomain.DATA_QUALITY,
                weight=1.0,
                penalty_severity=1.0,
            ),
            ComplianceRequirement(
                framework=ComplianceFramework.EU_AI_ACT,
                requirement_id="Art. 13",
                name="Transparency",
                description="Transparency and provision of information to users",
                domain=RiskDomain.TRANSPARENCY,
                weight=0.9,
                penalty_severity=0.8,
            ),
            ComplianceRequirement(
                framework=ComplianceFramework.EU_AI_ACT,
                requirement_id="Art. 14",
                name="Human Oversight",
                description="Human oversight of high-risk AI systems",
                domain=RiskDomain.HUMAN_OVERSIGHT,
                weight=1.0,
                penalty_severity=1.0,
            ),

            # SOC 2 Requirements
            ComplianceRequirement(
                framework=ComplianceFramework.SOC2,
                requirement_id="CC1.1",
                name="Control Environment",
                description="Commitment to integrity and ethical values",
                domain=RiskDomain.ACCOUNTABILITY,
                weight=0.8,
                penalty_severity=0.7,
            ),
            ComplianceRequirement(
                framework=ComplianceFramework.SOC2,
                requirement_id="CC6.1",
                name="Logical Access Security",
                description="Restrict logical access to systems",
                domain=RiskDomain.SECURITY,
                weight=1.0,
                penalty_severity=0.9,
            ),

            # NIST AI RMF Requirements
            ComplianceRequirement(
                framework=ComplianceFramework.NIST_AI,
                requirement_id="GOVERN",
                name="Governance",
                description="Culture of risk management is cultivated",
                domain=RiskDomain.MODEL_GOVERNANCE,
                weight=0.9,
                penalty_severity=0.6,
            ),
            ComplianceRequirement(
                framework=ComplianceFramework.NIST_AI,
                requirement_id="MEASURE",
                name="Measure",
                description="AI risks are identified and measured",
                domain=RiskDomain.MONITORING,
                weight=0.8,
                penalty_severity=0.6,
            ),
        ]

        for req in requirements:
            self._requirements[req.id] = req

    def add_control(self, control: ComplianceControl) -> str:
        """Add a compliance control"""
        self._controls[control.id] = control
        return control.id

    def add_requirement(self, requirement: ComplianceRequirement) -> str:
        """Add a compliance requirement"""
        self._requirements[requirement.id] = requirement
        return requirement.id

    def update_control_effectiveness(
        self,
        control_id: str,
        effectiveness: ControlEffectiveness,
        implementation_score: float,
        evidence: List[str] = None,
    ) -> bool:
        """Update control effectiveness assessment"""
        if control_id not in self._controls:
            return False

        control = self._controls[control_id]
        control.effectiveness = effectiveness
        control.implementation_score = max(0, min(100, implementation_score))
        control.last_tested = datetime.utcnow()

        if evidence:
            control.evidence_collected = evidence

        return True

    def update_requirement_status(
        self,
        requirement_id: str,
        status: ComplianceStatus,
        compliance_score: float,
        gap_description: str = "",
    ) -> bool:
        """Update requirement compliance status"""
        if requirement_id not in self._requirements:
            return False

        req = self._requirements[requirement_id]
        req.status = status
        req.compliance_score = max(0, min(100, compliance_score))
        req.gap_description = gap_description

        return True

    async def calculate_score(
        self,
        model_id: str,
        model_name: str = "",
        applicable_frameworks: List[ComplianceFramework] = None,
        model_metadata: Dict[str, Any] = None,
    ) -> ComplianceRiskScore:
        """Calculate comprehensive compliance risk score"""
        metadata = model_metadata or {}

        # Calculate domain scores
        domain_scores = await self._calculate_domain_scores(
            applicable_frameworks or list(ComplianceFramework)
        )

        # Calculate framework scores
        framework_scores = await self._calculate_framework_scores(
            applicable_frameworks or list(ComplianceFramework)
        )

        # Calculate overall scores
        overall_compliance = self._calculate_overall_compliance(
            domain_scores, framework_scores
        )
        overall_risk = 100 - overall_compliance

        # Determine status
        status = self._determine_status(overall_compliance)

        # Identify risk factors and mitigating factors
        risk_factors, mitigating = self._identify_factors(domain_scores, framework_scores)

        # Generate recommendations
        high, medium, low = self._generate_recommendations(
            domain_scores, framework_scores
        )

        # Create score
        score = ComplianceRiskScore(
            model_id=model_id,
            model_name=model_name,
            overall_risk_score=round(overall_risk, 2),
            overall_compliance_score=round(overall_compliance, 2),
            status=status,
            domain_scores=domain_scores,
            framework_scores=framework_scores,
            risk_factors=risk_factors,
            mitigating_factors=mitigating,
            high_priority_actions=high,
            medium_priority_actions=medium,
            low_priority_actions=low,
        )

        # Store score
        self._scores[model_id] = score

        # Add to history
        if model_id not in self._history:
            self._history[model_id] = []
        self._history[model_id].append(score)

        # Trim history
        if len(self._history[model_id]) > 100:
            self._history[model_id] = self._history[model_id][-100:]
            
        # Persist to DB
        db = SessionLocal()
        try:
            orm_score = ORMComplianceScore(
                id=score.id,
                model_id=score.model_id,
                overall_risk_score=score.overall_risk_score,
                overall_compliance_score=score.overall_compliance_score,
                status=score.status,
                domain_scores=[d.model_dump() for d in score.domain_scores],
                framework_scores=[f.model_dump() for f in score.framework_scores],
                risk_factors=score.risk_factors,
                mitigating_factors=score.mitigating_factors,
                recommendations={
                    "high": score.high_priority_actions,
                    "medium": score.medium_priority_actions,
                    "low": score.low_priority_actions
                },
                calculated_at=score.calculated_at
            )
            db.merge(orm_score) # merge allows updating if exists, though ID is UUID
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error("Failed to persist compliance score", error=str(e))
        finally:
            db.close()

        logger.info(
            "Compliance score calculated",
            model_id=model_id,
            compliance_score=overall_compliance,
            risk_score=overall_risk,
            status=status.value,
        )

        return score

    async def _calculate_domain_scores(
        self, frameworks: List[ComplianceFramework]
    ) -> List[DomainRiskScore]:
        """Calculate risk scores by domain"""
        domain_scores = []

        for domain in RiskDomain:
            # Get controls for this domain
            domain_controls = [
                c for c in self._controls.values()
                if c.domain == domain and any(f in frameworks for f in c.frameworks)
            ]

            # Calculate control effectiveness score
            if domain_controls:
                control_score = sum(c.implementation_score for c in domain_controls) / len(domain_controls)
            else:
                control_score = 50  # Neutral if no controls

            # Get requirements for this domain
            domain_reqs = [
                r for r in self._requirements.values()
                if r.domain == domain and r.framework in frameworks
            ]

            # Calculate requirement compliance
            if domain_reqs:
                req_score = sum(r.compliance_score for r in domain_reqs) / len(domain_reqs)
            else:
                req_score = 50

            # Combined score (higher = less risk)
            combined_compliance = (control_score + req_score) / 2
            risk_score = 100 - combined_compliance

            # Get domain weight
            weight = self.config.domain_weights.get(domain.value, 1.0)

            # Generate factors
            factors = []
            if control_score < 50:
                factors.append(f"Weak controls in {domain.value}")
            if req_score < 50:
                factors.append(f"Low requirement compliance in {domain.value}")

            # Generate recommendations
            recommendations = []
            if risk_score > 60:
                low_controls = [c for c in domain_controls if c.implementation_score < 50]
                for c in low_controls[:3]:
                    recommendations.append(f"Improve '{c.name}' implementation")

            domain_scores.append(DomainRiskScore(
                domain=domain,
                score=round(risk_score, 2),
                weight=weight,
                contributing_factors=factors,
                recommendations=recommendations,
            ))

        return domain_scores

    async def _calculate_framework_scores(
        self, frameworks: List[ComplianceFramework]
    ) -> List[FrameworkComplianceScore]:
        """Calculate compliance scores by framework"""
        framework_scores = []

        for framework in frameworks:
            # Get requirements for this framework
            reqs = [
                r for r in self._requirements.values()
                if r.framework == framework
            ]

            if not reqs:
                continue

            # Calculate compliance
            total_weight = sum(r.weight for r in reqs)
            if total_weight == 0:
                continue

            weighted_score = sum(r.compliance_score * r.weight for r in reqs) / total_weight
            met_count = sum(1 for r in reqs if r.status == ComplianceStatus.COMPLIANT)

            # Determine status
            if weighted_score >= self.config.compliant_threshold:
                status = ComplianceStatus.COMPLIANT
            elif weighted_score >= self.config.partial_threshold:
                status = ComplianceStatus.PARTIAL
            else:
                status = ComplianceStatus.NON_COMPLIANT

            # Identify critical gaps
            critical_gaps = [
                f"{r.requirement_id}: {r.gap_description}"
                for r in reqs
                if r.status == ComplianceStatus.NON_COMPLIANT and r.mandatory
            ][:5]

            # Priority actions
            non_compliant = [r for r in reqs if r.compliance_score < 60]
            non_compliant.sort(key=lambda x: x.penalty_severity, reverse=True)
            actions = [
                f"Address {r.requirement_id} ({r.name})"
                for r in non_compliant[:5]
            ]

            framework_scores.append(FrameworkComplianceScore(
                framework=framework,
                overall_score=round(weighted_score, 2),
                status=status,
                requirements_met=met_count,
                requirements_total=len(reqs),
                critical_gaps=critical_gaps,
                priority_actions=actions,
            ))

        return framework_scores

    def _calculate_overall_compliance(
        self,
        domain_scores: List[DomainRiskScore],
        framework_scores: List[FrameworkComplianceScore],
    ) -> float:
        """Calculate overall compliance score"""
        # Weight domain scores
        total_domain_weight = sum(d.weight for d in domain_scores)
        if total_domain_weight > 0:
            domain_compliance = sum(
                (100 - d.score) * d.weight for d in domain_scores
            ) / total_domain_weight
        else:
            domain_compliance = 50

        # Average framework scores
        if framework_scores:
            framework_compliance = sum(f.overall_score for f in framework_scores) / len(framework_scores)
        else:
            framework_compliance = 50

        # Combined (60% domain, 40% framework)
        return domain_compliance * 0.6 + framework_compliance * 0.4

    def _determine_status(self, compliance_score: float) -> ComplianceStatus:
        """Determine overall compliance status"""
        if compliance_score >= self.config.compliant_threshold:
            return ComplianceStatus.COMPLIANT
        elif compliance_score >= self.config.partial_threshold:
            return ComplianceStatus.PARTIAL
        else:
            return ComplianceStatus.NON_COMPLIANT

    def _identify_factors(
        self,
        domain_scores: List[DomainRiskScore],
        framework_scores: List[FrameworkComplianceScore],
    ) -> Tuple[List[str], List[str]]:
        """Identify risk and mitigating factors"""
        risk_factors = []
        mitigating = []

        # From domain scores
        for domain in domain_scores:
            if domain.score > 60:
                risk_factors.append(f"High risk in {domain.domain.value}: {domain.score:.0f}")
            elif domain.score < 30:
                mitigating.append(f"Strong compliance in {domain.domain.value}")
            risk_factors.extend(domain.contributing_factors)

        # From framework scores
        for fw in framework_scores:
            if fw.status == ComplianceStatus.NON_COMPLIANT:
                risk_factors.append(f"Non-compliant with {fw.framework.value}")
            elif fw.status == ComplianceStatus.COMPLIANT:
                mitigating.append(f"Fully compliant with {fw.framework.value}")

        return risk_factors[:10], mitigating[:10]

    def _generate_recommendations(
        self,
        domain_scores: List[DomainRiskScore],
        framework_scores: List[FrameworkComplianceScore],
    ) -> Tuple[List[str], List[str], List[str]]:
        """Generate prioritized recommendations"""
        high = []
        medium = []
        low = []

        # Critical domain issues
        for domain in domain_scores:
            if domain.score > 70:
                high.extend(domain.recommendations[:2])
            elif domain.score > 50:
                medium.extend(domain.recommendations[:2])
            else:
                low.extend(domain.recommendations[:1])

        # Framework gaps
        for fw in framework_scores:
            if fw.status == ComplianceStatus.NON_COMPLIANT:
                high.extend(fw.priority_actions[:2])
            elif fw.status == ComplianceStatus.PARTIAL:
                medium.extend(fw.priority_actions[:2])

        # Controls needing attention
        stale_controls = [
            c for c in self._controls.values()
            if c.last_tested and (datetime.utcnow() - c.last_tested).days > c.test_frequency_days
        ]
        for c in stale_controls[:3]:
            medium.append(f"Re-test control: {c.name}")

        # Deduplicate
        high = list(dict.fromkeys(high))[:5]
        medium = list(dict.fromkeys(medium))[:5]
        low = list(dict.fromkeys(low))[:5]

        return high, medium, low

    def get_score(self, model_id: str) -> Optional[ComplianceRiskScore]:
        """Get compliance score for a model"""
        return self._scores.get(model_id)

    def get_history(
        self, model_id: str, days: int = 30
    ) -> List[ComplianceRiskScore]:
        """Get historical scores for a model"""
        if model_id not in self._history:
            return []

        cutoff = datetime.utcnow() - timedelta(days=days)
        return [
            s for s in self._history[model_id]
            if s.calculated_at > cutoff
        ]

    def get_trend(self, model_id: str) -> Dict[str, Any]:
        """Get compliance trend for a model"""
        history = self.get_history(model_id)
        if len(history) < 2:
            return {"trend": "insufficient_data"}

        scores = [s.overall_compliance_score for s in history]
        recent = sum(scores[-5:]) / min(5, len(scores))
        older = sum(scores[:-5]) / max(1, len(scores) - 5)

        diff = recent - older
        if diff > 5:
            trend = "improving"
        elif diff < -5:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "recent_average": round(recent, 2),
            "change": round(diff, 2),
        }

    def get_controls_by_framework(
        self, framework: ComplianceFramework
    ) -> List[ComplianceControl]:
        """Get controls applicable to a framework"""
        return [
            c for c in self._controls.values()
            if framework in c.frameworks
        ]

    def get_requirements_by_framework(
        self, framework: ComplianceFramework
    ) -> List[ComplianceRequirement]:
        """Get requirements for a framework"""
        return [
            r for r in self._requirements.values()
            if r.framework == framework
        ]

    def get_gap_analysis(
        self, model_id: str
    ) -> Dict[str, Any]:
        """Get compliance gap analysis"""
        score = self._scores.get(model_id)
        if not score:
            return {"error": "No score found for model"}

        gaps = {
            "overall_status": score.status.value,
            "compliance_score": score.overall_compliance_score,
            "risk_score": score.overall_risk_score,
            "critical_domains": [],
            "framework_gaps": [],
            "control_gaps": [],
            "recommended_timeline": "",
        }

        # Critical domains
        for domain in score.domain_scores:
            if domain.score > 60:
                gaps["critical_domains"].append({
                    "domain": domain.domain.value,
                    "risk_score": domain.score,
                    "factors": domain.contributing_factors,
                })

        # Framework gaps
        for fw in score.framework_scores:
            if fw.status != ComplianceStatus.COMPLIANT:
                gaps["framework_gaps"].append({
                    "framework": fw.framework.value,
                    "score": fw.overall_score,
                    "gaps": fw.critical_gaps,
                })

        # Control gaps
        for control in self._controls.values():
            if control.implementation_score < 50:
                gaps["control_gaps"].append({
                    "control": control.name,
                    "score": control.implementation_score,
                    "domain": control.domain.value,
                })

        # Timeline recommendation
        if score.status == ComplianceStatus.NON_COMPLIANT:
            gaps["recommended_timeline"] = "30-60 days to address critical gaps"
        elif score.status == ComplianceStatus.PARTIAL:
            gaps["recommended_timeline"] = "60-90 days to achieve full compliance"
        else:
            gaps["recommended_timeline"] = "Ongoing maintenance"

        return gaps

    def get_stats(self) -> Dict[str, Any]:
        """Get compliance scoring statistics"""
        status_counts = {}
        total_compliance = 0

        for score in self._scores.values():
            status_counts[score.status.value] = status_counts.get(
                score.status.value, 0
            ) + 1
            total_compliance += score.overall_compliance_score

        avg_compliance = total_compliance / len(self._scores) if self._scores else 0

        return {
            "models_scored": len(self._scores),
            "average_compliance": round(avg_compliance, 2),
            "by_status": status_counts,
            "total_controls": len(self._controls),
            "total_requirements": len(self._requirements),
        }


# Global compliance scorer
compliance_scorer: Optional[ComplianceRiskScorer] = None


def get_compliance_scorer() -> ComplianceRiskScorer:
    """Get the global compliance scorer"""
    global compliance_scorer
    if compliance_scorer is None:
        compliance_scorer = ComplianceRiskScorer()
    return compliance_scorer

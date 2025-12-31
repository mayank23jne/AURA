"""Policy Optimization Algorithms for AURA Platform"""

import asyncio
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class OptimizationGoal(str, Enum):
    """Optimization goals for policies"""
    MINIMIZE_CONFLICTS = "minimize_conflicts"
    MAXIMIZE_COVERAGE = "maximize_coverage"
    IMPROVE_EFFICIENCY = "improve_efficiency"
    BALANCE_ALL = "balance_all"
    REDUCE_COMPLEXITY = "reduce_complexity"
    ENHANCE_SECURITY = "enhance_security"


class ConflictType(str, Enum):
    """Types of policy conflicts"""
    CONTRADICTION = "contradiction"
    OVERLAP = "overlap"
    GAP = "gap"
    REDUNDANCY = "redundancy"
    INCONSISTENCY = "inconsistency"
    HIERARCHY_VIOLATION = "hierarchy_violation"


class PolicyRule(BaseModel):
    """A rule within a policy"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    condition: str
    action: str
    priority: int = 0
    scope: List[str] = Field(default_factory=list)
    enabled: bool = True


class Policy(BaseModel):
    """A governance policy"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    version: str = "1.0"
    rules: List[PolicyRule] = Field(default_factory=list)
    scope: List[str] = Field(default_factory=list)
    priority: int = 0
    enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class PolicyConflict(BaseModel):
    """A detected policy conflict"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conflict_type: ConflictType
    policy_ids: List[str]
    rule_ids: List[str] = Field(default_factory=list)
    description: str
    severity: str = "medium"  # low, medium, high, critical
    resolution_suggestions: List[str] = Field(default_factory=list)
    auto_resolvable: bool = False


class CoverageGap(BaseModel):
    """A gap in policy coverage"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    area: str
    description: str
    risk_level: str = "medium"
    suggested_rules: List[str] = Field(default_factory=list)


class OptimizationResult(BaseModel):
    """Result of policy optimization"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal: OptimizationGoal
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Analysis results
    conflicts_found: List[PolicyConflict] = Field(default_factory=list)
    coverage_gaps: List[CoverageGap] = Field(default_factory=list)
    redundancies: List[str] = Field(default_factory=list)

    # Scores
    original_score: float = 0.0
    optimized_score: float = 0.0
    improvement: float = 0.0

    # Recommendations
    recommendations: List[str] = Field(default_factory=list)
    auto_fixes_applied: int = 0
    manual_fixes_needed: int = 0

    # Optimized policies
    optimized_policies: List[Policy] = Field(default_factory=list)


class OptimizerConfig(BaseModel):
    """Configuration for policy optimizer"""
    enable_auto_fix: bool = False
    max_iterations: int = 100
    convergence_threshold: float = 0.01
    conflict_weight: float = 0.4
    coverage_weight: float = 0.3
    efficiency_weight: float = 0.2
    complexity_weight: float = 0.1


class PolicyOptimizer:
    """
    Policy optimization engine.

    Features:
    - Conflict detection and resolution
    - Coverage analysis
    - Redundancy elimination
    - Multi-objective optimization
    - Efficiency scoring
    - Automatic recommendations
    """

    def __init__(self, config: OptimizerConfig = None):
        self.config = config or OptimizerConfig()
        self._policies: Dict[str, Policy] = {}
        self._results: List[OptimizationResult] = []

        logger.info("PolicyOptimizer initialized")

    def add_policy(self, policy: Policy) -> str:
        """Add a policy to optimize"""
        self._policies[policy.id] = policy
        return policy.id

    def remove_policy(self, policy_id: str) -> bool:
        """Remove a policy"""
        if policy_id in self._policies:
            del self._policies[policy_id]
            return True
        return False

    async def optimize(
        self,
        goal: OptimizationGoal = OptimizationGoal.BALANCE_ALL,
        policy_ids: List[str] = None,
    ) -> OptimizationResult:
        """Run optimization on policies"""
        # Get policies to optimize
        if policy_ids:
            policies = [self._policies[pid] for pid in policy_ids if pid in self._policies]
        else:
            policies = list(self._policies.values())

        if not policies:
            return OptimizationResult(goal=goal)

        # Calculate original score
        original_score = self._calculate_score(policies, goal)

        # Detect conflicts
        conflicts = await self._detect_conflicts(policies)

        # Analyze coverage
        coverage_gaps = await self._analyze_coverage(policies)

        # Find redundancies
        redundancies = await self._find_redundancies(policies)

        # Generate recommendations
        recommendations = await self._generate_recommendations(
            policies, conflicts, coverage_gaps, redundancies
        )

        # Apply auto-fixes if enabled
        optimized_policies = policies.copy()
        auto_fixes = 0

        if self.config.enable_auto_fix:
            optimized_policies, auto_fixes = await self._apply_auto_fixes(
                policies, conflicts, redundancies
            )

        # Calculate optimized score
        optimized_score = self._calculate_score(optimized_policies, goal)

        result = OptimizationResult(
            goal=goal,
            conflicts_found=conflicts,
            coverage_gaps=coverage_gaps,
            redundancies=redundancies,
            original_score=round(original_score, 2),
            optimized_score=round(optimized_score, 2),
            improvement=round(optimized_score - original_score, 2),
            recommendations=recommendations,
            auto_fixes_applied=auto_fixes,
            manual_fixes_needed=len(conflicts) + len(coverage_gaps) - auto_fixes,
            optimized_policies=optimized_policies,
        )

        self._results.append(result)

        logger.info(
            "Policy optimization completed",
            goal=goal.value,
            original_score=original_score,
            optimized_score=optimized_score,
            conflicts=len(conflicts),
        )

        return result

    async def _detect_conflicts(self, policies: List[Policy]) -> List[PolicyConflict]:
        """Detect conflicts between policies"""
        conflicts = []

        # Check each pair of policies
        for i, policy1 in enumerate(policies):
            for policy2 in policies[i + 1:]:
                # Check for contradictions
                contradictions = self._find_contradictions(policy1, policy2)
                conflicts.extend(contradictions)

                # Check for overlaps
                overlaps = self._find_overlaps(policy1, policy2)
                conflicts.extend(overlaps)

                # Check for hierarchy violations
                violations = self._find_hierarchy_violations(policy1, policy2)
                conflicts.extend(violations)

        return conflicts

    def _find_contradictions(
        self, policy1: Policy, policy2: Policy
    ) -> List[PolicyConflict]:
        """Find contradictory rules between policies"""
        conflicts = []

        for rule1 in policy1.rules:
            for rule2 in policy2.rules:
                # Check if rules have overlapping scope
                if self._scopes_overlap(rule1.scope, rule2.scope):
                    # Check if conditions are similar but actions differ
                    if self._conditions_similar(rule1.condition, rule2.condition):
                        if self._actions_contradict(rule1.action, rule2.action):
                            conflicts.append(PolicyConflict(
                                conflict_type=ConflictType.CONTRADICTION,
                                policy_ids=[policy1.id, policy2.id],
                                rule_ids=[rule1.id, rule2.id],
                                description=(
                                    f"Rules '{rule1.name}' and '{rule2.name}' have "
                                    f"contradictory actions for similar conditions"
                                ),
                                severity="high",
                                resolution_suggestions=[
                                    f"Review and align actions of '{rule1.name}' and '{rule2.name}'",
                                    "Consider merging into a single rule with clear priority",
                                    "Add explicit scope differentiation",
                                ],
                            ))

        return conflicts

    def _find_overlaps(
        self, policy1: Policy, policy2: Policy
    ) -> List[PolicyConflict]:
        """Find overlapping rules"""
        conflicts = []

        for rule1 in policy1.rules:
            for rule2 in policy2.rules:
                if rule1.id == rule2.id:
                    continue

                # Check for complete overlap
                if (
                    self._scopes_overlap(rule1.scope, rule2.scope)
                    and self._conditions_similar(rule1.condition, rule2.condition)
                    and self._actions_similar(rule1.action, rule2.action)
                ):
                    conflicts.append(PolicyConflict(
                        conflict_type=ConflictType.OVERLAP,
                        policy_ids=[policy1.id, policy2.id],
                        rule_ids=[rule1.id, rule2.id],
                        description=(
                            f"Rules '{rule1.name}' and '{rule2.name}' significantly overlap"
                        ),
                        severity="low",
                        resolution_suggestions=[
                            "Consider consolidating into a single rule",
                            "Differentiate scope or conditions",
                        ],
                        auto_resolvable=True,
                    ))

        return conflicts

    def _find_hierarchy_violations(
        self, policy1: Policy, policy2: Policy
    ) -> List[PolicyConflict]:
        """Find hierarchy violations between policies"""
        conflicts = []

        # Check if lower priority policy overrides higher priority
        if policy1.priority < policy2.priority:  # Higher number = lower priority
            for rule1 in policy1.rules:
                for rule2 in policy2.rules:
                    if (
                        self._scopes_overlap(rule1.scope, rule2.scope)
                        and self._conditions_similar(rule1.condition, rule2.condition)
                        and rule1.priority > rule2.priority
                    ):
                        conflicts.append(PolicyConflict(
                            conflict_type=ConflictType.HIERARCHY_VIOLATION,
                            policy_ids=[policy1.id, policy2.id],
                            rule_ids=[rule1.id, rule2.id],
                            description=(
                                f"Rule in lower priority policy overrides higher priority policy"
                            ),
                            severity="medium",
                            resolution_suggestions=[
                                "Adjust rule priorities to match policy hierarchy",
                                "Move rule to appropriate policy level",
                            ],
                        ))

        return conflicts

    async def _analyze_coverage(self, policies: List[Policy]) -> List[CoverageGap]:
        """Analyze policy coverage for gaps"""
        gaps = []

        # Define expected coverage areas
        expected_areas = [
            "data_privacy",
            "model_fairness",
            "transparency",
            "security",
            "accountability",
            "human_oversight",
            "error_handling",
            "audit_logging",
        ]

        # Find covered areas
        covered_areas: Set[str] = set()
        for policy in policies:
            for rule in policy.rules:
                covered_areas.update(rule.scope)

        # Identify gaps
        for area in expected_areas:
            if area not in covered_areas:
                gaps.append(CoverageGap(
                    area=area,
                    description=f"No policy rules cover {area}",
                    risk_level="high" if area in ["data_privacy", "security"] else "medium",
                    suggested_rules=[
                        f"Add rule for {area} monitoring",
                        f"Define {area} compliance requirements",
                    ],
                ))

        return gaps

    async def _find_redundancies(self, policies: List[Policy]) -> List[str]:
        """Find redundant rules across policies"""
        redundancies = []
        seen_rules: Dict[str, Tuple[str, str]] = {}  # (condition, action) -> (policy_id, rule_id)

        for policy in policies:
            for rule in policy.rules:
                key = (rule.condition.lower().strip(), rule.action.lower().strip())
                if key in seen_rules:
                    orig_policy, orig_rule = seen_rules[key]
                    redundancies.append(
                        f"Rule '{rule.name}' in policy {policy.id} is redundant "
                        f"with rule in policy {orig_policy}"
                    )
                else:
                    seen_rules[key] = (policy.id, rule.id)

        return redundancies

    async def _generate_recommendations(
        self,
        policies: List[Policy],
        conflicts: List[PolicyConflict],
        gaps: List[CoverageGap],
        redundancies: List[str],
    ) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        # Conflict-based recommendations
        if len(conflicts) > 5:
            recommendations.append(
                "High number of conflicts detected. Consider consolidating policies."
            )

        critical_conflicts = [c for c in conflicts if c.severity == "critical"]
        if critical_conflicts:
            recommendations.append(
                f"Address {len(critical_conflicts)} critical conflicts immediately."
            )

        # Coverage-based recommendations
        high_risk_gaps = [g for g in gaps if g.risk_level == "high"]
        if high_risk_gaps:
            areas = ", ".join(g.area for g in high_risk_gaps)
            recommendations.append(
                f"High-risk coverage gaps in: {areas}. Add policies urgently."
            )

        # Redundancy-based recommendations
        if len(redundancies) > len(policies):
            recommendations.append(
                "Significant policy redundancy. Consider consolidation to improve maintainability."
            )

        # Efficiency recommendations
        total_rules = sum(len(p.rules) for p in policies)
        if total_rules > 100:
            recommendations.append(
                f"High rule count ({total_rules}). Consider grouping related rules."
            )

        # General recommendations
        if not recommendations:
            recommendations.append("Policies are well-optimized. Continue regular reviews.")

        return recommendations

    async def _apply_auto_fixes(
        self,
        policies: List[Policy],
        conflicts: List[PolicyConflict],
        redundancies: List[str],
    ) -> Tuple[List[Policy], int]:
        """Apply automatic fixes to policies"""
        fixed_policies = [p.model_copy(deep=True) for p in policies]
        fixes_applied = 0

        # Fix auto-resolvable conflicts
        for conflict in conflicts:
            if conflict.auto_resolvable:
                if conflict.conflict_type == ConflictType.OVERLAP:
                    # Remove duplicate rule from lower priority policy
                    if len(conflict.rule_ids) >= 2:
                        rule_to_remove = conflict.rule_ids[1]
                        for policy in fixed_policies:
                            policy.rules = [
                                r for r in policy.rules
                                if r.id != rule_to_remove
                            ]
                        fixes_applied += 1

        return fixed_policies, fixes_applied

    def _calculate_score(
        self, policies: List[Policy], goal: OptimizationGoal
    ) -> float:
        """Calculate optimization score for policies"""
        scores = {
            "conflict": self._conflict_score(policies),
            "coverage": self._coverage_score(policies),
            "efficiency": self._efficiency_score(policies),
            "complexity": self._complexity_score(policies),
        }

        if goal == OptimizationGoal.MINIMIZE_CONFLICTS:
            return scores["conflict"] * 100
        elif goal == OptimizationGoal.MAXIMIZE_COVERAGE:
            return scores["coverage"] * 100
        elif goal == OptimizationGoal.IMPROVE_EFFICIENCY:
            return scores["efficiency"] * 100
        elif goal == OptimizationGoal.REDUCE_COMPLEXITY:
            return scores["complexity"] * 100
        else:  # BALANCE_ALL
            return (
                scores["conflict"] * self.config.conflict_weight
                + scores["coverage"] * self.config.coverage_weight
                + scores["efficiency"] * self.config.efficiency_weight
                + scores["complexity"] * self.config.complexity_weight
            ) * 100

    def _conflict_score(self, policies: List[Policy]) -> float:
        """Calculate conflict-free score (higher is better)"""
        # Simplified: count potential conflict pairs
        total_rule_pairs = 0
        conflict_pairs = 0

        all_rules = [r for p in policies for r in p.rules]
        for i, rule1 in enumerate(all_rules):
            for rule2 in all_rules[i + 1:]:
                total_rule_pairs += 1
                if self._scopes_overlap(rule1.scope, rule2.scope):
                    if self._conditions_similar(rule1.condition, rule2.condition):
                        conflict_pairs += 1

        if total_rule_pairs == 0:
            return 1.0

        return 1.0 - (conflict_pairs / total_rule_pairs)

    def _coverage_score(self, policies: List[Policy]) -> float:
        """Calculate coverage score (higher is better)"""
        expected_areas = {
            "data_privacy", "model_fairness", "transparency",
            "security", "accountability", "human_oversight",
            "error_handling", "audit_logging"
        }

        covered = set()
        for policy in policies:
            for rule in policy.rules:
                covered.update(rule.scope)

        return len(covered & expected_areas) / len(expected_areas)

    def _efficiency_score(self, policies: List[Policy]) -> float:
        """Calculate efficiency score (higher is better)"""
        total_rules = sum(len(p.rules) for p in policies)
        enabled_rules = sum(
            1 for p in policies for r in p.rules if r.enabled
        )

        if total_rules == 0:
            return 1.0

        # Penalize for too many rules or disabled rules
        efficiency = enabled_rules / max(total_rules, 1)

        # Penalize for excessive rules
        if total_rules > 50:
            efficiency *= 50 / total_rules

        return min(1.0, efficiency)

    def _complexity_score(self, policies: List[Policy]) -> float:
        """Calculate simplicity score (higher is better)"""
        # Lower complexity is better
        total_rules = sum(len(p.rules) for p in policies)
        avg_condition_length = 0

        conditions = [r.condition for p in policies for r in p.rules]
        if conditions:
            avg_condition_length = sum(len(c) for c in conditions) / len(conditions)

        # Score based on rule count and condition complexity
        complexity = (total_rules / 100) + (avg_condition_length / 200)
        return max(0, 1.0 - min(1.0, complexity))

    def _scopes_overlap(self, scope1: List[str], scope2: List[str]) -> bool:
        """Check if two scopes overlap"""
        if not scope1 or not scope2:
            return True  # Empty scope means global
        return bool(set(scope1) & set(scope2))

    def _conditions_similar(self, cond1: str, cond2: str) -> bool:
        """Check if two conditions are similar"""
        # Simplified similarity check
        c1 = cond1.lower().strip()
        c2 = cond2.lower().strip()

        if c1 == c2:
            return True

        # Check for significant overlap in words
        words1 = set(c1.split())
        words2 = set(c2.split())

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2) / max(len(words1), len(words2))
        return overlap > 0.7

    def _actions_contradict(self, action1: str, action2: str) -> bool:
        """Check if two actions contradict each other"""
        a1 = action1.lower()
        a2 = action2.lower()

        contradicting_pairs = [
            ("allow", "deny"),
            ("enable", "disable"),
            ("require", "prohibit"),
            ("accept", "reject"),
            ("approve", "deny"),
        ]

        for word1, word2 in contradicting_pairs:
            if (word1 in a1 and word2 in a2) or (word2 in a1 and word1 in a2):
                return True

        return False

    def _actions_similar(self, action1: str, action2: str) -> bool:
        """Check if two actions are similar"""
        return action1.lower().strip() == action2.lower().strip()

    def get_results(self) -> List[OptimizationResult]:
        """Get optimization results"""
        return self._results

    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics"""
        if not self._results:
            return {"total_optimizations": 0}

        latest = self._results[-1]
        improvements = [r.improvement for r in self._results]

        return {
            "total_optimizations": len(self._results),
            "total_policies": len(self._policies),
            "latest_score": latest.optimized_score,
            "average_improvement": sum(improvements) / len(improvements),
            "total_conflicts_resolved": sum(r.auto_fixes_applied for r in self._results),
        }


# Global optimizer instance
policy_optimizer: Optional[PolicyOptimizer] = None


def get_policy_optimizer() -> PolicyOptimizer:
    """Get the global policy optimizer"""
    global policy_optimizer
    if policy_optimizer is None:
        policy_optimizer = PolicyOptimizer()
    return policy_optimizer

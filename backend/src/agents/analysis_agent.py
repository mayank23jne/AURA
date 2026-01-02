"""Analysis Agent - Deep analysis and insight generation"""

import json
from datetime import datetime
from typing import Any, Dict, List

import structlog
from langchain_core.tools import Tool

from src.core.base_agent import BaseAgent
from src.core.models import (
    AgentConfig,
    KnowledgeItem,
    KnowledgeType,
    TaskRequest,
    TaskResponse,
)

logger = structlog.get_logger()


class AnalysisAgent(BaseAgent):
    """
    Agent specialized in analyzing test results and generating insights.

    Capabilities:
    - Pattern recognition across audits
    - Root cause analysis
    - Trend prediction
    - Anomaly detection
    - Compliance risk scoring
    """

    print('Analysis Agent')

    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                name="analysis",
                llm_model="gpt-4",
                llm_provider="openai",
            )
        super().__init__(config)

        self._analysis_cache: Dict[str, Dict[str, Any]] = {}
        self._patterns: List[Dict[str, Any]] = []

        logger.info("AnalysisAgent initialized", agent_id=self.id)

    def _init_tools(self) -> List[Any]:
        """Initialize analysis-specific tools"""
        return [
            Tool(
                name="analyze_patterns",
                func=self._analyze_patterns_sync,
                description="Analyze patterns in audit results",
            ),
            Tool(
                name="detect_anomalies",
                func=self._detect_anomalies_sync,
                description="Detect anomalies in results",
            ),
            Tool(
                name="predict_trends",
                func=self._predict_trends_sync,
                description="Predict compliance trends",
            ),
        ]

    async def process_task(self, task: TaskRequest) -> TaskResponse:
        """Process analysis-related tasks"""
        start_time = datetime.utcnow()

        try:
            if task.task_type == "analyze_results":
                result = await self._analyze_results(task.parameters)
            elif task.task_type == "root_cause_analysis":
                result = await self._root_cause_analysis(task.parameters)
            elif task.task_type == "detect_patterns":
                result = await self._detect_patterns(task.parameters)
            elif task.task_type == "predict_trends":
                result = await self._predict_trends(task.parameters)
            elif task.task_type == "anomaly_detection":
                result = await self._detect_anomalies(task.parameters)
            elif task.task_type == "comparative_analysis":
                result = await self._comparative_analysis(task.parameters)
            elif task.task_type == "risk_scoring":
                result = await self._calculate_risk_score(task.parameters)
            else:
                result = await self._handle_generic_task(task)

            self.metrics.tasks_completed += 1

            return TaskResponse(
                task_id=task.task_id,
                status="success",
                result=result,
                agent_id=self.id,
                execution_time_ms=int(
                    (datetime.utcnow() - start_time).total_seconds() * 1000
                ),
            )

        except Exception as e:
            self.metrics.tasks_failed += 1
            logger.error("Task failed", task_id=task.task_id, error=str(e))

            return TaskResponse(
                task_id=task.task_id,
                status="failure",
                error=str(e),
                agent_id=self.id,
                execution_time_ms=int(
                    (datetime.utcnow() - start_time).total_seconds() * 1000
                ),
            )

    async def _analyze_results(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis of audit results"""
        audit_id = params.get("audit_id")
        results = params.get("results", [])
        model_id = params.get("model_id")

        prompt = f"""Perform comprehensive analysis of these audit results:

Audit ID: {audit_id}
Model: {model_id}
Total Results: {len(results)}
Results Summary: {json.dumps(results[:20])}  # First 20 for context

Analyze:
1. Overall compliance status
2. Failure patterns and clusters
3. Severity distribution
4. Policy-specific insights
5. Performance trends
6. Risk areas

Provide:
- Key findings (top 5)
- Critical issues requiring immediate attention
- Recommendations for improvement
- Confidence scores for each finding

Return as structured JSON."""

        response = await self.invoke_llm(prompt)

        # Store analysis for learning
        await self.learn({
            "type": "audit_analysis",
            "domain": "compliance",
            "audit_id": audit_id,
            "key_findings": response,
        })

        return {
            "audit_id": audit_id,
            "model_id": model_id,
            "analysis": response,
            "results_analyzed": len(results),
        }

    async def _root_cause_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform root cause analysis on failures"""
        failures = params.get("failures", [])
        context = params.get("context", {})

        prompt = f"""Perform root cause analysis on these compliance failures:

Failures: {json.dumps(failures)}
Context: {json.dumps(context)}

Apply systematic analysis:
1. Identify immediate causes
2. Trace to underlying root causes
3. Map causal chains
4. Identify systemic issues
5. Determine contributing factors

Use techniques:
- 5 Whys
- Fishbone/Ishikawa
- Fault tree analysis

Return:
- Root causes ranked by impact
- Causal relationships
- Evidence supporting each cause
- Remediation suggestions per root cause"""

        response = await self.invoke_llm(prompt)

        return {
            "failures_analyzed": len(failures),
            "root_cause_analysis": response,
        }

    async def _detect_patterns(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Detect patterns across multiple audits"""
        audit_data = params.get("audit_data", [])
        pattern_types = params.get("pattern_types", ["failure", "success", "temporal"])

        prompt = f"""Detect patterns in this audit data:

Data Points: {len(audit_data)}
Sample Data: {json.dumps(audit_data[:10])}
Pattern Types to Find: {pattern_types}

Identify:
1. Recurring failure patterns
2. Success patterns to replicate
3. Temporal patterns (time-based)
4. Correlation patterns
5. Policy-specific patterns
6. Model behavior patterns

For each pattern:
- Description
- Frequency/strength
- Affected policies/models
- Statistical significance
- Actionable insight

Return as JSON array of patterns."""

        response = await self.invoke_llm(prompt)

        # Store discovered patterns
        try:
            patterns = json.loads(response)
            self._patterns.extend(patterns if isinstance(patterns, list) else [patterns])
        except json.JSONDecodeError:
            pass

        return {
            "patterns_found": response,
            "data_points_analyzed": len(audit_data),
        }

    async def _predict_trends(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Predict future compliance trends"""
        historical_data = params.get("historical_data", [])
        prediction_horizon = params.get("horizon_days", 30)

        prompt = f"""Predict compliance trends based on historical data:

Historical Data Points: {len(historical_data)}
Sample: {json.dumps(historical_data[:10])}
Prediction Horizon: {prediction_horizon} days

Predict:
1. Overall compliance score trend
2. Likely failure areas
3. Risk emergence predictions
4. Policy effectiveness trends
5. Resource requirement forecasts

Include:
- Point predictions with confidence intervals
- Key trend drivers
- Early warning indicators
- Recommended preemptive actions"""

        response = await self.invoke_llm(prompt)

        return {
            "predictions": response,
            "horizon_days": prediction_horizon,
            "based_on_data_points": len(historical_data),
        }

    async def _detect_anomalies(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in audit results"""
        data = params.get("data", [])
        sensitivity = params.get("sensitivity", "medium")

        prompt = f"""Detect anomalies in this compliance data:

Data Points: {len(data)}
Sample: {json.dumps(data[:10])}
Sensitivity: {sensitivity}

Identify:
1. Statistical outliers
2. Behavioral anomalies
3. Pattern deviations
4. Unexpected results
5. Potential data quality issues

For each anomaly:
- Type and description
- Anomaly score
- Potential causes
- Recommended investigation steps
- Risk assessment"""

        response = await self.invoke_llm(prompt)

        return {
            "anomalies": response,
            "data_points_analyzed": len(data),
            "sensitivity": sensitivity,
        }

    async def _comparative_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Compare results across models or time periods"""
        comparison_type = params.get("comparison_type", "models")
        dataset_a = params.get("dataset_a", [])
        dataset_b = params.get("dataset_b", [])
        labels = params.get("labels", ["A", "B"])

        prompt = f"""Perform comparative analysis:

Comparison Type: {comparison_type}
{labels[0]} Data Points: {len(dataset_a)}
{labels[1]} Data Points: {len(dataset_b)}

Compare:
1. Overall performance metrics
2. Policy-specific differences
3. Failure pattern differences
4. Strength and weakness areas
5. Statistical significance of differences

Provide:
- Side-by-side metrics
- Key differences with impact assessment
- Recommendations based on comparison
- Visualization suggestions"""

        response = await self.invoke_llm(prompt)

        return {
            "comparison_type": comparison_type,
            "labels": labels,
            "comparison_results": response,
        }

    async def _calculate_risk_score(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate compliance risk score"""
        model_id = params.get("model_id")
        audit_results = params.get("audit_results", [])
        historical_data = params.get("historical_data", [])

        prompt = f"""Calculate compliance risk score:

Model: {model_id}
Recent Audit Results: {len(audit_results)} tests
Historical Data Points: {len(historical_data)}

Calculate risk score (0-100) considering:
1. Current compliance failures
2. Severity of issues
3. Historical trends
4. Pattern-based risks
5. External risk factors

Provide:
- Overall risk score
- Risk breakdown by category
- Key risk drivers
- Risk trajectory (improving/stable/worsening)
- Mitigation priorities"""

        response = await self.invoke_llm(prompt)

        return {
            "model_id": model_id,
            "risk_assessment": response,
        }

    async def _handle_generic_task(self, task: TaskRequest) -> Dict[str, Any]:
        """Handle generic analysis tasks"""
        prompt = f"""Perform analysis for this task:

Task Type: {task.task_type}
Description: {task.description}
Parameters: {task.parameters}

Provide detailed analytical output."""

        response = await self.invoke_llm(prompt)
        return {"analysis": response}

    async def _update_strategies(self, experience: Dict[str, Any]):
        """Update analysis strategies based on learning"""
        if experience.get("type") == "audit_analysis":
            # Store analytical patterns
            if self.knowledge_base:
                await self.knowledge_base.store(
                    KnowledgeItem(
                        id=f"analysis_{datetime.utcnow().timestamp()}",
                        knowledge_type=KnowledgeType.INSIGHT,
                        domain="analysis",
                        content=experience,
                        source_agent=self.name,
                        confidence=0.85,
                    )
                )

    def _analyze_patterns_sync(self, data: str) -> str:
        return "Patterns analyzed"

    def _detect_anomalies_sync(self, data: str) -> str:
        return "Anomalies detected"

    def _predict_trends_sync(self, data: str) -> str:
        return "Trends predicted"

    def get_discovered_patterns(self) -> List[Dict[str, Any]]:
        """Get all discovered patterns"""
        return self._patterns

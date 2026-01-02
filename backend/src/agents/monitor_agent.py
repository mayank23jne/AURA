"""Monitor Agent - Continuous compliance monitoring"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List

import structlog
from langchain_core.tools import Tool

from src.core.base_agent import BaseAgent
from src.core.models import (
    AgentConfig,
    KnowledgeItem,
    KnowledgeType,
    MessageType,
    TaskRequest,
    TaskResponse,
)
from src.infrastructure.event_stream import EventType

logger = structlog.get_logger()
from src.db import SessionLocal
from src.models.orm import ORMAlert, ORMMetric


class MonitorAgent(BaseAgent):
    """
    Agent for continuous compliance monitoring.

    Capabilities:
    - Real-time model behavior tracking
    - Drift detection
    - Incident detection and alerting
    - Performance degradation monitoring
    - Compliance metric tracking
    """
    print('Monitor Agent')
    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                name="monitor",
                llm_model="gpt-4",
                llm_provider="openai",
            )
        super().__init__(config)

        self._monitored_models: Dict[str, Dict[str, Any]] = {}
        self._alerts: List[Dict[str, Any]] = []
        self._metrics_history: Dict[str, List[Dict[str, Any]]] = {}
        self._monitoring = False

        # Load from DB
        self._load_from_db()

        logger.info("MonitorAgent initialized with Database backend", agent_id=self.id)
    
    def _load_from_db(self):
        """Load alerts and metrics from database"""
        db = SessionLocal()
        try:
            # Load alerts
            alerts = db.query(ORMAlert).filter(ORMAlert.status == "open").all()
            for alert in alerts:
                self._alerts.append(alert.to_dict())
            
            # Load recent metrics (simplified)
            # In a real system we might not load ALL metrics, just recent ones for context
            pass
        except Exception as e:
            logger.error("Failed to load monitor data", error=str(e))
        finally:
            db.close()

    def _init_tools(self) -> List[Any]:
        """Initialize monitoring-specific tools"""
        return [
            Tool(
                name="check_drift",
                func=self._check_drift_sync,
                description="Check for compliance drift",
            ),
            Tool(
                name="analyze_metrics",
                func=self._analyze_metrics_sync,
                description="Analyze monitoring metrics",
            ),
        ]

    async def process_task(self, task: TaskRequest) -> TaskResponse:
        """Process monitoring-related tasks"""
        start_time = datetime.utcnow()

        try:
            if task.task_type == "start_monitoring":
                result = await self._start_monitoring(task.parameters)
            elif task.task_type == "check_compliance":
                result = await self._check_compliance(task.parameters)
            elif task.task_type == "detect_drift":
                result = await self._detect_drift(task.parameters)
            elif task.task_type == "analyze_behavior":
                result = await self._analyze_behavior(task.parameters)
            elif task.task_type == "generate_alert":
                result = await self._generate_alert(task.parameters)
            elif task.task_type == "predict_issues":
                result = await self._predict_issues(task.parameters)
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

    async def _start_monitoring(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Start monitoring a model"""
        model_id = params.get("model_id")
        metrics = params.get("metrics", ["compliance_score", "response_time", "error_rate"])
        thresholds = params.get("thresholds", {})

        self._monitored_models[model_id] = {
            "start_time": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "thresholds": thresholds,
            "status": "active",
        }

        self._metrics_history[model_id] = []

        logger.info("Started monitoring", model_id=model_id, metrics=metrics)

        return {
            "model_id": model_id,
            "status": "monitoring_started",
            "metrics": metrics,
        }

    async def _check_compliance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check current compliance status"""
        model_id = params.get("model_id")

        if model_id not in self._monitored_models:
            return {"error": f"Model {model_id} not being monitored"}

        # Simulate compliance check
        prompt = f"""Evaluate current compliance status for model {model_id}:

Monitoring Data: {len(self._metrics_history.get(model_id, []))} data points
Thresholds: {self._monitored_models[model_id].get('thresholds', {})}

Assess:
1. Current compliance score
2. Trend (improving/stable/degrading)
3. Risk indicators
4. Areas of concern

Return JSON with status, score, and recommendations."""

        response = await self.invoke_llm(prompt)

        return {
            "model_id": model_id,
            "compliance_status": response,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def _detect_drift(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Detect compliance drift"""
        model_id = params.get("model_id")
        baseline = params.get("baseline", {})
        current = params.get("current", {})

        prompt = f"""Detect compliance drift for model {model_id}:

Baseline Metrics: {baseline}
Current Metrics: {current}

Analyze:
1. Statistical drift in key metrics
2. Behavioral changes
3. Performance degradation
4. Pattern deviations

For each drift detected:
- Metric affected
- Drift magnitude
- Severity (low/medium/high/critical)
- Likely cause
- Recommended action

Return as JSON."""

        response = await self.invoke_llm(prompt)

        # Check if alert needed
        drift_detected = "drift" in response.lower() and "critical" in response.lower()

        if drift_detected:
            await self._generate_alert({
                "model_id": model_id,
                "type": "drift",
                "severity": "high",
                "details": response,
            })

        return {
            "model_id": model_id,
            "drift_analysis": response,
            "alert_generated": drift_detected,
        }

    async def _analyze_behavior(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model behavior patterns"""
        model_id = params.get("model_id")
        behavior_data = params.get("behavior_data", [])

        prompt = f"""Analyze behavior patterns for model {model_id}:

Behavior Data Points: {len(behavior_data)}
Sample: {behavior_data[:10] if behavior_data else 'No data'}

Analyze:
1. Response patterns
2. Error patterns
3. Performance patterns
4. Compliance patterns
5. Anomalies

Identify:
- Normal behavior baseline
- Deviations from baseline
- Concerning patterns
- Positive patterns to maintain"""

        response = await self.invoke_llm(prompt)

        return {
            "model_id": model_id,
            "behavior_analysis": response,
        }

    async def _generate_alert(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate and send an alert"""
        model_id = params.get("model_id")
        alert_type = params.get("type", "general")
        severity = params.get("severity", "medium")
        details = params.get("details", "")

        alert = {
            "id": f"alert_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "model_id": model_id,
            "type": alert_type,
            "severity": severity,
            "details": details,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "open",
        }

        self._alerts.append(alert)

        # Persist alert
        db = SessionLocal()
        try:
            orm_alert = ORMAlert(
                id=alert["id"],
                model_id=model_id,
                type=alert_type,
                severity=severity,
                details=str(details) if isinstance(details, (dict, list)) else details,
                status="open",
                timestamp=datetime.utcnow()
            )
            db.add(orm_alert)
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error("Failed to persist alert", error=str(e))
        finally:
            db.close()

        # Notify orchestrator
        if self.message_queue:
            await self.collaborate(
                "orchestrator",
                MessageType.ALERT,
                alert,
                priority=8 if severity == "critical" else 5,
            )

        logger.warning(
            "Alert generated",
            alert_id=alert["id"],
            model_id=model_id,
            severity=severity,
        )

        return alert

    async def _predict_issues(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Predict upcoming compliance issues"""
        model_id = params.get("model_id")
        horizon_hours = params.get("horizon_hours", 24)

        history = self._metrics_history.get(model_id, [])

        prompt = f"""Predict compliance issues for model {model_id}:

Historical Data Points: {len(history)}
Prediction Horizon: {horizon_hours} hours

Predict:
1. Likely issues to occur
2. Probability and timing
3. Expected severity
4. Early warning signs to watch
5. Preventive actions

Return predictions with confidence scores."""

        response = await self.invoke_llm(prompt)

        return {
            "model_id": model_id,
            "predictions": response,
            "horizon_hours": horizon_hours,
        }

    async def _handle_generic_task(self, task: TaskRequest) -> Dict[str, Any]:
        """Handle generic monitoring tasks"""
        prompt = f"""Process this monitoring task:

Task Type: {task.task_type}
Description: {task.description}
Parameters: {task.parameters}"""

        response = await self.invoke_llm(prompt)
        return {"response": response}

    async def _update_strategies(self, experience: Dict[str, Any]):
        """Update monitoring strategies"""
        pass

    def _check_drift_sync(self, model_id: str) -> str:
        return "Drift checked"

    def _analyze_metrics_sync(self, metrics: str) -> str:
        return "Metrics analyzed"

    def get_alerts(self, status: str = None) -> List[Dict[str, Any]]:
        """Get alerts, optionally filtered by status"""
        if status:
            return [a for a in self._alerts if a["status"] == status]
        return self._alerts

    def get_monitored_models(self) -> List[str]:
        """Get list of monitored models"""
        return list(self._monitored_models.keys())

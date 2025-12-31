"""Report Agent - Intelligent reporting and communication"""

import json
from datetime import datetime
from typing import Any, Dict, List

import structlog
from langchain_core.tools import Tool

from src.core.base_agent import BaseAgent
from src.core.models import AgentConfig, TaskRequest, TaskResponse

logger = structlog.get_logger()


class ReportAgent(BaseAgent):
    """
    Agent for intelligent reporting and communication.

    Capabilities:
    - Natural language report generation
    - Executive summary creation
    - Visualization suggestions
    - Stakeholder-specific formatting
    - Insight prioritization
    """
    print('Report Agent')
    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                name="report",
                llm_model="gpt-4",
                llm_provider="openai",
            )
        super().__init__(config)

        self._report_templates: Dict[str, str] = {}
        self._generated_reports: List[Dict[str, Any]] = []

        logger.info("ReportAgent initialized", agent_id=self.id)

    def _init_tools(self) -> List[Any]:
        """Initialize report-specific tools"""
        return [
            Tool(
                name="generate_summary",
                func=self._generate_summary_sync,
                description="Generate executive summary",
            ),
            Tool(
                name="format_report",
                func=self._format_report_sync,
                description="Format report for specific audience",
            ),
        ]

    async def process_task(self, task: TaskRequest) -> TaskResponse:
        """Process reporting tasks"""
        start_time = datetime.utcnow()
        print('report processtask', task)
        try:
            if task.task_type == "generate_audit_report":
                result = await self._generate_audit_report(task.parameters)
            elif task.task_type == "executive_summary":
                result = await self._generate_executive_summary(task.parameters)
            elif task.task_type == "technical_report":
                result = await self._generate_technical_report(task.parameters)
            elif task.task_type == "compliance_dashboard":
                result = await self._generate_dashboard_data(task.parameters)
            elif task.task_type == "trend_narrative":
                result = await self._generate_trend_narrative(task.parameters)
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

    async def _generate_audit_report(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        audit_id = params.get("audit_id")
        results = params.get("results", {})
        analysis = params.get("analysis", {})
        audience = params.get("audience", "general")

        prompt = f"""Generate a comprehensive audit report:

Audit ID: {audit_id}
Audience: {audience}
Results Summary: {json.dumps(results)}
Analysis: {json.dumps(analysis)}

Structure the report with:
1. Executive Summary (2-3 sentences)
2. Key Findings (bullet points)
3. Compliance Score and Trend
4. Critical Issues Requiring Action
5. Detailed Results by Policy
6. Recommendations (prioritized)
7. Next Steps

Adjust language and detail level for {audience} audience.
Format in Markdown."""

        response = await self.invoke_llm(prompt)

        report = {
            "audit_id": audit_id,
            "generated_at": datetime.utcnow().isoformat(),
            "audience": audience,
            "content": response,
            "format": "markdown",
        }

        self._generated_reports.append(report)

        return report

    async def _generate_executive_summary(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary"""
        data = params.get("data", {})
        max_length = params.get("max_length", 500)

        prompt = f"""Generate an executive summary for C-level leadership:

Data: {json.dumps(data)}
Maximum Length: {max_length} words

Requirements:
1. Lead with the most critical insight
2. Include compliance score and trend
3. Highlight business impact
4. Summarize key risks
5. Provide clear action items

Use clear, non-technical language.
Be concise but comprehensive."""

        response = await self.invoke_llm(prompt)

        return {
            "summary": response,
            "word_count": len(response.split()),
        }

    async def _generate_technical_report(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed technical report"""
        audit_id = params.get("audit_id")
        results = params.get("results", [])
        include_raw = params.get("include_raw_data", False)

        prompt = f"""Generate a detailed technical report for engineering teams:

Audit ID: {audit_id}
Total Results: {len(results)}
Results Sample: {json.dumps(results[:20])}

Include:
1. Technical methodology
2. Test coverage analysis
3. Failure categorization with stack traces/logs
4. Performance metrics
5. Root cause analysis details
6. Remediation code examples
7. Integration points affected
8. API/endpoint specific issues

Use technical terminology appropriate for engineers.
Include code snippets where relevant."""

        response = await self.invoke_llm(prompt)

        return {
            "audit_id": audit_id,
            "technical_report": response,
            "format": "markdown",
        }

    async def _generate_dashboard_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data for compliance dashboard"""
        time_range = params.get("time_range_days", 30)
        models = params.get("models", [])

        prompt = f"""Generate dashboard data structure for compliance monitoring:

Time Range: {time_range} days
Models: {models if models else 'All models'}

Generate JSON data for:
1. Overall compliance score (current and trend)
2. Score breakdown by policy category
3. Issue count by severity
4. Top 5 failing policies
5. Model comparison matrix
6. Timeline of compliance scores
7. Alert summary
8. Upcoming audit schedule

Return as structured JSON suitable for chart rendering."""

        response = await self.invoke_llm(prompt)

        return {
            "dashboard_data": response,
            "time_range_days": time_range,
        }

    async def _generate_trend_narrative(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate narrative explanation of trends"""
        trend_data = params.get("trend_data", [])
        focus_areas = params.get("focus_areas", [])

        prompt = f"""Generate a narrative analysis of compliance trends:

Trend Data: {json.dumps(trend_data)}
Focus Areas: {focus_areas}

Write a clear narrative that:
1. Describes the overall trajectory
2. Explains key inflection points
3. Identifies driving factors
4. Compares to industry benchmarks
5. Projects future direction
6. Recommends actions based on trends

Make it engaging and actionable."""

        response = await self.invoke_llm(prompt)

        return {
            "narrative": response,
            "focus_areas": focus_areas,
        }

    async def _handle_generic_task(self, task: TaskRequest) -> Dict[str, Any]:
        """Handle generic reporting tasks"""
        prompt = f"""Generate report content for:

Task Type: {task.task_type}
Description: {task.description}
Parameters: {task.parameters}"""

        response = await self.invoke_llm(prompt)
        return {"content": response}

    async def _update_strategies(self, experience: Dict[str, Any]):
        """Update reporting strategies"""
        pass

    def _generate_summary_sync(self, data: str) -> str:
        return "Summary generated"

    def _format_report_sync(self, report: str) -> str:
        return "Report formatted"

    def get_reports(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get generated reports"""
        return self._generated_reports[-limit:]

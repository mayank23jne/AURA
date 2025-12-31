"""Auto-Remediation Engine for AURA Platform"""

import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import uuid

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class RemediationStatus(str, Enum):
    """Status of a remediation action"""
    PENDING = "pending"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class RemediationType(str, Enum):
    """Types of remediation actions"""
    CONFIGURATION = "configuration"
    MODEL_UPDATE = "model_update"
    POLICY_UPDATE = "policy_update"
    DATA_CORRECTION = "data_correction"
    ACCESS_CONTROL = "access_control"
    ALERT_RESPONSE = "alert_response"
    COMPLIANCE_FIX = "compliance_fix"
    PERFORMANCE_TUNING = "performance_tuning"


class ApprovalLevel(str, Enum):
    """Approval levels for remediation"""
    AUTO = "auto"
    REVIEW = "review"
    APPROVE = "approve"
    DUAL_APPROVE = "dual_approve"


class RemediationAction(BaseModel):
    """Definition of a remediation action"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    action_type: RemediationType
    handler: str = ""  # Handler function name

    # Trigger conditions
    trigger_conditions: List[str] = Field(default_factory=list)
    issue_types: List[str] = Field(default_factory=list)

    # Approval and risk
    approval_level: ApprovalLevel = ApprovalLevel.REVIEW
    risk_level: str = "medium"  # low, medium, high, critical
    reversible: bool = True

    # Execution
    timeout_seconds: int = 300
    retry_count: int = 3
    parameters: Dict[str, Any] = Field(default_factory=dict)

    # Metadata
    enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    use_count: int = 0


class RemediationTask(BaseModel):
    """A remediation task instance"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    action_id: str
    action_name: str
    action_type: RemediationType

    # Issue reference
    issue_id: str
    issue_type: str
    issue_description: str

    # Status
    status: RemediationStatus = RemediationStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Approval
    approval_level: ApprovalLevel
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None

    # Execution
    parameters: Dict[str, Any] = Field(default_factory=dict)
    result: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    retry_count: int = 0

    # Rollback
    rollback_data: Dict[str, Any] = Field(default_factory=dict)
    rolled_back_at: Optional[datetime] = None
    rolled_back_by: Optional[str] = None

    # Audit
    audit_trail: List[Dict[str, Any]] = Field(default_factory=list)


class RemediationConfig(BaseModel):
    """Configuration for remediation engine"""
    auto_approve_low_risk: bool = True
    max_concurrent_tasks: int = 5
    default_timeout_seconds: int = 300
    retention_days: int = 90
    require_dual_approval_critical: bool = True
    enable_auto_rollback: bool = True
    rollback_window_hours: int = 24


class RemediationEngine:
    """
    Auto-remediation engine for compliance issues.

    Features:
    - Automated issue remediation
    - Approval workflows
    - Risk-based execution
    - Rollback capabilities
    - Audit trail
    - Custom action handlers
    """

    def __init__(self, config: RemediationConfig = None):
        self.config = config or RemediationConfig()
        self._actions: Dict[str, RemediationAction] = {}
        self._tasks: Dict[str, RemediationTask] = {}
        self._handlers: Dict[str, Callable] = {}
        self._running = False
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._executor_task: Optional[asyncio.Task] = None

        # Initialize default actions
        self._initialize_default_actions()

        logger.info("RemediationEngine initialized")

    def _initialize_default_actions(self):
        """Initialize default remediation actions"""
        defaults = [
            RemediationAction(
                name="Disable Non-Compliant Model",
                description="Disable a model that fails compliance checks",
                action_type=RemediationType.MODEL_UPDATE,
                issue_types=["compliance_failure", "critical_audit_failure"],
                approval_level=ApprovalLevel.APPROVE,
                risk_level="high",
                reversible=True,
                handler="disable_model",
            ),
            RemediationAction(
                name="Update Model Configuration",
                description="Update model configuration parameters",
                action_type=RemediationType.CONFIGURATION,
                issue_types=["configuration_drift", "parameter_violation"],
                approval_level=ApprovalLevel.REVIEW,
                risk_level="medium",
                reversible=True,
                handler="update_config",
            ),
            RemediationAction(
                name="Enforce Rate Limits",
                description="Apply stricter rate limits to problematic endpoints",
                action_type=RemediationType.PERFORMANCE_TUNING,
                issue_types=["rate_limit_violation", "resource_exhaustion"],
                approval_level=ApprovalLevel.AUTO,
                risk_level="low",
                reversible=True,
                handler="enforce_rate_limits",
            ),
            RemediationAction(
                name="Rotate Access Credentials",
                description="Rotate compromised or expired credentials",
                action_type=RemediationType.ACCESS_CONTROL,
                issue_types=["credential_exposure", "credential_expiry"],
                approval_level=ApprovalLevel.APPROVE,
                risk_level="high",
                reversible=False,
                handler="rotate_credentials",
            ),
            RemediationAction(
                name="Apply Policy Update",
                description="Apply updated compliance policy",
                action_type=RemediationType.POLICY_UPDATE,
                issue_types=["policy_violation", "regulation_change"],
                approval_level=ApprovalLevel.REVIEW,
                risk_level="medium",
                reversible=True,
                handler="apply_policy",
            ),
            RemediationAction(
                name="Trigger Retraining",
                description="Trigger model retraining for drift correction",
                action_type=RemediationType.MODEL_UPDATE,
                issue_types=["model_drift", "performance_degradation"],
                approval_level=ApprovalLevel.APPROVE,
                risk_level="high",
                reversible=False,
                handler="trigger_retrain",
            ),
            RemediationAction(
                name="Quarantine Data",
                description="Quarantine problematic data records",
                action_type=RemediationType.DATA_CORRECTION,
                issue_types=["data_quality_issue", "pii_exposure"],
                approval_level=ApprovalLevel.REVIEW,
                risk_level="medium",
                reversible=True,
                handler="quarantine_data",
            ),
            RemediationAction(
                name="Acknowledge Alert",
                description="Auto-acknowledge known non-critical alerts",
                action_type=RemediationType.ALERT_RESPONSE,
                issue_types=["known_issue", "transient_error"],
                approval_level=ApprovalLevel.AUTO,
                risk_level="low",
                reversible=True,
                handler="acknowledge_alert",
            ),
        ]

        for action in defaults:
            self._actions[action.id] = action

    async def start(self):
        """Start the remediation engine"""
        self._running = True
        self._executor_task = asyncio.create_task(self._executor_loop())
        logger.info("Remediation engine started")

    async def stop(self):
        """Stop the remediation engine"""
        self._running = False
        if self._executor_task:
            self._executor_task.cancel()
            try:
                await self._executor_task
            except asyncio.CancelledError:
                pass
        logger.info("Remediation engine stopped")

    def register_action(self, action: RemediationAction) -> str:
        """Register a remediation action"""
        self._actions[action.id] = action
        logger.info("Remediation action registered", action_id=action.id, name=action.name)
        return action.id

    def register_handler(self, handler_name: str, handler: Callable):
        """Register a handler function for actions"""
        self._handlers[handler_name] = handler
        logger.info("Handler registered", handler=handler_name)

    async def create_task(
        self,
        issue_id: str,
        issue_type: str,
        issue_description: str,
        parameters: Dict[str, Any] = None,
        action_id: Optional[str] = None,
    ) -> Optional[RemediationTask]:
        """Create a remediation task for an issue"""
        # Find matching action
        if action_id:
            action = self._actions.get(action_id)
        else:
            action = self._find_matching_action(issue_type)

        if not action or not action.enabled:
            logger.warning(
                "No matching remediation action",
                issue_type=issue_type,
                issue_id=issue_id,
            )
            return None

        # Create task
        task = RemediationTask(
            action_id=action.id,
            action_name=action.name,
            action_type=action.action_type,
            issue_id=issue_id,
            issue_type=issue_type,
            issue_description=issue_description,
            approval_level=action.approval_level,
            parameters=parameters or {},
        )

        # Add audit entry
        task.audit_trail.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "task_created",
            "details": f"Task created for issue {issue_id}",
        })

        # Store task
        self._tasks[task.id] = task

        # Auto-approve if applicable
        if self._should_auto_approve(action):
            await self.approve_task(task.id, "system")

        logger.info(
            "Remediation task created",
            task_id=task.id,
            action=action.name,
            issue_id=issue_id,
        )

        return task

    def _find_matching_action(self, issue_type: str) -> Optional[RemediationAction]:
        """Find a matching action for an issue type"""
        for action in self._actions.values():
            if not action.enabled:
                continue
            if issue_type in action.issue_types:
                return action
        return None

    def _should_auto_approve(self, action: RemediationAction) -> bool:
        """Check if action should be auto-approved"""
        if action.approval_level == ApprovalLevel.AUTO:
            return True

        if (
            self.config.auto_approve_low_risk
            and action.risk_level == "low"
            and action.approval_level == ApprovalLevel.REVIEW
        ):
            return True

        return False

    async def approve_task(
        self, task_id: str, approved_by: str, notes: str = ""
    ) -> bool:
        """Approve a remediation task"""
        task = self._tasks.get(task_id)
        if not task:
            return False

        if task.status != RemediationStatus.PENDING:
            return False

        action = self._actions.get(task.action_id)

        # Check for dual approval requirement
        if (
            action
            and action.risk_level == "critical"
            and self.config.require_dual_approval_critical
        ):
            if not task.approved_by:
                # First approval
                task.approved_by = approved_by
                task.audit_trail.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "event": "first_approval",
                    "by": approved_by,
                    "notes": notes,
                })
                logger.info("First approval received", task_id=task_id, by=approved_by)
                return True
            elif task.approved_by == approved_by:
                logger.warning("Same user cannot provide dual approval")
                return False

        # Approve and queue for execution
        task.status = RemediationStatus.APPROVED
        task.approved_by = approved_by
        task.approved_at = datetime.utcnow()

        task.audit_trail.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "approved",
            "by": approved_by,
            "notes": notes,
        })

        # Add to execution queue
        await self._task_queue.put(task_id)

        logger.info("Task approved", task_id=task_id, by=approved_by)
        return True

    async def reject_task(
        self, task_id: str, rejected_by: str, reason: str = ""
    ) -> bool:
        """Reject a remediation task"""
        task = self._tasks.get(task_id)
        if not task:
            return False

        if task.status != RemediationStatus.PENDING:
            return False

        task.status = RemediationStatus.CANCELLED
        task.completed_at = datetime.utcnow()

        task.audit_trail.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "rejected",
            "by": rejected_by,
            "reason": reason,
        })

        logger.info("Task rejected", task_id=task_id, by=rejected_by, reason=reason)
        return True

    async def _executor_loop(self):
        """Background loop to execute approved tasks"""
        while self._running:
            try:
                # Get task with timeout
                try:
                    task_id = await asyncio.wait_for(
                        self._task_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                task = self._tasks.get(task_id)
                if not task or task.status != RemediationStatus.APPROVED:
                    continue

                await self._execute_task(task)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in executor loop", error=str(e))

    async def _execute_task(self, task: RemediationTask):
        """Execute a remediation task"""
        action = self._actions.get(task.action_id)
        if not action:
            task.status = RemediationStatus.FAILED
            task.error = "Action not found"
            return

        task.status = RemediationStatus.IN_PROGRESS
        task.started_at = datetime.utcnow()

        task.audit_trail.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "execution_started",
        })

        logger.info(
            "Executing remediation task",
            task_id=task.id,
            action=action.name,
        )

        try:
            # Get handler
            handler = self._handlers.get(action.handler)
            if not handler:
                # Use default handler
                handler = self._default_handler

            # Execute with timeout
            result = await asyncio.wait_for(
                self._run_handler(handler, task, action),
                timeout=action.timeout_seconds,
            )

            # Success
            task.status = RemediationStatus.COMPLETED
            task.result = result or {}
            task.completed_at = datetime.utcnow()

            # Update action use count
            action.use_count += 1

            task.audit_trail.append({
                "timestamp": datetime.utcnow().isoformat(),
                "event": "execution_completed",
                "result": task.result,
            })

            logger.info(
                "Remediation task completed",
                task_id=task.id,
                action=action.name,
            )

        except asyncio.TimeoutError:
            task.status = RemediationStatus.FAILED
            task.error = "Execution timeout"
            task.completed_at = datetime.utcnow()

            task.audit_trail.append({
                "timestamp": datetime.utcnow().isoformat(),
                "event": "execution_timeout",
            })

            logger.error("Task execution timeout", task_id=task.id)

        except Exception as e:
            task.status = RemediationStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.utcnow()
            task.retry_count += 1

            task.audit_trail.append({
                "timestamp": datetime.utcnow().isoformat(),
                "event": "execution_failed",
                "error": str(e),
            })

            # Retry if applicable
            if task.retry_count < action.retry_count:
                task.status = RemediationStatus.APPROVED
                await self._task_queue.put(task.id)
                logger.warning(
                    "Task failed, retrying",
                    task_id=task.id,
                    retry=task.retry_count,
                )
            else:
                logger.error(
                    "Task failed after retries",
                    task_id=task.id,
                    error=str(e),
                )

    async def _run_handler(
        self,
        handler: Callable,
        task: RemediationTask,
        action: RemediationAction,
    ) -> Dict[str, Any]:
        """Run a handler function"""
        # Store rollback data before execution
        if action.reversible:
            task.rollback_data = await self._capture_state(task)

        # Execute handler
        if asyncio.iscoroutinefunction(handler):
            result = await handler(task, action)
        else:
            result = handler(task, action)

        return result or {}

    async def _capture_state(self, task: RemediationTask) -> Dict[str, Any]:
        """Capture state before remediation for rollback"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "task_id": task.id,
            "parameters": task.parameters.copy(),
        }

    async def _default_handler(
        self, task: RemediationTask, action: RemediationAction
    ) -> Dict[str, Any]:
        """Default handler for actions without specific handlers"""
        logger.info(
            "Default handler executing",
            task_id=task.id,
            action=action.name,
        )

        # Simulate execution
        await asyncio.sleep(0.1)

        return {
            "message": f"Action '{action.name}' executed successfully",
            "action_type": action.action_type.value,
        }

    async def rollback_task(
        self, task_id: str, rolled_back_by: str, reason: str = ""
    ) -> bool:
        """Rollback a completed remediation task"""
        task = self._tasks.get(task_id)
        if not task:
            return False

        if task.status != RemediationStatus.COMPLETED:
            return False

        action = self._actions.get(task.action_id)
        if not action or not action.reversible:
            return False

        # Check rollback window
        if task.completed_at:
            elapsed = datetime.utcnow() - task.completed_at
            if elapsed > timedelta(hours=self.config.rollback_window_hours):
                logger.warning(
                    "Rollback window expired",
                    task_id=task_id,
                    completed_at=task.completed_at,
                )
                return False

        # Perform rollback
        try:
            handler = self._handlers.get(f"{action.handler}_rollback")
            if handler:
                if asyncio.iscoroutinefunction(handler):
                    await handler(task, task.rollback_data)
                else:
                    handler(task, task.rollback_data)

            task.status = RemediationStatus.ROLLED_BACK
            task.rolled_back_at = datetime.utcnow()
            task.rolled_back_by = rolled_back_by

            task.audit_trail.append({
                "timestamp": datetime.utcnow().isoformat(),
                "event": "rolled_back",
                "by": rolled_back_by,
                "reason": reason,
            })

            logger.info("Task rolled back", task_id=task_id, by=rolled_back_by)
            return True

        except Exception as e:
            logger.error("Rollback failed", task_id=task_id, error=str(e))
            return False

    def get_task(self, task_id: str) -> Optional[RemediationTask]:
        """Get a task by ID"""
        return self._tasks.get(task_id)

    def list_tasks(
        self,
        status: Optional[RemediationStatus] = None,
        action_type: Optional[RemediationType] = None,
        limit: int = 100,
    ) -> List[RemediationTask]:
        """List tasks with optional filters"""
        tasks = list(self._tasks.values())

        if status:
            tasks = [t for t in tasks if t.status == status]
        if action_type:
            tasks = [t for t in tasks if t.action_type == action_type]

        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks[:limit]

    def get_pending_approvals(self) -> List[RemediationTask]:
        """Get tasks pending approval"""
        return [
            t for t in self._tasks.values()
            if t.status == RemediationStatus.PENDING
        ]

    def get_actions(self) -> List[RemediationAction]:
        """Get all remediation actions"""
        return list(self._actions.values())

    def get_audit_trail(self, task_id: str) -> List[Dict[str, Any]]:
        """Get audit trail for a task"""
        task = self._tasks.get(task_id)
        if not task:
            return []
        return task.audit_trail

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        status_counts = {}
        for task in self._tasks.values():
            status_counts[task.status.value] = status_counts.get(
                task.status.value, 0
            ) + 1

        completed = [
            t for t in self._tasks.values()
            if t.status == RemediationStatus.COMPLETED
        ]

        success_rate = len(completed) / len(self._tasks) if self._tasks else 0

        return {
            "total_actions": len(self._actions),
            "total_tasks": len(self._tasks),
            "by_status": status_counts,
            "pending_approvals": len(self.get_pending_approvals()),
            "success_rate": round(success_rate * 100, 2),
            "total_executions": sum(a.use_count for a in self._actions.values()),
        }


# Global remediation engine
remediation_engine: Optional[RemediationEngine] = None


def get_remediation_engine() -> RemediationEngine:
    """Get the global remediation engine"""
    global remediation_engine
    if remediation_engine is None:
        remediation_engine = RemediationEngine()
    return remediation_engine

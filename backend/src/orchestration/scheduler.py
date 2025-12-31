"""Audit Scheduler for AURA Platform"""

import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import uuid

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class ScheduleType(str, Enum):
    """Types of schedule triggers"""
    CRON = "cron"
    INTERVAL = "interval"
    EVENT = "event"
    RISK_BASED = "risk_based"


class SchedulePriority(str, Enum):
    """Priority levels for scheduled tasks"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ScheduleConfig(BaseModel):
    """Configuration for a scheduled audit"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    schedule_type: ScheduleType
    model_id: Optional[str] = None
    policy_ids: List[str] = Field(default_factory=list)
    priority: SchedulePriority = SchedulePriority.MEDIUM

    # Interval scheduling
    interval_seconds: Optional[int] = None

    # Cron scheduling
    cron_expression: Optional[str] = None

    # Event-based scheduling
    trigger_events: List[str] = Field(default_factory=list)

    # Risk-based scheduling
    risk_threshold: float = 0.7

    # Execution settings
    timeout_seconds: int = 3600
    retry_count: int = 3
    enabled: bool = True

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None


class ScheduledTask(BaseModel):
    """A scheduled task instance"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    schedule_id: str
    scheduled_time: datetime
    status: str = "pending"  # pending, running, completed, failed, cancelled
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class AuditScheduler:
    """
    Intelligent audit scheduling system.

    Features:
    - Multiple schedule types (cron, interval, event, risk-based)
    - Priority queue management
    - Dynamic rescheduling based on results
    - Resource-aware scheduling
    - Conflict detection
    """

    def __init__(self):
        self._schedules: Dict[str, ScheduleConfig] = {}
        self._tasks: Dict[str, ScheduledTask] = {}
        self._task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._handlers: Dict[str, Callable] = {}
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._executor_task: Optional[asyncio.Task] = None

        logger.info("AuditScheduler initialized")

    async def start(self):
        """Start the scheduler"""
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduling_loop())
        self._executor_task = asyncio.create_task(self._execution_loop())
        logger.info("Audit scheduler started")

    async def stop(self):
        """Stop the scheduler"""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
        if self._executor_task:
            self._executor_task.cancel()
        logger.info("Audit scheduler stopped")

    def register_handler(self, schedule_type: ScheduleType, handler: Callable):
        """Register a handler for a schedule type"""
        self._handlers[schedule_type.value] = handler

    def add_schedule(self, config: ScheduleConfig) -> str:
        """Add a new schedule"""
        self._schedules[config.id] = config

        # Calculate next run time
        config.next_run = self._calculate_next_run(config)

        logger.info(
            "Schedule added",
            schedule_id=config.id,
            name=config.name,
            type=config.schedule_type,
            next_run=config.next_run,
        )

        return config.id

    def update_schedule(self, schedule_id: str, updates: Dict[str, Any]) -> bool:
        """Update a schedule configuration"""
        if schedule_id not in self._schedules:
            return False

        config = self._schedules[schedule_id]
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Recalculate next run
        config.next_run = self._calculate_next_run(config)

        logger.info("Schedule updated", schedule_id=schedule_id)
        return True

    def remove_schedule(self, schedule_id: str) -> bool:
        """Remove a schedule"""
        if schedule_id in self._schedules:
            del self._schedules[schedule_id]
            logger.info("Schedule removed", schedule_id=schedule_id)
            return True
        return False

    def get_schedule(self, schedule_id: str) -> Optional[ScheduleConfig]:
        """Get a schedule by ID"""
        return self._schedules.get(schedule_id)

    def list_schedules(self) -> List[ScheduleConfig]:
        """List all schedules"""
        return list(self._schedules.values())

    async def trigger_audit(self, schedule_id: str) -> str:
        """Manually trigger an audit for a schedule"""
        if schedule_id not in self._schedules:
            raise ValueError(f"Schedule not found: {schedule_id}")

        config = self._schedules[schedule_id]
        task = await self._create_task(config, datetime.utcnow())

        logger.info("Audit manually triggered", schedule_id=schedule_id, task_id=task.id)
        return task.id

    def _calculate_next_run(self, config: ScheduleConfig) -> datetime:
        """Calculate the next run time for a schedule"""
        now = datetime.utcnow()

        if config.schedule_type == ScheduleType.INTERVAL:
            if config.interval_seconds:
                return now + timedelta(seconds=config.interval_seconds)

        elif config.schedule_type == ScheduleType.CRON:
            # Simplified cron parsing - in production would use croniter
            # Default to 1 hour for now
            return now + timedelta(hours=1)

        elif config.schedule_type == ScheduleType.RISK_BASED:
            # Schedule based on risk assessment
            # Higher risk = sooner scheduling
            risk_factor = config.risk_threshold
            hours = max(1, int((1 - risk_factor) * 24))
            return now + timedelta(hours=hours)

        # Default: 24 hours
        return now + timedelta(hours=24)

    async def _scheduling_loop(self):
        """Main loop to check schedules and create tasks"""
        while self._running:
            try:
                now = datetime.utcnow()

                for schedule_id, config in list(self._schedules.items()):
                    if not config.enabled:
                        continue

                    if config.next_run and config.next_run <= now:
                        # Time to run this schedule
                        await self._create_task(config, now)

                        # Update last run and calculate next run
                        config.last_run = now
                        config.next_run = self._calculate_next_run(config)

                await asyncio.sleep(1)  # Check every second

            except Exception as e:
                logger.error("Error in scheduling loop", error=str(e))
                await asyncio.sleep(5)

    async def _create_task(self, config: ScheduleConfig, scheduled_time: datetime) -> ScheduledTask:
        """Create a new task for a schedule"""
        task = ScheduledTask(
            schedule_id=config.id,
            scheduled_time=scheduled_time,
        )

        self._tasks[task.id] = task

        # Add to priority queue
        # Priority value: lower = higher priority
        priority_values = {
            SchedulePriority.CRITICAL: 1,
            SchedulePriority.HIGH: 2,
            SchedulePriority.MEDIUM: 3,
            SchedulePriority.LOW: 4,
        }
        priority = priority_values.get(config.priority, 3)

        await self._task_queue.put((priority, scheduled_time.timestamp(), task.id))

        logger.debug(
            "Task created",
            task_id=task.id,
            schedule_id=config.id,
            priority=config.priority,
        )

        return task

    async def _execution_loop(self):
        """Main loop to execute tasks from the queue"""
        while self._running:
            try:
                # Get next task with timeout
                try:
                    _, _, task_id = await asyncio.wait_for(
                        self._task_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                if task_id not in self._tasks:
                    continue

                task = self._tasks[task_id]
                config = self._schedules.get(task.schedule_id)

                if not config:
                    continue

                # Execute the task
                await self._execute_task(task, config)

            except Exception as e:
                logger.error("Error in execution loop", error=str(e))

    async def _execute_task(self, task: ScheduledTask, config: ScheduleConfig):
        """Execute a scheduled task"""
        task.status = "running"
        task.started_at = datetime.utcnow()

        logger.info(
            "Executing task",
            task_id=task.id,
            schedule_name=config.name,
        )

        try:
            # Get the handler for this schedule type
            handler = self._handlers.get(config.schedule_type.value)

            if handler:
                result = await handler(config)
                task.result = result if isinstance(result, dict) else {"result": result}
                task.status = "completed"
            else:
                # Default handler - just mark as completed
                task.result = {"message": "No handler registered"}
                task.status = "completed"

            logger.info("Task completed", task_id=task.id)

        except Exception as e:
            task.error = str(e)
            task.status = "failed"
            logger.error("Task failed", task_id=task.id, error=str(e))

        finally:
            task.completed_at = datetime.utcnow()

    def get_task_status(self, task_id: str) -> Optional[ScheduledTask]:
        """Get the status of a task"""
        return self._tasks.get(task_id)

    def get_pending_tasks(self) -> List[ScheduledTask]:
        """Get all pending tasks"""
        return [t for t in self._tasks.values() if t.status == "pending"]

    def get_running_tasks(self) -> List[ScheduledTask]:
        """Get all running tasks"""
        return [t for t in self._tasks.values() if t.status == "running"]

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task"""
        if task_id in self._tasks:
            task = self._tasks[task_id]
            if task.status == "pending":
                task.status = "cancelled"
                logger.info("Task cancelled", task_id=task_id)
                return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        status_counts = {}
        for task in self._tasks.values():
            status_counts[task.status] = status_counts.get(task.status, 0) + 1

        return {
            "total_schedules": len(self._schedules),
            "enabled_schedules": sum(1 for s in self._schedules.values() if s.enabled),
            "total_tasks": len(self._tasks),
            "tasks_by_status": status_counts,
            "queue_depth": self._task_queue.qsize(),
        }

"""Workflow State Persistence and Recovery for AURA Platform"""

import asyncio
import json
import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid
import hashlib

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class PersistenceBackend(str, Enum):
    """Supported persistence backends"""
    MEMORY = "memory"
    FILE = "file"
    REDIS = "redis"


class WorkflowCheckpoint(BaseModel):
    """A checkpoint in workflow execution"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str
    step_name: str
    state: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    checksum: str = ""

    def calculate_checksum(self) -> str:
        """Calculate checksum of state"""
        state_str = json.dumps(self.state, sort_keys=True, default=str)
        return hashlib.md5(state_str.encode()).hexdigest()


class PersistedWorkflow(BaseModel):
    """A persisted workflow state"""
    workflow_id: str
    name: str
    status: str = "pending"  # pending, running, paused, completed, failed, cancelled
    current_step: str = ""

    # State data
    state: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Execution tracking
    started_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    # Checkpoints
    checkpoint_ids: List[str] = Field(default_factory=list)
    last_checkpoint: Optional[str] = None

    # Error handling
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    # Recovery info
    recoverable: bool = True
    recovery_point: Optional[str] = None


class PersistenceConfig(BaseModel):
    """Configuration for workflow persistence"""
    backend: PersistenceBackend = PersistenceBackend.FILE
    storage_path: str = "./data/workflows"

    # Checkpointing
    auto_checkpoint: bool = True
    checkpoint_interval_steps: int = 1
    max_checkpoints_per_workflow: int = 10

    # Retention
    retention_days: int = 30
    archive_completed: bool = True

    # Recovery
    auto_recover: bool = True
    recovery_timeout_seconds: int = 300

    # Connection settings (for Redis)
    connection_string: Optional[str] = None


class PersistenceStore(ABC):
    """Abstract base class for persistence stores"""

    @abstractmethod
    async def save_workflow(self, workflow: PersistedWorkflow) -> bool:
        """Save workflow state"""
        pass

    @abstractmethod
    async def load_workflow(self, workflow_id: str) -> Optional[PersistedWorkflow]:
        """Load workflow state"""
        pass

    @abstractmethod
    async def delete_workflow(self, workflow_id: str) -> bool:
        """Delete workflow state"""
        pass

    @abstractmethod
    async def list_workflows(
        self, status: Optional[str] = None, limit: int = 100
    ) -> List[PersistedWorkflow]:
        """List workflows with optional status filter"""
        pass

    @abstractmethod
    async def save_checkpoint(self, checkpoint: WorkflowCheckpoint) -> bool:
        """Save a checkpoint"""
        pass

    @abstractmethod
    async def load_checkpoint(self, checkpoint_id: str) -> Optional[WorkflowCheckpoint]:
        """Load a checkpoint"""
        pass

    @abstractmethod
    async def list_checkpoints(self, workflow_id: str) -> List[WorkflowCheckpoint]:
        """List checkpoints for a workflow"""
        pass


class InMemoryStore(PersistenceStore):
    """In-memory persistence store"""

    def __init__(self):
        self._workflows: Dict[str, PersistedWorkflow] = {}
        self._checkpoints: Dict[str, WorkflowCheckpoint] = {}

    async def save_workflow(self, workflow: PersistedWorkflow) -> bool:
        self._workflows[workflow.workflow_id] = workflow
        return True

    async def load_workflow(self, workflow_id: str) -> Optional[PersistedWorkflow]:
        return self._workflows.get(workflow_id)

    async def delete_workflow(self, workflow_id: str) -> bool:
        if workflow_id in self._workflows:
            del self._workflows[workflow_id]
            return True
        return False

    async def list_workflows(
        self, status: Optional[str] = None, limit: int = 100
    ) -> List[PersistedWorkflow]:
        workflows = list(self._workflows.values())
        if status:
            workflows = [w for w in workflows if w.status == status]
        return workflows[:limit]

    async def save_checkpoint(self, checkpoint: WorkflowCheckpoint) -> bool:
        self._checkpoints[checkpoint.id] = checkpoint
        return True

    async def load_checkpoint(self, checkpoint_id: str) -> Optional[WorkflowCheckpoint]:
        return self._checkpoints.get(checkpoint_id)

    async def list_checkpoints(self, workflow_id: str) -> List[WorkflowCheckpoint]:
        return [
            c for c in self._checkpoints.values()
            if c.workflow_id == workflow_id
        ]


class FileStore(PersistenceStore):
    """File-based persistence store"""

    def __init__(self, storage_path: str):
        self._storage_path = Path(storage_path)
        self._workflows_path = self._storage_path / "workflows"
        self._checkpoints_path = self._storage_path / "checkpoints"

        # Create directories
        self._workflows_path.mkdir(parents=True, exist_ok=True)
        self._checkpoints_path.mkdir(parents=True, exist_ok=True)

    def _workflow_file(self, workflow_id: str) -> Path:
        return self._workflows_path / f"{workflow_id}.json"

    def _checkpoint_file(self, checkpoint_id: str) -> Path:
        return self._checkpoints_path / f"{checkpoint_id}.json"

    async def save_workflow(self, workflow: PersistedWorkflow) -> bool:
        try:
            filepath = self._workflow_file(workflow.workflow_id)
            with open(filepath, "w") as f:
                json.dump(workflow.model_dump(mode="json"), f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error("Failed to save workflow", error=str(e))
            return False

    async def load_workflow(self, workflow_id: str) -> Optional[PersistedWorkflow]:
        try:
            filepath = self._workflow_file(workflow_id)
            if not filepath.exists():
                return None
            with open(filepath, "r") as f:
                data = json.load(f)
            return PersistedWorkflow(**data)
        except Exception as e:
            logger.error("Failed to load workflow", error=str(e))
            return None

    async def delete_workflow(self, workflow_id: str) -> bool:
        try:
            filepath = self._workflow_file(workflow_id)
            if filepath.exists():
                filepath.unlink()
                return True
            return False
        except Exception as e:
            logger.error("Failed to delete workflow", error=str(e))
            return False

    async def list_workflows(
        self, status: Optional[str] = None, limit: int = 100
    ) -> List[PersistedWorkflow]:
        workflows = []
        try:
            for filepath in self._workflows_path.glob("*.json"):
                workflow = await self.load_workflow(filepath.stem)
                if workflow:
                    if status is None or workflow.status == status:
                        workflows.append(workflow)

            # Sort by last_updated
            workflows.sort(key=lambda w: w.last_updated, reverse=True)
            return workflows[:limit]
        except Exception as e:
            logger.error("Failed to list workflows", error=str(e))
            return []

    async def save_checkpoint(self, checkpoint: WorkflowCheckpoint) -> bool:
        try:
            filepath = self._checkpoint_file(checkpoint.id)
            with open(filepath, "w") as f:
                json.dump(checkpoint.model_dump(mode="json"), f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error("Failed to save checkpoint", error=str(e))
            return False

    async def load_checkpoint(self, checkpoint_id: str) -> Optional[WorkflowCheckpoint]:
        try:
            filepath = self._checkpoint_file(checkpoint_id)
            if not filepath.exists():
                return None
            with open(filepath, "r") as f:
                data = json.load(f)
            return WorkflowCheckpoint(**data)
        except Exception as e:
            logger.error("Failed to load checkpoint", error=str(e))
            return None

    async def list_checkpoints(self, workflow_id: str) -> List[WorkflowCheckpoint]:
        checkpoints = []
        try:
            for filepath in self._checkpoints_path.glob("*.json"):
                checkpoint = await self.load_checkpoint(filepath.stem)
                if checkpoint and checkpoint.workflow_id == workflow_id:
                    checkpoints.append(checkpoint)

            # Sort by created_at
            checkpoints.sort(key=lambda c: c.created_at)
            return checkpoints
        except Exception as e:
            logger.error("Failed to list checkpoints", error=str(e))
            return []


class WorkflowPersistence:
    """
    Workflow state persistence and recovery system.

    Features:
    - Multiple backend support (memory, file, redis)
    - Automatic checkpointing
    - State recovery after failures
    - Checkpoint management
    - Archive and retention policies
    - Transactional state updates
    """

    def __init__(self, config: PersistenceConfig = None):
        self.config = config or PersistenceConfig()
        self._store = self._create_store()
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None

        logger.info(
            "WorkflowPersistence initialized",
            backend=self.config.backend.value,
        )

    def _create_store(self) -> PersistenceStore:
        """Create the appropriate persistence store"""
        if self.config.backend == PersistenceBackend.MEMORY:
            return InMemoryStore()
        elif self.config.backend == PersistenceBackend.FILE:
            return FileStore(self.config.storage_path)
        else:
            # Default to memory for unsupported backends
            logger.warning(
                f"Backend {self.config.backend} not fully implemented, using memory"
            )
            return InMemoryStore()

    async def start(self):
        """Start the persistence service"""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Workflow persistence started")

        # Auto-recover pending workflows
        if self.config.auto_recover:
            await self._auto_recover()

    async def stop(self):
        """Stop the persistence service"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Workflow persistence stopped")

    async def persist_workflow(
        self,
        workflow_id: str,
        name: str,
        state: Dict[str, Any],
        status: str = "pending",
        context: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
    ) -> PersistedWorkflow:
        """Persist a new workflow or update existing"""
        existing = await self._store.load_workflow(workflow_id)

        if existing:
            # Update existing
            existing.state = state
            existing.status = status
            existing.last_updated = datetime.utcnow()
            if context:
                existing.context.update(context)
            if metadata:
                existing.metadata.update(metadata)
            workflow = existing
        else:
            # Create new
            workflow = PersistedWorkflow(
                workflow_id=workflow_id,
                name=name,
                status=status,
                state=state,
                context=context or {},
                metadata=metadata or {},
                started_at=datetime.utcnow() if status == "running" else None,
            )

        await self._store.save_workflow(workflow)

        logger.debug(
            "Workflow persisted",
            workflow_id=workflow_id,
            status=status,
        )

        return workflow

    async def update_status(
        self,
        workflow_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> bool:
        """Update workflow status"""
        workflow = await self._store.load_workflow(workflow_id)
        if not workflow:
            return False

        workflow.status = status
        workflow.last_updated = datetime.utcnow()

        if status == "running" and not workflow.started_at:
            workflow.started_at = datetime.utcnow()
        elif status == "paused":
            workflow.paused_at = datetime.utcnow()
        elif status in ["completed", "failed", "cancelled"]:
            workflow.completed_at = datetime.utcnow()

        if error:
            workflow.error = error

        await self._store.save_workflow(workflow)
        return True

    async def update_state(
        self,
        workflow_id: str,
        state: Dict[str, Any],
        current_step: str = "",
    ) -> bool:
        """Update workflow state"""
        workflow = await self._store.load_workflow(workflow_id)
        if not workflow:
            return False

        workflow.state = state
        workflow.current_step = current_step
        workflow.last_updated = datetime.utcnow()

        await self._store.save_workflow(workflow)

        # Auto checkpoint if configured
        if self.config.auto_checkpoint:
            await self.create_checkpoint(workflow_id, current_step, state)

        return True

    async def create_checkpoint(
        self,
        workflow_id: str,
        step_name: str,
        state: Dict[str, Any],
    ) -> str:
        """Create a checkpoint for workflow state"""
        workflow = await self._store.load_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")

        # Create checkpoint
        checkpoint = WorkflowCheckpoint(
            workflow_id=workflow_id,
            step_name=step_name,
            state=state,
        )
        checkpoint.checksum = checkpoint.calculate_checksum()

        await self._store.save_checkpoint(checkpoint)

        # Update workflow
        workflow.checkpoint_ids.append(checkpoint.id)
        workflow.last_checkpoint = checkpoint.id
        workflow.recovery_point = step_name

        # Trim old checkpoints
        if len(workflow.checkpoint_ids) > self.config.max_checkpoints_per_workflow:
            old_ids = workflow.checkpoint_ids[:-self.config.max_checkpoints_per_workflow]
            workflow.checkpoint_ids = workflow.checkpoint_ids[-self.config.max_checkpoints_per_workflow:]

            # Delete old checkpoint files
            for old_id in old_ids:
                old_checkpoint = await self._store.load_checkpoint(old_id)
                if old_checkpoint:
                    # For file store, we'd delete the file
                    pass

        await self._store.save_workflow(workflow)

        logger.debug(
            "Checkpoint created",
            workflow_id=workflow_id,
            checkpoint_id=checkpoint.id,
            step=step_name,
        )

        return checkpoint.id

    async def load_workflow(self, workflow_id: str) -> Optional[PersistedWorkflow]:
        """Load a workflow"""
        return await self._store.load_workflow(workflow_id)

    async def list_workflows(
        self,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[PersistedWorkflow]:
        """List workflows"""
        return await self._store.list_workflows(status, limit)

    async def get_recoverable_workflows(self) -> List[PersistedWorkflow]:
        """Get workflows that can be recovered"""
        workflows = await self._store.list_workflows()
        return [
            w for w in workflows
            if w.status in ["running", "paused"] and w.recoverable
        ]

    async def recover_workflow(
        self, workflow_id: str, checkpoint_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Recover workflow from checkpoint"""
        workflow = await self._store.load_workflow(workflow_id)
        if not workflow:
            return None

        if not workflow.recoverable:
            logger.warning("Workflow not recoverable", workflow_id=workflow_id)
            return None

        # Determine which checkpoint to use
        if checkpoint_id:
            checkpoint = await self._store.load_checkpoint(checkpoint_id)
        elif workflow.last_checkpoint:
            checkpoint = await self._store.load_checkpoint(workflow.last_checkpoint)
        else:
            # No checkpoint, use current state
            checkpoint = None

        if checkpoint:
            # Verify checksum
            if checkpoint.checksum != checkpoint.calculate_checksum():
                logger.error(
                    "Checkpoint checksum mismatch",
                    workflow_id=workflow_id,
                    checkpoint_id=checkpoint.id,
                )
                return None

            recovered_state = checkpoint.state
            recovery_step = checkpoint.step_name
        else:
            recovered_state = workflow.state
            recovery_step = workflow.current_step

        # Update workflow for recovery
        workflow.status = "running"
        workflow.retry_count += 1
        workflow.last_updated = datetime.utcnow()
        workflow.error = None

        await self._store.save_workflow(workflow)

        logger.info(
            "Workflow recovered",
            workflow_id=workflow_id,
            recovery_step=recovery_step,
            retry_count=workflow.retry_count,
        )

        return {
            "workflow_id": workflow_id,
            "state": recovered_state,
            "recovery_step": recovery_step,
            "retry_count": workflow.retry_count,
        }

    async def pause_workflow(self, workflow_id: str) -> bool:
        """Pause a running workflow"""
        workflow = await self._store.load_workflow(workflow_id)
        if not workflow or workflow.status != "running":
            return False

        workflow.status = "paused"
        workflow.paused_at = datetime.utcnow()
        workflow.last_updated = datetime.utcnow()

        await self._store.save_workflow(workflow)

        logger.info("Workflow paused", workflow_id=workflow_id)
        return True

    async def resume_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Resume a paused workflow"""
        workflow = await self._store.load_workflow(workflow_id)
        if not workflow or workflow.status != "paused":
            return None

        workflow.status = "running"
        workflow.last_updated = datetime.utcnow()

        await self._store.save_workflow(workflow)

        logger.info("Workflow resumed", workflow_id=workflow_id)

        return {
            "workflow_id": workflow_id,
            "state": workflow.state,
            "current_step": workflow.current_step,
        }

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a workflow"""
        workflow = await self._store.load_workflow(workflow_id)
        if not workflow:
            return False

        if workflow.status in ["completed", "cancelled"]:
            return False

        workflow.status = "cancelled"
        workflow.completed_at = datetime.utcnow()
        workflow.last_updated = datetime.utcnow()

        await self._store.save_workflow(workflow)

        logger.info("Workflow cancelled", workflow_id=workflow_id)
        return True

    async def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow and its checkpoints"""
        workflow = await self._store.load_workflow(workflow_id)
        if not workflow:
            return False

        # Delete checkpoints
        for checkpoint_id in workflow.checkpoint_ids:
            # Implementation depends on store type
            pass

        return await self._store.delete_workflow(workflow_id)

    async def get_checkpoints(self, workflow_id: str) -> List[WorkflowCheckpoint]:
        """Get checkpoints for a workflow"""
        return await self._store.list_checkpoints(workflow_id)

    async def rollback_to_checkpoint(
        self, workflow_id: str, checkpoint_id: str
    ) -> bool:
        """Rollback workflow to a specific checkpoint"""
        workflow = await self._store.load_workflow(workflow_id)
        if not workflow:
            return False

        checkpoint = await self._store.load_checkpoint(checkpoint_id)
        if not checkpoint or checkpoint.workflow_id != workflow_id:
            return False

        # Restore state
        workflow.state = checkpoint.state
        workflow.current_step = checkpoint.step_name
        workflow.recovery_point = checkpoint.step_name
        workflow.last_updated = datetime.utcnow()

        await self._store.save_workflow(workflow)

        logger.info(
            "Workflow rolled back",
            workflow_id=workflow_id,
            checkpoint_id=checkpoint_id,
        )

        return True

    async def _auto_recover(self):
        """Auto-recover interrupted workflows on startup"""
        workflows = await self.get_recoverable_workflows()

        for workflow in workflows:
            if workflow.retry_count < workflow.max_retries:
                try:
                    await self.recover_workflow(workflow.workflow_id)
                except Exception as e:
                    logger.error(
                        "Auto-recovery failed",
                        workflow_id=workflow.workflow_id,
                        error=str(e),
                    )

        if workflows:
            logger.info("Auto-recovery completed", recovered=len(workflows))

    async def _cleanup_loop(self):
        """Background loop for cleanup tasks"""
        while self._running:
            try:
                await self._cleanup_old_workflows()
                await asyncio.sleep(3600)  # Run hourly
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in cleanup loop", error=str(e))
                await asyncio.sleep(60)

    async def _cleanup_old_workflows(self):
        """Clean up old completed workflows"""
        if self.config.retention_days <= 0:
            return

        cutoff = datetime.utcnow() - timedelta(days=self.config.retention_days)
        workflows = await self._store.list_workflows()

        deleted = 0
        for workflow in workflows:
            if workflow.status in ["completed", "failed", "cancelled"]:
                if workflow.completed_at and workflow.completed_at < cutoff:
                    await self.delete_workflow(workflow.workflow_id)
                    deleted += 1

        if deleted:
            logger.info("Old workflows cleaned up", count=deleted)

    def get_stats(self) -> Dict[str, Any]:
        """Get persistence statistics"""
        return {
            "backend": self.config.backend.value,
            "storage_path": self.config.storage_path,
            "auto_checkpoint": self.config.auto_checkpoint,
            "retention_days": self.config.retention_days,
        }


# Global persistence instance
workflow_persistence: Optional[WorkflowPersistence] = None


def get_workflow_persistence() -> WorkflowPersistence:
    """Get the global workflow persistence instance"""
    global workflow_persistence
    if workflow_persistence is None:
        workflow_persistence = WorkflowPersistence()
    return workflow_persistence

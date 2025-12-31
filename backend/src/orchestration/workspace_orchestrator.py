# src/orchestration/workspace_orchestrator.py
"""Workspace Orchestrator

This module implements a lightweight orchestrator that watches the platform's
event stream and proactively suggests new workspaces based on simple heuristics.
For now the heuristics are placeholder logic – when three events of any type
are observed within a short window, a suggestion is generated.

The orchestrator stores suggestions in an in‑memory list that can be queried
via the API endpoints defined in `src/api/workspaces.py`.
"""

import time
import uuid
from datetime import datetime
from collections import deque
from typing import Deque, Dict, List, Optional

from src.infrastructure.event_stream import EventStream, EventType, Event
from src.models.workspace_suggestion import WorkspaceSuggestion, Base
from src.db import SessionLocal

# Ensure tables are created
Base.metadata.create_all(bind=SessionLocal().bind)


class WorkspaceOrchestrator:
    """Orchestrator that listens to the EventStream and creates suggestions.

    The orchestrator persists suggestions in a SQLite database via SQLAlchemy.
    """

    def __init__(self, event_stream: EventStream):
        self.event_stream = event_stream
        # Keep a short‑term history of recent events (timestamp, Event)
        self._recent_events: Deque[Event] = deque(maxlen=20)
        # Subscribe to all events – we use a wildcard subscription
        self._subscription_id = self.event_stream.subscribe(
            event_types=[et for et in EventType],
            callback=self._handle_event,
        )

    async def _handle_event(self, event: Event):
        """Callback invoked for every emitted event.

        Records the event and attempts to generate a suggestion based on recent activity.
        """
        self._recent_events.append(event)
        await self._maybe_generate_suggestion()

    async def _maybe_generate_suggestion(self):
        # Simple heuristic: if we have at least 3 events in the last 30 s
        now = datetime.utcnow()
        recent = [e for e in self._recent_events if (now - e.timestamp).total_seconds() <= 30]
        if len(recent) >= 3 and not self._has_pending_suggestion():
            suggestion = WorkspaceSuggestion(
                id=str(uuid.uuid4()),
                title="New Workspace: Data‑Prep for Q4 Forecast",
                description="A workspace pre‑populated with notebooks and pipelines for Q4 data preparation.",
                resources=[
                    {"type": "dashboard", "id": "audit_dashboard"},
                    {"type": "notebook", "id": "data_prep_notebook"},
                ],
            )
            # Persist suggestion to DB
            db = SessionLocal()
            try:
                db.add(suggestion)
                db.commit()
                db.refresh(suggestion)
            finally:
                db.close()
            # Emit an event so UI can react (optional)
            await self.event_stream.emit_simple(
                event_type=EventType.SYSTEM_ALERT,
                source="workspace_orchestrator",
                data={"suggestion_id": suggestion.id},
            )

    def _has_pending_suggestion(self) -> bool:
        db = SessionLocal()
        try:
            count = (
                db.query(WorkspaceSuggestion)
                .filter(WorkspaceSuggestion.dismissed == False, WorkspaceSuggestion.accepted == False)
                .count()
            )
            return count > 0
        finally:
            db.close()

    # Public API used by the FastAPI endpoints
    def list_pending(self) -> List[Dict]:
        db = SessionLocal()
        try:
            suggestions = (
                db.query(WorkspaceSuggestion)
                .filter(WorkspaceSuggestion.dismissed == False, WorkspaceSuggestion.accepted == False)
                .all()
            )
            return [s.to_dict() for s in suggestions]
        finally:
            db.close()

    def accept(self, suggestion_id: str) -> bool:
        db = SessionLocal()
        try:
            suggestion = (
                db.query(WorkspaceSuggestion)
                .filter(WorkspaceSuggestion.id == suggestion_id, WorkspaceSuggestion.dismissed == False)
                .first()
            )
            if suggestion:
                suggestion.accepted = True
                db.commit()
                return True
            return False
        finally:
            db.close()

    def dismiss(self, suggestion_id: str) -> bool:
        db = SessionLocal()
        try:
            suggestion = (
                db.query(WorkspaceSuggestion)
                .filter(WorkspaceSuggestion.id == suggestion_id, WorkspaceSuggestion.accepted == False)
                .first()
            )
            if suggestion:
                suggestion.dismissed = True
                db.commit()
                return True
            return False
        finally:
            db.close()

    # Cleanup – unsubscribe when the platform shuts down
    async def shutdown(self):
        self.event_stream.unsubscribe(self._subscription_id)

# Helper to create a singleton orchestrator – the platform will instantiate it
_orchestrator_instance: Optional[WorkspaceOrchestrator] = None

def get_workspace_orchestrator(event_stream: EventStream) -> WorkspaceOrchestrator:
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = WorkspaceOrchestrator(event_stream)
    return _orchestrator_instance

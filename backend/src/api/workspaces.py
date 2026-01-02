# src/api/workspaces.py
"""FastAPI endpoints for Self‑Organizing Workspace suggestions.
These endpoints allow the frontend to fetch pending suggestions and to accept or dismiss them.
"""

from fastapi import APIRouter, HTTPException, Request

from src.infrastructure.event_stream import EventStream
from src.orchestration.workspace_orchestrator import get_workspace_orchestrator, WorkspaceOrchestrator

router = APIRouter()

# Dependency to get the orchestrator – assumes a global EventStream instance is available.
# In the platform startup we will create the orchestrator and store it in the FastAPI app state.

def _get_orchestrator(request: Request) -> WorkspaceOrchestrator:
    if not hasattr(request.app.state, "workspace_orchestrator"):
        raise HTTPException(status_code=500, detail="Workspace orchestrator not initialized")
    return request.app.state.workspace_orchestrator

@router.get("/workspaces/suggestions")
async def list_suggestions(request: Request):
    orchestrator = _get_orchestrator(request)
    return {"suggestions": orchestrator.list_pending()}

@router.post("/workspaces/suggestions/{suggestion_id}/accept")
async def accept_suggestion(suggestion_id: str, request: Request):
    orchestrator = _get_orchestrator(request)
    if orchestrator.accept(suggestion_id):
        return {"status": "accepted", "suggestion_id": suggestion_id}
    raise HTTPException(status_code=404, detail="Suggestion not found or already dismissed")

@router.post("/workspaces/suggestions/{suggestion_id}/dismiss")
async def dismiss_suggestion(suggestion_id: str, request: Request):
    orchestrator = _get_orchestrator(request)
    if orchestrator.dismiss(suggestion_id):
        return {"status": "dismissed", "suggestion_id": suggestion_id}
    raise HTTPException(status_code=404, detail="Suggestion not found or already accepted")

# src/api/websocket.py
"""WebSocket routes for streaming workspace suggestions.
The endpoint periodically sends the list of pending suggestions to connected clients.
"""

import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List

router = APIRouter()

@router.websocket("/ws/workspaces/suggestions")
async def suggestions_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Retrieve orchestrator from app state via websocket.app
            orchestrator = getattr(websocket.app.state, "workspace_orchestrator", None)
            if orchestrator:
                suggestions = orchestrator.list_pending()
                await websocket.send_json({"suggestions": suggestions})
            await asyncio.sleep(2)  # poll interval
    except WebSocketDisconnect:
        # Client disconnected
        pass
    except Exception as e:
        # Log error if needed (omitted for brevity)
        await websocket.close()

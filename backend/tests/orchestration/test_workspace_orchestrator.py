import pytest
import asyncio
import sys
from unittest.mock import MagicMock, AsyncMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Mock internal dependencies to avoid cascading imports of missing 3rd party libs
sys.modules["src.orchestration.workflow_engine"] = MagicMock()
sys.modules["src.core.base_agent"] = MagicMock()
sys.modules["src.infrastructure.message_bus"] = MagicMock()
# Also mock langchain just in case
sys.modules["langchain_anthropic"] = MagicMock()
sys.modules["langchain_core"] = MagicMock() 

from src.models.workspace_suggestion import Base, WorkspaceSuggestion
# Import the module to be tested
import src.orchestration.workspace_orchestrator as orchestrator_module
from src.orchestration.workspace_orchestrator import WorkspaceOrchestrator, get_workspace_orchestrator

# For EventStream, we might need to mock it if it imports things that fail.
# If src.infrastructure.event_stream fails, we should mock it too.
# But let's try with just the langchain mock first.
from src.infrastructure.event_stream import EventStream, Event, EventType

# Setup in-memory DB for testing
@pytest.fixture(scope="function")
def test_db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create a session
    session = TestingSessionLocal()
    
    # We need to patch the SessionLocal used in the orchestrator module
    # We'll return a factory that returns our session, but better: 
    # since the code creates a new session each time: db = SessionLocal()
    # we need the mock to return our session.
    # However, if the code closes the session, we might have issues if we reuse it.
    # So we should let the factory create new sessions bound to the same engine.
    
    yield TestingSessionLocal
    
    Base.metadata.drop_all(engine)

@pytest.fixture
def mock_event_stream():
    stream = MagicMock(spec=EventStream)
    stream.subscribe.return_value = "sub-123"
    stream.emit_simple = AsyncMock()
    return stream

@pytest.fixture
def orchestrator(mock_event_stream, test_db_session, monkeypatch):
    # Patch the SessionLocal in the module
    monkeypatch.setattr(orchestrator_module, "SessionLocal", test_db_session)
    
    orch = WorkspaceOrchestrator(mock_event_stream)
    # Reset internal state if needed (though new instance is clean)
    return orch

@pytest.mark.asyncio
async def test_suggestion_generation(orchestrator, mock_event_stream):
    # Simulate 3 events in quick succession
    event1 = Event(event_type=EventType.AUDIT_STARTED, source="test", data={})
    event2 = Event(event_type=EventType.AUDIT_COMPLETED, source="test", data={})
    event3 = Event(event_type=EventType.SYSTEM_METRIC, source="test", data={})
    
    # Process events
    await orchestrator._handle_event(event1)
    await orchestrator._handle_event(event2)
    
    # Should not have suggestions yet
    assert len(orchestrator.list_pending()) == 0
    
    # Third event triggers suggestion
    await orchestrator._handle_event(event3)
    
    pending = orchestrator.list_pending()
    assert len(pending) == 1
    assert pending[0]["title"] == "New Workspace: Data‑Prep for Q4 Forecast"
    
    # Verify alert was emitted
    mock_event_stream.emit_simple.assert_called_once()
    assert mock_event_stream.emit_simple.call_args[1]["event_type"] == EventType.SYSTEM_ALERT

@pytest.mark.asyncio
async def test_accept_suggestion(orchestrator, mock_event_stream):
    # Trigger a suggestion
    for _ in range(3):
        await orchestrator._handle_event(Event(event_type=EventType.SYSTEM_METRIC, source="test"))
    
    pending = orchestrator.list_pending()
    suggestion_id = pending[0]["id"]
    
    # Accept it
    result = orchestrator.accept(suggestion_id)
    assert result is True
    
    # Should no longer be pending
    assert len(orchestrator.list_pending()) == 0

@pytest.mark.asyncio
async def test_dismiss_suggestion(orchestrator, mock_event_stream):
    # Trigger a suggestion
    for _ in range(3):
        await orchestrator._handle_event(Event(event_type=EventType.SYSTEM_METRIC, source="test"))
    
    pending = orchestrator.list_pending()
    suggestion_id = pending[0]["id"]
    
    # Dismiss it
    result = orchestrator.dismiss(suggestion_id)
    assert result is True
    
    # Should no longer be pending
    assert len(orchestrator.list_pending()) == 0

@pytest.mark.asyncio
async def test_no_double_suggestion(orchestrator, mock_event_stream):
    # Trigger a suggestion
    for _ in range(3):
        await orchestrator._handle_event(Event(event_type=EventType.SYSTEM_METRIC, source="test"))
        
    assert len(orchestrator.list_pending()) == 1
    
    # Trigger more events
    for _ in range(3):
        await orchestrator._handle_event(Event(event_type=EventType.SYSTEM_METRIC, source="test"))
        
    # Should still satisfy condition but not generate new one if one is pending
    assert len(orchestrator.list_pending()) == 1

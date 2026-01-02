
import pytest
import os
import uuid
from datetime import datetime
from src.db import init_db, SessionLocal
from src.models.orm import ORMPolicy, ORMAudit, ORMComplianceScore, ORMModel
from src.agents.policy_agent import PolicyAgent
from src.core.audit_repository import AuditRepository
from src.core.compliance_scoring import ComplianceRiskScorer, ComplianceRiskScore, ComplianceStatus
from src.core.models import PolicyDefinition, AgentConfig

# Initialize DB for tests
init_db()

@pytest.fixture
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

def test_policy_persistence():
    """Test that policies persist across agent re-initialization"""
    unique_id = f"test-policy-{uuid.uuid4()}"
    
    # 1. Create agent and add policy
    agent1 = PolicyAgent()
    policy = PolicyDefinition(
        id=unique_id,
        name="Persistence Test Policy",
        description="Testing persistence",
        category="test",
        rules=[{"text": "Rule 1"}]
    )
    agent1.add_policy(policy)
    
    # 2. Re-initialize agent (simulate restart)
    agent2 = PolicyAgent()
    
    # 3. Verify policy exists in new agent
    retrieved = agent2.get_policy(unique_id)
    assert retrieved is not None
    assert retrieved.id == unique_id
    assert retrieved.name == "Persistence Test Policy"
    
    # Clean up
    agent2.delete_policy(unique_id)

@pytest.mark.asyncio
async def test_audit_persistence():
    """Test that audits persist across repository re-initialization"""
    unique_id = f"test-audit-{uuid.uuid4()}"
    
    # 1. Create repo and save audit
    repo1 = AuditRepository()
    audit_data = {
        "audit_id": unique_id,
        "model_id": "test-model",
        "status": "compliant",
        "compliance_score": 95.0,
        "risk_score": 5.0,
        "saved_at": datetime.utcnow().isoformat()
    }
    await repo1.save_audit(audit_data)
    
    # 2. Re-initialize repo
    repo2 = AuditRepository()
    
    # 3. Verify audit exists
    retrieved = await repo2.get_audit(unique_id)
    assert retrieved is not None
    assert retrieved["audit_id"] == unique_id
    assert retrieved["compliance_score"] == 95.0
    
    # Clean up
    await repo2.delete_audit(unique_id)

@pytest.mark.asyncio
async def test_compliance_score_persistence():
    """Test that compliance scores persist across scorer re-initialization"""
    unique_model_id = f"test-model-{uuid.uuid4()}"
    unique_score_id = f"test-score-{uuid.uuid4()}"
    
    # 1. Create scorer and save score
    scorer1 = ComplianceRiskScorer()
    score = ComplianceRiskScore(
        id=unique_score_id,
        model_id=unique_model_id,
        overall_risk_score=10.0,
        overall_compliance_score=90.0,
        status=ComplianceStatus.COMPLIANT
    )
    # Manually adding to internal store and persisting to simulate calculation result
    scorer1._scores[unique_model_id] = score
    
    # Persist manually as calculate_score is complex to mock fully here
    db = SessionLocal()
    orm_score = ORMComplianceScore(
        id=score.id,
        model_id=score.model_id,
        overall_risk_score=score.overall_risk_score,
        overall_compliance_score=score.overall_compliance_score,
        status=score.status,
        domain_scores=[],
        framework_scores=[],
        recommendations={}
    )
    db.merge(orm_score)
    db.commit()
    db.close()
    
    # 2. Re-initialize scorer
    scorer2 = ComplianceRiskScorer()
    # verify it loaded history
    
    # 3. Verify score exists in history
    assert unique_model_id in scorer2._scores
    loaded_score = scorer2._scores[unique_model_id]
    assert loaded_score.overall_compliance_score == 90.0
    assert loaded_score.id == unique_score_id
    
    # Clean up
    db = SessionLocal()
    db.query(ORMComplianceScore).filter(ORMComplianceScore.id == unique_score_id).delete()
    db.commit()
    db.close()

@pytest.mark.asyncio
async def test_monitor_persistence():
    """Test that monitor alerts persist"""
    from src.agents.monitor_agent import MonitorAgent
    from src.models.orm import ORMAlert
    
    unique_alert_id = f"alert_{uuid.uuid4()}"
    
    # 1. Create agent and generate alert (simulated)
    agent1 = MonitorAgent()
    alert = {
        "id": unique_alert_id,
        "model_id": "test_model_monitor",
        "type": "drift",
        "severity": "high",
        "details": "Test drift detected"
    }
    # Manually append and persist since _generate_alert is internal/async complex
    # But let's use the private method if possible or just mimic what it does
    result = await agent1._generate_alert(alert)
    generated_id = result["id"]
    
    # 2. Re-initialize
    agent2 = MonitorAgent()
    
    # 3. Verify alert loaded
    # loaded = [a for a in agent2._alerts if a["id"] == unique_alert_id] # OLD
    loaded = [a for a in agent2._alerts if a["id"] == generated_id]
    assert len(loaded) == 1
    assert loaded[0]["severity"] == "high"
    
    # Clean up
    db = SessionLocal()
    db.query(ORMAlert).filter(ORMAlert.id == generated_id).delete()
    db.commit()
    db.close()

@pytest.mark.asyncio
async def test_learning_persistence():
    """Test that learning history persists"""
    from src.agents.learning_agent import LearningAgent
    from src.models.orm import ORMLearning
    
    unique_audit_id = f"audit_{uuid.uuid4()}"
    
    # 1. Create agent and learn
    agent1 = LearningAgent()
    params = {
        "audit_id": unique_audit_id,
        "results": [],
        "policies": []
    }
    # Mock LLM response for _learn_from_audit to allow it to proceed? 
    # Or just manually persist. _learn_from_audit calls invoke_llm.
    # We can't easily mock LLM here without patching.
    # Let's manually inject into DB to verify LOAD logic, which is the key persistence check.
    
    db = SessionLocal()
    learning = ORMLearning(
        audit_id=unique_audit_id,
        insights={"test": "insight"},
        timestamp=datetime.utcnow()
    )
    db.add(learning)
    db.commit()
    db.close()
    
    # 2. Initialize agent
    agent2 = LearningAgent()
    
    # 3. Verify loaded
    loaded = [l for l in agent2._learning_history if l["audit_id"] == unique_audit_id]
    assert len(loaded) == 1
    assert loaded[0]["insights"]["test"] == "insight"
    
    # Clean up
    db = SessionLocal()
    db.query(ORMLearning).filter(ORMLearning.audit_id == unique_audit_id).delete()
    db.commit()
    db.close()

@pytest.mark.asyncio
async def test_workflow_persistence():
    """Test that workflow executions persist"""
    from src.orchestration.workflow_engine import WorkflowEngine
    from src.models.orm import ORMWorkflowExecution
    from src.orchestration.workflow_engine import WorkflowStatus
    
    from src.db import init_db, engine
    init_db() # Force ensure tables exist
    
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"DEBUG: Tables in DB: {tables}")

    engine1 = WorkflowEngine()
    
    # 1. Create a dummy workflow
    workflow_id = "test_workflow"
    # We need to manually inject a workflow execution because execute_workflow is complex
    # But let's use the internal persistence method or just mock the state dict and call _persist
    
    execution_id = f"exec_{uuid.uuid4()}"
    state = {
        "workflow_id": workflow_id,
        "execution_id": execution_id,
        "status": "running",
        "current_node": "test_node",
        "messages": ["started"],
        "audit_state": {},
        "errors": [],
        "start_time": datetime.utcnow().isoformat(),
        "end_time": None,
        "metadata": {}
    }
    
    # Persist
    engine1._persist_execution(state)
    
    # 2. Re-initialize
    engine2 = WorkflowEngine()
    
    # 3. Retrieve status
    loaded_state = engine2.get_execution_status(execution_id)
    
    assert loaded_state is not None, f"Failed to load execution {execution_id}"
    assert loaded_state["workflow_id"] == workflow_id
    assert loaded_state["status"] == "running"
    
    # Clean up
    db = SessionLocal()
    db.query(ORMWorkflowExecution).filter(ORMWorkflowExecution.execution_id == execution_id).delete()
    db.commit()
    db.close()

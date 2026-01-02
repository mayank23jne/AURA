
import asyncio
import sys
import os
import uuid
from dotenv import load_dotenv
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.getcwd())
load_dotenv()

from src.db import get_engine, SessionLocal
from src.core.model_registry import ModelRegistry, ModelConfig, ModelProvider, ModelType, ModelStatus
from src.core.audit_repository import AuditRepository
from src.agents.audit_agent import AuditAgent
from src.core.models import AgentConfig
from sqlalchemy import text

async def verify_all():
    print("🚀 Starting Comprehensive Feature Verification...")
    results = {"mysql": False, "poe": False, "audit_persistence": False}

    # 1. Verify MySQL
    print("\n[1/3] Verifying MySQL Database...")
    try:
        engine = get_engine()
        if engine.dialect.name == 'mysql':
            print("   ✅ Engine is configured for MySQL")
            with engine.connect() as conn:
                ver = conn.execute(text("SELECT VERSION()")).fetchone()[0]
                print(f"   ✅ Connected! Server Version: {ver}")
            results["mysql"] = True
        else:
            print(f"   ❌ Engine is {engine.dialect.name}, expected mysql")
    except Exception as e:
        print(f"   ❌ Database Error: {e}")

    # 2. Verify Poe Integration in Registry
    print("\n[2/3] Verifying Poe Model Support...")
    try:
        registry = ModelRegistry()
        poe_config = ModelConfig(
            name=f"verify-poe-{uuid.uuid4().hex[:6]}",
            model_type=ModelType.API,
            provider=ModelProvider.POE,
            model_name="Claude-3-Opus",
            api_key="mock-key",
            status=ModelStatus.ACTIVE # Ensure active
        )
        
        # Test Registration
        model_id = registry.register(poe_config)
        print(f"   ✅ Registered Poe model with ID: {model_id}")
        
        # Test Provider Logic (Mocking the HTTP call to avoid actual API error)
        # We need to mock httpx.AsyncClient.post to be an async function returning a response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Poe says hello"}}],
            "usage": {}
        }

        # Create an async mock for the post method
        async_mock = AsyncMock(return_value=mock_response)
        
        with patch('httpx.AsyncClient.post', side_effect=async_mock) as mock_post:
            
            response = await registry.invoke(model_id, "Hello")
            
            # Verify it used the correct URL
            call_args = mock_post.call_args
            if call_args and "api.poe.com" in call_args[0][0]:
                 print("   ✅ Poe API URL verified")
            
            if response.get("response") == "Poe says hello":
                print("   ✅ Poe Invocation Logic verified (Mocked)")
                results["poe"] = True
            else:
                print(f"   ❌ Invocation unexpected response: {response}")
                
        # Cleanup
        registry.unregister(model_id)
        
    except Exception as e:
        print(f"   ❌ Registry Error: {e}")

    # 3. Verify Audit Persistence
    print("\n[3/3] Verifying Audit Agent Persistence...")
    try:
        # Mock dependencies for Agent
        agent_config = AgentConfig(name="audit_verifier", llm_provider="openai")
        agent = AuditAgent(agent_config)
        agent.invoke_llm = AsyncMock(return_value='{"passed": true, "score": 1.0, "details": "simulated"}')
        
        # Manually create a "completed" audit result structure to test SAVE only
        # (avoiding running full audit flow which requires more mocks)
        audit_id = f"audit-{uuid.uuid4()}"
        dummy_result = {
            "audit_id": audit_id,
            "model_id": "test-model",
            "compliance_score": 0.95,
            "total_tests": 10,
            "passed": 9,
            "failed": 1,
            "status": "completed",
            "start_time": "2024-01-01T00:00:00",
            "end_time": "2024-01-01T00:01:00",
            "results": []
        }
        
        # Test saving capability
        repo = AuditRepository()
        await repo.save_audit(dummy_result)
        print("   ✅ Called save_audit()")
        
        # Verify it's in DB
        saved = await repo.get_audit(audit_id)
        if saved and saved["audit_id"] == audit_id:
            print(f"   ✅ Persistence Confirmed: Found audit {audit_id} in DB")
            results["audit_persistence"] = True
        else:
            print("   ❌ Save failed: Could not retrieve audit")
            
    except Exception as e:
        # If it fails due to AsyncMock import (python < 3.8/mocks)
        if "AsyncMock" in str(e) or "name 'AsyncMock' is not defined" in str(e):
             # Simple patch fallback for test
             print("   ⚠️  (Skipping execution flow, verifying DB write directly)")
             # We just tested DB write via Repository above actually.
    except ImportError:
        pass

    print("\n--- Summary ---")
    all_passed = all(results.values())
    print(f"Overall Result: {'✅ PASSED' if all_passed else '❌ FAILED'}")
    return all_passed

class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)

if __name__ == "__main__":
    success = asyncio.run(verify_all())
    if not success:
        sys.exit(1)

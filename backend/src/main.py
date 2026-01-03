"""Main entry point for AURA Agentic Platform"""

import asyncio
import signal
import sys
from dotenv import load_dotenv

load_dotenv()
from typing import Any, Dict, List, Optional

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Response, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import get_settings
from src.agents import (
    AnalysisAgent,
    AuditAgent,
    LearningAgent,
    MonitorAgent,
    OrchestratorAgent,
    PolicyAgent,
    RemediationAgent,
    ReportAgent,
    TestingAgent,
    CollectiveIntelligenceCoordinator,
)
from src.core.models import (
    AgentConfig,
    TaskRequest,
    PackageDefinition,
    PackageCreateRequest,
    PackageUpdateRequest,
)
from src.db import SessionLocal
from src.models.orm import ORMPackage
from src.core.model_registry import (
    get_model_registry,
    ModelConfig,
    ModelType,
    ModelProvider,
    ModelStatus,
)
from src.infrastructure.event_stream import EventStream
from src.infrastructure.message_bus import InMemoryMessageBus
from src.knowledge.knowledge_base import InMemoryKnowledgeBase
from src.orchestration.scheduler import AuditScheduler
from src.orchestration.workflow_engine import WorkflowEngine
from src.utils.logging import setup_logging
from src.core.audit_repository import get_audit_repository
from src.dashboard.components.reports import generate_report_pdf
import pandas as pd
import io
import csv

logger = structlog.get_logger()


class AURAPlatform:
    """
    Main AURA Agentic Platform orchestrator.

    Initializes and manages all platform components including:
    - Agent ecosystem
    - Message bus
    - Knowledge base
    - Event stream
    - Workflow engine
    - Scheduler
    """

    def __init__(self):
        self.settings = get_settings()
        self._running = False

        # Core infrastructure
        self.message_bus = InMemoryMessageBus()
        self.knowledge_base = InMemoryKnowledgeBase()
        self.event_stream = EventStream()
        self.workflow_engine = WorkflowEngine()
        self.scheduler = AuditScheduler()

        # Agents
        self.agents: Dict[str, any] = {}

        logger.info("AURA Platform initialized")

    async def start(self):
        """Start the AURA platform"""
        logger.info("Starting AURA Platform...")

        # Start infrastructure
        await self.message_bus.start()
        await self.event_stream.start()
        await self.scheduler.start()

        # Initialize agents
        await self._init_agents()

        self._running = True
        logger.info("AURA Platform started successfully")

    async def stop(self):
        """Stop the AURA platform"""
        logger.info("Stopping AURA Platform...")

        self._running = False

        # Shutdown agents
        for name, agent in self.agents.items():
            await agent.shutdown()

        # Stop infrastructure
        await self.scheduler.stop()
        await self.event_stream.stop()
        await self.message_bus.stop()

        logger.info("AURA Platform stopped")

    async def _init_agents(self):
        """Initialize all platform agents"""
        settings = self.settings

        # Create agent configurations
        default_config = {
            "llm_model": settings.llm.default_model,
            "llm_provider": settings.llm.default_provider,
            "rate_limit_rpm": settings.agent.rate_limit_rpm,
            "retry_attempts": settings.agent.retry_attempts,
            "timeout_seconds": settings.agent.timeout_seconds,
        }

        # Initialize core agents
        agents_to_create = [
            ("orchestrator", OrchestratorAgent),
            ("policy", PolicyAgent),
            ("audit", AuditAgent),
            ("testing", TestingAgent),
            ("analysis", AnalysisAgent),
            ("learning", LearningAgent),
            ("monitor", MonitorAgent),
            ("report", ReportAgent),
            ("remediation", RemediationAgent),
            ("collective_intelligence", CollectiveIntelligenceCoordinator),
        ]

        for name, agent_class in agents_to_create:
            config = AgentConfig(name=name, **default_config)
            agent = agent_class(config)

            # Inject dependencies
            agent.message_queue = self.message_bus
            agent.knowledge_base = self.knowledge_base

            self.agents[name] = agent

            # Register with orchestrator
            if name != "orchestrator":
                self.agents["orchestrator"].register_agent(name, agent.id)

            logger.info(f"Agent initialized: {name}", agent_id=agent.id)

        # Pass all agents to orchestrator for workflow execution
        orchestrator = self.agents.get("orchestrator")
        if orchestrator:
            orchestrator.set_platform_agents(self.agents)

    def get_agent(self, name: str):
        """Get an agent by name"""
        return self.agents.get(name)

    async def run_audit(
        self,
        model_id: str,
        policy_ids: list = None,
        test_count: int = None,
        frameworks: list = None,
    ) -> dict:
        """Run a compliance audit"""
        orchestrator = self.agents.get("orchestrator")
        if not orchestrator:
            raise RuntimeError("Orchestrator agent not initialized")

        task = TaskRequest(
            task_id=f"audit_{model_id}",
            task_type="start_audit",
            description=f"Run compliance audit for model {model_id}",
            parameters={
                "model_id": model_id,
                "policy_ids": policy_ids or [],
                "test_count": test_count or self.settings.audit.default_test_count,
                "frameworks": frameworks or ["aura-native"],
            },
            requester="api",
        )

        response = await orchestrator.process_task(task)
        return response.result

    def get_status(self) -> dict:
        """Get platform status"""
        return {
            "running": self._running,
            "agents": {
                name: {
                    "id": agent.id,
                    "status": agent.status.value,
                    "metrics": agent.get_metrics().model_dump(),
                }
                for name, agent in self.agents.items()
            },
            "message_bus": {
                "queue_depths": {
                    topic: self.message_bus.get_queue_depth(topic)
                    for topic in self.agents.keys()
                }
            },
            "knowledge_base": self.knowledge_base.get_stats(),
            "event_stream": self.event_stream.get_stats(),
            "scheduler": self.scheduler.get_stats(),
        }


# Global platform instance
platform: Optional[AURAPlatform] = None


# FastAPI app
app = FastAPI(
    title="AURA Agentic Platform",
    description="Autonomous AI Governance System",
    version="0.1.0",
)

# CORS middleware
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API
class AuditRequest(BaseModel):
    model_id: str
    policy_ids: list = []
    test_count: int = 100
    frameworks: list = ["aura-native"]


class PolicyGenerationRequest(BaseModel):
    regulation_text: str
    regulation_name: str


class TaskRequestModel(BaseModel):
    agent_name: str
    task_type: str
    description: str = ""
    parameters: dict = {}


class ModelRegistrationRequest(BaseModel):
    name: str
    description: str = ""
    model_type: str = "api"  # api, ollama, huggingface, uploaded
    provider: str = "openai"  # openai, anthropic, ollama, huggingface, custom
    endpoint_url: str = ""
    api_key: str = ""
    model_name: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    tags: list = []


class ModelUpdateRequest(BaseModel):
    name: str = None
    description: str = None
    endpoint_url: str = None
    api_key: str = None
    model_name: str = None
    temperature: float = None
    max_tokens: int = None
    status: str = None
    tags: list = None


class ModelInvokeRequest(BaseModel):
    prompt: str
    max_tokens: int = None
    temperature: float = None


# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize platform on startup"""
    global platform
    
    # Initialize DB tables
    from src.db import init_db
    init_db()
    
    setup_logging(settings.monitoring.log_level, settings.monitoring.log_format)
    platform = AURAPlatform()
    await platform.start()
    # Initialize Workspace Orchestrator and store in app state for API access
    from src.orchestration.workspace_orchestrator import get_workspace_orchestrator
    orchestrator = get_workspace_orchestrator(platform.event_stream)
    app.state.workspace_orchestrator = orchestrator
    # Include the workspace suggestion router
    from src.api.workspaces import router as workspaces_router
    app.include_router(workspaces_router)
    # Include the websocket router for streaming suggestions
    from src.api.websocket import router as websocket_router
    app.include_router(websocket_router)



@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global platform
    if platform:
        await platform.stop()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "AURA Agentic Platform",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "platform_running": platform._running if platform else False}


@app.get("/status")
async def get_status():
    """Get platform status"""
    if not platform:
        raise HTTPException(status_code=500, detail="Platform not initialized")
    return platform.get_status()


@app.post("/audit")
async def run_audit(request: AuditRequest):
    """Run a compliance audit"""
    if not platform:
        raise HTTPException(status_code=500, detail="Platform not initialized")

    result = await platform.run_audit(
        model_id=request.model_id,
        policy_ids=request.policy_ids,
        test_count=request.test_count,
        frameworks=request.frameworks,
    )

    # Save audit to persistent storage
    try:
        audit_repo = get_audit_repository()
        await audit_repo.save_audit(result)
        logger.info("Audit saved to repository", audit_id=result.get("audit_id"))
    except Exception as e:
        logger.error("Failed to save audit to repository", error=str(e))
        # Don't fail the request if saving fails

    return result


@app.get("/audits")
async def list_audits(
    model_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """List audit history with optional filters"""
    try:
        audit_repo = get_audit_repository()
        audits = await audit_repo.list_audits(
            model_id=model_id,
            status=status,
            limit=limit,
            offset=offset
        )
        return {
            "audits": audits,
            "count": len(audits),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error("Failed to list audits", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list audits: {str(e)}")


@app.get("/audits/{audit_id}")
async def get_audit(audit_id: str):
    """Get a specific audit by ID"""
    try:
        audit_repo = get_audit_repository()
        audit = await audit_repo.get_audit(audit_id)

        if not audit:
            raise HTTPException(status_code=404, detail=f"Audit {audit_id} not found")

        return audit
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get audit", error=str(e), audit_id=audit_id)
        raise HTTPException(status_code=500, detail=f"Failed to get audit: {str(e)}")


@app.get("/audits/{audit_id}/pdf")
async def get_audit_pdf(audit_id: str):
    """Get a PDF version of the audit report"""
    try:
        audit_repo = get_audit_repository()
        audit = await audit_repo.get_audit(audit_id)
        print('audit report report', audit)
        if not audit:
            raise HTTPException(status_code=404, detail=f"Audit {audit_id} not found")

        pdf_bytes = generate_report_pdf(audit)

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=audit-report-{audit_id}.pdf"
            }
        )
    except Exception as e:
        logger.error("Failed to generate audit PDF", error=str(e), audit_id=audit_id)
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")


@app.post("/policies/generate")
async def generate_policies(request: PolicyGenerationRequest):
    """Generate policies from regulatory text"""
    if not platform:
        raise HTTPException(status_code=500, detail="Platform not initialized")

    policy_agent = platform.get_agent("policy")
    if not policy_agent:
        raise HTTPException(status_code=500, detail="Policy agent not available")

    task = TaskRequest(
        task_id="policy_generation",
        task_type="generate_from_regulation",
        description="Generate policies from regulation",
        parameters={
            "regulation_text": request.regulation_text,
            "regulation_name": request.regulation_name,
        },
        requester="api",
    )

    response = await policy_agent.process_task(task)
    return response.result


@app.get("/policies")
async def list_policies():
    """List all policies"""
    if not platform:
        raise HTTPException(status_code=500, detail="Platform not initialized")

    policy_agent = platform.get_agent("policy")
    if not policy_agent:
        raise HTTPException(status_code=500, detail="Policy agent not available")

    policies = policy_agent.list_policies()
    return {"policies": [p.model_dump() for p in policies]}


class PolicyCreateRequest(BaseModel):
    """Request model for creating a policy"""
    id: str
    name: str
    description: str
    category: str
    severity: str
    package_id: Optional[str] = None
    rules: List[Dict[str, Any]] = []
    version: str = "1.0.0"
    active: bool = True
    regulatory_references: List[str] = []


class PolicyUpdateRequest(BaseModel):
    """Request model for updating a policy"""
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    severity: Optional[str] = None
    package_id: Optional[str] = None
    rules: Optional[List[Dict[str, Any]]] = None
    version: Optional[str] = None
    active: Optional[bool] = None
    regulatory_references: Optional[List[str]] = None


@app.post("/policies")
async def create_policy(request: PolicyCreateRequest):
    """Create a new policy manually"""
    if not platform:
        raise HTTPException(status_code=500, detail="Platform not initialized")

    policy_agent = platform.get_agent("policy")
    if not policy_agent:
        raise HTTPException(status_code=500, detail="Policy agent not available")

    # Check if policy ID already exists
    existing = policy_agent.get_policy(request.id)
    if existing:
        raise HTTPException(status_code=400, detail=f"Policy with ID '{request.id}' already exists")

    # Create PolicyDefinition
    from src.core.models import PolicyDefinition
    policy = PolicyDefinition(
        id=request.id,
        name=request.name,
        description=request.description,
        category=request.category,
        severity=request.severity,
        package_id=request.package_id,
        rules=request.rules,
        version=request.version,
        active=request.active,
        regulatory_references=request.regulatory_references,
    )

    policy_agent.add_policy(policy)
    return {"message": "Policy created successfully", "policy": policy.model_dump()}


@app.get("/policies/{policy_id}")
async def get_policy(policy_id: str):
    """Get a specific policy by ID"""
    if not platform:
        raise HTTPException(status_code=500, detail="Platform not initialized")

    policy_agent = platform.get_agent("policy")
    if not policy_agent:
        raise HTTPException(status_code=500, detail="Policy agent not available")

    policy = policy_agent.get_policy(policy_id)
    if not policy:
        raise HTTPException(status_code=404, detail=f"Policy not found: {policy_id}")

    return {"policy": policy.model_dump()}


@app.put("/policies/{policy_id}")
async def update_policy(policy_id: str, request: PolicyUpdateRequest):
    """Update an existing policy"""
    if not platform:
        raise HTTPException(status_code=500, detail="Platform not initialized")

    policy_agent = platform.get_agent("policy")
    if not policy_agent:
        raise HTTPException(status_code=500, detail="Policy agent not available")

    # Convert request to dict, excluding None values
    updates = {k: v for k, v in request.model_dump().items() if v is not None}

    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")

    updated_policy = policy_agent.update_policy(policy_id, updates)
    if not updated_policy:
        raise HTTPException(status_code=404, detail=f"Policy not found: {policy_id}")

    return {"message": "Policy updated successfully", "policy": updated_policy.model_dump()}


@app.delete("/policies/{policy_id}")
async def delete_policy(policy_id: str):
    """Delete a policy"""
    if not platform:
        raise HTTPException(status_code=500, detail="Platform not initialized")

    policy_agent = platform.get_agent("policy")
    if not policy_agent:
        raise HTTPException(status_code=500, detail="Policy agent not available")

    success = policy_agent.delete_policy(policy_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Policy not found: {policy_id}")

    return {"message": f"Policy '{policy_id}' deleted successfully"}


@app.post("/policies/upload")
async def upload_policies(file: UploadFile = File(...)):
    """Upload policies from Excel or CSV file"""
    if not platform:
        raise HTTPException(status_code=500, detail="Platform not initialized")

    policy_agent = platform.get_agent("policy")
    if not policy_agent:
        raise HTTPException(status_code=500, detail="Policy agent not available")

    content = await file.read()
    filename = file.filename.lower()
    
    policies_to_add = []
    
    try:
        if filename.endswith('.csv'):
            stream = io.StringIO(content.decode('utf-8'))
            reader = csv.DictReader(stream)
            for row in reader:
                policies_to_add.append(row)
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(content))
            policies_to_add = df.to_dict(orient='records')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload .csv, .xls, or .xlsx")

        created_count = 0
        from src.core.models import PolicyDefinition
        
        for p_data in policies_to_add:
            # Map fields (handle variations in column names)
            p_id = p_data.get('policy_id') or p_data.get('id') or f"up_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{created_count}"
            name = p_data.get('policy_name') or p_data.get('name', 'Unnamed Policy')
            desc = p_data.get('description', '')
            cat = p_data.get('category', 'general')
            sev = p_data.get('severity', 'medium')
            ver = str(p_data.get('version', '1.0.0'))
            active = str(p_data.get('active', 'true')).lower() == 'true'
            
            # Rules handling
            rules_raw = p_data.get('rules', '')
            rules = []
            if isinstance(rules_raw, str) and rules_raw:
                # Semicolon separated rules
                rule_lines = rules_raw.split(';')
                for i, r_text in enumerate(rule_lines):
                    if r_text.strip():
                        rules.append({"id": f"{p_id}-r-{i+1}", "text": r_text.strip(), "severity": sev})
            elif isinstance(rules_raw, list):
                rules = rules_raw

            policy = PolicyDefinition(
                id=str(p_id),
                name=str(name),
                description=str(desc),
                category=str(cat),
                severity=str(sev),
                version=ver,
                active=active,
                rules=rules,
                regulatory_references=p_data.get('regulatory_references') or []
            )
            policy_agent.add_policy(policy)
            created_count += 1

        return {"message": f"Successfully imported {created_count} policies", "count": created_count}

    except Exception as e:
        logger.error("Failed to upload policies", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.post("/tasks")
async def submit_task(request: TaskRequestModel):
    """Submit a task to a specific agent"""
    if not platform:
        raise HTTPException(status_code=500, detail="Platform not initialized")

    agent = platform.get_agent(request.agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {request.agent_name}")

    task = TaskRequest(
        task_id=f"task_{request.task_type}",
        task_type=request.task_type,
        description=request.description,
        parameters=request.parameters,
        requester="api",
    )

    response = await agent.process_task(task)
    return response.model_dump()


@app.get("/agents")
async def list_agents():
    """List all agents"""
    if not platform:
        raise HTTPException(status_code=500, detail="Platform not initialized")

    return {
        "agents": [
            {
                "name": name,
                "id": agent.id,
                "status": agent.status.value,
            }
            for name, agent in platform.agents.items()
        ]
    }


@app.get("/agents/{agent_name}/metrics")
async def get_agent_metrics(agent_name: str):
    """Get metrics for a specific agent"""
    if not platform:
        raise HTTPException(status_code=500, detail="Platform not initialized")

    agent = platform.get_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_name}")

    return agent.get_metrics().model_dump()


# Model Management Endpoints
@app.get("/models")
async def list_models(
    model_type: str = None,
    provider: str = None,
    status: str = None
):
    """List all registered models"""
    registry = get_model_registry()

    # Convert string params to enums if provided
    type_filter = ModelType(model_type) if model_type else None
    provider_filter = ModelProvider(provider) if provider else None
    status_filter = ModelStatus(status) if status else None

    models = registry.list_models(
        model_type=type_filter,
        provider=provider_filter,
        status=status_filter
    )

    return {
        "models": [m.model_dump() for m in models],
        "total": len(models)
    }


@app.get("/models/stats")
async def get_model_stats():
    """Get model registry statistics"""
    registry = get_model_registry()
    return registry.get_stats()


@app.post("/models")
async def register_model(request: ModelRegistrationRequest):
    """Register a new model"""
    registry = get_model_registry()

    config = ModelConfig(
        name=request.name,
        description=request.description,
        model_type=ModelType(request.model_type),
        provider=ModelProvider(request.provider),
        endpoint_url=request.endpoint_url or None,
        api_key=request.api_key or None,
        model_name=request.model_name,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        tags=request.tags,
        status=ModelStatus.INACTIVE
    )

    model_id = registry.register(config)
    return {"model_id": model_id, "status": "registered"}


@app.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get model details"""
    registry = get_model_registry()
    model = registry.get(model_id)

    if not model:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    return model.model_dump()


@app.put("/models/{model_id}")
async def update_model(model_id: str, request: ModelUpdateRequest):
    """Update model configuration"""
    registry = get_model_registry()

    updates = {k: v for k, v in request.model_dump().items() if v is not None}

    if "status" in updates:
        updates["status"] = ModelStatus(updates["status"])

    model = registry.update(model_id, updates)

    if not model:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    return model.model_dump()


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a model"""
    registry = get_model_registry()

    if not registry.unregister(model_id):
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    return {"status": "deleted", "model_id": model_id}


@app.post("/models/{model_id}/test")
async def test_model(model_id: str):
    """Test model connectivity and response"""
    registry = get_model_registry()

    model = registry.get(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    result = await registry.test_model(model_id)
    return result.model_dump()


@app.post("/models/{model_id}/invoke")
async def invoke_model(model_id: str, request: ModelInvokeRequest):
    """Invoke a model with a prompt"""
    registry = get_model_registry()

    model = registry.get(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    kwargs = {}
    if request.max_tokens:
        kwargs["max_tokens"] = request.max_tokens
    if request.temperature:
        kwargs["temperature"] = request.temperature

    result = await registry.invoke(model_id, request.prompt, **kwargs)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.post("/models/{model_id}/upload")
async def upload_model_file(model_id: str):
    """Upload a model file (placeholder - needs file upload handling)"""
    registry = get_model_registry()

    model = registry.get(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    # Note: Full file upload implementation would need:
    # from fastapi import File, UploadFile
    # async def upload_model_file(model_id: str, file: UploadFile = File(...)):

    return {
        "status": "placeholder",
        "message": "File upload endpoint - implement with multipart form data",
        "model_id": model_id
    }


# Packages CRUD

@app.get("/packages")
async def list_packages():
    """List all packages"""
    db = SessionLocal()
    try:
        packages = db.query(ORMPackage).all()
        result = []
        for p in packages:
             d = p.to_dict()
             d["policies_count"] = len(p.policies) if p.policies else 0
             result.append(d)
        return {"packages": result}
    finally:
        db.close()


@app.post("/packages")
async def create_package(request: PackageCreateRequest):
    """Create a new package"""
    db = SessionLocal()
    try:
        existing = db.query(ORMPackage).filter(ORMPackage.id == request.id).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"Package '{request.id}' already exists")
        
        pkg = ORMPackage(
            id=request.id,
            name=request.name,
            description=request.description
        )
        db.add(pkg)
        db.commit()
        db.refresh(pkg)
        return {"message": "Package created", "package": pkg.to_dict()}
    finally:
        db.close()


@app.get("/packages/{package_id}")
async def get_package(package_id: str):
    """Get package details"""
    db = SessionLocal()
    try:
        pkg = db.query(ORMPackage).filter(ORMPackage.id == package_id).first()
        if not pkg:
            raise HTTPException(status_code=404, detail="Package not found")
        d = pkg.to_dict()
        d["policies_count"] = len(pkg.policies) if pkg.policies else 0
        return {"package": d}
    finally:
        db.close()


@app.put("/packages/{package_id}")
async def update_package(package_id: str, request: PackageUpdateRequest):
    """Update a package"""
    db = SessionLocal()
    try:
        pkg = db.query(ORMPackage).filter(ORMPackage.id == package_id).first()
        if not pkg:
            raise HTTPException(status_code=404, detail="Package not found")
        
        if request.name is not None:
            pkg.name = request.name
        if request.description is not None:
            pkg.description = request.description
            
        db.commit()
        db.refresh(pkg)
        return {"message": "Package updated", "package": pkg.to_dict()}
    finally:
        db.close()


@app.delete("/packages/{package_id}")
async def delete_package(package_id: str):
    """Delete a package"""
    db = SessionLocal()
    try:
        pkg = db.query(ORMPackage).filter(ORMPackage.id == package_id).first()
        if not pkg:
             raise HTTPException(status_code=404, detail="Package not found")
        
        # Unlink policies before delete
        if pkg.policies:
             for p in pkg.policies:
                 p.package_id = None
             db.commit()

        db.delete(pkg)
        db.commit()
        return {"message": "Package deleted"}
    finally:
        db.close()


def main():
    """Main entry point"""
    settings = get_settings()
    setup_logging(settings.monitoring.log_level, settings.monitoring.log_format)

    logger.info(
        "Starting AURA Agentic Platform",
        host=settings.api.host,
        port=settings.api.port,
    )

    uvicorn.run(
        "src.main:app",
        host=settings.api.host,
        port=settings.api.port,
        workers=settings.api.workers,
        reload=settings.api.reload,
    )


if __name__ == "__main__":
    main()

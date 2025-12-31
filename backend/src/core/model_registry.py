"""Model Registry for AURA Platform

Supports:
- API-based models (OpenAI, Anthropic, custom endpoints)
- Local models (Ollama, llama.cpp)
- Uploaded model files (ONNX, PyTorch, SafeTensors)
"""

import asyncio
import os
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import structlog
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from src.db import SessionLocal
from src.models.orm import ORMModel

logger = structlog.get_logger()


class ModelType(str, Enum):
    """Types of models supported"""
    API = "api"  # External API endpoints
    OLLAMA = "ollama"  # Local Ollama models
    HUGGINGFACE = "huggingface"  # HuggingFace models
    UPLOADED = "uploaded"  # Uploaded model files


class ModelProvider(str, Enum):
    """Model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    POE = "poe"
    CUSTOM = "custom"


class ModelStatus(str, Enum):
    """Model status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TESTING = "testing"
    ERROR = "error"


class ModelConfig(BaseModel):
    """Configuration for a registered model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    description: str = ""
    model_type: ModelType
    provider: ModelProvider

    # Connection settings
    endpoint_url: Optional[str] = None
    api_key: Optional[str] = None
    model_name: str = ""  # e.g., "gpt-4", "llama2", "mistral"

    # For uploaded models
    file_path: Optional[str] = None
    file_format: Optional[str] = None  # onnx, pt, safetensors

    # Model parameters
    temperature: float = 0.7
    max_tokens: int = 8096
    timeout_seconds: int = 60

    # Metadata
    status: ModelStatus = ModelStatus.INACTIVE
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_tested: Optional[datetime] = None
    test_result: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

    # Usage tracking
    total_requests: int = 0
    total_tokens: int = 0
    avg_latency_ms: float = 0.0


class ModelTestResult(BaseModel):
    """Result of testing a model"""
    model_id: str
    success: bool
    latency_ms: float
    response: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelRegistry:
    """
    Central registry for managing models in AURA platform.

    Supports registering, testing, and invoking various model types.
    """

    def __init__(self, storage_path: str = "./data/models"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize with default models
        self._init_default_models()

        logger.info("Model registry initialized with Database backend")

    def _init_default_models(self):
        """Initialize registry with default API models"""
        db = SessionLocal()
        try:
            # Check if any models exist
            if db.query(ORMModel).count() > 0:
                return

            # Get API keys from environment variables
            openai_key = os.getenv("OPENAI_API_KEY", "")
            anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")

            default_models = [
                ModelConfig(
                    id="openai-gpt4",
                    name="GPT-4",
                    description="OpenAI GPT-4 model - requires OPENAI_API_KEY env var",
                    model_type=ModelType.API,
                    provider=ModelProvider.OPENAI,
                    model_name="gpt-4",
                    api_key=openai_key if openai_key else None,
                    status=ModelStatus.ACTIVE if openai_key else ModelStatus.INACTIVE,
                    tags=["llm", "openai", "production"]
                ),
                ModelConfig(
                    id="openai-gpt35",
                    name="GPT-3.5 Turbo",
                    description="OpenAI GPT-3.5 Turbo - requires OPENAI_API_KEY env var",
                    model_type=ModelType.API,
                    provider=ModelProvider.OPENAI,
                    model_name="gpt-3.5-turbo",
                    api_key=openai_key if openai_key else None,
                    status=ModelStatus.ACTIVE if openai_key else ModelStatus.INACTIVE,
                    tags=["llm", "openai", "fast", "cheap"]
                ),
                ModelConfig(
                    id="anthropic-claude",
                    name="Claude 3 Sonnet",
                    description="Anthropic Claude 3 Sonnet - requires ANTHROPIC_API_KEY env var",
                    model_type=ModelType.API,
                    provider=ModelProvider.ANTHROPIC,
                    model_name="claude-3-sonnet-20240229",
                    api_key=anthropic_key if anthropic_key else None,
                    status=ModelStatus.ACTIVE if anthropic_key else ModelStatus.INACTIVE,
                    tags=["llm", "anthropic", "production"]
                ),
                ModelConfig(
                    id="nexi-llm",
                    name="Nexi LLM",
                    description="NexiLLM IAW Llama 3.2 3B Instruct model for compliance",
                    model_type=ModelType.API,
                    provider=ModelProvider.CUSTOM,
                    model_name="NanoMatriX/NexiLLM-IAW-Llama-3.2-3B-Instruct-SFT-v0.1-GPTQ-INT4",
                    endpoint_url="http://ec2-35-175-202-40.compute-1.amazonaws.com/tokens_generator/api/v1/chat/completions",
                    status=ModelStatus.ACTIVE,
                    tags=["llm", "nexi", "compliance", "llama"]
                ),
            ]

            for model in default_models:
                orm_model = ORMModel(**model.model_dump())
                db.add(orm_model)
            
            db.commit()

            logger.info("Default models initialized",
                    total=len(default_models),
                    openai_configured=bool(openai_key),
                    anthropic_configured=bool(anthropic_key))
        except Exception as e:
            db.rollback()
            logger.error("Failed to init default models", error=str(e))
        finally:
            db.close()

    def register(self, config: ModelConfig) -> str:
        """Register a new model"""
        print('register model', config)

        if not config.id:
            config.id = str(uuid.uuid4())[:8]

        db = SessionLocal()
        try:
            # Check if exists
            existing = db.query(ORMModel).filter(ORMModel.id == config.id).first()
            if existing:
                # Update existing
                for key, value in config.model_dump().items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
            else:
                # Create new
                orm_model = ORMModel(**config.model_dump())
                db.add(orm_model)
            
            db.commit()
            logger.info("Model registered", model_id=config.id, name=config.name)
            return config.id
        except Exception as e:
            db.rollback()
            logger.error("Failed to register model", error=str(e))
            raise
        finally:
            db.close()

    def unregister(self, model_id: str) -> bool:
        """Unregister a model"""
        db = SessionLocal()
        try:
            model = db.query(ORMModel).filter(ORMModel.id == model_id).first()
            if model:
                # Delete uploaded file if exists
                if model.file_path and os.path.exists(model.file_path):
                    os.remove(model.file_path)

                db.delete(model)
                db.commit()
                logger.info("Model unregistered", model_id=model_id)
                return True
            return False
        except Exception as e:
            db.rollback()
            logger.error("Failed to unregister model", error=str(e))
            return False
        finally:
            db.close()

    def get(self, model_id: str) -> Optional[ModelConfig]:
        """Get a model by ID"""
        db = SessionLocal()
        try:
            orm_model = db.query(ORMModel).filter(ORMModel.id == model_id).first()
            if orm_model:
                return ModelConfig(**orm_model.to_dict())
            return None
        finally:
            db.close()

    def list_models(self,
                    model_type: Optional[ModelType] = None,
                    provider: Optional[ModelProvider] = None,
                    status: Optional[ModelStatus] = None,
                    tags: Optional[List[str]] = None) -> List[ModelConfig]:
        """List models with optional filtering"""
        db = SessionLocal()
        try:
            query = db.query(ORMModel)

            if model_type:
                query = query.filter(ORMModel.model_type == model_type)

            if provider:
                query = query.filter(ORMModel.provider == provider)

            if status:
                query = query.filter(ORMModel.status == status)

            # Filtering by tags in JSON is database-specific and tricky with simple ORM.
            # We'll filter in Python for now to keep it DB-agnostic (SQLite/MySQL).
            orms = query.all()
            
            models = [ModelConfig(**m.to_dict()) for m in orms]

            if tags:
                models = [m for m in models if any(t in m.tags for t in tags)]

            return models
        finally:
            db.close()

    def update(self, model_id: str, updates: Dict[str, Any]) -> Optional[ModelConfig]:
        """Update model configuration"""
        db = SessionLocal()
        try:
            model = db.query(ORMModel).filter(ORMModel.id == model_id).first()
            if not model:
                return None

            for key, value in updates.items():
                if hasattr(model, key):
                    setattr(model, key, value)

            db.commit()
            logger.info("Model updated", model_id=model_id)
            return ModelConfig(**model.to_dict())
        except Exception as e:
            db.rollback()
            logger.error("Failed to update model", error=str(e))
            return None
        finally:
            db.close()

    async def test_model(self, model_id: str) -> ModelTestResult:
        """Test a model's connectivity and response"""
        model = self.get(model_id)
        if not model:
            return ModelTestResult(
                model_id=model_id,
                success=False,
                latency_ms=0,
                error="Model not found"
            )

        start_time = datetime.utcnow()
        print("Testing model:", model_id, model.name)
        try:
            if model.model_type == ModelType.API:
                result = await self._test_api_model(model)
            elif model.model_type == ModelType.OLLAMA:
                result = await self._test_ollama_model(model)
            elif model.model_type == ModelType.HUGGINGFACE:
                result = await self._test_huggingface_model(model)
            elif model.model_type == ModelType.UPLOADED:
                result = await self._test_uploaded_model(model)
            else:
                result = ModelTestResult(
                    model_id=model_id,
                    success=False,
                    latency_ms=0,
                    error=f"Unknown model type: {model.model_type}"
                )

            # Update model status
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            result.latency_ms = latency

            model.last_tested = datetime.utcnow()
            model.test_result = "success" if result.success else result.error
            model.status = ModelStatus.ACTIVE if result.success else ModelStatus.ERROR
            
            # Update DB
            self.update(model_id, {
                "last_tested": model.last_tested,
                "test_result": model.test_result,
                "status": model.status
            })

            return result

        except Exception as e:
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Update DB on error
            try:
                self.update(model_id, {
                    "status": ModelStatus.ERROR,
                    "test_result": str(e)
                })
            except:
                pass

            return ModelTestResult(
                model_id=model_id,
                success=False,
                latency_ms=latency,
                error=str(e)
            )

    async def _test_api_model(self, model: ModelConfig) -> ModelTestResult:
        """Test an API-based model"""
        test_prompt = "Say 'Hello, AURA!' in exactly those words."

        if model.provider == ModelProvider.OPENAI:
            return await self._test_openai(model, test_prompt)
        elif model.provider == ModelProvider.ANTHROPIC:
            return await self._test_anthropic(model, test_prompt)
        elif model.provider == ModelProvider.POE:
            return await self._test_poe(model, test_prompt)
        elif model.provider == ModelProvider.CUSTOM:
            return await self._test_custom_api(model, test_prompt)
        else:
            return ModelTestResult(
                model_id=model.id,
                success=False,
                latency_ms=0,
                error=f"Unknown provider: {model.provider}"
            )

    async def _test_openai(self, model: ModelConfig, prompt: str) -> ModelTestResult:
        """Test OpenAI model"""
        api_key = model.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return ModelTestResult(
                model_id=model.id,
                success=False,
                latency_ms=0,
                error="OpenAI API key not configured"
            )

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 50
                },
                timeout=model.timeout_seconds
            )

            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return ModelTestResult(
                    model_id=model.id,
                    success=True,
                    latency_ms=0,
                    response=content
                )
            else:
                return ModelTestResult(
                    model_id=model.id,
                    success=False,
                    latency_ms=0,
                    error=f"API error: {response.status_code} - {response.text}"
                )

    async def _test_anthropic(self, model: ModelConfig, prompt: str) -> ModelTestResult:
        """Test Anthropic model"""
        api_key = model.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return ModelTestResult(
                model_id=model.id,
                success=False,
                latency_ms=0,
                error="Anthropic API key not configured"
            )

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": model.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 50
                },
                timeout=model.timeout_seconds
            )

            if response.status_code == 200:
                data = response.json()
                content = data["content"][0]["text"]
                return ModelTestResult(
                    model_id=model.id,
                    success=True,
                    latency_ms=0,
                    response=content
                )
            else:
                return ModelTestResult(
                    model_id=model.id,
                    success=False,
                    latency_ms=0,
                    error=f"API error: {response.status_code} - {response.text}"
                )

    async def _test_poe(self, model: ModelConfig, prompt: str) -> ModelTestResult:
        """Test Poe API model"""
        api_key = model.api_key or os.getenv("POE_API_KEY")
        if not api_key:
            return ModelTestResult(
                model_id=model.id,
                success=False,
                latency_ms=0,
                error="Poe API key not configured"
            )

        # Poe uses OpenAI-compatible endpoint
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.poe.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 50
                },
                timeout=model.timeout_seconds
            )

            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return ModelTestResult(
                    model_id=model.id,
                    success=True,
                    latency_ms=0,
                    response=content
                )
            else:
                return ModelTestResult(
                    model_id=model.id,
                    success=False,
                    latency_ms=0,
                    error=f"Poe API error: {response.status_code} - {response.text}"
                )

    async def _test_custom_api(self, model: ModelConfig, prompt: str) -> ModelTestResult:
        """Test custom API endpoint"""
        if not model.endpoint_url:
            return ModelTestResult(
                model_id=model.id,
                success=False,
                latency_ms=0,
                error="Endpoint URL not configured"
            )

        async with httpx.AsyncClient() as client:
            headers = {"Content-Type": "application/json"}
            if model.api_key:
                headers["Authorization"] = f"Bearer {model.api_key}"

            response = await client.post(
                model.endpoint_url,
                headers=headers,
                json={
                    "prompt": prompt,
                    "max_tokens": 50
                },
                timeout=model.timeout_seconds
            )
            print("custom api response:", response.status_code, response.text)
            if response.status_code == 200:
                return ModelTestResult(
                    model_id=model.id,
                    success=True,
                    latency_ms=0,
                    response=str(response.json())
                )
            else:
                return ModelTestResult(
                    model_id=model.id,
                    success=False,
                    latency_ms=0,
                    error=f"API error: {response.status_code}"
                )

    async def _test_ollama_model(self, model: ModelConfig) -> ModelTestResult:
        """Test Ollama local model"""
        endpoint = model.endpoint_url or "http://localhost:11434"

        async with httpx.AsyncClient() as client:
            # First check if Ollama is running
            try:
                health = await client.get(f"{endpoint}/api/tags", timeout=5)
                if health.status_code != 200:
                    return ModelTestResult(
                        model_id=model.id,
                        success=False,
                        latency_ms=0,
                        error="Ollama not running or not accessible"
                    )
            except Exception as e:
                return ModelTestResult(
                    model_id=model.id,
                    success=False,
                    latency_ms=0,
                    error=f"Cannot connect to Ollama: {str(e)}"
                )

            # Test the model
            response = await client.post(
                f"{endpoint}/api/generate",
                json={
                    "model": model.model_name,
                    "prompt": "Say 'Hello, AURA!'",
                    "stream": False
                },
                timeout=model.timeout_seconds
            )

            if response.status_code == 200:
                data = response.json()
                return ModelTestResult(
                    model_id=model.id,
                    success=True,
                    latency_ms=0,
                    response=data.get("response", "")
                )
            else:
                return ModelTestResult(
                    model_id=model.id,
                    success=False,
                    latency_ms=0,
                    error=f"Ollama error: {response.status_code}"
                )

    async def _test_huggingface_model(self, model: ModelConfig) -> ModelTestResult:
        """Test HuggingFace model"""
        # For HuggingFace Inference API
        api_key = model.api_key or os.getenv("HUGGINGFACE_API_KEY")

        if model.endpoint_url:
            # Custom endpoint (Inference Endpoints)
            endpoint = model.endpoint_url
        else:
            # Standard Inference API
            endpoint = f"https://api-inference.huggingface.co/models/{model.model_name}"

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                endpoint,
                headers=headers,
                json={"inputs": "Say 'Hello, AURA!'"},
                timeout=model.timeout_seconds
            )

            if response.status_code == 200:
                return ModelTestResult(
                    model_id=model.id,
                    success=True,
                    latency_ms=0,
                    response=str(response.json())
                )
            else:
                return ModelTestResult(
                    model_id=model.id,
                    success=False,
                    latency_ms=0,
                    error=f"HuggingFace error: {response.status_code} - {response.text}"
                )

    async def _test_uploaded_model(self, model: ModelConfig) -> ModelTestResult:
        """Test uploaded model file"""
        if not model.file_path or not os.path.exists(model.file_path):
            return ModelTestResult(
                model_id=model.id,
                success=False,
                latency_ms=0,
                error="Model file not found"
            )

        # Check file exists and is readable
        try:
            file_size = os.path.getsize(model.file_path)
            return ModelTestResult(
                model_id=model.id,
                success=True,
                latency_ms=0,
                response=f"Model file verified: {file_size} bytes"
            )
        except Exception as e:
            return ModelTestResult(
                model_id=model.id,
                success=False,
                latency_ms=0,
                error=str(e)
            )

    async def invoke(self, model_id: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Invoke a model with a prompt"""
        model = self.get(model_id)
        if not model:
            return {"error": "Model not found", "model_id": model_id}

        if model.status != ModelStatus.ACTIVE:
            return {"error": f"Model not active: {model.status}", "model_id": model_id}

        start_time = datetime.utcnow()

        try:
            if model.model_type == ModelType.API:
                result = await self._invoke_api_model(model, prompt, **kwargs)
            elif model.model_type == ModelType.OLLAMA:
                result = await self._invoke_ollama_model(model, prompt, **kwargs)
            else:
                result = {"error": f"Invocation not supported for {model.model_type}"}

            # Update usage stats
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Persist usage stats
            db = SessionLocal()
            try:
                orm_model = db.query(ORMModel).filter(ORMModel.id == model_id).first()
                if orm_model:
                    orm_model.total_requests += 1
                    orm_model.avg_latency_ms = (
                        (orm_model.avg_latency_ms * (orm_model.total_requests - 1) + latency)
                        / orm_model.total_requests
                    )
                    db.commit()
            except Exception as e:
                logger.error("Failed to update usage stats", error=str(e))
            finally:
                db.close()

            return result

        except Exception as e:
            return {"error": str(e), "model_id": model_id}

    async def _invoke_api_model(self, model: ModelConfig, prompt: str, **kwargs) -> Dict[str, Any]:
        """Invoke API model"""
        if model.provider == ModelProvider.OPENAI:
            api_key = model.api_key or os.getenv("OPENAI_API_KEY")
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": kwargs.get("max_tokens", model.max_tokens),
                        "temperature": kwargs.get("temperature", model.temperature)
                    },
                    timeout=model.timeout_seconds
                )

                if response.status_code == 200:
                    data = response.json()
                    return {
                        "response": data["choices"][0]["message"]["content"],
                        "usage": data.get("usage", {}),
                        "model": model.model_name
                    }
                else:
                    return {"error": f"API error: {response.status_code}"}

        elif model.provider == ModelProvider.ANTHROPIC:
            api_key = model.api_key or os.getenv("ANTHROPIC_API_KEY")
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "Content-Type": "application/json",
                        "anthropic-version": "2023-06-01"
                    },
                    json={
                        "model": model.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": kwargs.get("max_tokens", model.max_tokens)
                    },
                    timeout=model.timeout_seconds
                )

                if response.status_code == 200:
                    data = response.json()
                    return {
                        "response": data["content"][0]["text"],
                        "usage": data.get("usage", {}),
                        "model": model.model_name
                    }
                else:
                    return {"error": f"API error: {response.status_code}"}

        elif model.provider == ModelProvider.POE:
            api_key = model.api_key or os.getenv("POE_API_KEY")
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.poe.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": kwargs.get("max_tokens", model.max_tokens),
                        "temperature": kwargs.get("temperature", model.temperature)
                    },
                    timeout=model.timeout_seconds
                )

                if response.status_code == 200:
                    data = response.json()
                    return {
                        "response": data["choices"][0]["message"]["content"],
                        "usage": data.get("usage", {}),
                        "model": model.model_name
                    }
                else:
                    return {"error": f"Poe API error: {response.status_code}"}

        return {"error": f"Unknown provider: {model.provider}"}

    async def _invoke_ollama_model(self, model: ModelConfig, prompt: str, **kwargs) -> Dict[str, Any]:
        """Invoke Ollama model"""
        endpoint = model.endpoint_url or "http://localhost:11434"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{endpoint}/api/generate",
                json={
                    "model": model.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", model.temperature),
                        "num_predict": kwargs.get("max_tokens", model.max_tokens)
                    }
                },
                timeout=model.timeout_seconds
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    "response": data.get("response", ""),
                    "model": model.model_name,
                    "eval_count": data.get("eval_count", 0)
                }
            else:
                return {"error": f"Ollama error: {response.status_code}"}

    def save_uploaded_model(self, model_id: str, file_content: bytes, filename: str) -> str:
        """Save an uploaded model file"""
        model = self.get(model_id)
        if not model:
            raise ValueError(f"Model not found: {model_id}")

        # Determine file extension
        ext = Path(filename).suffix.lower()
        if ext not in ['.onnx', '.pt', '.pth', '.safetensors', '.bin']:
            raise ValueError(f"Unsupported file format: {ext}")

        # Save file
        file_path = self.storage_path / f"{model_id}{ext}"
        with open(file_path, 'wb') as f:
            f.write(file_content)

        # Update model
        model.file_path = str(file_path)
        model.file_format = ext[1:]  # Remove dot

        logger.info("Model file saved", model_id=model_id, path=str(file_path))
        return str(file_path)

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        models = self.list_models()

        return {
            "total_models": len(models),
            "by_type": {
                t.value: len([m for m in models if m.model_type == t])
                for t in ModelType
            },
            "by_status": {
                s.value: len([m for m in models if m.status == s])
                for s in ModelStatus
            },
            "by_provider": {
                p.value: len([m for m in models if m.provider == p])
                for p in ModelProvider
            }
        }


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Get or create the global model registry"""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry

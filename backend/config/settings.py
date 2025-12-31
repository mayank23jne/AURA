"""Configuration settings for AURA Agentic Platform"""

import os
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class LLMSettings(BaseSettings):
    """LLM provider settings"""
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    default_provider: str = Field(default="openai", env="DEFAULT_LLM_PROVIDER")
    default_model: str = Field(default="gpt-4", env="DEFAULT_LLM_MODEL")
    temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")
    max_tokens: int = Field(default=4096, env="LLM_MAX_TOKENS")


class DatabaseSettings(BaseSettings):
    """Database settings"""
    mysql_url: str = Field(
        default="mysql+mysqlconnector://root:password@localhost:3306/aura_db",
        env="MYSQL_URL"
    )
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    vector_db_type: str = Field(default="chroma", env="VECTOR_DB_TYPE")
    chroma_persist_dir: str = Field(default="./data/chroma", env="CHROMA_PERSIST_DIR")


class MessageBusSettings(BaseSettings):
    """Message bus settings"""
    bus_type: str = Field(default="memory", env="MESSAGE_BUS_TYPE")  # memory, kafka, redis
    kafka_brokers: str = Field(default="localhost:9092", env="KAFKA_BROKERS")
    kafka_group_id: str = Field(default="aura-agents", env="KAFKA_GROUP_ID")


class AgentSettings(BaseSettings):
    """Agent configuration settings"""
    rate_limit_rpm: int = Field(default=60, env="AGENT_RATE_LIMIT_RPM")
    rate_limit_tpm: int = Field(default=100000, env="AGENT_RATE_LIMIT_TPM")
    retry_attempts: int = Field(default=3, env="AGENT_RETRY_ATTEMPTS")
    timeout_seconds: int = Field(default=60, env="AGENT_TIMEOUT_SECONDS")
    memory_type: str = Field(default="conversation_summary", env="AGENT_MEMORY_TYPE")


class AuditSettings(BaseSettings):
    """Audit configuration settings"""
    default_test_count: int = Field(default=100, env="AUDIT_DEFAULT_TEST_COUNT")
    parallel_execution: bool = Field(default=True, env="AUDIT_PARALLEL_EXECUTION")
    batch_size: int = Field(default=10, env="AUDIT_BATCH_SIZE")
    early_stopping: bool = Field(default=True, env="AUDIT_EARLY_STOPPING")
    confidence_threshold: float = Field(default=0.95, env="AUDIT_CONFIDENCE_THRESHOLD")


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings"""
    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    prometheus_port: int = Field(default=8000, env="PROMETHEUS_PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")


class APISettings(BaseSettings):
    """API server settings"""
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8080, env="API_PORT")
    workers: int = Field(default=1, env="API_WORKERS")
    reload: bool = Field(default=True, env="API_RELOAD")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")


class Settings(BaseSettings):
    """Main settings container"""
    app_name: str = "AURA Agentic Platform"
    app_version: str = "0.1.0"
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")

    # Sub-settings
    llm: LLMSettings = LLMSettings()
    database: DatabaseSettings = DatabaseSettings()
    message_bus: MessageBusSettings = MessageBusSettings()
    agent: AgentSettings = AgentSettings()
    audit: AuditSettings = AuditSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    api: APISettings = APISettings()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance"""
    return settings

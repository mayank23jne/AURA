"""Schema Registry for message versioning and validation"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Type
from enum import Enum

import structlog
from pydantic import BaseModel, ValidationError, Field

logger = structlog.get_logger()


class SchemaVersion(BaseModel):
    """Schema version information"""
    version: str
    schema_class: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    deprecated: bool = False
    deprecation_message: Optional[str] = None


class SchemaCompatibility(str, Enum):
    """Schema compatibility modes"""
    BACKWARD = "backward"  # New schema can read old data
    FORWARD = "forward"    # Old schema can read new data
    FULL = "full"          # Both backward and forward
    NONE = "none"          # No compatibility guarantee


class SchemaRegistry:
    """
    Registry for message schema versioning and validation.

    Features:
    - Schema versioning with semantic versioning
    - Backward/forward compatibility checking
    - Schema validation
    - Schema migration support
    - Deprecation management
    """

    def __init__(self):
        self._schemas: Dict[str, Dict[str, Type[BaseModel]]] = {}
        self._versions: Dict[str, List[SchemaVersion]] = {}
        self._compatibility_mode: Dict[str, SchemaCompatibility] = {}
        self._migrations: Dict[str, Dict[str, callable]] = {}

        logger.info("SchemaRegistry initialized")

    def register_schema(
        self,
        schema_name: str,
        version: str,
        schema_class: Type[BaseModel],
        compatibility: SchemaCompatibility = SchemaCompatibility.BACKWARD,
    ) -> bool:
        """Register a new schema version"""
        if schema_name not in self._schemas:
            self._schemas[schema_name] = {}
            self._versions[schema_name] = []
            self._compatibility_mode[schema_name] = compatibility

        # Check compatibility with existing versions
        if self._versions[schema_name]:
            if not self._check_compatibility(schema_name, version, schema_class):
                logger.error(
                    "Schema compatibility check failed",
                    schema=schema_name,
                    version=version,
                )
                return False

        # Register the schema
        self._schemas[schema_name][version] = schema_class
        self._versions[schema_name].append(
            SchemaVersion(
                version=version,
                schema_class=schema_class.__name__,
            )
        )

        logger.info(
            "Schema registered",
            schema=schema_name,
            version=version,
            class_name=schema_class.__name__,
        )
        return True

    def validate(
        self,
        schema_name: str,
        data: Dict[str, Any],
        version: Optional[str] = None,
    ) -> tuple[bool, Optional[BaseModel], Optional[str]]:
        """
        Validate data against a schema.

        Returns:
            Tuple of (is_valid, parsed_model, error_message)
        """
        if schema_name not in self._schemas:
            return False, None, f"Schema not found: {schema_name}"

        # Use latest version if not specified
        if version is None:
            version = self.get_latest_version(schema_name)
            if version is None:
                return False, None, f"No versions registered for schema: {schema_name}"

        if version not in self._schemas[schema_name]:
            return False, None, f"Version not found: {schema_name}@{version}"

        schema_class = self._schemas[schema_name][version]

        try:
            model = schema_class(**data)
            return True, model, None
        except ValidationError as e:
            return False, None, str(e)

    def get_latest_version(self, schema_name: str) -> Optional[str]:
        """Get the latest version of a schema"""
        if schema_name not in self._versions:
            return None

        versions = [v for v in self._versions[schema_name] if not v.deprecated]
        if not versions:
            return None

        # Sort by semantic version
        sorted_versions = sorted(
            versions,
            key=lambda v: [int(x) for x in v.version.split(".")],
            reverse=True,
        )
        return sorted_versions[0].version

    def get_schema(
        self,
        schema_name: str,
        version: Optional[str] = None,
    ) -> Optional[Type[BaseModel]]:
        """Get a schema class by name and version"""
        if schema_name not in self._schemas:
            return None

        if version is None:
            version = self.get_latest_version(schema_name)

        return self._schemas[schema_name].get(version)

    def deprecate_version(
        self,
        schema_name: str,
        version: str,
        message: str = "",
    ) -> bool:
        """Mark a schema version as deprecated"""
        if schema_name not in self._versions:
            return False

        for v in self._versions[schema_name]:
            if v.version == version:
                v.deprecated = True
                v.deprecation_message = message
                logger.warning(
                    "Schema version deprecated",
                    schema=schema_name,
                    version=version,
                    message=message,
                )
                return True

        return False

    def register_migration(
        self,
        schema_name: str,
        from_version: str,
        to_version: str,
        migration_fn: callable,
    ):
        """Register a migration function between versions"""
        key = f"{from_version}->{to_version}"

        if schema_name not in self._migrations:
            self._migrations[schema_name] = {}

        self._migrations[schema_name][key] = migration_fn
        logger.info(
            "Migration registered",
            schema=schema_name,
            from_version=from_version,
            to_version=to_version,
        )

    def migrate(
        self,
        schema_name: str,
        data: Dict[str, Any],
        from_version: str,
        to_version: str,
    ) -> Optional[Dict[str, Any]]:
        """Migrate data from one version to another"""
        if schema_name not in self._migrations:
            return None

        key = f"{from_version}->{to_version}"
        if key not in self._migrations[schema_name]:
            # Try to find migration path
            path = self._find_migration_path(schema_name, from_version, to_version)
            if not path:
                return None

            # Apply migrations in sequence
            current_data = data
            for i in range(len(path) - 1):
                step_key = f"{path[i]}->{path[i+1]}"
                if step_key in self._migrations[schema_name]:
                    current_data = self._migrations[schema_name][step_key](current_data)
                else:
                    return None
            return current_data

        return self._migrations[schema_name][key](data)

    def _find_migration_path(
        self,
        schema_name: str,
        from_version: str,
        to_version: str,
    ) -> Optional[List[str]]:
        """Find a migration path between versions using BFS"""
        if schema_name not in self._migrations:
            return None

        # Build adjacency list
        graph: Dict[str, List[str]] = {}
        for key in self._migrations[schema_name]:
            src, dst = key.split("->")
            if src not in graph:
                graph[src] = []
            graph[src].append(dst)

        # BFS to find path
        from collections import deque
        queue = deque([(from_version, [from_version])])
        visited = {from_version}

        while queue:
            current, path = queue.popleft()
            if current == to_version:
                return path

            for neighbor in graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def _check_compatibility(
        self,
        schema_name: str,
        new_version: str,
        new_schema: Type[BaseModel],
    ) -> bool:
        """Check if new schema is compatible with existing versions"""
        compatibility = self._compatibility_mode.get(
            schema_name, SchemaCompatibility.NONE
        )

        if compatibility == SchemaCompatibility.NONE:
            return True

        # Get the latest version to compare against
        latest_version = self.get_latest_version(schema_name)
        if not latest_version:
            return True

        latest_schema = self._schemas[schema_name][latest_version]

        # Get fields from both schemas
        new_fields = set(new_schema.model_fields.keys())
        old_fields = set(latest_schema.model_fields.keys())

        if compatibility == SchemaCompatibility.BACKWARD:
            # New schema should be able to read old data
            # All old required fields must exist in new schema
            old_required = {
                k for k, v in latest_schema.model_fields.items()
                if v.is_required()
            }
            return old_required.issubset(new_fields)

        elif compatibility == SchemaCompatibility.FORWARD:
            # Old schema should be able to read new data
            # All new required fields must exist in old schema
            new_required = {
                k for k, v in new_schema.model_fields.items()
                if v.is_required()
            }
            return new_required.issubset(old_fields)

        elif compatibility == SchemaCompatibility.FULL:
            # Both backward and forward compatible
            old_required = {
                k for k, v in latest_schema.model_fields.items()
                if v.is_required()
            }
            new_required = {
                k for k, v in new_schema.model_fields.items()
                if v.is_required()
            }
            return old_required.issubset(new_fields) and new_required.issubset(old_fields)

        return True

    def list_schemas(self) -> Dict[str, List[str]]:
        """List all registered schemas and their versions"""
        return {
            name: [v.version for v in versions]
            for name, versions in self._versions.items()
        }

    def get_schema_info(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a schema"""
        if schema_name not in self._versions:
            return None

        return {
            "name": schema_name,
            "versions": [v.model_dump() for v in self._versions[schema_name]],
            "compatibility": self._compatibility_mode.get(
                schema_name, SchemaCompatibility.NONE
            ).value,
            "latest_version": self.get_latest_version(schema_name),
        }


# Global schema registry instance
schema_registry = SchemaRegistry()


def get_schema_registry() -> SchemaRegistry:
    """Get the global schema registry instance"""
    return schema_registry

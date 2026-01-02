"""Knowledge Base implementation for AURA agents"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

import structlog
from pydantic import BaseModel

from src.core.models import KnowledgeItem, KnowledgeType

logger = structlog.get_logger()


class KnowledgeBase(ABC):
    """Abstract base class for knowledge base implementations"""

    @abstractmethod
    async def store(self, item: KnowledgeItem) -> str:
        """Store a knowledge item"""
        pass

    @abstractmethod
    async def retrieve(self, item_id: str) -> Optional[KnowledgeItem]:
        """Retrieve a knowledge item by ID"""
        pass

    @abstractmethod
    async def search(
        self, query: str, top_k: int = 5, filters: Dict[str, Any] = None
    ) -> List[KnowledgeItem]:
        """Search for knowledge items using semantic similarity"""
        pass

    @abstractmethod
    async def update(self, item_id: str, updates: Dict[str, Any]) -> bool:
        """Update a knowledge item"""
        pass

    @abstractmethod
    async def delete(self, item_id: str) -> bool:
        """Delete a knowledge item"""
        pass

    @abstractmethod
    async def flush(self):
        """Flush any pending writes"""
        pass


class InMemoryKnowledgeBase(KnowledgeBase):
    """
    In-memory knowledge base for development and testing.

    Features:
    - Simple text-based search
    - Metadata filtering
    - Knowledge versioning
    - Conflict detection
    """

    def __init__(self):
        self._items: Dict[str, KnowledgeItem] = {}
        self._embeddings: Dict[str, List[float]] = {}
        self._indices: Dict[str, List[str]] = {
            "by_type": {},
            "by_domain": {},
            "by_source": {},
        }

        logger.info("InMemoryKnowledgeBase initialized")

    async def store(self, item: KnowledgeItem) -> str:
        """Store a knowledge item"""
        # Generate ID if not present
        if not item.id:
            item.id = str(uuid.uuid4())

        # Check for conflicts
        existing = await self._find_conflicts(item)
        if existing:
            # Merge or version based on confidence
            item = await self._resolve_conflict(item, existing)

        # Store the item
        self._items[item.id] = item

        # Update indices
        self._update_indices(item)

        logger.debug(
            "Knowledge stored",
            item_id=item.id,
            type=item.knowledge_type,
            domain=item.domain,
        )

        return item.id

    async def retrieve(self, item_id: str) -> Optional[KnowledgeItem]:
        """Retrieve a knowledge item by ID"""
        return self._items.get(item_id)

    async def search(
        self, query: str, top_k: int = 5, filters: Dict[str, Any] = None
    ) -> List[KnowledgeItem]:
        """Search for knowledge items"""
        results = []
        query_lower = query.lower()

        for item in self._items.values():
            # Apply filters
            if filters:
                if not self._matches_filters(item, filters):
                    continue

            # Simple text matching (would use embeddings in production)
            score = self._calculate_relevance(item, query_lower)
            if score > 0:
                results.append((score, item))

        # Sort by score and return top_k
        results.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in results[:top_k]]

    async def update(self, item_id: str, updates: Dict[str, Any]) -> bool:
        """Update a knowledge item"""
        if item_id not in self._items:
            return False

        item = self._items[item_id]

        # Create new version
        for key, value in updates.items():
            if hasattr(item, key):
                setattr(item, key, value)

        item.version += 1
        item.timestamp = datetime.utcnow()

        # Update indices
        self._update_indices(item)

        logger.debug("Knowledge updated", item_id=item_id, version=item.version)
        return True

    async def delete(self, item_id: str) -> bool:
        """Delete a knowledge item"""
        if item_id in self._items:
            del self._items[item_id]
            # Clean up indices
            for index in self._indices.values():
                if isinstance(index, dict):
                    for key, ids in list(index.items()):
                        if item_id in ids:
                            index[key].remove(item_id)
            return True
        return False

    async def flush(self):
        """Flush any pending writes (no-op for in-memory)"""
        pass

    def _update_indices(self, item: KnowledgeItem):
        """Update search indices"""
        # Index by type
        type_key = item.knowledge_type.value if isinstance(item.knowledge_type, KnowledgeType) else str(item.knowledge_type)
        if type_key not in self._indices["by_type"]:
            self._indices["by_type"][type_key] = []
        if item.id not in self._indices["by_type"][type_key]:
            self._indices["by_type"][type_key].append(item.id)

        # Index by domain
        if item.domain not in self._indices["by_domain"]:
            self._indices["by_domain"][item.domain] = []
        if item.id not in self._indices["by_domain"][item.domain]:
            self._indices["by_domain"][item.domain].append(item.id)

        # Index by source
        if item.source_agent not in self._indices["by_source"]:
            self._indices["by_source"][item.source_agent] = []
        if item.id not in self._indices["by_source"][item.source_agent]:
            self._indices["by_source"][item.source_agent].append(item.id)

    def _matches_filters(self, item: KnowledgeItem, filters: Dict[str, Any]) -> bool:
        """Check if item matches filters"""
        for key, value in filters.items():
            if hasattr(item, key):
                item_value = getattr(item, key)
                if isinstance(item_value, Enum):
                    item_value = item_value.value
                if item_value != value:
                    return False
        return True

    def _calculate_relevance(self, item: KnowledgeItem, query: str) -> float:
        """Calculate relevance score for an item"""
        score = 0.0
        content_str = str(item.content).lower()

        # Check content
        if query in content_str:
            score += 1.0

        # Check domain
        if query in item.domain.lower():
            score += 0.5

        # Check tags
        for tag in item.tags:
            if query in tag.lower():
                score += 0.3

        # Weight by confidence
        score *= item.confidence

        return score

    async def _find_conflicts(self, new_item: KnowledgeItem) -> Optional[KnowledgeItem]:
        """Find potentially conflicting knowledge items"""
        for item in self._items.values():
            if (
                item.knowledge_type == new_item.knowledge_type
                and item.domain == new_item.domain
                and str(item.content) == str(new_item.content)
            ):
                return item
        return None

    async def _resolve_conflict(
        self, new_item: KnowledgeItem, existing: KnowledgeItem
    ) -> KnowledgeItem:
        """Resolve conflict between knowledge items"""
        # Keep higher confidence item
        if new_item.confidence > existing.confidence:
            new_item.version = existing.version + 1
            return new_item
        else:
            # Update existing with new timestamp
            existing.timestamp = datetime.utcnow()
            existing.version += 1
            return existing

    async def get_by_type(self, knowledge_type: KnowledgeType) -> List[KnowledgeItem]:
        """Get all items of a specific type"""
        type_key = knowledge_type.value
        item_ids = self._indices["by_type"].get(type_key, [])
        return [self._items[id] for id in item_ids if id in self._items]

    async def get_by_domain(self, domain: str) -> List[KnowledgeItem]:
        """Get all items in a specific domain"""
        item_ids = self._indices["by_domain"].get(domain, [])
        return [self._items[id] for id in item_ids if id in self._items]

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        type_counts = {}
        for type_key, ids in self._indices["by_type"].items():
            type_counts[type_key] = len(ids)

        return {
            "total_items": len(self._items),
            "items_by_type": type_counts,
            "domains": list(self._indices["by_domain"].keys()),
            "sources": list(self._indices["by_source"].keys()),
        }


class ChromaKnowledgeBase(KnowledgeBase):
    """
    ChromaDB-based knowledge base for production deployments.

    Features:
    - Vector embeddings for semantic search
    - Persistent storage
    - Metadata filtering
    - Batch operations
    """

    def __init__(
        self,
        collection_name: str = "aura_knowledge",
        persist_directory: str = "./chroma_db",
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None

        logger.info(
            "ChromaKnowledgeBase initialized",
            collection=collection_name,
            persist_dir=persist_directory,
        )

    async def connect(self):
        """Initialize ChromaDB connection"""
        try:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.Client(
                Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=self.persist_directory,
                )
            )

            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info("ChromaDB connected", collection=self.collection_name)

        except ImportError:
            logger.warning("chromadb not installed, using mock mode")
        except Exception as e:
            logger.error("Failed to connect to ChromaDB", error=str(e))
            raise

    async def store(self, item: KnowledgeItem) -> str:
        """Store a knowledge item with embeddings"""
        if not self._collection:
            logger.warning("Collection not initialized")
            return ""

        if not item.id:
            item.id = str(uuid.uuid4())

        # Prepare document for storage
        document = str(item.content)
        metadata = {
            "knowledge_type": item.knowledge_type.value if isinstance(item.knowledge_type, KnowledgeType) else str(item.knowledge_type),
            "domain": item.domain,
            "source_agent": item.source_agent,
            "confidence": item.confidence,
            "version": item.version,
            "timestamp": item.timestamp.isoformat(),
            "tags": ",".join(item.tags),
        }

        # Add custom metadata
        for key, value in item.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                metadata[f"meta_{key}"] = value

        try:
            self._collection.add(
                ids=[item.id],
                documents=[document],
                metadatas=[metadata],
            )

            logger.debug("Knowledge stored in ChromaDB", item_id=item.id)
            return item.id

        except Exception as e:
            logger.error("Failed to store in ChromaDB", error=str(e))
            return ""

    async def retrieve(self, item_id: str) -> Optional[KnowledgeItem]:
        """Retrieve a knowledge item by ID"""
        if not self._collection:
            return None

        try:
            result = self._collection.get(ids=[item_id], include=["documents", "metadatas"])

            if result["ids"]:
                return self._result_to_item(
                    result["ids"][0],
                    result["documents"][0],
                    result["metadatas"][0],
                )

        except Exception as e:
            logger.error("Failed to retrieve from ChromaDB", error=str(e))

        return None

    async def search(
        self, query: str, top_k: int = 5, filters: Dict[str, Any] = None
    ) -> List[KnowledgeItem]:
        """Search using semantic similarity"""
        if not self._collection:
            return []

        try:
            where = None
            if filters:
                where = {k: v for k, v in filters.items()}

            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            items = []
            for i in range(len(results["ids"][0])):
                item = self._result_to_item(
                    results["ids"][0][i],
                    results["documents"][0][i],
                    results["metadatas"][0][i],
                )
                items.append(item)

            return items

        except Exception as e:
            logger.error("Failed to search ChromaDB", error=str(e))
            return []

    async def update(self, item_id: str, updates: Dict[str, Any]) -> bool:
        """Update a knowledge item"""
        if not self._collection:
            return False

        try:
            # Get existing
            existing = await self.retrieve(item_id)
            if not existing:
                return False

            # Apply updates
            for key, value in updates.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)

            existing.version += 1
            existing.timestamp = datetime.utcnow()

            # Re-store
            self._collection.update(
                ids=[item_id],
                documents=[str(existing.content)],
                metadatas=[{
                    "knowledge_type": existing.knowledge_type.value if isinstance(existing.knowledge_type, KnowledgeType) else str(existing.knowledge_type),
                    "domain": existing.domain,
                    "source_agent": existing.source_agent,
                    "confidence": existing.confidence,
                    "version": existing.version,
                    "timestamp": existing.timestamp.isoformat(),
                    "tags": ",".join(existing.tags),
                }],
            )

            return True

        except Exception as e:
            logger.error("Failed to update in ChromaDB", error=str(e))
            return False

    async def delete(self, item_id: str) -> bool:
        """Delete a knowledge item"""
        if not self._collection:
            return False

        try:
            self._collection.delete(ids=[item_id])
            return True
        except Exception as e:
            logger.error("Failed to delete from ChromaDB", error=str(e))
            return False

    async def flush(self):
        """Persist changes to disk"""
        if self._client:
            self._client.persist()

    def _result_to_item(
        self, item_id: str, document: str, metadata: Dict[str, Any]
    ) -> KnowledgeItem:
        """Convert ChromaDB result to KnowledgeItem"""
        return KnowledgeItem(
            id=item_id,
            knowledge_type=KnowledgeType(metadata.get("knowledge_type", "experience")),
            domain=metadata.get("domain", "general"),
            content={"text": document},
            source_agent=metadata.get("source_agent", "unknown"),
            confidence=float(metadata.get("confidence", 1.0)),
            version=int(metadata.get("version", 1)),
            timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.utcnow().isoformat())),
            tags=metadata.get("tags", "").split(",") if metadata.get("tags") else [],
        )

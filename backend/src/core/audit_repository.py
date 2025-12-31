"""Audit History Repository for AURA Platform

This module provides persistent storage for audit results.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from src.db import SessionLocal
from src.models.orm import ORMAudit

logger = structlog.get_logger(__name__)


class AuditRepository:
    """Repository for storing and retrieving audit history."""

    def __init__(self):
        """Initialize audit repository."""
        logger.info("AuditRepository initialized with Database backend")
    
    def _get_audit_file(self, audit_id: str) -> Path:
        """Deprecated: Get file path for audit ID."""
        return Path(f"./data/audits/{audit_id}.json")

    async def save_audit(self, audit_data: Dict[str, Any]) -> bool:
        """Save audit result to storage.

        Args:
            audit_data: Audit data dictionary

        Returns:
            True if successful, False otherwise
        """
        audit_id = audit_data.get("audit_id")
        if not audit_id:
            logger.error("Cannot save audit without audit_id")
            return False

        if "saved_at" not in audit_data:
            audit_data["saved_at"] = datetime.now().isoformat()

        db = SessionLocal()
        try:
            # Check if exists
            existing = db.query(ORMAudit).filter(ORMAudit.audit_id == audit_id).first()
            
            if existing:
                existing.model_id = audit_data.get("model_id")
                existing.status = audit_data.get("status", "unknown")
                existing.compliance_score = float(audit_data.get("compliance_score", 0.0))
                existing.risk_score = float(audit_data.get("risk_score", 0.0))
                existing.audit_data = audit_data
                existing.saved_at = datetime.utcnow()
            else:
                audit = ORMAudit(
                    audit_id=audit_id,
                    model_id=audit_data.get("model_id"),
                    status=audit_data.get("status", "unknown"),
                    compliance_score=float(audit_data.get("compliance_score", 0.0)),
                    risk_score=float(audit_data.get("risk_score", 0.0)),
                    audit_data=audit_data,
                    saved_at=datetime.utcnow()
                )
                db.add(audit)
            
            db.commit()
            logger.info("Audit saved", audit_id=audit_id)
            return True

        except Exception as e:
            db.rollback()
            logger.error("Failed to save audit", error=str(e), audit_id=audit_id)
            return False
        finally:
            db.close()

    async def get_audit(self, audit_id: str) -> Optional[Dict[str, Any]]:
        """Get audit by ID.

        Args:
            audit_id: Audit ID

        Returns:
            Audit data if found, None otherwise
        """
        db = SessionLocal()
        try:
            audit = db.query(ORMAudit).filter(ORMAudit.audit_id == audit_id).first()
            if audit:
                return audit.to_dict()
            return None

        except Exception as e:
            logger.error("Failed to load audit", error=str(e), audit_id=audit_id)
            return None
        finally:
            db.close()

    async def list_audits(
        self,
        model_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List audits with optional filters."""
        db = SessionLocal()
        try:
            query = db.query(ORMAudit)

            if model_id:
                query = query.filter(ORMAudit.model_id == model_id)
            if status:
                query = query.filter(ORMAudit.status == status)

            # Sort by saved_at desc
            query = query.order_by(ORMAudit.saved_at.desc())
            
            # Pagination
            audits = query.offset(offset).limit(limit).all()
            
            return [a.to_dict() for a in audits]

        except Exception as e:
            logger.error("Failed to list audits", error=str(e))
            return []
        finally:
            db.close()

    async def delete_audit(self, audit_id: str) -> bool:
        """Delete audit by ID."""
        db = SessionLocal()
        try:
            audit = db.query(ORMAudit).filter(ORMAudit.audit_id == audit_id).first()
            if audit:
                db.delete(audit)
                db.commit()
                logger.info("Audit deleted", audit_id=audit_id)
                return True
            return False

        except Exception as e:
            db.rollback()
            logger.error("Failed to delete audit", error=str(e), audit_id=audit_id)
            return False
        finally:
            db.close()

    async def count_audits(
        self,
        model_id: Optional[str] = None,
        status: Optional[str] = None
    ) -> int:
        """Count audits with optional filters."""
        db = SessionLocal()
        try:
            query = db.query(ORMAudit)
            if model_id:
                query = query.filter(ORMAudit.model_id == model_id)
            if status:
                query = query.filter(ORMAudit.status == status)
            return query.count()
        finally:
            db.close()


# Global instance
_audit_repository: Optional[AuditRepository] = None


def get_audit_repository() -> AuditRepository:
    """Get or create global audit repository instance."""
    global _audit_repository
    if _audit_repository is None:
        _audit_repository = AuditRepository()
    return _audit_repository

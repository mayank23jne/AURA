"""Security Module for AURA Platform

Provides cryptographic functions for:
- Data signing (HMAC-SHA256)
- Signature verification
- Immutability checks
"""

import hmac
import hashlib
import json
import os
import base64
from typing import Any, Dict, Optional
import structlog

logger = structlog.get_logger()

# Default secret for development - override with env var in production
_DEFAULT_SECRET = "aura-dev-secret-do-not-use-in-prod"


def _get_secret() -> bytes:
    """Get the signing secret key"""
    secret = os.getenv("AURA_SECRET_KEY", _DEFAULT_SECRET)
    if secret == _DEFAULT_SECRET:
        logger.warning("Using default development secret. Set AURA_SECRET_KEY in production.")
    return secret.encode("utf-8")


def _prepare_data(data: Dict[str, Any]) -> bytes:
    """Prepare dictionary data for signing (canonical JSON)"""
    # Sort keys for deterministic output
    # Exclude existing signature fields to avoid recursion
    clean_data = {
        k: v for k, v in data.items() 
        if k not in ["signature", "integrity_hash", "signed_at", "saved_at"]
    }
    dumped = json.dumps(clean_data, sort_keys=True, separators=(",", ":"), default=str)
    return dumped.encode("utf-8")


def sign_data(data: Dict[str, Any]) -> str:
    """
    Generate HMAC-SHA256 signature for a dictionary.
    
    Args:
        data: The dictionary to sign
        
    Returns:
        Base64 encoded signature string
    """
    try:
        secret = _get_secret()
        payload = _prepare_data(data)
        
        signature = hmac.new(
            secret,
            payload,
            hashlib.sha256
        ).digest()
        
        return base64.b64encode(signature).decode("utf-8")
        
    except Exception as e:
        logger.error("Failed to sign data", error=str(e))
        raise ValueError(f"Signing failed: {e}")


def verify_signature(data: Dict[str, Any], signature: str) -> bool:
    """
    Verify the signature of a dictionary.
    
    Args:
        data: The signed data (containing the fields to verify)
        signature: The expected signature string
        
    Returns:
        True if valid, False otherwise
    """
    try:
        expected = sign_data(data)
        # Constant-time comparison to prevent timing attacks
        return hmac.compare_digest(expected, signature)
    except Exception as e:
        logger.error("Signature verification error", error=str(e))
        return False


def hash_data(data: Dict[str, Any]) -> str:
    """
    Generate SHA-256 hash of data (without secret).
    Useful for integrity checks/checksums.
    """
    payload = _prepare_data(data)
    return hashlib.sha256(payload).hexdigest()

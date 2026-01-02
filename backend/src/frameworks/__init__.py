"""
Framework integrations for AURA platform.

This package provides adapters for external AI testing frameworks like AI Verify and PyRIT.
"""

from .framework_adapter import FrameworkAdapter, FrameworkRegistry

__all__ = ["FrameworkAdapter", "FrameworkRegistry"]

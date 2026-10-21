"""Compatibility imports; implementation lives in app.services.llm_service."""

from app.services.llm_service import EstimationError, estimate

__all__ = ['EstimationError', 'estimate']

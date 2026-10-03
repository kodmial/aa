"""Safety routing boundary."""

from __future__ import annotations

from aa.safety.emergency import (
    build_emergency_response,
    contains_prohibited_medical_advice,
    detect_acute_category,
    is_emergency,
)
from aa.safety.router import SafetyDecision, SafetyResult, SafetyRouter

__all__ = [
    "SafetyDecision",
    "SafetyResult",
    "SafetyRouter",
    "build_emergency_response",
    "contains_prohibited_medical_advice",
    "detect_acute_category",
    "is_emergency",
]

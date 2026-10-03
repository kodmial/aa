"""Safety routing boundary."""

from __future__ import annotations

from aa.safety.emergency import (
    EmergencyCategory,
    EmergencyClassification,
    classify_emergency,
    detect_language,
    is_emergency,
)
from aa.safety.response import (
    EMERGENCY_RESPONSE_EN,
    EMERGENCY_RESPONSE_RU,
    UNSAFE_RESPONSE_PATTERNS,
    build_emergency_response,
)
from aa.safety.router import SafetyDecision, SafetyResult, SafetyRouter

__all__ = [
    "EMERGENCY_RESPONSE_EN",
    "EMERGENCY_RESPONSE_RU",
    "UNSAFE_RESPONSE_PATTERNS",
    "EmergencyCategory",
    "EmergencyClassification",
    "SafetyDecision",
    "SafetyResult",
    "SafetyRouter",
    "build_emergency_response",
    "classify_emergency",
    "detect_language",
    "is_emergency",
]

"""Safety routing boundary."""

from __future__ import annotations

from aa.safety.emergency import (
    EmergencyCategory,
    EmergencyClassification,
    classify_emergency,
    detect_language,
    is_emergency,
)
from aa.safety.outbound import (
    OUTBOUND_SAFETY_MAX_REPAIRS,
    SAFE_RECOVERY_INSTRUCTION,
    OutboundSafetyVerdict,
    classify_outbound_safety,
    is_outbound_safe,
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
    "OUTBOUND_SAFETY_MAX_REPAIRS",
    "SAFE_RECOVERY_INSTRUCTION",
    "UNSAFE_RESPONSE_PATTERNS",
    "EmergencyCategory",
    "EmergencyClassification",
    "OutboundSafetyVerdict",
    "SafetyDecision",
    "SafetyResult",
    "SafetyRouter",
    "build_emergency_response",
    "classify_emergency",
    "classify_outbound_safety",
    "detect_language",
    "is_emergency",
    "is_outbound_safe",
]

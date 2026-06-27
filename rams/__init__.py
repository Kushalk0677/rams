"""RAMS: Resource-Adaptive Model Switching for Edge AI.

Dynamically switches among YOLOv8 detector tiers (NANO, SMALL, MEDIUM)
based on real-time system resource pressure, with a safety override
for vulnerable road user detection.

Exports:
    RAMSController, Tier, ModelLibrary, ResourceMonitor,
    make_policy, ThresholdPolicy, PredictivePolicy, SafetyPolicy
"""

from rams.controller import RAMSController
from rams.models     import Tier, ModelLibrary
from rams.monitor    import ResourceMonitor
from rams.policy     import make_policy, ThresholdPolicy, PredictivePolicy, SafetyPolicy

__all__ = [
    "RAMSController",
    "Tier", "ModelLibrary",
    "ResourceMonitor",
    "make_policy", "ThresholdPolicy", "PredictivePolicy", "SafetyPolicy",
]

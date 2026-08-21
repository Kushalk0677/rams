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
from rams.energy     import (PowerProfile, TDPEstimate, TelemetryEnergyEstimate,
                             estimate_profile_energy, estimate_tdp_energy, load_power_profile)
from rams.policy     import (make_policy, ThresholdPolicy, PredictivePolicy,
                             SafetyPolicy, FixedTierPolicy, PAPER_POLICY_LABELS)

__all__ = [
    "RAMSController",
    "Tier", "ModelLibrary",
    "ResourceMonitor",
    "make_policy", "ThresholdPolicy", "PredictivePolicy", "SafetyPolicy",
    "FixedTierPolicy", "PAPER_POLICY_LABELS",
    "PowerProfile", "TDPEstimate", "TelemetryEnergyEstimate",
    "estimate_profile_energy", "estimate_tdp_energy", "load_power_profile",
]

"""Regression tests for the letter-revision runtime contracts."""

from __future__ import annotations

from rams.models import Tier, _onnx_provider_plan
from rams.monitor import compute_pressure
from rams.policy import FixedTierPolicy, SafetyPolicy, SafetyTwoLevelPolicy, _bbox_area_fraction, make_policy


def test_gpu_pressure_only_applies_when_an_accelerator_is_present() -> None:
    cpu_only = compute_pressure(50, 25, None, None)
    gpu_busy = compute_pressure(50, 25, None, None, gpu_util=95, gpu_mem=0.8)
    assert gpu_busy > cpu_only


def test_coreml_backend_uses_coreml_first_with_cpu_fallback() -> None:
    providers, options = _onnx_provider_plan("coreml")
    assert providers[0][0] == "CoreMLExecutionProvider"
    assert providers[1] == "CPUExecutionProvider"
    assert options["ModelFormat"] == "MLProgram"
    assert options["MLComputeUnits"] == "ALL"


def test_normalized_bbox_area_is_resolution_invariant() -> None:
    nano = {"xyxy": [0, 0, 32, 32], "image_width": 320, "image_height": 320}
    medium = {"xyxy": [0, 0, 64, 64], "image_width": 640, "image_height": 640}
    assert _bbox_area_fraction(nano) == _bbox_area_fraction(medium) == 0.01


def test_two_level_policy_uses_source_image_area_when_present() -> None:
    policy = SafetyTwoLevelPolicy(hysteresis_window=1, proximity_window_s=100,
                                  near_area_fraction=0.02)
    near = [{"class": "person", "conf": 0.9, "xyxy": [0, 0, 200, 200],
             "image_width": 1000, "image_height": 1000}]
    assert policy.select_tier(0.95, Tier.NANO, near) == Tier.MEDIUM


def test_precise_policy_aliases_preserve_legacy_constructors() -> None:
    assert make_policy("ewma_smoothed").name == "predictive"
    assert make_policy("variance_adaptive_ewma").name == "adaptive"
    assert make_policy("vru_retention").name == "safety"
    assert make_policy("vru_retention2").name == "safety2"


def test_fixed_tier_policy_uses_controller_policy_contract() -> None:
    policy = FixedTierPolicy(Tier.NANO)
    assert policy.select_tier(0.0, Tier.MEDIUM) == Tier.NANO
    assert policy.select_tier(1.0, Tier.SMALL) == Tier.NANO


def test_retention_experiment_reads_each_policy_active_lock_state() -> None:
    from experiments.exp12_retention_sensitivity import retention_lock_active

    detection = [{"class": "person", "conf": 0.9, "xyxy": [0, 0, 10, 10],
                  "image_width": 100, "image_height": 100}]
    for policy in (SafetyPolicy(proximity_window_s=10),
                   SafetyTwoLevelPolicy(proximity_window_s=10)):
        assert retention_lock_active(policy) is False
        policy.observe(detection)
        assert retention_lock_active(policy) is True


def test_summary_intervals_are_bounded() -> None:
    from benchmark.run import bootstrap_ci, wilson_interval

    low, high = bootstrap_ci([1.0, 2.0, 3.0], seed=7, n_resamples=100)
    assert low <= high
    rate_low, rate_high = wilson_interval(3, 10)
    assert 0.0 <= rate_low <= rate_high <= 1.0


def test_summary_uses_block_means_when_paired_blocks_are_available() -> None:
    from benchmark.run import compute_summary

    records = [
        {"policy": "threshold", "load_profile": "idle", "block": 0,
         "latency_ms": 1.0, "tier": "NANO", "vru_detected": False,
         "backend": "unit", "tdp_energy_estimate_j": None,
         "energy_profile_estimate_j": None},
        {"policy": "threshold", "load_profile": "idle", "block": 1,
         "latency_ms": 3.0, "tier": "NANO", "vru_detected": False,
         "backend": "unit", "tdp_energy_estimate_j": None,
         "energy_profile_estimate_j": None},
    ]
    group = compute_summary(records)["groups"][0]
    assert group["ci_unit"] == "block_mean"
    assert group["n_blocks"] == 2


def test_burst_schedule_has_distinct_on_and_recovery_windows() -> None:
    from benchmark.run import burst_target_intensity

    targets = [burst_target_intensity(index, 0.85, 2, 3) for index in range(7)]
    assert targets == [0.85, 0.85, 0.0, 0.0, 0.0, 0.85, 0.85]


def test_steady_profiles_use_process_workers_scaled_to_host_cores(monkeypatch) -> None:
    from benchmark.run import LoadInjector

    monkeypatch.setattr("benchmark.load_injector.os.cpu_count", lambda: 16)
    assert LoadInjector(0.0).worker_count == 0
    assert LoadInjector(0.50).worker_count == 15
    assert LoadInjector(0.75).worker_count == 15
    assert LoadInjector(0.75).reserve_logical_cpus == 1
    # Fifteen workers on a 16-logical-core host need an 80% duty cycle to
    # request a 75% whole-host load.
    assert LoadInjector(0.75)._duty.value == 0.8


def test_steady_load_summary_does_not_report_burst_windows() -> None:
    from benchmark.run import compute_summary

    records = [
        {"policy": "threshold", "load_profile": "moderate", "block": 0,
         "latency_ms": 1.0, "tier": "NANO", "vru_detected": False,
         "backend": "unit", "load_intensity": 0.5,
         "tdp_energy_estimate_j": None, "energy_profile_estimate_j": None},
        {"policy": "threshold", "load_profile": "moderate", "block": 0,
         "latency_ms": 2.0, "tier": "NANO", "vru_detected": False,
         "backend": "unit", "load_intensity": 0.5,
         "tdp_energy_estimate_j": None, "energy_profile_estimate_j": None},
    ]
    group = compute_summary(records)["groups"][0]
    assert group["burst_on_frames"] == group["burst_off_frames"] == 0
    assert group["burst_on_latency_mean"] is None


def test_tdp_energy_is_explicitly_an_estimate() -> None:
    from rams.energy import estimate_tdp_energy

    estimate = estimate_tdp_energy(100.0, 20.0, "test power budget")
    assert estimate is not None
    assert estimate.joules == 2.0
    assert estimate.is_measured is False


def test_telemetry_profile_estimate_uses_dynamic_utilization_and_cap() -> None:
    from rams.energy import PowerProfile, estimate_profile_energy

    profile = PowerProfile.from_mapping({
        "name": "unit", "source": "unit test", "idle_w": 2.0,
        "cpu_dynamic_w": 10.0, "memory_dynamic_w": 4.0,
        "gpu_dynamic_w": 20.0, "max_power_w": 25.0,
    })
    estimate = estimate_profile_energy(
        100.0, profile, cpu_percent=50.0, memory_percent=50.0,
        gpu_utilization_percent=100.0,
    )
    assert estimate.estimated_power_w == 25.0  # 29 W raw, capped to profile budget
    assert estimate.estimated_energy_j == 2.5
    assert estimate.is_measured is False


def test_replay_manifest_hashes_every_frame(tmp_path) -> None:
    from benchmark.run import build_replay_manifest

    frame = tmp_path / "frame.png"
    frame.write_bytes(b"deterministic-frame")
    manifest = build_replay_manifest([frame], seed=5, run_id="unit")
    assert manifest["seed"] == 5
    assert manifest["frames"][0]["name"] == "frame.png"
    assert len(manifest["frames"][0]["sha256"]) == 64

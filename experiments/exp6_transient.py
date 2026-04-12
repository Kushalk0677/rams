"""
Experiment 6 — Transient Spike Response
========================================
Injects periodic sudden CPU bursts (spikes) and measures how each
policy responds: time-to-downgrade, latency spike magnitude, and
time-to-recover after the spike clears.

This is a *dynamic* test — unlike experiments 1–3 which measure
steady-state behaviour under fixed load.

Why it matters for the paper
-----------------------------
  - Threshold policy: reactive — responds only after hysteresis window.
  - Predictive policy: anticipatory — EWMA forecast rises before the
    spike peaks, so it downgrades earlier.
  - Adaptive policy: fastest — high-variance alpha means it reacts to
    spike onset more aggressively than fixed EWMA.
  - Safety policies: same responsiveness as threshold, but tier floor
    is maintained when a VRU is present during the spike.

Design
------
Each trial is 300 inferences (~60 s at 5 Hz inference rate).
Spikes are injected at inference indices 60, 130, 200 (every ~14 s).
Each spike:  intensity 0.95 for 50 inferences (~10 s), then back to
             baseline 0.15 (light idle).

Outputs
-------
  results/exp6_transient_trace_<policy>.png  (latency + tier timeline)
  results/exp6_response_summary.png          (summary metrics bar chart)
  results/exp6_transient.csv / .json
  results/exp6_latex.tex
"""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.utils import (
    save_records_csv, save_stats_json, _get_mpl, TrialRecord,
)
from rams.controller import RAMSController
from rams.models import Tier

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

ALL_POLICIES = ["threshold", "predictive", "safety", "adaptive", "safety2"]

# Spike schedule: list of (spike_start_idx, spike_end_idx)
SPIKE_SCHEDULE = [(60, 110), (140, 190), (220, 270)]
TOTAL_N        = 310
BASELINE_INT   = 0.15   # light background load
SPIKE_INT      = 0.95   # burst intensity


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic load injector with on-the-fly intensity control
# ─────────────────────────────────────────────────────────────────────────────

class DynamicLoadInjector:
    """Adjustable-intensity load injector."""

    def __init__(self, initial_intensity: float = 0.0):
        self._intensity = max(0.0, min(1.0, initial_intensity))
        self._stop      = threading.Event()
        self._threads: list[threading.Thread] = []

    @property
    def intensity(self) -> float:
        return self._intensity

    @intensity.setter
    def intensity(self, v: float):
        self._intensity = max(0.0, min(1.0, v))

    def _burn(self):
        while not self._stop.is_set():
            _ = sum(i * i for i in range(2000))
            time.sleep(max(0.0, (1.0 - self._intensity) * 0.0005))

    def start(self):
        for _ in range(4):
            t = threading.Thread(target=self._burn, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.0)

    def __enter__(self):  self.start();  return self
    def __exit__(self, *_): self.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Response metrics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpikeResponse:
    spike_id:         int
    downgrade_delay:  Optional[int]    # inferences until first tier drop (None = never)
    recovery_delay:   Optional[int]    # inferences after spike end until MEDIUM restored
    max_latency_ms:   float            # peak latency during spike
    baseline_lat_ms:  float            # pre-spike mean latency


def analyse_spike(
    tiers:      list[str],
    latencies:  list[float],
    spike_start: int,
    spike_end:   int,
    spike_id:    int,
) -> SpikeResponse:
    pre_lat = latencies[max(0, spike_start - 20): spike_start]
    baseline_lat = statistics.mean(pre_lat) if pre_lat else 0.0
    max_lat = max(latencies[spike_start: spike_end]) if spike_start < len(latencies) else 0.0

    # Downgrade delay: first inference in spike window where tier drops below MEDIUM
    downgrade_delay: Optional[int] = None
    for i in range(spike_start, min(spike_end, len(tiers))):
        if tiers[i] in ("NANO", "SMALL"):
            downgrade_delay = i - spike_start
            break

    # Recovery delay: first inference after spike end where tier returns to MEDIUM
    recovery_delay: Optional[int] = None
    for i in range(spike_end, len(tiers)):
        if tiers[i] == "MEDIUM":
            recovery_delay = i - spike_end
            break

    return SpikeResponse(
        spike_id=spike_id,
        downgrade_delay=downgrade_delay,
        recovery_delay=recovery_delay,
        max_latency_ms=max_lat,
        baseline_lat_ms=baseline_lat,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-policy trial
# ─────────────────────────────────────────────────────────────────────────────

def run_policy_spike_trial(
    policy_name: str,
    simulate: bool = True,
) -> tuple[list[TrialRecord], list[SpikeResponse]]:

    records:   list[TrialRecord]   = []
    responses: list[SpikeResponse] = []
    injector = DynamicLoadInjector(initial_intensity=BASELINE_INT)

    with injector:
        with RAMSController(simulate=simulate, policy=policy_name) as ctrl:
            time.sleep(0.4)
            prev_tier = ctrl.current_tier

            for idx in range(TOTAL_N):
                # Update load intensity based on spike schedule
                in_spike = any(s <= idx < e for s, e in SPIKE_SCHEDULE)
                injector.intensity = SPIKE_INT if in_spike else BASELINE_INT

                ctrl.set_pressure_override(SPIKE_INT * 0.93 if in_spike else BASELINE_INT * 0.28)
                res = ctrl.infer()
                cur_tier = ctrl.current_tier
                vru = any(
                    d.get("class", "").lower() in
                    {"person", "pedestrian", "cyclist", "bicycle"}
                    for d in res.get("detections", [])
                )
                records.append(TrialRecord(
                    label=policy_name,
                    group="transient",
                    latency_ms=res["latency_ms"],
                    pressure=res.get("pressure", 0.0),
                    tier=res["tier"],
                    n_detections=len(res.get("detections", [])),
                    vru_detected=vru,
                    switch_occurred=(cur_tier != prev_tier),
                    accuracy_proxy=float(res.get("accuracy_proxy", 0.0)),
                ))
                prev_tier = cur_tier

    tiers     = [r.tier       for r in records]
    latencies = [r.latency_ms for r in records]

    for sid, (s_start, s_end) in enumerate(SPIKE_SCHEDULE):
        resp = analyse_spike(tiers, latencies, s_start, s_end, sid + 1)
        responses.append(resp)

    return records, responses


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

TIER_Y = {"MEDIUM": 3, "SMALL": 2, "NANO": 1}
TIER_COLORS_PLT = {"NANO": "#e74c3c", "SMALL": "#f39c12", "MEDIUM": "#2ecc71"}


def trace_plot(records: list[TrialRecord], policy_name: str, out: Path):
    plt = _get_mpl()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    idxs     = list(range(len(records)))
    latencies = [r.latency_ms for r in records]
    tier_ys   = [TIER_Y.get(r.tier, 2) for r in records]

    # Shade spike windows
    for s, e in SPIKE_SCHEDULE:
        ax1.axvspan(s, e, alpha=0.10, color="#e74c3c")
        ax2.axvspan(s, e, alpha=0.10, color="#e74c3c")

    ax1.plot(idxs, latencies, linewidth=0.9, color="#2980b9", alpha=0.85)
    ax1.set_ylabel("Latency (ms)", fontsize=10)
    ax1.set_title(f"Exp 6 — Transient Spike Response  [{policy_name}]",
                  fontsize=12, fontweight="bold")
    ax1.grid(alpha=0.25)

    # Tier timeline (step plot)
    colors = [TIER_COLORS_PLT.get(r.tier, "#95a5a6") for r in records]
    ax2.scatter(idxs, tier_ys, c=colors, s=8, alpha=0.8, zorder=3)
    ax2.step(idxs, tier_ys, linewidth=0.7, color="#555", alpha=0.4, zorder=2)
    ax2.set_yticks([1, 2, 3])
    ax2.set_yticklabels(["NANO", "SMALL", "MEDIUM"], fontsize=9)
    ax2.set_xlabel("Inference index", fontsize=10)
    ax2.set_ylabel("Active tier", fontsize=10)
    ax2.grid(alpha=0.25)

    # Legend for spikes
    from matplotlib.patches import Patch
    ax1.legend([Patch(facecolor="#e74c3c", alpha=0.2, label="Spike window")],
               ["Spike window"], fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Trace saved → {out.name}")


def summary_bar(
    policy_names:  list[str],
    all_responses: dict[str, list[SpikeResponse]],
    out: Path,
):
    plt = _get_mpl()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    colors = ["#3266ad", "#1D9E75", "#D85A30", "#8e44ad", "#c0392b"]

    def _mean_opt(vals):
        vs = [v for v in vals if v is not None]
        return statistics.mean(vs) if vs else float("nan")

    downgrade_delays = [_mean_opt([r.downgrade_delay for r in all_responses[p]])
                        for p in policy_names]
    recovery_delays  = [_mean_opt([r.recovery_delay  for r in all_responses[p]])
                        for p in policy_names]
    max_latencies    = [statistics.mean([r.max_latency_ms for r in all_responses[p]])
                        for p in policy_names]

    import math
    def clean(vals):
        return [v if not math.isnan(v) else 0 for v in vals]

    x = list(range(len(policy_names)))

    axes[0].bar(x, clean(downgrade_delays), color=colors[:len(policy_names)], alpha=0.85)
    axes[0].set_title("Downgrade delay (inferences)", fontsize=10, fontweight="bold")
    axes[0].set_xticks(x); axes[0].set_xticklabels(policy_names, fontsize=8, rotation=20)
    axes[0].set_ylabel("Inferences"); axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, clean(recovery_delays), color=colors[:len(policy_names)], alpha=0.85)
    axes[1].set_title("Recovery delay (inferences)", fontsize=10, fontweight="bold")
    axes[1].set_xticks(x); axes[1].set_xticklabels(policy_names, fontsize=8, rotation=20)
    axes[1].set_ylabel("Inferences"); axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(x, max_latencies, color=colors[:len(policy_names)], alpha=0.85)
    axes[2].set_title("Peak latency during spike (ms)", fontsize=10, fontweight="bold")
    axes[2].set_xticks(x); axes[2].set_xticklabels(policy_names, fontsize=8, rotation=20)
    axes[2].set_ylabel("ms"); axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("Exp 6 — Transient Spike Response Summary", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Summary chart → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(simulate: bool = True):
    print("\n" + "═" * 62)
    print("  Experiment 6 — Transient Spike Response")
    print(f"  N={TOTAL_N} per policy  simulate={simulate}")
    print(f"  Spike schedule: {SPIKE_SCHEDULE}")
    print("═" * 62)

    all_records:   list[TrialRecord]                   = []
    all_responses: dict[str, list[SpikeResponse]]      = {}

    for policy_name in ALL_POLICIES:
        print(f"\n  policy={policy_name} ...", flush=True)
        records, responses = run_policy_spike_trial(policy_name, simulate)
        all_records.extend(records)
        all_responses[policy_name] = responses

        for resp in responses:
            print(
                f"    spike {resp.spike_id}: "
                f"downgrade_delay={resp.downgrade_delay}  "
                f"recovery_delay={resp.recovery_delay}  "
                f"peak_lat={resp.max_latency_ms:.1f} ms"
            )

        trace_plot(records, policy_name,
                   RESULTS / f"exp6_trace_{policy_name}.png")

    summary_bar(ALL_POLICIES, all_responses,
                RESULTS / "exp6_response_summary.png")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print(f"  {'Policy':<14} {'Dwngrade(inf)':>14} {'Recovery(inf)':>14} {'Peak lat (ms)':>14}")
    print("─" * 70)
    for p in ALL_POLICIES:
        resps = all_responses[p]
        dd = [r.downgrade_delay for r in resps if r.downgrade_delay is not None]
        rd = [r.recovery_delay  for r in resps if r.recovery_delay  is not None]
        pl = [r.max_latency_ms  for r in resps]
        dd_s = f"{statistics.mean(dd):.1f}" if dd else "N/A"
        rd_s = f"{statistics.mean(rd):.1f}" if rd else "N/A"
        print(f"  {p:<14} {dd_s:>14} {rd_s:>14} {statistics.mean(pl):>14.1f}")
    print("═" * 70)

    # ── Save ─────────────────────────────────────────────────────────────────
    save_records_csv(all_records, RESULTS / "exp6_transient.csv")

    import json
    summary_data = {}
    for p in ALL_POLICIES:
        summary_data[p] = [
            {
                "spike_id":        r.spike_id,
                "downgrade_delay": r.downgrade_delay,
                "recovery_delay":  r.recovery_delay,
                "max_latency_ms":  round(r.max_latency_ms, 2),
                "baseline_lat_ms": round(r.baseline_lat_ms, 2),
            }
            for r in all_responses[p]
        ]
    (RESULTS / "exp6_transient.json").write_text(
        json.dumps(summary_data, indent=2))
    print(f"  JSON → results/exp6_transient.json\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate",    action="store_true", default=True)
    parser.add_argument("--no-simulate", dest="simulate", action="store_false")
    args = parser.parse_args()
    run(simulate=args.simulate)

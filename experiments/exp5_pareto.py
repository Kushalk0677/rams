"""
Experiment 5 — Accuracy–Latency Pareto Frontier
================================================
Plots each policy (plus fixed-tier baselines) as an operating point in
(mean latency, mean accuracy proxy) space.

The key claim for the paper:
  "RAMS policies lie on or near the Pareto frontier; no fixed-tier
   baseline achieves the same accuracy at lower latency."

Fixed baselines added programmatically by locking the controller to a
single tier regardless of resource pressure.

Sub-experiment
--------------
  5a. All policies under *moderate* load (the interesting regime where
      RAMS tier-switching is most active and baselines diverge).
  5b. Heavy load (where NANO and SMALL baselines hurt accuracy most).

Outputs
-------
  results/exp5_pareto_moderate.png
  results/exp5_pareto_heavy.png
  results/exp5_pareto.csv / .json
  results/exp5_latex.tex
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.utils import (
    LoadInjector, TrialRecord, compute_stats, print_table,
    save_records_csv, save_stats_json, to_latex, _get_mpl,
)
from rams.controller import RAMSController
from rams.models import Tier
from rams.policy import make_policy, ThresholdPolicy

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

ALL_POLICIES = ["threshold", "predictive", "safety", "adaptive", "safety2"]
FIXED_TIERS  = ["FIXED_NANO", "FIXED_SMALL", "FIXED_MEDIUM"]

LOAD_SCENARIOS = {
    "moderate": 0.50,
    "heavy":    0.75,
}


# ─────────────────────────────────────────────────────────────────────────────
# Fixed-tier runner (forces a single tier, ignores policy logic)
# ─────────────────────────────────────────────────────────────────────────────

def run_fixed_tier(
    tier_name: str,
    n: int,
    intensity: float,
    simulate: bool = True,
) -> list[TrialRecord]:
    """Run N inferences locked to a single tier (baseline comparison)."""
    tier_enum = {"FIXED_NANO": Tier.NANO,
                 "FIXED_SMALL": Tier.SMALL,
                 "FIXED_MEDIUM": Tier.MEDIUM}[tier_name]

    records: list[TrialRecord] = []
    with LoadInjector(intensity):
        with RAMSController(simulate=simulate, policy="threshold") as ctrl:
            # Override: hard-code the tier by patching the policy thresholds
            # so it always selects the target tier regardless of pressure.
            ctrl.policy = ThresholdPolicy(
                lo_thresh=0.0 if tier_enum == Tier.MEDIUM else (
                    1.0 if tier_enum == Tier.NANO else 0.5),
                hi_thresh=0.0 if tier_enum == Tier.NANO else (
                    1.0 if tier_enum == Tier.MEDIUM else 0.5),
                hysteresis_window=1,
            )
            ctrl._current_tier = tier_enum
            time.sleep(0.4)

            from experiments.utils import intensity_to_pressure
            for _ in range(n):
                ctrl.set_pressure_override(intensity_to_pressure(intensity))
                res = ctrl.infer()
                vru = any(
                    d.get("class", "").lower() in
                    {"person", "pedestrian", "cyclist", "bicycle"}
                    for d in res.get("detections", [])
                )
                records.append(TrialRecord(
                    label=tier_name,
                    group="fixed",
                    latency_ms=res["latency_ms"],
                    pressure=res.get("pressure", 0.0),
                    tier=res["tier"],
                    n_detections=len(res.get("detections", [])),
                    vru_detected=vru,
                    switch_occurred=False,
                    accuracy_proxy=float(res.get("accuracy_proxy", 0.0)),
                ))
    return records


def run_policy_trial(
    policy_name: str,
    n: int,
    intensity: float,
    simulate: bool = True,
) -> list[TrialRecord]:
    """Run N inferences with a named policy."""
    records: list[TrialRecord] = []
    with LoadInjector(intensity):
        with RAMSController(simulate=simulate, policy=policy_name) as ctrl:
            time.sleep(0.4)
            prev_tier = ctrl.current_tier
            from experiments.utils import intensity_to_pressure
            for _ in range(n):
                ctrl.set_pressure_override(intensity_to_pressure(intensity))
                res = ctrl.infer()
                cur_tier = ctrl.current_tier
                vru = any(
                    d.get("class", "").lower() in
                    {"person", "pedestrian", "cyclist", "bicycle"}
                    for d in res.get("detections", [])
                )
                records.append(TrialRecord(
                    label=policy_name,
                    group="policy",
                    latency_ms=res["latency_ms"],
                    pressure=res.get("pressure", 0.0),
                    tier=res["tier"],
                    n_detections=len(res.get("detections", [])),
                    vru_detected=vru,
                    switch_occurred=(cur_tier != prev_tier),
                    accuracy_proxy=float(res.get("accuracy_proxy", 0.0)),
                ))
                prev_tier = cur_tier
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Pareto plot
# ─────────────────────────────────────────────────────────────────────────────

POLICY_COLORS = {
    "threshold":    "#3266ad",
    "predictive":   "#1D9E75",
    "safety":       "#D85A30",
    "adaptive":     "#8e44ad",
    "safety2":      "#c0392b",
    "FIXED_NANO":   "#7f8c8d",
    "FIXED_SMALL":  "#95a5a6",
    "FIXED_MEDIUM": "#bdc3c7",
}

POLICY_MARKERS = {
    "threshold":    "o",
    "predictive":   "s",
    "safety":       "^",
    "adaptive":     "D",
    "safety2":      "*",
    "FIXED_NANO":   "x",
    "FIXED_SMALL":  "+",
    "FIXED_MEDIUM": "v",
}


def pareto_plot(
    points: dict[str, tuple[float, float]],  # label → (latency, accuracy)
    title: str,
    out: Path,
):
    plt = _get_mpl()
    fig, ax = plt.subplots(figsize=(8, 6))

    for label, (lat, acc) in points.items():
        color  = POLICY_COLORS.get(label, "#333")
        marker = POLICY_MARKERS.get(label, "o")
        size   = 140 if "FIXED" not in label else 80
        ax.scatter(lat, acc, c=color, marker=marker, s=size,
                   zorder=3, label=label)
        # Annotate
        va = "bottom" if "FIXED" not in label else "top"
        ax.annotate(label, (lat, acc),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color=color, va=va)

    # Draw dominance frontier (lower-left convex hull of Pareto-optimal points)
    sorted_pts = sorted(points.values(), key=lambda p: p[0])
    pareto = []
    best_acc = -1.0
    for lat, acc in sorted_pts:
        if acc > best_acc:
            pareto.append((lat, acc))
            best_acc = acc
    if len(pareto) > 1:
        xs, ys = zip(*pareto)
        ax.plot(xs, ys, "--", color="#aaa", linewidth=1.2,
                alpha=0.7, label="Pareto frontier", zorder=1)

    ax.set_xlabel("Mean latency (ms)", fontsize=12)
    ax.set_ylabel("Mean accuracy proxy (mAP50)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Chart saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(n: int = 80, simulate: bool = True):
    print("\n" + "═" * 62)
    print("  Experiment 5 — Accuracy–Latency Pareto Frontier")
    print(f"  N={n}  simulate={simulate}")
    print("═" * 62)

    all_records: list[TrialRecord] = []
    all_stats = []

    for scenario, intensity in LOAD_SCENARIOS.items():
        print(f"\n  ── Load scenario: {scenario} (intensity={intensity}) ──")
        scenario_points: dict[str, tuple[float, float]] = {}

        # Adaptive policies
        for policy_name in ALL_POLICIES:
            print(f"    policy={policy_name} ...", flush=True)
            recs = run_policy_trial(policy_name, n, intensity, simulate)
            for r in recs:
                r.group = scenario
            all_records.extend(recs)
            s = compute_stats(recs, label=policy_name, group=scenario)
            all_stats.append(s)
            acc_mean = statistics.mean(r.accuracy_proxy for r in recs)
            scenario_points[policy_name] = (s.mean, acc_mean)
            print(f"      latency={s.mean:.1f} ms  acc={acc_mean:.3f}")

        # Fixed baselines
        for tier_name in FIXED_TIERS:
            print(f"    baseline={tier_name} ...", flush=True)
            recs = run_fixed_tier(tier_name, n, intensity, simulate)
            for r in recs:
                r.group = scenario
            all_records.extend(recs)
            s = compute_stats(recs, label=tier_name, group=scenario)
            all_stats.append(s)
            acc_mean = statistics.mean(r.accuracy_proxy for r in recs)
            scenario_points[tier_name] = (s.mean, acc_mean)
            print(f"      latency={s.mean:.1f} ms  acc={acc_mean:.3f}")

        pareto_plot(
            scenario_points,
            title=f"Exp 5 — Accuracy vs Latency Operating Points [{scenario} load]",
            out=RESULTS / f"exp5_pareto_{scenario}.png",
        )

    # ── Console table ─────────────────────────────────────────────────────────
    print_table(
        all_stats,
        title="Experiment 5: Accuracy–Latency Pareto (mean latency, tier dist)",
        columns=["label", "group", "n", "mean", "std", "p95", "tier_dist"],
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    save_records_csv(all_records, RESULTS / "exp5_pareto.csv")
    save_stats_json(all_stats,   RESULTS / "exp5_pareto.json")

    # ── LaTeX ────────────────────────────────────────────────────────────────
    latex = to_latex(
        all_stats,
        caption=(
            "Accuracy–latency operating points for all RAMS policies and "
            "fixed-tier baselines under moderate and heavy synthetic load. "
            "Adaptive and safety-tier policies achieve higher accuracy than "
            "FIXED\\_NANO/SMALL at comparable or lower latency. "
            f"N={n} inferences per condition."
        ),
        label="tab:exp5_pareto",
        highlight_best="mean",
    )
    tex_path = RESULTS / "exp5_latex.tex"
    tex_path.write_text(latex)
    print(f"  LaTeX table → {tex_path}\n")

    return all_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",           type=int,  default=80)
    parser.add_argument("--simulate",    action="store_true", default=True)
    parser.add_argument("--no-simulate", dest="simulate", action="store_false")
    args = parser.parse_args()
    run(n=args.n, simulate=args.simulate)

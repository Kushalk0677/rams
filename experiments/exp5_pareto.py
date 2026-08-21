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
import random
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
from rams.policy import FixedTierPolicy, PAPER_POLICY_LABELS

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

def _read_frame(path: Path):
    import cv2
    frame = cv2.imread(str(path))
    if frame is None:
        raise IOError(f"Could not read frame: {path}")
    return frame


def run_controller_trial(
    label: str,
    policy: str | FixedTierPolicy,
    frame_paths: list[Path] | None,
    n: int,
    intensity: float,
    simulate: bool = True,
    block: int = 0,
) -> list[TrialRecord]:
    """Run every method through the same controller and replay path."""
    records: list[TrialRecord] = []
    with LoadInjector(intensity):
        with RAMSController(simulate=simulate, policy=policy) as ctrl:
            # Let process-based steady load settle before the replay begins.
            time.sleep(1.0)
            prev_tier = ctrl.current_tier
            for index in range(n):
                path = frame_paths[index % len(frame_paths)] if frame_paths else None
                frame = _read_frame(path) if path else None
                # Publication runs must use the controller's measured pressure.
                # The previous unconditional synthetic override used a fixed
                # pre-calibration pressure mapping, which could disagree with
                # device-specific calibrated thresholds.  Keep that mechanism
                # only for simulated runs, where no real telemetry exists.
                if simulate:
                    from experiments.utils import intensity_to_pressure
                    ctrl.set_pressure_override(intensity_to_pressure(intensity))
                res = ctrl.infer(frame=frame)
                cur_tier = ctrl.current_tier
                vru = any(
                    d.get("class", "").lower() in
                    {"person", "pedestrian", "cyclist", "bicycle"}
                    for d in res.get("detections", [])
                )
                records.append(TrialRecord(
                    label=label,
                    group="paired",
                    latency_ms=res["end_to_end_ms"],
                    pressure=res.get("pressure", 0.0),
                    tier=res["tier"],
                    n_detections=len(res.get("detections", [])),
                    vru_detected=vru,
                    switch_occurred=(cur_tier != prev_tier),
                    accuracy_proxy=float(res.get("accuracy_proxy", 0.0)),
                    frame_name=path.name if path else "",
                    block=block,
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

def run(n: int = 80, simulate: bool = True, frames: str | None = None,
        blocks: int = 1, seed: int = 20260710, scenarios: list[str] | None = None):
    """Run paired, frame-aware Pareto cells.

    Each block randomizes method order while every method sees the same frame
    slice. Paper-facing runs must pass ``frames`` and ``simulate=False``.
    """
    print("\n" + "═" * 62)
    print("  Experiment 5 — Accuracy–Latency Pareto Frontier")
    print(f"  N={n}  simulate={simulate}")
    print("═" * 62)

    frame_paths = None
    if frames:
        directory = Path(frames)
        frame_paths = sorted(list(directory.glob("*.png")) + list(directory.glob("*.jpg")) +
                             list(directory.glob("*.jpeg")))
        if not frame_paths:
            raise ValueError(f"No replay frames found in {frames}")
        random.Random(seed).shuffle(frame_paths)
    elif not simulate:
        raise ValueError("Real Experiment 5 runs require --frames for paired replay")

    all_records: list[TrialRecord] = []
    all_stats = []
    selected_scenarios = scenarios or list(LOAD_SCENARIOS)
    unknown_scenarios = set(selected_scenarios) - set(LOAD_SCENARIOS)
    if unknown_scenarios:
        raise ValueError(f"Unknown load scenarios: {', '.join(sorted(unknown_scenarios))}")

    for scenario in selected_scenarios:
        intensity = LOAD_SCENARIOS[scenario]
        print(f"\n  ── Load scenario: {scenario} (intensity={intensity}) ──")
        scenario_points: dict[str, tuple[float, float]] = {}

        methods: list[tuple[str, str | FixedTierPolicy]] = [(name, name) for name in ALL_POLICIES]
        methods.extend([
            ("FIXED_NANO", FixedTierPolicy(Tier.NANO)),
            ("FIXED_SMALL", FixedTierPolicy(Tier.SMALL)),
            ("FIXED_MEDIUM", FixedTierPolicy(Tier.MEDIUM)),
        ])
        # `n` is deliberately per block, matching the paper replay protocol.
        # This keeps a 10-block x 200-frame run at the intended sample size.
        per_block = n
        grouped: dict[str, list[TrialRecord]] = {label: [] for label, _ in methods}
        for block in range(blocks):
            order = methods[:]
            random.Random(seed + block).shuffle(order)
            start = block * per_block
            block_frames = frame_paths[start:start + per_block] if frame_paths else None
            if frame_paths and not block_frames:
                block_frames = frame_paths[:per_block]
            for label, policy in order:
                print(f"    block={block + 1}/{blocks} method={label} ...", flush=True)
                recs = run_controller_trial(label, policy, block_frames, per_block,
                                            intensity, simulate, block)
                for record in recs:
                    record.group = scenario
                grouped[label].extend(recs)
                all_records.extend(recs)

        for label, recs in grouped.items():
            stats = compute_stats(recs, label=label, group=scenario)
            all_stats.append(stats)
            acc_mean = statistics.mean(record.accuracy_proxy for record in recs)
            scenario_points[label] = (stats.mean, acc_mean)
            print(f"      {label}: latency={stats.mean:.1f} ms  acc_proxy={acc_mean:.3f}")

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
    scenario_tag = "_".join(selected_scenarios)
    save_records_csv(all_records, RESULTS / f"exp5_pareto_{scenario_tag}.csv")
    save_stats_json(all_stats,   RESULTS / f"exp5_pareto_{scenario_tag}.json")

    # ── LaTeX ────────────────────────────────────────────────────────────────
    latex = to_latex(
        all_stats,
        caption=(
            "Accuracy–latency operating points for all RAMS policies and "
            "fixed-tier baselines under paired moderate and heavy replay load. "
            "Accuracy is a tier-level proxy and must be interpreted alongside "
            "policy-level detection metrics. "
            f"N={n} inferences per condition."
        ),
        label="tab:exp5_pareto",
        highlight_best="mean",
    )
    tex_path = RESULTS / f"exp5_latex_{scenario_tag}.tex"
    tex_path.write_text(latex)
    print(f"  LaTeX table → {tex_path}\n")

    return all_stats


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",           type=int,  default=80,
                        help="frames per randomized paired block")
    parser.add_argument("--frames",      type=str, default=None,
                        help="KITTI replay directory; required for --no-simulate")
    parser.add_argument("--blocks",      type=int, default=1,
                        help="randomized paired blocks per scenario")
    parser.add_argument("--seed",        type=int, default=20260710)
    parser.add_argument("--scenarios",   default="moderate,heavy",
                        help="comma-separated subset of moderate,heavy")
    parser.add_argument("--simulate",    action="store_true", default=True)
    parser.add_argument("--no-simulate", dest="simulate", action="store_false")
    args = parser.parse_args()
    scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]
    run(n=args.n, simulate=args.simulate, frames=args.frames,
        blocks=args.blocks, seed=args.seed, scenarios=scenarios)

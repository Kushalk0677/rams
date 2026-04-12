"""
Experiment 4 — Safety Override Analysis
Measures how the SafetyPolicy's VRU class-conditional override
affects tier selection and latency compared to the base threshold
policy, across varying VRU detection rates and proximity windows.

Sub-experiments
---------------
  4a. VRU rate vs tier lock rate       (vary simulated VRU frequency)
  4b. Proximity window sweep           (vary proximity_window_s)
  4c. Confidence threshold sweep       (vary min_conf)

Outputs
-------
  results/exp4_safety_override.csv / .json
  results/exp4_vru_rate_vs_lock.png
  results/exp4_proximity_window.png
  results/exp4_conf_threshold.png
  results/exp4_latex.tex
"""

import argparse
import random
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.utils import (
    compute_stats, print_table, to_latex,
    save_records_csv, save_stats_json, _get_mpl, TrialRecord, LoadInjector,
)
from rams.controller import RAMSController
from rams.models import Tier
from rams.policy import SafetyPolicy, ThresholdPolicy

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

VULNERABLE_CLASSES = ["person", "cyclist", "pedestrian"]
HEAVY_INTENSITY    = 0.70    # pressure level that would normally force NANO


# ─────────────────────────────────────────────────────────────────────────────
# Specialised runner: injects synthetic VRU detections at a given rate
# ─────────────────────────────────────────────────────────────────────────────

def run_with_vru(
    label:           str,
    group:           str,
    policy,
    n:               int,
    intensity:       float,
    vru_inject_rate: float,   # probability per inference of injecting a VRU detection
    simulate:        bool = True,
) -> list[TrialRecord]:
    records: list[TrialRecord] = []

    with LoadInjector(intensity):
        with RAMSController(simulate=simulate, policy=policy) as ctrl:
            time.sleep(0.4)
            prev_tier = ctrl.current_tier

            for _ in range(n):
                res = ctrl.infer()

                # Optionally inject a synthetic VRU into the result
                injected_vru = False
                if random.random() < vru_inject_rate:
                    res.setdefault("detections", []).append({
                        "class": random.choice(VULNERABLE_CLASSES),
                        "conf":  random.uniform(0.55, 0.95),
                    })
                    injected_vru = True

                # Feed detections to safety policy override
                if hasattr(ctrl.policy, "select_tier"):
                    ctrl.policy.select_tier(
                        pressure=res.get("pressure", 0.0),
                        last_tier=ctrl.current_tier,
                        recent_detections=res.get("detections", []),
                    )

                cur_tier = ctrl.current_tier
                vru_present = injected_vru or any(
                    d.get("class", "").lower() in {"person", "cyclist", "pedestrian"}
                    for d in res.get("detections", [])
                )

                records.append(TrialRecord(
                    label=label,
                    group=group,
                    latency_ms=res["latency_ms"],
                    pressure=res.get("pressure", 0.0),
                    tier=res["tier"],
                    n_detections=len(res.get("detections", [])),
                    vru_detected=vru_present,
                    switch_occurred=(cur_tier != prev_tier),
                ))
                prev_tier = cur_tier

    return records


def tier_lock_rate(records: list[TrialRecord]) -> float:
    """Fraction of inferences where tier was NOT downgraded to NANO
    despite high pressure — proxy for safety override effectiveness."""
    total = len(records)
    if total == 0:
        return 0.0
    not_nano = sum(1 for r in records if r.tier != "NANO")
    return not_nano / total


# ─────────────────────────────────────────────────────────────────────────────
# Sub-experiment 4a: VRU injection rate sweep
# ─────────────────────────────────────────────────────────────────────────────

def exp4a_vru_rate(n: int, simulate: bool) -> list[TrialRecord]:
    print("\n  [4a] VRU injection rate sweep ...")
    VRU_RATES = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    records   = []

    for rate in VRU_RATES:
        label = f"vru={rate:.1f}"
        # Safety policy
        safety_pol = SafetyPolicy(lo_thresh=0.45, hi_thresh=0.72,
                                  proximity_window_s=0.5, min_conf=0.40)
        recs_safe = run_with_vru(label, "safety", safety_pol, n,
                                 HEAVY_INTENSITY, rate, simulate)
        # Threshold baseline (no VRU awareness)
        thresh_pol = ThresholdPolicy(lo_thresh=0.45, hi_thresh=0.72)
        recs_thresh = run_with_vru(label, "threshold", thresh_pol, n,
                                   HEAVY_INTENSITY, rate, simulate)
        records.extend(recs_safe)
        records.extend(recs_thresh)

        lock_s = tier_lock_rate(recs_safe)
        lock_t = tier_lock_rate(recs_thresh)
        print(f"    VRU rate={rate:.1f}  "
              f"safety lock={lock_s:.2f}  threshold lock={lock_t:.2f}")

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Sub-experiment 4b: Proximity window sweep
# ─────────────────────────────────────────────────────────────────────────────

def exp4b_proximity_window(n: int, simulate: bool) -> list[TrialRecord]:
    print("\n  [4b] Proximity window sweep ...")
    WINDOWS = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0]
    records = []

    for win in WINDOWS:
        label = f"w={win:.2f}s"
        pol = SafetyPolicy(proximity_window_s=win, min_conf=0.40)
        recs = run_with_vru(label, "proximity_window", pol, n,
                            HEAVY_INTENSITY, vru_inject_rate=0.5, simulate=simulate)
        records.extend(recs)
        lock = tier_lock_rate(recs)
        s = compute_stats(recs, label=label, group="proximity_window")
        print(f"    window={win:.2f}s  lock_rate={lock:.2f}  mean={s.mean:.1f} ms")

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Sub-experiment 4c: Confidence threshold sweep
# ─────────────────────────────────────────────────────────────────────────────

def exp4c_conf_threshold(n: int, simulate: bool) -> list[TrialRecord]:
    print("\n  [4c] Confidence threshold sweep ...")
    CONFS   = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    records = []

    for conf in CONFS:
        label = f"conf={conf:.2f}"
        pol = SafetyPolicy(proximity_window_s=0.5, min_conf=conf)
        recs = run_with_vru(label, "conf_threshold", pol, n,
                            HEAVY_INTENSITY, vru_inject_rate=0.5, simulate=simulate)
        records.extend(recs)
        lock = tier_lock_rate(recs)
        s = compute_stats(recs, label=label, group="conf_threshold")
        print(f"    min_conf={conf:.2f}  lock_rate={lock:.2f}  mean={s.mean:.1f} ms")

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(n: int = 50, simulate: bool = True):
    print("\n" + "═" * 60)
    print("  Experiment 4 — Safety Override Analysis")
    print(f"  N={n}  simulate={simulate}  load=heavy ({HEAVY_INTENSITY})")
    print("═" * 60)

    all_records = []
    all_records.extend(exp4a_vru_rate(n, simulate))
    all_records.extend(exp4b_proximity_window(n, simulate))
    all_records.extend(exp4c_conf_threshold(n, simulate))

    # Stats per (label, group)
    from collections import defaultdict
    groups: dict[tuple, list] = defaultdict(list)
    for r in all_records:
        groups[(r.label, r.group)].append(r)
    all_stats = [compute_stats(recs, lbl, grp)
                 for (lbl, grp), recs in groups.items()]

    print_table(
        [s for s in all_stats if s.group in ("safety", "threshold")],
        title="Exp 4a — VRU Rate vs Policy (heavy load)",
        columns=["label", "group", "n", "mean", "std", "tier_dist", "switch_rate"],
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    save_records_csv(all_records, RESULTS / "exp4_safety_override.csv")
    save_stats_json(all_stats,   RESULTS / "exp4_safety_override.json")

    # ── Plots ────────────────────────────────────────────────────────────────
    plt = _get_mpl()

    # 4a plot: VRU injection rate vs lock rate for safety vs threshold
    vru_rates_unique = sorted({
        float(s.label.split("=")[1])
        for s in all_stats if s.group in ("safety", "threshold")
    })

    def lock_series(policy_group):
        out = []
        for rate in vru_rates_unique:
            label = f"vru={rate:.1f}"
            recs  = groups.get((label, policy_group), [])
            out.append(tier_lock_rate(recs))
        return out

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(vru_rates_unique, lock_series("safety"),    "o-",
            color="#2ecc71", linewidth=2, label="SafetyPolicy")
    ax.plot(vru_rates_unique, lock_series("threshold"), "s--",
            color="#e74c3c", linewidth=2, label="ThresholdPolicy")
    ax.set_xlabel("VRU injection rate (per inference)", fontsize=11)
    ax.set_ylabel("Non-NANO fraction (lock rate)", fontsize=11)
    ax.set_title("Exp 4a — VRU Detection Rate vs Safety Lock Rate",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "exp4_vru_rate_vs_lock.png", dpi=150)
    plt.close(fig)
    print(f"\n  Chart saved → results/exp4_vru_rate_vs_lock.png")

    # 4b plot: proximity window vs lock rate + mean latency
    prox_stats = [s for s in all_stats if s.group == "proximity_window"]
    prox_wins  = [float(s.label.split("=")[1].rstrip("s")) for s in prox_stats]
    prox_lock  = [tier_lock_rate(groups[(s.label, s.group)]) for s in prox_stats]
    prox_lat   = [s.mean for s in prox_stats]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.plot(prox_wins, prox_lock, "o-", color="#2ecc71", linewidth=2, label="Lock rate")
    ax2.plot(prox_wins, prox_lat,  "s--", color="#2980b9", linewidth=2, label="Mean lat (ms)")
    ax1.set_xlabel("Proximity window (s)", fontsize=11)
    ax1.set_ylabel("Non-NANO fraction", fontsize=11, color="#2ecc71")
    ax2.set_ylabel("Mean latency (ms)", fontsize=11, color="#2980b9")
    ax1.set_title("Exp 4b — Proximity Window vs Lock Rate & Latency",
                  fontsize=12, fontweight="bold")
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, fontsize=9)
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "exp4_proximity_window.png", dpi=150)
    plt.close(fig)
    print(f"  Chart saved → results/exp4_proximity_window.png")

    # 4c plot: confidence threshold vs lock rate
    conf_stats = [s for s in all_stats if s.group == "conf_threshold"]
    confs      = [float(s.label.split("=")[1]) for s in conf_stats]
    conf_lock  = [tier_lock_rate(groups[(s.label, s.group)]) for s in conf_stats]
    conf_lat   = [s.mean for s in conf_stats]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.plot(confs, conf_lock, "o-", color="#2ecc71", linewidth=2, label="Lock rate")
    ax2.plot(confs, conf_lat,  "s--", color="#2980b9", linewidth=2, label="Mean lat (ms)")
    ax1.set_xlabel("Minimum confidence threshold", fontsize=11)
    ax1.set_ylabel("Non-NANO fraction", fontsize=11, color="#2ecc71")
    ax2.set_ylabel("Mean latency (ms)", fontsize=11, color="#2980b9")
    ax1.set_title("Exp 4c — Confidence Threshold vs Lock Rate & Latency",
                  fontsize=12, fontweight="bold")
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, fontsize=9)
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "exp4_conf_threshold.png", dpi=150)
    plt.close(fig)
    print(f"  Chart saved → results/exp4_conf_threshold.png")

    # ── LaTeX ────────────────────────────────────────────────────────────────
    latex = to_latex(
        [s for s in all_stats if s.group in ("safety", "threshold")],
        caption="Safety override analysis: SafetyPolicy vs ThresholdPolicy under heavy load "
                "with varying VRU injection rates. Lock rate = fraction of inferences "
                "where tier was not downgraded to NANO.",
        label="tab:exp4_safety",
        highlight_best="mean",
    )
    tex_path = RESULTS / "exp4_latex.tex"
    tex_path.write_text(latex)
    print(f"  LaTeX table → {tex_path}\n")

    return all_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",           type=int, default=50)
    parser.add_argument("--simulate",    action="store_true", default=True)
    parser.add_argument("--no-simulate", dest="simulate", action="store_false")
    args = parser.parse_args()
    run(n=args.n, simulate=args.simulate)

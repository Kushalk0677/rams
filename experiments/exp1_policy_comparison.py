"""
Experiment 1 — Policy Comparison
Compares ThresholdPolicy, PredictivePolicy, and SafetyPolicy across
five load profiles (idle → burst).

Outputs
-------
  results/exp1_policy_comparison.csv
  results/exp1_policy_comparison.json
  results/exp1_latency_by_policy.png
  results/exp1_latency_by_profile.png
  results/exp1_tier_dist_<policy>.png
  results/exp1_latex.tex
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.utils import (
    run_trial, compute_stats, print_table, to_latex,
    bar_chart, tier_pie, save_records_csv, save_stats_json,
    line_chart,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

POLICIES = ["threshold", "predictive", "safety"]

LOAD_PROFILES = {
    "idle":     0.00,
    "light":    0.25,
    "moderate": 0.50,
    "heavy":    0.75,
    "burst":    1.00,
}


def run(n: int = 60, simulate: bool = True):
    print("\n" + "═" * 60)
    print("  Experiment 1 — Policy Comparison")
    print(f"  N={n} per cell  |  simulate={simulate}")
    print("═" * 60)

    all_records = []
    all_stats   = []

    for policy in POLICIES:
        for profile, intensity in LOAD_PROFILES.items():
            print(f"\n  [{policy:>12}  ×  {profile:<8}]  R≈{intensity:.2f} ...", flush=True)
            recs = run_trial(
                label=policy,
                group=profile,
                policy=policy,
                n=n,
                intensity=intensity,
                simulate=simulate,
            )
            all_records.extend(recs)
            s = compute_stats(recs, label=policy, group=profile)
            all_stats.append(s)
            print(f"    mean={s.mean:.1f} ms  std={s.std:.1f}  p95={s.p95:.1f}  "
                  f"switches={s.switch_rate:.3f}")

    # ── Console table ────────────────────────────────────────────────────────
    print_table(
        all_stats,
        title="Experiment 1: Policy Comparison (mean ± std latency, ms)",
        columns=["label", "group", "n", "mean", "std", "p95", "tier_dist", "switch_rate"],
    )

    # ── Save raw data ────────────────────────────────────────────────────────
    save_records_csv(all_records, RESULTS / "exp1_policy_comparison.csv")
    save_stats_json(all_stats,   RESULTS / "exp1_policy_comparison.json")

    # ── Plots ────────────────────────────────────────────────────────────────
    # 1. Mean latency by policy (grouped by load profile)
    bar_chart(
        all_stats,
        x_attr="label",
        group_attr="group",
        title="Exp 1 — Mean Latency by Policy",
        xlabel="Policy",
        ylabel="Mean latency (ms)",
        out=RESULTS / "exp1_latency_by_policy.png",
    )

    # 2. Mean latency by load profile (grouped by policy)
    bar_chart(
        all_stats,
        x_attr="group",
        group_attr="label",
        title="Exp 1 — Mean Latency by Load Profile",
        xlabel="Load profile",
        ylabel="Mean latency (ms)",
        out=RESULTS / "exp1_latency_by_profile.png",
    )

    # 3. Tier distribution pie per policy (aggregated over all profiles)
    from collections import defaultdict
    for policy in POLICIES:
        agg: dict[str, float] = defaultdict(float)
        n_total = 0
        for s in all_stats:
            if s.label == policy:
                for t, frac in s.tier_dist.items():
                    agg[t] += frac * s.n
                n_total += s.n
        if n_total > 0:
            dist = {t: v / n_total for t, v in agg.items()}
            tier_pie(
                dist,
                title=f"{policy} — tier distribution",
                out=RESULTS / f"exp1_tier_dist_{policy}.png",
            )

    # 4. Latency trace across one profile for all policies
    from collections import defaultdict
    profile_traces: dict[str, list[float]] = {}
    # use "heavy" as representative
    for policy in POLICIES:
        trace = [r.latency_ms for r in all_records
                 if r.label == policy and r.group == "heavy"]
        if trace:
            profile_traces[policy] = trace
    if profile_traces:
        line_chart(
            profile_traces,
            title="Exp 1 — Latency Trace under Heavy Load",
            xlabel="Inference index",
            ylabel="Latency (ms)",
            out=RESULTS / "exp1_latency_trace_heavy.png",
        )

    # ── LaTeX ────────────────────────────────────────────────────────────────
    latex = to_latex(
        all_stats,
        caption="Policy comparison: mean latency (ms) ± std across load profiles. "
                "Bold = best mean per row. N=" + str(n) + " inferences per cell.",
        label="tab:exp1_policy",
        highlight_best="mean",
    )
    tex_path = RESULTS / "exp1_latex.tex"
    tex_path.write_text(latex)
    print(f"  LaTeX table → {tex_path}\n")

    return all_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",           type=int,  default=60)
    parser.add_argument("--simulate",    action="store_true", default=True)
    parser.add_argument("--no-simulate", dest="simulate", action="store_false")
    args = parser.parse_args()
    run(n=args.n, simulate=args.simulate)

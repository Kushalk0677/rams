"""
Experiment 2 — Load Profile Sweep
Runs a single policy (default: safety) across a fine-grained
intensity sweep [0.0 → 1.0] and captures how tier distribution,
latency, and switching frequency respond to rising pressure.

Outputs
-------
  results/exp2_load_sweep.csv
  results/exp2_load_sweep.json
  results/exp2_latency_vs_load.png
  results/exp2_tier_dist_vs_load.png
  results/exp2_switches_vs_load.png
  results/exp2_latex.tex
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.utils import (
    run_trial, compute_stats, print_table, to_latex,
    bar_chart, line_chart, save_records_csv, save_stats_json,
    _get_mpl,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

# Fine-grained intensity steps
INTENSITIES = [round(i * 0.1, 1) for i in range(11)]  # 0.0, 0.1, … 1.0


def run(n: int = 50, policy: str = "safety", simulate: bool = True):
    print("\n" + "═" * 60)
    print("  Experiment 2 — Load Profile Sweep")
    print(f"  Policy={policy}  N={n}  simulate={simulate}")
    print("═" * 60)

    all_records = []
    all_stats   = []

    for intensity in INTENSITIES:
        label = f"{intensity:.1f}"
        print(f"  intensity={label} ...", flush=True)
        recs = run_trial(
            label=label,
            group=policy,
            policy=policy,
            n=n,
            intensity=intensity,
            simulate=simulate,
        )
        all_records.extend(recs)
        s = compute_stats(recs, label=label, group=policy)
        all_stats.append(s)
        tier_str = "  ".join(f"{k}:{v:.0%}" for k, v in sorted(s.tier_dist.items()))
        print(f"    mean={s.mean:.1f} ms  R̄={s.pressure_mean:.3f}  tiers={tier_str}")

    # ── Console table ────────────────────────────────────────────────────────
    print_table(
        all_stats,
        title=f"Experiment 2: Load Sweep — policy={policy}",
        columns=["label", "n", "pressure_mean", "mean", "std", "p95",
                 "tier_dist", "switch_rate"],
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    save_records_csv(all_records, RESULTS / "exp2_load_sweep.csv")
    save_stats_json(all_stats,   RESULTS / "exp2_load_sweep.json")

    # ── Plots ────────────────────────────────────────────────────────────────
    plt = _get_mpl()

    intensities = [float(s.label) for s in all_stats]
    means  = [s.mean  for s in all_stats]
    stds   = [s.std   for s in all_stats]
    p95s   = [s.p95   for s in all_stats]
    sw     = [s.switch_rate for s in all_stats]
    press  = [s.pressure_mean for s in all_stats]

    # Plot 1: latency vs load intensity
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(intensities, means, "o-", color="#2980b9", label="Mean latency", linewidth=2)
    ax.fill_between(intensities,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.2, color="#2980b9", label="±1 std")
    ax.plot(intensities, p95s, "s--", color="#e74c3c", label="P95 latency", linewidth=1.5)
    ax.set_xlabel("Load intensity", fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title(f"Exp 2 — Latency vs Load Intensity  [{policy}]",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "exp2_latency_vs_load.png", dpi=150)
    plt.close(fig)
    print(f"  Chart saved → results/exp2_latency_vs_load.png")

    # Plot 2: tier distribution stacked bar vs load
    TIER_ORDER  = ["NANO", "SMALL", "MEDIUM"]
    TIER_COLORS = {"NANO": "#e74c3c", "SMALL": "#f39c12", "MEDIUM": "#2ecc71"}

    fig, ax = plt.subplots(figsize=(9, 4))
    bottoms = [0.0] * len(all_stats)
    for tier in TIER_ORDER:
        fracs = [s.tier_dist.get(tier, 0.0) for s in all_stats]
        ax.bar(intensities, fracs, bottom=bottoms, width=0.07,
               label=tier, color=TIER_COLORS[tier], alpha=0.9)
        bottoms = [b + f for b, f in zip(bottoms, fracs)]
    ax.set_xlabel("Load intensity", fontsize=11)
    ax.set_ylabel("Tier fraction", fontsize=11)
    ax.set_title(f"Exp 2 — Tier Distribution vs Load Intensity  [{policy}]",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "exp2_tier_dist_vs_load.png", dpi=150)
    plt.close(fig)
    print(f"  Chart saved → results/exp2_tier_dist_vs_load.png")

    # Plot 3: switch rate + pressure vs load (dual axis)
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()
    ax1.plot(intensities, sw,    "o-", color="#8e44ad", linewidth=2, label="Switch rate")
    ax2.plot(intensities, press, "s--", color="#95a5a6", linewidth=1.5, label="R̄(t)")
    ax1.set_xlabel("Load intensity", fontsize=11)
    ax1.set_ylabel("Switches / inference", fontsize=11, color="#8e44ad")
    ax2.set_ylabel("Mean pressure R(t)", fontsize=11, color="#95a5a6")
    ax1.set_title(f"Exp 2 — Switch Rate & Pressure vs Load  [{policy}]",
                  fontsize=12, fontweight="bold")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, fontsize=9)
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "exp2_switches_vs_load.png", dpi=150)
    plt.close(fig)
    print(f"  Chart saved → results/exp2_switches_vs_load.png")

    # ── LaTeX ────────────────────────────────────────────────────────────────
    latex = to_latex(
        all_stats,
        caption=f"Load sweep results for the {policy} policy. "
                f"N={n} per intensity level.",
        label="tab:exp2_sweep",
        highlight_best="mean",
    )
    tex_path = RESULTS / "exp2_latex.tex"
    tex_path.write_text(latex)
    print(f"  LaTeX table → {tex_path}\n")

    return all_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",           type=int,  default=50)
    parser.add_argument("--policy",      default="safety",
                        choices=["threshold", "predictive", "safety"])
    parser.add_argument("--simulate",    action="store_true", default=True)
    parser.add_argument("--no-simulate", dest="simulate", action="store_false")
    args = parser.parse_args()
    run(n=args.n, policy=args.policy, simulate=args.simulate)

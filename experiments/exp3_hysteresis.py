"""
Experiment 3 — Hysteresis Sensitivity
Sweeps the hysteresis_window parameter [1 … 10] for the threshold
policy under moderate and heavy load.  Shows the tradeoff between
responsiveness (low window) and stability (high window).

Outputs
-------
  results/exp3_hysteresis.csv
  results/exp3_hysteresis.json
  results/exp3_switches_vs_window.png
  results/exp3_latency_vs_window.png
  results/exp3_latex.tex
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.utils import (
    run_trial, compute_stats, print_table, to_latex,
    save_records_csv, save_stats_json, _get_mpl,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

WINDOWS    = [1, 2, 3, 4, 5, 7, 10]
TEST_LOADS = {"moderate": 0.50, "heavy": 0.75}


def run(n: int = 60, simulate: bool = True):
    print("\n" + "═" * 60)
    print("  Experiment 3 — Hysteresis Sensitivity")
    print(f"  N={n}  simulate={simulate}")
    print("═" * 60)

    all_records = []
    all_stats   = []

    for window in WINDOWS:
        for profile, intensity in TEST_LOADS.items():
            label = f"w={window}"
            print(f"  window={window}  profile={profile} ...", flush=True)
            recs = run_trial(
                label=label,
                group=profile,
                policy="threshold",
                n=n,
                intensity=intensity,
                simulate=simulate,
                policy_kwargs={"hysteresis_window": window},
            )
            all_records.extend(recs)
            s = compute_stats(recs, label=label, group=profile)
            all_stats.append(s)
            print(f"    switch_rate={s.switch_rate:.3f}  mean={s.mean:.1f} ms")

    # ── Console ──────────────────────────────────────────────────────────────
    print_table(
        all_stats,
        title="Experiment 3: Hysteresis Window Sensitivity",
        columns=["label", "group", "n", "mean", "std", "p95", "switch_rate", "tier_dist"],
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    save_records_csv(all_records, RESULTS / "exp3_hysteresis.csv")
    save_stats_json(all_stats,   RESULTS / "exp3_hysteresis.json")

    # ── Plots ────────────────────────────────────────────────────────────────
    plt = _get_mpl()

    for profile in TEST_LOADS:
        subset = [s for s in all_stats if s.group == profile]
        wins   = [int(s.label.split("=")[1]) for s in subset]
        sw     = [s.switch_rate for s in subset]
        means  = [s.mean        for s in subset]
        stds   = [s.std         for s in subset]

        # switches vs window
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(wins, sw, "o-", color="#8e44ad", linewidth=2, markersize=7)
        ax.axvline(x=3, color="#e74c3c", linestyle="--", alpha=0.6, label="default (w=3)")
        ax.set_xlabel("Hysteresis window (samples)", fontsize=11)
        ax.set_ylabel("Switches / inference", fontsize=11)
        ax.set_title(f"Exp 3 — Switching Rate vs Hysteresis Window  [{profile}]",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        fig.tight_layout()
        out = RESULTS / f"exp3_switches_vs_window_{profile}.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"  Chart saved → {out.name}")

        # latency vs window
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(wins, means, "o-", color="#2980b9", linewidth=2, markersize=7,
                label="Mean latency")
        ax.fill_between(wins,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.2, color="#2980b9")
        ax.axvline(x=3, color="#e74c3c", linestyle="--", alpha=0.6, label="default (w=3)")
        ax.set_xlabel("Hysteresis window (samples)", fontsize=11)
        ax.set_ylabel("Mean latency (ms)", fontsize=11)
        ax.set_title(f"Exp 3 — Latency vs Hysteresis Window  [{profile}]",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        fig.tight_layout()
        out = RESULTS / f"exp3_latency_vs_window_{profile}.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"  Chart saved → {out.name}")

    # ── LaTeX ────────────────────────────────────────────────────────────────
    latex = to_latex(
        all_stats,
        caption="Hysteresis window sensitivity for the threshold policy. "
                "Lower switch rate indicates more stable tier selection; "
                "higher switch rate indicates faster adaptation.",
        label="tab:exp3_hysteresis",
        highlight_best="switch_rate",
    )
    tex_path = RESULTS / "exp3_latex.tex"
    tex_path.write_text(latex)
    print(f"  LaTeX table → {tex_path}\n")

    return all_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",           type=int, default=60)
    parser.add_argument("--simulate",    action="store_true", default=True)
    parser.add_argument("--no-simulate", dest="simulate", action="store_false")
    args = parser.parse_args()
    run(n=args.n, simulate=args.simulate)

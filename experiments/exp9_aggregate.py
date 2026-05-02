"""
Experiment 9 — Multi-Device Aggregator
========================================
Reads all exp9_*.json files from results/multidevice/ and produces
cross-device comparison tables, plots, and a LaTeX table for the paper.

Usage
-----
  python experiments/exp9_aggregate.py
  python experiments/exp9_aggregate.py --results-dir results/multidevice/

Outputs
-------
  results/exp9_cross_device_latency.png
  results/exp9_cross_device_tiers.png
  results/exp9_latex.tex
  results/exp9_summary.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


# ---------------------------------------------------------------------------
# Load all device result files
# ---------------------------------------------------------------------------

def load_all_results(results_dir: Path, latest_only: bool = True) -> list[dict]:
    """Load all exp9_*.json files and flatten into a list of group records."""
    files = sorted(results_dir.glob("exp9_*.json"))
    if not files:
        logger.error("No exp9_*.json files found in %s", results_dir)
        sys.exit(1)

    if latest_only:
        latest_by_target = {}
        for f in files:
            with open(f) as fp:
                data = json.load(fp)
            key = (data.get("device", "unknown"), data.get("backend", "unknown"))
            ts = data.get("timestamp") or f.stat().st_mtime
            previous = latest_by_target.get(key)
            if previous is None or ts > previous[0]:
                latest_by_target[key] = (ts, f)
        files = sorted(v[1] for v in latest_by_target.values())

    all_groups = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        device  = data.get("device", "unknown")
        backend = data.get("backend", "unknown")
        logger.info("Loaded %s: device=%s backend=%s groups=%d",
                    f.name, device, backend, len(data.get("groups", [])))
        for g in data.get("groups", []):
            g["device"]  = device
            g["backend"] = backend
            all_groups.append(g)

    logger.info("Total groups loaded: %d", len(all_groups))
    return all_groups


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_cross_device_latency(groups: list[dict]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available")
        return

    # Focus on the moderate load profile for the main comparison
    TARGET_PROFILE = "moderate"
    devices = sorted(set(g["device"] for g in groups))
    policies = ["threshold", "predictive", "adaptive", "safety", "safety2"]
    policies = [p for p in policies if any(g["policy"] == p for g in groups)]

    filtered = [g for g in groups if g["load_profile"] == TARGET_PROFILE]

    # Build matrix: device × policy → mean latency
    data = defaultdict(dict)
    for g in filtered:
        data[g["device"]][g["policy"]] = g["latency_mean"]

    x = np.arange(len(policies))
    width = 0.8 / max(len(devices), 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    palette = ["#5b8dd9", "#e07b54", "#4caf7d", "#9b59b6", "#f39c12"]

    for i, device in enumerate(devices):
        vals = [data[device].get(p, 0) for p in policies]
        offset = (i - len(devices) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width * 0.9,
               label=device, color=palette[i % len(palette)],
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=15)
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title(f"Cross-device latency comparison — {TARGET_PROFILE} load")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()

    out = RESULTS_DIR / "exp9_cross_device_latency.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


def plot_cross_device_tier_dist(groups: list[dict]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    TARGET_PROFILE = "moderate"
    TARGET_POLICY  = "safety"

    filtered = [g for g in groups
                if g["load_profile"] == TARGET_PROFILE and g["policy"] == TARGET_POLICY]

    if not filtered:
        logger.warning("No data for policy=%s profile=%s", TARGET_POLICY, TARGET_PROFILE)
        return

    devices = [g["device"] for g in filtered]
    tiers = ["NANO", "SMALL", "MEDIUM"]
    colors = {"NANO": "#e07b54", "SMALL": "#5b8dd9", "MEDIUM": "#4caf7d"}

    x = np.arange(len(devices))
    bottoms = np.zeros(len(devices))
    fig, ax = plt.subplots(figsize=(9, 4))

    for tier in tiers:
        vals = []
        for g in filtered:
            total = g["n"]
            count = g["tier_counts"].get(tier, 0)
            vals.append(count / total * 100 if total > 0 else 0)
        ax.bar(x, vals, bottom=bottoms, label=tier,
               color=colors[tier], edgecolor="white", linewidth=0.5)
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(devices, rotation=15)
    ax.set_ylabel("Tier usage (%)")
    ax.set_title(f"Tier distribution — {TARGET_POLICY} policy, {TARGET_PROFILE} load")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 110)
    plt.tight_layout()

    out = RESULTS_DIR / "exp9_cross_device_tiers.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def write_latex(groups: list[dict]):
    """Write a cross-device latency table for the paper."""
    TARGET_PROFILES = ["idle", "moderate", "heavy", "burst"]
    devices  = sorted(set(g["device"]  for g in groups))
    policies = ["threshold", "predictive", "adaptive", "safety", "safety2"]
    policies = [p for p in policies if any(g["policy"] == p for g in groups)]

    # Build lookup
    lookup = {}
    for g in groups:
        lookup[(g["device"], g["policy"], g["load_profile"])] = g

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Mean inference latency (ms) across devices, policies, and load profiles. "
                 r"Tier distributions in parentheses: (N/S/M) = \% NANO/SMALL/MEDIUM.}")
    lines.append(r"\label{tab:multidevice}")

    col_spec = "ll" + "r" * len(TARGET_PROFILES)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header = "Device & Policy & " + " & ".join(p.capitalize() for p in TARGET_PROFILES) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for d_idx, device in enumerate(devices):
        if d_idx > 0:
            lines.append(r"\midrule")
        for p_idx, policy in enumerate(policies):
            device_col = r"\multirow{" + str(len(policies)) + r"}{*}{" + device + "}" if p_idx == 0 else ""
            cells = []
            for profile in TARGET_PROFILES:
                g = lookup.get((device, policy, profile))
                if g:
                    tc = g["tier_counts"]
                    total = g["n"]
                    n_pct = round(tc.get("NANO",   0) / total * 100)
                    s_pct = round(tc.get("SMALL",  0) / total * 100)
                    m_pct = round(tc.get("MEDIUM", 0) / total * 100)
                    cells.append(f"{g['latency_mean']:.1f} ({n_pct}/{s_pct}/{m_pct})")
                else:
                    cells.append("—")
            lines.append(f"{device_col} & {policy} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out = RESULTS_DIR / "exp9_latex.tex"
    with open(out, "w") as f:
        f.write("\n".join(lines))
    logger.info("LaTeX table -> %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp 9 — Multi-Device Aggregator")
    parser.add_argument("--results-dir", default=None,
                        help="Directory containing exp9_*.json files (default: results/multidevice/)")
    parser.add_argument("--all-runs", action="store_true",
                        help="Aggregate every exp9 JSON instead of only the latest run per device/backend")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR / "multidevice"
    groups = load_all_results(results_dir, latest_only=not args.all_runs)

    # Save combined summary
    out_json = RESULTS_DIR / "exp9_summary.json"
    with open(out_json, "w") as f:
        json.dump(groups, f, indent=2)
    logger.info("Combined summary -> %s", out_json)

    plot_cross_device_latency(groups)
    plot_cross_device_tier_dist(groups)
    write_latex(groups)

    # Print console summary — moderate load, all policies × devices
    print("\n" + "=" * 80)
    print("Cross-device summary (moderate load)")
    print(f"{'Device':<16} {'Policy':<14} {'Mean ms':>8} {'P95 ms':>8} {'Switches':>9} {'VRU%':>6}")
    print("-" * 80)
    for g in sorted(groups, key=lambda x: (x["device"], x["policy"])):
        if g["load_profile"] != "moderate":
            continue
        print(f"{g['device']:<16} {g['policy']:<14} "
              f"{g['latency_mean']:>8.1f} {g['latency_p95']:>8.1f} "
              f"{g['tier_switches']:>9} {g['vru_rate']*100:>5.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()

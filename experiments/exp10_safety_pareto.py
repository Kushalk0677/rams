"""
Experiment 10 — Safety-Latency Pareto Frontier (Real VRU Recall)
=================================================================
Plots each policy as an operating point in (mean latency, VRU recall)
space, using ground-truth VRU recall from exp8 rather than the simulated
vru_rate from the benchmark.

This is the key ESL figure: it shows that the safety policy lies on the
Pareto frontier of the latency-recall tradeoff, and that adaptive/threshold
policies trade recall for speed.

Also sweeps the proximity_window_s parameter for the safety policy to
produce a recall-latency tradeoff curve (the safety policy Pareto arc).

Usage
-----
  python experiments/exp10_safety_pareto.py \\
      --exp8-json results/exp8_accuracy_kitti.json \\
      --benchmark-json results/rams_<timestamp>_kitti_calibrated.json \\
      --dataset kitti

  # With proximity window sweep
  python experiments/exp10_safety_pareto.py \\
      --exp8-json results/exp8_accuracy_kitti.json \\
      --benchmark-json results/rams_<timestamp>_kitti_calibrated.json \\
      --dataset kitti \\
      --sweep-proximity \\
      --frames /data/kitti/images/val \\
      --n 200

Outputs
-------
  results/exp10_pareto_<dataset>.png
  results/exp10_proximity_sweep_<dataset>.png   (if --sweep-proximity)
  results/exp10_latex.tex
"""

import argparse
import json
import logging
import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)

VRU_CLASSES = {"person", "pedestrian", "cyclist", "bicycle", "motorbike", "motorcycle", "rider"}

LOAD_PROFILES = {
    "idle": 0.00, "light": 0.25, "moderate": 0.50, "heavy": 0.75, "burst": 1.00,
}

# Fixed-tier recall (from exp8 results — filled in at plot time from exp8 JSON)
TIER_LATENCY_SIM = {"NANO": 18.0, "SMALL": 32.0, "MEDIUM": 58.0}


# ---------------------------------------------------------------------------
# Load experiment data
# ---------------------------------------------------------------------------

def load_exp8(exp8_json: str) -> dict[str, float]:
    """Return {tier: vru_recall} from exp8 output."""
    with open(exp8_json) as f:
        data = json.load(f)
    return {t["tier"]: t["vru_recall"] for t in data["tiers"]}


def load_benchmark(benchmark_json: str) -> list[dict]:
    """Load benchmark JSON (existing format) groups."""
    with open(benchmark_json) as f:
        data = json.load(f)
    return data.get("groups", [])


def policy_latency(groups: list[dict], policy: str, profile: str = "moderate") -> float | None:
    for g in groups:
        if g["policy"] == policy and g["load_profile"] == profile:
            return g["latency_mean"]
    return None


def policy_tier_recall(groups: list[dict], tier_recall: dict[str, float],
                       policy: str, profile: str = "moderate") -> float | None:
    """
    Compute weighted VRU recall for a policy based on its tier distribution.
    weighted_recall = Σ (tier_fraction * tier_recall)
    """
    for g in groups:
        if g["policy"] == policy and g["load_profile"] == profile:
            total = g["n"]
            weighted = sum(
                (count / total) * tier_recall.get(tier, 0.0)
                for tier, count in g["tier_counts"].items()
            )
            return weighted
    return None


# ---------------------------------------------------------------------------
# Proximity window sweep
# ---------------------------------------------------------------------------

class LoadInjector:
    def __init__(self, intensity: float = 0.0):
        self.intensity = intensity
        self._stop = threading.Event()
        self._threads = []

    def _burn(self):
        while not self._stop.is_set():
            _ = sum(i * i for i in range(1000))
            time.sleep(max(0.0, (1.0 - self.intensity) * 0.001))

    def start(self):
        n = max(0, int(self.intensity * 4))
        for _ in range(n):
            t = threading.Thread(target=self._burn, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.0)


def sweep_proximity_window(frames_dir: str, n: int,
                           windows: list[float], tier_recall: dict[str, float]) -> list[dict]:
    """
    For each proximity_window_s value, run the safety policy under moderate load
    and compute (mean_latency, weighted_vru_recall).
    """
    from rams.controller import RAMSController
    import cv2
    import statistics

    frame_paths = sorted(
        list(Path(frames_dir).glob("*.jpg")) +
        list(Path(frames_dir).glob("*.png"))
    )
    if not frame_paths:
        logger.error("No frames found in %s", frames_dir)
        return []

    results = []
    intensity = LOAD_PROFILES["moderate"]

    for window in windows:
        logger.info("Proximity window sweep: window=%.2fs", window)
        injector = LoadInjector(intensity=intensity)
        injector.start()
        latencies = []
        tier_list = []
        frame_idx = 0

        try:
            # Patch the safety policy proximity window
            import rams.policy as policy_module
            orig_init = policy_module.SafetyPolicy.__init__

            def patched_init(self, *args, **kwargs):
                orig_init(self, *args, **kwargs)
                self.proximity_window_s = window

            policy_module.SafetyPolicy.__init__ = patched_init

            with RAMSController(simulate=False, policy="safety") as ctrl:
                time.sleep(0.5)
                for i in range(n):
                    path = frame_paths[frame_idx % len(frame_paths)]
                    frame_idx += 1
                    frame = cv2.imread(str(path))
                    result = ctrl.infer(frame=frame)
                    latencies.append(result["latency_ms"])
                    tier_list.append(result["tier"])

            policy_module.SafetyPolicy.__init__ = orig_init
        finally:
            injector.stop()

        total = len(tier_list)
        tier_counts = {t: tier_list.count(t) for t in set(tier_list)}
        weighted_recall = sum(
            (count / total) * tier_recall.get(tier, 0.0)
            for tier, count in tier_counts.items()
        )

        results.append({
            "proximity_window_s": window,
            "latency_mean":       round(statistics.mean(latencies), 2),
            "latency_p95":        round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
            "weighted_vru_recall": round(weighted_recall, 4),
            "tier_counts":        tier_counts,
        })
        logger.info("  window=%.2f  latency=%.1f ms  recall=%.3f",
                    window, results[-1]["latency_mean"], weighted_recall)

    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_pareto(policy_points: list[dict], fixed_tier_points: list[dict], dataset: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Fixed tier baselines
    tier_colors = {"NANO": "#e07b54", "SMALL": "#5b8dd9", "MEDIUM": "#4caf7d"}
    for pt in fixed_tier_points:
        ax.scatter(pt["latency_mean"], pt["vru_recall"],
                   color=tier_colors.get(pt["tier"], "gray"),
                   marker="s", s=120, zorder=5,
                   label=f"Fixed-{pt['tier']}")
        ax.annotate(pt["tier"], (pt["latency_mean"], pt["vru_recall"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)

    # Policy operating points
    policy_colors = {
        "threshold":  "#333333",
        "predictive": "#9b59b6",
        "adaptive":   "#1abc9c",
        "safety":     "#e74c3c",
        "safety2":    "#e67e22",
    }
    for pt in policy_points:
        ax.scatter(pt["latency_mean"], pt["vru_recall"],
                   color=policy_colors.get(pt["policy"], "gray"),
                   marker="o", s=140, zorder=6,
                   label=pt["policy"])
        ax.annotate(pt["policy"], (pt["latency_mean"], pt["vru_recall"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)

    ax.set_xlabel("Mean latency (ms)")
    ax.set_ylabel("Weighted VRU recall")
    ax.set_title(f"Safety-latency Pareto frontier — {dataset.upper()} (moderate load)")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = RESULTS_DIR / f"exp10_pareto_{dataset}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


def plot_proximity_sweep(sweep_results: list[dict], dataset: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    windows  = [r["proximity_window_s"]  for r in sweep_results]
    latencies = [r["latency_mean"]        for r in sweep_results]
    recalls  = [r["weighted_vru_recall"] for r in sweep_results]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    color_lat = "#5b8dd9"
    color_rec = "#e07b54"

    ax1.plot(windows, latencies, "o-", color=color_lat, label="Mean latency (ms)")
    ax1.set_xlabel("Proximity window (s)")
    ax1.set_ylabel("Mean latency (ms)", color=color_lat)
    ax1.tick_params(axis="y", labelcolor=color_lat)

    ax2 = ax1.twinx()
    ax2.plot(windows, recalls, "s--", color=color_rec, label="Weighted VRU recall")
    ax2.set_ylabel("Weighted VRU recall", color=color_rec)
    ax2.tick_params(axis="y", labelcolor=color_rec)
    ax2.set_ylim(0, 1.05)

    ax1.set_title(f"Safety policy: proximity window sweep — {dataset.upper()}")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)

    plt.tight_layout()
    out = RESULTS_DIR / f"exp10_proximity_sweep_{dataset}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

def write_latex(policy_points: list[dict], fixed_tier_points: list[dict],
                dataset: str, sweep_results: list[dict] | None = None):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Safety-latency operating points under moderate load — "
                 + dataset.upper() + r". VRU recall is weighted by tier distribution.}")
    lines.append(r"\label{tab:pareto_" + dataset + "}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Policy / Tier & Mean latency (ms) & Weighted VRU recall & Tier dist. (N/S/M\%) \\")
    lines.append(r"\midrule")

    for pt in fixed_tier_points:
        lines.append(f"Fixed-{pt['tier']} & {pt['latency_mean']:.1f} & "
                     f"{pt['vru_recall']:.3f} & 100/0/0 \\\\")

    lines.append(r"\midrule")

    for pt in policy_points:
        tc = pt.get("tier_counts", {})
        total = sum(tc.values()) or 1
        n_pct = round(tc.get("NANO",   0) / total * 100)
        s_pct = round(tc.get("SMALL",  0) / total * 100)
        m_pct = round(tc.get("MEDIUM", 0) / total * 100)
        lines.append(f"{pt['policy']} & {pt['latency_mean']:.1f} & "
                     f"{pt['vru_recall']:.3f} & {n_pct}/{s_pct}/{m_pct} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = RESULTS_DIR / "exp10_latex.tex"
    with open(out, "w") as f:
        f.write("\n".join(lines))
    logger.info("LaTeX -> %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp 10 — Safety-Latency Pareto")
    parser.add_argument("--exp8-json",       required=True,
                        help="exp8_accuracy_<dataset>.json from Experiment 8")
    parser.add_argument("--benchmark-json",  required=True,
                        help="Benchmark JSON with per-policy latency and tier counts")
    parser.add_argument("--dataset",         required=True, choices=["kitti", "coco"])
    parser.add_argument("--profile",         default="moderate",
                        help="Load profile to use for Pareto comparison (default: moderate)")
    parser.add_argument("--sweep-proximity", action="store_true",
                        help="Also sweep proximity_window_s for the safety policy")
    parser.add_argument("--frames",          default=None,
                        help="Image directory (required for --sweep-proximity)")
    parser.add_argument("--n",               type=int, default=200)
    args = parser.parse_args()

    tier_recall = load_exp8(args.exp8_json)
    logger.info("Tier recall from exp8: %s", tier_recall)

    groups = load_benchmark(args.benchmark_json)

    policies = ["threshold", "predictive", "adaptive", "safety", "safety2"]
    policy_points = []
    for policy in policies:
        latency = policy_latency(groups, policy, args.profile)
        recall  = policy_tier_recall(groups, tier_recall, policy, args.profile)
        if latency is None or recall is None:
            logger.warning("No data for policy=%s profile=%s", policy, args.profile)
            continue
        g = next((g for g in groups
                  if g["policy"] == policy and g["load_profile"] == args.profile), {})
        policy_points.append({
            "policy":      policy,
            "latency_mean": latency,
            "vru_recall":  recall,
            "tier_counts": g.get("tier_counts", {}),
        })

    # Fixed-tier baselines — use exp8 recall directly, latency from tier profiles
    tier_latencies = {"NANO": 18.0, "SMALL": 32.0, "MEDIUM": 58.0}
    fixed_tier_points = [
        {"tier": t, "latency_mean": tier_latencies[t], "vru_recall": tier_recall.get(t, 0.0)}
        for t in ["NANO", "SMALL", "MEDIUM"] if t in tier_recall
    ]

    sweep_results = None
    if args.sweep_proximity:
        if not args.frames:
            logger.error("--frames required for --sweep-proximity")
            sys.exit(1)
        windows = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
        sweep_results = sweep_proximity_window(args.frames, args.n, windows, tier_recall)
        plot_proximity_sweep(sweep_results, args.dataset)

        import json as _json
        sweep_out = RESULTS_DIR / f"exp10_proximity_sweep_{args.dataset}.json"
        with open(sweep_out, "w") as f:
            _json.dump(sweep_results, f, indent=2)
        logger.info("Proximity sweep JSON -> %s", sweep_out)

    plot_pareto(policy_points, fixed_tier_points, args.dataset)
    write_latex(policy_points, fixed_tier_points, args.dataset, sweep_results)

    print("\n" + "=" * 65)
    print(f"Pareto operating points — {args.dataset.upper()} ({args.profile} load)")
    print(f"{'Policy':<14} {'Latency (ms)':>13} {'VRU Recall':>11}")
    print("-" * 65)
    for pt in sorted(fixed_tier_points, key=lambda x: x["latency_mean"]):
        print(f"Fixed-{pt['tier']:<8} {pt['latency_mean']:>13.1f} {pt['vru_recall']:>11.3f}")
    for pt in sorted(policy_points, key=lambda x: x["latency_mean"]):
        print(f"{pt['policy']:<14} {pt['latency_mean']:>13.1f} {pt['vru_recall']:>11.3f}")
    print("=" * 65)


if __name__ == "__main__":
    main()

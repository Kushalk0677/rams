"""
experiments/utils.py
Shared utilities: trial runner, statistics, console table,
LaTeX table, and matplotlib chart helpers.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── repo root on path ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rams.controller import RAMSController
from rams.policy import BasePolicy, make_policy


# ─────────────────────────────────────────────────────────────────────────────
# Load injector
# ─────────────────────────────────────────────────────────────────────────────

class LoadInjector:
    def __init__(self, intensity: float):
        self.intensity = max(0.0, min(1.0, intensity))
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def _burn(self):
        while not self._stop.is_set():
            _ = sum(i * i for i in range(2000))
            time.sleep(max(0.0, (1.0 - self.intensity) * 0.0005))

    def start(self):
        n = max(0, round(self.intensity * 4))
        for _ in range(n):
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
# Trial record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrialRecord:
    label:          str
    group:          str
    latency_ms:     float
    pressure:       float
    tier:           str
    n_detections:   int
    vru_detected:   bool  = False
    switch_occurred:bool  = False
    accuracy_proxy: float = 0.0   # per-inference mAP50 proxy


# ─────────────────────────────────────────────────────────────────────────────
# Core runner
# ─────────────────────────────────────────────────────────────────────────────

def run_trial(
    label:       str,
    group:       str,
    policy:      str | BasePolicy,
    n:           int,
    intensity:   float,
    simulate:    bool = True,
    policy_kwargs: dict | None = None,
    warmup:      int  = 5,
) -> list[TrialRecord]:
    """Run one experimental cell and return per-inference records."""
    records: list[TrialRecord] = []

    with LoadInjector(intensity) as _inj:
        with RAMSController(
            simulate=simulate,
            policy=policy if isinstance(policy, str) else policy,
            policy_kwargs=policy_kwargs or {},
        ) as ctrl:
            time.sleep(0.4)   # monitor warm-up

            prev_tier = ctrl.current_tier
            # warm-up passes (excluded from results)
            for _ in range(warmup):
                ctrl.infer()

            for _ in range(n):
                res = ctrl.infer()
                cur_tier = ctrl.current_tier

                vru = any(
                    d.get("class", "").lower() in
                    {"person", "pedestrian", "cyclist", "bicycle"}
                    for d in res.get("detections", [])
                )

                records.append(TrialRecord(
                    label=label,
                    group=group,
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
# Statistics helper
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Stats:
    label:        str
    group:        str
    n:            int
    mean:         float
    std:          float
    p50:          float
    p95:          float
    tier_dist:    dict[str, float]   # tier → fraction
    switch_rate:  float              # switches / inference
    vru_rate:     float              # VRU detections / inference
    pressure_mean:float

    def ci95(self) -> float:
        """95% CI half-width assuming normality."""
        if self.n < 2:
            return 0.0
        return 1.96 * self.std / math.sqrt(self.n)

    def overlaps(self, other: "Stats") -> bool:
        """Conservative overlap check via non-overlapping std ranges."""
        lo_a, hi_a = self.mean - self.std, self.mean + self.std
        lo_b, hi_b = other.mean - other.std, other.mean + other.std
        return not (hi_a < lo_b or hi_b < lo_a)


def compute_stats(records: list[TrialRecord], label: str, group: str) -> Stats:
    lats = [r.latency_ms for r in records]
    sl   = sorted(lats)
    tiers = [r.tier for r in records]
    unique = sorted(set(tiers))
    return Stats(
        label=label,
        group=group,
        n=len(records),
        mean=statistics.mean(lats),
        std=statistics.stdev(lats) if len(lats) > 1 else 0.0,
        p50=statistics.median(lats),
        p95=sl[max(0, int(len(sl) * 0.95) - 1)],
        tier_dist={t: tiers.count(t) / len(tiers) for t in unique},
        switch_rate=sum(r.switch_occurred for r in records) / len(records),
        vru_rate=sum(r.vru_detected for r in records) / len(records),
        pressure_mean=statistics.mean(r.pressure for r in records),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Console table
# ─────────────────────────────────────────────────────────────────────────────

def print_table(rows: list[Stats], title: str = "", columns: list[str] | None = None):
    cols = columns or ["label", "group", "n", "mean", "std", "p95", "tier_dist", "switch_rate"]

    col_map = {
        "label":         ("Label",        14, lambda s: s.label),
        "group":         ("Group",        14, lambda s: s.group),
        "n":             ("N",             5, lambda s: str(s.n)),
        "mean":          ("Mean ms",       8, lambda s: f"{s.mean:.1f}"),
        "std":           ("±Std",          7, lambda s: f"{s.std:.1f}"),
        "p50":           ("P50 ms",        7, lambda s: f"{s.p50:.1f}"),
        "p95":           ("P95 ms",        7, lambda s: f"{s.p95:.1f}"),
        "ci95":          ("CI95",          7, lambda s: f"±{s.ci95():.1f}"),
        "tier_dist":     ("Tier dist",    22, lambda s: "  ".join(
                              f"{k}:{v:.0%}" for k, v in sorted(s.tier_dist.items()))),
        "switch_rate":   ("Sw/inf",        7, lambda s: f"{s.switch_rate:.3f}"),
        "vru_rate":      ("VRU/inf",       7, lambda s: f"{s.vru_rate:.3f}"),
        "pressure_mean": ("R̄(t)",          6, lambda s: f"{s.pressure_mean:.3f}"),
    }

    headers   = [col_map[c][0] for c in cols]
    widths    = [col_map[c][1] for c in cols]
    formatters= [col_map[c][2] for c in cols]

    sep = "  ".join("─" * w for w in widths)
    hdr = "  ".join(f"{h:<{w}}" for h, w in zip(headers, widths))

    if title:
        print(f"\n{'═' * len(sep)}")
        print(f"  {title}")
    print(f"{'═' * len(sep)}")
    print(hdr)
    print(sep)
    for s in rows:
        vals = [f(s) for f in formatters]
        print("  ".join(f"{v:<{w}}" for v, w in zip(vals, widths)))
    print(f"{'═' * len(sep)}\n")


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table
# ─────────────────────────────────────────────────────────────────────────────

def to_latex(
    rows: list[Stats],
    caption: str = "",
    label: str = "tab:results",
    highlight_best: str = "mean",   # column to bold-minimise
) -> str:
    best_val = min(getattr(s, highlight_best) for s in rows)

    def bold_if_best(s: Stats) -> bool:
        return abs(getattr(s, highlight_best) - best_val) < 0.5

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{llrrrrrl}",
        r"\toprule",
        r"Label & Group & $N$ & Mean (ms) & $\pm$Std & P95 (ms) & Sw/inf & Tier dist \\",
        r"\midrule",
    ]

    for s in rows:
        mean_str = f"\\textbf{{{s.mean:.1f}}}" if bold_if_best(s) else f"{s.mean:.1f}"
        tier_str = ", ".join(f"{k}:{v:.0%}" for k, v in sorted(s.tier_dist.items()))
        lines.append(
            f"  {s.label} & {s.group} & {s.n} & {mean_str} & "
            f"{s.std:.1f} & {s.p95:.1f} & {s.switch_rate:.3f} & {tier_str} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


TIER_COLORS_PLT = {"NANO": "#e74c3c", "SMALL": "#f39c12", "MEDIUM": "#2ecc71"}
POLICY_MARKERS  = {"threshold": "o", "predictive": "s", "safety": "^"}


def bar_chart(
    rows: list[Stats],
    x_attr: str,
    y_attr: str = "mean",
    err_attr: str = "std",
    group_attr: str = "group",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Mean latency (ms)",
    out: Path | None = None,
):
    plt = _get_mpl()
    groups  = sorted({getattr(s, group_attr) for s in rows})
    labels  = sorted({getattr(s, x_attr)     for s in rows})
    x       = range(len(labels))
    n_groups= len(groups)
    width   = 0.8 / max(n_groups, 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = plt.cm.tab10.colors

    for i, grp in enumerate(groups):
        grp_rows = [next((s for s in rows if getattr(s, x_attr) == lbl
                          and getattr(s, group_attr) == grp), None)
                    for lbl in labels]
        ys   = [getattr(r, y_attr)   if r else 0 for r in grp_rows]
        errs = [getattr(r, err_attr) if r else 0 for r in grp_rows]
        offset = (i - n_groups / 2 + 0.5) * width
        ax.bar([xi + offset for xi in x], ys, width,
               label=grp, color=colors[i % 10], alpha=0.85,
               yerr=errs, capsize=3, error_kw={"linewidth": 1})

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel(xlabel or x_attr, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if out:
        fig.savefig(out, dpi=150)
        print(f"  Chart saved → {out}")
    plt.close(fig)


def line_chart(
    series: dict[str, list[float]],     # label → values
    title: str = "",
    xlabel: str = "Inference index",
    ylabel: str = "Latency (ms)",
    out: Path | None = None,
):
    plt = _get_mpl()
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.tab10.colors

    for i, (lbl, vals) in enumerate(series.items()):
        ax.plot(vals, label=lbl, color=colors[i % 10], linewidth=1.2, alpha=0.85)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if out:
        fig.savefig(out, dpi=150)
        print(f"  Chart saved → {out}")
    plt.close(fig)


def tier_pie(
    tier_dist: dict[str, float],
    title: str = "",
    out: Path | None = None,
):
    plt = _get_mpl()
    fig, ax = plt.subplots(figsize=(4, 4))
    labels  = list(tier_dist.keys())
    sizes   = [tier_dist[k] for k in labels]
    colors  = [TIER_COLORS_PLT.get(k, "#95a5a6") for k in labels]
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%",
           startangle=90, textprops={"fontsize": 11})
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
        print(f"  Chart saved → {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_records_csv(records: list[TrialRecord], path: Path):
    if not records:
        return
    fields = list(records[0].__dataclass_fields__.keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: getattr(r, k) for k in fields})


def save_stats_json(stats_list: list[Stats], path: Path):
    data = []
    for s in stats_list:
        d = {k: getattr(s, k) for k in s.__dataclass_fields__}
        data.append(d)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic pressure injection (for sandboxed environments)
# ─────────────────────────────────────────────────────────────────────────────

# Maps load intensity (0-1) to a realistic R(t) value based on the
# compute_pressure() formula with typical CPU/mem/temp readings.
# Calibrated so that idle→light stays below lo_thresh (0.45),
# moderate straddles lo_thresh, heavy pushes into SMALL range,
# burst crosses hi_thresh and forces NANO.
INTENSITY_TO_PRESSURE = {
    0.00: 0.08,   # idle:     CPU ~10%
    0.10: 0.14,
    0.20: 0.20,
    0.25: 0.28,   # light:    CPU ~35%
    0.30: 0.33,
    0.40: 0.39,
    0.50: 0.52,   # moderate: straddles lo_thresh 0.45 — SMALL territory
    0.60: 0.61,
    0.70: 0.67,
    0.75: 0.72,   # heavy:    right at hi_thresh — oscillates SMALL/NANO
    0.80: 0.77,
    0.90: 0.85,
    1.00: 0.93,   # burst:    CPU ~100% — NANO
}


def intensity_to_pressure(intensity: float, noise: float = 0.03) -> float:
    """
    Convert a LoadInjector intensity value to a synthetic R(t) reading.
    Adds small Gaussian noise to simulate monitor jitter.
    """
    import random, math
    keys = sorted(INTENSITY_TO_PRESSURE.keys())
    # Linear interpolation
    if intensity <= keys[0]:
        base = INTENSITY_TO_PRESSURE[keys[0]]
    elif intensity >= keys[-1]:
        base = INTENSITY_TO_PRESSURE[keys[-1]]
    else:
        for i in range(len(keys) - 1):
            if keys[i] <= intensity <= keys[i + 1]:
                lo, hi = keys[i], keys[i + 1]
                t = (intensity - lo) / (hi - lo)
                base = INTENSITY_TO_PRESSURE[lo] * (1 - t) + INTENSITY_TO_PRESSURE[hi] * t
                break
    return max(0.0, min(1.0, base + random.gauss(0, noise)))


def run_trial(
    label:       str,
    group:       str,
    policy:      str | BasePolicy,
    n:           int,
    intensity:   float,
    simulate:    bool = True,
    policy_kwargs: dict | None = None,
    warmup:      int  = 5,
    inject_pressure: bool = True,   # use synthetic R(t) override
) -> list[TrialRecord]:
    """Run one experimental cell and return per-inference records."""
    records: list[TrialRecord] = []

    with LoadInjector(intensity) as _inj:
        with RAMSController(
            simulate=simulate,
            policy=policy if isinstance(policy, str) else policy,
            policy_kwargs=policy_kwargs or {},
        ) as ctrl:
            time.sleep(0.4)   # monitor warm-up

            prev_tier = ctrl.current_tier
            for _ in range(warmup):
                if inject_pressure:
                    ctrl.set_pressure_override(intensity_to_pressure(intensity))
                ctrl.infer()

            for _ in range(n):
                if inject_pressure:
                    ctrl.set_pressure_override(intensity_to_pressure(intensity))
                res = ctrl.infer()
                cur_tier = ctrl.current_tier

                vru = any(
                    d.get("class", "").lower() in
                    {"person", "pedestrian", "cyclist", "bicycle"}
                    for d in res.get("detections", [])
                )

                records.append(TrialRecord(
                    label=label,
                    group=group,
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

"""
Experiment 7 — Safety-Weighted Accuracy Score (SWAS)
=====================================================
Introduces and evaluates the *Safety-Weighted Accuracy Score* (SWAS),
a novel composite metric that jointly evaluates accuracy and safety
behaviour in a single scalar.

Definition
----------
For a sequence of N inferences:

    SWAS = (1/N) * Σ_i [ acc_i * (1 + β * vru_i) ] / (1 + β)

where
    acc_i   = per-inference accuracy proxy (mAP50 of selected tier)
    vru_i   = 1 if a VRU is detected in inference i, else 0
    β       = safety weight factor (default 2.0)

Interpretation
    - β=0  reduces SWAS to plain mean accuracy (no safety weighting).
    - β>0  up-weights inferences where VRU is present: a policy that
           selects high-accuracy tiers when VRU is present earns a
           higher SWAS than one that opportunistically downgrades.
    - SWAS ∈ (0, 1] for normalised mAP proxies.

Why it matters
    SWAS bridges the "accuracy vs latency" Pareto plot (Exp 5) with a
    safety-aware objective. A policy that achieves high accuracy *only*
    when VRU is present outscores one that uniformly uses a mid tier.
    This is the core quantitative claim of the paper.

Sub-experiments
---------------
  7a. SWAS vs β sensitivity sweep — shows how the metric rewards safety
      policies more as β increases.
  7b. SWAS vs VRU injection rate — shows robustness of the score across
      scene compositions.
  7c. SWAS per load profile per policy (the main comparison table).
  7d. Fixed-tier baseline comparison — RAMS policies vs FIXED_NANO /
      FIXED_SMALL / FIXED_MEDIUM under SWAS; demonstrates RAMS dominance.
  7e. SWAS efficiency decomposition — separates VRU-present vs VRU-absent
      accuracy contributions to show the safety override earns its SWAS
      gain rather than just raising overall accuracy uniformly.
  7f. SWAS vs mean latency scatter — joint safety-efficiency frontier;
      shows SafetyPolicy / safety2 reach higher SWAS without proportional
      latency cost compared to fixed-tier alternatives.

Outputs
-------
  results/exp7_swas_main.png
  results/exp7_swas_beta_sweep.png
  results/exp7_swas_vru_rate.png
  results/exp7_swas_baselines.png        [7d]
  results/exp7_swas_efficiency.png       [7e]
  results/exp7_swas_latency_scatter.png  [7f]
  results/exp7_swas.csv / .json
  results/exp7_latex.tex
  results/exp7_baselines_latex.tex       [7d]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.utils import (
    LoadInjector, TrialRecord, save_records_csv, _get_mpl,
    intensity_to_pressure,
)
from rams.controller import RAMSController
from rams.models import Tier
from rams.policy import ThresholdPolicy

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

ALL_POLICIES    = ["threshold", "predictive", "safety", "adaptive", "safety2"]
FIXED_BASELINES = ["FIXED_NANO", "FIXED_SMALL", "FIXED_MEDIUM"]

LOAD_SCENARIOS = {
    "idle":     0.00,
    "moderate": 0.50,
    "heavy":    0.75,
    "burst":    1.00,
}

BETA_DEFAULT = 2.0

POLICY_COLORS = {
    "threshold":    "#3266ad",
    "predictive":   "#1D9E75",
    "safety":       "#D85A30",
    "adaptive":     "#8e44ad",
    "safety2":      "#c0392b",
    "FIXED_NANO":   "#aaaaaa",
    "FIXED_SMALL":  "#777777",
    "FIXED_MEDIUM": "#444444",
}

POLICY_HATCHES = {
    "FIXED_NANO":   "//",
    "FIXED_SMALL":  "\\\\",
    "FIXED_MEDIUM": "xx",
}


# ---------------------------------------------------------------------------
# SWAS computation
# ---------------------------------------------------------------------------

def compute_swas(records: list[TrialRecord], beta: float = BETA_DEFAULT) -> float:
    """
    Safety-Weighted Accuracy Score.

    SWAS = mean( acc_i * (1 + beta * vru_i) ) / (1 + beta)

    Normalised by (1+beta) so SWAS <= 1 always.
    """
    if not records:
        return 0.0
    total = sum(
        r.accuracy_proxy * (1.0 + beta * float(r.vru_detected))
        for r in records
    )
    return total / (len(records) * (1.0 + beta))


def compute_swas_decomposed(records: list[TrialRecord],
                             beta: float = BETA_DEFAULT) -> dict:
    """
    Decomposes SWAS into VRU-present and VRU-absent contributions.

    Keys returned:
        swas        — overall SWAS
        swas_vru    — mean accuracy on VRU-present inferences
        swas_novru  — mean accuracy on VRU-absent inferences
        vru_rate    — fraction of inferences with VRU
        delta_vru   — acc(VRU) - acc(no-VRU): safety gain per event
        base_acc    — unweighted mean accuracy
        efficiency  — fraction of SWAS attributable to VRU bonus weight
                      = (SWAS - baseline_swas) / (SWAS + eps)
                      where baseline_swas uses base_acc uniformly
    """
    if not records:
        return {}

    vru_recs   = [r for r in records if r.vru_detected]
    novru_recs = [r for r in records if not r.vru_detected]

    acc_vru   = statistics.mean(r.accuracy_proxy for r in vru_recs)   if vru_recs   else 0.0
    acc_novru = statistics.mean(r.accuracy_proxy for r in novru_recs) if novru_recs else 0.0
    base_acc  = statistics.mean(r.accuracy_proxy for r in records)

    swas     = compute_swas(records, beta)
    vru_rate = len(vru_recs) / len(records)

    # Baseline SWAS: what a policy with uniform accuracy = base_acc would score
    baseline_swas = base_acc * (1.0 + beta * vru_rate) / (1.0 + beta)
    efficiency = (swas - baseline_swas) / (swas + 1e-9)

    return {
        "swas":       swas,
        "swas_vru":   acc_vru,
        "swas_novru": acc_novru,
        "vru_rate":   vru_rate,
        "delta_vru":  acc_vru - acc_novru,
        "base_acc":   base_acc,
        "efficiency": efficiency,
    }


@dataclass
class SWASResult:
    policy:       str
    scenario:     str
    n:            int
    swas:         float
    mean_acc:     float
    vru_rate:     float
    mean_lat:     float
    tier_medium:  float
    tier_small:   float
    tier_nano:    float


# ---------------------------------------------------------------------------
# Trial runners
# ---------------------------------------------------------------------------

def run_swas_trial(
    policy_name: str,
    n: int,
    intensity: float,
    simulate: bool = True,
) -> list[TrialRecord]:
    records: list[TrialRecord] = []
    with LoadInjector(intensity):
        with RAMSController(simulate=simulate, policy=policy_name) as ctrl:
            time.sleep(0.4)
            prev_tier = ctrl.current_tier

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
                    group="swas",
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


def run_fixed_tier_trial(
    tier_name: str,
    n: int,
    intensity: float,
    simulate: bool = True,
) -> list[TrialRecord]:
    """Run N inferences locked to a single tier — fixed-model baseline."""
    tier_enum = {
        "FIXED_NANO":   Tier.NANO,
        "FIXED_SMALL":  Tier.SMALL,
        "FIXED_MEDIUM": Tier.MEDIUM,
    }[tier_name]

    records: list[TrialRecord] = []
    with LoadInjector(intensity):
        with RAMSController(simulate=simulate, policy="threshold") as ctrl:
            ctrl.policy = ThresholdPolicy(
                lo_thresh=0.0 if tier_enum == Tier.MEDIUM else (
                    1.0 if tier_enum == Tier.NANO else 0.5),
                hi_thresh=0.0 if tier_enum == Tier.NANO else (
                    1.0 if tier_enum == Tier.MEDIUM else 0.5),
                hysteresis_window=1,
            )
            ctrl._current_tier = tier_enum
            time.sleep(0.4)

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


# ---------------------------------------------------------------------------
# Plots — original (7a / 7b / 7c)
# ---------------------------------------------------------------------------

def swas_main_plot(results: list[SWASResult], out: Path):
    plt = _get_mpl()
    scenarios = list(LOAD_SCENARIOS.keys())
    policies  = ALL_POLICIES
    x         = list(range(len(scenarios)))
    n_pol     = len(policies)
    width     = 0.75 / n_pol

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, pol in enumerate(policies):
        ys = []
        for scen in scenarios:
            match = [r for r in results if r.policy == pol and r.scenario == scen]
            ys.append(match[0].swas if match else 0.0)
        offset = (i - n_pol / 2 + 0.5) * width
        ax.bar([xi + offset for xi in x], ys, width,
               label=pol, color=POLICY_COLORS.get(pol, "#888"), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.set_xlabel("Load scenario", fontsize=11)
    ax.set_ylabel(f"SWAS  (beta={BETA_DEFAULT})", fontsize=11)
    ax.set_title("Exp 7 — Safety-Weighted Accuracy Score by Policy and Load",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 0.65)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Chart saved -> {out.name}")


def beta_sweep_plot(records_by_policy: dict[str, list[TrialRecord]], out: Path):
    plt = _get_mpl()
    betas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    fig, ax = plt.subplots(figsize=(9, 5))

    for pol, recs in records_by_policy.items():
        ys = [compute_swas(recs, beta=b) for b in betas]
        ax.plot(betas, ys, marker="o", linewidth=2,
                color=POLICY_COLORS.get(pol, "#888"), label=pol)

    ax.set_xlabel("Safety weight beta", fontsize=11)
    ax.set_ylabel("SWAS", fontsize=11)
    ax.set_title("Exp 7b — SWAS vs Safety Weight beta  [moderate load]",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Chart saved -> {out.name}")


def swas_vs_vru_rate_plot(policy_names, n, intensity, simulate, out):
    plt = _get_mpl()
    vru_rates = []
    swas_by_pol: dict[str, list[float]] = {p: [] for p in policy_names}

    print("    Running VRU-rate sweep (5 sub-runs per policy) ...", flush=True)
    for _rep in range(5):
        run_vru_rates = []
        for pol in policy_names:
            recs = run_swas_trial(pol, n // 5, intensity, simulate)
            vr   = sum(r.vru_detected for r in recs) / len(recs)
            swas = compute_swas(recs)
            swas_by_pol[pol].append(swas)
            run_vru_rates.append(vr)
        vru_rates.append(statistics.mean(run_vru_rates))

    fig, ax = plt.subplots(figsize=(9, 5))
    for pol in policy_names:
        ax.plot(vru_rates, swas_by_pol[pol], marker="o", linewidth=1.5,
                color=POLICY_COLORS.get(pol, "#888"), label=pol, alpha=0.85)

    ax.set_xlabel("Observed VRU detection rate", fontsize=11)
    ax.set_ylabel(f"SWAS  (beta={BETA_DEFAULT})", fontsize=11)
    ax.set_title("Exp 7c — SWAS vs VRU Detection Rate  [moderate load]",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Chart saved -> {out.name}")


# ---------------------------------------------------------------------------
# Exp 7d — Fixed-tier SWAS baseline comparison
# ---------------------------------------------------------------------------

def run_exp7d(n: int, simulate: bool) -> dict[str, dict[str, float]]:
    """
    Computes SWAS for all RAMS policies AND fixed-tier baselines under
    moderate and heavy load.

    Returns {label: {scenario: swas}}.
    """
    print("\n  -- Exp 7d: Fixed-tier SWAS baseline comparison --", flush=True)
    scenarios = {"moderate": 0.50, "heavy": 0.75}
    swas_table: dict[str, dict[str, float]] = {}

    for label in ALL_POLICIES + FIXED_BASELINES:
        swas_table[label] = {}
        for scen, intensity in scenarios.items():
            print(f"    {label:14s}  {scen} ...", flush=True)
            if label in FIXED_BASELINES:
                recs = run_fixed_tier_trial(label, n, intensity, simulate)
            else:
                recs = run_swas_trial(label, n, intensity, simulate)
            swas_table[label][scen] = compute_swas(recs)
            print(f"      SWAS = {swas_table[label][scen]:.4f}")

    return swas_table


def baseline_comparison_plot(swas_table: dict[str, dict[str, float]], out: Path):
    """
    Side-by-side grouped bars: RAMS policies (solid) vs fixed baselines
    (hatched). Every RAMS adaptive policy should sit clearly above the
    fixed-tier alternatives — this is the "strong comparison" visual.
    """
    plt = _get_mpl()
    scenarios  = ["moderate", "heavy"]
    all_labels = ALL_POLICIES + FIXED_BASELINES
    n_lab  = len(all_labels)
    width  = 0.70 / n_lab
    x      = list(range(len(scenarios)))

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, label in enumerate(all_labels):
        ys     = [swas_table.get(label, {}).get(s, 0.0) for s in scenarios]
        offset = (i - n_lab / 2 + 0.5) * width
        hatch  = POLICY_HATCHES.get(label, None)
        ax.bar([xi + offset for xi in x], ys, width,
               label=label,
               color=POLICY_COLORS.get(label, "#888"),
               hatch=hatch,
               alpha=0.82,
               edgecolor="white" if hatch is None else "#333")

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in scenarios], fontsize=11)
    ax.set_xlabel("Load scenario", fontsize=11)
    ax.set_ylabel(f"SWAS  (beta={BETA_DEFAULT})", fontsize=11)
    ax.set_title(
        "Exp 7d — SWAS: RAMS Adaptive Policies vs Fixed-Tier Baselines\n"
        "(hatched bars = fixed single-tier; solid = adaptive RAMS)",
        fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Chart saved -> {out.name}")


def baseline_latex(swas_table: dict[str, dict[str, float]], n: int) -> str:
    """
    LaTeX table: rows = policies + baselines, cols = moderate / heavy.
    Bold = best SWAS per column. Baselines separated by midrule.
    """
    scenarios = ["moderate", "heavy"]
    best = {s: max(swas_table[l].get(s, 0.0) for l in swas_table)
            for s in scenarios}

    def cell(label, scen):
        v = swas_table.get(label, {}).get(scen)
        if v is None:
            return "--"
        s = f"{v:.3f}"
        return f"\\textbf{{{s}}}" if abs(v - best[scen]) < 0.0005 else s

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        (r"\caption{SWAS ($\beta=" + f"{BETA_DEFAULT:.1f}" + r"$) for RAMS adaptive"
         r" policies vs fixed-tier baselines under moderate and heavy load."
         f" $N={n}$. Bold = best per column."
         r" Fixed baselines lock to a single tier regardless of system pressure.}"),
        r"\label{tab:exp7d_baselines}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Method & Moderate & Heavy \\",
        r"\midrule",
    ]
    for label in ALL_POLICIES:
        lines.append(
            f"  {label} & {cell(label, 'moderate')} & {cell(label, 'heavy')} \\\\"
        )
    lines.append(r"\midrule")
    for label in FIXED_BASELINES:
        lines.append(
            f"  {label} & {cell(label, 'moderate')} & {cell(label, 'heavy')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Exp 7e — SWAS efficiency decomposition
# ---------------------------------------------------------------------------

def run_exp7e(n: int, simulate: bool,
              intensity: float = 0.75) -> dict[str, dict]:
    """
    For each RAMS policy under heavy load, decomposes SWAS into:
      - acc on VRU-present frames (should be HIGH for safety policies)
      - acc on VRU-absent frames  (can be lower — normal resource reduction)
      - delta = acc_vru - acc_novru  (the override's per-event lift)
      - efficiency ratio

    A non-adaptive policy has delta ~= 0 (same accuracy regardless of scene).
    SafetyPolicy / safety2 show positive delta: the VRU override holds the
    tier up specifically when it matters.
    """
    print("\n  -- Exp 7e: SWAS efficiency decomposition (heavy load) --",
          flush=True)
    results: dict[str, dict] = {}
    for pol in ALL_POLICIES:
        print(f"    {pol} ...", flush=True)
        recs = run_swas_trial(pol, n, intensity, simulate)
        results[pol] = compute_swas_decomposed(recs)
        d = results[pol]
        print(f"      SWAS={d['swas']:.4f}  delta_vru={d['delta_vru']:+.4f}"
              f"  efficiency={d['efficiency']:.4f}")
    return results


def efficiency_plot(decomp: dict[str, dict], out: Path):
    """
    Grouped bar chart (left axis): acc_vru (solid) and acc_novru (hatched).
    Line overlay (right axis): efficiency ratio.

    Visually proves that safety policies allocate higher accuracy *specifically*
    during VRU events rather than uniformly raising all inferences.
    """
    plt = _get_mpl()
    policies = ALL_POLICIES
    xs    = list(range(len(policies)))
    width = 0.30

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    acc_vru   = [decomp[p]["swas_vru"]   for p in policies]
    acc_novru = [decomp[p]["swas_novru"] for p in policies]
    effic     = [decomp[p]["efficiency"] for p in policies]

    ax1.bar([x - width / 2 for x in xs], acc_vru, width,
            label="Accuracy | VRU present",
            color=[POLICY_COLORS.get(p, "#888") for p in policies],
            alpha=0.85)
    ax1.bar([x + width / 2 for x in xs], acc_novru, width,
            label="Accuracy | no VRU",
            color=[POLICY_COLORS.get(p, "#888") for p in policies],
            alpha=0.40, hatch="//", edgecolor="#333")

    ax2.plot(xs, effic, marker="D", linewidth=2, color="#222",
             label="SWAS efficiency", zorder=5)
    ax2.axhline(0, color="#bbb", linewidth=0.8, linestyle="--")

    ax1.set_xticks(xs)
    ax1.set_xticklabels(policies, fontsize=10)
    ax1.set_xlabel("Policy", fontsize=11)
    ax1.set_ylabel("Mean accuracy proxy (mAP50)", fontsize=11)
    ax2.set_ylabel("SWAS efficiency ratio", fontsize=11)
    ax1.set_ylim(0.0, 0.75)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper right")
    ax1.set_title(
        "Exp 7e — SWAS Efficiency Decomposition  [heavy load]\n"
        "Solid = acc on VRU frames  |  Hatched = acc on non-VRU frames  |"
        "  Diamond = efficiency ratio",
        fontsize=11, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Chart saved -> {out.name}")


# ---------------------------------------------------------------------------
# Exp 7f — SWAS vs mean latency scatter (safety-efficiency frontier)
# ---------------------------------------------------------------------------

@dataclass
class FrontierPoint:
    label:    str
    kind:     str   # "adaptive" | "fixed"
    scenario: str
    swas:     float
    mean_lat: float
    vru_rate: float


def run_exp7f(n: int, simulate: bool) -> list[FrontierPoint]:
    """
    Collects (SWAS, mean_latency) for every policy + fixed baseline under
    moderate and heavy load, to plot the joint safety-efficiency frontier.
    Points in the top-left (high SWAS, low latency) are dominant.
    """
    print("\n  -- Exp 7f: SWAS vs latency scatter (frontier) --", flush=True)
    scenarios = {"moderate": 0.50, "heavy": 0.75}
    points: list[FrontierPoint] = []

    for label in ALL_POLICIES + FIXED_BASELINES:
        for scen, intensity in scenarios.items():
            print(f"    {label:14s}  {scen} ...", flush=True)
            if label in FIXED_BASELINES:
                recs = run_fixed_tier_trial(label, n, intensity, simulate)
            else:
                recs = run_swas_trial(label, n, intensity, simulate)

            swas  = compute_swas(recs)
            lat   = statistics.mean(r.latency_ms for r in recs)
            vru_r = sum(r.vru_detected for r in recs) / len(recs)
            kind  = "fixed" if label in FIXED_BASELINES else "adaptive"

            points.append(FrontierPoint(
                label=label, kind=kind, scenario=scen,
                swas=swas, mean_lat=lat, vru_rate=vru_r))
            print(f"      SWAS={swas:.4f}  lat={lat:.1f} ms")

    return points


def frontier_scatter_plot(points: list[FrontierPoint], out: Path):
    """
    Scatter: x = mean latency (ms), y = SWAS.
    Circle = moderate load, triangle = heavy load.
    Colour = policy (grey for fixed baselines).
    Upper-left is the Pareto-optimal region (high safety, low latency).
    """
    plt = _get_mpl()
    fig, ax = plt.subplots(figsize=(10, 6))

    MARKERS = {"moderate": "o", "heavy": "^"}

    # Fixed baselines — background layer, grey
    for pt in points:
        if pt.kind != "fixed":
            continue
        m = MARKERS.get(pt.scenario, "s")
        ax.scatter(pt.mean_lat, pt.swas, marker=m, s=90,
                   color=POLICY_COLORS.get(pt.label, "#aaa"),
                   edgecolors="#555", linewidths=1.2, alpha=0.55, zorder=2)
        ax.annotate(
            f"{pt.label}\n({pt.scenario})",
            (pt.mean_lat, pt.swas),
            textcoords="offset points", xytext=(6, -10),
            fontsize=7, color="#555",
            arrowprops=dict(arrowstyle="-", color="#999", lw=0.8),
        )

    # RAMS adaptive policies — foreground
    seen: set[str] = set()
    for pt in points:
        if pt.kind != "adaptive":
            continue
        m = MARKERS.get(pt.scenario, "s")
        lbl = pt.label if pt.label not in seen else None
        ax.scatter(pt.mean_lat, pt.swas, marker=m, s=130,
                   color=POLICY_COLORS.get(pt.label, "#888"),
                   edgecolors="white", linewidths=1.0,
                   label=lbl, zorder=4, alpha=0.92)
        seen.add(pt.label)
        ax.annotate(
            f"{pt.label}\n({pt.scenario})",
            (pt.mean_lat, pt.swas),
            textcoords="offset points", xytext=(5, 6),
            fontsize=7.5, color=POLICY_COLORS.get(pt.label, "#333"),
        )

    ax.text(0.03, 0.96,
            "< lower latency\n^ higher SWAS\n(top-left = Pareto optimal)",
            transform=ax.transAxes, fontsize=8, color="#555",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5",
                      edgecolor="#ccc", alpha=0.8))

    for scen, m in MARKERS.items():
        ax.scatter([], [], marker=m, color="grey", s=60,
                   label=f"load={scen}")

    ax.set_xlabel("Mean inference latency (ms)", fontsize=11)
    ax.set_ylabel(f"SWAS  (beta={BETA_DEFAULT})", fontsize=11)
    ax.set_title(
        "Exp 7f — Safety-Efficiency Frontier: SWAS vs Mean Latency\n"
        "(RAMS adaptive policies vs fixed-tier baselines; "
        "circle=moderate, triangle=heavy)",
        fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Chart saved -> {out.name}")


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def swas_to_latex(results: list[SWASResult], beta: float, n: int) -> str:
    scenarios = list(LOAD_SCENARIOS.keys())
    policies  = ALL_POLICIES

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        (r"\caption{Safety-Weighted Accuracy Score (SWAS, $\beta="
         + f"{beta:.1f}" + r"$) per policy and load profile. $N="
         + str(n) + r"$ inferences per cell. Bold = best SWAS per load column.}}"),
        r"\label{tab:exp7_swas}",
        r"\begin{tabular}{l" + "r" * len(scenarios) + "}",
        r"\toprule",
        "Policy & " + " & ".join(s.capitalize() for s in scenarios) + r" \\",
        r"\midrule",
    ]

    best: dict[str, float] = {
        scen: max((r.swas for r in results if r.scenario == scen), default=0.0)
        for scen in scenarios
    }

    for pol in policies:
        row_vals = []
        for scen in scenarios:
            match = [r for r in results if r.policy == pol and r.scenario == scen]
            if match:
                v = match[0].swas
                s = f"\\textbf{{{v:.3f}}}" if abs(v - best[scen]) < 0.001 else f"{v:.3f}"
            else:
                s = "--"
            row_vals.append(s)
        lines.append(f"  {pol} & " + " & ".join(row_vals) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(n: int = 100, simulate: bool = True):
    print("\n" + "=" * 62)
    print("  Experiment 7 — Safety-Weighted Accuracy Score (SWAS)")
    print(f"  N={n}  beta={BETA_DEFAULT}  simulate={simulate}")
    print("=" * 62)

    all_records: list[TrialRecord] = []
    results:     list[SWASResult]  = []
    moderate_recs: dict[str, list[TrialRecord]] = {}

    # -- 7a / 7b / 7c: all load scenarios, all RAMS policies ------------------
    for scenario, intensity in LOAD_SCENARIOS.items():
        print(f"\n  -- Load scenario: {scenario} (intensity={intensity}) --")

        for policy_name in ALL_POLICIES:
            print(f"    policy={policy_name} ...", flush=True)
            recs = run_swas_trial(policy_name, n, intensity, simulate)
            for r in recs:
                r.group = scenario
            all_records.extend(recs)

            swas     = compute_swas(recs, beta=BETA_DEFAULT)
            mean_acc = statistics.mean(r.accuracy_proxy for r in recs)
            vru_rate = sum(r.vru_detected for r in recs) / len(recs)
            mean_lat = statistics.mean(r.latency_ms for r in recs)
            tiers    = [r.tier for r in recs]
            n_tot    = len(tiers)

            results.append(SWASResult(
                policy=policy_name,
                scenario=scenario,
                n=n,
                swas=swas,
                mean_acc=mean_acc,
                vru_rate=vru_rate,
                mean_lat=mean_lat,
                tier_medium=tiers.count("MEDIUM") / n_tot,
                tier_small=tiers.count("SMALL")   / n_tot,
                tier_nano=tiers.count("NANO")      / n_tot,
            ))

            if scenario == "moderate":
                moderate_recs[policy_name] = recs

            print(f"      SWAS={swas:.4f}  acc={mean_acc:.3f}  "
                  f"vru_rate={vru_rate:.3f}  lat={mean_lat:.1f} ms")

    # Console summary
    print("\n" + "=" * 72)
    print(f"  {'Policy':<14} {'idle':>8} {'moderate':>10} {'heavy':>8} {'burst':>8}")
    print("-" * 72)
    for pol in ALL_POLICIES:
        row = f"  {pol:<14}"
        for scen in LOAD_SCENARIOS:
            match = [r for r in results if r.policy == pol and r.scenario == scen]
            row += f" {match[0].swas:>10.4f}" if match else f"{'--':>10}"
        print(row)
    print("=" * 72)

    # 7a: main bar chart
    swas_main_plot(results, RESULTS / "exp7_swas_main.png")

    # 7b: beta sweep
    beta_sweep_plot(moderate_recs, RESULTS / "exp7_swas_beta_sweep.png")

    # 7c: VRU rate sweep
    swas_vs_vru_rate_plot(
        ALL_POLICIES, n, LOAD_SCENARIOS["moderate"], simulate,
        RESULTS / "exp7_swas_vru_rate.png",
    )

    # -- 7d: Fixed-tier SWAS baseline comparison ------------------------------
    swas_baselines = run_exp7d(n, simulate)
    baseline_comparison_plot(swas_baselines, RESULTS / "exp7_swas_baselines.png")
    tex_7d = baseline_latex(swas_baselines, n)
    (RESULTS / "exp7_baselines_latex.tex").write_text(tex_7d)
    print("  LaTeX (7d) -> results/exp7_baselines_latex.tex")

    # -- 7e: SWAS efficiency decomposition ------------------------------------
    decomp = run_exp7e(n, simulate, intensity=0.75)
    efficiency_plot(decomp, RESULTS / "exp7_swas_efficiency.png")

    # -- 7f: SWAS vs latency scatter ------------------------------------------
    frontier_pts = run_exp7f(n, simulate)
    frontier_scatter_plot(frontier_pts, RESULTS / "exp7_swas_latency_scatter.png")

    # -- Persist --------------------------------------------------------------
    save_records_csv(all_records, RESULTS / "exp7_swas.csv")

    (RESULTS / "exp7_swas.json").write_text(
        json.dumps([dataclasses.asdict(r) for r in results], indent=2))
    print("  JSON -> results/exp7_swas.json")

    latex = swas_to_latex(results, beta=BETA_DEFAULT, n=n)
    (RESULTS / "exp7_latex.tex").write_text(latex)
    print("  LaTeX (7c) -> results/exp7_latex.tex\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",           type=int,  default=100)
    parser.add_argument("--simulate",    action="store_true", default=True)
    parser.add_argument("--no-simulate", dest="simulate", action="store_false")
    args = parser.parse_args()
    run(n=args.n, simulate=args.simulate)

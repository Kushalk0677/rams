#!/usr/bin/env python3
"""
Aggregate results from multiple laptops into a single summary.

Usage:
    # After copying all result CSVs into results/
    python scripts/aggregate.py

    # Or point to a specific directory
    python scripts/aggregate.py --dir /path/to/results
"""

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_csvs(results_dir: Path) -> list[dict]:
    all_records = []
    for csv_file in sorted(results_dir.glob("*.csv")):
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            for row in rows:
                row["_source_file"] = csv_file.name
                # coerce numeric fields
                for field in ("latency_ms", "pressure", "cpu_pct", "mem_pct", "load_intensity"):
                    try:
                        row[field] = float(row[field]) if row[field] not in ("", "None") else None
                    except (ValueError, KeyError):
                        pass
            all_records.extend(rows)
            print(f"  Loaded {len(rows):>5} records from {csv_file.name}")
    return all_records


def summarise(records: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in records:
        key = (r.get("policy"), r.get("load_profile"))
        groups[key].append(r)

    table = []
    for (policy, profile), recs in sorted(groups.items()):
        latencies = [r["latency_ms"] for r in recs if isinstance(r.get("latency_ms"), float)]
        tiers = [r.get("tier") for r in recs]
        tier_counts = {t: tiers.count(t) for t in sorted(set(tiers))}
        sources = sorted({r["_source_file"] for r in recs})

        if not latencies:
            continue

        sorted_lat = sorted(latencies)
        table.append({
            "policy":          policy,
            "load_profile":    profile,
            "n_records":       len(recs),
            "n_sources":       len(sources),
            "latency_mean_ms": round(statistics.mean(latencies), 2),
            "latency_std_ms":  round(statistics.stdev(latencies) if len(latencies) > 1 else 0.0, 2),
            "latency_p50_ms":  round(statistics.median(latencies), 2),
            "latency_p95_ms":  round(sorted_lat[int(len(sorted_lat) * 0.95)], 2),
            "tier_distribution": tier_counts,
            "sources":         sources,
        })

    return {"groups": table, "total_records": len(records)}


def print_table(summary: dict):
    print("\n" + "=" * 80)
    print(f"  Total records: {summary['total_records']}")
    print("=" * 80)
    fmt = "{:<14} {:<12} {:>7}  {:>9}  {:>9}  {:>9}  {}"
    print(fmt.format("Policy", "Profile", "N", "Mean ms", "P50 ms", "P95 ms", "Tiers"))
    print("-" * 80)
    for g in summary["groups"]:
        tier_str = "  ".join(f"{k}:{v}" for k, v in g["tier_distribution"].items())
        print(fmt.format(
            g["policy"], g["load_profile"], g["n_records"],
            g["latency_mean_ms"], g["latency_p50_ms"], g["latency_p95_ms"],
            tier_str,
        ))
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Aggregate RAMS benchmark results")
    parser.add_argument("--dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not args.dir.exists():
        print(f"Results directory not found: {args.dir}")
        sys.exit(1)

    print(f"\nLoading CSVs from: {args.dir}")
    records = load_csvs(args.dir)

    if not records:
        print("No CSV files found.")
        sys.exit(1)

    summary = summarise(records)
    print_table(summary)

    out = args.out or (args.dir / "aggregated_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAggregated summary saved → {out}\n")


if __name__ == "__main__":
    main()

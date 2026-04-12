#!/usr/bin/env python3
"""
RAMS Live Demo
Runs continuous adaptive inference and prints a live status table.

Usage:
    python scripts/live_demo.py --simulate --policy safety --fps 5
    python scripts/live_demo.py --no-simulate --policy threshold
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rams.controller import RAMSController

TIER_COLORS = {
    "NANO":   "\033[91m",   # red
    "SMALL":  "\033[93m",   # yellow
    "MEDIUM": "\033[92m",   # green
}
RESET = "\033[0m"
CLEAR_LINE = "\033[2K\r"

BAR_WIDTH = 30


def pressure_bar(p: float) -> str:
    filled = int(p * BAR_WIDTH)
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
    color = "\033[92m" if p < 0.45 else ("\033[93m" if p < 0.72 else "\033[91m")
    return f"{color}[{bar}]{RESET} {p:.3f}"


def main():
    parser = argparse.ArgumentParser(description="RAMS Live Demo")
    parser.add_argument("--policy",     default="safety", choices=["threshold", "predictive", "safety"])
    parser.add_argument("--fps",        type=float, default=4.0, help="Target inferences per second")
    parser.add_argument("--simulate",   action="store_true", default=True)
    parser.add_argument("--no-simulate", dest="simulate", action="store_false")
    parser.add_argument("--duration",   type=float, default=0, help="Run for N seconds then stop (0=forever)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    interval = 1.0 / args.fps
    t_start  = time.monotonic()
    n_frames = 0
    n_switches = 0
    last_tier = None

    print(f"\n  RAMS Live Demo  |  policy={args.policy}  |  simulate={args.simulate}")
    print("  Press Ctrl+C to stop.\n")

    try:
        with RAMSController(simulate=args.simulate, policy=args.policy) as ctrl:
            while True:
                t0 = time.monotonic()

                result  = ctrl.infer()
                tier    = result["tier"]
                lat     = result["latency_ms"]
                pressure = result.get("pressure", 0.0)
                cpu     = result.get("cpu_pct", 0.0)
                mem     = result.get("mem_pct", 0.0)
                temp    = result.get("cpu_temp")
                n_det   = len(result.get("detections", []))

                n_frames += 1
                if tier != last_tier and last_tier is not None:
                    n_switches += 1
                last_tier = tier

                elapsed = time.monotonic() - t_start
                fps_actual = n_frames / elapsed if elapsed > 0 else 0.0

                color = TIER_COLORS.get(tier, "")
                temp_str = f"{temp:.0f}°C" if temp else "N/A"

                line = (
                    f"  Tier: {color}{tier:<6}{RESET}"
                    f"  Latency: {lat:6.1f} ms"
                    f"  R(t): {pressure_bar(pressure)}"
                    f"  CPU:{cpu:4.0f}%  MEM:{mem:4.0f}%"
                    f"  Temp:{temp_str:>6}"
                    f"  Det:{n_det:3d}"
                    f"  Switches:{n_switches:4d}"
                    f"  FPS:{fps_actual:4.1f}"
                )
                print(f"{CLEAR_LINE}{line}", end="", flush=True)

                if args.duration > 0 and elapsed >= args.duration:
                    break

                sleep = max(0.0, interval - (time.monotonic() - t0))
                time.sleep(sleep)

    except KeyboardInterrupt:
        pass

    elapsed = time.monotonic() - t_start
    print(f"\n\n  Ran {n_frames} frames in {elapsed:.1f}s  |  {n_switches} tier switches\n")


if __name__ == "__main__":
    main()

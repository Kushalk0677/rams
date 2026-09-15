"""Prepare KITTI-raw temporal transition manifests for RAMS follow-up analysis.

This tool deliberately performs no detector inference.  It first verifies that
the KITTI raw image archive has its matching tracklet labels, then identifies
pedestrian/cyclist entry events with a VRU-free look-back window.  The emitted
manifest is the fixed input to a later controller replay; it is not paper
evidence by itself.

The analysis is named ``pre-entry elevation`` rather than anticipation.  RAMS
is reactive: a higher tier before a new ground-truth entry can only arise from
an earlier detection-conditioned lock, not from a prediction of the entry.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


VRU_TRACKLET_TYPES = {"pedestrian", "cyclist"}


@dataclass(frozen=True)
class Tracklet:
    object_type: str
    first_frame: int
    last_frame: int


@dataclass(frozen=True)
class SequenceAudit:
    sequence: str
    image_dir: str
    n_images: int
    n_timestamps: int
    fps: float | None
    tracklet_path: str | None
    n_vru_tracklets: int | None


def _timestamp_seconds(value: str) -> float | None:
    """Parse KITTI timestamps, accepting their nanosecond fractional field."""
    value = value.strip()
    if not value:
        return None
    if "." in value:
        prefix, fraction = value.split(".", 1)
        value = f"{prefix}.{fraction[:6].ljust(6, '0')}"
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f").timestamp()
    except ValueError:
        return None


def _fps(timestamp_path: Path) -> tuple[int, float | None]:
    if not timestamp_path.is_file():
        return 0, None
    values = [_timestamp_seconds(line) for line in timestamp_path.read_text(encoding="utf-8").splitlines()]
    parsed = [value for value in values if value is not None]
    deltas = [right - left for left, right in zip(parsed, parsed[1:]) if right > left]
    if not deltas:
        return len(values), None
    return len(values), round(1.0 / statistics.median(deltas), 6)


def _tracklet_index(root: Path) -> dict[str, Path]:
    """Map raw drive names to matching official ``tracklet_labels.xml`` files."""
    if not root.is_dir():
        return {}
    index: dict[str, Path] = {}
    for path in root.rglob("tracklet_labels.xml"):
        index.setdefault(path.parent.name, path)
    return index


def parse_tracklets(path: Path) -> list[Tracklet]:
    """Read the standard KITTI raw tracklet XML without requiring MATLAB tools."""
    root = ET.parse(path).getroot()
    container = root.find(".//tracklets")
    if container is None:
        raise ValueError(f"No <tracklets> element in {path}")
    parsed: list[Tracklet] = []
    for item in container.findall("./item"):
        object_type = (item.findtext("objectType") or "").strip().lower()
        first_text = item.findtext("first_frame")
        poses = item.findall("./poses/item")
        if not object_type or first_text is None or not poses:
            continue
        first = int(first_text)
        parsed.append(Tracklet(object_type, first, first + len(poses) - 1))
    return parsed


def audit_dataset(raw_root: Path, tracklet_root: Path, max_sequences: int | None = None) -> list[SequenceAudit]:
    labels = _tracklet_index(tracklet_root)
    audits: list[SequenceAudit] = []
    for date_dir in sorted(path for path in raw_root.iterdir() if path.is_dir() and path.name[:4].isdigit()):
        for drive_dir in sorted(path for path in date_dir.glob("*_sync") if path.is_dir()):
            image_dir = drive_dir / "image_02" / "data"
            images = sorted(image_dir.glob("*.png")) if image_dir.is_dir() else []
            timestamp_count, fps = _fps(drive_dir / "image_02" / "timestamps.txt")
            tracklet_path = labels.get(drive_dir.name)
            tracklets = parse_tracklets(tracklet_path) if tracklet_path else None
            audits.append(SequenceAudit(
                sequence=drive_dir.name,
                image_dir=str(image_dir),
                n_images=len(images),
                n_timestamps=timestamp_count,
                fps=fps,
                tracklet_path=str(tracklet_path) if tracklet_path else None,
                n_vru_tracklets=(sum(track.object_type in VRU_TRACKLET_TYPES for track in tracklets)
                                 if tracklets is not None else None),
            ))
            if max_sequences is not None and len(audits) >= max_sequences:
                return audits
    return audits


def _active_frames(tracklets: Iterable[Tracklet], n_images: int) -> set[int]:
    active: set[int] = set()
    for track in tracklets:
        if track.object_type not in VRU_TRACKLET_TYPES:
            continue
        active.update(range(max(0, track.first_frame), min(n_images - 1, track.last_frame) + 1))
    return active


def build_transition_manifest(audits: list[SequenceAudit], window_s: float, min_events: int) -> dict:
    """Build eligible entry and matched control windows from verified labels.

    Every entry must have an entire look-back window without *any* labelled VRU.
    Controls have no VRU in either their look-back or forthcoming window.  This
    makes the manifest suitable for testing pre-entry carry-over, not oracle
    triggering.
    """
    events: list[dict] = []
    controls: list[dict] = []
    for audit in audits:
        if not audit.tracklet_path:
            continue
        tracklets = parse_tracklets(Path(audit.tracklet_path))
        active = _active_frames(tracklets, audit.n_images)
        fps = audit.fps
        if fps is None or fps <= 0:
            continue
        # Timestamp parsing has microsecond rounding noise; map a 0.5-s
        # controller window to the nearest camera-frame count rather than
        # spuriously rounding a nominal 10-Hz stream from five to six frames.
        window_frames = max(1, int(round(window_s * fps)))
        entries = sorted(
            (track for track in tracklets
             if track.object_type in VRU_TRACKLET_TYPES and track.first_frame >= window_frames),
            key=lambda track: (track.first_frame, track.last_frame, track.object_type),
        )
        for track in entries:
            history = range(track.first_frame - window_frames, track.first_frame)
            if any(frame in active for frame in history):
                continue
            events.append({
                "sequence": audit.sequence,
                "entry_frame": track.first_frame,
                "track_last_frame": track.last_frame,
                "object_type": track.object_type,
                "window_frames": window_frames,
                "pre_frames": list(history),
            })

        # Deterministic, non-overlapping candidate controls from the same route.
        for frame in range(window_frames, max(window_frames, audit.n_images - window_frames), window_frames):
            lookback = range(frame - window_frames, frame)
            future = range(frame, frame + window_frames)
            if not any(index in active for index in (*lookback, *future)):
                controls.append({
                    "sequence": audit.sequence,
                    "control_frame": frame,
                    "window_frames": window_frames,
                    "pre_frames": list(lookback),
                })

    # A deterministic same-sequence control is selected later from this pool.
    return {
        "analysis": "KITTI raw pre-entry elevation",
        "interpretation": "reactive carry-over only; not VRU anticipation",
        "window_seconds": window_s,
        "vru_tracklet_types": sorted(VRU_TRACKLET_TYPES),
        "eligible_transition_events": events,
        "matched_control_candidates": controls,
        "n_transition_events": len(events),
        "minimum_events": min_events,
        "status": "ready" if len(events) >= min_events else "underpowered",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit/prepare KITTI raw temporal VRU transitions")
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--tracklet-root", type=Path,
                        help="Root containing matching drive/tracklet_labels.xml files; defaults to --raw-root")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepare", action="store_true", help="Build a transition manifest after a complete audit")
    parser.add_argument("--labelled-only", action="store_true",
                        help="Restrict preparation to drives with official KITTI raw tracklets. "
                             "Use this when a raw image collection contains dates for which KITTI released no labels.")
    parser.add_argument("--window-s", type=float, default=0.5)
    parser.add_argument("--min-events", type=int, default=30)
    parser.add_argument("--smoke", action="store_true", help="Audit only the first drive; never run inference")
    args = parser.parse_args()

    raw_root = args.raw_root.expanduser().resolve()
    tracklet_root = (args.tracklet_root or args.raw_root).expanduser().resolve()
    if not raw_root.is_dir():
        parser.error(f"--raw-root is not a directory: {raw_root}")
    audits = audit_dataset(raw_root, tracklet_root, max_sequences=1 if args.smoke else None)
    missing = [audit.sequence for audit in audits if audit.tracklet_path is None]
    eligible_audits = [audit for audit in audits if audit.tracklet_path]
    selected_audits = eligible_audits if args.labelled_only else audits
    selected_missing = [audit.sequence for audit in selected_audits if audit.tracklet_path is None]
    payload: dict = {
        "raw_root": str(raw_root),
        "tracklet_root": str(tracklet_root),
        "sequences": [asdict(audit) for audit in audits],
        "n_sequences": len(audits),
        "n_images": sum(audit.n_images for audit in audits),
        "n_missing_tracklet_labels": len(missing),
        "missing_tracklet_label_sequences": missing,
        "labelled_only": args.labelled_only,
        "n_selected_sequences": len(selected_audits),
        "selected_sequences": [audit.sequence for audit in selected_audits],
        "excluded_unlabelled_sequences": missing if args.labelled_only else [],
        "status": ("complete_labelled_subset" if args.labelled_only and eligible_audits
                   else "complete" if not missing else "incomplete_tracklet_labels"),
        "next_step": ("Run with --prepare --labelled-only to restrict analysis to officially labelled drives."
                      if missing and not args.labelled_only
                      else "Run with --prepare to create the fixed transition manifest."),
    }
    if args.prepare and not selected_missing and selected_audits:
        payload["transition_manifest"] = build_transition_manifest(selected_audits, args.window_s, args.min_events)
    elif args.prepare:
        payload["prepare_blocked"] = "Official tracklet_labels.xml files are required for ground-truth transition events."

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"KITTI raw audit: {payload['status']} | sequences={payload['n_sequences']} | selected={payload['n_selected_sequences']} | missing_labels={len(missing)}")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

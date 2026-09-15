from __future__ import annotations

from pathlib import Path

from experiments.exp13_temporal_lead_time import audit_dataset, build_transition_manifest, parse_tracklets


TRACKLETS = """<boost_serialization><tracklets><count>2</count>
<item><objectType>Pedestrian</objectType><first_frame>6</first_frame><poses><item/><item/></poses></item>
<item><objectType>Car</objectType><first_frame>0</first_frame><poses><item/><item/></poses></item>
</tracklets></boost_serialization>"""


def _raw_fixture(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    drive = raw / "2011_09_26" / "2011_09_26_drive_0001_sync"
    image_dir = drive / "image_02" / "data"
    image_dir.mkdir(parents=True)
    for index in range(12):
        (image_dir / f"{index:010d}.png").write_bytes(b"png")
    (drive / "image_02" / "timestamps.txt").write_text(
        "\n".join(
            f"2011-09-26 13:00:{index // 10:02d}.{(index % 10) * 100000:06d}"
            for index in range(12)
        ),
        encoding="utf-8",
    )
    (drive / "tracklet_labels.xml").write_text(TRACKLETS, encoding="utf-8")
    return raw, drive


def test_parse_standard_tracklets(tmp_path: Path) -> None:
    _, drive = _raw_fixture(tmp_path)
    tracklets = parse_tracklets(drive / "tracklet_labels.xml")
    assert [(track.object_type, track.first_frame, track.last_frame) for track in tracklets] == [
        ("pedestrian", 6, 7), ("car", 0, 1)
    ]


def test_transition_manifest_requires_vru_free_history(tmp_path: Path) -> None:
    raw, _ = _raw_fixture(tmp_path)
    audits = audit_dataset(raw, raw)
    assert audits[0].fps is not None
    assert abs(audits[0].fps - 10.0) < 1e-3
    manifest = build_transition_manifest(audits, window_s=0.5, min_events=1)
    assert manifest["status"] == "ready"
    assert manifest["n_transition_events"] == 1
    event = manifest["eligible_transition_events"][0]
    assert event["entry_frame"] == 6
    assert event["pre_frames"] == [1, 2, 3, 4, 5]


def test_transition_manifest_sorts_multiple_vru_entries(tmp_path: Path) -> None:
    raw, drive = _raw_fixture(tmp_path)
    (drive / "tracklet_labels.xml").write_text(
        """<boost_serialization><tracklets><count>2</count>
        <item><objectType>Cyclist</objectType><first_frame>6</first_frame><poses><item/></poses></item>
        <item><objectType>Pedestrian</objectType><first_frame>6</first_frame><poses><item/></poses></item>
        </tracklets></boost_serialization>""",
        encoding="utf-8",
    )
    manifest = build_transition_manifest(audit_dataset(raw, raw), window_s=0.5, min_events=1)
    assert [event["entry_frame"] for event in manifest["eligible_transition_events"]] == [6, 6]
    assert [event["object_type"] for event in manifest["eligible_transition_events"]] == ["cyclist", "pedestrian"]


def test_missing_tracklets_are_reported(tmp_path: Path) -> None:
    raw, drive = _raw_fixture(tmp_path)
    (drive / "tracklet_labels.xml").unlink()
    audits = audit_dataset(raw, raw)
    assert audits[0].tracklet_path is None
    assert audits[0].n_vru_tracklets is None

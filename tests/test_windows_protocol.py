"""Regression checks for the Windows validation protocol boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_windows_runner_rejects_non_onnx_backend() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_paper_suite.py",
            "--platform",
            "windows",
            "--backend",
            "pytorch",
            "--phase",
            "calibration",
            "--simulate",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "requires --backend onnx" in completed.stderr

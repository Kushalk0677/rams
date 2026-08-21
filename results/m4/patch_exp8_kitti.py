#!/usr/bin/env python3
"""Patch exp8_accuracy_per_tier.py so the KITTI mAP path can use a local dataset.

Changes one line: the hardcoded `data_arg = "kitti.yaml"` becomes a lookup of
the RAMS_KITTI_YAML environment variable, falling back to the old behaviour when
the variable is unset. Adds `import os` if absent. Writes a .bak first and is
safe to run twice.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET = Path("experiments/exp8_accuracy_per_tier.py")
OLD = '                data_arg = "kitti.yaml"'
NEW = ('                data_arg = os.environ.get("RAMS_KITTI_YAML", "kitti.yaml")')


def main() -> int:
    if not TARGET.is_file():
        sys.exit(f"ERROR: {TARGET} not found -- run this from the package root")

    source = TARGET.read_text()

    if "RAMS_KITTI_YAML" in source:
        print("Already patched -- nothing to do.")
        return 0

    if OLD not in source:
        sys.exit(
            "ERROR: could not find the expected line:\n"
            f"  {OLD.strip()}\n"
            "The file may have been modified. Patch it by hand."
        )

    backup = TARGET.with_suffix(".py.bak")
    shutil.copy2(TARGET, backup)
    print(f"Backup written: {backup}")

    patched = source.replace(OLD, NEW, 1)

    if "\nimport os\n" not in patched:
        patched = patched.replace("import logging\n", "import logging\nimport os\n", 1)
        print("Added: import os")

    TARGET.write_text(patched)
    print("Patched: data_arg now honours $RAMS_KITTI_YAML")
    print("\nVerify with:  grep -n 'RAMS_KITTI_YAML' experiments/exp8_accuracy_per_tier.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

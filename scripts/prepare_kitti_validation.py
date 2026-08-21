"""Create the fixed RAMS KITTI validation replay without overwriting data.

The evaluation protocol uses sorted KITTI training-frame indices 5981 through
7480 inclusive. This helper has no inference dependencies so a package operator
can verify the dataset layout before installing or running a model backend.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


START_INDEX = 5981
END_INDEX = 7480
EXPECTED_COUNT = END_INDEX - START_INDEX + 1


def sorted_files(directory: Path, pattern: str) -> list[Path]:
    return sorted(directory.glob(pattern), key=lambda path: path.name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the fixed 1,500-frame RAMS KITTI validation replay")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Directory containing kitti/images/training/image_2 and kitti/labels/training/label_2")
    parser.add_argument("--reuse-existing", action="store_true",
                        help="Accept an already complete 1,500-image and 1,500-label replay without copying")
    args = parser.parse_args()

    root = args.data_root.expanduser().resolve() / "kitti"
    image_source = root / "images" / "training" / "image_2"
    label_source = root / "labels" / "training" / "label_2"
    image_destination = root / "images" / "val"
    label_destination = root / "labels" / "val"
    for path in (image_source, label_source):
        if not path.is_dir():
            parser.error(f"Required KITTI source directory is missing: {path}")

    existing_images = sorted_files(image_destination, "*.png") if image_destination.exists() else []
    existing_labels = sorted_files(label_destination, "*.txt") if label_destination.exists() else []
    if image_destination.exists() or label_destination.exists():
        if args.reuse_existing and len(existing_images) == EXPECTED_COUNT and len(existing_labels) == EXPECTED_COUNT:
            print(f"Reusing existing validation replay: {image_destination} ({EXPECTED_COUNT} images), "
                  f"{label_destination} ({EXPECTED_COUNT} labels)")
            return 0
        parser.error(
            "Validation destination already exists. Refusing to overwrite it. "
            "Verify it contains 1,500 images and labels, then use --reuse-existing if appropriate."
        )

    images = sorted_files(image_source, "*.png")
    labels = sorted_files(label_source, "*.txt")
    if len(images) < END_INDEX + 1 or len(labels) < END_INDEX + 1:
        parser.error(
            f"KITTI source extraction is incomplete: found {len(images)} images and {len(labels)} labels; "
            f"need at least {END_INDEX + 1} of each."
        )

    image_destination.mkdir(parents=True)
    label_destination.mkdir(parents=True)
    for source in images[START_INDEX:END_INDEX + 1]:
        shutil.copy2(source, image_destination / source.name)
    for source in labels[START_INDEX:END_INDEX + 1]:
        shutil.copy2(source, label_destination / source.name)
    print(f"Created {EXPECTED_COUNT}-frame validation replay:")
    print(f"  images: {image_destination}")
    print(f"  labels: {label_destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

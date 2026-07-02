from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate prepared Anti-UAV directory structure.")
    parser.add_argument("data_root")
    args = parser.parse_args()
    root = Path(args.data_root)
    counts = {}
    for split in ("train", "val", "test"):
        split_root = root / split
        counts[split] = len([p for p in split_root.iterdir() if p.is_dir()]) if split_root.exists() else 0
    print({"data_root": str(root.resolve()), "sequence_counts": counts})


if __name__ == "__main__":
    main()

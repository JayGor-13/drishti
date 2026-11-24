"""Check that a converted Anti-UAV410 frames folder is identical to the one used for the paper.

It counts sequences and frames per split and computes a fingerprint over every frame's relative path
and file size, plus the full contents of every label file. The reference values below come from the
copy used for all reported results (converted on 29 Sep 2026 from the official Anti-UAV410.zip with
sha256 339e0e56e84d620acfd2e315df39ba42a6fd5bdc958fcc0c16e2130c6b41e055).

Usage:  python scripts/verify_antiuav410.py --frames-root /workspace/data/antiuav410_frames
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

REFERENCE = {
    "counts": {"train": (200, 213995), "val": (90, 94711), "test": (120, 129691)},
    "fingerprint": "71f830112dc28f809bb4f5741762856da0d01f5b38ea4f1bbd833ecd7a71b271",
}


def fingerprint(root: Path) -> tuple[dict, str]:
    digest = hashlib.sha256()
    counts = {}
    for split in ("train", "val", "test"):
        seqs = sorted(p for p in (root / split).iterdir() if p.is_dir())
        frames = 0
        for seq in seqs:
            label = seq / "infrared.json"
            digest.update(f"{split}/{seq.name}/infrared.json\n".encode())
            digest.update(label.read_bytes())
            for f in sorted((seq / "infrared").iterdir()):
                digest.update(f"{split}/{seq.name}/infrared/{f.name}:{f.stat().st_size}\n".encode())
                frames += 1
        counts[split] = (len(seqs), frames)
    return counts, digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a converted Anti-UAV410 frames folder.")
    parser.add_argument("--frames-root", required=True)
    args = parser.parse_args()
    counts, fp = fingerprint(Path(args.frames_root))
    ok = True
    for split, (seqs, frames) in counts.items():
        want = REFERENCE["counts"][split]
        flag = "OK" if (seqs, frames) == want else "MISMATCH"
        ok &= flag == "OK"
        print(f"{split:5s} {seqs:4d} sequences {frames:7d} frames  (expected {want[0]} / {want[1]})  {flag}")
    same = fp == REFERENCE["fingerprint"]
    print(f"fingerprint {fp}  {'OK, identical to the paper copy' if same else 'MISMATCH'}")
    raise SystemExit(0 if ok and same else 1)


if __name__ == "__main__":
    main()

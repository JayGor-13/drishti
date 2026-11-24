"""Pause or resume a running DRISHTI training job (and its data-loader workers) without losing progress.

Pausing suspends the processes: no CPU/GPU compute is used, but GPU memory (~4 GB) and RAM stay allocated.
Resuming continues exactly where it stopped. To free GPU memory completely, stop the job instead;
scripts/run_all_stages.py then resumes from the last saved epoch (<stage>_latest.pt).

Usage:  python scripts/pause_resume_training.py pause
        python scripts/pause_resume_training.py resume
"""
from __future__ import annotations

import os
import sys

import psutil

MARKERS = ("run_all_stages", "drishti_v2.experiments.run_training")


def targets() -> list[psutil.Process]:
    roots = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        if proc.pid == os.getpid():
            continue
        cmd = " ".join(proc.info["cmdline"] or [])
        if any(marker in cmd for marker in MARKERS):
            roots.append(proc)
    found = {p.pid: p for p in roots}
    for root in roots:
        for child in root.children(recursive=True):
            if child.pid != os.getpid():
                found[child.pid] = child
    return list(found.values())


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"pause", "resume"}:
        raise SystemExit(__doc__)
    procs = targets()
    for proc in procs:
        (proc.suspend if sys.argv[1] == "pause" else proc.resume)()
    print(f"{sys.argv[1]}d {len(procs)} processes: {sorted(p.pid for p in procs)}")


if __name__ == "__main__":
    main()

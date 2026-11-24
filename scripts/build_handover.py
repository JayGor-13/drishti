"""Build the complete handover zip: code + results (JSON summaries, tables, logs, tracker scores) + weights + docs.

Run from the drishti-master folder (the one that holds runs/, weights/ and drishti-master/):
  python drishti-master/scripts/build_handover.py
Produces drishti_handover_complete_<date>.zip next to runs/ and refreshes dgx_transfer/drishti_code.zip.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # .../drishti-master (holds runs/, weights/, drishti-master/)
CODE = ROOT / "drishti-master"
DGX_ZIP = ROOT.parent / "dgx_transfer" / "drishti_code.zip"
SKIP_RUN_DIRS = {"pretrained", "smoke_synthetic", "prev_infrared_fast_stride24", "check_enc_resnet18", "check_enc_simple",
                 "cvpr_seed42", "antiuav410_seed42"}
DOCS = ["PAPER_PLAN.md", "HOW_TO_RESUME.md", "PAPER_NOTES.md", "CHANGES_2026-09-28.md", "PLAN_CVPR2027.md", "DATASET.md",
        "BASELINES_AntiUAV410_trackers.md"]
README = """# Handover package ({date})

Start here, in this order:
1. PAPER_PLAN.md     what the idea is, how to frame it, what to put in the paper, which baselines, which figures
2. PAPER_NOTES.md    the exact protocol, the main numbers, negative results, where every number lives
3. results/TABLES.md every table, generated from results/ (re-generate: python code/scripts/make_tables.py --runs results/runs --trackers results/antiuav410_scores --out results/TABLES.md)
4. HOW_TO_RESUME.md  data download + extraction, Docker images, the one script behind every number, the launchers, the weights
5. CHANGES_2026-09-28.md  the dated log of every experiment and why (sections 24-33 matter for the paper)

code/      the complete repository (run anything from HOW_TO_RESUME.md with it)
results/   result files of every run (no raw predictions), the analyses, the queue logs, the 17 tracker scores
weights/   the final system (3 seeds) and the P2 variant, Ultralytics checkpoints
"""


def main() -> None:
    os.chdir(ROOT)
    # 1. tables
    subprocess.run([sys.executable, str(CODE / "scripts" / "make_tables.py"), "--runs", "runs", "--trackers",
                    str(ROOT.parent / "antiuav410" / "scores"), "--out", "runs/TABLES.md"], check=True, stdout=subprocess.DEVNULL)
    # 2. the code tree = everything the DGX zip holds (+ keep it in sync with the working copy)
    z0 = zipfile.ZipFile(DGX_ZIP)
    code_names = sorted(n for n in z0.namelist() if not n.endswith("/"))
    z0.close()
    with zipfile.ZipFile(DGX_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for n in code_names:
            z.write(CODE / n[5:], n)
    # 3. the handover zip
    out = ROOT / f"drishti_handover_complete_{time.strftime('%Y-%m-%d')}.zip"
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for cn in code_names:
            z.write(CODE / cn[5:], cn); n += 1
        for f in glob.glob("runs/**/*.json", recursive=True) + glob.glob("runs/**/*.csv", recursive=True):
            rel = f.replace("\\", "/"); parts = rel.split("/")
            if any(p.startswith("raw_") for p in parts) or "/train/" in rel or "/logs/" in rel or parts[1] in SKIP_RUN_DIRS:
                continue
            z.write(f, "results/" + rel); n += 1
        z.write("runs/TABLES.md", "results/TABLES.md"); n += 1
        for f in glob.glob("runs/logs/run_*.log"):
            z.write(f, "results/" + f.replace("\\", "/")); n += 1
        for f in glob.glob(str(ROOT.parent / "antiuav410" / "scores" / "*_test.json")):
            z.write(f, "results/antiuav410_scores/" + os.path.basename(f)); n += 1
        for f in glob.glob("weights/*.pt"):
            z.write(f, "weights/" + os.path.basename(f)); n += 1
        for d in DOCS:
            if (CODE / d).exists():
                z.write(CODE / d, d); n += 1
        z.writestr("README_FIRST.md", README.format(date=time.strftime("%d %b %Y"))); n += 1
    print(f"{out.name}: {n} files, {out.stat().st_size / 1e6:.1f} MB")
    print(f"{DGX_ZIP}: {len(code_names)} files")


if __name__ == "__main__":
    main()

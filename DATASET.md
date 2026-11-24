# Getting exactly the same Anti-UAV410 frames

Every number in the paper uses one specific copy of Anti-UAV410. Follow these steps and the last one
proves your frames are byte-for-byte the same set. Run them on the DGX; `ROOT=/raid/scratch/<your user>/drishti`.

## 1. Download the official dataset (9.36 GB)

Source: the official Anti-UAV410 repository, https://github.com/HwangBo94/Anti-UAV410 (TPAMI 2023).
Pick one of the three options.

**Option A, Google Drive from the command line (recommended):**
```
mkdir -p ROOT/data && cd ROOT/data
python3 -m pip install --user gdown
python3 -m gdown 1zsdazmKS3mHaEZWS2BnqbYHPEcIaH5WR -O Anti-UAV410.zip
```
**Option B, browser:** https://drive.google.com/file/d/1zsdazmKS3mHaEZWS2BnqbYHPEcIaH5WR/view , then copy
it to `ROOT/data/Anti-UAV410.zip` with `scp`.

**Option C, Baidu:** https://pan.baidu.com/s/1R-L9gKIRowMgjjt52n48-g?pwd=a410 (password `a410`).

## 2. Check you have the identical file

```
cd ROOT/data
ls -l Anti-UAV410.zip          # must be 9361681896 bytes
sha256sum Anti-UAV410.zip      # must be 339e0e56e84d620acfd2e315df39ba42a6fd5bdc958fcc0c16e2130c6b41e055
```
If the checksum differs, the download is incomplete or a different release: download again.

## 3. Convert it to our frame layout (about 15 to 30 minutes)

```
cd ROOT
docker run --rm -u $(id -u):$(id -g) -v $PWD:/workspace -w /workspace/code drishti:latest \
  python scripts/prepare_antiuav410.py --zip /workspace/data/Anti-UAV410.zip --out /workspace/data/antiuav410_frames
```
Expected output:
```
test_frames: 129691
test_sequences: 120
train_frames: 213995
train_sequences: 200
val_frames: 94711
val_sequences: 90
```

What the conversion does, so nothing is hidden:
- Official layout `<split>/<video>/*.jpg` + `<split>/<video>/IR_label.json` becomes
  `antiuav410_frames/<split>/<video>/infrared/*.jpg` + `antiuav410_frames/<split>/<video>/infrared.json`.
- Every JPG is copied **byte for byte** from the zip (no re-encoding, no resizing). The label file is
  copied unchanged; it keeps `exist`, `gt_rect` (x, y, w, h in pixels, top-left corner) and the
  per-frame attribute flags (`LR`, `OC`, `FM`, `SV`, `IC`, `DBC`, `TS`) used for per-attribute results.
- Frame *i* of a video is the *i*-th JPG in sorted file-name order; label entry *i* belongs to it.
- Resizing (448x448 for our model, 640 for YOLO) happens inside each method, never on disk.
- The official train / val / test split is used as is: **do not move videos between splits**.

## 4. Prove your frames are identical to the paper copy

```
cd ROOT
docker run --rm -u $(id -u):$(id -g) -v $PWD:/workspace -w /workspace/code drishti:latest \
  python scripts/verify_antiuav410.py --frames-root /workspace/data/antiuav410_frames
```
It must print `OK` for all three splits and
`fingerprint 71f830112dc28f809bb4f5741762856da0d01f5b38ea4f1bbd833ecd7a71b271  OK, identical to the paper copy`.
The fingerprint covers every frame's path and size and the full content of every label file.

**Shortcut:** if you can read the advisor's DGX folder, copying is faster than downloading:
`cp -r /raid/scratch/akshita_sit/drishti/data/antiuav410_frames ROOT/data/` and then run step 4 to verify.

## 5. Formats the baselines need (made from the verified frames)

- **YOLO11:** `scripts/baselines/yolo_antiuav.py` builds its own YOLO view (`--yolo-root`) from these frames.
- **COCO (DEIM / D-FINE, Gaze-DETR and other DETR-style code):**
```
docker run --rm -u $(id -u):$(id -g) -v $PWD:/workspace -w /workspace/code drishti:latest \
  python scripts/export_coco.py --frames-root /workspace/data/antiuav410_frames --modality infrared \
  --out /workspace/data/coco_a410 --train-stride 4
```
  Writes `coco_a410/{train,val,test}_infrared.json`; `file_name` is relative to `antiuav410_frames`,
  so set the image root of the baseline to `ROOT/data/antiuav410_frames`. Train and val use every 4th
  frame (enough for training); **test always uses every frame**.
- **Original layout (2025 challenge winner):** their code reads `IR_label.json` per video, i.e. the
  official zip unpacked as is: `cd ROOT/data && python3 -m zipfile -e Anti-UAV410.zip antiuav410_official`.

## 6. Optional: reproduce the 17-tracker reference table

Official raw outputs of the trackers retrained on Anti-UAV410 (350 MB,
sha256 `79b70e0e56c212bfb19223a22b007ad61e2cb18bf127c16319982f6c61cf6cea`):
```
cd ROOT/data && python3 -m gdown 1AlLpoMorj-7bKA1zqo1DkuEZ9h0jQs_- -O tracker_results.zip
cd ROOT
docker run --rm -u $(id -u):$(id -g) -v $PWD:/workspace -w /workspace/code drishti:latest \
  python scripts/import_tracker_results.py --results-zip /workspace/data/tracker_results.zip --out /workspace/data/tracker_preds
```
Then score each tracker with `scripts/score_predictions.py ... --pred-dir /workspace/data/tracker_preds/<tracker>`.
SiamDT must come out at 68.17 Anti-UAV accuracy (the TPAMI paper reports 68.19); that checks the metric code.

## Anti-UAV300 (not needed for baselines)

Our older development set (318 videos) is only used for our own ablations. 150 of Anti-UAV410's 200
train videos and 88 of its 120 test videos are Anti-UAV300 videos, so **never train anything scored on
Anti-UAV410 with Anti-UAV300 val or test data**.

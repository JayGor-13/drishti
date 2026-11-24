# Paper notes: what we found, what the numbers are, and how to use this results package

Written 23 Oct 2026 for whoever drafts the paper. Plain words; every number below comes from a file in this
package (see "Where each number lives"). Deadline: CVPR 2027, registration 10 Nov, submission 16 Nov 2026.

## 1. The story in one paragraph

Anti-UAV410 is filmed by a camera that follows the drone: in the labels, the drone's centre moves only
2.5 px per frame (median), under 1 px in 21% of frames. So plain frame differencing, the standard "motion
input" (used by the 2025 challenge winner), lights up the moving background more than the drone. We (1)
compensate the camera's motion before differencing, (2) difference over a longer gap (12 frames) so the
drone's own small displacement accumulates, and (3) train the box regression with a Gaussian box distance
(NWD) instead of IoU, which is unstable for 10-px boxes. On a 6.4-GFLOP YOLO11n this gives 70.3 tracking
accuracy on the Anti-UAV410 test set (3 seeds, +- 0.09): +4.1 over the same detector on plain frames, above
every detector we trained (15 of them, incl. RT-DETR-l at 72 GFLOPs and DINO) and above the best published
tracker (SiamDT 68.2), which is given the first-frame box. A second, independent finding: the usual way of
scoring detectors on this benchmark (a confidence threshold of 0.1 or more) under-reports them by about 5
points, because the metric punishes any empty frame and only 2% of frames are empty; the 17 published
trackers never abstain. With one fair protocol for everyone, cheap detectors lead the benchmark.

## 2. The protocol (say this exactly; it is what makes the comparison fair)

- Dataset: Anti-UAV410 official split, 200 / 90 / 120 sequences, infrared, every frame (213,995 / 94,711 /
  129,691). Test labels are public. `DATASET.md` has the download and frame extraction.
- Every method predicts every frame of val and test; at most 10 boxes per frame, any score.
- Post-processing (duplicate merging IoU 0.3, one box per frame, presence threshold from 0.001 to 0.5,
  optional smoothing) is chosen on VAL for each method and applied once to TEST (`scripts/sweep_postprocess.py`).
  Val always picks the lowest threshold: on this metric, never abstaining is optimal (section 1).
- Metrics: the Anti-UAV410 tracking accuracy (official formula, our implementation reproduces SiamDT's
  published 68.19), mAP@50 and mAP@75 (COCO-style), precision / recall at the chosen threshold.
- Trackers: the official raw results of the 17 retrained trackers from the Anti-UAV410 authors, scored with
  the same code (Track 1: they receive the first-frame box; we do not; say so in the table).
- Training of every detector: 40 epochs on every 4th training frame (more data did not help, section 4),
  COCO-pretrained weights, 640 input, seed 42 (3 seeds where stated), checkpoint chosen on val.

## 3. Main results (Anti-UAV410 test)

Final system = YOLO11n-640 + compensated frame differences (gaps 1 and 12) + NWD loss.

| Method | Track acc. | mAP50 | mAP75 | GFLOPs |
| --- | --- | --- | --- | --- |
| **Ours, final (3 seeds)** | **70.30 +- 0.09** | 90.2 | 42.9 | 6.4 (+ ~8 ms CPU) |
| Ours with a P2 head (yolov8n-p2) | 70.29 | 90.7 | 43.8 | ~12 |
| Ours, gap 8 only (3 seeds) | 69.17 +- 0.18 | 88.8 | 43.5 | 6.4 |
| YOLO11n, 2025 winner's input (signed diffs t-1, t-2) | 67.00 | 86.0 | 41.4 | 6.4 |
| YOLO11n, plain frames | 66.22 | 84.2 | 43.3 | 6.4 |
| YOLO26n (640) / YOLO11s (640) | 66.33 / 66.14 | 85.6 / 85.1 | 43.8 / 43.2 | 6.0 / 21.5 |
| RT-DETR-l (512) | 65.41 | 84.7 | 35.1 | 72 |
| DINO-4scale R50 / Deformable-DETR R50 | 66.32 / 64.54 | 84.6 / 83.1 | 41.8 / 33.2 | large |
| YOLOv8n / v9t / v10n / 11n / 12n / 26n (512) | 66.0 / 66.0 / 65.2 / 65.4 / 65.4 / 67.2 | | | |
| SSTNet (TGRS'24) / CHAL (CVPR'26), their code, trained from scratch | 55.35 / 54.25 | 71.0 / 70.5 | | |
| Best tracker, SiamDT (given the first-frame box) | 68.17 | | | |

The sequence decoder (Viterbi over all candidate boxes with an absent state, `scripts/decode_candidates.py`)
adds +0.1 to +0.5 on every detector; its row for the final system is produced by `dgx/run_rescore.sh`
(`decode_raw.json` in each run folder) and should be added when available.

## 4. What did not work (keep a short "negative results" paragraph; reviewers like it)

- A separate motion network on top (our original DRISHTI model): 0.0 under fair scoring, 8x slower.
- A learned judge between sources, a learned box refiner, a zoom-in second look, temporal box smoothing:
  no gain. Optical flow, time-surface, delta/delta-delta, background-subtraction and local-contrast inputs:
  no gain or worse. CLAHE, box-loss weight, rectangular training: worse. Training on every frame: no gain.
- Early stopping / checkpoint selection on Ultralytics' own val mAP is noisy; we choose checkpoints on val
  tracking accuracy (`weights_choice.json`).

## 5. Tables still being produced (DGX, `dgx/run_final_night.sh`, ~6 h)

Ablation of the final system: minus compensation, minus NWD, minus the motion input; the final recipe on
YOLO26n (transfer). Plus the decoder row for every run. `scripts/make_tables.py` prints all tables from
the result files once those land.

## 6. Figures to make (data is in this package)

1. Gap sweep: tracking accuracy vs frame gap (3, 5, 8, 12, 16) with the seed band at gap 8.
2. The threshold effect: tracking accuracy vs presence threshold for 3 detectors (from `postprocess_sweep_low.json` grids).
3. Per-size / per-attribute detection rate: appearance-only vs ours (`candidates_dyn_k8/ablation_v2/by_attribute.json`).
4. The camera-following fact: histogram of per-frame drone displacement (from the test labels).
5. Qualitative: frames where plain differencing fails and compensated differencing finds the drone.

## 7. Where each number lives

- `runs/baseline_<name>/postprocess_sweep_low.json` (or `postprocess_sweep.json`): `chosen_on_val.test` = the
  table numbers; `grid` = the full val sweep. `cost.json`: GFLOPs, params, latency, seed, checkpoint choice.
  `decode_raw.json`: the decoder rows. `score_test_raw.json`: raw (all boxes, threshold 0.3) test score.
- `runs/candidates_dyn_k8/`: the candidate-pool analyses (`analysis_test.json` = ceiling, `ablation_v2/` =
  decoder ablations and per-attribute breakdown, `smooth_ext.json` = box smoothing).
- `runs/annotation_noise_test.json`: label-consistency bound (0.90 IoU). `runs/system_cost.json`: timing.
- `antiuav410_scores/<tracker>_test.json`: the 17 trackers under our scorer.
- `CHANGES_2026-09-28.md`: the dated log of every experiment and why (sections 24 to 33 are the ones that
  matter for the paper); `PLAN_CVPR2027.md`: the plan and the idea backlog with outcomes.
- `scripts/make_tables.py`: regenerates every table above from these files.

## 8. Caveats to state in the paper

- Gaze-DETR's table (arXiv) reports mAP50 about 1 to 2 points above ours for the same detectors; their frame
  counts suggest they scored val. We re-ran everything under one protocol and do not copy numbers.
- The 2025 challenge winner's training code is not released; their row is their exact input on our YOLO11n.
- PConv and L2SKNet could not be reproduced (no training code / no code). CHAL's public code is missing one
  file (CBAM), restored from parts in the repository; CHAL and SSTNet were trained from scratch (their
  weights are only on Baidu Pan) with a shortened schedule, so their rows are lower bounds.
- Single-benchmark paper; say so. Anti-UAV300 is a near-subset of Anti-UAV410 and does not count as a second one.

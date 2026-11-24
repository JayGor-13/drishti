# Anti-UAV410 test set: 17 retrained trackers, scored with our code (29 Sep 2026)

Source: official raw outputs (`Trained_with_antiuav410`) from https://github.com/HwangBo94/Anti-UAV410,
converted with `scripts/import_tracker_results.py`, scored with `scripts/score_predictions.py` (threshold 0.3, 120 test sequences).
Check: SiamDT scores 68.17 here vs 68.19 reported in the Anti-UAV410 TPAMI paper, so our metric code matches the official one.
All trackers are GIVEN the first-frame box; our detector is not. Report them in a separate block.

| Tracker | Anti-UAV acc. (%) | mAP@50 | mAP@75 | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| SiamDT | 68.17 | 82.8 | 46.0 | 86.1 | 87.7 | 86.7 |
| GlobalTrack | 66.43 | 80.7 | 43.0 | 84.4 | 85.9 | 85.0 |
| DropTrack | 60.13 | 75.0 | 29.3 | 78.2 | 79.4 | 78.7 |
| MixFormerV2 | 59.63 | 72.6 | 33.8 | 76.2 | 77.4 | 76.7 |
| AiATrack | 59.54 | 74.3 | 26.3 | 77.8 | 79.1 | 78.3 |
| Stark-ST101 | 57.13 | 70.4 | 27.1 | 74.3 | 75.4 | 74.7 |
| KeepTrack | 56.80 | 68.6 | 24.8 | 73.3 | 74.5 | 73.8 |
| DiMP | 56.65 | 68.2 | 34.7 | 71.5 | 72.6 | 71.9 |
| SwinTrack-Base | 55.65 | 68.6 | 28.0 | 72.3 | 73.4 | 72.7 |
| ToMP | 55.08 | 66.7 | 29.1 | 70.5 | 71.5 | 70.9 |
| SwinTrack-Tiny | 52.20 | 64.9 | 31.7 | 68.1 | 68.9 | 68.4 |
| ATOM | 51.36 | 63.0 | 29.0 | 65.5 | 66.4 | 65.8 |
| SiamBAN | 47.31 | 59.2 | 19.0 | 62.5 | 63.4 | 62.8 |
| SiamCAR | 46.93 | 59.4 | 26.2 | 60.7 | 61.5 | 61.0 |
| KYS | 44.77 | 54.0 | 22.0 | 57.0 | 57.8 | 57.3 |
| ROAM | 43.03 | 54.7 | 18.1 | 56.5 | 57.3 | 56.8 |
| TCTrack | 41.64 | 54.5 | 16.6 | 55.4 | 56.2 | 55.7 |

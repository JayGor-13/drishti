# Paper plan, in plain language (23 Oct 2026)

Target: CVPR 2027 (registration 10 Nov, submission 16 Nov). Fallback: the CVPR 2027 Anti-UAV workshop.
Everything here is backed by numbers in `results/TABLES.md`; `PAPER_NOTES.md` has the protocol and file guide.

## 1. What the idea is (the message of the paper)

Infrared anti-drone video is filmed by cameras that follow the drone. We measured it: in the test labels the
drone's centre moves only 2.5 px per frame (median), under 1 px in one frame out of five. So the standard
"motion input" for drone detectors, the plain difference between consecutive frames, mostly shows the
background sliding past, not the drone. Three small, cheap changes fix this for a nano detector:

1. **Compensate the camera.** Measure the global shift of the background between frames (phase correlation at
   quarter resolution, a few milliseconds) and align past frames before differencing. The background cancels;
   the drone, which does not move with it, stays.
2. **Difference over a long gap.** With the camera following, the drone's own displacement is tiny per frame
   but adds up: differencing against the frame 12 steps back (plus the previous frame) shows it. The gap sweep
   (3, 5, 8, 12, 16) is a clean curve.
3. **Train the boxes with a size-aware distance (NWD).** For 10-pixel boxes, IoU is unstable (a 1-px shift
   changes it a lot), so the usual box loss and label assignment are noisy. Modelling boxes as Gaussians
   fixes that, a known trick from tiny-object detection, not used in anti-UAV before.

Result: YOLO11n (2.6 M parameters, 6.4 GFLOPs) goes from 66.2 to **70.3 tracking accuracy** (3 seeds,
+- 0.09) on Anti-UAV410 test, above every detector we trained (15, including RT-DETR-l at 72 GFLOPs and
DINO) and above the best published tracker (SiamDT, 68.2), which is handed the drone's first-frame box.

**The second message (evaluation).** The benchmark's tracking-accuracy metric punishes any empty frame, and
only 2% of frames are empty, so never abstaining is the winning strategy; the 17 published trackers never
abstain. Detectors scored with the usual confidence threshold (0.1 or more) lose about 5 points for nothing.
Scoring everyone the same way changes the ranking: RT-DETR-l looked best among detectors; under fair scoring
every small YOLO beats it. State this as a finding and a recommendation; it is also what makes our
comparison honest.

## 2. How to frame it (what we claim and what we do not)

- Claim: a *recipe* (camera-motion-compensated long-gap differencing + a tiny-object box loss) that makes
  the cheapest detectors the strongest anti-drone detectors, plus a corrected evaluation protocol.
- Do not claim: a new architecture, a new loss, or that motion input is our idea (the 2025 challenge winner
  used raw differences; our input beats their exact input by 3.3 points, say that instead).
- Be explicit that the gains come from understanding the data (the camera-following fact) rather than from
  model capacity: training on 4x more frames gave nothing, a larger detector gave nothing, and a 72-GFLOP
  DETR is 5 points below a 6-GFLOP YOLO with our input.
- Honest scope: one benchmark (the only one with public test labels and a per-frame protocol). Say it.

## 3. What goes in the paper

**Title idea:** "The camera follows the drone: motion-compensated inputs make nano detectors the best
anti-UAV detectors" (or something less playful; keep "camera follows the drone", it is the insight).

**Sections and their evidence**
1. Introduction: the camera-following fact (figure: histogram of per-frame drone displacement), the two
   messages above, the headline numbers.
2. Related work: anti-UAV detectors and trackers (Anti-UAV410 paper, the 2025 challenge winner, Gaze-DETR
   as the latest detector with its caveat), moving infrared small-target detection (CHAL, SSTNet, MI-DETR,
   decoupled-motion), tiny-object losses (NWD, SimD), camera-motion compensation in video.
3. Method: the three pieces, each with its motivation and one equation; the sequence decoder as an
   optional post-process (+0.1 to +0.5 on every detector); the evaluation protocol.
4. Experiments:
   - Main table (TABLES.md "Main table"): ours (3 seeds), ours + P2, the 15 detectors, SSTNet and CHAL,
     with GFLOPs; trackers in a separate block labelled "given the first-frame box".
   - Build-up table: plain -> gap 8 -> gap 12 -> + compensation -> + NWD -> final; and the ablation (final
     minus each piece), from `run_final_night.sh`.
   - Motion-input table (the gap sweep, winner's input, flow, time surface, delta, background subtraction).
   - The evaluation finding: old vs fair scoring for every detector (from `run_rescore.log` / the two
     sweep files per run), the "always output" observation, the trackers' abstention rate (0.0 to 1.0%).
   - Transfer: the recipe on YOLO11s and YOLO26n; the P2 variant for box quality.
   - Where the gains come from: per-size and per-attribute detection rates (thermal crossover, clutter).
   - Cost: GFLOPs, latency, the ~8 ms CPU for compensation (`system_cost.json`, `cost.json`).
5. Negative results (short, honest): the separate motion network, the learned judge, refiner, zoom, box
   smoothing, more data, larger input, other inputs. Reviewers trust a paper more for this.
6. Limitations: single benchmark; compensation assumes a mostly rigid background shift (pan/tilt), fails
   under strong parallax or zoom; label noise bounds mAP75 (our label-consistency check: 0.90 IoU).

## 4. Baselines to compare against (all already run under one protocol)

- Detectors at 512, Gaze-DETR's comparison set: YOLOv8n, v9t, v10n, YOLO11n, YOLO12n, YOLO26n, RT-DETR-l,
  DINO-4scale R50, Deformable-DETR R50. Plus YOLO11n/11s/26n at 640 (plain) as the direct controls.
- Motion-aware detectors: the 2025 winner's input on YOLO11n (their code is not released; say so), CHAL
  (CVPR'26) and SSTNet (TGRS'24) with their own code trained from scratch (weights only on Baidu; say so,
  and that their rows are lower bounds).
- Trackers: the 17 official Anti-UAV410 tracker results, re-scored by us (reference block only).
- Do not include: Gaze-DETR (arXiv only; their numbers appear to be on val), PConv / L2SKNet (no usable code).

## 5. Figures (5 is enough)

1. Per-frame drone displacement histogram + a frame pair showing raw vs compensated differencing.
2. Gap sweep curve with the seed band.
3. Old-vs-fair scoring: tracking accuracy vs presence threshold for three detectors.
4. Per-size / per-attribute detection rate: plain input vs ours.
5. Accuracy vs GFLOPs scatter: all detectors + ours (+ P2), trackers as hollow markers.

## 6. What is finished and what is not (as of 23 Oct)

Finished: every number in sections 3 to 5 except the ablation rows of the final system, the YOLO26n
transfer row and the decoder row for the final system (one queued DGX job, `run_final_night.sh`, ~6 h;
`HOW_TO_RESUME.md` section 5). Not needed any more: student baselines (we ran them all).
Writing has not started. Suggested split: method + evaluation finding first (they are the paper), tables
from `TABLES.md`, figures from the result files listed in `PAPER_NOTES.md` section 7.

## 7. Go / no-go (2 Nov)

Main track is defensible if the writing makes the two messages sharp and the ablation confirms that each
piece contributes (compensation and NWD each about +0.7 to +0.8 alone, +1.1 together with gap 12). If a
reviewer-sceptic reading is "a recipe paper", the Anti-UAV workshop takes the same paper unchanged.

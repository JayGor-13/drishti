# DRISHTI: plan and next steps for CVPR 2027 (main track)

Written 2026-09-28 for the student running the experiments. Plain words on purpose.
Deadline: paper registration 10 November 2026, submission 16 November 2026 (seven weeks).

---

## FOCUS (decided 17 Oct 2026; overrides everything below where they differ)

**One idea, one paper.** *A learned policy that decides, frame by frame, which expert to trust
(appearance = a dense detector, motion = our network, memory = where the drone should be) and when to
spend effort (run the expensive motion expert only when needed).*

**Motivation (the story in one paragraph).** Appearance detectors such as YOLO are accurate on clear
drones but miss tiny, blurred or cluttered ones. Motion finds those but gives loose boxes and false
alarms. Neither is best everywhere, and the per-attribute analysis shows *where* each one wins. Fixed rules
(highest score wins, score averaging) do not know which situation they are in. A small learned judge
does, and it can also skip the motion expert on easy frames, so we get the accuracy of both at close to the
cost of one.

Everything else we built (Chain-of-Focus, memory crops, the MoE inside the motion network) is part of
the motion expert. It is described briefly and checked by ablation, but **not claimed as a contribution**.

**Order of work.**
1. v6 (the judge): running. Claim holds only if judge > calibrated fusion > plain v5 > YOLO on test.
2. Frame-dynamics YOLO11n: the strongest *simple* competitor (YOLO that sees frame differences at its input,
   the idea of the 2025 challenge winner). Built 17 Oct; runs alongside v6.
3. v6.1 (when to spend effort): after v6 results. Offline first, on the saved candidates: learn "is this
   frame hard?" and run the motion expert only then; report accuracy vs fraction of frames where it runs.
4. Baselines (table below), seeds 2 and 3 of the final setup, ablations. Go/no-go on 2 Nov.

**Baselines (revised 17 Oct).** We do not compare against Gaze-DETR itself (arXiv only, not reviewed).
Instead we use the *same comparison set as Gaze-DETR's Table III* (arXiv 2607.19040), because it is the
most complete recent list of detectors on Anti-UAV410, and we drop the ones that exist only on arXiv.
Their protocol: 512 input, all frames, mAP@50 + P/R/F1, IoU 0.5. We report that **and** our Anti-UAV
tracking accuracy, all from our own runs (never copied numbers). Sanity check: their YOLO11n gets
84.74 mAP@50 at 512; ours gets 84.2 at 640, so the protocols agree.
Caution: their text lists the val and test frame counts swapped (129,691 is test, 94,711 is val), so
their numbers may be on val; another reason to rerun everything ourselves.

| Group (reviewer question) | Baseline | Venue | Who | Status |
| --- | --- | --- | --- | --- |
| Single-frame detectors (is one strong detector enough?) | YOLOv8n, YOLOv9t, YOLOv10n, YOLO11n, YOLO12n, YOLO26n | Ultralytics / ECCV'24 / NeurIPS'24 | us, `dgx/run_detector_baselines.sh` | built, to run |
| | RT-DETR-l | CVPR'24 | us, same script | built, to run |
| | DINO-DETR, Deformable-DETR (R50) | ICLR'23 / ICLR'21 | student (mmdetection or detrex) | to do |
| Infrared small-target detectors | PConv | AAAI'25 | student | to do |
| | L2SKNet ("Saliency at the helm") | TGRS'25 | student | to do |
| Temporal / moving-target detectors (isn't a video detector enough?) | SSTNet | TGRS'24 | student | to do |
| | CHAL | CVPR'26 | student | to do |
| Motion fed into a detector (our key competitor) | Frame-dynamics YOLO11n | ours, isolates the idea | us | built, to run |
| | 2025 Anti-UAV challenge winner, their code | CVPR'25 workshop | student | to do |
| Hand-made rules / ceiling | Highest score wins (v5), calibrated fusion, ours only, YOLO only, oracle | — | automatic in v6 | running |
| Reference only (different task) | 17 trackers given the first-frame box | — | done | separate block |

Dropped: Gaze-DETR, YOLOv6n, YOLOv7, YOLOv13n (arXiv only in their table), and from our earlier list
DEIM, YOLOV++, MOCID (not in the standard comparison; only if time is left after 2 Nov).

**Ablations (one table):** judge without memory cues; without agreement cues; without previous-frame cues;
mlp vs attention judge; judge trained on train vs on val; motion expert without Chain-of-Focus.

**Status 18 Oct.** Round 7 answered the big question: our motion network adds +3.4 tracking accuracy on
top of the strongest single detector (frame-dynamics YOLO: 63.5 -> 66.9), because the two sources find
different drones (15.2% of frames missed by each alone, 11.7% together). Round 8 measured where the rest
is: 8.3% of frames where neither source produces any candidate (mostly small drones), not loose boxes
(found drones overlap 0.81) and not bad choices (judge dead, oracle +0.8).

**The one idea, restated:** *two complementary ways of seeing a drone (how it looks, how it moves),
combined by evidence accumulated over time rather than frame by frame.* Dropped for good: the learned
judge (v6), adaptive computation (v6.1), tracker feedback, Track-1 comparisons as a main result, the box
refiner (v7: no gain, 19 Oct).

**Status 19 Oct (all of rounds 8 to 10 in; details in CHANGES section 20).** Best single small model is
our k=8 frame-difference YOLO11n (64.5 / 89.2 / 43.1), 2.7 tracking points above the winner's exact input.
Full-pool ceiling: 7.0% of frames invisible to both eyes, ~2.6% recoverable from low-score boxes. Labels
are consistent to 0.90 IoU, so box quality has room, but a crop refiner does not reach it. Next: (1) two-eye
fusion with the k=8 eye (`dgx/run_fusion_variant.sh`), (2) build the decoder + weighted box merging on the
saved candidates, (3) seeds and the ablation table. Go/no-go 2 Nov unchanged.

**Status 20 Oct: first full system.** Gap-8 YOLO11n + our motion network + sequence decoder: **68.7 online
/ 68.9 with a 15-frame delay** (test, all settings chosen on val), above every detector baseline (best
RT-DETR-l 65.4 at 72 GFLOPs) and above the best tracker given the first-frame box (SiamDT 68.2). Gains: +3.3
input, +3.0 our network, +0.6 to +0.8 decoding. Remaining: seeds for every component, ablations, student
baselines, paper. mAP75 (43) is the known weakness.

**Idea backlog (17 Oct), ranked by fit to the measured problem.** Attack = CM (complete misses, ~10% of
frames: need better proposals) or LB (loose boxes: need tighter boxes). Cost = GPU/person time.

| # | Idea | Borrowed from | Attacks | Cost | Ours / novel? | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Track-before-detect: accumulate weak detections along plausible paths over 10-30 frames before declaring | radar / sonar | CM | medium, CPU on saved boxes, ~2 days | high on this benchmark; keeps the temporal story | candidate headline if round 7 says our network is redundant |
| 2 | Time-surface motion channel (per-pixel decaying "time since last change") | event cameras | CM (slow / hovering drones) | cheap, 2.3 h per YOLO training | medium; a measured comparison of motion encodings (diff / LDMI / time surface) | run alongside v7 |
| 3 | Multi-lag frame differences at the input, gap sweep (k = 3 / 5 / 8) and the winner's exact variant (signed, lags 1 and 2) | **2025 winner's recipe** (arXiv 2505.04917 Eq. 1); our frame-dynamics run is a near-copy of it, already built and run (63.5 / 87.8) | CM | 2.3 h each | **not ours**; the baseline row | gap sweep + `winner` in the ablation batch |
| 4 | Optical-flow channel [frame, diff, flow magnitude] | 2025 winner tried it; differences beat flow in their ablation | CM | low: Farneback on CPU (hours) + 2.3 h | none | low priority, last in the batch |
| 5 | v7 refiner, centre-keypoint variant (Gaussian centre on the crop, size at the peak) | CenterNet (your idea 3); our own heatmap head does this | LB | medium, ~1 day | low as a design, fine as a refiner variant | only if the plain v7 refiner shows the box problem is fixable |
| 6 | Box-loss weight up in YOLO (`box=15`); CIoU instead of GIoU in the refiner | (your idea 4) | LB | one flag, 2.3 h | ablation | one run each, not a version |
| 7 | Per-sequence test-time adaptation (re-estimate normalisation statistics per sequence) | TENT / SATATrack | both, mildly | cheap, inference only | low-medium | ablation if time |
| 8 | Annotation-noise check: how jittery are Anti-UAV410's own boxes on 10-px drones? | label-quality work | tells us the hard ceiling of mAP75 | 1 h script | not a contribution; can save a week | do now |
| 9 | Distillation from a large detector into YOLO11n | knowledge distillation | efficiency claim | ~2 days | low | only if the efficiency claim is needed |
| 10 | Mid-level feature fusion of appearance and motion inside one detector | MI-DETR, SSTNet | both | high (days) | low; the field's current direction | no |
| 11 | **Sequence decoding of candidate boxes with an absent state** (Viterbi / HMM, short fixed lookahead to stay online); smooth paths through low-score boxes can win | speech pitch tracking (pYIN, Mauch & Dixon 2014; Taylor & Milner 2013) = radar DP track-before-detect; the concrete recipe for idea 1 | CM, flicker | medium, CPU on saved raw boxes (10 per frame at score >= 0.001), ~2 days | high on this benchmark | **candidate temporal contribution**; first re-measure the ceiling on the full candidate pool |
| 12 | Presence hysteresis + hangover (higher threshold to start, lower to continue, hold a few frames) | VAD (G.729B, WebRTC) | flicker | very cheap, post-processing chosen on val | low, honest engineering | fold into 11 |
| 13 | Posterior smoothing along a track before thresholding, lockout after a trigger | keyword spotting (Chen, Parada, Heigold 2014) | flicker | same pass as 12 | low | fold into 11 |
| 14 | Delta / delta-delta motion channels (regression-based first and second temporal derivatives) as the principled form of the lag sweep | MFCC deltas (Furui 1986) | CM (slow / accelerating drones) | 2.3 h per run | modest; clean ablation story | add to the input ablation |
| 15 | Shift-and-add frames along a hypothesised velocity (coherent integration) | delay-and-sum beamforming; "synthetic tracking" in astronomy | CM (frames where no detector box exists at all) | medium-high | medium | only if the ceiling shows complete misses with no box at any score |
| 16 | Randomly blank the motion channels for spans of training images | SpecAugment | robustness when the drone hovers | cheap at view-building time | low | if time |
| 17 | Running-minimum background estimate before differencing | minimum-statistics noise tracking (Martin 2001) | CM | cheap, input only | low; camera motion limits it | no for now |
| 18 | Self-supervised pretraining by predicting future frame features on the 440k unlabeled frames | CPC / wav2vec (Oord 2018) | all, indirectly | days | medium | too slow for 16 Nov |

**Backlog status, 21 Oct (after the scoring fix; see CHANGES sections 24-25).** Run and closed: 1/11/12/13
decoder (+0.2 once thresholds are fair), 2 time surface (worse), 3 gap sweep (8-12 best), 4 flow (no gain),
5 refiner (no gain), 6 box-loss weight (worse), 8 label noise (labels consistent to 0.90 IoU). Running now
(`dgx/run_ideas.sh`, round 14): 14 delta/delta-delta input, 16 motion-channel blanking, 17 background
subtraction, 7 test-time augmentation, plus input resolution 960. Not run: 15 shift-and-add (only if 14 and 17
fail), 18 self-supervised pretraining (too slow), 9/10.

**Literature review, 21 Oct (tiny objects / dynamic background / thermal crossover), what transfers.**
| Problem (our loss) | Idea in the literature | Adapted to our framework | Cost |
| --- | --- | --- | --- |
| Loose boxes on tiny drones (~20 pts) | IoU is unstable for 10-px boxes; NWD (Wang et al. 2022) and SimD (arXiv 2407.02394) replace it in label assignment and loss (+4 AP on very tiny objects) | NWD/SimD assignment and box loss inside YOLO11n (patch the assigner and BboxLoss) | 1 day code + 1 run |
| Tiny drones lost by stride-8 features | P2 (stride-4) detection head, lossless downsampling (LR-Net); PConv itself was built on yolov8n-p2 | yolov8n-p2 / yolo11-p2 variant with the gap-k input | 1 run, ~2x FLOPs |
| Dynamic background, camera motion (DBC 7% of frames, 74% found) | Decoupled global vs local motion (arXiv 2606.15286, +4.5 mAP50 on dynamic IRDST); deformable alignment + motion-compensation loss (DFAR, PR 2026) | Ego-motion-compensated differencing at the input: estimate the global background motion between t-k and t (homography / phase correlation at low resolution), warp t-k, then difference. Cheap cousin of both papers | ~10 ms/frame + 1 run |
| Thermal crossover (IC 7%, 60% found), dim targets | DeepPro (arXiv 2506.12766): detection as per-pixel 1-D temporal anomaly; 0.2M params, 185 fps, +4 Pd on SNR<=3; needs a static background | A temporal-profile channel: robust z-score of the current pixel against its own (compensated) history over T frames, replacing |diff t-k| when combined with compensation; plus a local-contrast channel (LCM-style) | 1 run each |
| Background modelling | Low-rank + sparse (TenRPCANet, arXiv 2509.07654) | Too heavy; cite only | - |
Proposed framework (if the pieces work): compensated temporal evidence at the input (short gap, long gap,
temporal anomaly, local contrast) + a tiny-object head (P2 + NWD) + the decoder/box smoothing. Order of
attack: compensation first (cheap, may lift everything), then NWD, then P2, then the profile channel.

## FINAL PLAN (decided 29 Sep 2026; overrides sections 4 and 6 where they differ)

**Datasets, checked against the official sources:**

| Role | Dataset | Why | Status |
| --- | --- | --- | --- |
| Main benchmark | **Anti-UAV410** (TPAMI 2023): 410 IR videos, 200 train / 90 val / 120 test (verified from the files) | Test labels are public and complete (all 120 test sequences, with per-frame attributes); raw outputs of 17 retrained trackers are downloaded and scored with our own code, no retraining | Ready: `Akshita_Projects/antiuav410/frames` in our layout |
| Development set for ablations | **Anti-UAV300** (our current data): 318 videos, 160 / 67 / 91 | Already extracted; ablations run here on GPU 2. **Not a second independent dataset**: 150 of Anti-UAV410's 200 train videos and 88 of its 120 test videos are Anti-UAV300 videos | Ready |
| Not used as a benchmark | 4th Anti-UAV challenge 2025 | Test labels are hidden (the winner had to build its own offline val set) and the scoring server closed in March 2025, so no comparable test number is possible. We still cite its leaderboard and run its winning code as a baseline on Anti-UAV410 | — |
| Dropped | NPS-Drones / ARD100 (RGB) | No time; only if everything else is done by 2 Nov | — |

**Task setting.** Our method is a *detector that needs no first-frame box* (this is the challenge's Track 2 setting, "Detection & Tracking"). Trackers in the Anti-UAV410 table get the first-frame box; we list them in a separate block labelled "given first-frame box" so the comparison is honest.

**Overlap, measured by sequence name (29 Sep).** Anti-UAV410 is essentially a superset of Anti-UAV300 with the same split assignment: its test set has 88 Anti-UAV300 test videos, 2 Anti-UAV300 val videos and **zero** Anti-UAV300 train videos. Consequences:
- Our Anti-UAV300 checkpoint may be fine-tuned on Anti-UAV410 train: no test leakage. This saves most of a full training run. (4 Anti-UAV300 train videos sit in Anti-UAV410 val; negligible for checkpoint selection, mention it in the paper.)
- Never train on Anti-UAV300 val or test for an Anti-UAV410 number (they overlap Anti-UAV410 test).
- Anti-UAV300 is reported only as a development set, not as evidence of generality. If a second dataset is needed for generality, use a different source (CST Anti-UAV, 2025, or NPS-Drones), time permitting.

**Baselines:**

| Group | Methods | Retrain? | Where the number comes from |
| --- | --- | --- | --- |
| Trackers (free) | 17 trackers retrained on Anti-UAV410 (SiamDT, MixFormerV2, DropTrack, SwinTrack, ToMP, Stark, AiATrack, ...) | No | Official raw outputs, scored with our `score_predictions.py` |
| Dense detectors (efficiency curve) | DEIM + D-FINE (CVPR25/ICLR25) N and S; YOLO11 n and s; at 448 and 640 | Yes, on Anti-UAV410 train | Our scorer |
| Anti-UAV SOTA | 2025 challenge winner (Frame Dynamics detector, CVPRW25) | Yes, on Anti-UAV410 train | Our scorer |
| Video detector | YOLOV++ online (IJCV 2026) | Yes | Our scorer |
| Cited only | Gaze-DETR 87.06 mAP@50 on Anti-UAV410 (arXiv Jul 2026) | Try their code; else cite | Their paper |
| If time | CHAL (CVPR26), MOCID (AAAI25) | Yes | Our scorer |

**GPUs.** Laptop (RTX 4090 16 GB): our model only. GPU 2: baselines and ablations.

**Milestones.**

| Date | Milestone |
| --- | --- |
| 1 Oct | Anti-UAV300 stage 1 done (fix already confirmed: heatmap peak within 5 px 72% after 3 epochs, was 0.2%) |
| 29 Sep (done) | Anti-UAV410 downloaded, verified and converted; 17 retrained trackers scored on its test set |
| 5 Oct | Baseline repos train 1 epoch on Anti-UAV410 (GPU 2) |
| 2–15 Oct | Our model on Anti-UAV410: start from the Anti-UAV300 stage 1 checkpoint, fine-tune stage 1 on Anti-UAV410 train, then stages 2–4 (seed 42); ablations A0–A10 on Anti-UAV300 on GPU 2 |
| 12 Oct | Gate 1: our Anti-UAV410 stage 1 heatmap and recall look healthy |
| 20 Oct–2 Nov | Seeds 2 and 3 on Anti-UAV410; baselines trained and scored; Anti-UAV300 full model finished |
| 2 Nov | Gate 2: are we on the accuracy-vs-GFLOPs Pareto front on Anti-UAV410? If no, retarget to the CVPR 2027 Anti-UAV workshop |
| 3–9 Nov | Freeze tables, latency on real hardware, figures |
| 10 Nov | Register paper |
| 16 Nov | Submit |

---

## 1. Where we are

The pipeline trains end to end and reaches 67.0% mAP@50 on the infrared validation clips, but the
motion heatmap that the whole paper is built around is not actually working, and the paper's
comparison and ablation tables are not backed by any run in the results folder. Both must be fixed
before a CVPR submission.

**The idea in one paragraph.** Instead of running a heavy detector on the full 448x448 frame, the
model (1) subtracts local background motion with a parameter-free filter (LDMI), (2) runs a tiny CNN
to get a heatmap of "where something moved", (3) cuts 8 small crops at the heatmap peaks plus fixed
edge and grid positions, (4) encodes each crop, fuses the last 5 frames with a causal transformer,
(5) routes through a sparse top-2 mixture of experts, and (6) predicts a box per crop. A simple
tracker feeds the next frame's crop positions.

**What the results folder shows** (run finished 2026-09-05, infrared only, clip stride 24, best
checkpoint at finetune epoch 9):

| Stage | mAP@50 (%) | mAP@75 (%) | Recall (%) | Tracking acc. (%) | Heatmap peak within 5 px (%) |
| --- | --- | --- | --- | --- | --- |
| Stage 1 (detector, 80 ep) | 28.7 | 4.2 | 35.3 | 14.1 | 1.0 |
| Stage 2 (temporal, 30 ep) | 60.0 | 18.0 | 65.0 | 38.0 | 0.2 |
| Stage 3 (MoE, 20 ep) | 60.3 | 16.7 | 65.4 | 37.7 | 0.2 |
| Finetune best (ep 9 of 10) | 67.0 | 22.9 | 71.8 | 44.0 | 0.2 |

**Three things the numbers tell us.**

1. **The motion heatmap is dead.** The heatmap peak lands within 5 px of the drone in only 0.2% of
   frames, and the heatmap training loss fell to 0.00001 by epoch 3 of stage 1. That is not
   "solved", it is "collapsed": the loss had a bug (section 3) that rewards an all-zero heatmap.
   So the detections come from the fixed edge and grid crops and the large 4x crop scale, not
   from motion guidance. The paper's main claim is currently not what the model does.
2. **Tables 3 to 6 and the 32.4 FPS Jetson number have no matching run**, log or script output in
   the results folder. The ablation script evaluated untrained models, so it cannot have produced
   those numbers. These tables have to be produced for real or removed.
3. **Several paper numbers do not match the code.** The checkpoint has 6.97 M parameters, not
   44.8 M. The MoE experts are 2.1 M of those and the MoE layer is 0.008 GFLOPs, so top-2 routing
   cannot give a "60% compute reduction". The model was trained on infrared frames (1 channel)
   with LDMI scales 7/15/31/63 plus a Sobel channel; the paper describes RGB with scales 15/31.
   Two of the eight experts receive about 1% of tokens, so "balanced utilization" is overstated.

**What is genuinely good.** Training is stable across all four stages, the temporal transformer
stage gives a real jump (28.7 to 60.0 mAP@50), the code base is clean and modular with unit tests,
and the crop-based design is cheap: about 7 M parameters and 4.3 GFLOPs per frame. That is a solid
base to fix and build on.

---

## 2. Reframing the paper for the CVPR main track

**Short answer: yes, the paper needs a new frame.** The current draft is a systems paper with five
"innovations" and a claim to beat state of the art. Neither survives review:

- Five loosely tied components read as a bag of tricks. CVPR wants one claim, tested from several
  sides.
- "Beats SOTA" is not true. Gaze-DETR (arXiv, July 2026, code public) reports 87.06 mAP@50 on
  Anti-UAV410; the Anti-UAV410 benchmark lists 50+ trackers. Our 67.0 is on a subsampled
  validation set and will likely drop under the honest protocol.
- The MoE claim is hollow (0.6% of compute). The LDMI "theorem" restates a well-known fact (camera
  rotation gives locally uniform flow) and local-mean subtraction of frame differences is a classic
  background-suppression trick; calling it a mathematical guarantee invites attack.

**The frame that can work: "where to look, not how deep".** The defensible, testable and novel claim
is about *budgeted attention*: for tiny targets on edge hardware, a fixed budget of K small crops,
scheduled across motion evidence, spatial priors (edges, periodic grid) and tracker feedback, with a
causal per-crop temporal memory, gives a better accuracy-per-GFLOP than dense detectors, and
degrades gracefully as the budget shrinks. Working title:

> *Where to Look, Not How Deep: Budgeted Causal Glimpse Detection for Tiny UAVs from Moving Sensors*

**The three claims the paper makes, and the evidence each needs.**

| Claim | Evidence to produce | Main figure or table |
| --- | --- | --- |
| A. A K-crop budget beats dense detection at equal compute | Ours at K = 4, 8, 16 and crop scales, versus DEIM/D-FINE, YOLO11 and YOLOV++ at several input resolutions and model sizes; all on our split, our evaluator | Figure 1: mAP@50 versus GFLOPs and versus Jetson latency (Pareto curve) |
| B. Where the budget goes matters | Ablation over crop sources (motion, edge, grid, guided, random, dense fallback) and over the scheduler; the new per-source attribution metrics | Table: source ablation with "true positives by source" |
| C. Causal per-crop memory is what makes glimpses work | Temporal window 1/3/5/9, with and without slot alignment across frames, with and without tracker feedback; the stage-2 jump (+31 mAP) is the seed of this evidence | Table plus a qualitative figure of aligned crop slots over time |

**What moves out of the headline.** The MoE goes to the ablation table (keep it only if it helps
accuracy; otherwise remove it and say so). The subpixel offset head becomes one line. LDMI becomes
"a parameter-free ego-motion suppression prior" with an empirical comparison to homography
alignment and to raw frame differences, not a theorem. The motion heatmap is one proposal source
among several, so the paper does not stand or fall on it, although the fix in section 3 should
make it a strong source.

**Related work the new frame must cite** (reviewers will check): glimpse and zoom-in detection
(Dynamic Zoom-in Network, CVPR 2018; ClusDet, ICCV 2019; Uzkent and Ermon, WACV 2020; AdaZoom),
streaming perception (StreamYOLO, CVPR 2022), video detection with causal memory (YOLOV++),
Anti-UAV benchmark line (Anti-UAV410 and SiamDT, TPAMI 2023; 4th challenge winner, CVPRW 2025;
Gaze-DETR), moving infrared small target detection (DTUM, MOCID, CHAL CVPR 2026), and
drone-to-drone detection from moving cameras (Dogfight, CVPR 2021; TransVisDrone, ICRA 2023).

**Two datasets, not one.** CVPR reviewers expect generality. Primary: Anti-UAV410 official
train/val/test split (infrared, per-attribute numbers for free). Secondary: NPS-Drones or ARD100
(RGB, moving camera) so the "moving sensor" claim is shown in RGB too. If time runs out, the second
dataset can be evaluation-only with a model trained on the first.

**Go/no-go for the main track.** By 2 November we need: ours on the Pareto front (claim A) on at
least one dataset, honest protocol, three seeds. If it is not, submit to the CVPR 2027 Anti-UAV
workshop (deadline usually March) with the same paper and more seeds. That is a good outcome; a
rejected main-track paper with weak baselines is not.

---

## 3. What must improve (ordered by how much a reviewer cares)

Items 1 to 4 are must-do; without them the paper is rejected on soundness. Items 5 to 9 make it a
good paper.

1. **Make the motion heatmap actually learn (bug fixed, retrain stage 1).** The heatmap loss marks a
   pixel as "positive" only where the target map equals exactly 1.0, but the Gaussian target was
   drawn at a fractional centre, so no pixel ever reached 1.0. The loss therefore had zero
   positives and only pushed the heatmap towards zero. Fix (done, see section 5): draw the Gaussian
   at the rounded integer pixel so the peak is exactly 1.0, as CenterNet does, and let the offset
   head handle the rounding error. After retraining stage 1, check that `heatmap_peak_within_5px`
   rises from 0.2% to well above 50% before touching anything else.
2. **Evaluate the honest way: whole sequences, streaming, official metric.** Today the evaluator
   scores only the last frame of every 5-frame clip, samples clips every 24 frames, never uses the
   tracker, and computes the Anti-UAV tracking accuracy over that subsample. Fix (script added):
   run every sequence frame by frame through the streaming path with the tracker on, compute the
   official accuracy per sequence, average. Report on the official test split and add per-attribute
   numbers (tiny, small, occlusion, dynamic background) from the Anti-UAV410 annotations.
3. **Replace the unverified tables with real runs.** Baselines, ablations and FPS must come from
   logged runs that live next to the checkpoints. Three seeds for the main model, one per ablation.
4. **Fix the compute claims.** Measured: 6.97 M parameters, 4.28 GFLOPs per frame in streaming
   mode (MotionCNN 1.68, crop encoder 2.58, MoE 0.008). Sparse routing saves 0.6% of compute.
   Measure FPS on a real Jetson Orin Nano with TensorRT FP16; if no Jetson is available, report GPU
   and CPU latency and say so.
5. **Let the detector use the crops it was designed for.** The box head predicts the centre through
   a sigmoid, so a box can only land within +/-32 px of the crop centre, even though the encoder
   also sees the 2x and 4x crops. The training assignment marks the nearest crop positive even when
   the drone is 150 px away, which teaches the head to fire on crops that cannot localise. Two small
   changes: only make a crop positive when the drone lies inside its 64 px window (otherwise ignore
   it in the loss), and let the centre regress over the largest crop scale. Then measure, with the
   new per-source report, which crop type each true positive comes from.
6. **Give the motion gate a real target.** Its loss is `1 - confidence`, so it learns to always say
   "confident" and never switches to dense mode. Supervise it with whether the heatmap's top peak is
   within a small radius of the drone.
7. **Use the tracker at evaluation.** Guided crops were never used in evaluation, so "tracker
   feedback" was untested. The new evaluator turns it on, and off for the ablation row.
8. **Address the MoE routing.** Experts 0 and 7 get about 1% of tokens. Try 4 experts or a higher
   balance weight, and report the ablation either way; drop MoE if it does not help.
9. **Make the paper match the code.** Infrared, 1 channel, LDMI scales 7/15/31/63 plus Sobel,
   16-channel motion input, encoder unfrozen in stage 4 at learning rate 2e-6, 6.97 M parameters.
   One method name throughout. Anonymised author block.

---

## 4. Baselines to run

Run the six "must" rows on exactly our split and protocol, with their published code and
recommended training recipe, and keep the run logs next to ours. All links were opened on
2026-09-28.

| Priority | Method | Venue, year | What it is | Why it is the right comparison | Code |
| --- | --- | --- | --- | --- | --- |
| Must | A Simple Detector with Frame Dynamics is a Strong Tracker | CVPR 2025 Anti-UAV Workshop, best paper, 1st place Track 1 | Detector fed with frame difference and optical flow plus trajectory filtering | Same task, same data family, the current winner on infrared UAV tracking | https://github.com/facias914/A-Simple-Detector-is-a-Strong-Tracker |
| Must | SiamDT on Anti-UAV410 | TPAMI 2023 | Official benchmark tracker, 50+ re-trained trackers already scored | Free, credible comparison on the official test split with per-attribute numbers | https://github.com/HwangBo94/Anti-UAV410 |
| Must | CHAL | CVPR 2026 | Moving infrared small target detector; background modelled as a spatio-temporal field, targets as anomalies | Newest top-venue method on our problem class; reports mAP50, precision, recall, F1 | https://github.com/UESTC-nnLab/CHAL |
| Must | MOCID | AAAI 2025 | Motion context and displacement learning for moving infrared small targets | Closest published idea to our motion-displacement loss | https://github.com/TanzanOY/MOCID |
| Must | DEIM with D-FINE | CVPR 2025 (DEIM), ICLR 2025 (D-FINE) | State-of-the-art real-time single-frame DETR | The "strong single-frame detector" row and the dense side of the Pareto curve; run at 3 sizes and 3 resolutions | https://github.com/Intellindust-AI-Lab/DEIM |
| Must | YOLOV++ | IJCV 2026 (YOLOV++), AAAI 2023 (YOLOV) | Video object detector with an online (causal) mode | The "strong video detector" row; fair causal comparison | https://github.com/YuHengsss/YOLOV |
| Should | Gaze-DETR | arXiv July 2026 | Priority-map guided DETR for infrared weak-small UAV detection; 87.06 mAP50 on Anti-UAV410 | Sets the accuracy bar; its priority map is close to our heatmap idea | https://github.com/nliu-25/Gaze-DETR-Top-Down-Guidance-Through-Priority-Maps-for-Infrared-Weak-Small-UAV-Detection-with-DETR |
| Should | TransVisDrone | ICRA 2023 | Spatio-temporal transformer for drone-to-drone detection from moving cameras (NPS-Drones, FL-Drones, AOT) | Lets us show the method on RGB moving-camera data | https://github.com/tusharsangam/TransVisDrone |
| Should | DTUM | TNNLS 2023 | Standard multi-frame infrared small target baseline | Almost every 2024 to 2026 paper in this line compares to it | https://github.com/TinaLRJ/Multi-frame-infrared-small-target-detection-DTUM |
| Should | Ultralytics YOLO11 (n/s/m) | software, 2024 | Standard single-frame detector family | Cheap extra points on the dense side of the Pareto curve | https://github.com/ultralytics/ultralytics |

**Rules for a fair table.** Same 448x448 input (plus higher resolutions for the Pareto curve), same
split, same evaluation script (ours). Parameters, GFLOPs and latency for every row from one script on
one machine. Say which rows are trackers (need the first-frame box) and which are detectors.

---

## 5. Code changes made on 2026-09-28 and why

Full detail with commands: `CHANGES_2026-09-28.md`. None of these change the architecture; existing
checkpoints still load. All unit tests pass except one pre-existing failure that needs OpenCV.

| Change | What was wrong | What changed | Why it matters |
| --- | --- | --- | --- |
| Heatmap target peak = 1.0 (`stage_losses.py`, `motion_cnn.py`, `focal_loss.py`) | Gaussian drawn at a fractional centre, so no pixel hit 1.0 and the focal loss had zero positives; heatmap collapsed to zero | Centre rounded to the integer heatmap pixel (CenterNet style); offset head keeps the sub-pixel part; loss accepts targets within 1e-4 of 1.0 | Reason "heatmap peak within 5 px" is 0.2%. Stage 1 must be retrained. New tests in `tests/test_heatmap_targets.py` |
| Evaluator reports crop-source attribution (`evaluator.py`) | Nothing measured whether detections came from motion peaks or fixed crops | New metrics: share of top boxes and of true positives per source; how often the drone lies inside any 64 px crop window | Turns the "motion-guided" claim into a number |
| New sequence-level streaming evaluator (`drishti_v2/experiments/run_sequence_eval.py`) | Old evaluator: one frame per clip, every 24 frames, no tracker | Every frame of every sequence, causal streaming, tracker on (or `--no-tracker`), per-sequence metrics averaged, CSV and JSON, FPS | The protocol reviewers expect. Test in `tests/test_sequence_eval.py` |
| New parameter and FLOP script (`scripts/count_flops.py`) | Paper numbers came from nowhere | Params per module, GFLOPs per streaming frame, dense-MoE / no-LDMI / 16-crop-fallback variants, latency; gate mode pinned so counts are deterministic | Real numbers: 6.97 M params, 4.28 GFLOPs, MoE saving 0.6%, 16-crop fallback 6.88 GFLOPs |
| Ablation script loads a trained checkpoint (`run_ablation.py`) | Evaluated random-weight models | `--checkpoint` loads all matching tensors; variants needing a retrain are flagged `retrain_required` | Inference-time ablations are valid; the rest are clearly marked |
| `train_full.py` learning rates | Every stage used 1e-4 | Per-stage 1e-4 / 5e-5 / 1e-5 / 2e-6, epochs 80 / 30 / 20 / 10, matching the logged run | Prevents a wrecked MoE and fine-tune stage |

**Recommended but not made** (they change training behaviour; decide after the stage 1 retrain):
gate crop positives on the drone being inside the crop window, regress the box centre over the
largest crop scale, give the motion gate a real supervision signal. Each is 10 to 20 lines in
`stage_losses.py`, `detection_head.py` and `pipeline.py`. Put each behind a config flag.

---

## 6. Week-by-week plan

Baselines run in parallel with our retrain from day one because they do not depend on it.

| Dates | Our model | Baselines and ablations | Paper |
| --- | --- | --- | --- |
| 29 Sep to 5 Oct | Stage 1 retrain with the fix; honest sequence eval of the old checkpoint; FLOPs script on the GPU box | Clone the six "must" baselines; one-epoch sanity run each on our frame layout | Outline with the new frame (section 2) |
| 6 to 12 Oct | Box, assignment and gate fixes behind config flags | Data prep for the second dataset (NPS-Drones or ARD100) | Related work draft |
| **12 Oct** | **Go/no-go on the heatmap: does `heatmap_peak_within_5px` clear 50%?** | | |
| 13 to 26 Oct | Full four-stage training, three seeds; K = 4 / 8 / 16 budget sweep | Train and evaluate baselines at 3 sizes x 3 resolutions (Pareto points) | Method section matching the code |
| 20 Oct to 2 Nov | Sequence eval on val and test, all seeds | Ablation retrains: no LDMI, no Sobel, no temporal, window 1/3/9, dense MoE, 4 experts, no edge, no grid, no motion, no tracker, random crops | Figure 1 (Pareto) and tables filled from JSON |
| **2 Nov** | **Go/no-go for main track: are we on the Pareto front with honest numbers?** | | |
| 2 to 8 Nov | Jetson Orin Nano TensorRT FP16 latency (mean and p95, 200 warm frames) | Freeze all tables | Full draft, limitations, anonymisation |
| 9 to 16 Nov | | | Register 10 Nov, submit 16 Nov, supplementary (code, per-sequence CSVs) 23 Nov |

## 7. Next steps for this week (checklist)

- [ ] Pull the new code, run `python -m pytest -q`, read `CHANGES_2026-09-28.md`.
- [ ] Run the sequence evaluator on the existing `finetune_best.pt` on val, tracker on and off:
      `python -m drishti_v2.experiments.run_sequence_eval --config configs/default.yaml --frames-root <frames> --split val --checkpoint <ckpt> --device cuda` (add `--no-tracker` for the second run). Send both `summary_*.json`.
- [ ] Retrain stage 1 only with the heatmap fix, same config. Watch `train_heatmap`: it should stay above 1 instead of falling to 0.00001. Send `stage1_eval_metrics.json`; the key number is `heatmap_peak_within_5px`.
- [ ] Run `python scripts/count_flops.py --config configs/default.yaml --checkpoint <ckpt> --device cuda` and save the output next to the checkpoints.
- [ ] Clone the six "must" baselines and confirm each trains for one epoch on our frames layout. Do not tune yet.
- [ ] Start a `RUNS.md` in the results folder: one line per run with the exact command, config, seed, checkpoint path and output JSON. Every number in the paper must trace to a line here.

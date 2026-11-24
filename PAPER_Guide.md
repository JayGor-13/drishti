# Paper writing guide: narrative, structure, results, and comparisons

Prepared 5 November 2026 from the supplied implementation and saved experiment summaries. This document is a writing blueprint, not a finished manuscript.

## 1. The central story

**Working title:** When the Camera Follows: Relative Motion for Efficient Infrared Drone Detection

**Research question:** How should a small detector use temporal information when the camera follows the target?

**Main argument:** Target-following cameras change the relationship between image motion and useful detection evidence. Background edges can move strongly while the drone remains nearly fixed in the image. Aligning the background before differencing, then combining short and long temporal intervals, can provide better temporal evidence to an otherwise unchanged detector.

**Method in one sentence:** Feed a standard YOLO11n the current thermal frame and two background-compensated absolute differences, at gaps 1 and 12, and train with a size-relative Gaussian Wasserstein similarity in assignment and box regression.

**Result in one sentence:** The final saved YOLO11n configuration achieves **70.30 ± 0.09 tracking accuracy across three seeds**, compared with **66.22** for the saved plain-frame baseline, while retaining **6.4 detector GFLOPs** and adding temporal preprocessing.

The paper should connect acquisition conditions, temporal evidence, and detector performance. Frame differencing, camera compensation, and NWD are established components; the contribution must come from their justified use and the evidence explaining when they help. Treat NWD as supporting supervision while its separate benefit remains uncertain.

## 2. Proposed contribution statements

Use these as the basis for the introduction:

1. **Temporal representation under target-following cameras.** We formulate the input problem in terms of target motion relative to the background, explaining why small image displacement alone is insufficient to characterize the temporal detection signal.
2. **A compact input for an unchanged detector.** We study background-aligned short- and long-gap differences within the existing three-channel input, alongside a size-relative Wasserstein assignment and regression variant.
3. **An empirical study of accuracy and evaluation.** We compare detector families, temporal encodings, component interactions, computational requirements, and the sensitivity of benchmark scores to presence thresholds.

After completing the motion analysis, contribution 1 can become a stronger statement about an *empirically characterized* failure mechanism. Until then, describe it as a formulation and hypothesis. Do not claim a new architecture, a new loss family, a proven optimal gap, or general superiority across anti-UAV benchmarks.

## 3. How to structure the paper

The page allocations below are a working eight-page writing budget, not a statement of venue rules. References are separate in this planning budget.

| Section | Approximate pages | What the reader should learn |
|---|---:|---|
| Abstract and introduction | 1.0 | The acquisition problem, proposed representation, main observed result, contributions |
| Related work | 0.5 | How this study relates to frame dynamics, camera alignment, infrared detection, and NWD |
| Understanding the temporal signal | 1.0 | Why relative motion and target scale matter; measured support for the hypothesis |
| Method | 1.0 | Exactly how frames become a detector input and how training changes |
| Experimental setup | 0.5 | Dataset, initialization, training, checkpoint selection, metrics and output policy |
| Results and analysis | 2.5 | Main comparisons, ablations, gap study, transfer, failures and efficiency |
| Evaluation sensitivity | 0.5 | How presence decisions affect the reported tracking metric |
| Limitations and conclusion | 0.5 | Scope, failure conditions and supported conclusions |
| Layout reserve | 0.5 | Captions, floats and final editing |

### 3.1 Abstract

Use this order: acquisition condition → temporal-input problem → representation → principal results → scope.

Include the three-seed YOLO11n tracking result, unchanged detector computation, and the fact that preprocessing is additional. Mention improved AP50 without implying improved AP75. Keep the decoder and long lists of baselines out of the abstract. Do not headline the 2.5 px/frame observation until its statistics are independently reproduced from the supplied labels.

### 3.2 Introduction

Write five connected paragraphs:

1. Explain why small infrared drones are difficult to detect and why inexpensive temporal evidence is attractive.
2. Introduce the following-camera condition and how it can make background differences stronger than target differences.
3. State the research question and distinguish image displacement from background-relative displacement.
4. Introduce background alignment and complementary temporal intervals within a fixed detector architecture.
5. Give the principal observed result and the three contributions, with scope proportional to the completed evidence.

An effective transition is: “The usefulness of temporal differences depends on how the video is acquired. We therefore examine temporal input construction while holding detector architecture fixed.”

### 3.3 Related work

Organize by relationship to the contribution:

| Area | What to discuss | How to position our work |
|---|---|---|
| Anti-UAV detection and tracking | Anti-UAV410 and initialized versus uninitialized settings | Separate detector comparisons from first-frame-box tracker references |
| Frame dynamics | Peng et al.'s input-level temporal representation | Closest input control; acknowledge it explicitly |
| Compensation and temporal differencing | Classical alignment and short/long temporal differences | Established tools; explain the acquisition-specific question we investigate |
| Moving infrared target detection | SSTNet, CHAL and other verified relevant work | Compare public-code experiments with their actual training budgets stated |
| Small-box supervision | NWD and related assignment/regression similarities | Our implementation adapts established supervision; it is not a new distance family |

Useful verified starting sources are [Peng et al., CVPR Workshops 2025](https://openaccess.thecvf.com/content/CVPR2025W/Anti-UAV/html/Peng_A_Simple_Detector_with_Frame_Dynamics_is_a_Strong_Tracker_CVPRW_2025_paper.html), [NWD](https://arxiv.org/abs/2110.13389), and the [official Anti-UAV410 repository](https://github.com/HwangBo94/Anti-UAV410). Expand and verify the bibliography before submission. The Peng paper reports first place in Track 1 and second in Track 2; avoid an ambiguous “Track-2 winner” label.

### 3.4 Understanding the temporal signal

This section should explain the method rather than merely repeat the result table. Measure target image displacement, estimated background displacement, and their difference. The approximate relative displacement over gap g is:

```text
relative displacement = current target center
                      − historical target center
                      − background translation over the same interval
```

Compare it with target size. Show whether alignment reduces background difference energy and whether the resulting signal helps detection. The previously reported 2.5 px/frame median and 21% below one pixel are image-space observations; they do not alone prove background dominance or explain an optimal compensated gap of 12.

Required evidence: split-wise distributions, target/background energy comparisons, and accuracy change against reliable background-motion estimates. Label these as pending until measured. Use training/validation to set analysis rules rather than tuning the recipe again on test.

### 3.5 Method

Present the implemented pipeline in this order:

1. **Input definition:** current thermal frame + compensated gap-1 difference + compensated gap-12 difference.
2. **Background translation:** quarter-resolution phase correlation with a Hann window and reliability checks.
3. **Temporal alignment:** accumulate consecutive translations and warp the historical frame to the current background.
4. **Difference channels:** absolute difference, gain 4, clipping, sequence-start handling, and zero-difference exposed borders.
5. **Supervision:** Gaussian box representation and size-relative similarity, with separate assignment and regression definitions.
6. **Objective and output:** state the retained classification/DFL terms and the top-score presence decision.

Explain why the architecture remains unchanged: the constructed image still has three channels. The training modification changes assignment and the regression similarity, not just one scalar loss term. Importantly, the current patch normalizes by predicted-box size in assignment and ground-truth size in regression.

Use the existing [equation-based method section](paper_materials/METHOD_SECTION.md) as the technical source. Keep the central equations in the main paper and move the full standard BCE/DFL expansion and implementation table to the supplement.

### 3.6 Experimental setup

Describe Anti-UAV410's recorded split: **200 / 90 / 120 sequences**, containing **213,995 / 94,711 / 129,691 frames**. Explain the intended 40-epoch YOLO recipe, COCO initialization, every-fourth-frame training sampling, input size, seeds, checkpoint policy and validation-selected postprocessing.

Do not describe all historical runs as identically trained: earlier stopping and checkpoint rules differ or lack provenance. Distinguish recorded settings from settings verified for each run. Verify every-frame prediction coverage using raw files; the current sweep loader can omit sequences whose outputs are missing.

Define tracking accuracy with its nonlinear visible-miss penalty and equal sequence averaging. Label the saved AP values **sequence-averaged AP50/AP75**; they are not standard dataset-global COCO AP. State that output NMS/top-1 can change AP's candidate set.

Disclose repeated test inspection during development. Validation-selected postprocessing is not equivalent to a completely untouched test set.

## 4. Main results and model comparisons

All scores below are percentages from saved validation-selected summaries. GFLOPs refer to the detector, excluding temporal input construction. Values reproduce existing runs; they do not establish equivalent historical training budgets. Except where a mean is shown, these are individual saved runs.

### 4.1 Our configurations and direct controls

| Configuration | Tracking accuracy | Sequence AP50 | Sequence AP75 | Detector GFLOPs |
|---|---:|---:|---:|---:|
| Plain YOLO11n, 640 | 66.22 | 84.16 | 43.31 | 6.44 |
| Signed short-gap input on YOLO11n | 67.00 | 86.01 | 41.43 | 6.44 |
| Final YOLO11n, seed 42 | 70.39 | 89.95 | 42.93 | 6.44 |
| Final YOLO11n, seed 1 | 70.30 | 90.50 | 43.18 | 6.44 |
| Final YOLO11n, seed 2 | 70.21 | 90.29 | 42.63 | 6.44 |
| **Final YOLO11n, three-seed mean ± sample SD** | **70.30 ± 0.09** | **90.25 ± 0.28** | **42.91 ± 0.27** | **6.44** |
| Plain YOLO26n, 640 | 66.33 | 85.57 | 43.83 | 6.03 |
| Recipe on YOLO26n, one seed | 70.76 | 90.58 | 43.53 | 6.03 |
| Recipe on YOLOv8n-P2 | 70.29 | 90.75 | 43.84 | 12.35 |

**How to discuss these results:**

- The observed final-versus-plain YOLO11n gain is approximately **4.09 tracking points**, with about **6.09 AP50 points**. These compare a three-seed final mean with a historical single-run baseline.
- Against the signed short-gap input control, the final mean is approximately **3.30 tracking points higher**. This is a comparison of input representations on our detector, not a reproduction of the predecessor's full system.
- AP75 changes from 43.31 to 42.91. Describe improved detection at moderate overlap, without claiming tighter localization.
- YOLO26n improves by approximately **4.43 tracking points** in its single-seed saved comparison. Call this transfer within evaluated detector families, not independent-benchmark generalization.
- The P2 configuration improves some AP values but approximately doubles detector computation. It does not improve tracking accuracy over the final YOLO11n mean.

### 4.2 Comparisons with other detector models

| Detector | Input size | Tracking accuracy | Sequence AP50 | Sequence AP75 | Detector GFLOPs |
|---|---:|---:|---:|---:|---:|
| YOLOv8n | 512 | 66.02 | 83.9 | 42.3 | 5.2 |
| YOLOv9t | 512 | 65.97 | 83.9 | 42.3 | 5.0 |
| YOLOv10n | 512 | 65.20 | 83.6 | 40.7 | 5.4 |
| YOLO11n | 512 | 65.43 | 83.3 | 41.5 | 4.1 |
| YOLO12n | 512 | 65.42 | 83.9 | 41.0 | 4.1 |
| YOLO26n | 512 | 67.20 | 86.0 | 43.6 | 3.8 |
| YOLO11n | 640 | 66.22 | 84.2 | 43.3 | 6.4 |
| YOLO11s | 640 | 66.14 | 85.1 | 43.2 | 21.5 |
| YOLO26n | 640 | 66.33 | 85.6 | 43.8 | 6.0 |
| RT-DETR-l | 512 | 65.41 | 84.7 | 35.1 | 72.1 |
| DINO-4scale R50 | Confirm from run configuration | 66.32 | 84.6 | 41.8 | Not recorded |
| Deformable-DETR R50 | Confirm from run configuration | 64.54 | 83.1 | 33.2 | Not recorded |
| **Final YOLO11n, three-seed mean** | **640** | **70.30** | **90.25** | **42.91** | **6.44** |

**How to discuss these results:** final YOLO11n exceeds the saved appearance-only detector scores, including larger YOLO11s and RT-DETR-l. This supports investigating temporal representation alongside model capacity. It does not establish that larger networks or additional data can never help. Input sizes differ across rows and must remain visible.

### 4.3 Comparisons with motion-aware methods

| Method or temporal representation | Tracking accuracy | Sequence AP50 | Sequence AP75 | Comparison meaning |
|---|---:|---:|---:|---|
| Signed differences to t−1 and t−2 | 67.00 | 86.0 | 41.4 | Closest predecessor's input reimplemented on YOLO11n |
| One difference plus optical-flow magnitude | 68.36 | 87.8 | 43.1 | Implemented input alternative on YOLO11n |
| Time surface | 67.75 | 87.2 | 42.1 | Implemented temporal-history input |
| Delta / delta-delta | 67.71 | 86.8 | 39.9 | Implemented polynomial temporal input |
| Median background subtraction | 68.57 | 88.1 | 41.4 | Implemented background-history input |
| Compensated gaps 1 and 8 | 69.88 | 90.0 | 42.9 | Alignment variant without the complete final recipe |
| SSTNet | 55.35 | 71.0 | 20.4 | Authors' code, trained from scratch on a shortened schedule |
| CHAL | 54.25 | 70.5 | 21.9 | Authors' code, trained from scratch on a shortened schedule |
| **Final YOLO11n, three-seed mean** | **70.30** | **90.25** | **42.91** | **Compensated gaps 1 and 12 plus NWD variant** |

SSTNet and CHAL have recorded costs of 123.6 and 334.1 GFLOPs respectively. Their historical training budgets differ materially from the pretrained YOLO setup. Present them as qualified reproduction attempts, not definitive estimates of their published capabilities. Likewise, an implemented optical-flow channel does not represent every possible flow-based detector.

### 4.4 Comparisons with initialized trackers

These methods receive the first-frame target box; our detector does not. Keep this table or block explicitly separate. Scores come from official raw tracker results evaluated by the local scorer, rather than retraining these trackers ourselves.

| Tracker | Tracking accuracy | Tracker | Tracking accuracy |
|---|---:|---|---:|
| SiamDT | 68.17 | ToMP | 55.08 |
| GlobalTrack | 66.43 | SwinTrack-Tiny | 52.20 |
| DropTrack | 60.13 | ATOM | 51.36 |
| MixFormerV2 | 59.63 | SiamBAN | 47.31 |
| AiATrack | 59.54 | SiamCAR | 46.93 |
| Stark-ST101 | 57.13 | KYS | 44.77 |
| KeepTrack | 56.80 | ROAM | 43.03 |
| DiMP | 56.65 | TCTrack | 41.64 |
| SwinTrack-Base | 55.65 | — | — |

Safe wording: “The final detector's saved tracking score is above the initialized tracker references under the local scorer, while the initialization conditions differ.” Avoid suggesting an identical task setting or superiority on every metric: SiamDT's sequence AP75 is 46.0, above our 42.9.

## 5. Ablations and what they actually establish

### 5.1 Component interactions

These are saved single-seed comparisons. Report validation alongside test, and do not describe every row as a cumulative gain under a fully matched protocol.

| YOLO11n configuration | Validation tracking | Test tracking | Test sequence AP50 | Test sequence AP75 |
|---|---:|---:|---:|---:|
| Plain | 71.30 | 66.22 | 84.16 | 43.31 |
| Plain + NWD | 71.01 | 66.50 | 85.20 | 43.51 |
| Gaps 1 and 12 | 73.20 | 69.59 | 88.65 | 44.06 |
| Gaps 1 and 12 + NWD | 73.26 | 69.20 | 88.44 | 42.21 |
| Compensated gaps 1 and 12 | **73.40** | 69.72 | 89.72 | 42.82 |
| Final combination, seed 42 | 73.35 | **70.39** | **89.95** | 42.93 |

Write the interpretation carefully: temporal input gives the clearest observed gain. At gap 12, compensation alone adds about 0.13 test points; NWD without compensation lowers the score. The complete combination performs best on test, while compensation without NWD is slightly better on validation. This motivates a component-interaction study and paired multi-seed controls, rather than a claim that every part independently improves performance.

### 5.2 Temporal interval

| Long gap, alongside gap 1 | Validation tracking | Test tracking |
|---|---:|---:|
| 3 | 72.92 | 68.24 |
| 5 | 72.95 | 68.97 |
| 8 | 73.02 | 69.18 |
| 12 | **73.20** | **69.59** |
| 16 | 72.90 | 69.16 |

The historical uncompensated sweep peaks at 12 on both splits. Explain it as an observed curve, not proof of a theoretical optimum. Retrieve the original selection record and match training schedules before claiming the gap was selected through an untouched protocol.

## 6. Evaluation sensitivity, efficiency, and negative experiments

### 6.1 Presence-threshold sensitivity

| Model | Saved default configuration | Validation-selected configuration |
|---|---:|---:|
| Plain YOLO11n | 58.28 | 66.22 |
| Final YOLO11n, seed 42 | 65.46 | 70.39 |
| RT-DETR-l | 62.04 | 65.41 |

Explain the miss penalty and the role of presence decisions. The saved configurations change several postprocessing settings together; isolate threshold-only behavior before assigning the whole difference to thresholds. Tracking already uses the highest-score box, whereas AP depends on the candidate set. Avoid calling a threshold “common in published work” without citations or recommending never abstaining for deployment.

### 6.2 Efficiency

State that final YOLO11n has approximately **2.59 million parameters and 6.44 detector GFLOPs**. The saved cost file reports approximately **10.95 ms** on an H200, but the timing loop includes temporal input construction and `model.predict`, with defaults differing from the explicit evaluation settings. This is not isolated GPU inference latency.

Do not add the old 7.56 ms preprocessing measurement to that value. Measure the actual final pipeline at batch 1, with consistent output settings, warmup, median/p95, hardware and memory, before reporting complete runtime or edge-device claims.

### 6.3 Optional decoder

Seed-42 final YOLO11n changes from **70.39 to 70.92**, and the YOLO26n recipe changes from **70.76 to 71.32**, with the lag-15 decoder. Keep these supplementary because they use 15 frames of future context. Do not combine a single-seed decoder score with the three-seed final mean or describe lag-15 inference as causal.

### 6.4 Negative experiments

Include a short discussion of the recorded alternatives: larger input sizes, every-frame training, learned refinement, zoom refinement and alternative temporal inputs. Report their actual settings in the supplement. Some variants improve certain scores slightly, so avoid saying that all alternatives failed. These experiments delimit the tested configurations; they do not prove that data, capacity or refinement are inherently unhelpful.

## 7. Figures and tables to prepare

| Artifact | Content | Status |
|---|---|---|
| Figure 1 | Real frame pair, raw/compensated differences and predictions; include a failure | Requires frames and predictions |
| Figure 2 | Target, background and relative displacement; accuracy gain versus camera motion | Analysis required |
| Figure 3 | Gap curves with validation/test and uncertainty where available | Historical values available; matched runs needed |
| Figure 4 | Final-versus-plain results by size/attribute and qualitative failure cases | Final-model analysis required |
| Figure 5, if space permits | Accuracy versus detector GFLOPs; preprocessing cost separately annotated | Saved network costs available |
| Main table | Direct controls, detector models, final mean and transfer | Saved results available |
| Ablation table | Fixed-gap compensation × NWD, with both splits and paired seeds | Single-seed rows available |
| Efficiency table | Parameters, detector FLOPs, verified pipeline latency and memory | Complete timing required |

Use the strongest evidence in the main paper. Move full tracker tables, all input variants, decoder results, expanded equations and detailed training configurations to the supplement. Do not reuse older-model attribute analyses as final-model results.

## 8. What must be completed before final claims

1. Freeze the current recipe and recover raw outputs, dataset coverage, exact environments and training provenance.
2. Run matched controls with paired seeds for plain, predecessor input, gap 12, gap 12 + compensation, gap 12 + NWD, and the final combination.
3. Reproduce the motion statistics and measure whether compensation helps under the predicted background-motion conditions.
4. Add final-model size, attribute, absence and failure analyses, plus complete runtime measurements.
5. Evaluate the frozen method on an independent dataset if available; documented Anti-UAV300 overlap prevents treating it as automatically independent.
6. Verify literature and bibliography, revise the abstract to match the final evidence, then compile and inspect the publication-format paper.

If NWD remains inconsistent, present it as an optional variant. If the motion analysis supports the mechanism, make that relationship central. The manuscript should follow the evidence rather than preserve a predetermined three-component novelty claim.

## 9. Writing order and result sources

Write method and experimental protocol first, then the principal results and ablations. Build the motion-analysis section when measurements exist. Write the introduction and related work around the supported interpretation, and write the abstract and conclusion last.

Sources for this guide:

- [All saved comparison tables](../results/TABLES.md): broad model, tracker, temporal-input and negative-experiment summaries.
- [Generated primary tables](paper_materials/RESULT_TABLES.md): primary comparisons, validation/test ablations and exact source map.
- [Result manifest](paper_materials/result_manifest.json): hashes of the primary result-summary sources.
- [All 63 selected run summaries](../results/AUDITED_RUNS.csv): detailed values for analysis.
- [Equation-based method](paper_materials/METHOD_SECTION.md) and [implementation correspondence](paper_materials/METHOD_IMPLEMENTATION.md): technical source for writing the method.

For detector scores, the source is `results/runs/<run>/postprocess_sweep_low.json`, falling back to `postprocess_sweep.json`, under `chosen_on_val.val` and `chosen_on_val.test`. Model cost comes from `cost.json`; trackers use `results/antiuav410_scores/<tracker>_test.json`. Differences should be calculated from unrounded JSON values, not rounded table cells. The existing summaries have not been independently recomputed from raw predictions in this workspace.

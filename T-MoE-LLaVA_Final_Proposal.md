# T-MoE-LLaVA 2.0: Event-Driven, Motion-Conditioned Sparse Architecture for Video LLMs

## Executive Summary
Following a rigorous multi-agent council debate, the original T-MoE-LLaVA proposal was found to contain critical theoretical and architectural flaws, specifically regarding naive camera motion handling, spatial misalignment in the CFCR loss, and catastrophic interference. This refined proposal overhauls the architecture into a **bio-inspired, event-driven temporal MoE** that decouples spatial semantics from kinematic dynamics, offering $O(1)$ compute for static video regions while remaining theoretically sound.

---

## 1. Cross-Agent Debate & Flaw Resolution

Before synthesizing the final architecture, the council addressed several critical vulnerabilities identified by the Adversarial Critic:

*   **Flaw 1: The "Camera Motion" Fallacy:** 
    *   *Critique:* Naive absolute frame differences ($\Delta = |f_t - f_{t-1}|$) fail during camera pans/zooms, falsely flagging the entire frame as dynamic.
    *   *Resolution:* Replaced with a lightweight 3D CNN (X3D-Tiny) or Ego-Motion Disentangled Optical Flow to capture true kinematic motion ($\delta^t$) robustly.
*   **Flaw 2: Spatial Misalignment in CFCR:**
    *   *Critique:* Penalizing the same spatial index $i$ across frames ignores object movement, forcing the router to discard semantic meaning.
    *   *Resolution (Theorist):* Introduced an attention-based alignment matrix $A_{i,j}$ into the CFCR loss to track shifting backgrounds, ensuring consistency is applied to the correct semantic patches.
*   **Flaw 3: Linear Bottleneck & Math Contradictions:**
    *   *Critique:* The term $(1 - \delta^t)$ conflates vectors and scalars, and linear addition $W_x x + W_\delta \delta$ washes out motion.
    *   *Resolution:* Defined a scalar motion confidence score $m_i^t = \sigma(W_m \delta_i^t)$. Upgraded the router to use concatenation $P_i^t = \text{softmax}(W_r [x_i^t \oplus \delta_i^t])$.
*   **Contradiction: Explicit vs. Implicit Experts:**
    *   *Debate:* Innovator suggested explicitly forcing experts to be "kinematic" or "semantic". The Architect and Theorist argued this requires unviable contrastive losses and violates MoE best practices.
    *   *Resolution (Chairman):* Adopted the **Architect's Implicit Experts** (8 experts, Top-2 routing) driven by a modality-aware router, allowing natural specialization via standard load-balancing without explicit forcing.

---

## 2. Final Synthesized Research Solution

### Core Idea
Transition from "incremental frame-difference routing" to an **Event-Driven, Motion-Conditioned Sparse Architecture**. The network uses motion $\delta$ to decide *where* to send a token, and a **Temporal Cache** to drop compute for static tokens entirely, mimicking event-based vision.

### Mathematical Formulation
1.  **Motion Representation:**
    *   Robust extraction: $\delta_i^t = \text{X3D-Tiny}(f_{t-w:t+w})_i$
    *   Motion Confidence Scalar: $m_i^t = \sigma(W_m \delta_i^t) \in [0, 1]$
2.  **Modality-Aware Router:**
    *   $P_i^t = \text{softmax}(W_r [x_i^t \oplus \delta_i^t])$
3.  **Refined CFCR Loss (Cross-Frame Consistent Routing):**
    *   $L_{cfcr} = \frac{1}{T \cdot N} \sum_{t=1}^{T-1} \sum_{i,j} A_{i,j} (1 - m_i^t) \cdot \text{JSD}(P_i^t \parallel P_j^{t+1})$
    *   *Note: JSD (Jensen-Shannon Divergence) prevents the vanishing gradients caused by L2 norm on the probability simplex. $A_{i,j}$ provides spatial tracking.*
4.  **Total Loss:**
    *   $L_{total} = L_{ar} + \alpha L_{aux} + \beta L_{cfcr}$

### Architecture Design
*   **Two-Pathway Input:** Visual tokens $V^t$ from CLIP-Large, and Motion embeddings $\delta^t$ from X3D-Tiny.
*   **Event-Based Temporal Token Cache:** If $m_i^t < \epsilon$ (motion is negligible), the router bypasses the MoE layer entirely and fetches the expert output for that spatial patch from frame $t-1$.
*   **LLM Backbone:** 8 Implicit Experts using Top-2 Routing. The router is modality-aware, applying the motion condition only to video tokens, while text tokens use standard sparse routing.

### Training Strategy
*   **Stage I (Modality Alignment):** Freeze LLM and CLIP. Train only the X3D-Tiny motion encoder and projection heads on large-scale image/video-text pairs.
*   **Stage II (Dense Upcycling):** Train the model as a dense network on general instruction tuning datasets (e.g., LLaVA-Instruct, Video-ChatGPT) using a low learning rate to preserve linguistic capabilities.
*   **Stage III (MoE Init & CFCR):** Initialize MoE experts with dense MLP weights + noise. Apply a **linear warmup** to $\alpha$ and $\beta$ over 500 steps to prevent catastrophic interference between the CFCR consistency and MoE load-balancing.
*   **Stage IV (High-Quality Temporal Tuning):** Train on high-quality long-form temporal reasoning datasets (e.g., ShareGPT4Video, NExT-QA). *Crucial: Avoid fine-tuning on benchmark subsets to prevent data leakage claims.*

### Experimental Plan
*   **Baselines:** Compare against dense models with the same *total* parameters, and dense models with the same *active* parameters (e.g., Video-LLaVA).
*   **Ablation Studies:**
    *   Ablate the Modality-Aware Router vs. standard routing.
    *   Ablate CFCR ($\beta = 0$) to show its impact on temporal hallucination reduction.
*   **Efficiency Metrics:** Report FLOPs, Active Parameters, and Inference Latency (tokens/sec) to empirically prove the O(1) efficiency gains of the Temporal Token Cache.

### Novelty Justification
This proposal fundamentally rethinks video redundancy. While spatial MoEs ignore the temporal physics of video, T-MoE-LLaVA decouples *what* an object is from *what it is doing*. By combining the CFCR loss to stabilize routing with an Event-Based Temporal Cache to eliminate redundant compute, this architecture represents a paradigm shift in efficient, kinematics-aware Video LLMs.

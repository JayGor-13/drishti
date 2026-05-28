# T-MoE-LLaVA 2.0: Single-Frame Dry Run Mathematical Trace

This document provides a step-by-step mathematical trace of the data flow through the **Temporally-Conditioned Homogeneous Experts (T-CHE)** architecture. 

We trace a **single-frame dry run ($T=1$)** containing:
*   **Input**: One image $f_1$ and a text question $Q$ ("Describe this frame.").
*   **Target**: The autoregressive generation of the first answer token $y_1$.

---

## Stage 1: Input & Feature Extraction (I/O to Token Projection)

### 1. Visual Pathway
The raw input image $f_1 \in \mathbb{R}^{H \times W \times 3}$ is processed by a frozen visual encoder (CLIP-Large/336) to yield grid-based patch tokens.

$$\text{CLIP}(f_1) = Z^1 \in \mathbb{R}^{P \times C}$$

Where:
*   $P$ is the number of spatial patches (e.g., $24 \times 24 = 576$ patches for a $336\times336$ image with patch size $14$).
*   $C = 1024$ (the CLIP embedding dimension).

The patch tokens $Z^1$ are projected to the LLM's hidden dimension $d$ using a learned multi-layer perceptron (Visual MLP):

$$V^1 = \text{GELU}(Z^1 W_{proj, 1} + b_{proj, 1}) W_{proj, 2} + b_{proj, 2} \in \mathbb{R}^{P \times d}$$

Where:
*   $W_{proj, 1} \in \mathbb{R}^{C \times 2d}$, $b_{proj, 1} \in \mathbb{R}^{2d}$
*   $W_{proj, 2} \in \mathbb{R}^{2d \times d}$, $b_{proj, 2} \in \mathbb{R}^{d}$
*   $d = 4096$ (LLM backbone hidden dimension).

---

### 2. Motion Pathway (Kinematics)
Because we are feeding a single frame ($T=1$), the temporal change is zero. We compute the motion tokens using a dummy/zero initialization or identical inputs.

$$\Delta_{1, 1} = |f_1 - f_1| = \mathbf{0} \in \mathbb{R}^{H \times W \times 3}$$

The lightweight motion encoder (X3D-Tiny) extracts spatial motion tokens:

$$\text{X3D-Tiny}(\Delta_{1, 1}) = M^1 = \mathbf{0} \in \mathbb{R}^{P \times M}$$

Where $M = 256$ is the motion channel dimension.

The motion tokens $M^1$ are projected to the LLM hidden dimension $d$ using a linear projection head:

$$\delta^1 = M^1 W_{mot\_proj} + b_{mot\_proj} = \mathbf{0} \in \mathbb{R}^{P \times d}$$

Where:
*   $W_{mot\_proj} \in \mathbb{R}^{M \times d}$
*   $b_{mot\_proj} \in \mathbb{R}^{d}$

**Motion Confidence Score ($m_i^1$):**
For each patch index $i \in \{1, \dots, P\}$:

$$m_i^1 = \sigma(W_m \delta_i^1 + b_m)$$

Since $\delta_i^1 = \mathbf{0}$, if the bias is initialized to a large negative value $b_m = -5.0$:

$$m_i^1 = \sigma(-5.0) \approx 0.0067 \to 0 \quad (\text{classified as static})$$

---

### 3. Text Prompt Encoding
The text question $Q$ is tokenized into $N_{text}$ discrete IDs, which are mapped through the LLM word embedding lookup matrix $E_{emb}$:

$$T_{text} = \text{Embedding}(Q) \in \mathbb{R}^{N_{text} \times d}$$

---

## Stage 2: Sequence Assembly & Position Embedding

All tokens are concatenated into a unified 1D sequence $X^{(0)}$:

$$X^{(0)} = [V^1 \parallel \delta^1 \parallel T_{text}] \in \mathbb{R}^{(P + P + N_{text}) \times d}$$

For each token $x_j^{(0)}$ at sequence position $j$:

$$x_j^{(0)} = X_j^{(0)} + PE_{spatial}(j) + PE_{temporal}(t)$$

Where:
*   $PE_{spatial}(j)$ is the spatial positional encoding based on patch coordinates.
*   $PE_{temporal}(t)$ is the temporal positional encoding (since $T=1$, $t=1$ for all visual and motion tokens; $t=0$ or a default placeholder for text tokens).

---

## Stage 3: LLM Backbone & MCR-MoE Layer Trace

The sequence passes through $L = 32$ stacked transformer layers. For an MoE layer $l$:

### 1. Attention Block
The tokens first undergo multi-head self-attention:

$$X^{(l-0.5)} = \text{SelfAttention}(\text{LN}(X^{(l-1)})) + X^{(l-1)}$$

Where:
*   $\text{LN}$ is Layer Normalization.
*   $X^{(l-0.5)} \in \mathbb{R}^{(2P + N_{text}) \times d}$.

---

### 2. Temporally-Aware Router Block
For each token $x_j \in X^{(l-0.5)}$ at index $j$:

**A. Temporal Context Vector Extraction ($c_j$):**
For $T=1$, there is no historical frame context ($t-k$ for $k > 0$ does not exist). Thus, the causal temporal context defaults to:

$$c_j = x_j$$

**B. Modality-Aware Router Routing Probability ($h$):**
The router projects the concatenated representation of the current token and its temporal context:

$$g_j = W_{g, 1} x_j + W_{g, 2} c_j \in \mathbb{R}^N$$

Where:
*   $W_{g, 1}, W_{g, 2} \in \mathbb{R}^{N \times d}$
*   $N = 8$ (Total number of experts).

The gating weights $h$ are computed via a softmax with temperature scaling $\tau$:

$$h(x_j, c_j) = \text{softmax}\left(\frac{g_j}{\tau}\right) \in \mathbb{R}^N$$

---

### 3. Event-Based Temporal Token Cache Bypassing
If $x_j$ is a visual token ($j \le P$) and its motion confidence is static ($m_j^1 < \epsilon$):
*   We check if a cached FFN output $FFN_{cache}(j)$ from the previous frame exists.
*   *Note: Since this is $T=1$ (the first frame), no cache exists yet. The token must be computed.*
*   We compute the MoE forward pass and cache the output for frame $t=2$:

$$FFN_{cache}(j) \leftarrow \text{MoE}(x_j)$$

If this were $T > 1$ and $m_j^t < \epsilon$, we would bypass the experts completely:

$$\text{MoE}(x_j) = FFN_{cache}(j)$$

---

### 4. Expert Execution
We select the indices $\mathcal{I}_j = \text{TopK}(h(x_j, c_j), K)$ where $K = 2$.

For each selected expert $i \in \mathcal{I}_j$, we execute its SwiGLU FFN:

$$E_i(x_j) = \left( \text{SiLU}(x_j W_{gate, i}) \odot (x_j W_{up, i}) \right) W_{down, i} \in \mathbb{R}^d$$

Where:
*   $W_{gate, i}, W_{up, i} \in \mathbb{R}^{d \times d_{ff}}$ ($d_{ff} = 14336$)
*   $W_{down, i} \in \mathbb{R}^{d_{ff} \times d}$
*   $\odot$ is the Hadamard (element-wise) product.

---

### 5. Combining Outputs
The output of the MoE layer for token $x_j$ is the weighted sum of the active experts' outputs:

$$y_j = x_j + \sum_{i \in \mathcal{I}_j} h_i(x_j, c_j) \cdot E_i(x_j)$$

The full sequence block output $X^{(l)}$ is formed by repeating this for all tokens $j \in \{1, \dots, 2P + N_{text}\}$.

---

## Stage 4: Autoregressive Decoder Output

After the final layer $L$:

$$X^{(L)} \in \mathbb{R}^{(2P + N_{text}) \times d}$$

We extract the final token representation (which corresponds to the last text token in the prompt sequence):

$$x_{last} = X^{(L)}_{-1} \in \mathbb{R}^d$$

This token is normalized and projected through the LM head to yield logits over the vocabulary $V$:

$$\text{logits} = \text{LN}(x_{last}) W_{lm\_head} \in \mathbb{R}^{|V|}$$

Where $W_{lm\_head} \in \mathbb{R}^{d \times |V|}$.

The probability distribution for the next token $y_1$ is:

$$p(y_1 \mid f_1, Q) = \text{softmax}(\text{logits}) \in \mathbb{R}^{|V|}$$

We sample the token $y_1$ (e.g., via argmax or nucleus sampling). For the next step ($y_2$), $y_1$ is appended to $T_{text}$, and the sequence length grows to $2P + N_{text} + 1$.

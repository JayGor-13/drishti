# Google Colab GPU & TPU Execution Guide: Micro-MoE

This guide explains how to set up, verify, and run the **Micro-MoE** training and hyperparameter experimentation sweep in a Google Colab notebook environment utilizing GPUs or TPUs.

---

## 1. Setting up Google Colab

### 1.1. Choose Runtime Hardware
1. In Google Colab, go to **Runtime** > **Change runtime type**.
2. Under **Hardware accelerator**, select:
   - **T4 GPU** or **A100 GPU** (Recommended for standard PyTorch CUDA execution).
   - **TPU v2** (For PyTorch XLA execution).
3. Click **Save**.

### 1.2. Uploading the Codebase
You can upload the `Micro-MoE` folder to your Google Drive and mount it in Colab:
```python
# Run this inside a Colab cell to mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Change directory to your uploaded Micro-MoE codebase
%cd /content/drive/MyDrive/Micro-MoE
```

---

## 2. Setting Up Python Environment & Verification

Colab comes pre-installed with PyTorch and Matplotlib. Ensure the python path is configured and run the unit test suite to verify everything functions:

```bash
# 1. Install/Verify pytest
!pip install pytest -q

# 2. Run pytest verification suite
!pytest
```

---

## 3. Running Hyperparameter Sweeps on GPU (CUDA)

Since the codebase is fully device-agnostic, the runner will automatically detect and place tensors on the selected GPU device when you pass `--device cuda`.

### 3.1. Execution Command
To run a complete sweep of all 15 configurations for 100 training steps each:
```bash
!python run_experiments.py --sweep --steps 100 --device cuda --output-dir colab_experiment_results
```

To run a single specific configuration (e.g., `Control`):
```bash
!python run_experiments.py --experiment Control --steps 100 --device cuda --output-dir colab_experiment_results
```

---

## 4. Visualizing Results Inline in Colab

After the run finishes, the script generates performance plots in the output directory. You can display them directly inside your notebook cells using the following Python code:

```python
from IPython.display import Image, display

# 1. Display Learning Curves (Total Loss & AR Loss)
print("--- Learning Curves ---")
display(Image("colab_experiment_results/learning_curves.png"))

# 2. Display Caching vs Loss Tradeoff (Pareto Frontier)
print("--- Caching vs Accuracy Pareto Frontier ---")
display(Image("colab_experiment_results/caching_vs_loss_pareto.png"))

# 3. Display Expert Routing Stability
print("--- Routing Dynamics ---")
display(Image("colab_experiment_results/routing_dynamics.png"))
```

---

## 5. Running on TPUs (Google Cloud TPU v2/v3/v4)

PyTorch uses the **PyTorch XLA** compiler to run models on Google TPUs.

### 5.1. Install PyTorch XLA in Colab
If your Colab notebook is running on a TPU runtime, install the PyTorch XLA compiler extension:
```bash
!pip install torch-xla -q
```

### 5.2. Running the Sweep on TPU
In PyTorch XLA, the TPU device is mapped to an `xla` device index. You can run the experiment runner by passing `--device xla`:
```bash
!python run_experiments.py --sweep --steps 100 --device xla --output-dir colab_tpu_results
```

*Note: For large scale training, PyTorch XLA recommends using the multiprocessing launcher (`xla_spawn.py` or standard `torch.distributed`). For our edge model scale (Micro-MoE), single-device XLA routing via `--device xla` works directly.*

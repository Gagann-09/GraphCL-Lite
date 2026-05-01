# 🚀 GraphCL-Lite: Self-Supervised Graph Neural Networks

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-EE4C2C.svg)
![TorchGeometric](https://img.shields.io/badge/PyG-2.7.0-3C2179.svg)

**GraphCL-Lite** is a compact, research-grade implementation of *contrastive learning* for Graph Neural Networks (GNNs). It provides a hands-on demonstration of how adding a self-supervised NT-Xent contrastive loss to a standard Graph Convolutional Network (GCN) significantly boosts semi-supervised node classification performance on the **Cora** citation network.

Dive into the world of Graph Contrastive Learning, complete with extensive evaluations, baseline comparisons, and a fully executable Jupyter Notebook!

---

## ✨ Key Features & Recent Updates

- **🌟 NEW: Enhanced GRACE Evaluation Suite** – Comprehensive benchmarking introducing **GRACE+Linear** and **GRACE+Finetune** baselines.
- **🌟 NEW: Alpha Sensitivity Analysis** – An experimental evaluation pipeline configured to iterate through specific alpha values (`0.1` through `0.9`) to systematically assess model performance sensitivity.
- **Robust Graph Augmentations** – Uses random feature masking and edge dropout to create robust contrasting views.
- **Dual-Objective Training** – Seamlessly combines supervised cross-entropy with NT-Xent contrastive loss (`L = α·L_cls + (1‑α)·L_con`).
- **Deep Visualisations** – Explore loss/accuracy curves, t-SNE embedding plots, graph sub-structures, and confusion matrices.
- **Production-Ready Validation** – Built-in dataset validation checks for NaNs/Infinities and enforces memory limits before training starts.
- **Ablation Studies** – Direct comparisons between baseline GCNs (no contrastive loss) and GraphCL-Lite.

---

## 🛠️ Tech Stack

| Category | Library | Version |
|----------|---------|---------|
| **Deep Learning** | PyTorch | 2.8.0 |
| **Graph Utilities** | torch-geometric | 2.7.0 |
| **Numerical** | NumPy | 2.4.3 |
| **Plotting** | Matplotlib | 3.10.6 |
| **Plotting Style** | Seaborn | 0.13.2 |
| **ML Utilities** | scikit-learn | 1.7.2 |
| **Graph Analysis** | NetworkX | 3.3 |

---

## 📂 Project Structure

```text
.
├── build_notebook.py                             # Assembles the Jupyter notebook
├── GraphCL_Lite_GNN_MiniProject.ipynb            # Original unexecuted notebook
├── GraphCL_Lite_GNN_MiniProject_executed.ipynb   # Fully executed notebook with latest runs
├── validate_nb.py                                # Syntax-check for notebook cells
├── requirements.txt                              # Pin-pointed dependencies
├── data/                                         # Cora dataset (downloaded by PyG)
└── README.md                                     # You are here!
```

---

## 🚀 Setup & Installation

Get started in seconds:

```bash
# 1️⃣ Clone the repository
git clone https://github.com/Gagann-09/GraphCL-Lite.git
cd GraphCL-Lite

# 2️⃣ Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# 3️⃣ Install dependencies
pip install -r requirements.txt
```

---

## 💡 Usage Guide

### 1. Generate the Notebook (Optional)
If you want to reconstruct the notebook from source scripts:
```bash
python build_notebook.py
```
This generates `GraphCL_Lite_GNN_MiniProject.ipynb` in your directory.

### 2. Run the Notebook
Launch Jupyter to explore the interactive workflow:
```bash
jupyter notebook GraphCL_Lite_GNN_MiniProject_executed.ipynb
```
**What the notebook covers:**
1. Loads and validates the Cora dataset.
2. Visualises the raw graph and class distributions.
3. Defines the `ContrastiveGCN` model.
4. Trains with the dual objective.
5. Evaluates sensitivity over different `alpha` thresholds.
6. Benchmarks against **GRACE+Linear** and **GRACE+Finetune** baselines.
7. Plots training curves, t-SNE embeddings, and performs the ablation study.

### 3. Validate Notebook Syntax (Optional)
```bash
python validate_nb.py
```
Checks for any syntax errors within the notebook cells.

---

## ⚙️ Configuration

- **Device Selection:** Automatically selects `CUDA` if available, gracefully falling back to `CPU`.
- **Hyper-parameters:** Easily editable within the notebook:
  - `epochs` (default: 200)
  - `lr` (default: 0.01)
  - `alpha` – Weight for classification vs contrastive loss (evaluated dynamically across [0.1, 0.9])
  - `feat_drop` / `edge_drop` – Augmentation probabilities (default: 0.2)

---

## 🔒 Security & Best Practices

- **Zero Secrets:** No embedded API keys, passwords, or tokens.
- **Data Safety:** Dataset validation prevents crashes by enforcing a 500 MB memory ceiling and checking for data anomalies.
- **Reproducibility:** Fixed seeds (`SEED = 42`) and CuDNN deterministic mode ensure your results are consistent every time.

---

## ⚠️ Limitations & Known Issues

- Tuned specifically for the **Cora** dataset. Larger graphs may require batch processing (e.g., using `NeighborLoader`) or memory-efficient sampling.
- Augmentations are limited to feature masking and edge dropout.
- Assumes a recent version of `torch-geometric` (`>=2.7.0`).

---

<p align="center">
  <i>If you find this project useful, please consider giving it a ⭐ on GitHub!</i>
</p>

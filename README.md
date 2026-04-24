# GraphCL‑Lite Mini‑Project

## Overview

**GraphCL‑Lite** is a compact research‑grade implementation of *contrastive learning* for Graph Neural Networks (GNNs). It demonstrates how adding a self‑supervised NT‑Xent contrastive loss to a standard 2‑layer Graph Convolutional Network (GCN) improves semi‑supervised node classification on the **Cora** citation‑network dataset.

The entire workflow – from data loading and validation, through model definition, training, visualisation, to an ablation study – is encapsulated in a single Jupyter notebook:

- `GraphCL_Lite_GNN_MiniProject.ipynb`
- The notebook can be generated (or regenerated) by running `build_notebook.py`.

---

## Features

- **Dataset validation** – checks for NaNs, infinities, and memory limits before training.
- **Graph augmentations** – random feature masking and edge dropout to create two views for contrastive learning.
- **Dual‑objective training** – combines supervised cross‑entropy with NT‑Xent contrastive loss (`L = α·L_cls + (1‑α)·L_con`).
- **Early stopping** based on validation accuracy.
- **Extensive visualisations** – loss/accuracy curves, graph sub‑structure visualisation, confusion matrix, and t‑SNE embedding plots.
- **Ablation study** – baseline GCN (no contrastive loss) vs. GraphCL‑Lite.
- **Reproducibility** – fixed random seeds, dataset integrity checks, no external secrets.

---

## Tech Stack

| Category | Library | Version |
|----------|---------|---------|
| Deep learning | PyTorch | 2.8.0 |
| Graph utilities | torch‑geometric | 2.7.0 |
| Numerical | NumPy | 2.4.3 |
| Plotting | Matplotlib | 3.10.6 |
| Plotting style | Seaborn | 0.13.2 |
| ML utilities | scikit‑learn | 1.7.2 |
| Graph analysis | NetworkX | 3.3 |
| Image handling (unused) | torchvision | 0.23.0 |

---

## Project Structure

```text
.
├── build_notebook.py                # Assembles the Jupyter notebook
├── GraphCL_Lite_GNN_MiniProject.ipynb   # Executable notebook (generated)
├── validate_nb.py                  # Syntax‑check for notebook cells
├── requirements.txt                # Pin‑pointed dependencies
├── data/                           # Cora dataset (downloaded by torch_geometric)
│   ├── Cora/...
├── .gitignore
└── README.md
```

---

## Setup & Installation

```bash
# 1️⃣ Clone the repository
git clone <repo‑url>
cd <repo‑directory>

# 2️⃣ Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# 3️⃣ Install exact dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Generate the notebook (optional)
```bash
python build_notebook.py
```
This command writes `GraphCL_Lite_GNN_MiniProject.ipynb` next to the script.

### 2. Open the notebook
```bash
jupyter notebook GraphCL_Lite_GNN_MiniProject.ipynb
```
Run the cells sequentially. The notebook will:
1. Load and validate the Cora dataset.
2. Visualise the raw graph and class distribution.
3. Define the `ContrastiveGCN` model.
4. Train with the dual objective (default `α=0.5`).
5. Plot training curves and evaluate on the test set.
6. Perform the ablation study and visualise embeddings.

### 3. Validate notebook syntax (optional)
```bash
python validate_nb.py
```
The script reports any syntax errors in code cells.

---

## Configuration

- **Device selection** – the script automatically selects CUDA if available, otherwise CPU.
- **Hyper‑parameters** (editable in the notebook):
  - `epochs` (default 200)
  - `lr` (default 0.01)
  - `weight_decay` (default 5e‑4)
  - `alpha` – weight for the classification loss (default 0.5)
  - `feat_drop` / `edge_drop` – augmentation probabilities (default 0.2)
  - `sample_size` – number of nodes sampled for contrastive loss (max 512)
- No external configuration files (`.env`) are required.

---

## Security Notes

- **No secrets** – the project does not embed API keys, passwords, or tokens.
- **Dataset validation** – checks for NaNs/Inf and enforces a 500 MB memory ceiling before moving data to the accelerator.
- **Deterministic behaviour** – seeds are fixed (`SEED = 42`) and CuDNN deterministic mode is enabled.
- **Safe file access** – all file paths are resolved under the repository root (`./data`).

---

## Limitations & Known Issues

- The implementation is tuned for the **Cora** dataset only; larger graphs may require batch processing or memory‑efficient sampling.
- Contrastive augmentation is limited to simple feature masking and edge dropout; more sophisticated graph augmentations are not provided.
- The notebook assumes a recent version of `torch‑geometric`; mismatched library versions could cause import errors.

---

# PharmXAI-3D: Explainable SciML for Protein-Ligand Binding Affinity

PharmXAI-3D is still a work in progress, but the aim is to be an advanced Scientific Machine Learning (SciML) pipeline engineered to predict absolute protein-ligand binding affinities ($pK_d$) directly from raw 3D crystallographic coordinates, utilizing SE(3)-invariant continuous filter convolutions and physics-targeted multi-task regularization.

## 🚀 Core Utilities & Applications

1. **High-Throughput Virtual Screening (HTVS):**
   The goal is for the model to acts as a highly accurate oracle for ranking massive chemical libraries. Predicts absolute binding affinity ($pK_d$) in milliseconds, bypassing the inaccuracy and computational cost of classic docking algorithms (e.g., AutoDock Vina).

2. **Lead Optimization (Ligand Pharmacophore Extraction):**
   We aim to provide medicinal chemists with a visual Saliency Map of the ligand. It mathematically isolates the "untouchable" functional groups essential for binding (Magenta) versus the "modifiable scaffolds" (Cyan) where functional groups can be altered to improve ADMET profiles without destroying affinity.

3. **Mechanistic Toxicity Analysis (Complex Mapping):**
   Holistically maps the orthosteric binding pocket to identify critical protein-ligand interaction vectors (e.g., specific H-bonds or $\pi$-stacking). Enables rational drug design to purposefully avoid specific receptor residues known to cause off-target toxicity.

---

## 🏗️ Architectural Phases

- **Phase 1 (Topological Ingestion):** Strict 6Å spatial radius pocket isolation to prevent memory bottlenecks. Computes a dense 11D physical prior vector (Atomic Number, Mass, Aromaticity, Formal Charge, Hybridization) avoiding hardcoded chemical rules.
- **Phase 2 (Continuous SE(3) Convolutions):** Employs `ContinuousFilterConv` (SchNet-like topology). Euclidean distances are expanded into 32-bin Radial Basis Functions (RBF) for continuous spatial gradients.
- **Phase 3 (Physics-Targeted Regularization):** Solves catastrophic overfitting via dual-vector regularization (Dropout, L2) and a Multi-Task Auxiliary Head that predicts a global Lennard-Jones steric potential proxy, forcing the latent manifold to physically understand spatial crowding.
- **Phase 4 (Explainable AI / Saliency):** Computes the partial derivative $\frac{\partial (pK_d)}{\partial (Dist)}$. Extracts a continuous mathematical mapping of structural sensitivity projected into an interactive `3Dmol.js` Web Viewer.

---

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/lapsx/PharmXAI-3D.git
cd PharmXAI-3D
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **Dataset Preparation:**
   Download the [PDBbind Dataset](http://www.pdbbind.org.cn/) (General & Refined sets) and extract them into the `raw/` directory.

---

## 💻 Usage

### 1. Training the Model
Execute the data processing and training pipeline. The model will auto-checkpoint the best weights (`pharm_model_weights_best.pth`).
```bash
python3 data_processor.py
python3 train.py
```

### 2. Custom Inference & Virtual Screening (`predict_custom.py`)
Run advanced inferences on arbitrary `.pdb` and `.sdf` files. The system automatically performs **Smina Docking** and **PINN Relaxation (Induced-Fit)** via `torch.autograd` to optimize spatial clashes before predicting the final affinity.

**Commands and Arguments (Tags):**
- `-r` or `--receptor`: Path to the receptor `.pdb` structure (Required).
- `-l` or `--ligand`: Path to the drug/ligand `.sdf` structure (Required).
- `-n` or `--native`: Path to the native crystal ligand `.pdb`. Activates **Targeted Docking**, using the native ligand as a bounding box.
- `-c` or `--compare`: Path to the native crystal ligand `.pdb`. Activates **Blind Docking** (Full scan) and compares the Center of Mass of the prediction with the true native structure, scientifically validating the docking pose.
- `-t` or `--true-affinity`: Allows injecting a numerical value for the true experimental pKd (e.g., `-t 8.5`) to calculate and plot the absolute error margin.
- `--no-minimize`: Disables PINN autograd physical relaxation (Not recommended).
- `--no-explain`: Disables Saliency Map / HTML generation.

**Usage Examples:**
```bash
# Targeted Docking (High-Throughput Screening)
python3 predict_custom.py -r target.pdb -l drug.sdf -n native_crystal.pdb

# Blind Docking with True Affinity Scientific Validation
python3 predict_custom.py -r target.pdb -l drug.sdf -c native_crystal.pdb -t 8.5
```

**Web and PyMOL Integration:**
For each inference, the script automatically generates **two spatial visualization files**:
1. `pharmacophore_...html`: An interactive standalone 3D web viewer.
2. `pharmacophore_...pml`: An executable **PyMOL** macro script. To open it professionally on your desktop, simply drag and drop the `.pml` file into the PyMOL window or run `pymol script.pml`. It will automatically load the complexes, coloring schemes, and Saliency Map pseudoatoms.

### 3. XAI Extraction (Command Line Interface)
Extract continuous mathematical mappings of structural sensitivity projected into an interactive standalone `3Dmol.js` Web Viewer.

```bash
# Pure Ligand Pharmacophore Extraction (Default: Dopamine D3 Receptor - 3pbl)
python3 explain.py

# Inject a specific PDB ID dynamically (e.g., Serotonin 5-HT2B bound to Ergotamine)
python3 explain.py 5tvn

# Map the holistic complex (Receptor + Ligand) for a specific target
python3 explain_complex.py 5tvn
```

### 4. Build GPCR Mini-Library
Download our automated validation set to test cross-reactivity across Dopamine, Adenosine, and Opioid receptors (Delta, Kappa, Mu) directly from RCSB and PubChem:
```bash
python3 build_mini_library.py
```

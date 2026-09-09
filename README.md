# PhysScore

**A graph neural network scoring function for protein–ligand binding affinity, with
physical constraints in the loss.** Work in progress.

Given the 3D coordinates of a protein pocket and a ligand posed inside it, the network
predicts binding affinity ($pK_d$) in milliseconds. It is built on SE(3)-**invariant**
continuous-filter convolutions (SchNet-style): the network sees only interatomic
distances, so its output is unchanged by rotation or translation of the complex.

It reaches **Pearson R = 0.754** on the CASF-2016 core set with **127k parameters** —
second among the 35 scoring functions benchmarked below, and one to two orders of
magnitude smaller than the deep-learning methods in the same accuracy band.

---

## Scope

Read this before anything else — the project started as a pharmacophore-explainability
experiment (hence the old `PharmXAI` name) and became a scoring function along the way.

**What it is.** A scoring function. You give it a pocket and a posed ligand; it
returns an affinity estimate. Its use is **ordering a library** — deciding which of
a thousand candidates to test first.

**What it is not.**

- **Not a docking program.** It does not search for poses, and it is weak at
  recognising a correct one (30th of 35 — see below). It needs smina, Vina, or Glide
  to place the ligand first. It is a **re-scorer**.
- **Not an absolute affinity predictor.** A predicted $K_d$ can be off by a factor of
  tens. Rank compounds with it; do not report a number from it.
- **Not a pharmacophore tool.** The gradient maps it produces are
  model-audit instruments, not database search queries. See *Model interpretation*.

---

## What the model actually does

**It ranks compounds well. It measures absolute affinity poorly.** That distinction
decides what the tool is good for, so it is stated up front.

Evaluated on the CASF-2016 core set — 285 complexes never seen during training or
model selection:

| metric | value | reading |
|---|---|---|
| Spearman ρ | **0.746** | ranks compounds reliably |
| Pearson R | **0.754** | R² = 0.568 |
| RMSE | 1.486 $pK_d$ | typical error of ~31× in $K_d$ |
| MAE | 1.185 $pK_d$ | |
| slope of predicted vs. true | **0.424** | predictions compressed toward the mean |
| baseline (predicting the mean) | 2.170 RMSE | |

The slope of 0.424 is the number that characterises the model. An ideal predictor
gives 1.0. At 0.424 it systematically under-predicts strong binders and
over-predicts weak ones, spanning 2.83–10.67 where the truth spans 2.07–11.82.
Together with an RMSE of 1.486 log units, this means a predicted $K_d$ can be off by
a factor of tens. **Do not use it to report an absolute affinity.** Use it to order
a library, which is what a Spearman ρ of 0.746 supports.

### Position among published scoring functions

The CASF-2016 package ships reference scores for 34 scoring functions on these same
285 complexes. Recomputing Pearson R for all of them under identical conditions:

| rank | method | R |
|---|---|---|
| 1 | deltaVinaRF20 | 0.816 |
| **2** | **PhysScore (this work)** | **0.754** |
| 3 | X-Score | 0.631 |
| 4 | deltaSAS | 0.625 |
| 7 | AutoDock Vina | 0.604 |
| — | *median of the 34* | *0.537* |

Second of 35, ahead of X-Score and AutoDock Vina. The scope of that claim matters:
the 34 references are the classical scoring functions bundled with CASF-2016.
Deep-learning methods published since reach R ≈ 0.70–0.85 on the same set, so this
model now sits inside that band rather than below it — but at its lower end, and with
far fewer parameters than the methods at the top.

### It does not find poses — it scores poses it is given

CASF-2016 grades a scoring function on four separate powers, and the numbers above are
only the first. On **docking power** — given a site and ~79 candidate poses of the same
ligand, put the correct one on top — the same model ranks **30th of 35**:

| | top1 % |
|---|---|
| AutodockVina | 90.1 |
| *median of the 34 references* | *64.9* |
| **PhysScore (this work)** | **47.0** |
| contact-count baseline | 30.9 |
| chance | 25.2 |

**2nd of 35 at scoring, 30th of 35 at docking.** No reference function has a profile
that lopsided, and the cause is not mysterious: every PDBbind complex is a correct
pose, so nothing in training ever asked the network to place a minimum at the native
geometry.

The deficit is specific. Split by scale, PhysScore *beats* AutodockVina on global trend
(rho 0.374 vs 0.334) and loses 2.5× within 3 Å of the native pose (0.249 vs 0.614). All
43 points of top1 come from that neighbourhood: the model separates a plausible pose
from an absurd one, and cannot choose between two plausible ones.

So this is a **re-scorer**. It needs a docking program to find the pose — which is
exactly its role in `predict_custom.py`, where smina docks and the network scores
afterwards. `EXPERIMENTOS.md` §4 has the full measurement and the three measurement
traps found along the way.

---

## Architecture

| | |
|---|---|
| trainable parameters | 127,090 |
| message passing | 4 continuous-filter convolution layers, 64 channels |
| distance encoding | 32 radial basis functions over 0–10 Å |
| atom features | 13 per atom |
| readout | per-role pooling: mean **and** sum, separately for ligand and pocket |
| heads | $pK_d$ regression + steric auxiliary |

**Graph construction.** The protein is truncated to a 6 Å shell around the ligand.
Edges are covalent (within 2 Å, intramolecular) and interaction (within 5 Å, between
ligand and pocket). Each atom's role — ligand or protein — is an explicit embedded
feature; without it the network sees an undifferentiated cloud of atoms and cannot
tell which part is the drug.

**Atom features.** Atomic number, mass, degree, aromaticity, formal charge, H-bond
donor, H-bond acceptor, and six hybridization one-hots.

*A note on hydrogens.* PDBbind's structure files contain hydrogens, but they are not
experimental: the headers read `GENERATED BY X-TOOL`, and N–H distances measure
1.0400 Å with a standard deviation of 0.0003 Å — geometric placement, not
measurement. They are therefore a deterministic function of the heavy atoms, carrying
no independent information while inflating the graph by roughly half. They are
removed, but what depends on them is kept: donor and acceptor flags are computed
*while the hydrogens are still present* and survive their removal as atom properties.

---

## Where the physics is

The network is not a black box fitted to labels alone. Four places where physical
structure is imposed rather than learned:

**Symmetry, in the architecture.** The convolutions are SE(3)-invariant: the model
consumes interatomic *distances*, never coordinates. Rotating or translating a complex
cannot change its prediction, because the rotated complex is literally the same input.
This is a hard constraint, not a learned approximation — and a consequence worth
knowing is that rotational data augmentation would achieve nothing here.

**A Lennard-Jones auxiliary target, in the loss.** A second head predicts a steric
proxy computed from geometry, at loss weight 0.1: σ = 3.5 Å over non-bonded edges only
(bonded atoms do not interact through Lennard-Jones in a force field), averaged per
edge and log-compressed to tame the repulsive tail. Its correlation with $pK_d$ is
−0.09, which is the intent — an auxiliary target strongly correlated with the primary
one would be a shortcut rather than a regulariser.

**Chemistry that survives preprocessing.** Donor and acceptor flags are computed while
hydrogens are still present and kept after their removal; ionisation state is assigned
at pH ~7 (see *Architecture* above). The information is distilled, not discarded.

**Energy minimisation at inference.** Before prediction the ligand pose is relaxed
against a physical energy — Lennard-Jones over contacts, a harmonic restraint on bond
lengths, an anchor to the docked pose. Only ligand atoms move, the pocket is held
fixed, and **the network is not in the loop**. An earlier version moved atoms to
maximise the model's own predicted affinity, and since the reported affinity was then
measured on the optimised pose, it inflated the result by construction.

**In progress: a stationarity constraint.** If the score is to behave like an energy,
the net force and torque on the ligand must vanish at the crystallographic pose. That
is a variational residual — the closest well-posed analogue here to a PINN's PDE
residual, since no PDE has $pK_d$ as its solution ($pK_d$ is a *free* energy, with
entropy and desolvation inside). It is imposed on the 6 rigid-body degrees of freedom
and paired with a ranking term against perturbed poses, because stationarity alone is
degenerate: a constant function satisfies it perfectly. This targets the docking-power
gap above and trains a **separate** head, leaving the affinity head untouched.

---

## Model interpretation

Secondary to the scoring function, and deliberately modest in what it claims.

`explain.py` computes $|\partial (pK_d)/\partial(\text{distance})|$ for every edge by
backpropagation and renders the top percentile in an interactive 3D viewer.
`consenso_farmacoforo.py` aggregates that signal across the five chemically distinct
ligands binding one target, using the CASF-2016 cluster structure, and reports which
pocket residues rank high for all five.

**These are sensitivity maps of the model, not pharmacophores.** They answer which
interatomic distances most influence *this model's* prediction — model-relative, and
not usable as a database search query. A classical pharmacophore is a ligand-side
object with defined geometry; these are receptor-side residue rankings. They are not
the same thing, and the project no longer claims otherwise.

Their legitimate use is **auditing**. If a map highlights a known catalytic hydrogen
bond, that is evidence the network learned a real interaction; if it highlights a
solvent-exposed tail, that is evidence of a learned shortcut. Both are worth knowing,
and the second is why the tool exists.

Two safeguards are built in: crystallographic waters are excluded (`HOH` numbering is
per-structure and never corresponds between entries), and pocket overlap across the
five complexes is reported, with a warning below 20% — consensus by residue number
presupposes a shared numbering convention, and near-zero overlap means that
presupposition has failed.

---

## Installation

```bash
git clone https://github.com/Lapsx/GNN_physics_ScoringFunction.git
cd GNN_physics_ScoringFunction
pip install -r requirements.txt
```

**Data.** Download the [PDBbind](http://www.pdbbind.org.cn/) general set into `raw/`
with the index files in `index/`. For the standard benchmark, extract
`CASF-2016/power_screening/CoreSet.dat` from the CASF-2016 package and save it as
`core_set.dat` in the project root.

---

## Usage

### Training

```bash
python3 train.py
```

Processing the full PDBbind set takes several hours on first run and is cached in
`processed/`. The split is three-way and hermetic: the CASF core set is removed first
as **test**, the remaining refined set is **validation**, everything else is
**training** — 13,711 / 5,041 / 285 complexes. Since the checkpoint is selected on
validation, only the test number is reportable, and the script evaluates it once at
the end using the selected checkpoint.

Training stops after 30 epochs without improvement. In the reference run the best
model appeared at epoch 17 and training halted at 47 of a possible 300.

### Inference on arbitrary structures

```bash
# Targeted docking against a known binding site
python3 predict_custom.py -r target.pdb -l drug.sdf -n native_crystal.pdb

# Blind docking, validating the predicted pose against the crystal structure
python3 predict_custom.py -r target.pdb -l drug.sdf -c native_crystal.pdb -t 8.5
```

| flag | meaning |
|---|---|
| `-r`, `--receptor` | receptor `.pdb` (required) |
| `-l`, `--ligand` | ligand `.sdf` (required) |
| `-n`, `--native` | native ligand `.pdb`; targeted docking via bounding box |
| `-c`, `--compare` | native ligand `.pdb`; blind docking, compares centres of mass |
| `-t`, `--true-affinity` | experimental $pK_d$, for plotting the error |
| `--no-minimize` | skip the steric relaxation |
| `--no-explain` | skip saliency and HTML generation |

**Pose relaxation.** Before prediction the ligand pose is relaxed by minimising a
physical energy: Lennard-Jones over the contacts, a harmonic restraint on covalent
bond lengths, and an anchor to the docked pose. Only ligand atoms move, the pocket is
held fixed, and the network is not in the loop.

This replaces an earlier formulation that moved atoms to *maximise the model's own
predicted affinity*. That version had no physical energy term, deformed the protein
along with the ligand, and — because the reported affinity was then measured on the
optimised pose — inflated the result by construction.

Each inference writes an interactive `pharmacophore_*.html` viewer and an executable
`pharmacophore_*.pml` PyMOL macro.

### Consensus analysis

```bash
python3 consenso_farmacoforo.py 10          # by cluster id (1–57)
python3 consenso_farmacoforo.py --pdb 3u5j  # by any PDB id in the core set
python3 consenso_farmacoforo.py 10 --top 8  # residues considered salient per complex
```

Writes `consenso_cluster<N>.pml`, colouring unanimous residues magenta and partial
consensus orange.

### GPCR validation library

```bash
python3 build_mini_library.py
```

Downloads a cross-reactivity set spanning dopamine, adenosine and opioid receptors
from RCSB and PubChem.

---

## Limitations

**Absolute affinity is unreliable.** See the slope of 0.424 above. This is a ranker.

**Weak docking power.** 30th of 35 at picking the correct pose (see above). Use a
docking program to place the ligand; this model only scores what it is given.

**No target in the benchmark is new to the model.** Measured by 4-mer containment,
every one of the 285 CASF-2016 core complexes shares sequence with some training
protein: minimum 0.803, median 0.996, 62% above 0.99. A stratified "no close homolog"
test is impossible on this split, and that impossibility *is* the finding. It applies
to every method trained on PDBbind, so comparisons against the 34 classical functions
— which never saw these data — favour ML methods unfairly; comparisons between ML
methods remain fair.

Removing the 2,812 training complexes homologous to the core set, against a
same-size random control over three paired seeds, costs **0.053 ± 0.006** in R
(p = 0.004). **The honest pair of numbers is R = 0.753 on a known target and
R = 0.692 on a new one** — the latter still above the count baseline (0.592) and the
median classical function (0.537).

**The model overfits early.** Best validation arrives at epoch 17 of a possible 300,
after which training loss keeps falling while validation rises. This is the main open
lever for improvement. Note that rotational data augmentation would achieve nothing
here — the network is SE(3)-invariant, so its output does not change under rotation.
Coordinate noise would be the meaningful perturbation.

**Seed variance.** The headline numbers come from the reference run; across three
seeds the model gives R = 0.753 ± 0.016 and RMSE = 1.463 ± 0.040, so the reference
checkpoint sits inside the spread. Differences below ~0.03 in R are not
interpretable without paired multi-seed runs.

**Consensus needs consistent numbering.** The per-target analysis matches residues by
number, which fails when entries use different conventions. The tool detects and warns
about this but cannot repair it; structural superposition would be required.

**Contact count is available to the head.** The per-role readout hands
`log(n_contacts)` straight to the MLP, which is what fixed the earlier size-blind
mean pooling — but it also makes counting the most tempting shortcut available to the
network. Every new evaluation on this model is run alongside a trivial count baseline
for that reason.

---

## Data and references

- PDBbind v2020 general set — [pdbbind.org.cn](http://www.pdbbind.org.cn/)
- CASF-2016 benchmark, including the reference scoring functions used above
- SchNet: Schütt et al., *SchNet: A continuous-filter convolutional neural network for
  modeling quantum interactions*, NIPS 2017
- Docking via [smina](https://sourceforge.net/projects/smina/)

`EXPERIMENTOS.md` (Portuguese) is the bench record: how every number here was
obtained, what was tested and rejected, and what each result does and does not
license one to claim.

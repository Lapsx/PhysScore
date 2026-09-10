# PhysScore

**A graph neural network scoring function for protein–ligand binding affinity, with
physical constraints in the loss.** Work in progress.

Given the 3D coordinates of a protein pocket and a ligand posed inside it, the network
predicts binding affinity ($pK_d$) in milliseconds. It is built on SE(3)-**invariant**
continuous-filter convolutions (SchNet-style): the network sees only interatomic
distances, so its output is unchanged by rotation or translation of the complex.

It ships in two sizes. **`lite`** is a single 144k-parameter network: R = 0.752 on
CASF-2016 and 78.5% top1 docking power. **`full`** averages six checkpoints for
R = 0.819 and 83.9% top1 — 8th of 35 on pose selection, and above the best classical
scoring function on affinity — at six times the inference cost.

---

## Scope

Read this before anything else — the project started as a pharmacophore-explainability
experiment (hence the old `PharmXAI` name) and became a scoring function along the way.

**What it is.** A scoring function. You give it a pocket and a posed ligand; it
returns an affinity estimate. Its use is **ordering a library** — deciding which of
a thousand candidates to test first.

**What it is not.**

- **Not a docking program.** It does not *search* for poses — it scores poses handed
  to it. It has become good at *recognising* a correct one (8th of 35, see below),
  which it was not before the physics constraints were added, but it still needs
  smina, Vina, or Glide to generate candidates. It is a **re-scorer**, now a
  competent one.
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

| metric | `lite` (1 model) | `full` (6 models) |
|---|---|---|
| Pearson R | 0.752 ± 0.013 | **0.819** |
| RMSE ($pK_d$) | 1.470 ± 0.042 | **1.357** |
| MAE | 1.207 | 1.105 |
| slope of predicted vs. true | 0.458 ± 0.044 | 0.473 |
| **top1 docking power** | 78.5% ± 1.7 | **83.9%** |
| top3 docking power | 93.2% | 95.1% |
| RMSD of selected pose | 1.46 Å | **1.10 Å** |
| parameters | **144k** | 1.3M |
| baseline (predicting the mean) | 2.170 RMSE | |

`lite` figures are mean ± sd over three seeds. Docking power is the fraction of the
285 targets where the top-scored pose, out of ~79 candidates, is within 2 Å of the
crystal pose.

**The slope is the number that characterises the model, and it is the one that has not
moved.** An ideal predictor gives 1.0; at ~0.46 this one systematically under-predicts
strong binders and over-predicts weak ones. With an RMSE of ~1.4 log units, a predicted
$K_d$ can be off by a factor of tens. **Do not use it to report an absolute affinity.**

That is not a fixable calibration error, and it is worth being precise about why.
Under a squared-error loss the optimal prediction is the conditional mean, which is
*less* variable than the truth — compression is what a good imperfect predictor does.
Since the slope of predicted-vs-true equals R·σ̂/σ, no linear rescaling can push it past
**R**, and forcing it to 1.0 makes RMSE worse. Improving the slope means improving R.
There is no post-hoc shortcut.

### Position among published scoring functions

The CASF-2016 package ships reference scores for 34 scoring functions on these same
285 complexes. Recomputing Pearson R for all of them under identical conditions:

| rank | method | R |
|---|---|---|
| **—** | **PhysScore `full`** (6-model ensemble) | **0.819** |
| 1 | deltaVinaRF20 | 0.816 |
| **2** | **PhysScore `lite`** (single, 144k params) | **0.752** |
| 3 | X-Score | 0.631 |
| 4 | deltaSAS | 0.625 |
| 7 | AutoDock Vina | 0.604 |
| — | *median of the 34* | *0.537* |

`lite` is second of 35 as a single 144k-parameter model. `full` edges past
deltaVinaRF20, but it is an **ensemble** and every reference is a single model, so that
row is marked `—` rather than 1st: it is not a like-for-like comparison and should not
be reported as one.

The scope of the whole table matters too. The 34 references are the *classical* scoring
functions bundled with CASF-2016. Deep-learning methods published since reach
R ≈ 0.70–0.85 on the same set, so this model sits inside that band — with one to two
orders of magnitude fewer parameters than the methods at its top.

### Pose selection, and how the physics got it there

CASF-2016 grades a scoring function on four separate powers, and affinity is only the
first. **Docking power** asks a different question: given a site and ~79 candidate poses
of the same ligand, does the top-scored one match the crystal structure?

The first measurement was bad. The original model — trained only to predict affinity —
placed **30th of 35**, at 47.0% top1, barely above the contact-count baseline. The cause
was not subtle: every PDBbind complex is a *correct* pose, so nothing in training ever
asked the network to put a minimum at the native geometry.

Two physical constraints fixed it, both on a **separate head** so the affinity output
was never touched:

- **Stationarity** — at the crystallographic pose, the net force and torque on the
  ligand must vanish. This is the variational condition an energy satisfies at a
  minimum, imposed on the 6 rigid-body degrees of freedom.
- **Ranking** against rigidly perturbed poses, with the graph edges held fixed so that
  contact count is *identical* between native and perturbed — closing the counting
  shortcut before it exists.

Neither works alone: stationarity by itself is degenerate, since a constant function
satisfies it perfectly.

| | top1 % | |
|---|---|---|
| AutodockVina | 90.1 | 1st |
| deltaVinaRF20 | 89.1 | 2nd |
| ChemPLP@GOLD | 85.6 | 6th |
| **PhysScore `full`** | **83.9** | **8th** |
| GlideScore-XP | 83.5 | 9th |
| **PhysScore `lite`** | **78.5** | **14th** |
| *median of the 34 references* | *64.9* | |
| contact-count baseline | 30.9 | |
| chance | 25.2 | |

**From 30th to 8th, with the affinity R unchanged** (0.754 → 0.752, within seed noise).
Three controls back this up: the *affinity* head of the same checkpoint scores 40.4%,
so the gain is not diffuse; the contact-count baseline stays at 30.9%, so it is not a
size shortcut; and removing the crystal pose from the candidate set leaves 75.8%, so the
model is judging docked geometry rather than recognising the crystal.

It still does not *search* for poses. In `predict_custom.py`, smina generates candidates
and the network scores them — which is what a re-scorer does. `EXPERIMENTOS.md` §4 and
§5 have the full measurements and the three measurement traps found along the way.

### Where the ensemble helps, and what it costs

Averaging six checkpoints — three scalar, three with directional features — gives the
`full` numbers above. The mechanism is error decorrelation, and it is worth stating a
tension openly: **the directional features were tested on their own and rejected**
(§6 — no sustained gain in R, slope, or top1 across three paired seeds, at 2.3× the
parameters). They earn their place here only because they fail *differently* from the
scalar models, which is exactly what an ensemble needs. An ensemble of three scalar
seeds reaches R = 0.789; adding the three directional ones takes it to 0.819.

Note also that `full` is an ensemble while all 34 reference functions are single
models, so the affinity comparison flatters it.

---

## Architecture

| | |
|---|---|
| trainable parameters | 143,795 (`lite`) · 1.3M (`full`, 6 models) |
| message passing | 4 continuous-filter convolution layers, 64 channels |
| distance encoding | 32 radial basis functions over 0–10 Å |
| atom features | 13 per atom |
| readout | per-role pooling: mean **and** sum, separately for ligand and pocket |
| heads | $pK_d$ regression + steric auxiliary + pose quality |

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

**A stationarity constraint, in the loss.** If the score is to behave like an energy,
the net force and torque on the ligand must vanish at the crystallographic pose. That is
a variational residual — the closest well-posed analogue here to a PINN's PDE residual,
since no PDE has $pK_d$ as its solution ($pK_d$ is a *free* energy, with entropy and
desolvation inside). It is imposed on the 6 rigid-body degrees of freedom, not the 3N
Cartesian ones: the crystal pose is not the minimum of the ligand's *internal* geometry
according to the model, and demanding that would inject noise.

It is paired with a ranking term against rigidly perturbed poses, and the pairing is
load-bearing rather than decorative: **stationarity alone is degenerate**, because a
constant function zeroes force and torque perfectly. Measured at initialisation, the
pose head starts almost constant (|F| = 1e-4). Ranking gives the minimum its content;
stationarity gives it its shape. This is what took docking power from 30th to 8th of 35
(§5), and it lives on a separate head so the affinity output was never constrained.

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

**That is all you need to score a complex.** The trained weights ship with the
repository — 0.57 MB for `lite`, 5.1 MB for all six `full` checkpoints. Every number
reported here comes from them. No dataset download is required to use the model.

**Data, only if you want to retrain or reproduce.** Download the
[PDBbind](http://www.pdbbind.org.cn/) general set into `raw/` with the index files in
`index/`. For the benchmark, extract `CASF-2016/power_screening/CoreSet.dat` from the
CASF-2016 package and save it as `core_set.dat` in the project root.

---

## Usage

### Scoring a complex

The common case. Uses the shipped checkpoints; no training needed.

**Which one to use.** `lite` is one 144k-parameter network and one forward pass —
use it for screening large libraries, where throughput decides. `full` averages six
checkpoints for +0.067 in R and +5.4 points of top1 docking power, at six times the
cost — use it when you are scoring tens or hundreds of compounds and accuracy decides.

| | R | top1 docking | params | inference |
|---|---|---|---|---|
| `lite` | 0.752 ± 0.013 | 78.5% ± 1.7 | 144k | 1× |
| `full` | 0.819 | 83.9% | 1.3M | 6× |

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

### Retraining (optional)

Only needed to reproduce the numbers, to change the architecture, or to train on
another dataset. Requires the PDBbind download above. Note that a model retrained on
different data is a different model: the results reported here describe this
checkpoint on CASF-2016, and would have to be re-measured.

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

**Absolute affinity is unreliable.** See the slope of ~0.46 above. This is a ranker.

**It does not generate poses.** Docking power is now 8th of 35 (`full`) at *picking*
the correct pose from candidates, but the candidates have to come from somewhere. Use
smina, Vina, or Glide to place the ligand; this model scores what it is given.

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
median classical function (0.537). Both were measured on the pre-physics scalar model;
the homology penalty has not been re-measured for `lite` or `full`.

**The model overfits early.** Best validation arrives around epoch 17–19 of a possible
300, after which training loss keeps falling while validation rises. Note that rotational
data augmentation would achieve nothing here — the network is SE(3)-invariant, so its
output does not change under rotation. Coordinate noise would be the meaningful
perturbation.

**The slope has not moved.** 0.458 ± 0.044, essentially where it started. Adding
directional (angular) features was the obvious attack and it failed: no sustained gain
in R, slope, or top1 across three paired seeds, at 2.3× the parameters (§6). The model
*uses* the angular information intensively — ablating it shifts predictions by 2.14
$pK_d$ — it just does not convert into accuracy. This is the most serious open problem.

**Ensemble numbers are not like-for-like.** `full` averages six models; all 34 CASF
references are single models. Compare `lite` against them, not `full`.

**Seed variance.** `lite` figures are mean ± sd over three seeds (R = 0.752 ± 0.013).
Differences below ~0.03 in R are not interpretable without **paired** multi-seed runs —
the same seed for both configurations, so the difference is the configuration and not
the draw. This is not pedantry: the directional-features result looked like a +0.014
gain in a single run and turned out to be null (+0.018, p = 0.324, sign flipping across
seeds) once paired.

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

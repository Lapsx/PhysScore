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

## What the three numbers mean

CASF-2016 grades a scoring function on separate capabilities. They sound similar and
measure very different things, so here they are in plain terms before any numbers.

**Scoring power — "how strongly does this bind?"** Given a complex with the ligand in its
correct, crystallographically determined pose, predict the binding affinity. Reported as
Pearson **R** against experiment across 285 complexes. This is the easiest of the three:
the pose is handed to you and you only estimate a magnitude.

**Docking power — "which of these poses is the right one?"** Given one ligand and ~79
candidate poses of it inside the same pocket, put the correct one on top. Reported as
**top1 %** — the fraction of the 285 targets where the highest-scored pose is within 2 Å
of the crystal structure. This is what a scoring function must do to be useful *during*
docking, and it is a question about geometry, not potency.

**Screening power — "which compound out of a library binds here?"** Given one target and
a library of 285 ligands of which only 5 actually bind it, rank the library. Reported as
**enrichment factor (EF)**: how many times better than random the method is at putting the
real binders on top.

> EF needs an example, because it is the least intuitive of the three. Testing the top 1%
> of the library means testing **3 compounds out of 285**. Picking 3 at random, you would
> expect 3 × (5/285) ≈ **0.05 actives** — almost always none. **EF1% = 1.0 means the
> method is no better than that coin flip.** EF1% = 2.9 means it finds 2.9× more actives
> than random; EF1% = 12.1, the best classical function, means 12×. The ceiling at top 1%
> is 100. In bench terms: with EF1% = 4.0 you find an active in roughly 20% of targets by
> testing three compounds each.

A model can be excellent at one and poor at another, and this one is — which is most of
what this README is about.

---

## Results

| | `lite` (1 model, 144k params) | `full` (6 models, 1.3M) |
|---|---|---|
| **scoring** — Pearson R | **0.752 ± 0.013** · 2nd of 35 | 0.819 · above 1st¹ |
| **docking** — top1 % | 78.5 ± 1.7 · 14th of 35 | **83.9** · **8th of 35** |
| **screening** — EF1% | 4.00 · 14th of 35 | **7.06** · **8th of 35** |
| RMSE ($pK_d$) | 1.470 ± 0.042 | 1.357 |
| slope, predicted vs. true | 0.458 ± 0.044 | 0.473 |
| inference cost | 1× | 6× |

¹ `full` is an **ensemble**; all 34 reference functions are single models, so that cell
is not a like-for-like comparison and is deliberately not reported as "1st".

`lite` figures are mean ± sd over three seeds. All numbers are on the CASF-2016 core set —
285 complexes that never entered training or checkpoint selection.

---

## Scope

The project began as a pharmacophore-explainability experiment — hence the old `PharmXAI`
name — and became a scoring function along the way.

**What it is.** You give it a pocket and a ligand already posed inside it; it returns an
affinity estimate in milliseconds.

**What it is not.**

- **Not a docking program.** It does not *search* for poses. It got good at *recognising*
  a correct one — 8th of 35, up from 30th before the physics constraints — but the
  candidates must come from smina, Vina or Glide. It is a **re-scorer**.
- **Not an absolute affinity predictor.** A predicted $K_d$ can be off by a factor of
  tens. Rank with it; do not report a number from it.
- **Not a pharmacophore tool.** The gradient maps it produces audit the model; they are
  not database search queries. See *Model interpretation*.
- **Not competitive as a virtual screening tool.** deltaVinaRF20 beats it on all three
  capabilities. Read *What we found* before adopting it for anything.

---

## What we found

Three findings, in the order they were measured. The full bench record, including the
negative results and the measurement traps, is in `EXPERIMENTOS.md` (Portuguese).

### 1. Physical constraints in the loss fixed pose selection

The first measurement was bad: **30th of 35** at docking power, 47.0% top1, barely above
a baseline that just counts atomic contacts. The cause was not subtle — every complex in
PDBbind is a *correct* pose, so nothing in training ever asked the network to put a
minimum at the native geometry.

Splitting the deficit by scale located it precisely. The model already **beat AutoDock
Vina** at the global trend (rho 0.374 vs 0.334) and lost 2.5× within 3 Å of the native
pose (0.249 vs 0.614). It could separate a plausible pose from an absurd one and could not
choose between two plausible ones.

Two constraints, on a **separate head** so the affinity output was never touched:

- **Stationarity** — at the crystallographic pose the net force and torque on the ligand
  must vanish. This is the variational condition an energy satisfies at a minimum, and it
  is the closest well-posed analogue here to a PINN's PDE residual. (There is no PDE whose
  solution is $pK_d$: it is a *free* energy, with entropy and desolvation inside.)
- **Ranking** against rigidly perturbed poses, with the graph edges held fixed so contact
  count is *identical* between native and perturbed — closing the counting shortcut before
  it can be exploited.

Neither works alone: **stationarity by itself is degenerate**, because a constant function
zeroes force and torque perfectly, and the pose head starts out almost constant.

**Result: 47.0% → 78.5% top1 (83.9% ensemble), with R unchanged** (0.754 → 0.752, within
seed noise). Three controls: the *affinity* head of the same checkpoint scores 40.4%, so
the gain is not diffuse; the contact-count baseline stays at 30.9%, so it is not a size
shortcut; removing the crystal pose from the candidate set still gives 75.8%, so the model
judges docked geometry rather than recognising the crystal.

### 2. Richer geometry did not help — and that is the informative part

The obvious next step was to let the network see **angles**, not just distances. Hydrogen
bonds are directional: N–H···O is worth a lot at 180° and almost nothing at 90°, at the
same distance. The model was blind to that.

Equivariant PaiNN-style layers were added (SE(3) invariance verified to 2.2e-08). Across
**three paired seeds**, the effect on R, slope, RMSE, top1 and top3 was **null**. Only
pose-ranking rho survived (+0.095, p = 0.037), which is not the metric that matters.

And not for lack of use: ablating the vector channels shifts predictions by **2.14 $pK_d$**.
The network consumes the angular information intensively and does not convert it into
accuracy.

Put beside the previous section, this is the most transferable thing the project produced:

| | fine discrimination (rho within 3 Å) |
|---|---|
| baseline | 0.249 |
| + richer representation (angles) | 0.236 |
| **+ constraint on the objective (physics)** | **0.600** |

**Giving the model more capacity did not work; changing what is asked of it did.** The
data and the expressiveness were already there; the objective was missing.

### 3. The model measures the molecule, not the pair

This one recontextualises the headline number.

Take the score the model gives each ligand **averaged over all 57 targets** — including
the 56 wrong ones — and correlate it with true affinity:

| | R |
|---|---|
| linear regression on 8 **ligand-only** descriptors (no protein at all) | **0.563** |
| the model, with no knowledge of which target is correct | 0.548 |
| the model, on its own target | 0.752 |
| molecular weight alone | 0.496 |

**Stripped of target identity, the model does not beat a linear regression over molecular
descriptors.** Of the 0.752, roughly 0.56 is a property of the compound and only ~0.19
comes from modelling the interaction.

Three independent lines of evidence agree:

1. **Variance ratio** — the affinity score varies 3.3× more across ligands than across
   targets (0.305); the pose score is the reverse (1.253), i.e. it measures the pair.
2. **The ligand-only baseline** above.
3. **Ensembling does not help it.** Six models leave the affinity head's EF1% exactly
   where it was (2.90 → 2.87) while nearly doubling the pose head's (4.00 → 7.06).
   Averaging cancels random error, not systematic error — so an ensemble is a free test of
   which kind dominates.

**The cause is the label, not the architecture.** PDBbind's $pK_d$ is always a ligand's
affinity for *its own* target. Nothing in training ever showed the same ligand against a
wrong target, so "potent molecule" and "pair that binds" are indistinguishable in the
data, and the model learns the easier one.

This applies to **any** method trained on PDBbind, not just this one. The test that
reveals it — comparing against a regression that uses only half the input — is trivial to
run and rarely reported.

#### The attempted fix, which failed

Ligand decoys were generated — the same pocket with a compound that does not bind it — and
added as a ranking loss. Before training, the model scored real binders at 6.22 and
non-binders in the same pocket at 5.99: **a gap of 0.23 $pK_d$**, the most direct
demonstration of the blindness.

After training, the gap tripled (+1.89) **and screening got worse** (EF1% 2.90 → 2.30,
R 0.752 → 0.725). That combination has one reading: the model learned to recognise the
*artefact* of how the decoys were built, not complementarity. Rigid perturbation
transferred in section 1 because it preserves the chemistry of the pair and only corrupts
geometry; transplanting a whole ligand corrupts placement so crudely that the artefact
dominates.

Realistic negatives would require redocking. Whether they would help is genuinely open:
if the blindness comes from the label, they might; if the task is intrinsically dominated
by ligand properties — and a linear regression reaching 0.563 suggests much of it is —
nothing fixes it.

---

## How to read these results

**As a tool**, the honest summary is narrow. There is one case where it is competitive:
ranking compounds by affinity when the poses are already given, where it is 2nd of 35 with
144k parameters, ahead of X-Score (0.631) and AutoDock Vina (0.604). The niche is thin,
because anyone holding poses has already run a docking program that returns a score.

For virtual screening it does not beat the alternatives, and the two biases above should
be weighed before trusting any of these numbers in production.

**As a study**, it answers a question worth asking: *what do physical constraints teach a
GNN, and what do they not?* One positive result, one negative result, and a diagnosis of
why the second failed. The measurement discipline that produced them — trivial baselines
that caught four separate shortcuts, paired seeds that killed a result which looked
positive in a single run — is in `EXPERIMENTOS.md`.

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

**Much of the accuracy is a property of the ligand, not of the interaction.** Stripped of
target identity the model scores R = 0.548, against 0.563 for a linear regression over
eight ligand-only descriptors. See *What we found* §3 — this is the limitation with the
widest consequences, and it applies to any model trained on PDBbind.

**The model overfits early.** Best validation arrives around epoch 17–19 of a possible
300, after which training loss keeps falling while validation rises. Note that rotational
data augmentation would achieve nothing here — the network is SE(3)-invariant, so its
output does not change under rotation. Coordinate noise would be the meaningful
perturbation.

**The slope has not moved**, and this is the most serious open problem. At 0.458 ± 0.044
the model compresses predictions toward the mean. It is not a calibration error that
rescaling can fix: under squared-error loss the optimal prediction *is* the conditional
mean, and since the slope of predicted-vs-true equals R·σ̂/σ, no linear rescaling pushes it
past **R** — forcing it to 1.0 only makes RMSE worse. **Improving the slope means
improving R.** The obvious attack, angular features, failed (see *What we found* §2).

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

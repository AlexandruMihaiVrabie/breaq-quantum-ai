# Fraud on a Quantum Budget: a VQC robust to depolarizing noise

BreaQ 2026 (24–25 October 2026, CAMPUS Research Institute, Bucharest), track **Quantum AI (PennyLane)**, Task 2.
Team of two. Credit card fraud detection (ULB, OpenML 42175) with a 4-qubit **Variational Quantum Classifier**, made robust to depolarizing noise and accompanied by a cost ledger for real hardware.

## At a glance

- **What we built:** a VQC with 4 qubits and 32 parameters (PennyLane + PyTorch), evaluated with density matrices under depolarizing noise p = 0, 0.001, 0.01, 0.05, over 5 seeds.
- **Headline result (test set, p = 0.05, AUPRC, mean ± std over 5 seeds):**

  | Model | AUPRC | Note |
  |---|---|---|
  | Baseline VQC (ZZ ×2, 33 CNOT) | 0.405 ± 0.177 | loses ~30% compared to p = 0 |
  | Simplified VQC (ZZ ×1, 21 CNOT) | 0.617 ± 0.036 | Exp. 3, evaluated on test in Exp. 5 |
  | **Final VQC (STAR ZZ, Z₀Z₁ readout, 15 CNOT)** | **0.709 ± 0.033** | no degradation: 0.704 at p = 0 |
  | Logistic regression, same 4 inputs | 0.703 ± 0.010 | classical reference, zero quantum cost |

- **What we do NOT claim:** quantum advantage. The final VQC **matches** logistic regression on AUPRC but does not beat it: LR finds more frauds at the threshold chosen on validation (79.2 vs 72.0 out of 98) and is more stable across seeds.
- **Main caveat:** all numbers come from exact simulation (expectation values, no shots). With 1024 shots and p = 0.05, the advantage of the Z₀Z₁ observable disappears (see Exp. 11 and 12). For real hardware, Z₀ is the safer choice.
- **The test set is not blind.** It was used in Exp. 5 and again in Exp. 12, after research on validation continued in between.

## Contents

1. [The task and how we covered it](#1-the-task-and-how-we-covered-it)
2. [Data and protocol](#2-data-and-protocol)
3. [The model](#3-the-model)
4. [The 12 experiments](#4-the-12-experiments)
5. [Final results on test](#5-final-results-on-test)
6. [Cost ledger](#6-cost-ledger)
7. [What did not work](#7-what-did-not-work)
8. [Limitations](#8-limitations)
9. [Repository structure](#9-repository-structure)
10. [Environment setup and reproduction](#10-environment-setup-and-reproduction)
11. [Team workflow](#11-team-workflow)

---

## 1. The task and how we covered it

Task 2 asks for a fraud-detection VQC that withstands depolarizing noise, while tracking cost on real hardware. Full statement: `docs/Fraud_on_a_Quantum_Budget_Task.pdf`.

| Mandatory requirement | Where it is covered |
|---|---|
| Baseline = the Task 1 VQC, evaluated with `evaluate_under_noise` on 5 seeds | Exp. 2 (Model A: ZZ ×2, ansatz ×3), test in Exp. 5 |
| At least one noise-oriented improvement on ≥ 5 seeds (threshold selection does not count) | Exp. 3 (ZZ ×2 → ×1), Exp. 9 (STAR ZZ), Exp. 11–12 (Z₀Z₁ readout), all on 5 seeds |
| Classical reference on the same inputs (at least logistic regression) | Exp. 4, test in Exp. 5 and 12 |
| Cost ledger (circuit executions and shots for training and validation, formula or `qml.Tracker`) | [Section 6](#6-cost-ledger), `src/cost_ledger.py`, `cost_ledger.json` files under `results/` |

Constraints respected: PennyLane for the quantum part, PyTorch for the training loop, scikit-learn only for preprocessing, baselines and metrics. Classification comes from the quantum circuit (the score is `(1 − ⟨O⟩)/2`, with no classical layer after measurement). Every transformation is fitted on train only. Dataset: OpenML 42175.

## 2. Data and protocol

**Dataset:** Credit Card Fraud Detection (ULB, OpenML 42175), 284,807 transactions, 492 frauds (0.1727%).

| Set | Transactions | Frauds | Used for |
|---|---:|---:|---|
| Train | 170,883 | 295 | training and preprocessing fit |
| Train VQC/LR, 1:1 undersampling | 590 | 295 | actual training, resampled per seed |
| Validation | 56,962 | 99 | choosing epoch, threshold, architecture |
| Test | 56,962 | 98 | final results only (Exp. 5 and Exp. 12) |

- **Split:** stratified 60/20/20, `random_state = 42`, saved in `split_indices.npz`. Every experiment checks the SHA-256 hash of the split.
- **Seeds:** models are trained with seeds 0–4 (numpy, torch, PennyLane). Seed 42 belongs to the split. The two are never mixed up.
- **Noise:** depolarizing channel of probability p applied after every native gate, on every qubit it touches, on `default.mixed` (density matrices). p ∈ {0, 0.001, 0.01, 0.05}. Native gates: `RX, RY, RZ, Rot, Hadamard, CNOT, CZ`.
- **Threshold:** maximum F1 on validation **at the same p**, applied unchanged on test.
- **Metrics:** AUPRC (primary, emphasis on p = 0.01 and 0.05), AUROC, Spearman ρ between the noisy and noiseless score, precision, recall, F1, MCC. Mean ± std (`ddof = 1`) over 5 seeds, with paired per-seed differences. We never report accuracy alone, since the data is extremely imbalanced.
- **No leakage:** `StandardScaler` (on `Time` and `Amount`), `PCA(4)` and `MinMaxScaler(0, π)` are fitted only on the 590 training examples. One experiment (Exp. 10) shows why this choice matters.

## 3. The model

```
transaction (30 features)
   │  StandardScaler(Time, Amount) → PCA(4) → MinMax [0, π]     (fit on the 590 train examples)
   ▼
4 angles x₀…x₃
   │  ZZ feature map:  H · RZ(2xᵢ) on each qubit,
   │                   then CNOT · RZ(2(π−xᵢ)(π−xⱼ)) · CNOT on the chosen pairs
   │  EfficientSU2-style ansatz: 4 RY+RZ layers, 3 CNOT ladders  → 32 parameters
   ▼
measure ⟨O⟩  →  fraud score  s = (1 − ⟨O⟩) / 2
```

Training: Adam, lr 0.05, BCE, batch 32, 30 epochs, epoch chosen by validation AUPRC at p = 0.

| Variant | Feature map | CNOT | Noise locations | Readout |
|---|---|---:|---:|---|
| **A** baseline (Task 1) | ZZ ×2, all pairs | 33 | 126 | Z₀ |
| **B** (Exp. 3) | ZZ ×1, all pairs | 21 | 88 | Z₀ |
| **STAR** (Exp. 9) | ZZ ×1, only pairs (0,1) (0,2) (0,3) | 15 | 73 | Z₀ |
| **STAR + Z₀Z₁** (Exp. 11–12) | same as STAR | 15 | 73 | Z₀Z₁ |

The number of parameters (32) is the same in every variant: the improvement comes from the circuit structure and from what we measure, not from extra capacity.

## 4. The 12 experiments

Convention: **val** = validation only, **test** = final evaluation. All on 5 seeds. Detailed reports are in `results/result_expN/report.md` (Exp. 8–12) and the notebooks in `notebooks/`.

| Exp. | Question | Result | Data |
|---|---|---|---|
| 1 | How many legitimate examples per fraud at training time? (1:1, 1:3, 1:5) | **1:1** wins: AUPRC 0.568 ± 0.026 vs 0.534 and 0.498, at a 2–3× lower training cost | val |
| 2 | Baseline A (ZZ ×2) under noise | 93 gates, 33 CNOT, 126 noise locations. Degrades strongly with p | val |
| 3 | Model B: ZZ ×2 → ×1 | −28% gates, −36% CNOT, −30% noise locations. B > A in 5/5 seeds at p = 0.05 | val |
| 4 | Logistic regression on the same 4 angles | The classical reference. Val AUPRC 0.663 ± 0.025, independent of noise | val |
| 5 | **Final evaluation A / B / LR** | See [Section 5](#5-final-results-on-test). LR > B at every seed and p. B > A: +0.212 ± 0.183 at p = 0.05 | **test** |
| 6 | Threshold for higher recall | 3–4 extra frauds cost 13–15 extra false alarms. Not a noise improvement (AUPRC does not depend on the threshold) | val |
| 7 | Ansatz ×3 → ×2 (24 parameters) | **Negative:** val AUPRC 0.587 → 0.520 at p = 0.05. Better ρ, but worse classification. B stays the solution | val |
| 8 | Rotation fusion (`RY·RZ → Rot`, `H·RZ → Rot`) | Exactly equivalent without noise (error 1e-15). −20 gates, −20 noise locations. AUPRC +0.0094 ± 0.0068 at p = 0.05 (5/5 seeds), ρ +0.048. A real but small gain | val |
| 9 | Sparse ZZ: STAR vs LINEAR vs FULL | **STAR:** −6 CNOT, −15 noise locations, AUPRC 0.622 vs 0.587 at p = 0.05 (4/5 seeds). The gain comes from better noiseless classification, not from smaller degradation. **LINEAR:** negative (0.542) | val |
| 10 | PCA fitted on the whole train set (170,883) instead of the 590 balanced examples | **Strongly negative:** AUPRC 0.622 → 0.089 at p = 0.05, in 5/5 seeds. We stay with balanced PCA | val |
| 11 | Which observable do we measure? Z₀, Z₁, Z₂, Z₃, Z₀Z₁ | Z₀Z₁ wins by the rule fixed before the run: AUPRC 0.678 ± 0.028 vs 0.622 at p = 0.05 (4/5 seeds). But with 1024 shots it drops to 0.19 (Z₀: 0.52). The 75–80% recall target is not reached | val |
| 12 | **Final test evaluation: STAR + Z₀Z₁** | See [Section 5](#5-final-results-on-test). AUPRC does not degrade under noise; matches LR | **test** |

Note: `notebooks/` contains two files with the `exp8` prefix (`exp8_rotation_fusion`, and `exp8_foundation_checks`, which holds the shared checks: saved scores, the "alert budget" metric, the cost ledger) and two with the `exp11` prefix (`exp11_quantum_readout_optimization` and `exp11_finite_shot_verification`).

### What the chain taught us

1. **Fewer gates does not automatically mean better.** Ansatz ×2 (Exp. 7) and LINEAR (Exp. 9) shrink the circuit and lose on classification.
2. **Cutting CNOTs helps, and noise shows up in the structure.** B and STAR reduce the CNOT count (the noisiest gate) from 33 to 21 and then 15.
3. **Less noise in the scores does not mean a better AUPRC.** A uniform contraction toward 0.5 does not change the score ordering. AUPRC drops only when noise reorders the top of the ranking (Exp. 9).
4. **Preprocessing matters as much as the circuit.** A PCA fitted on the real population ruins everything (Exp. 10).
5. **A threshold does not transfer between circuits.** Two circuits that are equivalent without noise have differently contracted scores under noise (Exp. 8).

## 5. Final results on test

Official test set: 56,962 transactions, 98 frauds. Mean ± std over 5 seeds. Threshold from validation at the same p.

### AUPRC

| Model | p = 0 | p = 0.01 | p = 0.05 |
|---|---|---|---|
| A · baseline VQC, ZZ ×2 (Exp. 5) | 0.580 ± 0.053 | 0.564 ± 0.049 | 0.405 ± 0.177 |
| B · VQC ZZ ×1 (Exp. 5) | 0.639 ± 0.022 | 0.635 ± 0.015 | 0.617 ± 0.036 |
| STAR + Z₀ (Exp. 12) | 0.690 ± 0.030 | 0.687 ± 0.033 | 0.661 ± 0.058 |
| **STAR + Z₀Z₁ (Exp. 12)** | 0.704 ± 0.028 | **0.705 ± 0.028** | **0.709 ± 0.033** |
| Logistic regression (Exp. 4–5) | 0.703 ± 0.010 | 0.703 ± 0.010 | 0.703 ± 0.010 |

### Threshold metrics, p = 0.05

| Model | Recall | Precision | F1 | MCC | TP / 98 | FP |
|---|---|---|---|---|---|---|
| A | 0.622 ± 0.158 | 0.487 ± 0.198 | 0.537 ± 0.184 | 0.545 ± 0.177 | 61.0 | 82.6 |
| STAR + Z₀ | 0.720 ± 0.049 | 0.772 ± 0.035 | 0.744 ± 0.031 | 0.745 ± 0.031 | 70.6 | 21.0 |
| **STAR + Z₀Z₁** | 0.735 ± 0.044 | 0.768 ± 0.061 | 0.749 ± 0.032 | 0.750 ± 0.031 | 72.0 | 22.4 |
| Logistic regression | 0.808 ± 0.013 | 0.776 ± 0.029 | 0.791 ± 0.019 | 0.791 ± 0.018 | 79.2 | 23.0 |

### Paired AUPRC comparisons, p = 0.05 (seeds in which the first model is better)

| Comparison | Δ AUPRC |
|---|---|
| STAR + Z₀Z₁ − A (baseline) | **+0.304 ± 0.199 (5/5)** |
| STAR + Z₀Z₁ − STAR + Z₀ | +0.048 ± 0.064 (3/5) |
| STAR + Z₀Z₁ − logistic regression | +0.006 ± 0.033 (3/5), a tie |

### What we can and cannot say

**We can say:**
- Compared with the initial baseline, AUPRC at p = 0.05 rises from 0.41 to 0.71 in 5/5 seeds, and false alarms drop from ~83 to ~22.
- With the same circuit, changing only the measured observable (Z₀ → Z₀Z₁, zero extra gates), AUPRC does not degrade between p = 0 (0.704) and p = 0.05 (0.709).
- On AUPRC, the final VQC matches logistic regression on the same 4 inputs.

**We cannot say:**
- Quantum advantage over the classical model.
- That Z₀Z₁ robustly "beats" Z₀: it wins in 3/5 seeds on test, and the mean (+0.048) is below the std of the differences (0.064). The gain comes mostly from seeds in which Z₀ degrades a lot.
- That the test set is an independent validation (see [limitations](#8-limitations)).
- That the results transfer to 1024 shots: for real hardware our recommendation is Z₀ (Exp. 11, §7 and §10).

### The 18 frauds nobody catches

At p = 0.05, 18 of the 98 frauds are missed by Z₀Z₁, Z₀ **and** LR, identically in all 5 seeds. They are indistinguishable from legitimate transactions in the 4 PCA components. The limit comes from the representation (4 angles), not from the circuit and not from noise.

## 6. Cost ledger

Convention: **nominal** circuit executions and **nominal** shots (1024 per execution), kept separate from simulation time. Simulation is exact (`shots=None`), so no shot is actually drawn; the figures estimate what it would cost on hardware. Implementation: `src/cost_ledger.py`. Checked against `qml.Tracker`.

**Formulas (per seed):**
- Training (parameter shift): `epochs × N × (1 + 2P)` = 30 × 590 × 65 = **1,150,500** nominal executions (P = 32 parameters). With `qml.Tracker`, only parameters inside the causal cone count: 41 executions per example, i.e. 725,700.
- Validation after each epoch: 30 × 56,962 = **1,708,860**.
- Final validation (p = 0 once, plus 3 noise levels): **227,848**.
- **Training + validation total: 3,087,208 nominal / 2,662,408 with Tracker.**

| | Executions / seed | Shots / seed (× 1024) |
|---|---:|---:|
| Training + validation, A / B / STAR / Z₀ | 3,087,208 (nominal) | ~3.16 bn |
| Training + validation with Tracker, Z₀ | 2,662,408 | |
| Training + validation with Tracker, Z₀Z₁ | 2,874,808 (+8.0%) | |
| Inference on test, a single p | 56,962 | 58,329,088 |
| LR (classical) | 0 | 0 |

**What changes and what does not between variants:**
- The number of executions and shots does **not** change between A, B, STAR and Z₀: it depends on parameters, examples and epochs, not on gates.
- The **cost of each execution** changes: A → B → STAR reduces CNOTs from 33 to 21 and 15, i.e. −29% CNOT applications relative to B (46.3 M vs 64.8 M per seed). Noisy simulation time drops by ~20%.
- Z₀Z₁ has a more expensive gradient (53 vs 41 executions per example, because its causal cone contains 26 parameters instead of 20). Inference costs the same: Z₀ and Z₀Z₁ are read from the same bitstrings.

## 7. What did not work

Negative results, kept in the report because they changed our decisions:

- **Exp. 6, threshold for high recall:** 75–80% recall requires hundreds or thousands of false alarms. The threshold is not a noise improvement.
- **Exp. 7, ansatz ×2:** smaller circuit, AUPRC lower by 0.07–0.08.
- **Exp. 9, LINEAR:** same 15 CNOT as STAR, but AUPRC 0.542 at p = 0.05 and a 2.4× larger loss under noise.
- **Exp. 10, PCA on the whole train set:** AUPRC 0.622 → 0.089, recall 0.677 → 0.168, in 5/5 seeds.
- **Exp. 11, the 75–80% recall target:** not reached by any observable (maximum 69.5% on validation). Z₂ is worse than Z₀ at every noise level. "Smaller causal cone ⇒ more robust" is not confirmed.
- **Exp. 11, finite shots:** the Z₀Z₁ advantage disappears at 1024 shots (AUPRC 0.19 vs 0.52 for Z₀ at p = 0.05). It returns only at ~10⁵ shots.
- **Exp. 12, versus LR:** logistic regression finds more frauds at the threshold in 5/5 seeds.

## 8. Limitations

- **The test set was consulted twice** (Exp. 5 for A, B, LR; Exp. 12 for STAR). Research on validation continued in between. All variants evaluated on test are reported, not just the best one.
- **98 frauds in test:** one moved fraud is ~1 point of recall. Differences of 1–2 frauds are below the level of statistical noise.
- **5 seeds** means 5 trainings and subsamplings, not 5 independent splits. 3/5 wins are not conclusive.
- **Exact simulation, not shots.** Conclusions about Z₀Z₁ do not transfer to 1024 shots. The shot analysis was done on validation only and post hoc.
- **Optimistic thresholds:** max-F1 is chosen and evaluated on the same validation set. On test, precision drops for all models and FP rises by ~4–5.
- **Test is easier than validation:** all models, including LR, have an AUPRC 0.03–0.04 higher on test. This is not evidence of better generalization.
- **Noise model:** depolarizing only, the same p on all gates. We do not model readout errors, crosstalk or coherence times.
- **Scores are not calibrated probabilities** (trained on 1:1, the real prevalence is 0.17%).
- **Numerics:** Exp. 5 ran in float32, Exp. 9–12 in float64. The difference in scores is ≤ 1e-7 and changes no ranking.

## 9. Repository structure

```
.
├── README.md
├── CLAUDE.md                  # project working rules
├── PR_GUIDE.md                # checklist for author and reviewer
├── requirements.txt           # dependencies (pip freeze)
├── split_indices.npz          # shared 60/20/20 split, seed 42
├── data/                      # creditcard.csv (NOT in git)
├── docs/
│   ├── Fraud_on_a_Quantum_Budget_Task.pdf    # Task 2 statement
│   ├── regulament_participare.pdf
│   ├── experimente_proiect.md                # technical handoff, Exp. 1–7
│   └── STORIES.md                            # plan and log for extensions
├── notebooks/                 # one notebook per experiment (exp1 … exp12) + task1
├── notes/                     # working ideas
├── results/result_expN/       # config.json, reports, scores, weights, plots
└── src/                       # reusable code, imported by the notebooks
    ├── circuits.py            # VQC (A/B), AngleEncoder, scoring under noise
    ├── star_pca.py            # STAR ZZ and balanced vs full-train PCA (Exp. 10)
    ├── quantum_readout.py     # observables, training, scoring, causal cone (Exp. 11–12)
    ├── scores_io.py           # loading saved scores, split hash check
    ├── alert_budget.py        # "alert budget" metric (max TP at FP ≤ k)
    ├── cost_ledger.py         # executions and shots for training, validation, inference
    └── report.py              # mean ± std, paired tables
```

Each `results/result_expN/` holds `config.json` (the frozen configuration; it refuses to overwrite a different configuration), `report.md`, the trained weights (`weights/`), the scores (`scores/` or `test_scores/`), `train_runs.jsonl` and plots. `*.csv` files are git-ignored and are regenerated by running the notebook, which resumes from the saved results.

## 10. Environment setup and reproduction

Python 3.12, PennyLane 0.45.1, torch 2.14.1, scikit-learn 1.9.1, pandas, numpy, scipy, matplotlib. Do not change versions without telling the team.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
nbstripout --install               # notebook outputs stay out of git
python -m ipykernel install --user --name breaq
```

Quick check:

```bash
python -c "import pennylane, torch, sklearn; print(pennylane.__version__, torch.__version__, sklearn.__version__)"
```

**Dataset:** not in git. After cloning, create the `data/` folder and put the OpenML 42175 file (or Kaggle `mlg-ulb/creditcardfraud`) at `data/creditcard.csv`.

**Reproducing the results:**
1. Run the notebooks in the order `exp1` … `exp12` (each resumes from what it already saved under `results/`, so re-runs are fast).
2. Exp. 12 needs the weights from `results/result_exp9/weights/star/` and `results/result_exp11/weights/Z0Z1/`.
3. All seeds are fixed (numpy, torch, PennyLane) and stated at the top of each notebook; each experiment checks the split hash and reproduces the previous experiment's numbers before comparing.
4. Quantum circuits are slow: simulating 5 seeds × 4 noise levels on validation takes tens of minutes per model. For a quick check, read the `report.md` in each results folder.

## 11. Team workflow

- Everyone works on their own branch (`alex-...`, `coleg-...`), never directly on `main`. Changes reach `main` through Pull Requests, and the author never merges their own PR: the teammate approves it, following `PR_GUIDE.md`.
- `git pull --ff-only` before starting work. Small, frequent commits and regular pushes.
- Different files and notebooks per person, to avoid conflicts.
- Reusable code lives in `src/`, not copy-pasted between notebooks.
- Never committed: `.venv/`, `data/`, `*.csv`, `.env`, keys or passwords.
- Commit messages: short, imperative, no trailers.

## Deliverables

- Runnable end-to-end code with fixed seeds: this repository.
- Presentation (`.pdf` / `.pptx`) uploaded within the 24 hours. 3-minute talk plus 2 minutes of questions.

# Fraud on a Quantum Budget — document de predare tehnică (Experimentele 1–7)

> **Scop:** document de handoff pentru colegul care integrează arhitectura finală în repository și pregătește livrabilul Task 2.
>
> **Stare la 24.10.2026:** Experimentele 1–7 realizate; **Exp. 3 este VQC-ul principal**, **Exp. 2 este baseline-ul obligatoriu**, **Exp. 4 este referința clasică**. Exp. 5 conține evaluarea finală pe test a A/B/C. Exp. 6–7 sunt analize exploratorii exclusiv pe validation. Nu există, în rezultatele furnizate, un model hibrid LR+VQC sau XGBoost deja implementat.
>
> **Proveniența cifrelor:** raportările experimentelor comunicate de echipă, fișierul JSONL cu 45 evaluări finale (recalculate statistic pentru valorile TEST), notebookul oficial `task1_guided_vqc_fraud_detection.ipynb` și PDF-ul `Fraud on a Quantum Budget - Task.pdf`. Pentru unele sumarizări Exp. 1–4, Exp. 6–7 avem rapoartele de rulare, nu toate fișierele originale ale acestor experimente; valorile trebuie verificate cu CSV/JSONL din repository la integrare.

---

## 0. Rezumat pentru coleg — în 60 de secunde

Am urmărit detectarea fraudelor pe **ULB Credit Card Fraud** folosind un **Variational Quantum Classifier (VQC) PennyLane**, evaluat fără zgomot și sub zgomot depolarizant. Datasetul are **284.807 tranzacții, 492 fraude (0,1727%)**. Metrică principală a competiției: **AUPRC** (Average Precision), nu accuracy.

**Contribuția principală:** am redus numărul de repetări ale **ZZ feature map de la 2 la 1**, păstrând **4 qubiți, ansatz ×3, 32 parametri**. La nivel de zgomot `p=0.05`, pe **test** și 5 seeds, AUPRC a crescut de la **0,405 ± 0,177** (baseline) la **0,617 ± 0,036** (VQC îmbunătățit), adică **+0,212 puncte / +52% relativ**. Scăderea față de propriul scor fără zgomot s-a redus de la **~30%** la **~3%**. Circuitul a trecut de la **93 la 67 porți native**, **33 la 21 CNOT**, **126 la 88 locații de zgomot**, adâncime **45 → 31**.

**Limită importantă:** **Logistic Regression (LR)**, antrenată pe **aceleași 4 unghiuri** ca VQC, este mai bună: **AUPRC pe test 0,703 ± 0,010**, fără execuții cuantice. **Nu afirmăm avantaj cuantic față de modelul clasic**. Scopul confirmat este *îmbunătățirea robusteții unui VQC*.

**Rezultat negativ util:** reducerea suplimentară a **ansatz-ului ×3 → ×2** (Exp. 7) scade AUPRC pe validation `0,587 → 0,520` la `p=0.05`, deși circuitul devine mai mic. Acest experiment este o ablație și **nu înlocuiește Exp. 3**.

## 1. Cerințele obligatorii ale Task 2

Conform PDF-ului oficial:

1. Un **baseline VQC** din notebookul Task 1, evaluat pe **minimum 5 seeds**.
2. Cel puțin **o îmbunătățire orientată spre zgomot**, evaluată identic, pe minimum 5 seeds.
3. O **referință clasică**, cel puțin regresie logistică, pe **aceleași intrări** ca VQC.
4. Un **cost ledger** (execuții de circuit și shots estimate pentru antrenare/validare și metoda proprie).
5. Raportare **AUPRC, AUROC, precision, recall, F1, MCC și Spearman ρ** la `p=0, 0.001, 0.01, 0.05`, medie ± deviație standard, în special `p=0.01` și `p=0.05`.
6. **Split comun** train/val/test, fără leakage. Selectarea arhitecturii, epocii și pragului pe training/validation; testul pentru evaluarea finală.
7. Componenta cuantică se implementează cu **PennyLane**, componenta clasică a buclei VQC cu **PyTorch**. `scikit-learn` este permis pentru preprocessing, metrici și baseline-uri clasice.
8. **Modelul raportat pentru Task 2 trebuie să rămână un VQC**: dacă se adaugă un ensemble LR+VQC, el este analiză auxiliară și nu înlocuiește demonstrarea clasificării care vine din circuitul cuantic.

### Distincții metodologice obligatorii

- **AUPRC** e independentă de un prag unic; TP/FP, precision, recall, F1 și MCC depind de prag.
- Pentru fiecare seed și nivel `p`, pragul utilizat pe test a fost ales prin **maximizarea F1 pe validation la același `p`**.
- `validation` are numai **99 fraude** și a fost folosit pentru alegerea celei mai bune epoci și a pragurilor: metricile obținute acolo pot fi optimiste.
- **Testul din Exp. 5 a fost deja consultat**. Exp. 6–7 sau ideile viitoare se raportează ca **exploratorii pe validation**, nu ca noi performanțe independente pe test. Nu retunați pe test.
- Pentru rezultatele raportate pe 5 seeds este vorba despre **5 antrenări/inițializări și subeșantionări**, nu despre 5 splituri de test independente.

## 2. Date, split și preprocessing

| Set | Total | Fraude (`Class=1`) | Legitime (`Class=0`) | Utilizare |
|---|---:|---:|---:|---|
| ULB original | 284.807 | 492 | 284.315 | Sursă |
| Train comun, înainte de balancing | 170.883 | 295 | 170.588 | Doar antrenare și fit preprocessing |
| Train VQC/LR, undersampling 1:1 | **590** | **295** | **295** | Antrenare efectivă per seed |
| Validation | 56.962 | 99 | 56.863 | Alegere epocă, prag, arhitectură |
| Test | 56.962 | 98 | 56.864 | Evaluare finală, Exp. 5 |

- **Split:** stratificat `60/20/20`, `SEED=42`, salvat ca `split_indices.npz`; `SAMPLE_FRACTION=1.0`.
- **Date:** `data/creditcard.csv`; dacă este necesară descărcare, OpenML **42175** (nu 1597, care nu are `Time`). Coloană-țintă: `Class`; caracteristici brute: `Time`, `Amount`, `V1...V28` (30 caracteristici numerice).
- **Undersampling:** se păstrează toate cele 295 fraude de train și se extrag aleator 295 legitime, cu seed-ul respectivei antrenări; validation/test NU se echilibrează.
- **`AngleEncoder(4, seed)`:** `StandardScaler` pentru `Time` și `Amount`, apoi `PCA(n_components=4, random_state=seed)` pe reprezentarea numerică, apoi `MinMaxScaler(feature_range=(0, π))`, cu `clip` pentru unghiuri. `fit` numai pe trainingul permis (în pipeline-ul experimental, subsetul echilibrat aferent seed-ului); transformarea pe validation/test folosește exact același encoder. Asigurați aceeași ordine a coloanelor și a indicilor.
- Pentru Exp. 4 s-a verificat reproductibilitatea unghiurilor: ponderile VQC din Exp. 2/3 pe unghiurile reconstruite au reprodus AUPRC la `p=0` cu diferență ≤ `1e-12`.

**Atenție:** cele **295 de fraude** sunt **date de train**. Când spunem „66 fraude detectate din 98”, aceasta este media **pe setul de test** la `p=0.05`, nu evaluarea pe cele 295 de fraude de antrenare. Extrapolarea `66/98 × 295 ≈ 199` este doar ilustrativă.

## 3. Arhitectura cuantică comună și zgomotul

### Baseline oficial (Model A, Exp. 2)

- **4 qubiți**, **4 unghiuri** PCA, ZZ feature map `FEATURE_MAP_REPS=2`.
- Ansatz tip **EfficientSU2**, cu alternanțe `RY`, `RZ` și entanglement liniar `CNOT(i, i+1)`, `ANSATZ_REPS=3`.
- Forma parametrilor: `(ANSATZ_REPS + 1, N_QUBITS, 2)` ⇒ `(4,4,2)` ⇒ **32 parametri**.
- Readout: valoarea așteptată `⟨Z₀⟩`, convertită în scor prin `(1−⟨Z₀⟩)/2`.
- Antrenare: `torch.optim.Adam`, `learning_rate=0.05`, `BCELoss`, `batch_size=32`, **30 epoci**, fără zgomot; se salvează ponderile epocii cu AUPRC validation maxim.
- Circuit decomposed: **93 porți native**, **33 CNOT**, **depth 45**, **126 noise locations**.

### Modelul îmbunătățit (Model B, Exp. 3) — alegerea principală

- **Singura schimbare intenționată față de Modelul A:** `FEATURE_MAP_REPS=1` (ZZ ×1).
- `ANSATZ_REPS=3`, **32 parametri**, 4 qubiți și același preprocessing și protocol.
- **67 porți native**, **21 CNOT**, **depth 31**, **88 noise locations**.
- Față de A: porți **−28%**, CNOT **−36%**, adâncime **−31%**, locații de zgomot **−30%**.

### Zgomot simulat

- `qml.transforms.decompose(..., gate_set=NATIVE_GATES)` cu setul `{RX, RY, RZ, Rot, Hadamard, CNOT, CZ}`.
- `qml.noise.insert(..., qml.DepolarizingChannel, p, position="all")`, după **fiecare poartă, pe fiecare qubit atins**; pentru CNOT sunt două canale.
- Simulator de matrice de densitate `default.mixed`; valori **exacte** ale observabilelor, fără shots finite efective în experimentele raportate.
- Niveluri `p ∈ {0, 0.001, 0.01, 0.05}`. Spearman `ρ` compară scorurile noisy și noiseless pe aceleași tranzacții.

## 4. Registrul experimentelor 1–7

### Experimentul 1 — selecția raportului de undersampling (validation, fără zgomot)

**Întrebare:** câte exemple legitime introducem la antrenare pentru fiecare fraudă? Au fost testate **1:1, 1:3, 1:5** (fraudă:legitime) cu 5 seeds și 30 epoci, folosind reprezentarea cuantică baseline. Encoderul a fost menținut fix per seed în comparație pentru a izola efectul eșantionării.

| Raport | Fraude train | Legitime train | Total train | AUPRC validation, medie ± std | F1 validation, medie ± std | Execuții training / seed, estimare |
|---|---:|---:|---:|---:|---:|---:|
| **1:1** | 295 | 295 | **590** | **0,568 ± 0,026** | **0,657 ± 0,035** | **1.150.500** |
| 1:3 | 295 | 885 | 1.180 | 0,534 ± 0,045 | 0,596 ± 0,062 | 2.301.000 |
| 1:5 | 295 | 1.475 | 1.770 | 0,498 ± 0,067 | 0,580 ± 0,104 | 3.451.500 |

**Output:** 1:1 s-a dovedit cea mai bună dintre cele **3 variante încercate**, atât pe AUPRC cât și pe cost. Alegere folosită în Exp. 2–4 și Exp. 7. Nu e o demonstrație că 1:1 este optim pentru toate modelele sau că orice utilizare a mai multor legitime ar fi inferioară.

**Referințe repo:** `notebooks/exp1_without_noise_baseline.ipynb`, `results/result_exp1/` (inclusiv directoarele de rulări `test1_full_without_noise` / `test1_without_noise` dacă există în checkout).

### Experimentul 2 — baseline VQC original (Model A), zgomot pe validation

**Întrebare:** cât se degradează modelul din Task 1 când aplicăm modelul oficial de zgomot? 5 seeds, 30 epoci, 590 exemple train/seed; antrenare fără zgomot, evaluare noisy după selectarea ponderilor.

| p | AUPRC validation (medie ± std) | F1 (medie ± std) | Spearman ρ (aprox.) |
|---|---:|---:|---:|
| 0 | 0,568 ± 0,026 | 0,657 ± 0,035 | 1,000 |
| 0.001 | 0,568 ± 0,029 | 0,657 ± 0,036 | ~1,000 |
| 0.01 | 0,546 ± 0,032 | 0,659 ± 0,034 | 0,982 ± 0,010 |
| **0.05** | **0,381 ± 0,172** | **0,534 ± 0,187** | **0,684 ± 0,146** |

**Output:** degradare majoră la `p=0.05`; seed 4 se prăbușește la **AUPRC ~0,085** pe validation. Arhitectura cu feature map ZZ ×2 nu este robustă.

**Artefacte:** `notebooks/exp2_noise_baseline.ipynb`; `results/result_exp2/weights/seed_0.pt` ... `seed_4.pt`, `val_runs.jsonl`, tabele de sumar, grafice și `cost_ledger.csv`.

### Experimentul 3 — feature map redus, VQC îmbunătățit (Model B), validation

**Ipoteză:** reducerea `FEATURE_MAP_REPS` de la **2 la 1** reduce expunerea la zgomot, păstrând capacitatea ansatz-ului.

| p | AUPRC Exp. 2 | **AUPRC Exp. 3** | Diferență Exp.3−Exp.2 |
|---|---:|---:|---:|
| 0 | 0,568 ± 0,026 | **0,627 ± 0,028** | +0,059 |
| 0.001 | 0,568 ± 0,029 | **0,627 ± 0,028** | +0,059 |
| 0.01 | 0,546 ± 0,032 | **0,623 ± 0,024** | **+0,077** |
| **0.05** | 0,381 ± 0,172 | **0,587 ± 0,039** | **+0,206** |

La `p=0.05`: **F1 Exp. 3 = 0,680 ± 0,030**, `ρ = 0,875 ± 0,110`, față de F1 ~0,534 și `ρ ~0,684` pentru Exp. 2. Avantaj AUPRC la **5 din 5 seeds** (`p=0.05`; și la `p=0.01`). Pierderea medie AUPRC `p=0 → 0.05`: Exp. 3 **~0,041**, Exp. 2 **~0,187**.

**Output:** candidata cuantică principală, confirmată ulterior pe TEST de Exp. 5. Nu se reduce numărul de execuții estimat după formula cu 32 parametri, ci **numărul și adâncimea porților pe fiecare circuit**.

**Artefacte:** `notebooks/exp3_reduced_feature_map.ipynb`; `results/result_exp3/weights/`, `train_runs.jsonl`, `val_runs.jsonl`, `comparison_vs_exp2.csv`, grafice și ledger.

### Experimentul 4 — regresie logistică, referință clasică (validation)

**Model:** `sklearn.linear_model.LogisticRegression(max_iter=1000)`, pe **aceiași 4 unghiuri** și aceeași subeșantionare 1:1 / seed. 5 seeds, convergență LBFGS în 22–31 iterații; nu este afectată de zgomotul cuantic.

| Metrică validation | Medie ± std |
|---|---:|
| **AUPRC** | **0,663 ± 0,025** |
| AUROC | 0,965 ± 0,001 |
| Precision | 0,802 ± 0,050 |
| Recall | 0,715 ± 0,008 |
| F1 | 0,755 ± 0,023 |
| MCC | 0,756 ± 0,024 |

**Output:** LR depășește Exp. 2/3 în medie pe validation și la toate nivelurile de zgomot. Exp. 3 rămâne aproape la `p=0` (diferență AUPRC **−0,036** în medie). Pragurile alese pe validation pentru LR sunt în jur de **0,75–0,83**, explicabil și prin antrenarea pe un set 1:1; scorurile LR **nu sunt automat probabilități calibrate** pentru prevalența reală de 0,17%.

**Artefacte:** `notebooks/exp4_logistic_regression.ipynb`; `results/result_exp4/val_runs.jsonl`, tabele și grafice. Nu s-a salvat inițial obiectul modelului, dar coeficienții au putut fi reproduși ulterior exact.

### Experimentul 5 — evaluarea finală pe TEST (45 evaluări)

**Configurații fixate înainte de test:** A = Exp. 2, B = Exp. 3, C = Exp. 4. 2 VQC × 5 seeds × 4 niveluri p = **40**; LR × 5 seeds = **5** ⇒ total **45**. Ponderile VQC au fost reutilizate; LR reconstruită cu coeficienți/intercept/prag reproduse exact. Nu s-a reantrenat VQC-ul. Validarea valorilor reproducea outputul din Exp. 2/3/4. Reexecutarea din cache a necesitat **0/45 recalculări**.

**AUPRC final pe test, medie ± std:**

| Model | `p=0` | `p=0.001` | `p=0.01` | **`p=0.05`** |
|---|---:|---:|---:|---:|
| A — VQC original, ZZ ×2 | 0,580 ± 0,053 | 0,580 ± 0,053 | 0,564 ± 0,049 | **0,405 ± 0,177** |
| **B — VQC optimizat, ZZ ×1** | **0,639 ± 0,022** | **0,638 ± 0,021** | **0,635 ± 0,015** | **0,617 ± 0,036** |
| C — Logistic Regression | 0,703 ± 0,010 | 0,703 ± 0,010 | 0,703 ± 0,010 | **0,703 ± 0,010** |

**Alte metrici la `p=0.05` (TEST):**

| Model | F1 | MCC | Spearman ρ | TP (din 98 fraude) | FP (din 56.864 legitime) | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| A — VQC original | 0,537 ± 0,184 | 0,545 ± 0,177 | 0,686 ± 0,145 | **61,0 ± 15,5** | **82,6 ± 69,0** | 0,622 ± 0,158 | 0,487 ± 0,198 |
| **B — VQC optimizat** | **0,691 ± 0,022** | **0,691 ± 0,022** | **0,876 ± 0,109** | **66,0 ± 4,1** | **27,0 ± 6,0** | **0,673 ± 0,041** | **0,712 ± 0,040** |
| C — Logistic Regression | **0,791 ± 0,019** | **0,791 ± 0,018** | — | **79,2 ± 1,3** | **23,0 ± 3,8** | **0,808 ± 0,013** | **0,776 ± 0,029** |

**Interpretare:**

- La `p=0.05`, Modelul B are **+0,212 ± 0,183 AUPRC diferență pereche medie** față de A, cu scor mai bun la **5/5 seeds**. La `p=0.01`, +0,071 ± 0,059, în **4/5 seeds** (seed 4 ușor favorabil baseline-ului la zgomot mic).
- AUPRC scade de la `p=0` la `p=0.05`: A **0,580→0,405** (−0,176 ≈ −30%); B **0,639→0,617** (−0,022 ≈ −3%).
- În cazul dificil **seed 4, p=0.05**, A a obținut `AUPRC=0,0967`, `TP=36`, `FP=203`, în timp ce B: `AUPRC=0,6194`, `TP=65`, `FP=34`.
- **LR rămâne superioară ambelor VQC-uri**: AUPRC **0,703**, TP medii **79,2**, FP medii **23**. Nu susținem că VQC-ul depășește clasicul.
- Test A/B/C a fost deja văzut: folosiți aceste rezultate ca **evaluare finală încheiată**, nu ca buclă de selecție pentru experimente ulterioare.

**Artefacte:** `notebooks/exp5_final_test.ipynb`; `results/result_exp5/` cu rezultate pe seed, tabele statistice, comparații, grafice (`test_metrics_vs_p.png`, `degradation_and_stability.png`) și `scores/` (~40 MB, cache). **Nu comiteți automat cache-ul `scores/`**.

### Experimentul 6 — optimizarea pragului pentru recall (validation, exploratoriu)

**Obiectiv:** putem detecta mai multe fraude doar schimbând pragul, fără să reantrenăm VQC B? Testat pe validation, **99 fraude / 56.863 legitime**, 5 seeds, inclusiv `p=0,0.01,0.05`.

La **`p=0.05`**, rezultate medii:

| Strategie de prag | TP medii din 99 | FP medii | Precision medie | Recall mediu | Diferență față de F1-max |
|---|---:|---:|---:|---:|---|
| **Referință F1-max** | **63,6** | **24,6** | **0,725** | **0,642** | — |
| Max recall cu precision ≥60% | 66,8 | 37,8 | 0,641 | 0,675 | +3,2 TP, +13,2 FP |
| Max recall cu FP ≤50 | 67,0 | 39,4 | 0,632 | 0,677 | +3,4 TP, +14,8 FP |
| Recall țintă ≥75% | 75,0 | ~759,6 | 0,132 | 0,758 | +11,4 TP, ~+735 FP |
| Recall țintă ≥80% | 80,0 | ~1.996,6 | 0,042 | 0,808 | +16,4 TP, ~+1.972 FP |

**Output:** 3–4 fraude în plus costă ~13–15 alarme false în plus pe validation; pentru ≥75–80% recall numărul FP explodează. **AUPRC nu se schimbă prin modificarea pragului**, deci Exp. 6 **nu este îmbunătățirea cuantica anti-zgomot cerută de Task 2**. Valorile sunt optimiste: pragurile selectate și evaluate pe același validation. Exp. 6 **nu** a accesat testul.

**Artefacte:** `notebooks/exp6_threshold_recall.ipynb`; `results/result_exp6/strategies_per_seed.csv`, `strategies_summary.csv`, `tradeoff_vs_f1max.csv`, `pr_curves.png`, `tradeoff_frauds_vs_false_alarms.png`, `config.json`, `val_scores/` (~7 MB, regenerabil).

### Experimentul 7 — ansatz redus ×3 → ×2 (validation, ablație exploratorie)

**Ipoteză:** dacă și ansatz-ul e mai scurt, circuitul va fi mai ieftin și poate mai robust la zgomot. Singura modificare față de B: **`ANSATZ_REPS=2`**, cu `FEATURE_MAP_REPS=1` neschimbat. Ponderile inițiale sunt primele trei straturi din inițializarea Exp. 3 pentru perechile de seed; 5 seeds × 30 epoci, același dataset/split/encoder/noise model și prag F1-max pe validation. Test neaccesat.

**AUPRC și performanțe validation:**

| p | Exp. 3 AUPRC | Exp. 7 AUPRC | Exp. 3 F1 | Exp. 7 F1 | TP Exp. 3 | TP Exp. 7 | FP Exp. 3 | FP Exp. 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | **0,627 ± 0,028** | 0,553 ± 0,031 | **0,680** | 0,637 | 63,2 | 54,4 | 23,8 | 17,0 |
| 0.01 | **0,623 ± 0,024** | 0,541 ± 0,032 | **0,686** | 0,646 | 64,4 | 57,2 | 24,6 | 20,6 |
| **0.05** | **0,587 ± 0,039** | 0,520 ± 0,013 | **0,680** | 0,666 | 63,6 | 64,4 | 24,6 | 30,0 |

Diferența pereche AUPRC `Exp7−Exp3`: `p=0`: **−0,074 ± 0,045** (Exp3 câștigă 5/5), `p=.01`: **−0,081 ± 0,051** (Exp3 4/5), `p=.05`: **−0,067 ± 0,037** (Exp3 5/5).

**Structură și cost:**

| Indicator | Exp. 3 (feature ×1, ansatz ×3) | Exp. 7 (feature ×1, ansatz ×2) |
|---|---:|---:|
| Porți native | 67 | 56 |
| CNOT | 21 | 18 |
| Depth | 31 | 27 |
| Locații zgomot | 88 | 74 |
| Parametri trainabili | 32 | 24 |
| Parametri deplasați efectiv (`qml.Tracker`) | 20 | 12 |
| Parametri cu gradient nenul | 19 | 11 |
| Execuții training/seed, ledger teoretic | 1.150.500 | 867.300 |
| Execuții training/seed, măsurate `qml.Tracker` | 725.700 | 442.500 |
| Execuții training + validare/seed, ledger total | 3.087.208 | 2.804.008 |

La `p=0.05`, Spearman `ρ`: **0,875 ± 0,110** în Exp. 3 vs **0,978 ± 0,013** în Exp. 7. Exp. 7 este mai stabil ca ranking sub zgomot, dar **AUPRC și F1 sunt mai slabe**; AUPRC scade deja fără zgomot. Pierderea de capacitate/optimizarea mai grea sunt explicații **plauzibile**, nu cauzalitate dovedită.

**Output:** rezultat negativ important: **nu orice scurtare de circuit ajută**. **Exp. 3 rămâne modelul principal**, Exp. 7 se prezintă ca **ablație**. În Exp. 7 numărul execuțiilor estimat scade ~9% pentru training + validare, dar performanța AUPRC scade ~0,07–0,08.

**Artefacte:** `notebooks/exp7_reduced_ansatz.ipynb`, `results/result_exp7/`. Numele exacte ale tuturor CSV/PNG din acest director trebuie confirmate din checkout (raportul indică tabele de comparație, ledger, grafice și ponderi).

## 5. Matrice finală: ce intră în arhitectură și în prezentare

| Experiment | Rol | Intră în soluția obligatorie Task 2? | Date folosite pentru scorurile raportate |
|---|---|---|---|
| **1** | Alegerea subeșantionării 1:1 | Ca justificare de design | Validation |
| **2** | **Baseline oficial A, ZZ×2, ansatz×3** | **Da, obligatoriu** | Validation + TEST prin Exp.5 |
| **3** | **Contribuția B, ZZ×1, ansatz×3** | **Da, modelul principal** | Validation + TEST prin Exp.5 |
| **4** | **LR clasică C pe aceleași intrări** | **Da, referință obligatorie** | Validation + TEST prin Exp.5 |
| **5** | **Comparația finală A/B/C și ledger** | **Da, rezultatele oficiale** | TEST, 5 seeds |
| **6** | Ablație de prag/recall/FP | Anexă sau „ce nu a funcționat” | Doar validation |
| **7** | Ablație ansatz×2, rezultat negativ | Anexă/slide „trade-off” | Doar validation |

**Mesajul central pentru juriu:** simplificarea feature map-ului reduce complexitatea și îmbunătățește robustețea sub zgomot; o reducere suplimentară a ansatz-ului reduce costul, dar pierde AUPRC. LR rămâne un reper clasic mai puternic; concluziile noastre sunt despre **optimizarea VQC-ului**, nu superioritate cuantică generală.

## 6. Cost ledger: convenții care NU trebuie amestecate

Pentru **Exp. 2 și Exp. 3**, în convenția conservatoare a notebookului oficial (`1+2P` execuții pentru gradient și `P=32`), per seed:

| Categorie | Execuții / seed |
|---|---:|
| Train: `30 × 590 × (1 + 2×32)` | **1.150.500** |
| Validation după fiecare epocă: `30 × 56.962` | **1.708.860** |
| Evaluare finală validation la 4 valori `p`: `4 × 56.962` | **227.848** |
| Test final la 4 valori `p`: `4 × 56.962` | **227.848** |
| **Total VQC per seed în ledger final Exp. 5** | **3.315.056** |
| **Total VQC pentru 5 seeds** | **16.575.280** |
| **Shots nominale pentru 5 seeds la 1.024/execuție** | **16.973.086.720** |

Aceasta este estimarea **pentru un singur model VQC** (A sau B) pe 5 seeds; pentru ambele modelele totalurile se dublează. Nu confundați cu `qml.Tracker`, care numără mai puține evaluări de gradient când unele coordonate nu afectează citirea. Dacă se schimbă convenția, păstrați explicit atât formula conservatoare, cât și măsurătoarea Tracker; nu comparați totaluri construite pe convenții diferite.

**Simulator:** antrenare prin `backprop` și predicții exacte cu `default.qubit` / `default.mixed`, deci **nu au fost trase ~17 miliarde shots reale**. Timpii de simulare (raportați ~17 min A vs ~10 min B pentru scorarea noisy finală) nu sunt timpi hardware. Costurile sunt estimări ale rulării ipotetice pe dispozitiv cuantic.

**Exp. 7:** în comparația cu Exp. 3 pe training + validation, ledgerul raportat a fost **3.087.208 vs 2.804.008 execuții/seed**; nu adăugați un cost TEST pentru Exp. 7, fiindcă **nu a fost evaluat pe test**.

## 7. Propunere de structură software pentru integrare (NU presupune că aceste module există deja)

Păstrați notebookurile și artefactele istorice intacte. Colegul poate extrage funcțiile repetate într-un pachet reutilizabil, fără a altera numerica:

```text
repo/
├── data/creditcard.csv                  # dataset local, nu neapărat de urcat în Git
├── split_indices.npz                    # split comun FIX
├── notebooks/
│   ├── task1_guided_vqc_fraud_detection.ipynb
│   ├── exp1_without_noise_baseline.ipynb
│   ├── exp2_noise_baseline.ipynb
│   ├── exp3_reduced_feature_map.ipynb
│   ├── exp4_logistic_regression.ipynb
│   ├── exp5_final_test.ipynb
│   ├── exp6_threshold_recall.ipynb
│   └── exp7_reduced_ansatz.ipynb
├── results/result_exp1/ ... result_exp7/ # rezultate istorice imutabile
├── src/                                  # PROPUNERE, nu directoare deja confirmate
│   ├── data_pipeline.py                  # split, undersampling și AngleEncoder
│   ├── circuits.py                       # VQC A/B, feature_map, ansatz, init
│   ├── noise.py                          # noise qnode, p, native gates
│   ├── train.py                          # antrenare, istoric, selecție epocă
│   ├── evaluate.py                       # score, praguri validation, metrici
│   ├── classical.py                      # LR și controale compatibilitate
│   ├── cost_ledger.py                    # formula hardware vs qml.Tracker
│   └── config.py                         # config și seeds
├── scripts/
│   ├── run_task2.py                      # reconstruire rezultate A/B/C, dacă e necesar
│   └── verify_artifacts.py               # checks reproducibilitate
├── experimente_proiect.md                # ACEST FIȘIER
└── README.md
```

### Contracte de interfață recomandate

```python
# Pseudocod orientativ, NU implementare existentă
load_data_and_fixed_split() -> train_frame, val_frame, test_frame
fit_angle_encoder(train_balanced_seed, seed) -> encoder
transform_with_encoder(encoder, frame) -> ndarray[n, 4]
make_vqc(feature_map_reps: int, ansatz_reps: int = 3) -> circuit
load_weights(model_id: str, seed: int) -> parameter_tensor
predict_scores(model, encoder, weights, frame, noise_p: float) -> ndarray[n]
select_threshold(y_val, val_scores) -> float
compute_metrics(y, scores, threshold) -> dict
estimate_hardware_cost(config, n_train, n_val, n_test, shots=1024) -> dict
```

**Reguli de integrare:**

1. `seed=42` este seed-ul splitului, în timp ce **0–4** sunt seeds de antrenare/eșantionare. Nu le inversați.
2. Preprocesarea este **seed-specifică**; nu reutilizați același encoder pentru toate seeds, dacă vreți reproductibilitate.
3. Modelul principal este **B = feature map 1, ansatz 3**; A rămâne baseline; Exp. 7 **nu** îl înlocuiește.
4. `p=0` se scorează noiseless; `p>0` se scorează cu exact modelul oficial cu inserție de canale după porțile native.
5. Nu antrenați pe validation/test; nu selectați sau ajustați praguri pe test.
6. Pentru comparația cu LR folosiți **aceleași 4 unghiuri**, altfel comparația nu mai este echivalentă.
7. Nu pretindeți că semnalele scorurilor VQC/LR sunt probabilități calibrate (s-au antrenat pe 1:1).
8. Raportați metricile **macro pe seeds ca medie de metrici**, nu ca agregare de predicții între seeds. În mod special, mean(F1) nu este neapărat F1 din mean(TP/FP).
9. Salvați rezultatele pe seed, `p`, configurație, threshold și `val AUPRC`, plus codul/versiunea și parametrii pentru reproducere.
10. Verificați cu `sha256` că refactorizarea nu schimbă artefactele și cu teste numerice că predicțiile/metricile se reproduc (toleranțe explicite).
11. Rularea din cache a Exp. 5 a indicat `0/45` evaluări suplimentare; nu eliminați cache-ul până ce reproducibilitatea e demonstrată din configurație + ponderi.
12. **Nu împingeți implicit în Git** `creditcard.csv` (~144MB), `results/result_exp5/scores/` (~40MB) ori `results/result_exp6/val_scores/` (~7MB). Tratați-le prin `.gitignore` și instrucțiuni de regenerare, după acordul echipei.

### Fluxul end-to-end prezentabil

```text
ULB creditcard.csv (284.807, 492 fraude)
       │
       ▼
Split fix stratificat, seed 42 → train / validation / test (60/20/20)
       │
       ├─ TRAIN: undersampling 1:1 (295 fraude + 295 legitime per seed)
       │      └─ fit StandardScaler(Time,Amount) → PCA(4) → MinMax[0,π]
       │                      │
       │                      ├─ A: ZZ×2 + Ansatz×3, 4 qubiți → ⟨Z₀⟩ → scor
       │                      ├─ B: ZZ×1 + Ansatz×3, 4 qubiți → ⟨Z₀⟩ → scor
       │                      └─ C: Logistic Regression pe aceleași 4 unghiuri
       │
       ├─ VALIDATION: selectare epocă și prag F1-max per seed și p
       │
       └─ TEST: evaluare o dată, fără reantrenare, toate seeds și p
                       └─ AUPRC, AUROC, F1, MCC, recall, precision, Spearman ρ
```

## 8. Ce urmează și ce NU este încă implementat

- **Task 2 oficial:** modelul principal rămâne Exp. 3; finalizați integrarea, README, reproducibilitate, cost ledger și prezentare.
- **Scop suplimentar exprimat de mentor:** cât mai puține alarme false cu recall mare. Experimentul 6 a demonstrat pe validation că un simplu threshold pentru 75–80% recall ridică FP la sute/mii.
- **Posibile experimente viitoare, încă NERULATE în evidența actuală:** (a) weighted ensemble **LR+VQC**, (b) **LR → VQC în două etape** / intersecție de alerte, (c) **XGBoost/LightGBM** pe toate cele 30 de caracteristici, cu protocol separat. Acestea nu pot fi introduse în rezultatele oficiale ca și cum ar fi fost evaluate.
- Dacă se investighează viitoarele modele, comparați **TP maxim pentru buget fix FP≤3/5/10/20**, AUPRC și cost. Găsiți complementaritatea erorilor și comparați **întotdeauna cu LR singură** la același buget FP; regula AND poate reduce FP, dar și TP.
- **Ținta de 270 din 295** fraude + 4 FP reprezintă `recall≈91,5%` și `precision≈98,5%` într-un scenariu ipotetic, nu rezultat existent; echivalent pe validation ≈91 din 99 fraude, și nu există dovezi că este realizabilă cu setările curente.
- Fiindcă testul original a fost deja consultat, orice variantă nouă este **exploratorie** până la evaluare pe date independente noi.

## 9. Checklist pentru arhitectură, README și prezentare

- [ ] `README.md` include sursa datelor, versiuni, instalare și comandă de rulare end-to-end.
- [ ] Splitul fix `split_indices.npz` verificat; fără fitting pe validation/test.
- [ ] Modelele A/B/C și parametrii lor descriși exact; configurațiile nu sunt confundate.
- [ ] Cele 5 seeds, cele 4 valori `p` și modelul oficial de zgomot păstrate.
- [ ] Baseline și VQC îmbunătățit comparate în aceleași condiții.
- [ ] LR este comparator pe aceleași 4 unghiuri.
- [ ] Valorile finale prezentate explicit drept **TEST Exp. 5**; Exp. 1–4/6–7 drept **validation**.
- [ ] Media ± std și diferențele pereche pe seeds disponibile.
- [ ] Cost ledger defalcat training / validare / test; shots nominale distincte de shots reale în simulare.
- [ ] Un slide despre **ce nu a mers**: Exp. 7 și trade-off-ul FP din Exp. 6.
- [ ] Limitări clare: 98 fraude test, 5 seeds, fără garanții de semnificație statistică și fără avantaj cuantic demonstrat.
- [ ] Nu se raportează îmbunătățiri ulterioare selectate cu informație din test ca rezultate independente.

## 10. Referințe principale din proiect

1. **Organizatori:** `Fraud on a Quantum Budget - Task.pdf`, secțiunile *Task 2*, *Evaluation protocol*, *Deliverables*.
2. **Notebook oficial:** `notebooks/task1_guided_vqc_fraud_detection.ipynb` (sau calea efectivă confirmată în repository), secțiunile *Split, balance, encode*, *Training*, *Quantum hardware cost*, *Noise*, *Task 2*.
3. **Reproducibilitate și rezultate:** directoarele `results/result_exp1/` … `results/result_exp7/`, plus notebookurile asociate.
4. **Set de date:** ULB Credit Card Fraud / Kaggle `mlg-ulb/creditcardfraud` sau OpenML **42175**.

---

## 11. PROMPT DE PREDARE CĂTRE COLEGUL RESPONSABIL DE ARHITECTURĂ

> Următorul text poate fi lipit în Claude Code / ChatGPT într-o sesiune care are acces la repo. Recomand să i se furnizeze și acest fișier Markdown.

**Rol:** Ești inginerul responsabil de integrarea arhitecturii finale și a livrabilelor pentru Task 2 — *Fraud on a Quantum Budget*. Înainte de orice modificare, citește `experimente_proiect.md`, PDF-ul competiției, notebookul oficial și notebookurile/resultatele Experimentelor 1–7. Tratează documentul ca **hartă de navigare**, iar CSV-urile/JSONL-urile și codul din repository ca **sursa definitivă** pentru valorile care trebuie să fie exacte.

**Ce s-a realizat:** Exp. 1 a ales undersampling 1:1 (590 exemple train); Exp. 2 este VQC baseline ZZ×2 / ansatz×3; Exp. 3 este VQC îmbunătățit ZZ×1 / ansatz×3; Exp. 4 e regresia logistică pe aceleași 4 unghiuri; Exp. 5 este evaluarea finală pe TEST pentru A/B/C; Exp. 6 e analiza pragurilor de recall pe validation; Exp. 7 e ablația ansatz×2 cu rezultat negativ pe validation. Rezultatul central pe TEST la `p=0.05`: **AUPRC A=0,405±0,177; B=0,617±0,036; C=0,703±0,010** (5 seeds). **B** e soluția cuantică principală. Nu susține avantaj cuantic față de C.

**Sarcină:** proiectează unificarea codului pentru rularea A/B/C, separând clar încărcarea datelor și splitului, encoderul seed-specific, circuitul cuantic, antrenarea, zgomotul, evaluarea și cost ledger-ul. Păstrează formulele și funcțiile oficiale și reproducerea exactă a valorilor deja salvate. Scrie un `README` de reproducere și generează tabele/grafice pentru prezentare. Orice refactorizare trebuie verificată prin testele automate existente, diferență numerică de predicții și hash al artefactelor istorice. **Nu suprascrie Experimentele 1–7, nu retraina inutil, nu face tuning pe test și nu pretinde rezultate pentru modele care nu au fost implementate.**

**Livrabil pentru mine:** (1) schema arhitecturii, (2) lista modulelor și API-urilor, (3) fluxul de date și limitele de leakage, (4) reproducerile numerice A/B/C și ledger-ul costurilor, (5) checklist-ul de rulare și packaging, (6) planul prezentării pentru juriu cu un slide „negative findings”, (7) riscuri/limitări și ce rămâne de făcut. Prezintă rezultatele **validation** și **test** separat și notează explicit că Exp. 6/7 sunt exploratorii.


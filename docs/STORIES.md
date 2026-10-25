# STORIES – direcții de extindere după Exp. 1–7

Versiune: sâmbătă 2026-10-24, seara · Fază: 3 (stories și tasks) · Se pune în repo ca `docs/STORIES.md`.
Surse: notebook-urile `exp1`…`exp7`, `Fraud_on_a_Quantum_Budget_Task.pdf`, `CLAUDE.md`, `PR_GUIDE.md`. Cifrele citate vin din concluziile scrise în Exp. 5 și Exp. 7 (notebook-urile din Project knowledge nu au outputuri, deci nu am putut recalcula nimic).
Marcaje: **[fapt]** = scris în documente · **[presupunere]** = de verificat · **[recomandare]** = părerea arhitectului.

---

## 0. Legendă

Statusuri story: `TODO` · `IN PROGRESS` · `IN REVIEW` · `DONE` · `BLOCKED`
Marcaje task: `[ ]` de făcut · `[~]` în lucru · `[x]` gata
Un story este `DONE` doar când toate criteriile sunt bifate **și** codul este în `main` (PR verificat de celălalt coleg, conform `PR_GUIDE.md`).
Prioritate: `MUST` / `SHOULD` / `COULD`.

---

## 1. Unde suntem (fapte din Exp. 1–7)

| Exp. | Ce a măsurat | Ce a ieșit **[fapt]** |
|---|---|---|
| 1 | Raport fraude:legitime la antrenare (1:1, 1:3, 1:5), VQC baseline, fără zgomot, validation | Configurația aleasă pentru restul: **1:1** (Exp. 2). Cifrele nu sunt în fișierele citite. |
| 2 | Baseline A (ZZ×2, ansatz×3, 32 param.) sub zgomot, validation | 93 porți, 33 CNOT, 126 locuri de zgomot. |
| 3 | Model B: ZZ feature map ×1 (restul identic) | 67 porți, 21 CNOT, 88 locuri de zgomot; aceiași 32 de parametri. |
| 4 | Regresie logistică (LR) pe aceleași 4 unghiuri | Referința clasică obligatorie; cost cuantic zero. |
| 5 | **Evaluare finală pe test** A, B, C (=LR), 5 seeds, p ∈ {0, 0.001, 0.01, 0.05} | LR: AUPRC 0.703 ± 0.010; **LR > B la fiecare seed și la orice p** (B−C: −0.064 / −0.068 / −0.086 la p = 0 / 0.01 / 0.05). B > A: +0.071 ± 0.059 la p = 0.01 (4/5 seeds), +0.212 ± 0.183 la p = 0.05 (5/5). A pierde 0.176 AUPRC între p = 0 și 0.05 (−29%), B pierde 0.022 (−3%). ρ la p = 0.05: A 0.687 ± 0.145, B 0.876 ± 0.109. F1 la p = 0.05: A 0.537 ± 0.184, B 0.691 ± 0.022. Cost: 3.315.056 execuții/seed, identic pentru A și B. Circuitul B are −28% porți, −36% CNOT, −31% adâncime, −30% locuri de zgomot. |
| 6 | Alegerea pragului pentru recall mai mare, Model B, doar validation | Nu se numără ca îmbunătățire anti-zgomot (regulă din PDF). AUPRC nu depinde de prag. Cifrele nu sunt în fișierele citite. |
| 7 | Ansatz redus (×2, 24 param.) peste B, doar validation | **Rezultat negativ**: AUPRC −0.081 ± 0.051 la p = 0.01 și −0.067 ± 0.037 la p = 0.05 față de B; ρ la p = 0.05 mai bun (0.978 ± 0.013 vs 0.875 ± 0.110); cost antrenare −25% (formulă), antrenare + validare doar −9%. B rămâne soluția principală. |

Două lucruri care contează pentru tot ce urmează:

1. **Testul a fost deja folosit** (Exp. 5), iar Exp. 7 notează explicit că rezultatele de test au fost consultate. Orice variantă nouă evaluată pe test este o a doua privire. Regulile din secțiunea 3 (P4) există ca să nu devină cherry-picking.
2. **Scorurile salvate de Exp. 5** (`results/result_exp5/scores/{A|B|C}_seed{s}_p{p}.npz`, chei `val`, `test`, `seconds`) permit ca S9, S10a și S11 să nu ceară nicio simulare nouă sub zgomot. **[fapt: din codul Exp. 5]** Dacă `results/` nu e în git, colegul de pe Windows nu le are (vezi Q&A #3).

---

## 2. Risc de conformitate (citește înainte de a investi timp)

PDF-ul Task 2, secțiunea „Technical constraints”: *„The model has to stay a VQC: classical layers may prepare the inputs or rescale the circuit's output, but the classification must come from the quantum circuit.”* **[fapt]**

- **Direcția 1 (ansamblu LR + VQC)** și **Direcția 2 (LR filtrează, VQC verifică; eliminată din plan, fostul S10)**: decizia finală depinde parțial de LR. Textul nu spune explicit că e interzis, dar nici că e permis. **[presupunere]** Le tratăm ca **analize hibride suplimentare**, nu ca „îmbunătățirea anti-zgomot” obligatorie. Îmbunătățirea obligatorie rămâne B (ZZ×1), deja evaluată pe 5 seeds.
- **Direcția 3 (gradient boosting pe 30 de feature-uri; eliminată din plan, fostul S11)**: nu e un VQC. E o **referință extinsă** („plafonul clasic”). **Nu înlocuiește** referința clasică cerută de Task 2 (LR pe aceleași 4 unghiuri, deja făcută în Exp. 4).
- **Excepție conformă [recomandare] (eliminată din plan odată cu S10):** varianta 10b din S10 (validare mai ieftină la antrenare, folosind LR doar ca să aleagă pe ce tranzacții validăm) produce tot un VQC antrenat, deci intră în direcția „Cheaper training” din PDF și se poate prezenta ca îmbunătățire de cost.
- **Ce rămâne neacoperit [recomandare]:** direcții din PDF care sunt eligibile ca „îmbunătățire anti-zgomot” și nu apar în Exp. 1–7: noise-aware training, zero-noise extrapolation, SPSA, data re-uploading, alt readout. Dacă după S8–S10 mai e timp, acestea au valoare mai mare pentru jurizare decât S11.

---

## 3. Protocol comun de verificare (ca în Exp. 2–7)

Fiecare story de mai jos trebuie să respecte aceste verificări; criteriile de acceptare le citează ca `P1`…`P10`.

- **P1 – Split.** `sha256(split_indices.npz)` egal cu `split_sha256` din `results/result_exp5/config.json`; fraude 295 / 99 / 98; validation și test au câte 56.962 de tranzacții.
- **P2 – Modele fixe.** Ponderile A și B nu se reantrenează (în Exp. 10 doar varianta B a encoderului se reantrenează, în afara acestui protocol). Se verifică sha256 față de `results/result_exp5/config.json` și AUPRC-ul pe validation la p = 0 față de valoarea salvată (`atol=1e-12`).
- **P3 – Fit doar pe train.** Orice scaler, PCA, model sau hiperparametru nou se potrivește pe train. Parametrii noi (ponderi de ansamblu, praguri, hiperparametri GBM) se aleg **doar pe validation**.
- **P4 – Testul.** Nu se atinge în faza de dezvoltare. Fiecare configurație se **îngheață** (scrisă în `config.json` înainte) și se evaluează pe test **o singură dată**. Toate variantele evaluate pe test se raportează, nu doar cea mai bună, iar numărul total de evaluări pe test se declară în prezentare.
- **P5 – Seeds.** Seeds `[0, 1, 2, 3, 4]`; comparații **pereche pe seed**; medie ± std (`ddof=1`); numărul de seeds cu Δ > 0; toate seeds raportate, inclusiv cele nefavorabile.
- **P6 – Metrici.** AUPRC (accent pe p = 0.01 și 0.05), AUROC, Spearman ρ (doar pentru componenta cuantică), precision, recall, F1, MCC la pragul max-F1 de pe validation **la același p**. În plus, metrica „buget de alarme” (secțiunea 4).
- **P7 – Cost ledger.** Execuții nominale (antrenare: epoci × N × (1 + 2P); scorare: 1 execuție per tranzacție; 1024 shots per execuție), **separate** de timpul de simulare. Fiecare execuție se numără o singură dată.
- **P8 – Salvare.** Rezultatele se scriu incremental în `.jsonl`, iar `config.json` refuză suprascrierea unei configurații diferite (ca în Exp. 2–7).
- **P9 – Cod.** Funcțiile reutilizabile stau în `src/`, notebook-ul doar le apelează. `nbstripout` rămâne activ. Fiecare PR e verificat de celălalt coleg.
- **P10 – Concluzii.** Scrise **după** rulare, cu: ipoteza, rezultatul, ce ar fi însemnat un rezultat negativ, limitele (validation optimist; 98 de fraude pe test înseamnă că o fraudă mutată schimbă recall-ul cu ~1 pp).

---

## 4. Metrica „buget de alarme” (din poza cu cele 3 metode)

Poza cere „cele mai multe fraude la maximum 3, 5 sau 10 alarme false”. Definiție comună, implementată o singură dată în `src/alert_budget.py` (S8):

- Pentru un buget `k`: pe **validation** se alege pragul cu **TP maxim** sub constrângerea **FP ≤ k**; la egalitate, pragul cel mai mare (aceeași regulă ca `pick` din Exp. 6).
- Același prag se aplică **neschimbat** pe test. Se raportează TP, FP și recall **reale** pe test (FP poate depăși k pe test).
- `k ∈ {3, 5, 10}` (din poză) și `k = 50` (continuitate cu Exp. 6).
- **Limită [recomandare]:** la k = 3 pragul depinde de primele 3–4 poziții din clasament; varianța între seeds va fi mare. Se raportează mereu cu std și cu valorile per seed.
- Metrica este **suplimentară**: nu înlocuiește AUPRC, AUROC, ρ, F1, MCC cerute de juriu.

---

## 5. Rezumat

| ID | Story | Owner | Status | Prioritate |
|---|---|---|---|---|
| S0 | Setup mediu și repo | Alex | DONE | MUST |
| S8 | Fundație comună: scoruri salvate, buget de alarme, ledger de inferență | Alex | IN REVIEW | MUST |
| S9 | Direcția 1: ansamblu ponderat LR + VQC | Alex | TODO | SHOULD |
| S12 | Integrare în prezentare și tabel comparativ | Alex | TODO | MUST |
| E10 | **Experiment efectiv executat:** BALANCED vs FULL-TRAIN PCA pentru STAR ZZ (ablație de encoder; vezi nota de mai jos) | Alex | DONE (validation, neintegrat în `main`) | — |

**Nota despre numerotare (2026-10-25).** Blueprintul a folosit alte numere decât experimentele rulate efectiv. În repo: Exp. 8 = rotation fusion, Exp. 9 = ZZ feature map rar (STAR / LINEAR), iar **Exp. 10 = BALANCED PCA vs FULL-TRAIN PCA pentru STAR ZZ** (`notebooks/exp10_balanced_vs_full_train_pca.ipynb`, `results/result_exp10/`, `src/star_pca.py`). **S10 (LR + VQC în două etape) și S11 (gradient boosting) au fost eliminate din plan (2026-10-25)**, deci numele `exp10` / `exp11` nu mai sunt rezervate.

**Exp. 10 (executat, doar validation, 5 seeds):** STAR ZZ fix (15 CNOT, 73 locuri de zgomot). Singura schimbare: `PCA.fit` pe tot train-ul (170.883) în loc de cele 590 echilibrate; scalerele și VQC-ul rămân pe cele 590. Rezultat negativ: la p = 0.05, AUPRC 0.622 → 0.089, recall 0.677 → 0.168, FP 16.4 → 135, în toate cele 5 seeds. Recomandare: se păstrează BALANCED PCA. Testul nu a fost folosit. Detalii: `results/result_exp10/report.md`.

**Cine ce editează (fără suprapuneri):** Alex: `src/scores_io.py`, `src/alert_budget.py`, `src/report.py`, `src/cost_ledger.py`, `src/ensemble.py`, `src/star_pca.py`, notebook-urile `exp8`, `exp9`, `exp10`, `docs/`. Coleg: de redistribuit (S10 și S11 eliminate). Dacă unul dintre voi trebuie să modifice un fișier al celuilalt, o spune pe chat înainte.

**Plan de timp [recomandare, scenariul pesimist: final duminică 09:00]:**

| Interval (ora locală) | Alex | Coleg |
|---|---|---|
| 21:30–23:00 | S8 | (de redistribuit) |
| 23:00–01:30 | S9 | (de redistribuit) |
| 01:30–03:00 | S9 final pe validation, apoi înghețare | (de redistribuit) |
| 03:00–05:00 | evaluări finale pe test (P4), tabele | (de redistribuit) |
| 05:00–06:00 | S12 (slide-uri) | verificare încrucișată PR-uri |
| 06:00 | **înghețare cod** (cu ≥ 3 h înainte de 09:00) | |

Ceasurile se dau înapoi la 04:00 (devine 03:00): nu conta pe ora în plus. Dacă finalul real e 14:00, se mută doar înghețarea.

---

## S0 – Setup mediu și repo

**Owner:** Alex · **Status:** DONE · **Branch:** `main` · **Prioritate:** MUST · **Depinde de:** —
**Obiectiv:** mediu Python 3.12 cu dependențele din `requirements.txt`, dataset local în `data/creditcard.csv`, repo privat pe GitHub, colegul invitat.

**Criterii de acceptare:**
- [x] Mediu virtual funcțional (PennyLane 0.45.1, torch 2.14.1, scikit-learn 1.9.1)
- [x] `data/creditcard.csv` local, în `.gitignore`
- [x] Repo privat `breaq-quantum-ai`, colegul invitat
- [x] `split_indices.npz` generat (Exp. 3–7 îl verifică față de reconstrucția oficială)

---

## S8 – Fundație comună: scoruri salvate, buget de alarme, ledger de inferență

**Owner:** Alex · **Status:** IN REVIEW · **Branch:** `alex-exp8-foundation` · **Prioritate:** MUST · **Depinde de:** —
**Obiectiv:** o singură implementare, verificată, a încărcării scorurilor salvate, a metricii „buget de alarme” și a contabilității de cost, ca S9–S11 să se compare pe exact aceleași cifre și juriul să vadă cod curat în `src/` (nu funcții copiate între notebook-uri).

**Criterii de acceptare:**
- [x] P1 respectat; testul nu se încarcă (doar etichetele)
- [x] Din scorurile salvate se reproduc, la `atol=1e-9`: AUPRC și pragul max-F1 pe validation pentru A și B la toate (seed, p) față de `results/result_exp2/val_runs.jsonl` și `results/result_exp3/val_runs.jsonl`; AUPRC și F1 pe test față de `results/result_exp5/test_runs.jsonl`
- [x] `src/alert_budget.py` dă aceleași TP/FP ca `pick` din Exp. 6 pe scorurile B de pe validation (verificat pe cel puțin 3 (seed, p))
- [x] `src/cost_ledger.py` reproduce 1.150.500 execuții de antrenare și 1.708.860 de validare pe epoci (30 epoci, 590 exemple, 56.962 validare, P = 32)
- [x] Verificare de independență de batch: scorul lui B pe un subset de 5.000 de tranzacții (p = 0.01, seed 0) coincide cu scorul aceleiași tranzacții în setul complet, `atol=1e-8` (necesar pentru afirmațiile de cost din S10)
- [x] Tabel de referință „buget de alarme” pentru A, B, C pe **validation**, k ∈ {3, 5, 10, 50}, p ∈ {0, 0.01, 0.05}, 5 seeds, medie ± std
- [ ] PR deschis; reviewer: Coleg

**Tasks:**
- [x] S8-T1 – `src/scores_io.py`: încărcare scoruri val/test pentru A/B/C, etichete, verificare sha256 split (60 min) · owner: Alex · fișiere: `src/scores_io.py`
- [x] S8-T2 – `src/alert_budget.py`: `operating_points`, `threshold_at_fp_budget`, `alert_budget_table` (mutate din Exp. 6) (60 min) · owner: Alex · fișiere: `src/alert_budget.py`
- [x] S8-T3 – `src/report.py`: `mean_std`, tabel pereche (Δ medie ± std, seeds cu Δ > 0) (45 min) · owner: Alex · fișiere: `src/report.py`
- [x] S8-T4 – `src/cost_ledger.py`: cost antrenare, cost scorare pe N tranzacții, shots (45 min) · owner: Alex · fișiere: `src/cost_ledger.py`
- [x] S8-T5 – notebook `exp8_foundation_checks`: toate verificările de mai sus + independența de batch (75 min) · owner: Alex · fișiere: `notebooks/exp8_foundation_checks.ipynb`, `results/result_exp8/`
- [x] S8-T6 – tabelul de referință „buget de alarme” pe validation (45 min) · owner: Alex · fișiere: `results/result_exp8/`

**Note / riscuri:**
- Dacă `results/result_exp5/scores/` nu e în git, Alex trimite arhiva colegului (sau se regenerează: ~25–35 min după estimarea din Exp. 5).
- Verifică înainte să creezi fișiere în `src/`: poate există deja module cu aceleași nume.

---

## S9 – Direcția 1: ansamblu ponderat LR + VQC

**Owner:** Alex · **Status:** TODO · **Branch:** `alex-exp9-ensemble` · **Prioritate:** SHOULD · **Depinde de:** S8
**Obiectiv:** să aflăm, măsurat, dacă scorul VQC-ului B adaugă informație peste LR pe aceleași 4 unghiuri și cât din avantajul LR se păstrează la p = 0.01 și 0.05; pentru juriu contează ca analiză onestă a „valorii cuantice” (criteriul de gândire de cercetare), nu ca model final.

**Ipoteze**
- **H1 [presupunere]:** există o pondere `w` pentru LR, `0 < w < 1`, cu AUPRC pe validation mai mare decât LR singur, la p = 0.01 și 0.05, la cel puțin 4/5 seeds. Plauzibilitate: ZZ feature map codifică produse între features, pe care LR nu le vede.
- **H0:** `w* ≈ 1` la orice p, iar câștigul față de LR e mai mic decât std-ul între seeds ⇒ VQC-ul nu adaugă informație utilizabilă. **[fapt]** LR bate B la 5/5 seeds în Exp. 5, deci H0 este rezultatul mai probabil; se raportează ca rezultat negativ.
- **Ce înseamnă negativ:** „pe aceste 4 unghiuri, circuitul nu aduce informație complementară; plafonul e dat de intrări, nu de circuit.”

**Design (fără simulare nouă)**
- Scoruri: B sub zgomot (`B_seed{s}_p{p}.npz`) și LR (`C_seed{s}_p0.0.npz`), ambele deja salvate.
- Normalizare: funcția de repartiție empirică (ECDF) a scorurilor de **validation** ale aceluiași (model, seed, p), aplicată pe val și pe test. Transformare monotonă ⇒ `w = 0` reproduce B și `w = 1` reproduce LR (se verifică).
- Scor final: `s = w · F_LR(s_LR) + (1 − w) · F_B(s_B)`, `w ∈ {0, 0.1, …, 1}`.
- Alegerea lui `w`: pe validation, per p, după AUPRC mediu pe cele 5 seeds; verificare suplimentară cu 2 fold-uri stratificate în interiorul validation-ului (alegi `w` pe un fold, măsori pe celălalt) ca să vezi cât din câștig e supraadaptare la 99 de fraude.
- ρ pentru ansamblu = Spearman între scorul ansamblului cu zgomot și cel fără zgomot (B la p = 0 și același LR).

**Ablații:** (a) B singur, LR singur, A singur; (b) `w` fix 0.8 (cazul din poză) vs `w*`; (c) ansamblu cu A în loc de B.

**Cost [fapt/derivat]:** antrenarea B e deja plătită în Exp. 3; ansamblul adaugă 0 execuții de antrenare. Scorarea cere tot câte 1 execuție per tranzacție, deci costul de inferență este **egal** cu al lui B. Câștig de cost: niciunul; se spune explicit în ledger.

**Criterii de acceptare:**
- [ ] P1–P10 respectate
- [ ] `w = 0` reproduce AUPRC-ul lui B și `w = 1` pe cel al lui LR, la toate (seed, p) pe validation (`atol=1e-9`)
- [ ] Curba AUPRC(w) pe validation pentru p ∈ {0, 0.01, 0.05}, cu std între seeds și rezultatul pe fold-urile interne
- [ ] `w*` pentru fiecare p, scris în `config.json` **înainte** de evaluarea pe test
- [ ] Evaluare pe test o singură dată pentru `w*` și `w = 0.8`; raportate AUPRC, AUROC, ρ, precision, recall, F1, MCC și buget de alarme (k = 3, 5, 10, 50), medie ± std, comparate pereche cu B, LR și A
- [ ] Concluzie scrisă explicit: H1 susținută / respinsă / neconcludentă, cu motivul
- [ ] Ledger: execuții de scorare = cele ale lui B; antrenare nouă = 0
- [ ] Notă de conformitate în notebook: „analiză hibridă, nu model VQC” (secțiunea 2)

**Tasks:**
- [ ] S9-T1 – `src/ensemble.py`: ECDF pe validation, scor combinat, grilă de ponderi (60 min) · owner: Alex · fișiere: `src/ensemble.py`
- [ ] S9-T2 – notebook `exp9`: verificări `w = 0` / `w = 1`, curba AUPRC(w) pe validation (60 min) · owner: Alex · fișiere: `notebooks/exp9_weighted_ensemble.ipynb`
- [ ] S9-T3 – validare încrucișată internă pe validation (2 fold-uri stratificate) pentru `w*` (45 min) · owner: Alex · fișiere: `notebooks/exp9_weighted_ensemble.ipynb`
- [ ] S9-T4 – tabel buget de alarme pe validation pentru `w*`, `w = 0.8`, B, LR (45 min) · owner: Alex · fișiere: `results/result_exp9/`
- [ ] S9-T5 – înghețare `config.json` + evaluare unică pe test + tabele pereche (60 min) · owner: Alex · fișiere: `results/result_exp9/`
- [ ] S9-T6 – concluzii și ledger (30 min) · owner: Alex · fișiere: `notebooks/exp9_weighted_ensemble.ipynb`

**Note / riscuri:**
- La 98 de fraude pe test, diferențe de 1–2 puncte de recall sunt zgomot statistic.
- `w*` ales pe validation e optimist (aceeași limitare ca în Exp. 6).

---

## S12 – Integrare în prezentare și tabel comparativ

**Owner:** Alex · **Status:** TODO · **Branch:** `alex-s12-slides` · **Prioritate:** MUST · **Depinde de:** S8 (+ S9 și Exp. 10 în măsura în care sunt gata)
**Obiectiv:** o oră pentru slide-urile care acoperă extensiile, cu cifre luate **din fișierele de rezultate**, nu rescrise de mână.

**Criterii de acceptare:**
- [ ] Un tabel unic pe test: A, B, LR și fiecare extensie finalizată (S9, Exp. 10), la p = 0, 0.01, 0.05: AUPRC, ρ (unde are sens), F1, MCC, buget de alarme; medie ± std pe 5 seeds
- [ ] Un slide „Ce nu a mers”: Exp. 7 (ansatz redus) și rezultatele negative din S9 și Exp. 10 (FULL-TRAIN PCA), dacă apar
- [ ] Un slide de cost: ledger A, B și STAR (Exp. 9–10), cu execuții și shots
- [ ] Menționat numărul total de evaluări pe test (Exp. 5 + extensii) și faptul că testul a fost consultat în Exp. 5
- [ ] Marcat clar în prezentare ce este „model VQC” și ce este „analiză hibridă/clasică”
- [ ] Citări: Dinuț et al. (2026)

**Tasks:**
- [ ] S12-T1 – script care generează tabelul final din `results/result_exp5/`, `result_exp9/`, `result_exp10/` (60 min) · owner: Alex · fișiere: `docs/`
- [ ] S12-T2 – slide-uri pentru extensii și ce nu a mers (60 min) · owner: Alex · fișiere: `docs/`

---

## Q&A (întrebări clarificate)

| # | Întrebare | Răspuns | Sursa |
|---|---|---|---|
| 1 | Un ansamblu sau o cascadă cu LR respectă „clasificarea trebuie să vină din circuitul cuantic”? | **Deschis.** Textul permite straturi clasice care pregătesc intrările sau rescalează ieșirea; un scor amestecat cu LR nu e acoperit explicit. Le tratăm ca analize hibride. De întrebat mentorii/juriul. | PDF Task 2, „Technical constraints” |
| 2 | Avem voie XGBoost? | sklearn e permis pentru preprocesare, baseline-uri și metrici; XGBoost nu e în stack. Implicit folosim `HistGradientBoostingClassifier`. | PDF Task 2; `CLAUDE.md` (stack) |
| 3 | `results/` (scoruri, ponderi) e în git? | **Deschis.** Dacă nu, colegul de pe Windows are nevoie de arhiva cu `result_exp5/scores` și `result_exp3/weights`. | Verifică Alex (`git ls-files results/`) |
| 4 | Ora de final | **Deschis:** regulament duminică 09:00, site duminică 14:00. Planul de mai sus presupune 09:00. | Enunțul echipei |
| 5 | Se mai poate folosi testul după Exp. 5? | Da, dar o singură dată per configurație înghețată, cu toate variantele raportate (P4). | Recomandare, pe baza criteriului „no cherry-picking” din PDF |

## Decizii luate

| Data | Decizie | De ce |
|---|---|---|
| 2026-10-24 | Raport 1:1 pentru antrenare | Ales în Exp. 1 și păstrat în Exp. 2–7 |
| 2026-10-24 | B (ZZ×1, ansatz×3, 32 param.) = soluția principală | Exp. 5: robustețe mult mai bună decât A; Exp. 7 a respins ansatz×2 |
| 2026-10-24 | Extensiile S9 (și fostele S10–S11) sunt analize suplimentare, nu înlocuiesc îmbunătățirea obligatorie | Risc de conformitate (secțiunea 2) |
| 2026-10-25 | S10 (cascadă LR + VQC) și S11 (GBM) eliminate din plan; Exp. 10 = BALANCED vs FULL-TRAIN PCA | Decizie Alex; fără GBM, întrebarea cu XGBoost devine irelevantă |
| 2026-10-24 | **PROPUSĂ:** testul se folosește o singură dată per configurație înghețată | Testul a fost deja consultat în Exp. 5 |

## Jurnal de lucru

| Data/ora | Cine | Ce s-a făcut | Story/Task | PR |
|---|---|---|---|---|
| 2026-10-24 21:xx | Alex + Claude | Analiza Exp. 1–7, definirea S8–S12 | S8–S12 | — |
| 2026-10-25 | Alex + Claude | Exp. 10 executat: BALANCED vs FULL-TRAIN PCA pe STAR ZZ (reproducere Exp. 9 verificată, B reantrenat, 13 verificări trecute); rezultat negativ pentru B | E10 | — |

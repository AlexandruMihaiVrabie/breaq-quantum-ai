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

- **Direcția 1 (ansamblu LR + VQC)** și **Direcția 2 (LR filtrează, VQC verifică)**: decizia finală depinde parțial de LR. Textul nu spune explicit că e interzis, dar nici că e permis. **[presupunere]** Le tratăm ca **analize hibride suplimentare**, nu ca „îmbunătățirea anti-zgomot” obligatorie. Îmbunătățirea obligatorie rămâne B (ZZ×1), deja evaluată pe 5 seeds.
- **Direcția 3 (gradient boosting pe 30 de feature-uri)**: nu e un VQC. E o **referință extinsă** („plafonul clasic”). **Nu înlocuiește** referința clasică cerută de Task 2 (LR pe aceleași 4 unghiuri, deja făcută în Exp. 4).
- **Excepție conformă [recomandare]:** varianta 10b din S10 (validare mai ieftină la antrenare, folosind LR doar ca să aleagă pe ce tranzacții validăm) produce tot un VQC antrenat, deci intră în direcția „Cheaper training” din PDF și se poate prezenta ca îmbunătățire de cost.
- **Ce rămâne neacoperit [recomandare]:** direcții din PDF care sunt eligibile ca „îmbunătățire anti-zgomot” și nu apar în Exp. 1–7: noise-aware training, zero-noise extrapolation, SPSA, data re-uploading, alt readout. Dacă după S8–S10 mai e timp, acestea au valoare mai mare pentru jurizare decât S11.

---

## 3. Protocol comun de verificare (ca în Exp. 2–7)

Fiecare story de mai jos trebuie să respecte aceste verificări; criteriile de acceptare le citează ca `P1`…`P10`.

- **P1 – Split.** `sha256(split_indices.npz)` egal cu `split_sha256` din `results/result_exp5/config.json`; fraude 295 / 99 / 98; validation și test au câte 56.962 de tranzacții.
- **P2 – Modele fixe.** Ponderile A și B nu se reantrenează (excepție: S10-T6). Se verifică sha256 față de `results/result_exp5/config.json` și AUPRC-ul pe validation la p = 0 față de valoarea salvată (`atol=1e-12`).
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
| S8 | Fundație comună: scoruri salvate, buget de alarme, ledger de inferență | Alex | TODO | MUST |
| S9 | Direcția 1: ansamblu ponderat LR + VQC | Alex | TODO | SHOULD |
| S10 | Direcția 2: LR + VQC în două etape (10a inferență, 10b validare ieftină) | Coleg | TODO | SHOULD |
| S11 | Direcția 3: gradient boosting pe 30 de feature-uri (referință extinsă) | Coleg | TODO | COULD |
| S12 | Integrare în prezentare și tabel comparativ | Alex | TODO | MUST |

**Cine ce editează (fără suprapuneri):** Alex: `src/scores_io.py`, `src/alert_budget.py`, `src/report.py`, `src/cost_ledger.py`, `src/ensemble.py`, notebook-urile `exp8`, `exp9`, `docs/`. Coleg: `src/cascade.py`, `src/gbm_reference.py`, notebook-urile `exp10`, `exp11`. Dacă unul dintre voi trebuie să modifice un fișier al celuilalt, o spune pe chat înainte.

**Plan de timp [recomandare, scenariul pesimist: final duminică 09:00]:**

| Interval (ora locală) | Alex | Coleg |
|---|---|---|
| 21:30–23:00 | S8 | S11-T1…T3 (nu depind de S8) |
| 23:00–01:30 | S9 | S10a (S10-T1…T5) |
| 01:30–03:00 | S9 final pe validation, apoi înghețare | S10b (S10-T6…T7), S11-T4 |
| 03:00–05:00 | evaluări finale pe test (P4), tabele | idem pentru S10 și S11 |
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

**Owner:** Alex · **Status:** TODO · **Branch:** `alex-exp8-foundation` · **Prioritate:** MUST · **Depinde de:** —
**Obiectiv:** o singură implementare, verificată, a încărcării scorurilor salvate, a metricii „buget de alarme” și a contabilității de cost, ca S9–S11 să se compare pe exact aceleași cifre și juriul să vadă cod curat în `src/` (nu funcții copiate între notebook-uri).

**Criterii de acceptare:**
- [ ] P1 respectat; testul nu se încarcă (doar etichetele)
- [ ] Din scorurile salvate se reproduc, la `atol=1e-9`: AUPRC și pragul max-F1 pe validation pentru A și B la toate (seed, p) față de `results/result_exp2/val_runs.jsonl` și `results/result_exp3/val_runs.jsonl`; AUPRC și F1 pe test față de `results/result_exp5/test_runs.jsonl`
- [ ] `src/alert_budget.py` dă aceleași TP/FP ca `pick` din Exp. 6 pe scorurile B de pe validation (verificat pe cel puțin 3 (seed, p))
- [ ] `src/cost_ledger.py` reproduce 1.150.500 execuții de antrenare și 1.708.860 de validare pe epoci (30 epoci, 590 exemple, 56.962 validare, P = 32)
- [ ] Verificare de independență de batch: scorul lui B pe un subset de 5.000 de tranzacții (p = 0.01, seed 0) coincide cu scorul aceleiași tranzacții în setul complet, `atol=1e-8` (necesar pentru afirmațiile de cost din S10)
- [ ] Tabel de referință „buget de alarme” pentru A, B, C pe **validation**, k ∈ {3, 5, 10, 50}, p ∈ {0, 0.01, 0.05}, 5 seeds, medie ± std
- [ ] PR deschis; reviewer: Coleg

**Tasks:**
- [ ] S8-T1 – `src/scores_io.py`: încărcare scoruri val/test pentru A/B/C, etichete, verificare sha256 split (60 min) · owner: Alex · fișiere: `src/scores_io.py`
- [ ] S8-T2 – `src/alert_budget.py`: `operating_points`, `threshold_at_fp_budget`, `alert_budget_table` (mutate din Exp. 6) (60 min) · owner: Alex · fișiere: `src/alert_budget.py`
- [ ] S8-T3 – `src/report.py`: `mean_std`, tabel pereche (Δ medie ± std, seeds cu Δ > 0) (45 min) · owner: Alex · fișiere: `src/report.py`
- [ ] S8-T4 – `src/cost_ledger.py`: cost antrenare, cost scorare pe N tranzacții, shots (45 min) · owner: Alex · fișiere: `src/cost_ledger.py`
- [ ] S8-T5 – notebook `exp8_foundation_checks`: toate verificările de mai sus + independența de batch (75 min) · owner: Alex · fișiere: `notebooks/exp8_foundation_checks.ipynb`, `results/result_exp8/`
- [ ] S8-T6 – tabelul de referință „buget de alarme” pe validation (45 min) · owner: Alex · fișiere: `results/result_exp8/`

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

## S10 – Direcția 2: LR + VQC în două etape

**Owner:** Coleg · **Status:** TODO · **Branch:** `coleg-exp10-two-stage` · **Prioritate:** SHOULD · **Depinde de:** S8 (S10-T1 poate începe cu `sklearn` direct)
**Obiectiv:** să măsurăm dacă LR poate înlătura majoritatea tranzacțiilor evident legitime, astfel încât circuitul cuantic să fie rulat doar pe un subset, reducând execuțiile de scorare (criteriul „Cost” din PDF), fără să pierdem fraude față de LR singur la același buget de alarme.

Story-ul are două părți independente:
- **10a – cascadă la inferență** (hibrid, risc de conformitate, vezi secțiunea 2): LR alege candidații, B îi reordonează.
- **10b – validare ieftină la antrenare** (conformă): LR alege pe ce tranzacții de validation se scorează VQC-ul după fiecare epocă; modelul final rămâne un VQC antrenat. Țintește componenta care costă cel mai mult în ledger: **1.708.860 de execuții de validare pe seed** vs. 1.150.500 de antrenare **[fapt: PDF și Exp. 1]**.

**Ipoteze**
- **H1 (10a) [presupunere]:** la un recall al etapei 1 de 98–99% pe validation, B reordonează candidații suficient de bine încât, la același buget de alarme (k), recall-ul final ≥ recall-ul lui LR singur, cu mult mai puține execuții cuantice.
- **H0 (10a):** recall-ul final ≤ recall-ul lui LR singur (tavanul îl dă etapa 1; B nu adaugă informație, vezi S9), deci cascada doar economisește execuții fără câștig de detecție.
- **H1 (10b):** alegerea epocii pe subsetul de candidați duce la aceeași epocă sau la un AUPRC pe validation complet cu diferență sub 1 std față de modelul antrenat cu validare completă, cu costul de validare redus proporțional cu fracția de candidați.
- **H0 (10b):** epoca aleasă diferă și AUPRC-ul complet scade peste 1 std ⇒ validarea pe subset nu e sigură.
- **Necunoscută cheie [fapt: nu o știm]:** câte tranzacții de validation rămân candidate la recall 98–99%. Tot câștigul de cost depinde de asta; se măsoară în S10-T2, nu se estimează.

**Convenții (fixate înainte de rulare)**
- LR = cel din Exp. 4/5 (antrenat pe setul 1:1 al seed-ului). Prag etapa 1 `t1` = cel mai mare prag cu recall pe validation ≥ `r1`, `r1 ∈ {0.95, 0.98, 0.99}`. `t1` se aplică neschimbat pe test.
- Scor final pe toate tranzacțiile (necesar pentru AUPRC): candidații primesc `1 + s_B`, iar cei respinși primesc ECDF-ul scorului LR (în `[0, 1)`), deci sunt întotdeauna sub candidați. Convenția alternativă (respinșii = 0 constant) se raportează ca ablație, fiindcă AUPRC depinde de ea.
- Costul de inferență: `n_candidați` execuții per trecere (validation și test), nu 56.962.
- 10a reutilizează scorurile salvate ale lui B (corect doar dacă S8 confirmă independența de batch).

**Ablații:** (a) `r1`; (b) convenția pentru respinși; (c) B vs A ca etapă 2; (d) etapa 2 = ansamblul din S9 (doar dacă S9 e gata); (e) 10b: antrenare cu validare completă vs. validare pe candidați.

**Criterii de acceptare:**
- [ ] P1–P10 respectate
- [ ] Număr de candidați pe validation și pe test pentru fiecare `r1` (medie ± std pe seeds) și recall-ul real al etapei 1 pe test
- [ ] 10a: pentru fiecare `r1`, p ∈ {0, 0.01, 0.05}: AUPRC, AUROC, precision/recall/F1/MCC la prag max-F1 pe validation, buget de alarme (k = 3, 5, 10, 50), comparate pereche cu **LR singur** și cu B singur, la același buget de alarme
- [ ] 10a: ledger de inferență: execuții de scorare validation + test cu cascadă vs. fără, per nivel de zgomot și per seed
- [ ] 10b: model nou antrenat pe 5 seeds cu validare pe candidați; raportat: epoca aleasă vs. Exp. 3, AUPRC pe validation **complet** la p = 0, ρ față de modelul Exp. 3 (scoruri pe test fără zgomot), ledger de validare pe epoci (execuții/seed vs. 1.708.860)
- [ ] Configurația (`r1`, convenții) scrisă în `config.json` înainte de testul unic
- [ ] Concluzie explicită pe H1/H0, inclusiv dacă cascada **pierde fraude reale** față de LR singur (riscul din poză)
- [ ] Notă de conformitate: 10a hibrid; 10b VQC

**Tasks:**
- [ ] S10-T1 – `src/cascade.py`: etapa 1 (prag pe validation după recall țintă), scor final cu convenția de mai sus (60 min) · owner: Coleg · fișiere: `src/cascade.py`
- [ ] S10-T2 – notebook `exp10`: număr de candidați vs. `r1` pe validation (cifra cheie) (45 min) · owner: Coleg · fișiere: `notebooks/exp10_two_stage.ipynb`
- [ ] S10-T3 – 10a pe validation: metrici + buget de alarme vs. LR singur și B singur (75 min) · owner: Coleg · fișiere: `notebooks/exp10_two_stage.ipynb`
- [ ] S10-T4 – ledger de inferență pentru cascadă (45 min) · owner: Coleg · fișiere: `results/result_exp10/`
- [ ] S10-T5 – înghețare `r1` + evaluare unică pe test pentru 10a (60 min) · owner: Coleg · fișiere: `results/result_exp10/`
- [ ] S10-T6 – 10b: antrenare B cu validare pe candidați (5 seeds, aceleași ponderi inițiale și ordine de batch-uri ca în Exp. 3), salvare ponderi (90 min) · owner: Coleg · fișiere: `notebooks/exp10_two_stage.ipynb`, `results/result_exp10/weights/`
- [ ] S10-T7 – 10b: comparație pereche cu Exp. 3 pe validation, apoi test unic și ledger de validare (75 min) · owner: Coleg · fișiere: `results/result_exp10/`

**Note / riscuri:**
- Timp 10b: antrenarea fără zgomot a durat ~20–35 s/seed pentru 590 de exemple (estimare din Exp. 2); evaluările sub zgomot (15 treceri) au fost estimate la ~8–15 min în Exp. 2, deci 10b e fezabil în timp. **[fapt: estimări din notebook-uri, depind de mașină]**
- Dacă numărul de candidați la `r1 = 0.98` e mare (de ordinul zecilor de mii), 10a nu aduce economie reală: se raportează ca rezultat negativ și se trece la 10b.
- Modificarea setului de validare folosit pentru alegerea epocii schimbă o decizie de dezvoltare: pe test se evaluează o singură dată (P4).

---

## S11 – Direcția 3: gradient boosting pe 30 de feature-uri (referință extinsă)

**Owner:** Coleg · **Status:** TODO · **Branch:** `coleg-exp11-gbm` · **Prioritate:** COULD · **Depinde de:** S8 (doar T4; T1–T3 sunt independente)
**Obiectiv:** să arătăm cât din golful dintre VQC și clasic vine din **model** și cât din **intrări** (doar 4 unghiuri PCA), pentru a încadra onest rezultatele cuantice (recomandarea Bowles, Ahmed, Schuld 2024, citată în PDF).

**Statut [fapt]:** nu înlocuiește referința clasică obligatorie (LR pe aceleași 4 unghiuri, Exp. 4); în poză este descris tot ca „comparație extinsă, nu echivalentă cu baseline-ul oficial”.

**Decizie de dependență [recomandare]:** implicit `sklearn.ensemble.HistGradientBoostingClassifier` (fără dependențe noi; sklearn e permis pentru baseline-uri). XGBoost doar dacă ambii acceptați și `requirements.txt` se modifică printr-un PR separat anunțat echipei. Înainte de a scrie cod, verifică în documentația scikit-learn semnăturile și parametrii (`class_weight`, `early_stopping`, `random_state`) pentru versiunea 1.9.1; nu mă bazez pe memorie pentru API.

**Ipoteze**
- **H1 [presupunere]:** GBM pe 30 de feature-uri depășește clar LR pe 4 unghiuri la AUPRC pe validation ⇒ intrările (PCA la 4 componente) sunt gâtuirea principală, nu circuitul.
- **H0:** GBM pe 30 de feature-uri ≈ LR pe 4 unghiuri ⇒ 4 unghiuri conțin aproape tot semnalul disponibil.
- **Tabel 2×2 (separă cauzele):** {LR, GBM} × {4 unghiuri, 30 de feature-uri}, același set de antrenare și aceeași evaluare.

**Variante de date de antrenare (fixate înainte):** (a) setul 1:1 al seed-ului (590 de exemple, la fel ca VQC); (b) tot train-ul (170.883 de rânduri) cu ponderare de clase. Hiperparametri: grilă mică, aleasă doar pe validation.

**Cost:** 0 execuții cuantice; antrenare pe CPU. Se raportează timpul de antrenare, separat de ledger.

**Criterii de acceptare:**
- [ ] P1, P3, P4, P5, P6, P8, P9, P10 respectate (P2, P7 nu se aplică: nu există circuit)
- [ ] GBM antrenat doar pe train; orice scalare/selecție fitată pe train
- [ ] Tabel 2×2 pe validation (medie ± std, 5 seeds) și apoi pe test (o singură dată, configurație înghețată)
- [ ] GBM și LR raportate la **aceeași** valoare pentru toate p (modele clasice, fără zgomot cuantic), cu etichetă clară „referință extinsă, nu cerută de Task 2”
- [ ] Buget de alarme (k = 3, 5, 10, 50) pentru GBM și LR
- [ ] Concluzie scrisă: de unde vine golful (model vs. intrări), cu limitele (98 de fraude pe test)
- [ ] Dacă se adaugă dependențe: PR separat pentru `requirements.txt`, anunțat echipei

**Tasks:**
- [ ] S11-T1 – `src/gbm_reference.py`: antrenare GBM pe 30 de feature-uri (varianta a și b), seed fix (60 min) · owner: Coleg · fișiere: `src/gbm_reference.py`
- [ ] S11-T2 – GBM pe cele 4 unghiuri ale aceluiași `AngleEncoder` (reutilizează encoderul din Exp. 4/5) și LR pe 30 de feature-uri, pentru tabelul 2×2 (60 min) · owner: Coleg · fișiere: `notebooks/exp11_gradient_boosting.ipynb`
- [ ] S11-T3 – grilă mică de hiperparametri pe validation, fără a atinge testul (45 min) · owner: Coleg · fișiere: `notebooks/exp11_gradient_boosting.ipynb`
- [ ] S11-T4 – buget de alarme + înghețare + evaluare unică pe test (60 min) · owner: Coleg · fișiere: `results/result_exp11/`
- [ ] S11-T5 – **(COULD, doar dacă mai e timp)** alegerea celor 4 intrări ale VQC după importanța din GBM (calculată pe train), reantrenare B pe noile intrări, LR de referință pe **aceleași** intrări noi, evaluare sub zgomot pe validation (120 min) · owner: Coleg · fișiere: `notebooks/exp11_gradient_boosting.ipynb`

**Note / riscuri:**
- T5 schimbă intrările, deci și LR de referință trebuie refăcută pe ele (altfel violează „aceleași intrări”). Pe test se evaluează o singură dată.
- Dacă H0 iese adevărat, rezultatul se prezintă ca atare: „4 unghiuri nu sunt gâtuirea”.

---

## S12 – Integrare în prezentare și tabel comparativ

**Owner:** Alex · **Status:** TODO · **Branch:** `alex-s12-slides` · **Prioritate:** MUST · **Depinde de:** S8 (+ S9, S10, S11 în măsura în care sunt gata)
**Obiectiv:** o oră pentru slide-urile care acoperă extensiile, cu cifre luate **din fișierele de rezultate**, nu rescrise de mână.

**Criterii de acceptare:**
- [ ] Un tabel unic pe test: A, B, LR și fiecare extensie finalizată (S9, S10a, S10b, S11), la p = 0, 0.01, 0.05: AUPRC, ρ (unde are sens), F1, MCC, buget de alarme; medie ± std pe 5 seeds
- [ ] Un slide „Ce nu a mers”: Exp. 7 (ansatz redus) și rezultatele negative din S9/S10/S11, dacă apar
- [ ] Un slide de cost: ledger A, B, cascadă (10a), validare ieftină (10b), cu execuții și shots
- [ ] Menționat numărul total de evaluări pe test (Exp. 5 + extensii) și faptul că testul a fost consultat în Exp. 5
- [ ] Marcat clar în prezentare ce este „model VQC” și ce este „analiză hibridă/clasică”
- [ ] Citări: Dinuț et al. (2026), Bowles, Ahmed, Schuld (2024) (pentru S11)

**Tasks:**
- [ ] S12-T1 – script care generează tabelul final din `results/result_exp5/`, `result_exp9/`, `result_exp10/`, `result_exp11/` (60 min) · owner: Alex · fișiere: `docs/`
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
| 2026-10-24 | Extensiile S9–S11 sunt analize suplimentare, nu înlocuiesc îmbunătățirea obligatorie | Risc de conformitate (secțiunea 2) |
| 2026-10-24 | **PROPUSĂ:** GBM implicit prin scikit-learn, nu XGBoost | Fără dependențe noi |
| 2026-10-24 | **PROPUSĂ:** testul se folosește o singură dată per configurație înghețată | Testul a fost deja consultat în Exp. 5 |

## Jurnal de lucru

| Data/ora | Cine | Ce s-a făcut | Story/Task | PR |
|---|---|---|---|---|
| 2026-10-24 21:xx | Alex + Claude | Analiza Exp. 1–7, definirea S8–S12 | S8–S12 | — |

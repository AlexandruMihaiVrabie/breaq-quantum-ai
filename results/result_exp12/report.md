# Experimentul 12: evaluarea finală pe TEST a modelului STAR + Z₀Z₁

**Tip:** evaluare. Nimic nu a fost reantrenat, recalibrat sau reselectat.
**Cod:** `notebooks/exp12_final_test_evaluation.ipynb` (importă `src/quantum_readout.py`, nemodificat). PennyLane 0.45.1, torch 2.14.1 (float64), sklearn 1.9.1.
**Date:** test oficial, 56 962 de tranzacții, dintre care 98 fraude (split 60/20/20 stratificat, seed 42).
**Protocol:**
- `default.mixed`, valori de așteptare exacte;
- `DepolarizingChannel(p)` după fiecare poartă nativă;
- p ∈ {0, 0.001, 0.01, 0.05}, 5 seeds;
- prag = max-F1 pe **validation** la același p, aplicat neschimbat (`pred = score >= prag`).

> ⚠️ **Mențiune obligatorie.** Setul de test a fost folosit anterior în Exp. 5, pentru alte modele (A = Exp. 2, B = Exp. 3). Cercetarea a continuat după acel moment (Exp. 6–11). Evaluarea curentă **nu poate fi prezentată ca verificare complet independentă** de întregul proces exploratoriu.

## 0. Rezumat executiv

- **STAR + Z₀Z₁ se menține pe test.** AUPRC este **0.705 ± 0.028** la p = 0.01 și **0.709 ± 0.033** la p = 0.05. Practic nu se degradează cu zgomotul: AUPRC_drop (0 → 0.05) = −0.005 ± 0.018.
- **Față de STAR + Z₀, avantajul la p = 0.05 se păstrează în medie, dar e mai slab decât pe validation.**
  - Δ AUPRC = **+0.048 ± 0.064**, Z₀Z₁ mai bun în **3/5 seeds**, față de 4/5 pe validation.
  - La p = 0.01: Δ = +0.018 ± 0.026, tot 3/5 seeds.
  - Câștigul mediu vine mai ales din seed-urile unde Z₀ se degradează mult (seed 2: +0.131). În două seeds, Z₀Z₁ pierde puțin (−0.012 și −0.019).
- **Față de regresia logistică: egalitate pe AUPRC, LR mai bună la prag.**
  - Δ AUPRC = +0.006 ± 0.033 (3/5) la p = 0.05 și +0.001 ± 0.030 (2/5) la p = 0.01.
  - LR detectează însă mai multe fraude la pragul ales pe validation: **79.2 față de 72.0** din 98, mai bună în 5/5 seeds, cu FP similare (23.0 față de 22.4).
  - LR este și mult mai stabilă între seeds (std AUPRC 0.010 față de 0.033).
- **La p = 0.05, Z₀Z₁ detectează 72.0 ± 4.3 fraude din 98 (67–77) și dă 22.4 ± 8.9 alarme false (16–38)**, din 56 864 de tranzacții legitime.
- **Ipoteza din validation se confirmă parțial.** Z₀Z₁ ≥ Z₀ în medie și robust la zgomot: da. „Mai bun decât Z₀ în majoritatea clară a seeds” și „mai bun decât LR”: nu.
- **Rezerva principală rămâne cea din Exp. 11 §7.** Toate cifrele de aici sunt valori de așteptare exacte. Cu 1024 shots, la p = 0.05, Z₀Z₁ pierdea masiv pe validation (AUPRC 0.19, față de 0.52 pentru Z₀), pentru că semnalul lui e de ~2× mai comprimat. Rezultatul de pe test **nu se transferă** la bugetul de 1024 shots.

## 1. Verificări (30, toate trecute: `checks.csv`)

- **Split:** reconstruit cu seed 42 și identic cu `split_indices.npz`. Seturile nu se suprapun, iar acoperirea este completă (284 807).
  - SHA-256 identic cu Exp. 11 (`1aa0c2d8…`, convenția „array-uri concatenate”) și cu Exp. 5 (`0ea3d5c1…`, convenția „fișier”).
  - Dimensiuni: 170 883 / 295, 56 962 / 99, 56 962 / 98.
- **Circuit identic cu Exp. 11:** aceeași ordine a celor 58 de porți native (`config.json` Exp. 11), observabile `PauliZ(0)` și `PauliZ(0) @ PauliZ(1)`, 15 CNOT, 73 locuri de zgomot, 32 de parametri.
- **Encoder doar din train:** `AngleEncoder(4, seed)`, fitat pe cele 590 de exemple 1:1 (`n_samples_seen_ = 590`).
  - Unghiurile train + val au același SHA-256 ca în Exp. 9 și Exp. 11, deci este același encoder.
  - Hash-ul parametrilor encoderului (scaler, PCA, MinMax) este identic înainte și după transformarea testului: pe test nu s-a făcut niciun fit.
  - Clipping-ul [0, π] atinge 0.003–0.1% din unghiurile test.
- **Ponderi neschimbate:**
  - Z₀ din `result_exp9/weights/star`, Z₀Z₁ din `result_exp11/weights/Z0Z1`;
  - inițializarea, epoca aleasă și val AUPRC (p = 0) sunt reproduse exact;
  - SHA-256-urile sunt în `config.json`.
- **Praguri doar din validation:**
  - pentru VQC, max-F1 recalculat pe scorurile validation salvate de Exp. 11 dă exact pragurile din `metrics_per_seed.csv` (eroare 0);
  - pentru LR, pragul este identic cu Exp. 4;
  - pragurile au fost salvate (`thresholds.csv`, SHA-256 în `config.json`) **înainte** de calculul scorurilor test;
  - etichetele test au fost blocate până după înghețarea pragurilor.
- **Modelul de zgomot = Exp. 11:** primele 2 000 de scoruri validation, recalculate din ponderi la p = 0.001 / 0.01 / 0.05, coincid cu cele salvate (eroare 0). `default.mixed` la p = 0 dă aceleași scoruri ca `default.qubit` (7.6e-13).
- **Matricele de confuzie:** TP + FN = 98, FP + TN = 56 864 și totalul 56 962, la toate cele 60 de rânduri. Spearman ρ (p = 0) = 1. AUPRC este calculat din scoruri continue (≥ 1 000 de valori unice).
- **Verificarea #7 („Z₀ pe test reproduce Exp. 5”) nu se poate face literal.** STAR + Z₀ nu a fost evaluat în Exp. 5, care a testat A (ZZ ×2) și B (ZZ ×1, toate perechile); STAR a apărut abia în Exp. 9. Am făcut în schimb verificarea echivalentă a pipeline-ului (același encoder, `make_vqc`, `noisy_predict`):
  - modelul B din Exp. 3 reproduce scorurile test salvate de Exp. 5: eroare **0** la p = 0 (toate tranzacțiile) și **0** la p > 0 (2 000 de rânduri);
  - LR reproduce exact modelul C din Exp. 5;
  - metricile istorice Exp. 5, recalculate din scorurile lui, sunt identice cu `test_detailed.csv`.
  - *Observație tehnică:* Exp. 5 a rulat cu tipul implicit float32 din torch (nu a setat `set_default_dtype`). Cu float64, protocolul Exp. 9 / 11 / 12, scorurile cu zgomot diferă de ale Exp. 5 cu cel mult **1.0e-7**. Verificarea strictă a rulat cu setarea Exp. 5. Diferența nu afectează niciun clasament.

## 2. Rezultatul principal: test, medie ± std pe 5 seeds

| model | AUPRC p = 0 | AUPRC p = 0.01 | AUPRC p = 0.05 | AUROC p = 0.05 | recall p = 0.05 | precision p = 0.05 | F1 p = 0.05 | MCC p = 0.05 | TP / 98 | FP |
|---|---|---|---|---|---|---|---|---|---|---|
| **STAR + Z₀Z₁** (candidat) | 0.704 ± 0.028 | **0.705 ± 0.028** | **0.709 ± 0.033** | 0.962 ± 0.008 | 0.735 ± 0.044 | 0.768 ± 0.061 | 0.749 ± 0.032 | 0.750 ± 0.031 | 72.0 ± 4.3 | 22.4 ± 8.9 |
| STAR + Z₀ (Exp. 9) | 0.690 ± 0.030 | 0.687 ± 0.033 | 0.661 ± 0.058 | 0.971 ± 0.003 | 0.720 ± 0.049 | 0.772 ± 0.035 | 0.744 ± 0.031 | 0.745 ± 0.031 | 70.6 ± 4.8 | 21.0 ± 4.2 |
| Regresie logistică (clasic) | 0.703 ± 0.010 | 0.703 ± 0.010 | 0.703 ± 0.010 | 0.976 ± 0.003 | **0.808 ± 0.013** | 0.776 ± 0.029 | **0.791 ± 0.019** | **0.791 ± 0.018** | **79.2 ± 1.3** | 23.0 ± 3.8 |
| A: VQC ZZ ×2 (Exp. 5, istoric) | 0.580 ± 0.053 | 0.564 ± 0.049 | 0.405 ± 0.177 | 0.955 ± 0.024 | 0.622 ± 0.158 | 0.487 ± 0.198 | 0.537 ± 0.184 | 0.545 ± 0.177 | 61.0 ± 15.5 | 82.6 ± 69.0 |

La p = 0.01, Z₀Z₁ are recall 0.729 ± 0.064, precision 0.791 ± 0.021, F1 0.757 ± 0.030, TP 71.4 ± 6.3 și FP 19.0 ± 3.7. Tabelele complete (toate metricile, toate p, valori per seed) sunt în `test_summary_metrics.csv` și `test_metrics_per_seed.csv`.

AUPRC pe test per seed:

| | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| Z₀Z₁ p = 0.01 | 0.683 | 0.717 | 0.696 | 0.680 | 0.747 |
| Z₀ p = 0.01 | 0.663 | 0.731 | 0.650 | 0.683 | 0.706 |
| Z₀Z₁ p = 0.05 | 0.663 | 0.721 | 0.713 | 0.697 | 0.752 |
| Z₀ p = 0.05 | 0.675 | 0.740 | 0.582 | 0.638 | 0.672 |
| LR | 0.699 | 0.688 | 0.709 | 0.710 | 0.710 |

## 3. Comparații pereche pe AUPRC (`paired_test_comparison.csv`)

| comparație | p = 0 | p = 0.01 | p = 0.05 | bootstrap 95%, p = 0.05 (descriptiv) |
|---|---|---|---|---|
| A. Z₀Z₁ − Z₀ | +0.014 ± 0.020 (3/5) | +0.018 ± 0.026 (3/5) | **+0.048 ± 0.064 (3/5)** | [+0.007, +0.075] |
| B. Z₀Z₁ − LR | +0.001 ± 0.031 (2/5) | +0.001 ± 0.030 (2/5) | +0.006 ± 0.033 (3/5) | [−0.064, +0.064] |
| C. Z₀ − LR | −0.013 ± 0.035 (2/5) | −0.017 ± 0.039 (1/5) | −0.042 ± 0.066 (1/5) | [−0.102, +0.020] |
| D. Z₀Z₁ − A (Exp. 5) | +0.123 ± 0.048 (5/5) | +0.140 ± 0.042 (5/5) | +0.304 ± 0.199 (5/5) | [+0.237, +0.350] |

Între paranteze: seeds în care primul model e mai bun.

**Cum se citește bootstrap-ul.** Este stratificat pereche, cu 1 000 de reeșantionări ale tranzacțiilor test, aplicate media pe 5 seeds.
- Măsoară doar incertitudinea datorată eșantionului de test (98 de fraude), la modele fixe.
- **Nu** include variabilitatea între seeds, care aici este sursa dominantă.
- Intervalul pentru A exclude zero, dar cu 3/5 seeds câștigate nu putem vorbi de un avantaj robust.
- A fost calculat după evaluare și nu a fost folosit la nicio decizie.

**Metrici la prag** (Δ medie pe seeds, la p = 0.05):
- Z₀Z₁ − Z₀: recall +0.014 (4/5), TP +1.4, FP +1.4 (Z₀Z₁ are mai puține FP în doar 2/5 seeds).
- Z₀Z₁ − LR: recall −0.073 (0/5), TP −7.2, FP −0.6.

## 4. Degradarea cu zgomotul și robustețea

| | AUPRC_drop p = 0.01 | AUPRC_drop p = 0.05 | Spearman ρ p = 0.01 | Spearman ρ p = 0.05 |
|---|---|---|---|---|
| Z₀Z₁ | −0.001 ± 0.003 | **−0.005 ± 0.018** | 0.997 ± 0.004 | 0.857 ± 0.157 |
| Z₀ | +0.004 ± 0.007 | +0.029 ± 0.042 | 0.997 ± 0.002 | 0.890 ± 0.064 |
| A (Exp. 5) | +0.016 | +0.175 | 0.982 ± 0.010 | 0.687 ± 0.145 |

AUPRC_drop = AUPRC(p = 0) − AUPRC(p). O valoare negativă înseamnă că AUPRC nu scade.

- **Z₀Z₁ își păstrează AUPRC până la p = 0.05.** Pe seeds, drop-ul este între −0.022 și +0.023. Z₀ pierde până la 0.083 (seed 2) și 0.048 (seed 3), la fel ca pe validation.
- **Spearman ρ nu urmează AUPRC.** Z₀Z₁ are ρ mai mic și foarte variabil (seed-uri cu ρ scăzut). Zgomotul amestecă ordinea în masa de tranzacții legitime cu scor mic, dar păstrează vârful clasamentului, unde se decide AUPRC. Același fenomen a apărut în Exp. 11.

## 5. Validation vs test (`validation_vs_test.csv`)

| | AUPRC val → test, p = 0.05 | Δ | recall val → test | precision val → test | FP val → test |
|---|---|---|---|---|---|
| Z₀Z₁ | 0.678 → 0.709 | +0.031 | 0.695 → 0.735 | 0.792 → 0.768 | 18.2 → 22.4 |
| Z₀ | 0.622 → 0.661 | +0.039 | 0.677 → 0.720 | 0.804 → 0.772 | 16.4 → 21.0 |
| LR | 0.663 → 0.703 | +0.040 | 0.715 → 0.808 | 0.802 → 0.776 | 17.8 → 23.0 |

- **Toate modelele au AUPRC cu +0.03 – +0.04 mai mare pe test, inclusiv LR**, care nu are nimic cuantic și nu a văzut validation la antrenare. Explicația cea mai simplă: cu doar 98–99 de fraude per set, testul pare puțin „mai ușor” decât validation. Câștigul **nu** este o dovadă că modelul generalizează mai bine și nu validează definitiv modelul.
- **Precision scade puțin pe test la toate modelele** (−0.02 … −0.04), iar FP cresc cu ~4–5. Seturile au același număr de tranzacții legitime (56 863 față de 56 864), deci FP sunt direct comparabile. Pragurile max-F1, alese pe validation, sunt ușor optimiste.
- **Diferențele pereche se păstrează în semn, dar se micșorează.**
  - Z₀Z₁ − Z₀: +0.056 → +0.048 la p = 0.05, dar seeds câștigate 4/5 → 3/5.
  - Z₀Z₁ − LR: +0.015 → +0.006 (3/5 seeds), adică egalitate.

## 6. Analiza erorilor (`test_confusion_analysis.csv`)

La p = 0.05, medie pe 5 seeds:

| | fraude (din 98) |
|---|---|
| detectate de Z₀Z₁ / de Z₀ | 72.0 / 70.6 |
| detectate de ambele | 66.0 |
| doar de Z₀Z₁ / doar de Z₀ | 6.0 / 4.6 |
| ratate de ambele | 21.4 |
| ratate de Z₀Z₁, Z₀ **și** LR | **18 (identic în toate cele 5 seeds)** |

| | alarme false |
|---|---|
| Z₀Z₁ / Z₀ | 22.4 / 21.0 |
| comune | 16.8 |
| doar Z₀Z₁ / doar Z₀ | 5.6 / 4.2 |

- **Cele două observabile greșesc în mare parte pe aceleași tranzacții.** Diferența dintre ele ține de ~5–6 fraude per seed.
- **18 fraude sunt ratate de toate modelele, în toate seeds.** Ele sunt de nedistins de tranzacțiile legitime în cele 4 componente PCA. Limita vine din reprezentare (PCA 4), nu din circuit sau din zgomot.
- **Seed 0 la p = 0.05 este un caz izolat pentru Z₀Z₁:** 38 FP (20 doar ale lui Z₀Z₁) și precision 0.66. Pragul ales pe validation s-a transferat prost la acest seed. Explică cea mai mare parte a std-ului FP (±8.9).
- **Stabilitatea între seeds, la p = 0.05:** câte fraude sunt detectate în toate cele 5 seeds:

  | | fraude detectate în 5/5 seeds |
  |---|---|
  | Z₀Z₁ | 59 |
  | Z₀ | 58 |
  | LR | 77 |

  LR este mult mai consecventă.

## 7. Cost cuantic (`test_cost_ledger.json`)

### Inferența pe test

O execuție de circuit per tranzacție. Cifrele sunt identice pentru Z₀ și Z₀Z₁:

| | per (seed, p) | 5 seeds × 4 p (simulat aici) | dispozitiv real, un singur nivel, 5 seeds |
|---|---|---|---|
| execuții nominale | 56 962 | 1 139 240 | 284 810 |
| shots nominale (× 1024) | 58 329 088 | 1 166 581 760 | 291 645 440 |
| CNOT / locuri de zgomot per execuție | 15 / 73 | | |

### De ce costul inferenței este identic

Ambele observabile folosesc același circuit. Z₀ și Z₀Z₁ se citesc din **aceleași bitstring-uri**, deci shots nu se dublează.

### Ce înseamnă „nominal” aici

- **Simulatorul exact nu trage shots.** Execuțiile și shots de mai sus sunt o estimare nominală pentru hardware.
- **Timpul efectiv de simulare** pentru scorarea testului: 4.1 min (Z₀Z₁) și 4.6 min (Z₀).
- **Pragurile nu au cerut execuții noi pe validation:** s-au refolosit scorurile din Exp. 11, deja contabilizate acolo. Recalculările de verificare (60 000 de rânduri validation și 314 810 rânduri test pentru Exp. 5) sunt doar verificări și nu intră în costul protocolului.

### Antrenarea (istoric, Exp. 9 / 11, nerepetată)

Gradientul diferă între cele două observabile:

| | execuții / exemplu, parameter shift (Tracker) | total / seed (Tracker) |
|---|---|---|
| Z₀ | **41** | 2 662 408 |
| Z₀Z₁ | 53 | 2 874 808 (**+8.0%**) |

Diferența vine din conul cauzal: Z₀Z₁ are 26 de parametri în con, față de 20 pentru Z₀.

### Analiza finite-shot

Este alt protocol (Exp. 11, `finite_shot_cost_ledger.json`) și nu face parte din metricile exacte de aici.

## 8. Răspunsuri explicite

1. **AUPRC test la p = 0.01, STAR + Z₀Z₁:** 0.705 ± 0.028.
2. **AUPRC test la p = 0.05:** 0.709 ± 0.033.
3. **Medii ± SD pe 5 seeds (p = 0.05):**

   | | AUPRC |
   |---|---|
   | Z₀Z₁ | 0.709 ± 0.033 |
   | Z₀ | 0.661 ± 0.058 |
   | LR | 0.703 ± 0.010 |
   | A (Exp. 5) | 0.405 ± 0.177 |

   La p = 0.01: Z₀Z₁ 0.705 ± 0.028, Z₀ 0.687 ± 0.033.
4. **Fraude detectate din 98:** 72.0 ± 4.3 (67–77) la p = 0.05 și 71.4 ± 6.3 la p = 0.01. Recall 73.5% / 72.9%.
5. **Alarme false:** 22.4 ± 8.9 (16–38) la p = 0.05 și 19.0 ± 3.7 la p = 0.01, din 56 864 de tranzacții legitime, adică ~3.9 la 10 000.
6. **Z₀Z₁ depășește Z₀ în majoritatea seeds?** Formal da, dar la limită: 3/5 la p = 0.01 și la p = 0.05, față de 4/5 pe validation. Diferența medie este pozitivă (+0.018 / +0.048) și vine mai ales din seeds unde Z₀ se degradează puternic.
7. **Z₀Z₁ depășește LR?** Nu în mod susținut. Pe AUPRC este egalitate (+0.006 ± 0.033, 3/5 la p = 0.05; +0.001, 2/5 la p = 0.01). La pragul ales pe validation, LR detectează mai multe fraude în 5/5 seeds (79.2 față de 72.0) și are F1 mai mare (0.791 față de 0.749).
8. **Test vs validation:** AUPRC +0.031 pe test (0.678 → 0.709 la p = 0.05), la fel ca Z₀ (+0.039) și LR (+0.040). Este o diferență între seturi, nu un câștig propriu al modelului. Precision scade ușor, iar FP cresc cu ~4.
9. **Cost nominal:** inferența pe test costă 56 962 de execuții (58.3 M shots la 1024) per seed și nivel de zgomot, identic pentru Z₀ și Z₀Z₁. Antrenarea pe hardware ar costa ~2.87 M execuții / seed pentru Z₀Z₁, cu +8.0% față de Z₀.
10. **Ce putem afirma în 3 minute** (formulări susținute de date):
    - „Schimbând doar observabila măsurată (Z₀ → Z₀Z₁, același circuit, zero porți în plus), AUPRC-ul VQC-ului pe test **nu se degradează** sub zgomot depolarizant: 0.704 fără zgomot, 0.709 la p = 0.05, pe 5 seeds.”
    - „Față de baseline-ul inițial (VQC ZZ ×2, Exp. 5), AUPRC la p = 0.05 crește de la 0.41 la 0.71, în 5/5 seeds, iar alarmele false scad de la ~83 la ~22.”
    - „Pe AUPRC, VQC-ul final egalează regresia logistică pe aceleași 4 caracteristici (0.709 față de 0.703). Nu o depășește: LR prinde mai multe fraude la prag și este mai stabilă.”
    - „Rezultatele sunt în simulare exactă. Cu 1024 shots, avantajul Z₀Z₁ dispare, iar pe hardware real Z₀ ar fi alegerea mai sigură.”
    - Ce **nu** putem spune: avantaj cuantic; că Z₀Z₁ „bate” clasicul; că testul este o validare complet independentă.

## 9. Limitări

- **Testul nu este blind:** a fost folosit în Exp. 5, iar cercetarea a continuat după aceea (vezi mențiunea de la început).
- **Doar 98 de fraude în test.** O fraudă înseamnă ~1 punct de recall. Diferențele de 1–2 fraude între modele sunt sub nivelul zgomotului statistic.
- **5 seeds**: câștigurile 3/5 nu sunt concludente. Std-ul diferențelor (0.064 la p = 0.05) este mai mare decât media lor (0.048).
- **Valori exacte, nu shots.** Concluzia despre Z₀Z₁ nu se transferă la 1024 shots (Exp. 11 §7). Analiza finite-shot a fost făcută doar pe validation și este post hoc.
- **Pragurile max-F1 sunt alese și evaluate pe același validation** (optimist). Pe test, precision scade la toate modelele.
- **Diferența de tip numeric față de Exp. 5:** float32 implicit acolo, float64 aici, ~1e-7 pe scoruri. Nu afectează rezultatele.
- **Regula de decizie din Exp. 11 a ales Z₀Z₁**, dar recomandarea finală a raportului Exp. 11 (§0, §10) era să păstrăm Z₀ pentru soluția finală, din cauza shots. Exp. 12 evaluează Z₀Z₁ ca model înghețat, așa cum s-a cerut, și nu schimbă recomandarea pentru hardware.

## 10. Fișiere

`results/result_exp12/`:

| fișier | conținut |
|---|---|
| `config.json` | modele, hash-uri pentru split, ponderi, encodere, praguri și unghiuri test; mențiunea despre folosirea anterioară a testului |
| `checks.csv` | 30 de verificări |
| `thresholds.csv` | pragurile validation înghețate |
| `test_metrics_per_seed.csv` | o linie per (model, seed, p): metrici, TP / FP / FN / TN, ρ, MAE, RMSE, drop, prag, timp |
| `test_summary_metrics.csv` | medie, std, min, max și valori per seed |
| `paired_test_comparison.csv` | comparațiile A–D per metrică și p, cu Δ per seed, plus bootstrap-ul |
| `validation_vs_test.csv` | comparația validation / test |
| `test_confusion_analysis.csv` | analiza erorilor per seed, medii, stabilitate între seeds |
| `test_cost_ledger.json` | costul inferenței și al antrenării |
| `test_scores/` | 40 de fișiere `.npz` VQC și 5 LR, cu hash-ul ponderilor și pragul validation în fiecare |
| `plots/` | 10 grafice, `01_…` până la `10_…` |
| `report.md` | acest raport |

**Notă git:** `*.csv` este în `.gitignore`, ca la experimentele anterioare. CSV-urile nu vor intra în commit, decât dacă decideți altfel.

# Experimentul 10: BALANCED PCA vs FULL-TRAIN PCA pentru STAR ZZ

**Arhitectură fixă:** STAR ZZ din Exp. 9 (4 qubiți, ZZ pe (0,1) (0,2) (0,3), feature map ×1, ansatz ×3, 32 de parametri, readout ⟨Z₀⟩, **15 CNOT, 73 de locuri de zgomot**, identice în A și B).
**Date:** doar train (VQC: 590 de exemple 1:1 / seed) și validation (56 962 de tranzacții, 99 de fraude). **Testul nu a fost încărcat.** 5 seeds, p = 0 / 0.001 / 0.01 / 0.05, `default.mixed`, prag max-F1 pe validation la același p (estimare optimistă: pragul se alege și se evaluează pe același set).
**Notebook:** `notebooks/exp10_balanced_vs_full_train_pca.ipynb`; cod reutilizabil: `src/star_pca.py`. PennyLane 0.45.1, torch 2.14.1, sklearn 1.9.1.

## A. Întrebarea de cercetare

Care este impactul populației folosite la fit-ul PCA (cele 590 de exemple echilibrate vs. tot train-ul, 170.883) asupra performanței circuitului STAR ZZ, cu restul pipeline-ului neschimbat?

**Răspuns scurt:** schimbarea este puternic negativă. FULL-TRAIN PCA scade AUPRC la p = 0.05 de la 0.622 la 0.089, în toate cele 5 seeds, iar recall-ul de la 0.677 la 0.168. **H0 este susținută; H1 este respinsă în această configurație.** Recomandarea este să rămânem cu **BALANCED PCA**.

## B. Metodologie

| | A: BALANCED PCA (Exp. 1–9) | B: FULL-TRAIN PCA |
|---|---|---|
| `StandardScaler` (Time, Amount) | fit pe cele 590 | **identic** (același obiect, verificat) |
| `PCA(4).fit` | cele 590 | **cele 170.883 din train** standardizate cu scalerul de mai sus |
| `MinMaxScaler(0, π).fit` | proiecțiile PCA ale celor 590 | proiecțiile PCA-ului B ale **acelorași 590** |
| clipping la [0, π] în `transform` | da | da (comportament existent, neschimbat) |
| VQC antrenat pe | cele 590 | **aceleași 590** (tot train-ul intră doar în `PCA.fit`) |

**Constant:** circuitul STAR, readout, `s = (1 − ⟨Z₀⟩)/2`, Adam lr 0.05, BCE, batch 32, 30 de epoci, selecția epocii după AUPRC pe validation la p = 0, `init_weights(seed)` și ordinea mini-batch-urilor (hash SHA-256 identic între A și B pe fiecare seed), modelul oficial de zgomot, pragul max-F1 la același p, seeds [0–4].
**Modificat:** exclusiv populația din `PCA.fit`.
**Reantrenare:** B este antrenat de la zero. A folosește ponderile salvate în Exp. 9, după ce reantrenarea lor a fost verificată.

**Verificări trecute** (`checks.csv`): hash-ul split-ului = cel din Exp. 5 și 170.883 / 56.962 / 56.962 tranzacții cu 295 / 99 / 98 de fraude; `PCA_A.n_samples_ == 590` și `PCA_B.n_samples_ == 170883`; același StandardScaler și același MinMaxScaler-fit pe cele 590; 4 dimensiuni de intrare, unghiuri în [0, π]; 15 CNOT, 73 locuri de zgomot și 32 de parametri, cu secvența de porți identică cu `config.json` din Exp. 9; refit-ul lui B și reantrenarea lui B (seed 0) dau rezultate identice bit cu bit; `default.mixed` la p = 0 = `default.qubit`; **A sub zgomot reproduce Exp. 9 STAR la 4e-15** pe cele 20 de (seed, p) (AUPRC, prag, TP, FP, F1, MAE, ρ).

**Reproducerea referinței Exp. 9 (p = 0.05, 5 seeds, validation):** AUPRC 0.622, recall 0.6768 (≈ 67 din 99 de fraude), F1 0.735, FP 16.4, AUPRC la p = 0: 0.662. Valorile din cerință se regăsesc.

**Diferențe de reproducere, documentate (nu au schimbat baseline-ul):**
- sklearn 1.9 nu expune `n_samples_seen_` pe `PCA` (doar pe `IncrementalPCA`); am verificat `n_samples_`.
- Hash-ul brut al octeților unghiurilor A diferă de cel din `train_runs.jsonl` al Exp. 9 (diferență de ultim bit între sesiuni). Numeric, ponderile salvate în Exp. 9 dau pe unghiurile de acum exact AUPRC-ul salvat (eroare ≤ 1e-12).
- Reantrenarea lui A dă aceeași epocă aleasă și același istoric al loss-ului (≤ 1e-14), dar ponderile diferă cu 1–5e-9 (zgomot numeric amplificat peste 570 de pași Adam). Toleranța pentru ponderi reantrenate este 1e-6; rezultatele lui A folosesc ponderile originale din Exp. 9 (`exp9_reproduction.csv`).

## C. Rezultate cantitative

### C.1 Medie ± std (ddof = 1) pe 5 seeds, validation

| varianta | p | AUPRC | AUROC | precision | recall | F1 | MCC | TP | FP | FN | ρ Spearman | MAE | RMSE | ΔAUPRC vs p=0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A BALANCED | 0.0000 | 0.6617 ± 0.0226 | 0.9570 ± 0.0032 | 0.7995 ± 0.0354 | 0.6768 ± 0.0440 | 0.7327 ± 0.0373 | 0.7350 ± 0.0369 | 67.0 ± 4.4 | 16.8 ± 2.9 | 32.0 ± 4.4 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| A BALANCED | 0.0010 | 0.6606 ± 0.0228 | 0.9569 ± 0.0032 | 0.8014 ± 0.0327 | 0.6768 ± 0.0440 | 0.7335 ± 0.0361 | 0.7358 ± 0.0356 | 67.0 ± 4.4 | 16.6 ± 2.7 | 32.0 ± 4.4 | 1.0000 ± 0.0000 | 0.0130 ± 0.0012 | 0.0131 ± 0.0012 | 0.0011 ± 0.0013 |
| A BALANCED | 0.0100 | 0.6539 ± 0.0227 | 0.9558 ± 0.0044 | 0.8027 ± 0.0358 | 0.6747 ± 0.0425 | 0.7330 ± 0.0377 | 0.7354 ± 0.0373 | 66.8 ± 4.2 | 16.4 ± 2.9 | 32.2 ± 4.2 | 0.9968 ± 0.0016 | 0.1081 ± 0.0089 | 0.1085 ± 0.0089 | 0.0078 ± 0.0093 |
| A BALANCED | 0.0500 | 0.6221 ± 0.0411 | 0.9471 ± 0.0033 | 0.8043 ± 0.0424 | 0.6768 ± 0.0378 | 0.7346 ± 0.0337 | 0.7371 ± 0.0339 | 67.0 ± 3.7 | 16.4 ± 4.0 | 32.0 ± 3.7 | 0.8892 ± 0.0655 | 0.2710 ± 0.0186 | 0.2720 ± 0.0185 | 0.0396 ± 0.0440 |
| B FULL-TRAIN | 0.0000 | 0.1403 ± 0.0660 | 0.9161 ± 0.0134 | 0.3436 ± 0.2019 | 0.2020 ± 0.0357 | 0.2287 ± 0.0593 | 0.2476 ± 0.0696 | 20.0 ± 3.5 | 66.8 ± 62.2 | 79.0 ± 3.5 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| B FULL-TRAIN | 0.0010 | 0.1394 ± 0.0659 | 0.9158 ± 0.0133 | 0.3349 ± 0.1890 | 0.2020 ± 0.0357 | 0.2278 ± 0.0602 | 0.2454 ± 0.0692 | 20.0 ± 3.5 | 67.6 ± 62.3 | 79.0 ± 3.5 | 0.9999 ± 0.0000 | 0.0088 ± 0.0011 | 0.0094 ± 0.0011 | 0.0009 ± 0.0006 |
| B FULL-TRAIN | 0.0100 | 0.1304 ± 0.0651 | 0.9130 ± 0.0119 | 0.2648 ± 0.1214 | 0.1960 ± 0.0273 | 0.2130 ± 0.0562 | 0.2202 ± 0.0591 | 19.4 ± 2.7 | 75.6 ± 66.8 | 79.6 ± 2.7 | 0.9948 ± 0.0044 | 0.0726 ± 0.0074 | 0.0776 ± 0.0074 | 0.0099 ± 0.0110 |
| B FULL-TRAIN | 0.0500 | 0.0889 ± 0.0528 | 0.9020 ± 0.0083 | 0.1993 ± 0.1540 | 0.1677 ± 0.0408 | 0.1580 ± 0.0662 | 0.1681 ± 0.0713 | 16.6 ± 4.0 | 135.0 ± 155.9 | 82.4 ± 4.0 | 0.9314 ± 0.0602 | 0.1789 ± 0.0097 | 0.1911 ± 0.0091 | 0.0515 ± 0.0369 |

### C.2 Comparație pereche B − A pe același seed, p = 0.05

Cu 5 seeds raportăm medie, std, interval t de 95 % pentru Δ (n = 5) și testul semnului exact. Cel mai mic p posibil al testului semnului cu 5 seeds este 0.0625, deci testul nu poate depăși pragul 0.05. Nu declarăm superioritate sau inferioritate „statistică"; susținem concluzia pe mărimea (≈ −0.53 AUPRC) și pe consistența ei (5 din 5 seeds, fiecare între −0.47 și −0.58).

| metrică | A (BALANCED) | B (FULL-TRAIN) | Δ medie (B − A) | Δ std (ddof=1) | Δ medie [puncte procentuale] | IC95% t jos | IC95% t sus | seeds în care B e mai bun | test semn exact (bilateral) |
|---|---|---|---|---|---|---|---|---|---|
| AUPRC | 0.6221 | 0.0889 | -0.5332 | 0.0422 |  | -0.5856 | -0.4808 | 0.0000 | 0.0625 |
| AUROC | 0.9471 | 0.9020 | -0.0452 | 0.0093 |  | -0.0567 | -0.0337 | 0.0000 | 0.0625 |
| precision | 0.8043 | 0.1993 | -0.6050 | 0.1364 | -60.4977 | -0.7744 | -0.4356 | 0.0000 | 0.0625 |
| recall | 0.6768 | 0.1677 | -0.5091 | 0.0233 | -50.9091 | -0.5380 | -0.4802 | 0.0000 | 0.0625 |
| F1 | 0.7346 | 0.1580 | -0.5765 | 0.0764 | -57.6540 | -0.6714 | -0.4816 | 0.0000 | 0.0625 |
| MCC | 0.7371 | 0.1681 | -0.5690 | 0.0756 |  | -0.6629 | -0.4751 | 0.0000 | 0.0625 |
| TP | 67.0000 | 16.6000 | -50.4000 | 2.3022 |  | -53.2585 | -47.5415 | 0.0000 | 0.0625 |
| FP | 16.4000 | 135.0000 | 118.6000 | 155.0429 |  | -73.9112 | 311.1112 | 0.0000 | 0.0625 |
| Spearman rho | 0.8892 | 0.9314 | 0.0422 | 0.1095 |  | -0.0938 | 0.1783 | 3.0000 | 1.0000 |
| MAE scor | 0.2710 | 0.1789 | -0.0921 | 0.0141 |  | -0.1096 | -0.0746 | 5.0000 | 0.0625 |
| pierdere AUPRC față de p = 0 | 0.0396 | 0.0515 | 0.0118 | 0.0221 |  | -0.0156 | 0.0393 | 1.0000 | 0.3750 |

### C.3 Diferențe B − A la toate nivelurile de zgomot (inclusiv per seed)

| p | metrică | Δ medie (B − A) | Δ std (ddof=1) | seeds în care B e mai bun | Δ seed 0 | Δ seed 1 | Δ seed 2 | Δ seed 3 | Δ seed 4 |
|---|---|---|---|---|---|---|---|---|---|
| 0.0000 | AUPRC | -0.5214 | 0.0514 | 0.0000 | -0.5689 | -0.5428 | -0.5569 | -0.4456 | -0.4926 |
| 0.0010 | AUPRC | -0.5211 | 0.0507 | 0.0000 | -0.5679 | -0.5434 | -0.5555 | -0.4461 | -0.4927 |
| 0.0100 | AUPRC | -0.5234 | 0.0491 | 0.0000 | -0.5724 | -0.5388 | -0.5552 | -0.4489 | -0.5020 |
| 0.0500 | AUPRC | -0.5332 | 0.0422 | 0.0000 | -0.5804 | -0.5646 | -0.5306 | -0.4725 | -0.5179 |
| 0.0000 | F1 | -0.5040 | 0.0446 | 0.0000 | -0.5407 | -0.4705 | -0.5615 | -0.4605 | -0.4869 |
| 0.0010 | F1 | -0.5056 | 0.0461 | 0.0000 | -0.5445 | -0.4742 | -0.5642 | -0.4584 | -0.4869 |
| 0.0100 | F1 | -0.5200 | 0.0544 | 0.0000 | -0.5479 | -0.4664 | -0.6007 | -0.4835 | -0.5015 |
| 0.0500 | F1 | -0.5765 | 0.0764 | 0.0000 | -0.5984 | -0.4961 | -0.6854 | -0.5107 | -0.5922 |
| 0.0000 | FP | 50.0000 | 60.5351 | 1.0000 | 138.0000 | -9.0000 | 86.0000 | 14.0000 | 21.0000 |
| 0.0010 | FP | 51.0000 | 61.0246 | 1.0000 | 139.0000 | -7.0000 | 89.0000 | 13.0000 | 21.0000 |
| 0.0100 | FP | 59.2000 | 65.3506 | 0.0000 | 172.0000 | 13.0000 | 58.0000 | 20.0000 | 33.0000 |
| 0.0500 | FP | 118.6000 | 155.0429 | 0.0000 | 98.0000 | 26.0000 | 389.0000 | 11.0000 | 69.0000 |
| 0.0000 | TP | -47.0000 | 3.0822 | 0.0000 | -42.0000 | -49.0000 | -49.0000 | -46.0000 | -49.0000 |
| 0.0010 | TP | -47.0000 | 3.0822 | 0.0000 | -42.0000 | -49.0000 | -49.0000 | -46.0000 | -49.0000 |
| 0.0100 | TP | -47.4000 | 5.7271 | 0.0000 | -40.0000 | -46.0000 | -56.0000 | -47.0000 | -48.0000 |
| 0.0500 | TP | -50.4000 | 2.3022 | 0.0000 | -53.0000 | -47.0000 | -50.0000 | -50.0000 | -52.0000 |
| 0.0000 | recall | -0.4747 | 0.0311 | 0.0000 | -0.4242 | -0.4949 | -0.4949 | -0.4646 | -0.4949 |
| 0.0010 | recall | -0.4747 | 0.0311 | 0.0000 | -0.4242 | -0.4949 | -0.4949 | -0.4646 | -0.4949 |
| 0.0100 | recall | -0.4788 | 0.0578 | 0.0000 | -0.4040 | -0.4646 | -0.5657 | -0.4747 | -0.4848 |
| 0.0500 | recall | -0.5091 | 0.0233 | 0.0000 | -0.5354 | -0.4747 | -0.5051 | -0.5051 | -0.5253 |

### C.4 Toate metricile, toate nivelurile de zgomot, toate seeds

| varianta | seed | p | AUPRC | AUROC | precision | recall | F1 | MCC | TP | FP | FN | TN | threshold | Spearman rho | MAE scor | RMSE scor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A BALANCED | 0 | 0.0000 | 0.6217 | 0.9563 | 0.7561 | 0.6263 | 0.6851 | 0.6876 | 62 | 20 | 37 | 56843 | {:.4f} | 1.0000 | 0.0000 | 0.0000 |
| A BALANCED | 1 | 0.0000 | 0.6727 | 0.9604 | 0.7778 | 0.6364 | 0.7000 | 0.7031 | 63 | 18 | 36 | 56845 | {:.4f} | 1.0000 | 0.0000 | 0.0000 |
| A BALANCED | 2 | 0.0000 | 0.6687 | 0.9597 | 0.8090 | 0.7273 | 0.7660 | 0.7667 | 72 | 17 | 27 | 56846 | {:.4f} | 1.0000 | 0.0000 | 0.0000 |
| A BALANCED | 3 | 0.0000 | 0.6695 | 0.9524 | 0.8500 | 0.6869 | 0.7598 | 0.7637 | 68 | 12 | 31 | 56851 | {:.4f} | 1.0000 | 0.0000 | 0.0000 |
| A BALANCED | 4 | 0.0000 | 0.6760 | 0.9564 | 0.8046 | 0.7071 | 0.7527 | 0.7539 | 70 | 17 | 29 | 56846 | {:.4f} | 1.0000 | 0.0000 | 0.0000 |
| B FULL-TRAIN | 0 | 0.0000 | 0.0527 | 0.8966 | 0.1124 | 0.2020 | 0.1444 | 0.1487 | 20 | 158 | 79 | 56705 | {:.4f} | 1.0000 | 0.0000 | 0.0000 |
| B FULL-TRAIN | 1 | 0.0000 | 0.1298 | 0.9105 | 0.6087 | 0.1414 | 0.2295 | 0.2929 | 14 | 9 | 85 | 56854 | {:.4f} | 1.0000 | 0.0000 | 0.0000 |
| B FULL-TRAIN | 2 | 0.0000 | 0.1118 | 0.9314 | 0.1825 | 0.2323 | 0.2044 | 0.2044 | 23 | 103 | 76 | 56760 | {:.4f} | 1.0000 | 0.0000 | 0.0000 |
| B FULL-TRAIN | 3 | 0.0000 | 0.2238 | 0.9245 | 0.4583 | 0.2222 | 0.2993 | 0.3183 | 22 | 26 | 77 | 56837 | {:.4f} | 1.0000 | 0.0000 | 0.0000 |
| B FULL-TRAIN | 4 | 0.0000 | 0.1834 | 0.9177 | 0.3559 | 0.2121 | 0.2658 | 0.2738 | 21 | 38 | 78 | 56825 | {:.4f} | 1.0000 | 0.0000 | 0.0000 |
| A BALANCED | 0 | 0.0010 | 0.6202 | 0.9559 | 0.7654 | 0.6263 | 0.6889 | 0.6919 | 62 | 19 | 37 | 56844 | {:.4f} | 1.0000 | 0.0120 | 0.0120 |
| A BALANCED | 1 | 0.0010 | 0.6729 | 0.9604 | 0.7778 | 0.6364 | 0.7000 | 0.7031 | 63 | 18 | 36 | 56845 | {:.4f} | 1.0000 | 0.0123 | 0.0123 |
| A BALANCED | 2 | 0.0010 | 0.6656 | 0.9596 | 0.8090 | 0.7273 | 0.7660 | 0.7667 | 72 | 17 | 27 | 56846 | {:.4f} | 1.0000 | 0.0151 | 0.0151 |
| A BALANCED | 3 | 0.0010 | 0.6690 | 0.9523 | 0.8500 | 0.6869 | 0.7598 | 0.7637 | 68 | 12 | 31 | 56851 | {:.4f} | 1.0000 | 0.0126 | 0.0127 |
| A BALANCED | 4 | 0.0010 | 0.6751 | 0.9564 | 0.8046 | 0.7071 | 0.7527 | 0.7539 | 70 | 17 | 29 | 56846 | {:.4f} | 1.0000 | 0.0133 | 0.0134 |
| B FULL-TRAIN | 0 | 0.0010 | 0.0524 | 0.8967 | 0.1124 | 0.2020 | 0.1444 | 0.1487 | 20 | 158 | 79 | 56705 | {:.4f} | 1.0000 | 0.0079 | 0.0088 |
| B FULL-TRAIN | 1 | 0.0010 | 0.1295 | 0.9099 | 0.5600 | 0.1414 | 0.2258 | 0.2808 | 14 | 11 | 85 | 56852 | {:.4f} | 0.9999 | 0.0080 | 0.0085 |
| B FULL-TRAIN | 2 | 0.0010 | 0.1101 | 0.9309 | 0.1783 | 0.2323 | 0.2018 | 0.2019 | 23 | 106 | 76 | 56757 | {:.4f} | 0.9999 | 0.0103 | 0.0108 |
| B FULL-TRAIN | 3 | 0.0010 | 0.2229 | 0.9243 | 0.4681 | 0.2222 | 0.3014 | 0.3217 | 22 | 25 | 77 | 56838 | {:.4f} | 0.9999 | 0.0083 | 0.0087 |
| B FULL-TRAIN | 4 | 0.0010 | 0.1824 | 0.9172 | 0.3559 | 0.2121 | 0.2658 | 0.2738 | 21 | 38 | 78 | 56825 | {:.4f} | 1.0000 | 0.0097 | 0.0105 |
| A BALANCED | 0 | 0.0100 | 0.6180 | 0.9511 | 0.7654 | 0.6263 | 0.6889 | 0.6919 | 62 | 19 | 37 | 56844 | {:.4f} | 0.9980 | 0.0999 | 0.1003 |
| A BALANCED | 1 | 0.0100 | 0.6754 | 0.9605 | 0.7683 | 0.6364 | 0.6961 | 0.6988 | 63 | 19 | 36 | 56844 | {:.4f} | 0.9955 | 0.1031 | 0.1034 |
| A BALANCED | 2 | 0.0100 | 0.6464 | 0.9593 | 0.8182 | 0.7273 | 0.7701 | 0.7710 | 72 | 16 | 27 | 56847 | {:.4f} | 0.9951 | 0.1230 | 0.1234 |
| A BALANCED | 3 | 0.0100 | 0.6631 | 0.9514 | 0.8500 | 0.6869 | 0.7598 | 0.7637 | 68 | 12 | 31 | 56851 | {:.4f} | 0.9989 | 0.1063 | 0.1070 |
| A BALANCED | 4 | 0.0100 | 0.6664 | 0.9569 | 0.8118 | 0.6970 | 0.7500 | 0.7518 | 69 | 16 | 30 | 56847 | {:.4f} | 0.9964 | 0.1081 | 0.1087 |
| B FULL-TRAIN | 0 | 0.0100 | 0.0456 | 0.8972 | 0.1033 | 0.2222 | 0.1410 | 0.1494 | 22 | 191 | 77 | 56672 | {:.4f} | 0.9995 | 0.0657 | 0.0730 |
| B FULL-TRAIN | 1 | 0.0100 | 0.1366 | 0.9055 | 0.3469 | 0.1717 | 0.2297 | 0.2432 | 17 | 32 | 82 | 56831 | {:.4f} | 0.9943 | 0.0679 | 0.0719 |
| B FULL-TRAIN | 2 | 0.0100 | 0.0912 | 0.9262 | 0.1778 | 0.1616 | 0.1693 | 0.1681 | 16 | 74 | 83 | 56789 | {:.4f} | 0.9898 | 0.0827 | 0.0869 |
| B FULL-TRAIN | 3 | 0.0100 | 0.2143 | 0.9222 | 0.3962 | 0.2121 | 0.2763 | 0.2890 | 21 | 32 | 78 | 56831 | {:.4f} | 0.9912 | 0.0683 | 0.0716 |
| B FULL-TRAIN | 4 | 0.0100 | 0.1644 | 0.9137 | 0.3000 | 0.2121 | 0.2485 | 0.2512 | 21 | 49 | 78 | 56814 | {:.4f} | 0.9990 | 0.0781 | 0.0843 |
| A BALANCED | 0 | 0.0500 | 0.6133 | 0.9473 | 0.7558 | 0.6566 | 0.7027 | 0.7040 | 65 | 21 | 34 | 56842 | {:.4f} | 0.9053 | 0.2528 | 0.2539 |
| A BALANCED | 1 | 0.0500 | 0.6859 | 0.9505 | 0.7683 | 0.6364 | 0.6961 | 0.6988 | 63 | 19 | 36 | 56844 | {:.4f} | 0.8697 | 0.2677 | 0.2685 |
| A BALANCED | 2 | 0.0500 | 0.5761 | 0.9474 | 0.8111 | 0.7374 | 0.7725 | 0.7730 | 73 | 17 | 26 | 56846 | {:.4f} | 0.8744 | 0.2980 | 0.2987 |
| A BALANCED | 3 | 0.0500 | 0.6326 | 0.9488 | 0.8590 | 0.6768 | 0.7571 | 0.7621 | 67 | 11 | 32 | 56852 | {:.4f} | 0.9881 | 0.2802 | 0.2819 |
| A BALANCED | 4 | 0.0500 | 0.6025 | 0.9416 | 0.8272 | 0.6768 | 0.7444 | 0.7478 | 67 | 14 | 32 | 56849 | {:.4f} | 0.8084 | 0.2561 | 0.2571 |
| B FULL-TRAIN | 0 | 0.0500 | 0.0329 | 0.8986 | 0.0916 | 0.1212 | 0.1043 | 0.1036 | 12 | 119 | 87 | 56744 | {:.4f} | 0.9903 | 0.1665 | 0.1835 |
| B FULL-TRAIN | 1 | 0.0500 | 0.1213 | 0.8913 | 0.2623 | 0.1616 | 0.2000 | 0.2048 | 16 | 45 | 83 | 56818 | {:.4f} | 0.9414 | 0.1808 | 0.1913 |
| B FULL-TRAIN | 2 | 0.0500 | 0.0454 | 0.9049 | 0.0536 | 0.2323 | 0.0871 | 0.1085 | 23 | 406 | 76 | 56457 | {:.4f} | 0.8620 | 0.1923 | 0.2023 |
| B FULL-TRAIN | 3 | 0.0500 | 0.1600 | 0.9139 | 0.4359 | 0.1717 | 0.2464 | 0.2728 | 17 | 22 | 82 | 56841 | {:.4f} | 0.8764 | 0.1729 | 0.1810 |
| B FULL-TRAIN | 4 | 0.0500 | 0.0846 | 0.9011 | 0.1531 | 0.1515 | 0.1523 | 0.1508 | 15 | 83 | 84 | 56780 | {:.4f} | 0.9871 | 0.1817 | 0.1976 |

## D. Recall versus AUPRC

**Analiză exploratorie** (pragurile se aleg și se evaluează pe același validation). Regula: dintre pragurile curbei PR cu recall ≥ țintă, cel mai mare. AUPRC nu depinde de prag și este cel al modelului. „Alarme excesive" = precision < 10 % (convenție de proiect). Medii pe 5 seeds; tabelul complet per seed și p: `threshold_recall_analysis.csv`.

| p | target recall | varianta | recall | precision | F1 | TP | FP | FN | threshold | AUPRC | realizabil | alarme excesive (seeds din 5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0000 | 0.7000 | A BALANCED | 0.7071 | 0.6168 | 0.6464 | 70.0000 | 53.0000 | 29.0000 | 0.4890 | 0.6617 | 5 | 0 |
| 0.0000 | 0.7000 | B FULL-TRAIN | 0.7071 | 0.0218 | 0.0423 | 70.0000 | 3297.6000 | 29.0000 | 0.5085 | 0.1403 | 5 | 5 |
| 0.0000 | 0.7500 | A BALANCED | 0.7576 | 0.3053 | 0.4235 | 75.0000 | 211.0000 | 24.0000 | 0.3829 | 0.6617 | 5 | 0 |
| 0.0000 | 0.7500 | B FULL-TRAIN | 0.7576 | 0.0193 | 0.0375 | 75.0000 | 4140.6000 | 24.0000 | 0.4819 | 0.1403 | 5 | 5 |
| 0.0000 | 0.8000 | A BALANCED | 0.8081 | 0.1063 | 0.1860 | 80.0000 | 791.8000 | 19.0000 | 0.2919 | 0.6617 | 5 | 1 |
| 0.0000 | 0.8000 | B FULL-TRAIN | 0.8081 | 0.0163 | 0.0320 | 80.0000 | 5359.0000 | 19.0000 | 0.4547 | 0.1403 | 5 | 5 |
| 0.0000 | 0.8500 | A BALANCED | 0.8586 | 0.0337 | 0.0647 | 85.0000 | 2588.2000 | 14.0000 | 0.2384 | 0.6617 | 5 | 5 |
| 0.0000 | 0.8500 | B FULL-TRAIN | 0.8586 | 0.0104 | 0.0205 | 85.0000 | 8782.6000 | 14.0000 | 0.3996 | 0.1403 | 5 | 5 |
| 0.0500 | 0.7000 | A BALANCED | 0.7071 | 0.6939 | 0.6947 | 70.0000 | 34.4000 | 29.0000 | 0.5023 | 0.6221 | 5 | 0 |
| 0.0500 | 0.7000 | B FULL-TRAIN | 0.7071 | 0.0176 | 0.0342 | 70.0000 | 4021.0000 | 29.0000 | 0.4978 | 0.0889 | 5 | 5 |
| 0.0500 | 0.7500 | A BALANCED | 0.7576 | 0.3368 | 0.4462 | 75.0000 | 220.0000 | 24.0000 | 0.4855 | 0.6221 | 5 | 0 |
| 0.0500 | 0.7500 | B FULL-TRAIN | 0.7576 | 0.0164 | 0.0321 | 75.0000 | 4679.4000 | 24.0000 | 0.4959 | 0.0889 | 5 | 5 |
| 0.0500 | 0.8000 | A BALANCED | 0.8081 | 0.1117 | 0.1897 | 80.0000 | 947.4000 | 19.0000 | 0.4752 | 0.6221 | 5 | 3 |
| 0.0500 | 0.8000 | B FULL-TRAIN | 0.8081 | 0.0121 | 0.0238 | 80.0000 | 6887.2000 | 19.0000 | 0.4917 | 0.0889 | 5 | 5 |
| 0.0500 | 0.8500 | A BALANCED | 0.8586 | 0.0353 | 0.0674 | 85.0000 | 2801.0000 | 14.0000 | 0.4707 | 0.6221 | 5 | 5 |
| 0.0500 | 0.8500 | B FULL-TRAIN | 0.8586 | 0.0090 | 0.0178 | 85.0000 | 9985.0000 | 14.0000 | 0.4878 | 0.0889 | 5 | 5 |

- **A (BALANCED):** la p = 0.05, recall ≥ 70 % se obține cu precision 0.69 și 34 de alarme false în medie (față de 16.4 la pragul F1-optim, unde recall este 0.677). Recall ≥ 75 % costă deja 220 de alarme false (precision 0.34). La ≥ 80 % apar 947 de alarme false (precision 0.11; 3 din 5 seeds sub 10 %), iar la ≥ 85 % 2.801 (precision 0.035; 5 din 5 seeds). Deci un recall de ~70 % este realizabil fără explozie de alarme false, peste el costul crește brusc.
- **B (FULL-TRAIN):** chiar și ținta de 70 % cere ~4.000 de alarme false (precision 0.018) la p = 0.05, în toate cele 5 seeds. Nicio țintă nu este utilizabilă.
- Ajustarea pragului nu schimbă AUPRC și nu reduce zgomotul. Ea mută punctul de funcționare pe aceeași curbă PR.
- Răspuns la întrebarea „putem detecta mai multe fraude fără scădere importantă a AUPRC și fără creștere disproporționată a FP?": **cu varianta B nu**, iar cu A doar până în jurul a 70 %.

## E. Interpretarea cuantică

Cele 4 componente PCA, aduse în [0, π], devin unghiurile `RZ(2xᵢ)` din encodare și argumentele `RZ(2(π − xᵢ)(π − xⱼ))` ale celor 3 blocuri ZZ. Alt encoder produce alți vectori de unghiuri și deci alte stări pregătite, de unde reantrenarea obligatorie. Circuitul, numărul de porți, CNOT-urile și locurile de zgomot nu se schimbă.

**Acesta este un experiment asupra encoderului clasic, nu o nouă tehnică de eliminare a porților sau de reducere fizică a zgomotului.** Costul cuantic per execuție este același în A și B.

**Ce arată PCA-ul (§ pca_analysis.csv, pca_angle_dimensions.csv):**

| mărime | A BALANCED | B FULL-TRAIN |
|---|---|---|
| rânduri la PCA.fit | 590 | 170883 |
| EVR PC1 | 0.6685 | 0.1213 |
| EVR PC2 | 0.1135 | 0.0952 |
| EVR PC3 | 0.0705 | 0.0799 |
| EVR PC4 | 0.0342 | 0.0643 |
| EVR cumulat | 0.8867 | 0.3607 |
| varianță din tot train captată (suma) | 0.1805 | 0.3607 |
| |cos| PC1 vs PCA doar-legit | 0.214 | 0.993 |
| |cos| minim (4 componente) vs PCA doar-legit | 0.073 | 0.916 |

- PCA-ul lui B este practic PCA-ul tranzacțiilor legitime: cosinusul PC1 cu PCA-ul fitat doar pe legitime este 0.993, iar minimul pe cele 4 componente 0.916. Pentru A aceleași valori sunt 0.21 și 0.07, adică direcțiile lui A nu sunt cele ale clasei legitime. Cosinusul PC1 dintre A și B este doar ~0.31 (PC4: 0.003–0.04), deci cei doi encoderi proiectează pe subspații foarte diferite (`pca_component_similarity.csv`).
- Varianța explicată nu înseamnă separabilitate: B explică 36 % din varianța din tot train (față de 18 % pentru direcțiile lui A), dar are performanță mult mai slabă. Pe cele 590 de exemple 1:1, PC1 al lui A explică 67 %, pentru că în acel set clasele sunt egale și diferența dintre ele contribuie la varianță. Pe tot train-ul, fraudele sunt 0.17 %, deci contribuția lor la varianță este neglijabilă.
- Diagnostic per dimensiune, medii pe seeds, validation (AUROC univariat al unghiului; 0.5 = fără separare):

| varianta | dimensiune | unghi mediu legit | unghi mediu fraudă | d Cohen (fraudă − legit) | AUROC univariat al unghiului | proporție în afara domeniului (val) | valori val < 0 (înainte de clip) | valori val > π (înainte de clip) |
|---|---|---|---|---|---|---|---|---|
| A BALANCED | 0 | 3.0872 | 2.3535 | -1.4726 | 0.0576 | 0.0013 | 0.0000 | 75.0000 |
| A BALANCED | 1 | 2.4762 | 2.6513 | 1.2988 | 0.8797 | 0.0000 | 0.0000 | 0.0000 |
| A BALANCED | 2 | 1.5755 | 1.4496 | -0.5715 | 0.1727 | 0.0000 | 0.0000 | 2.0000 |
| A BALANCED | 3 | 1.2823 | 1.2711 | -0.0959 | 0.3727 | 0.0000 | 0.0000 | 0.0000 |
| B FULL-TRAIN | 0 | 2.9108 | 2.5499 | -0.8270 | 0.2608 | 0.0017 | 13.6000 | 84.2000 |
| B FULL-TRAIN | 1 | 1.3753 | 1.5660 | 0.8142 | 0.7493 | 0.0016 | 91.6000 | 0.0000 |
| B FULL-TRAIN | 2 | 2.7703 | 2.2727 | -1.2247 | 0.1414 | 0.0026 | 1.0000 | 149.8000 |
| B FULL-TRAIN | 3 | 1.1637 | 1.4411 | 0.8414 | 0.7440 | 0.0031 | 111.2000 | 66.8000 |

  Cea mai separatoare dimensiune a lui A are AUROC 0.058 (adică ~0.94 inversat, d Cohen −1.47), cea a lui B 0.141 (~0.86, d −1.22). În medie, separarea univariată este mai slabă în B (media |2·AUROC − 1| pe 4 dimensiuni ≈ 0.55, față de ≈ 0.64 în A, calculată din valorile de mai sus).
- Valorile care ies din domeniul de fit al MinMaxScaler (apoi tăiate de `np.clip`) sunt rare în ambele variante: ≤ 0.3 % din validation per dimensiune (A: 0.13 % pe dimensiunea 0, rest ~0; B: 0.16–0.31 %). Clipping-ul nu explică diferența.
- **Ce nu am demonstrat:** legătura cauzală dintre „PCA dominat de clasa legitimă" și performanța slabă este consistentă cu datele (PCA-ul lui B ≈ PCA doar-legit; separare univariată mai slabă; AUROC scade de la 0.947 la 0.902), dar nu am făcut o ablație care să o izoleze (de exemplu PCA pe alte raporturi legit:fraudă) și nu am calculat corelații pe 5 seeds.

**Zgomot (observație, nu avantaj):** B are MAE al scorurilor mai mic (0.179 vs 0.271) și ρ Spearman mai mare (0.931 vs 0.889) la p = 0.05. Asta nu înseamnă robustețe mai bună: scorurile lui B sunt mai puțin informative încă de la p = 0 (AUPRC 0.140), deci o perturbare mai mică a unor scoruri slabe nu ajută. Pierderea de AUPRC față de p = 0 este 0.040 pentru A (6 %) și 0.052 pentru B (37 % relativ); diferența nu este consistentă (B mai bun în 1 din 5 seeds).

## F. Rezultate negative și limitări

- **Rezultat negativ principal:** FULL-TRAIN PCA degradează toate metricile de clasificare la toate nivelurile de zgomot (AUPRC la p = 0: 0.662 → 0.140). Degradarea apare deja fără zgomot, deci vine din encoder, nu din zgomot.
- **Ce nu putem afirma:** că varianta A este optimă; că rezultatul se generalizează la alte topologii, alte numere de componente sau alte tehnici de fit (de exemplu PCA ponderat pe clase); că explicația „PCA dominat de legitime" este cauza (vezi mai sus); că performanța ar fi aceeași pe un test independent.
- **Validation folosit de mai multe ori:** alegerea epocii (AUPRC la p = 0), pragul max-F1 și pragurile pentru ținte de recall folosesc același validation pe care se raportează. Cifrele sunt optimiste. Testul nu a fost folosit.
- Doar 5 seeds: nu putem declara semnificație statistică (testul semnului exact are p minim 0.0625). Concluzia se sprijină pe mărimea și consistența efectului.
- Cele 590 de exemple 1:1 diferă pe seed; variația dintre seeds (std AUPRC 0.04 pentru A) este mică față de diferența A–B (0.53).
- Ordinea/semnul componentelor PCA poate varia între fit-uri; semnele PC2–PC4 ale lui B față de A diferă (`pca_component_similarity.csv`), ceea ce este normal și nu schimbă varianța explicată, dar reflectă faptul că encoderul B nu este o simplă re-scalare a lui A.

## G. Recomandare

Aplicând logica stabilită:
1. **Protocol și leakage:** respectate de ambele variante. `PCA.fit` al lui B vede doar rânduri din train, iar validation și testul nu intră în nicio potrivire.
2. **AUPRC la p = 0.05 (criteriu de protecție):** B pierde 0.533 față de A, cu mult peste toleranța de proiect de 0.02 (toleranță de proiect, nu regulă oficială).
3. **Recall și FP:** B are recall mai mic (−50.9 puncte procentuale) și de ~8 ori mai multe alarme false (135 vs 16.4).
4. Nu există un candidat cu recall superior care să intre în toleranța de 0.02 AUPRC.
5. Diferențele sunt consecvente între seeds, în defavoarea lui B.

**Decizie: se păstrează BALANCED PCA (A).** Rezultatele sunt exploratorii pe validation și nu reprezintă performanță pe un test independent.

## Cost ledger (`cost_ledger.json`)

Execuțiile și shots-urile sunt **nominale** (estimare pentru hardware: parameter shift 1 + 2·32 = 65 de execuții / exemplu, 1024 de shots / execuție). Simularea a folosit valori de așteptare exacte, fără shots. Circuitul este același în A și B, deci **FULL-TRAIN PCA nu reduce numărul de CNOT, de locuri de zgomot sau costul per execuție cuantică.** B adaugă doar cost clasic (un `PCA.fit` pe 170.883 de rânduri), neglijabil față de simulare.

| mărime | A BALANCED | B FULL-TRAIN |
|---|---|---|
| execuții nominale / seed | 3,087,208 | 3,087,208 |
| execuții nominale, 5 seeds | 15,436,040 | 15,436,040 |
| shots ipotetice (1024 / execuție), 5 seeds | 15,806,504,960 | 15,806,504,960 |
| pași Adam, 5 seeds | 2850 | 2850 |
| simulare antrenare, 5 seeds [s] | 81.9 (din Exp. 9) | 66.8 |
| simulare evaluare validation (4 niveluri p), 5 seeds [s] | 1144 | 1131 |
| PCA.fit clasic, 5 seeds [s] | 0.022 | 0.241 |
| rânduri la PCA.fit | 590 | 170883 |

Timpul de antrenare al lui A provine din Exp. 9 (altă sesiune); cel al lui B este măsurat acum, deci nu sunt comparabile exact.

## Fișiere

`config.json`, `checks.csv`, `exp9_reproduction.csv`, `metrics_per_seed.csv`, `summary_metrics.csv`, `paired_comparison.csv`, `pca_analysis.csv`, `pca_angle_dimensions.csv`, `pca_component_similarity.csv`, `threshold_recall_analysis.csv`, `cost_ledger.json`, `train_runs.jsonl`, `weights/full_train/`, `plots/1…9`. `scores_cache/` (scoruri pe validation, ~18 MB) este doar pentru reluarea rulării și nu se comite.

# Experimentul 9: ZZ feature map rar (STAR / LINEAR) și reducerea CNOT

**Model de referință:** Exp. 3 (ZZ ×1, ansatz ×3, 4 qubiți, 32 de parametri, readout ⟨Z₀⟩), ponderi salvate, **fără reantrenare**.
**Modificare unică:** setul de perechi cu interacțiune ZZ în feature map. Fiecare pereche exclusă elimină blocul întreg `CNOT → RZ → CNOT`.
**Date:** doar train (590 de exemple 1:1 / seed) și validation (56 962 de tranzacții, 99 de fraude). Testul nu a fost folosit. 5 seeds, p = 0 / 0.001 / 0.01 / 0.05, `default.mixed`, prag max-F1 pe validation la același p.
**Notebook:** `notebooks/exp9_sparse_zz_feature_map.ipynb`. PennyLane 0.45.1, torch 2.14.1, dtype implicit torch float64.

| Topologie | Perechi ZZ | Ponderi |
|---|---|---|
| FULL (Exp. 3) | (0,1) (0,2) (0,3) (1,2) (1,3) (2,3) | refolosite din `result_exp3/weights` |
| STAR | (0,1) (0,2) (0,3) | antrenate de la zero, `weights/star/` |
| LINEAR | (0,1) (1,2) (2,3) | antrenate de la zero, `weights/linear/` |

## 1. Ce am verificat înainte de a compara (toate trecute, `checks.csv`)

- **FULL = Exp. 3 poartă cu poartă:** tape-ul decompus este identic cu `vqc` oficial (`qml.equal`, 20 de intrări).
- **STAR / LINEAR = FULL fără blocurile excluse:** circuitele păstrează codificarea `H, RZ(2xᵢ)`, formula unghiului `2(π − xᵢ)(π − xⱼ)`, ordinea blocurilor rămase și ansatz-ul neschimbat.
- **Check 6.1 pe fiecare topologie:** feature map-ul coincide cu varianta construită cu `qml.IsingZZ` pe aceleași perechi (fidelitate 1 ± 2e-15).
- **Bucla de antrenare reproduce Exp. 3 exact:** FULL seed 0, reantrenat 30 de epoci în float64, are același istoric, aceeași epocă aleasă și aceleași ponderi (eroare 0.0). Deci STAR / LINEAR sunt antrenate în condițiile Exp. 3:
  - aceleași unghiuri (hash SHA-256 identic per seed);
  - aceeași inițializare `init_weights(seed)`;
  - aceeași ordine a mini-batch-urilor (hash identic cu FULL).
- **FULL sub zgomot:** recalculat aici, reproduce Exp. 8 (float64) exact (eroare 0) și Exp. 3 istoric (float32) la 7e-8.
- **`default.mixed` la p = 0 = `default.qubit`:** ≤ 7e-13 pentru STAR și LINEAR.

## 2. Structura circuitelor (decompoziție în `NATIVE_GATES`, numărată, nu hardcodată)

| | FULL (Exp. 3) | STAR | LINEAR |
|---|---:|---:|---:|
| Interacțiuni ZZ | 6 | 3 | 3 |
| Porți native | 67 | 58 | 58 |
| Porți 1 qubit / 2 qubiți | 46 / 21 | 43 / 15 | 43 / 15 |
| **CNOT** | **21** | **15 (−6, −29%)** | **15 (−6, −29%)** |
| **Adâncime** | **31** | **26 (−5, −16%)** | **25 (−6, −19%)** |
| **Locuri de zgomot** | **88** | **73 (−15, −17%)** | **73 (−15, −17%)** |
| din care feature map (porți / CNOT / zgomot) | 26 / 12 / 38 | 17 / 6 / 23 | 17 / 6 / 23 |
| Parametri antrenabili | 32 | 32 | 32 |
| Execuții / exemplu: nominal 1 + 2P / `qml.Tracker` | 65 / 41 | 65 / 41 | 65 / 41 |
| Parametri cu gradient nenul | 19 | 19 | 19 |

Estimarea preliminară (15 CNOT, ~73 de locuri de zgomot) este confirmată. Ansatz-ul rămâne identic (9 CNOT, 50 de locuri de zgomot în toate trei), deci întreaga reducere vine din feature map.

## 3. Rezultate pe validation (medie ± std pe 5 seeds)

| Metrică | p | FULL (Exp. 3) | STAR | LINEAR |
|---|---|---:|---:|---:|
| **AUPRC** | 0 | 0.6274 ± 0.0280 | **0.6617 ± 0.0226** | 0.6388 ± 0.0621 |
| | 0.001 | 0.6267 ± 0.0276 | **0.6606 ± 0.0228** | 0.6377 ± 0.0609 |
| | 0.01 | 0.6226 ± 0.0236 | **0.6539 ± 0.0227** | 0.6285 ± 0.0512 |
| | **0.05** | 0.5867 ± 0.0385 | **0.6221 ± 0.0411** | 0.5421 ± 0.0186 |
| AUROC | 0.05 | 0.9477 ± 0.0043 | 0.9471 ± 0.0033 | 0.9541 ± 0.0058 |
| F1 (prag max-F1, optimist) | 0.05 | 0.6800 ± 0.0303 | **0.7346 ± 0.0337** | 0.6721 ± 0.0400 |
| MCC | 0.05 | 0.6813 ± 0.0312 | **0.7371 ± 0.0339** | 0.6747 ± 0.0354 |
| Spearman ρ | 0.01 | 0.9956 ± 0.0045 | 0.9968 ± 0.0016 | 0.9970 ± 0.0021 |
| Spearman ρ | 0.05 | 0.8751 ± 0.1102 | 0.8892 ± 0.0655 | 0.9312 ± 0.0339 |

**Referință clasică (Exp. 4):** regresia logistică pe aceleași 4 unghiuri are AUPRC pe validation 0.6633 ± 0.0251. STAR la p = 0 este practic la nivelul ei (0.662), iar la p = 0.05 rămâne sub (0.622). Nu afirmăm avantaj cuantic.

### Diferențe pereche față de FULL (același seed, aceleași unghiuri, aceeași inițializare)

| Δ AUPRC | p = 0 | p = 0.01 | p = 0.05 |
|---|---:|---:|---:|
| STAR − FULL | +0.034 ± 0.039 (4/5) | +0.031 ± 0.033 (4/5) | **+0.035 ± 0.044 (4/5)** |
| LINEAR − FULL | +0.011 ± 0.080 (4/5) | +0.006 ± 0.059 (4/5) | **−0.045 ± 0.040 (1/5)** |

Valori per seed:

| Seed | STAR − FULL, p = 0 | STAR − FULL, p = 0.05 | LINEAR − FULL, p = 0 | LINEAR − FULL, p = 0.05 |
|---|---:|---:|---:|---:|
| 0 | −0.013 | +0.025 | +0.040 | −0.073 |
| 1 | +0.022 | +0.047 | +0.015 | −0.075 |
| 2 | +0.065 | −0.026 | +0.067 | −0.068 |
| 3 | +0.014 | +0.097 | −0.127 | +0.018 |
| 4 | +0.083 | +0.034 | +0.062 | −0.025 |

LINEAR seed 3 este un outlier: la p = 0 converge slab (AUPRC 0.528), deci câștigul lui la p = 0.05 nu reflectă robustețe.

## 4. Impactul direct al zgomotului: trei lucruri diferite

| | FULL | STAR | LINEAR |
|---|---:|---:|---:|
| **(a) Expunere structurală**: locuri de zgomot | 88 | 73 | 73 |
| **(b) Perturbarea scorurilor**: MAE, p = 0.01 | 0.143 ± 0.028 | 0.108 ± 0.009 (−24%, 5/5) | 0.086 ± 0.015 (−40%, 5/5) |
| MAE, p = 0.05 | 0.294 ± 0.051 | 0.271 ± 0.019 (−8%, 4/5) | 0.219 ± 0.025 (−25%, 5/5) |
| RMSE, p = 0.05 | 0.297 ± 0.052 | 0.272 ± 0.019 | 0.221 ± 0.026 |
| Factor de contracție \|⟨Z₀⟩\|(0.05) / \|⟨Z₀⟩\|(0) | 0.060 ± 0.035 | 0.109 ± 0.020 | 0.175 ± 0.050 |
| **(c) Clasificare**: pierdere AUPRC 0 → 0.05 (absolută) | 0.041 ± 0.047 | 0.040 ± 0.044 | **0.097 ± 0.071** |
| Pierdere AUPRC 0 → 0.05 (relativă) | 6.4% | 5.9% | **14.3%** |
| Std AUPRC între seeds, p = 0.05 | 0.039 | 0.041 | 0.019 |

**Cum se citește:**
- **(a) → (b) funcționează.** Cu 15 locuri de zgomot mai puțin, la p = 0.05 FULL păstrează 6% din amplitudinea semnalului ⟨Z₀⟩ fără zgomot, STAR 11% și LINEAR 18%. Scorurile sunt efectiv mai puțin perturbate, mai ales la p = 0.01.
- **(b) → (c) nu urmează automat.** O contracție uniformă spre 0.5 nu schimbă ordinea scorurilor, deci nici AUPRC-ul. AUPRC scade doar prin distorsiunea neuniformă a ordinii.
  - LINEAR are cea mai mică perturbare medie, dar cea mai mare pierdere de AUPRC (4/5 seeds pierd 0.10–0.16 între p = 0 și p = 0.05). Rangul general rămâne stabil (ρ = 0.93, AUROC 0.954), dar zona cu scor mare, unde se decide AUPRC-ul pe 99 de fraude, se reordonează.
  - STAR se degradează exact cât FULL (0.040 vs 0.041; Δ pereche −0.001 ± 0.070, 3/5).
- **Câștigul STAR nu vine din robustețe, ci din clasificarea fără zgomot.** Avantajul de ~+0.035 există deja la p = 0 și se păstrează la toate nivelurile de zgomot.
- MAE depinde și de scara scorurilor: LINEAR are scoruri fără zgomot mai apropiate de 0.5 (\|⟨Z₀⟩\| mediu 0.53 vs 0.63 la FULL). Factorul de contracție confirmă totuși că perturbarea relativă este mai mică.

## 5. Cost

| Per seed | FULL | STAR | LINEAR |
|---|---:|---:|---:|
| Antrenare, nominal 30 × 590 × 65 | 1 150 500 | 1 150 500 | 1 150 500 |
| Antrenare, `qml.Tracker` 30 × 590 × 41 | 725 700 | 725 700 | 725 700 |
| Validare după fiecare epocă, 30 × 56 962 | 1 708 860 | 1 708 860 | 1 708 860 |
| Validation finală (p = 0 o dată + 3 p cu zgomot) | 227 848 | 227 848 | 227 848 |
| **Total execuții, nominal / Tracker** | **3 087 208 / 2 662 408** | **identic** | **identic** |
| Shots nominale, 5 seeds (1024 / execuție, nominal) | 15.81 mld. | 15.81 mld. | 15.81 mld. |
| CNOT / execuție | 21 | 15 | 15 |
| Aplicări CNOT / seed (derivat, nominal) | 64.8 mil. | 46.3 mil. (−29%) | 46.3 mil. (−29%) |
| Timp simulare cu zgomot, 3 p / seed | 63.1 ± 6.5 s | 49.8 ± 2.1 s (−21%) | 52.0 ± 4.8 s (−18%) |
| Timp antrenare simulator / seed | 16.6 s (Exp. 3) | 16.4 s | 21.8 s* |

\* Diferența de timp la LINEAR vine din încărcarea mașinii (seeds 0–2), nu din circuit: au aceleași porți ca STAR.

- **Execuțiile și shot-urile nu se schimbă**, pentru că depind de parametri, exemple, epoci și valori p, nu de porți. Nu se schimbă nici parametrii care influențează citirea: 20 deplasați de parameter shift, 19 cu gradient nenul.
- **Se schimbă costul fiecărei execuții:** −29% CNOT, −13% porți, −16% / −19% adâncime.
- Shots-urile sunt nominale: simularea este exactă (`shots=None`), deci niciun shot nu a fost tras. FULL nu a fost reantrenat în Exp. 9; costul lui de antrenare este cel al Exp. 3.

## 6. Răspunsuri

1. **Câte CNOT am eliminat?** 6 din 21 (−29%), în ambele topologii: 15 CNOT, toate din feature map (12 → 6).
2. **Cu cât au scăzut locurile de zgomot?** Cu 15, de la 88 la 73 (−17%), în ambele.
3. **Cu cât a scăzut adâncimea?** STAR: 31 → 26 (−5, −16%). LINEAR: 31 → 25 (−6, −19%).
4. **Care dintre STAR și LINEAR clasifică mai bine?** **STAR**: AUPRC mai mare la fiecare p, cu cel mai mare decalaj la p = 0.05 (0.622 vs 0.542). Are și varianța mult mai mică la p = 0 (std 0.023 vs 0.062).
5. **AUPRC la p = 0 / 0.01 / 0.05?**
   - FULL: 0.627 / 0.623 / 0.587;
   - STAR: 0.662 / 0.654 / 0.622;
   - LINEAR: 0.639 / 0.629 / 0.542.
6. **Care se degradează cel mai puțin?**
   - Ca scoruri: LINEAR (MAE −25% la p = 0.05, 5/5 seeds).
   - Ca AUPRC: STAR și FULL sunt la egalitate (−0.040 / −0.041, ~6%), iar LINEAR pierde cel mai mult (−0.097, 14%).
   - Niciuna nu are o degradare de AUPRC semnificativ mai mică decât FULL.
7. **Spearman ρ s-a îmbunătățit?** Doar în medie, nu consistent.
   - La p = 0.05: FULL 0.875, STAR 0.889, LINEAR 0.931. Media crește în principal prin seed 3 (+0.29 la ambele); pe seeds, ambele câștigă doar la 2/5.
   - Std-ul lui ρ scade clar (0.110 → 0.066 / 0.034), deci comportamentul este mai uniform între seeds.
8. **Scorurile sunt mai puțin perturbate?** Da, măsurat.
   - MAE la p = 0.01: −24% STAR, −40% LINEAR (5/5 seeds).
   - Contracția semnalului ⟨Z₀⟩ la p = 0.05 se reduce de la 94% (FULL) la 89% / 82%.
   - O perturbare mai mică nu a însemnat însă un AUPRC mai bun pentru LINEAR.
9. **Câte fraude detectăm și câte alarme false?** Pe validation, din 99 de fraude și 56 863 de tranzacții legitime, la pragul max-F1 (optimist, ales pe același set):

   | | p = 0.01: TP / FP | p = 0.05: TP / FP |
   |---|---:|---:|
   | FULL | 64.4 / 24.6 | 63.6 / 24.6 |
   | STAR | 66.8 / 16.4 | 67.0 / 16.4 |
   | LINEAR | 61.0 / 17.0 | 63.8 / 26.6 |

   STAR detectează ~3 fraude în plus cu ~8 alarme false mai puțin decât FULL la p = 0.05, dar diferențele pe seed sunt mari (ΔFP −8.2 ± 10.2, 3/5).
10. **Este justificată reducerea CNOT?**
    - **Pentru STAR, da, pe validation:** −6 CNOT, −15 locuri de zgomot și −5 straturi de adâncime, fără pierderi de clasificare la niciun p (+0.035 AUPRC la p = 0.05, 4/5 seeds).
    - Justificarea nu este cea din ipoteză: STAR nu se degradează mai puțin, ci pornește de mai sus și se degradează la fel.
    - **Pentru LINEAR, nu:** aceeași reducere structurală, dar AUPRC mai slab la p = 0.05 (−0.045, 4/5 seeds) și o pierdere de 2.4× mai mare sub zgomot.
11. **Se reduce costul efectiv?**
    - **Per execuție, da:** −29% CNOT, −13% porți, −16% / −19% adâncime, −18% / −21% timp de simulare cu zgomot.
    - **Numărul de execuții și de shots, nu:** 3 087 208 nominal / 2 662 408 Tracker per seed, identic, pentru că parametrii și pașii optimizerului sunt aceiași.
12. **Ce arhitectură merită investigată mai departe?** **STAR**, ca singura dintre cele trei care e cel puțin la fel de bună ca FULL la toate nivelurile de zgomot cu un circuit mai mic. LINEAR nu.
13. **Merită combinat STAR cu Rotation Fusion (Exp. 8)?** Da: cele două modificări sunt independente.
    - Fuziunea atinge doar perechile `H + RZ` (codificare) și `RY + RZ` (ansatz), care există identic în STAR. RZ-urile din blocurile ZZ nu se fuzionează.
    - Fuziunea este exactă fără zgomot, deci se aplică pe ponderile STAR deja antrenate, fără reantrenare.
    - **Estimare structurală, nemăsurată:** 58 − 20 = 38 de porți, 73 − 20 = 53 de locuri de zgomot, aceleași 15 CNOT. Valorile trebuie confirmate în experimentul următor.
    - În Exp. 8, fuziunea a adus +0.009 AUPRC la p = 0.05 (5/5 seeds) pe FULL. Câștigul pe STAR nu poate fi presupus.
14. **A redus eliminarea ZZ capacitatea de reprezentare?** Depinde de topologie (indicii, nu cauzalitate dovedită).
    - **LINEAR, plauzibil da:**
      - loss-ul de antrenare rămâne mai mare decât la FULL (final 0.381 vs 0.362);
      - un seed nu învață bine (AUPRC 0.528 la p = 0);
      - se degradează de 2.4× mai mult sub zgomot.
    - **STAR, nu se vede o pierdere:**
      - loss-ul de antrenare este *mai mic* decât la FULL (0.333 vs 0.362);
      - AUPRC este mai mare deja la p = 0;
      - perechile (1,2), (1,3), (2,3) nu par necesare pentru citirea pe qubitul 0, iar fără ele optimizarea ajunge mai jos în loss.
    - Cu 590 de exemple, 99 de fraude pe validation și 5 seeds, explicația rămâne o ipoteză.

## 7. Tabel comparativ și verdict

| | FULL ZZ (Exp. 3) | STAR ZZ | LINEAR ZZ |
|---|---:|---:|---:|
| Interacțiuni ZZ | 6 | 3 | 3 |
| CNOT / porți / adâncime | 21 / 67 / 31 | 15 / 58 / 26 | 15 / 58 / 25 |
| Locuri de zgomot | 88 | 73 | 73 |
| AUPRC p = 0 | 0.627 ± 0.028 | **0.662 ± 0.023** | 0.639 ± 0.062 |
| AUPRC p = 0.01 | 0.623 ± 0.024 | **0.654 ± 0.023** | 0.629 ± 0.051 |
| AUPRC p = 0.05 | 0.587 ± 0.039 | **0.622 ± 0.041** | 0.542 ± 0.019 |
| Δ AUPRC vs FULL, p = 0.05 (seeds mai bune) | — | +0.035 ± 0.044 (4/5) | −0.045 ± 0.040 (1/5) |
| Pierdere AUPRC 0 → 0.05 | 0.041 (6.4%) | 0.040 (5.9%) | 0.097 (14.3%) |
| Spearman ρ, p = 0.05 | 0.875 ± 0.110 | 0.889 ± 0.066 | 0.931 ± 0.034 |
| MAE noisy–noiseless, p = 0.05 | 0.294 | 0.271 | **0.219** |
| F1 p = 0.05 (optimist) | 0.680 | **0.735** | 0.672 |
| TP / FP p = 0.05 (din 99 / 56 863) | 63.6 / 24.6 | **67.0 / 16.4** | 63.8 / 26.6 |
| Execuții / seed, nominal (Tracker) | 3 087 208 (2 662 408) | identic | identic |
| Timp simulare zgomot / seed | 63.1 s | 49.8 s | 52.0 s |

**Verdict.**
- **Ipoteza este confirmată doar parțial.** Reducerea ZZ elimină 6 CNOT și 15 locuri de zgomot și reduce perturbarea scorurilor. Robustețea AUPRC la zgomot nu se îmbunătățește în niciuna dintre topologii.
- **LINEAR este un rezultat negativ:** mai puține CNOT, scoruri mai puțin perturbate, dar clasificare mai slabă la p = 0.05.
- **STAR este cel mai bun compromis observat pe validation:** circuit cu 29% mai puține CNOT, AUPRC mai mare la fiecare p, mai puține alarme false. Avantajul vine însă dintr-o clasificare fără zgomot mai bună, nu dintr-o degradare mai mică.

**Rezerve:**
- Diferențele STAR − FULL au std comparabil cu media (+0.035 ± 0.044), cu un seed contra la fiecare p.
- Epoca și pragul sunt alese pe același validation. Testul a fost deja consultat în Exp. 5, deci STAR nu are o evaluare independentă.

**Recomandare:** **Exp. 3 rămâne modelul principal raportat** (singurul cu rezultate pe test în protocolul oficial). STAR este candidatul exploratoriu de urmărit: de combinat cu rotation fusion și prezentat ca analiză pe validation, nu ca rezultat final.

## Fișiere

| Fișier | Conținut |
|---|---|
| `config.json` | Perechile ZZ și ordinea exactă a operațiilor native, per topologie |
| `checks.csv` | Toate verificările, cu erorile maxime |
| `circuit_comparison.csv` | Structura circuitelor și execuțiile Tracker |
| `train_runs.jsonl`, `weights/{star,linear}/seed_*.pt` | Antrenarea: ponderile alese, istoricul pe 30 de epoci, ponderile inițiale |
| `val_runs.jsonl` | 60 de înregistrări (3 topologii × 5 seeds × 4 p): metrici, ρ, MAE / RMSE, timp |
| `val_summary.csv`, `noise_impact.csv` | Sumare pe topologie și p |
| `paired_differences.csv`, `comparison_vs_exp3.csv` | Diferențe pereche față de FULL |
| `cost_ledger.csv`, `simulator_time.csv` | Cost |
| `plots/` | 12 grafice: circuite, CNOT, locuri de zgomot, adâncime, AUPRC / F1 / ρ / MAE vs p, AUPRC per seed, Δ pereche, trade-off, curbe de antrenare |

Fișierele `*.csv` sunt ignorate de git (`.gitignore`), ca la celelalte experimente. Se regenerează din `val_runs.jsonl` / `train_runs.jsonl` rulând notebookul, care reia din rezultatele salvate.

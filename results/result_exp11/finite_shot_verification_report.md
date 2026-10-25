# Experimentul 11 · verificarea analizei finite-shot și a costului cuantic

**Ce este:** o analiză suplimentară a Exp. 11. Nu înlocuiește metricile oficiale. Nicio reantrenare și nicio simulare nouă pe toată validarea: refolosim scorurile exacte salvate (`scores/`, `default.mixed`, zgomot oficial) și ponderile salvate.
**Unde:** `notebooks/exp11_finite_shot_verification.ipynb`. Fișierele noi au prefixul `finite_shot_` (plus `finite_shot_reps/` și `plots/fs1…fs5`). Rezultatele originale ale Exp. 11 nu au fost modificate. Testul nu a fost accesat.
**Monte Carlo:** 50 de repetări independente pentru fiecare (observabilă × seed × p × N). N ∈ {1 024, 4 096, 10 000, 32 768, 100 000}; p ∈ {0, 0.001, 0.01, 0.05}; 5 observabile × 5 seeds. În total: 25 000 de realizări pe câte 56 962 de tranzacții.

## 0. Pe scurt

- **Analiza inițială este corectă** matematic și numeric. AUPRC la p = 0.05 și 1 024 shots se reproduce: Z₀ **0.519**, față de 0.520 inițial; Z₀Z₁ **0.197**, față de 0.188 inițial. Diferențele sunt în limita erorii Monte Carlo (|z| ≤ 2.6 per seed).
- **Eșantionarea PennyLane directă confirmă modelul binomial**, pe 60 de cazuri (tranzacții reale, `default.mixed`, zgomot oficial).
- **Recalibrarea pragului contează.** Pragul exact aplicat scorurilor cu shots produce mii de alarme false. Pragul recalculat repară F1-ul parțial (Z₀Z₁: 0.05 → 0.29 la 1 024 shots), dar nu recuperează valoarea exactă (0.74). Pierderea vine din informație, nu din prag.
- **Mecanismul:** la p = 0.05, semnalul lui Z₀Z₁ este comprimat într-o fereastră de ~0.03 în scor, iar eroarea standard la 1 024 shots este 0.016. Cele 56 863 de tranzacții legitime se întind peste fraude. Separabilitatea relativă (d′ ≈ 2.3) este însă aceeași pentru toate observabilele.
- **Când revine avantajul lui Z₀Z₁:** la p = 0.05, Z₀Z₁ depășește Z₀ consistent la AUPRC abia de la ~**32 768 shots** (4/5 seeds). Pe F1 și recall nu îl depășește consistent nici la 100 000 de shots. La p = 0.01, Z₀Z₁ rămâne mai bun și la 1 024 shots, dar cu puțin (+0.021 AUPRC, 4/5 seeds).

## 1. Verificarea matematică și a implementării inițiale

**Formula este validă pentru toate cele cinci observabile.**

- **Pentru Z_k:** rezultatul b_k = 1 corespunde valorii proprii −1, deci s = (1 − ⟨Z_k⟩)/2 = P(b_k = 1).
- **Pentru Z₀Z₁:** valoarea proprie a lui |b⟩ este (−1)^(b₀⊕b₁), deci s = P(b₀ ⊕ b₁ = 1), adică probabilitatea parității impare.
- **Distribuția estimatorului:** fiecare shot este o variabilă Bernoulli(s), i.i.d. (aceeași stare ρ, inclusiv zgomotul depolarizant), deci K ~ Binomial(N, s). Estimatorul ŝ = K/N = (1 − Ô)/2 este estimatorul standard al lui ⟨O⟩, transformat liniar, cu E[ŝ] = s și Var[ŝ] = s(1 − s)/N.

| Punct verificat | Rezultat |
|---|---|
| Probabilitatea eșantionată | s salvat = P(O = −1). Corect. |
| Convenția semnului | ŝ crește cu numărul de rezultate −1, la fel ca scorul exact. Corect. |
| Limitele [0, 1] | Scorurile sunt strict în (0, 1); `clip` nu modifică nimic. |
| Tipuri | float64 → `binomial` dă int64 → ŝ în float64. Corect (verificat numeric). |
| Independența | Fiecare tranzacție și fiecare repetare primește eșantioane proprii; nicio realizare nu este refolosită. Corect. Singura slăbiciune: inițial, rezultatele depindeau de ordinea buclei. Acum fiecare configurație are propriul generator `default_rng([2026, i_obs, seed, i_p, N])`. |
| Alinierea scoruri ↔ etichete | Scorurile salvate reproduc AUPRC-ul și pragurile oficiale cu eroare 0 (toate cele 100 de seturi). Pe seed 0, unghiurile recalculate și ponderile reproduc scorurile salvate exact. Corect. |
| Pragul | Inițial: doar max-F1 **recalculat** pe scorurile cu shots (B). Acum se raportează și A (pragul exact) și C (½ / ½). |
| Repetări și metrici | Inițial: 10 repetări, fără AUROC, Spearman, MCC, FN, TN. Acum: 50 de repetări și toate metricile. |

**Nu am găsit nicio eroare de calcul în analiza inițială.**

## 2. Verificare independentă cu PennyLane (`finite_shot_verification.csv`)

**De ce o copie a QNode-ului oficial:** `noisy_qnode` oficial cere `diff_method="backprop"`, care nu acceptă shots finite (`QuantumFunctionError` în PennyLane 0.45.1).

**Cum am construit copia:** același `default.mixed`, același `qml.noise.insert(DepolarizingChannel, p, position="all")`, aceeași descompunere în `NATIVE_GATES`, cu `diff_method=None`. Am verificat că probabilitățile exacte ale copiei coincid cu cele ale QNode-ului oficial (eroare < 1e-12). Modelul de zgomot nu s-a schimbat.

**Ce am testat:** pentru fiecare observabilă (modelul ei antrenat, seed 0), 6 tranzacții reale din validation, cu scoruri diferite (fraude și legitime cu scor maxim / median / minim), la p = 0 și p = 0.05. Pentru fiecare, 200 de repetări × 1 024 shots eșantionate direct cu `qml.sample(wires=[0, 1, 2, 3])`:

- ŝ = fracția de rezultate cu b_k = 1 pentru Z_k, respectiv cu b₀ ⊕ b₁ = 1 pentru Z₀Z₁;
- pentru Z₀Z₁ am verificat și identitatea pe aceleași shots: fracția impară = (1 − media lui z₀z₁)/2.

| Criteriu | Rezultat (60 de cazuri) |
|---|---|
| Media PennyLane vs s exact | \|z\| ≤ 2.4 (prag 4) |
| Varianța PennyLane / (s(1 − s)/N) | 0.77 – 1.21 (bandă χ² 99.9%: 0.70 – 1.36) |
| KS PennyLane vs binomial | p-value minim 0.006. Cu 60 de teste, un p < 0.01 este de așteptat întâmplător; criteriile pe medie și varianță trec toate. |
| Calea `qml.expval(O)` cu `shots=1024` (Z₀, Z₀Z₁; 100 de execuții) | \|z\| ≤ 1.9, raport varianță 0.84 – 1.15 |

**Concluzie:** eșantionarea binomială pe scorurile exacte este statistic echivalentă cu măsurarea directă în PennyLane, inclusiv pentru paritatea Z₀Z₁.

## 3. Rezultate Monte Carlo (`finite_shot_metrics.csv`, `finite_shot_summary.csv`)

**Cum citim tabelele:** medie pe 5 seeds a mediilor MC ± std între seeds. În paranteză este std MC medie din interiorul unui seed. Prag B = max-F1 recalculat pe scorurile cu shots.

**AUPRC la p = 0.05:**

| | 1 024 | 4 096 | 10 000 | 32 768 | 100 000 | exact (oficial) |
|---|---|---|---|---|---|---|
| Z₀ | **0.519** ± 0.041 (0.028) | 0.608 ± 0.025 (0.018) | 0.616 ± 0.034 (0.013) | 0.620 ± 0.038 (0.009) | 0.621 ± 0.040 (0.006) | 0.622 ± 0.041 |
| Z₁ | 0.386 ± 0.096 (0.035) | 0.606 ± 0.018 (0.019) | 0.634 ± 0.011 (0.015) | 0.648 ± 0.012 (0.010) | 0.654 ± 0.012 (0.007) | 0.657 ± 0.011 |
| Z₂ | 0.306 ± 0.143 (0.028) | 0.499 ± 0.026 (0.021) | 0.546 ± 0.049 (0.016) | 0.565 ± 0.064 (0.010) | 0.569 ± 0.067 (0.007) | 0.574 ± 0.072 |
| Z₃ | 0.168 ± 0.075 (0.035) | 0.520 ± 0.056 (0.029) | 0.590 ± 0.044 (0.018) | 0.615 ± 0.045 (0.013) | 0.620 ± 0.045 (0.010) | 0.623 ± 0.048 |
| Z₀Z₁ | 0.197 ± 0.115 (0.036) | 0.557 ± 0.066 (0.026) | 0.628 ± 0.037 (0.018) | 0.660 ± 0.031 (0.011) | 0.672 ± 0.028 (0.007) | **0.678** ± 0.028 |

**AUPRC la p = 0.01:**

| | 1 024 | 10 000 | 100 000 | exact |
|---|---|---|---|---|
| Z₀ | 0.650 | 0.654 | 0.654 | 0.654 |
| Z₀Z₁ | **0.670** | 0.677 | 0.677 | 0.677 |
| Z₃ | 0.669 | 0.676 | 0.677 | 0.677 |

La p = 0.01, 1 024 shots ajung pentru toate observabilele: AUPRC rămâne în 0.006 de valoarea exactă.

**AUROC și Spearman la p = 0.05:**

- **AUROC scade mai puțin** decât AUPRC: Z₀ 0.947 → 0.884, Z₀Z₁ 0.941 → 0.840 la 1 024 shots.
- **Spearman ρ(ŝ, s) este foarte mic** chiar la 100 000 de shots: 0.12 (Z₀) și 0.07 (Z₀Z₁) la 1 024; 0.66 și 0.42 la 100 000. Ordinea internă a masei de tranzacții legitime devine practic aleatoare; AUPRC depinde doar de coada de sus a clasamentului.

**Seeds vs Monte Carlo:**

- **La N mare domină variația între seeds:** pentru Z₀ la 100 000 de shots, std între seeds 0.040 față de std MC 0.006. Asta e variația modelelor, nu a măsurării.
- **La 1 024 shots, ambele surse contează:** pentru Z₀Z₁, std între seeds 0.115, std MC 0.036.
- **Nu am agregat repetările ca observații independente:** toate comparațiile între observabile folosesc n = 5 seeds.

## 4. Pragurile: A (exact original), B (recalculat), C (½ / ½) (`finite_shot_threshold_analysis.csv`)

**p = 0.05, medie pe seeds:**

| | shots | recall A | FP A | F1 A | recall B | FP B | F1 B | F1 C | F1 exact (oficial) | F1 exact C |
|---|---|---|---|---|---|---|---|---|---|---|
| Z₀ | 1 024 | 0.639 | **363** | 0.386 | 0.550 | 20.1 | **0.627** | 0.606 | 0.735 | 0.712 |
| Z₀ | 10 000 | 0.662 | 19.9 | 0.711 | 0.671 | 17.5 | 0.726 | 0.701 | | |
| Z₀ | 100 000 | 0.668 | 17.2 | 0.726 | 0.680 | 17.4 | 0.732 | 0.708 | | |
| Z₀Z₁ | 1 024 | 0.609 | **3 427** | 0.046 | 0.251 | 52.3 | **0.290** | 0.265 | 0.740 | 0.718 |
| Z₀Z₁ | 10 000 | 0.664 | 31.4 | 0.674 | 0.641 | 17.5 | 0.705 | 0.685 | | |
| Z₀Z₁ | 100 000 | 0.681 | 19.9 | 0.725 | 0.685 | 18.0 | 0.734 | 0.712 | | |

**Pragul exact original (A) nu se poate folosi pe scorurile cu shots.** Zgomotul de eșantionare lărgește distribuția legitimelor peste prag:
- Z₀Z₁ la 1 024 shots ajunge la 3 427 de alarme false (F1 0.05);
- recall-ul A, aparent mare (0.61), este plătit cu aceste alarme false.

**Recalibrarea (B) îmbunătățește mult F1 față de A** (Z₀: 0.39 → 0.63; Z₀Z₁: 0.05 → 0.29), dar **nu îmbunătățește recall-ul**: B alege un prag mai sus, cu recall mai mic și mult mai puține FP.

**Pierderea rămasă vine din informație, nu din prag.** B este cel mai bun prag posibil pe aceleași date (optimist), deci niciun prag nu recuperează diferența:

| F1 la 1 024 shots | exact | cu shots (B) | pierdere |
|---|---|---|---|
| Z₀ | 0.735 | 0.627 | 0.11 |
| Z₀Z₁ | 0.740 | 0.290 | 0.45 |

**Pragul ales onest (C)** stă cu ~0.02–0.03 sub B, la fel ca pe scorurile exacte (0.712 față de 0.735 pentru Z₀). Concluziile nu se schimbă.

**De ce scade recall-ul:** la N mic, scăderea recall-ului la pragul recalculat vine din pierderea informației cauzate de shots. Cu pragul vechi, recall-ul s-ar menține doar pe hârtie, cu sute sau mii de alarme false.

## 5. Separabilitate și shots necesare (`finite_shot_separability.csv`)

**p = 0.05, medii pe seeds, scoruri exacte:**

| | diferență medie fraudă − legitim | std legitime | lățimea intervalului de scor | contracție \|⟨O⟩\| | d′ | OVL | SNR la 1 024 | N_egal = 0.25/σ²_legit | N₉₅ | N_(exact − 0.01) |
|---|---|---|---|---|---|---|---|---|---|---|
| Z₀ | **0.056** | 0.0030 | 0.113 | 0.109 | 2.32 | 0.15 | **3.6** | 32 000 | **~3 200** | **~6 500** |
| Z₁ | 0.042 | 0.0027 | 0.077 | 0.078 | 2.30 | 0.16 | 2.7 | 39 000 | ~7 200 | ~30 000 |
| Z₂ | 0.044 | 0.0036 | 0.088 | 0.114 | 2.18 | 0.18 | 2.9 | 36 000 | ~10 000 | ~32 000 |
| Z₃ | 0.031 | 0.0022 | 0.054 | 0.064 | 2.32 | 0.17 | 2.0 | 57 000 | ~11 000 | ~31 000 |
| Z₀Z₁ | 0.031 | 0.0017 | 0.059 | 0.047 | 2.28 | 0.16 | 2.0 | 103 000 | **~18 000** | **~70 000** |

*Legendă:* d′ = diferență / std combinat al claselor; OVL = suprapunerea histogramelor; SNR = diferență / eroarea standard shot; N_egal = shots la care eroarea shot egalează dispersia legitimelor; N₉₅ și N_(exact − 0.01) = shots la care AUPRC atinge 95% din exact, respectiv exact − 0.01.

**De ce Z₀Z₁ are AUPRC exactă mare și AUPRC mică la 1 024 shots:**

1. **Separabilitatea relativă este aceeași** pentru toate observabilele: d′ ≈ 2.2–2.4, OVL ≈ 0.15–0.18. În valori exacte, clasamentul nu depinde de scară, iar Z₀Z₁ ordonează puțin mai bine fraudele din vârf (AUPRC 0.678).
2. **Zgomotul de shots are scară fixă** (≈ 0.5/√N = 0.016 la 1 024), indiferent de observabilă. La p = 0.05, scorul Z₀Z₁ este comprimat mai mult: contracție 0.047 față de 0.109 pentru Z₀. Toate cele 56 962 de scoruri stau într-o fereastră de ~0.06, iar dispersia legitimelor (0.0017) este de ~9 ori mai mică decât eroarea shot.
3. **Coada populației legitime decide AUPRC.** Exemplu, seed 0, cuantile exacte (graficul fs4):

   | seed 0, scoruri exacte | Z₀ | Z₀Z₁ |
   |---|---|---|
   | mediana legitimelor | 0.4695 | 0.4844 |
   | mediana fraudelor | 0.5427 | 0.5146 |
   | distanța, în erori standard shot (0.0156) | **4.7** | **1.9** |

   Cu 56 863 de legitime, în medie **~1 700 (3.0%)** depășesc mediana fraudelor la Z₀Z₁. La Z₀ sunt doar **~12** (0.02%), toate din coada distribuției exacte a legitimelor. Valorile sunt calculate exact, Σᵢ P(ŝᵢ > mediana), și confirmate prin eșantionare (1 768, respectiv 12, din 20 de realizări).
4. **Concluzie:** media diferenței dintre clase nu ajunge ca explicație. Contează câte erori standard separă vârful fraudelor de coada unei clase de ~575 de ori mai mare. Pentru Z₀Z₁ separarea este de ~2σ, pentru Z₀ de ~5σ.

**Shots necesare:**

- **La p = 0.05:**
  - Z₀ ajunge la 95% din AUPRC exactă la ~3 200 de shots;
  - Z₀Z₁ are nevoie de ~18 000 de shots pentru 95% din exact și de ~70 000 pentru a fi la 0.01 de exact.
- **La p ≤ 0.01:** 1 024 shots ajung pentru toate observabilele.

## 6. Comparația pereche Z₀Z₁ − Z₀ (`finite_shot_paired_comparison.csv`, graficele fs3, fs5)

Δ = diferența mediilor MC per seed; tabelul arată media pe 5 seeds și numărul de seeds cu Δ > 0. P_MC = probabilitatea ca o realizare Z₀Z₁ să depășească o realizare Z₀, în același seed.

| p = 0.05 | Δ AUPRC | seeds | P_MC | Δ F1 (B) | seeds | Δ recall (B) | seeds |
|---|---|---|---|---|---|---|---|
| 1 024 | −0.322 | 0/5 | 0.001 | −0.337 | 0/5 | −0.299 | 0/5 |
| 4 096 | −0.051 | 1/5 | 0.24 | −0.073 | 1/5 | −0.086 | 0/5 |
| 10 000 | +0.012 | 3/5 | 0.56 | −0.021 | 2/5 | −0.029 | 2/5 |
| 32 768 | **+0.040** | **4/5** | 0.73 | −0.004 | 2/5 | −0.008 | 2/5 |
| 100 000 | +0.051 | 4/5 | 0.79 | +0.002 | 3/5 | +0.006 | 3/5 |
| exact | +0.056 | 4/5 | — | +0.006 | 4/5 | +0.018 | 4/5 |

| p = 0.01 | Δ AUPRC | seeds | P_MC | Δ F1 (B) | seeds |
|---|---|---|---|---|---|
| 1 024 | +0.021 | 4/5 | 0.74 | +0.008 | 4/5 |
| 10 000 | +0.023 | 4/5 | 0.80 | +0.011 | 4/5 |
| exact | +0.023 | 4/5 | — | +0.012 | 4/5 |

**Comparația cu celelalte observabile:** la p = 0.05 și 1 024 shots, toate pierd față de Z₀ în 5/5 seeds. Z₁ îl depășește pe Z₀ la AUPRC de la ~32 768 de shots (4/5 seeds).

## 7. Cost pe bugete de shots (`finite_shot_cost_ledger.json`)

**Ce rămâne neschimbat:** execuțiile de circuit, aceleași ca în ledger-ul oficial al Exp. 11, care nu a fost modificat. Z₀Z₁ folosește aceleași shots per execuție ca Z₀.

**Shots per seed:**

| | N | antrenare (parameter shift, Tracker) | validare + inferență | total / seed | total 5 seeds | × față de 1 024 |
|---|---|---|---|---|---|---|
| Z₀ | 1 024 | 7.4 × 10⁸ | 2.0 × 10⁹ | 2.7 × 10⁹ | 1.36 × 10¹⁰ | 1 |
| Z₀ | 4 096 | 3.0 × 10⁹ | 7.9 × 10⁹ | 1.1 × 10¹⁰ | 5.45 × 10¹⁰ | 4 |
| Z₀Z₁ | 1 024 | 9.6 × 10⁸ | 2.0 × 10⁹ | 2.9 × 10⁹ | 1.47 × 10¹⁰ | 1 |
| Z₀Z₁ | 10 000 | 9.4 × 10⁹ | 1.9 × 10¹⁰ | 2.9 × 10¹⁰ | 1.44 × 10¹¹ | 9.8 |
| Z₀Z₁ | 32 768 | 3.1 × 10¹⁰ | 6.3 × 10¹⁰ | 9.4 × 10¹⁰ | 4.71 × 10¹¹ | 32 |
| Z₀Z₁ | 100 000 | 9.4 × 10¹⁰ | 1.9 × 10¹¹ | 2.9 × 10¹¹ | 1.44 × 10¹² | 98 |

**Inferența:** costă N shots per tranzacție, deci 5.8 × 10⁷ (1 024) până la 5.7 × 10⁹ (100 000) pentru toată validarea, la un nivel de zgomot.

**Comparația relevantă:** Z₀Z₁ la 32 768 de shots (primul buget la care îl depășește consistent pe Z₀ la AUPRC) costă **~35× mai multe shots totale** decât Z₀ la 1 024 (4.71 × 10¹¹ față de 1.36 × 10¹⁰). La același buget per execuție, Z₀Z₁ costă cu ~8% mai mult decât Z₀ (gradient cu mai mulți parametri în con) și rămâne în urmă până la ~10 000 de shots.

**Ce nu am simulat:** modelele au fost antrenate fără zgomot, cu valori exacte (backprop). Costul de antrenare cu shots este un scenariu de ledger; efectul shots asupra antrenării nu este simulat.

## 8. Protocolul oficial vs analiza finite-shot

| | Protocolul oficial Task 2 (valori exacte) | Analiza finite-shot (suplimentară) |
|---|---|---|
| AUPRC p = 0.01 | Z₀Z₁ 0.677 > Z₀ 0.654 (4/5 seeds) | La fel la 1 024 shots (+0.021, 4/5) |
| AUPRC p = 0.05 | Z₀Z₁ 0.678 > Z₀ 0.622 (4/5 seeds) | Z₀ 0.519 ≫ Z₀Z₁ 0.197 la 1 024 (5/5); Z₀Z₁ conduce consistent abia de la ~32 768 |
| F1 / recall p = 0.05 | Z₀Z₁ puțin mai bun (+0.006 F1, +1.8 TP) | Z₀Z₁ nu este mai bun consistent la niciun buget testat |
| Cost | Execuții nominale + Tracker | Shots × N; Z₀Z₁ are nevoie de ~32× bugetul de referință |

**Z₀Z₁ nu a eșuat în protocolul oficial:** acolo câștigă. Avantajul lui exact este însă **fragil la eșantionare finită** la p = 0.05: dispare sub ~10 000 de shots și se inversează puternic la 1 024.

## 9. Răspunsuri

1. **Este corectă analiza inițială cu 1 024 shots?** Da. Formula binomială este exactă pentru toate cele 5 observabile, inclusiv paritatea Z₀Z₁. Implementarea nu are erori de semn, de limite, de tip, de independență sau de aliniere. Eșantionarea PennyLane directă o confirmă statistic (60 de cazuri). Două lipsuri, acum corectate: doar 10 repetări și lipsa pragului A.
2. **Sunt confirmate 0.188 (Z₀Z₁) și 0.520 (Z₀)?** Da, în limita erorii MC: cu 50 de repetări rezultă Z₀Z₁ **0.197** ± 0.115 între seeds (std MC 0.036) și Z₀ **0.519** ± 0.041 (std MC 0.028).
3. **Îmbunătățește recalibrarea pragului recall-ul sau F1?** F1, da, mult față de pragul exact original (Z₀Z₁: 0.05 → 0.29; Z₀: 0.39 → 0.63), prin eliminarea a sute sau mii de alarme false. Recall-ul, nu: pragul recalculat are recall mai mic decât pragul vechi, iar acela era umflat de FP. Niciun prag nu recuperează F1-ul exact; pierderea este de informație.
4. **Câți shots necesită Z₀Z₁ ca să se apropie de performanța exactă la p = 0.05?** ~18 000 pentru 95% din AUPRC exactă și ~70 000 pentru a fi la 0.01 de ea. Prin comparație, Z₀ are nevoie de ~3 200, respectiv ~6 500. La p ≤ 0.01: 1 024 ajung pentru toate observabilele.
5. **De la ce buget depășește Z₀Z₁ consistent pe Z₀?** La AUPRC, de la ~32 768 de shots (+0.040, 4/5 seeds, P_MC 0.73); la 10 000 doar 3/5 seeds și +0.012. La F1 și recall, nu consistent nici la 100 000 (3/5 seeds, +0.002 F1). La p = 0.01, deja de la 1 024.
6. **Este realist acest buget?** Nu, în raport cu ledger-ul nominal. 32 768 de shots per execuție înseamnă 32× bugetul de referință: 4.7 × 10¹¹ shots totale pentru 5 seeds, față de 1.4 × 10¹⁰ pentru Z₀ la 1 024. Inferența ar costa 32 768 de shots per tranzacție.
7. **Ce model recomand strict pentru evaluarea oficială exactă?** **Z₀Z₁.** Este alegerea regulii fixate înainte de rulare: AUPRC la p = 0.05 de 0.678 față de 0.622, iar la p = 0.01 de 0.677 față de 0.654, ambele în 4/5 seeds. Câștigul de recall este mic (+1.8 fraude, +1.8 FP).
8. **Ce model recomand la 1 024 shots?** **Z₀.** La p = 0.05 diferența este de 0.32 AUPRC în favoarea lui Z₀ (5/5 seeds). La p = 0.01, Z₀Z₁ ar fi cu 0.02 mai bun, dar pierderea la p = 0.05 este de ~15 ori mai mare decât acest câștig. Z₀ are și cel mai ieftin gradient.
9. **Merită păstrat Z₀Z₁ ca soluție principală?** Recomandarea mea: **doar ca rezultat experimental documentat**, cu ambele cifre (exact și 1 024 shots); soluția principală rămâne **Z₀**. Motivul: avantajul lui Z₀Z₁ există doar cu valori exacte sau cu un buget de shots de ≥ ~30× referința, iar într-un task despre bugetul cuantic acesta este argumentul decisiv. Dacă echipa vrea să optimizeze strict tabelul oficial, Z₀Z₁ e justificat de regula noastră, dar cu limitarea spusă explicit în prezentare. Decizia vă aparține.

## 10. Limitări

- Pragurile B sunt alese și evaluate pe același validation, deci sunt optimiste; C le estimează onest, pe jumătăți de ~49–50 de fraude.
- Modelul de shots presupune shots i.i.d. și **nu include erori de readout** sau drift; pe hardware real, efectele ar fi cel puțin la fel de mari.
- Antrenarea nu a fost refăcută cu shots; ledger-ul de antrenare cu shots este un scenariu de cost.
- 99 de fraude și 5 seeds nu dau semnificație statistică formală; P_MC descrie doar variația măsurării, nu pe cea a modelelor.
- Interpolarea N₉₅ / N_(exact − 0.01) este liniară în log N între punctele grilei.

## Fișiere

- **Notebook:** `notebooks/exp11_finite_shot_verification.ipynb`.
- **Verificare:** `finite_shot_verification.csv` (PennyLane vs binomial vs exact).
- **Metrici:** `finite_shot_metrics.csv` (per observabilă × seed × p × N × metrică: exact, medie / std / min / max MC) și `finite_shot_summary.csv` (variație între seeds vs variație MC).
- **Comparații și praguri:** `finite_shot_paired_comparison.csv`, `finite_shot_threshold_analysis.csv`.
- **Separabilitate și cost:** `finite_shot_separability.csv`, `finite_shot_cost_ledger.json`.
- **Repetări brute:** `finite_shot_reps/<o>_p<p>_N<N>.npz` (5 seeds × 50 de repetări × 27 de metrici; fără scoruri brute).
- **Grafice:** `plots/fs1_auprc_vs_shots.png`, `fs2_recall_vs_shots_p0.05.png`, `fs3_z0_vs_z0z1_p0.05.png`, `fs4_score_distributions_p0.05.png`, `fs5_delta_auprc_z0z1_vs_z0.png`.

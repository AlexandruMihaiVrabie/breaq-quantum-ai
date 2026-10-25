# Experimentul 11: optimizarea observabilei de readout pentru STAR ZZ

**Model de pornire:** STAR ZZ din Exp. 9: 4 qubiți, perechi ZZ (0,1), (0,2), (0,3), ansatz ×3, 32 de parametri, 15 CNOT, 73 locuri de zgomot. Ponderile Z₀ sunt refolosite din `result_exp9/weights/star`.
**Modificare unică:** observabila măsurată O ∈ {Z₀, Z₁, Z₂, Z₃, Z₀Z₁}. Scorul este s(x) = (1 − ⟨O⟩)/2, fără strat clasic.
**Date:** train (590 de exemple 1:1 per seed) și validation (56 962 de tranzacții, 99 de fraude). **Testul nu a fost folosit.**
**Setup:** 5 seeds; p = 0 / 0.001 / 0.01 / 0.05; `default.mixed`, zgomot pe tot circuitul; prag max-F1 pe validation la același p (optimist, la fel pentru toate observabilele).
**Cod:** `notebooks/exp11_quantum_readout_optimization.ipynb`, `src/quantum_readout.py`. PennyLane 0.45.1, torch 2.14.1, float64.

## 0. Pe scurt

- **În protocolul oficial (valori exacte, matrici de densitate), Z₀Z₁ câștigă după regula fixată înainte de rulare.** La p = 0.05 are AUPRC **0.678 ± 0.028**, față de **0.622 ± 0.041** pentru Z₀: Δ = +0.056 ± 0.057, mai bun în 4/5 seeds. Recall-ul crește puțin: 69.5% față de 67.7%.
- **Ținta de recall 75–80% nu a fost atinsă.** Câștigul lui Z₀Z₁ înseamnă **+1.8 fraude detectate și +1.8 alarme false** în medie. La recall 75% fixat, Z₀ cere *mai puține* alarme false decât oricare alternativă.
- **Avantajul dispare cu 1024 shots** (analiză suplimentară, post hoc, §7). La p = 0.05, semnalul lui Z₀Z₁ este de aproape două ori mai comprimat decât al lui Z₀, iar cu 1024 shots per execuție AUPRC-ul lui scade la **0.19**; Z₀ păstrează **0.52**. Toate alternativele pierd față de Z₀ în 5/5 seeds. La p ≤ 0.01 shot-urile nu schimbă clasamentul.
- **Costul crește.** Parameter shift deplasează doar parametrii din conul cauzal: Z₀ are cel mai mic con, deci cel mai ieftin gradient (41 de execuții / exemplu, față de 49–53).
- **Recomandare:** păstrați **Z₀** ca readout al soluției finale și prezentați Z₀Z₁ ca rezultat al Exp. 11, adevărat pentru valori exacte, dar nu la bugetul de 1024 shots (§10).

## 1. Ce am verificat înainte de a compara (23 de verificări, toate trecute, `checks.csv`)

- **Splitul:** splitul oficial a fost reconstruit și comparat. SHA-256 (train + val + test) este `config.json → split_sha256`. Din test am verificat doar dimensiunile; features nu au fost încărcate.
- **Același circuit ca Exp. 9:** subseturile 1:1 și unghiurile (train + val) au același SHA-256 ca în Exp. 9. Circuitul STAR are exact ordinea de porți native salvată de Exp. 9.
- **Singura diferență este observabila:** cele 5 modele au porți identice (`qml.equal`, 20 de intrări); diferă doar măsurarea. Toate au 4 qubiți, 15 CNOT, 73 locuri de zgomot, 32 de parametri și 0 porți de rotire a bazei.
- **Observabilele, testate analitic:**
  - 16 stări de bază (±1) și produse de RY (cos θ, cos θ₀ cos θ₁);
  - ⟨O⟩ = Σ_b (−1)^paritate(b) P(b) pe circuitul STAR real, la p = 0 și 0.05 (eroare 2e-16).
- **Bucla de antrenare reproduce Exp. 9:** STAR + Z₀, seed 0, reantrenat, are același istoric, aceeași epocă, aceleași ponderi și aceeași ordine a mini-batch-urilor (eroare 0).
- **Condiții identice pentru toate modelele noi:** Z₁, Z₂, Z₃ și Z₀Z₁ au aceeași inițializare și aceeași ordine a mini-batch-urilor per seed ca Exp. 9 (hash comparat).
- **Z₀ reproduce Exp. 9 STAR exact** (AUPRC, AUROC, prag, F1, MAE, ρ, TP, FP; 5 seeds × 4 p; eroare 0). Cifrele istorice din cerință (AUPRC 0.622, recall 0.6768, TP ≈ 67) sunt confirmate.
- **Regresia logistică** pe aceleași unghiuri reproduce Exp. 4.
- **Zgomot:**
  - `default.mixed` la p = 0 dă aceleași scoruri ca `default.qubit` (≤ 7e-13);
  - toate cele 100 de seturi de scoruri sunt finite și în [0, 1];
  - inserarea manuală a canalelor = `noisy_qnode` oficial (≤ 4e-13).

## 2. Rezultatul principal: validation, medie ± std pe 5 seeds, prag max-F1

| p = 0.05 | AUPRC | AUROC | Spearman ρ | precision | recall | F1 | MCC | TP | FP |
|---|---|---|---|---|---|---|---|---|---|
| **Z₀** (Exp. 9) | 0.622 ± 0.041 | 0.947 ± 0.003 | 0.889 ± 0.066 | 0.804 ± 0.042 | 0.677 ± 0.038 | 0.735 ± 0.034 | 0.737 ± 0.034 | 67.0 ± 3.7 | 16.4 ± 4.0 |
| Z₁ | 0.657 ± 0.011 | 0.946 ± 0.007 | 0.947 ± 0.061 | 0.829 ± 0.038 | 0.661 ± 0.042 | 0.734 ± 0.016 | 0.739 ± 0.013 | 65.4 ± 4.2 | 13.8 ± 4.5 |
| Z₂ | 0.574 ± 0.072 | 0.936 ± 0.023 | 0.953 ± 0.057 | 0.741 ± 0.057 | 0.576 ± 0.076 | 0.647 ± 0.065 | 0.652 ± 0.064 | 57.0 ± 7.5 | 19.8 ± 4.5 |
| Z₃ | 0.623 ± 0.048 | 0.945 ± 0.014 | 0.884 ± 0.207 | 0.722 ± 0.084 | 0.691 ± 0.072 | 0.705 ± 0.070 | 0.705 ± 0.070 | 68.4 ± 7.1 | 26.8 ± 10.3 |
| **Z₀Z₁** | **0.678 ± 0.028** | 0.941 ± 0.012 | 0.855 ± 0.159 | 0.792 ± 0.043 | **0.695 ± 0.019** | 0.740 ± 0.028 | 0.741 ± 0.028 | 68.8 ± 1.9 | 18.2 ± 4.7 |
| LR (clasic) | 0.663 ± 0.025 | 0.965 ± 0.001 | — | 0.802 ± 0.051 | 0.715 ± 0.009 | 0.755 ± 0.023 | 0.757 ± 0.024 | 70.8 ± 0.8 | 17.8 ± 5.7 |

| AUPRC | p = 0 | p = 0.001 | p = 0.01 | p = 0.05 |
|---|---|---|---|---|
| Z₀ | 0.662 ± 0.023 | 0.661 ± 0.023 | 0.654 ± 0.023 | 0.622 ± 0.041 |
| Z₁ | 0.669 ± 0.007 | 0.669 ± 0.007 | 0.668 ± 0.005 | 0.657 ± 0.011 |
| Z₂ | 0.643 ± 0.029 | 0.643 ± 0.029 | 0.634 ± 0.035 | 0.574 ± 0.072 |
| Z₃ | **0.680 ± 0.012** | **0.680 ± 0.012** | 0.677 ± 0.011 | 0.623 ± 0.048 |
| Z₀Z₁ | 0.676 ± 0.018 | 0.675 ± 0.018 | **0.677 ± 0.020** | **0.678 ± 0.028** |

Diferențe pereche față de Z₀, cu numărul de seeds în care observabila nouă e mai bună (`paired_comparison.csv`):

| | Δ AUPRC p = 0 | Δ AUPRC p = 0.01 | Δ AUPRC p = 0.05 | Δ recall p = 0.05 | Δ TP / Δ FP p = 0.05 |
|---|---|---|---|---|---|
| Z₁ | +0.007 ± 0.028 (3/5) | +0.015 ± 0.028 (4/5) | +0.035 ± 0.050 (4/5) | −0.016 (1/5) | −1.6 / −2.6 |
| Z₂ | −0.019 ± 0.026 (1/5) | −0.020 ± 0.034 (1/5) | −0.048 ± 0.085 (2/5) | −0.101 (1/5) | −10.0 / +3.4 |
| Z₃ | +0.018 ± 0.016 (4/5) | +0.023 ± 0.015 (5/5) | +0.001 ± 0.039 (2/5) | +0.014 (3/5) | +1.4 / +10.4 |
| Z₀Z₁ | +0.014 ± 0.023 (3/5) | +0.023 ± 0.027 (4/5) | **+0.056 ± 0.057 (4/5)** | +0.018 (4/5) | +1.8 / +1.8 |

Regula de alegere a fost fixată înainte de rulare (§13 din notebook, `decision.csv`):
- **Z₀Z₁** trece ambele criterii (calitate: AUPRC mai bun în 4/5 seeds; recall în toleranța de 0.02, cu ΔFP ≤ ΔTP).
- **Z₁** trece doar criteriul de calitate. Recall-ul îi scade (1/5 seeds), deci nu câștigă pe recall.
- Între candidate, Z₀Z₁ are AUPRC medie maximă, deci regula îl alege.

## 3. Recall: țintă 75–80% (analiză secundară, exploratorie)

La pragul oficial max-F1, la p = 0.05, cel mai mare recall îl are Z₀Z₁: **69.5%**. Asta înseamnă 68.8 din 99 de fraude, față de 67.0 pentru Z₀, adică **+1.8 fraude** și **+1.8 alarme false** în medie. Ținta de 75–80% **nu este atinsă de nicio observabilă**.

Câte alarme false costă să forțezi recall-ul (`recall_precision_tradeoff.csv`, graficul 8; p = 0.05, medie pe seeds):

| recall țintă | Z₀ | Z₁ | Z₂ | Z₃ | Z₀Z₁ |
|---|---|---|---|---|---|
| 70% (70 fraude) | 34 FP | 52 FP | 149 FP | 42 FP | 38 FP |
| 75% (75 fraude) | **220 FP** | 538 FP | 1 299 FP | 854 FP | 509 FP |
| 80% (80 fraude) | **947 FP** | 1 625 FP | 3 390 FP | 2 803 FP | 1 574 FP |
| 85% (85 fraude) | **2 801 FP** | 3 967 FP | 7 483 FP | 6 049 FP | 4 092 FP |

Peste ~70%, curbele recall–FP ajung pe un platou: ultimele 20–30 de fraude primesc scoruri amestecate cu cele ale tranzacțiilor legitime, pentru *toate* observabilele. Peste 75%, Z₀ e cea mai ieftină (cu deviații standard mari între seeds). Mutarea pragului nu schimbă AUPRC și nu este o îmbunătățire cuantică sau anti-zgomot.

## 4. Robustețe vs performanță de bază

| | AUPRC p = 0 | AUPRC_drop (0 → 0.05) | Δ drop vs Z₀ (seeds mai bune) | Spearman ρ p = 0.05 | contracție \|⟨O⟩\| p = 0.05 |
|---|---|---|---|---|---|
| Z₀ | 0.662 | 0.040 ± 0.044 | — | 0.889 ± 0.066 | 0.109 |
| Z₁ | 0.669 | 0.012 ± 0.013 | −0.027 ± 0.052 (3/5) | 0.947 ± 0.061 | 0.078 |
| Z₂ | 0.643 | 0.068 ± 0.052 | +0.029 ± 0.071 (2/5) | 0.953 ± 0.057 | 0.114 |
| Z₃ | 0.680 | 0.057 ± 0.047 | +0.017 ± 0.027 (2/5) | 0.884 ± 0.207 | 0.064 |
| Z₀Z₁ | 0.676 | −0.003 ± 0.013 | −0.043 ± 0.050 (3/5) | 0.855 ± 0.159 | 0.047 |

- **Z₀Z₁: avantajul are două surse.** Performanța de bază e puțin mai bună (+0.014 la p = 0, 3/5 seeds), iar degradarea e mai mică (−0.043, 3/5 seeds). Degradarea mare a lui Z₀ vine mai ales din 2 seeds (seed 2: 0.093, seed 4: 0.074). Cu 3/5 seeds, robustețea mai mare **nu este dovedită**, doar sugerată.
- **Z₃ are cea mai bună performanță de bază** (AUPRC 0.680, recall 71.5% la p = 0, comparabil cu LR) și e mai bun decât Z₀ la p = 0.01 în 5/5 seeds. Se degradează însă cel mai mult până la p = 0.05: la seed 3, ρ scade la 0.52 și FP sare de la 16 la 43. Exemplu clar de „mai bun fără zgomot ≠ mai robust”.
- **Z₁ este cel mai stabil între seeds:** std AUPRC 0.011 la p = 0.05, față de 0.041 pentru Z₀. Degradarea e mică, dar recall-ul scade puțin.
- **Spearman ρ nu urmează AUPRC.** Z₀Z₁ are ρ mai mic decât Z₀, dar AUPRC mai mare: zgomotul amestecă ordinea în masa de tranzacții legitime cu scor mic, care nu contează pentru AUPRC, dar păstrează vârful clasamentului.

## 5. Conul cauzal: ce ne spune despre circuit (`causal_cone_analysis.csv`, graficele 10a/10b)

| | porți în con | CNOT în con | locuri de zgomot în con | parametri în con (pe fire) | parametri cu gradient ≠ 0 | per strat (gradient) |
|---|---|---|---|---|---|---|
| Z₀ | 43 / 58 | 12 / 15 | **52 / 73** | 20 | 19 | 8, 6, 4, 1 |
| Z₁ | 49 / 58 | 14 / 15 | 60 / 73 | 24 | 22 | 8, 8, 5, 1 |
| Z₂ | 52 / 58 | 15 / 15 | 64 / 73 | 26 | 23 | 8, 8, 6, 1 |
| Z₃ | 52 / 58 | 15 / 15 | 64 / 73 | 26 | 22 | 8, 8, 5, 1 |
| Z₀Z₁ | 51 / 58 | 14 / 15 | 63 / 73 | 26 | 24 | 8, 8, 6, 2 |

### Cum am calculat conul

Am calculat conul invers pe tape-ul real (decompus în porți native) și l-am verificat numeric:
- parametrii cu gradient 0 nu schimbă ⟨O⟩ nici la o perturbare de 1.234 rad;
- zgomotul pus **doar în afara conului** dă exact rezultatul fără zgomot, iar zgomotul pus **doar în con** dă exact rezultatul cu zgomot peste tot (5 puncte, p = 0.05, eroare ≤ 4e-13).

Explicația este imaginea Heisenberg: un canal depolarizant pe un fir unde operatorul evoluat este identitatea nu are efect. În toate evaluările, zgomotul a fost totuși aplicat întregului circuit.

### Ce determină conul

- **Conul îl dă direcția scării de CNOT din ansatz (0→1→2→3), nu topologia STAR.**
  - Mergând înapoi de la Z₀, din fiecare scară doar CNOT(0,1) atinge qubitul 0, deci conul crește cu un qubit per strat.
  - De la Z₃ (capătul scării), ultima scară adaugă deja q₂, q₁, q₀: conul acoperă toți qubiții după un singur strat.
  - De aceea Z₀ are cel mai mic con (52 de canale de zgomot efective), iar Z₂ și Z₃ cel mai mare (64).
- **Feature map-ul STAR (17 porți) este integral în conul fiecărei observabile.** Conul acoperă toate firele înainte să ajungă la el. Rolul de centru al qubitului 0 în STAR nu restrânge deci informația disponibilă pentru readout-urile periferice.
- **Structura Pauli restrânge conul și mai mult.** RZ-ul final pe qubitul măsurat comută cu Z, deci are gradient 0 mereu. Câteva RZ dinaintea scării de CNOT pe controale au și ele gradient 0 (Z₂, Z₃). Toate cele 4 unghiuri de intrare influențează fiecare observabilă.

### Ce nu explică conul

- **Ipoteza „con mai mic ⇒ mai robust” nu este susținută.** Z₀ are cele mai puține canale de zgomot în con, dar o degradare AUPRC mai mare decât Z₁ și Z₀Z₁.
- **Conul explică însă contracția semnalului.** Mai multe canale în con și o observabilă de greutate 2 dau o contracție mai puternică: |⟨O⟩| scade la 4.7% pentru Z₀Z₁, 6.4% pentru Z₃ și 10.9% pentru Z₀ (Z₂, cu 11.4%, este excepția).
- O contracție uniformă nu schimbă clasamentul la valori exacte. Pe hardware însă, devine decisivă (§7).

### Interpretarea lui Z₀Z₁

- **Ce măsoară Z₀Z₁:** o *corelație de paritate în baza computațională*:
  ⟨Z₀Z₁⟩ = Σ_b (−1)^(b₀ ⊕ b₁) P(b) = P(b₀ = b₁) − P(b₀ ≠ b₁).
  Scorul s = (1 − ⟨Z₀Z₁⟩)/2 = P(b₀ ≠ b₁) este probabilitatea parității impare a primilor doi qubiți.
- **Nu este o măsură a entanglementului.** Starea Bell și amestecul clasic ½(|00⟩⟨00| + |11⟩⟨11|) au amândouă ⟨Z₀Z₁⟩ = 1 și ⟨Z₀⟩ = ⟨Z₁⟩ = 0. Doar Bell este entanglată: criteriul PPT dă valoarea proprie minimă −0.5, față de 0 pentru amestec (verificat în notebook).
- **Nu putem afirma un avantaj cuantic al măsurării pe doi qubiți.** Z₀Z₁ este doar o altă funcție a acelorași rezultate de măsurare.

## 6. Cost cuantic (`cost_ledger.json`, `cost_ledger.csv`)

### Ce rămâne identic

- **Structura per execuție:** 58 de porți, 15 CNOT, 73 locuri de zgomot, nicio rotire a bazei, pentru toate observabilele.
- **Shots:** Z₀ și Z₁ se citesc din aceleași bitstring-uri, deci **shots pentru Z₀Z₁ = shots pentru Z₀**, nu dublu.
- **Pași Adam:** 30 × 19 = 570 per seed.

### Ce se schimbă: execuțiile parameter shift

PennyLane deplasează **doar parametrii din conul pe fire**. Am măsurat cu `qml.Tracker` și am verificat egalitatea cu conul:

| | execuții / exemplu (Tracker) | antrenare / seed | total antrenare + validare / seed | vs Z₀ | shots 5 seeds (×1024) |
|---|---|---|---|---|---|
| Z₀ | **41** | 725 700 | 2 662 408 | — | 1.36 × 10¹⁰ |
| Z₁ | 49 | 867 300 | 2 804 008 | +5.3% | 1.44 × 10¹⁰ |
| Z₂, Z₃, Z₀Z₁ | 53 | 938 100 | 2 874 808 | +8.0% | 1.47 × 10¹⁰ |

- **Totalul este dominat de validarea după fiecare epocă:** 1.71 M execuții / seed, identice pentru toate.
- **Pe componenta de gradient, Z₀Z₁ costă +29% față de Z₀.** Formula nominală 1 + 2P = 65 supraestimează costul pentru toate observabilele.
- **Potențial, neimplementat și neverificat:** dacă s-ar deplasa doar parametrii cu gradient ≠ 0, ar rămâne 39 / 45 / 47 / 45 / 49 execuții / exemplu. Nu atribuim această reducere niciunui model.

### Timpul pe simulator

- **Antrenarea:** ~16 s / seed pentru toate observabilele, adică 5.3 min pentru cele 20 de antrenări noi.
- **Evaluarea cu zgomot** a celor 5 observabile: 21.4 min.

## 7. Sensibilitatea la shots (analiză suplimentară, adăugată post hoc)

**De ce am adăugat-o:** după ce am văzut contracția semnalului la p = 0.05. Nu schimbă regula de alegere și se raportează separat.

**Metoda:** pentru o observabilă diagonală, s este exact probabilitatea rezultatului −1. Deci estimatorul cu N shots este ŝ = Binomial(N, s)/N. L-am eșantionat pe scorurile exacte salvate, cu 10 repetări per (observabilă, seed, p), `default_rng(42)`, fără erori de readout (`shot_noise_sensitivity.csv`, graficul 11).

| AUPRC, p = 0.05 | exact | 1024 shots | 10 000 shots | 100 000 shots | diferența de scor fraudă − legitim |
|---|---|---|---|---|---|
| Z₀ | 0.622 | **0.520** | 0.613 | 0.621 | **0.056** |
| Z₁ | 0.657 | 0.387 | 0.632 | 0.653 | 0.042 |
| Z₂ | 0.574 | 0.303 | 0.547 | 0.571 | 0.044 |
| Z₃ | 0.623 | 0.168 | 0.587 | 0.620 | 0.031 |
| Z₀Z₁ | **0.678** | 0.188 | 0.630 | **0.672** | 0.031 |

- **La p = 0.05 cu 1024 shots,** toate alternativele sunt sub Z₀ în **5/5 seeds**. Z₀Z₁ ajunge la recall 25% și 57 FP, față de 55% și 19 FP pentru Z₀.
- **Explicația:** eroarea standard per tranzacție (~0.016 în scor) este comparabilă cu diferența fraudă − legitim a lui Z₀Z₁ (0.031), dar de 3.5 ori mai mică decât a lui Z₀ (0.056).
- **La ce buget revine avantajul:** la 10⁴ shots, Z₀Z₁ ≈ Z₀ (+0.017, 3/5 seeds); abia la 10⁵ shots revine avantajul exact (+0.051, 4/5 seeds).
- **La p ≤ 0.01 shot-urile practic nu contează:** diferența fraudă − legitim este 0.29–0.53. Cu 1024 shots, Z₀Z₁ rămâne mai bun decât Z₀ cu +0.022 (4/5 seeds), iar Z₃ cu +0.024 (5/5 seeds).

## 8. Răspunsuri explicite

1. **AUPRC maximă la p = 0.05:** Z₀Z₁, 0.678 ± 0.028 (valori exacte). Cu 1024 shots: Z₀, 0.520.
2. **Recall maxim la pragul oficial (p = 0.05):** Z₀Z₁, 69.5% ± 1.9%. Z₃ urmează cu 69.1%, dar cu FP 26.8. La p = 0, Z₃ are cel mai mare recall: 71.5%.
3. **Fraude suplimentare:** +1.8 în medie față de Z₀ (68.8 față de 67.0 din 99; mai multe în 4/5 seeds). Nu este un salt spre 75–80%.
4. **False positives suplimentare:** +1.8 în medie (18.2 față de 16.4); mai puține decât Z₀ doar în 1/5 seeds. Pentru Z₃: +10.4.
5. **Creștere reală a robusteții:** nu este demonstrată. Z₀Z₁ și Z₁ au o degradare AUPRC medie mai mică, dar doar în 3/5 seeds, cu deviații standard mai mari decât diferența. Spearman ρ este *mai mic* pentru Z₀Z₁. Cu shots finite, Z₀ este cel mai robust la p = 0.05.
6. **Conul cauzal:** este determinat de scara de CNOT din ansatz, nu de STAR. Z₀ are cel mai mic con (52 / 73 de canale efective, 20 de parametri), Z₂ și Z₃ cel mai mare (64 / 73, 26 de parametri). Conul nu prezice degradarea AUPRC exactă, dar explică contracția semnalului și costul gradientului.
7. **Stabilitatea între seeds:** Z₁ (std AUPRC 0.011 la p = 0.05). Z₀Z₁ este al doilea (0.028), apoi Z₀ (0.041), Z₃ (0.048) și Z₂ (0.072).
8. **Costul cuantic:** per execuție, nu se schimbă. Pe gradient, crește cu +20% (Z₁) sau +29% (Z₂, Z₃, Z₀Z₁) față de Z₀. Pe total (antrenare + validare), crește cu +5.3% sau +8.0%. Shots per execuție: identice.
9. **Rezultate negative:**
   - Z₂ este mai slab decât Z₀ la toate nivelurile (recall −10 puncte procentuale la p = 0.05);
   - avantajul exact al lui Z₃ dispare la p = 0.05;
   - ținta de recall 75–80% nu este atinsă;
   - nicio observabilă nu depășește clar regresia logistică pe aceleași unghiuri (LR: AUPRC 0.663, recall 71.5%), cu excepția lui Z₀Z₁ la p = 0.05 *exact* (0.678 față de 0.663, diferență sub deviația standard);
   - avantajul lui Z₀Z₁ dispare la 1024 shots;
   - „con mai mic ⇒ mai robust” nu se confirmă.
10. **Ce merită păstrat:** vezi §10.

## 9. Ipoteze vs ce am măsurat

| Ipoteză inițială | Măsurat |
|---|---|
| H1: un alt readout citește informația mai bine | Parțial: Z₀Z₁ și Z₃ au AUPRC mai mare la p ≤ 0.01 (4/5, respectiv 5/5 seeds la p = 0.01), cu câștiguri mici (+0.02). |
| Recall 75–80% doar din observabilă | Nu: maxim 69.5% la pragul oficial. |
| Observabila schimbă robustețea | Neconcludent pe valori exacte (3/5 seeds). Negativ cu shots finite. |
| Con mai mic ⇒ zgomot mai mic | Structural da (52 față de 64 de canale), dar nu se vede în degradarea AUPRC exactă. |
| Readout-ul nu schimbă costul | Fals pentru gradient: parameter shift urmează conul (41 față de 53 de execuții / exemplu). |

## 10. Recomandare

- **Regula fixată înainte de rulare alege Z₀Z₁** pentru protocolul oficial (valori exacte, matrici de densitate).
- **Recomandarea mea: păstrați Z₀ ca readout al soluției finale**, din trei motive:
  1. câștigul de recall al lui Z₀Z₁ este de +1.8 fraude, plătit cu +1.8 FP;
  2. avantajul de AUPRC la p = 0.05 se pierde complet la bugetul de 1024 shots din cost ledger;
  3. Z₀ are cel mai ieftin gradient (41 față de 53 de execuții / exemplu).
- **În prezentare,** Z₀Z₁ merită arătat ca rezultat al Exp. 11: câștigă cu valori exacte, dar avantajul cere ≥ 10⁵ shots la p = 0.05.
- **Decizia finală este a echipei.** Dacă vreți să evaluăm strict ca juriul (valori exacte), Z₀Z₁ este justificat de regula noastră, dar cu limitarea de la §7 spusă explicit.

**Limitări:**
- precision / recall / F1 sunt optimiste (prag ales și evaluat pe același validation);
- 99 de fraude și 5 seeds nu dau semnificație statistică formală;
- analiza de shots ignoră erorile de readout și presupune shots independente per tranzacție;
- testul nu a fost folosit.

## Fișiere

- **Configurare și verificări:** `config.json`, `checks.csv`.
- **Structură și con cauzal:** `circuit_structure.csv`, `causal_cone_analysis.csv`, `causal_cone_parameters.csv`.
- **Antrenare:** `train_runs.jsonl`, `weights/<Z1|Z2|Z3|Z0Z1>/seed_<s>.pt` (Z₀ în `result_exp9/weights/star`).
- **Scoruri:** `scores/<o>_seed<s>_p<p>.npz` (scoruri validation + timp).
- **Metrici și comparații:** `metrics_per_seed.csv`, `summary_metrics.csv`, `paired_comparison.csv`, `paired_comparison_per_seed.csv`, `robustness_vs_baseline.csv`, `decision.csv`.
- **Recall, cost, shots:** `recall_precision_tradeoff.csv`, `cost_ledger.json`, `cost_ledger.csv`, `shot_noise_sensitivity.csv`, `shot_noise_paired_vs_z0.csv`.
- **Grafice:** `plots/1…11`. În `plots/` numerotarea urmează lista cerută: 1 AUPRC, 2 recall, 3 F1, 4 Spearman, 5–6 per seed, 7 PR, 8 recall–FP, 9 Δ pereche, 10a/10b con cauzal, 11 shots.

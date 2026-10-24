# Experimentul 8: fuziunea rotațiilor cuantice (rotation fusion)

**Model de bază:** Exp. 3 (ZZ ×1, ansatz ×3, 4 qubiți, 32 de parametri, readout ⟨Z₀⟩), cu ponderile salvate, **fără reantrenare**.
**Date:** doar validation (56 962 tranzacții, 99 fraude). Testul nu a fost folosit. 5 seeds, p = 0 / 0.001 / 0.01 / 0.05, `default.mixed`.
**Notebook:** `notebooks/exp8_rotation_fusion.ipynb`. **PennyLane 0.45.1**, torch 2.14.1, dtype implicit torch float64 (vezi §6).

## 1. Ce s-a fuzionat

Am parcurs circuitul Exp. 3 după decompoziția oficială în `NATIVE_GATES` și am căutat, pe fiecare qubit, secvențele maximale de porți de un qubit fără nicio poartă de doi qubiți între ele.

| Regiune | Secvență originală | Blocuri | După fuziune | Porți eliminate |
|---|---|---:|---|---:|
| feature map | `H → RZ(2xᵢ)` | 4 | `Rot(π, π/2, 2xᵢ)` | 4 |
| feature map (blocuri ZZ) | `RZ` izolat între două CNOT | 6 | neschimbat | 0 |
| ansatz | `RY(a) → RZ(b)` | 16 | `Rot(0, a, b)` | 16 |
| **Total** | | **26 secvențe** | **20 fuziuni** | **20** |

**Identități folosite** (convenția PennyLane: `Rot(φ, θ, ω) = RZ(ω)·RY(θ)·RZ(φ)`):
1. `RZ(b)·RY(a) = Rot(0, a, b)`: identitate **exactă**, fără fază globală. Aceleași ponderi, φ = 0 e o constantă, deci nu apar parametri antrenabili noi.
2. `H = i·RY(π/2)·RZ(π)`, de unde `RZ(θ)·H = i·Rot(π, π/2, θ)`. Diferența e doar faza globală `i`, care nu are efect fizic: stările și ⟨Z₀⟩ rămân aceleași.

**De ce nu am folosit transformările automate:**
- `merge_rotations` nu găsește nimic de unit, pentru că în circuit nu există rotații adiacente de același tip.
- `single_qubit_fusion` găsește aceleași 20 de blocuri, dar are trei probleme:
  - eșuează cu tensorii torch ai modelului;
  - rescrie și cele 6 RZ izolate ca `Rot`, ceea ce nu aduce niciun câștig;
  - adaugă o operație `GlobalPhase`.

Am folosit deci o fuziune manuală controlată. Ca verificare încrucișată, unitara ei coincide cu cea a `single_qubit_fusion` la 6e-17.

**Ce nu s-a schimbat:** cele 21 de CNOT, în aceeași ordine (verificat), feature map-ul, ansatz-ul, readout-ul, encoderul, ponderile și pragurile.

## 2. Echivalența fără zgomot (toleranță 1e-10, float64)

| Verificare | Comparații | Eroare max |
|---|---:|---:|
| Unitare per bloc `H→RZ ⇒ Rot` (validation + aleator [0, π]) | 4 000 | 5.0e-16 |
| Unitare per bloc `RY→RZ ⇒ Rot` | 16 000 | 0 (exact) |
| Unitara 16×16 a circuitului, până la fază globală | 1 000 | 1.1e-15 |
| ⟨Z₀⟩ pe toată validarea, 5 seeds | 284 810 | 2.7e-15 |
| Scor de fraudă pe toată validarea, 5 seeds | 284 810 | 1.3e-15 |
| ⟨Z₀⟩ pe 5 000 de unghiuri aleatoare × 5 seeds | 25 000 | 2.2e-15 |
| \|Δ AUPRC\| fără zgomot | 5 | 0 |

Faza globală observată este exact π/2 pentru blocurile H→RZ și 0 pentru RY→RZ, așa cum prezice teoria. **Circuitul fuzionat produce exact aceleași scoruri și predicții fără zgomot.**

## 3. Structura circuitelor

| | A: Exp. 3 | A: Exp. 8 | B: Exp. 3 | B: Exp. 8 | B′: Exp. 3 | B′: Exp. 8 |
|---|---:|---:|---:|---:|---:|---:|
| Porți native | 67 | **47** | 75 | 87 | 75 | 71 |
| Porți 1 qubit | 46 | 26 | 54 | 66 | 54 | 50 |
| CNOT | 21 | 21 | 21 | 21 | 21 | 21 |
| Adâncime | 31 | **26** | 33 | 36 | 33 | 32 |
| Locuri de zgomot | 88 | **68** | 96 | 108 | 96 | 92 |

Cele trei coloane-pereche:
- **A** este modelul oficial (`Rot` și H native).
- **B** descompune `Rot → RZ(φ)·RY(θ)·RZ(ω)` (inclusiv `RZ(0)`) și `H → RZ·RX·RZ`, pentru **ambele** circuite.
- **B′** e doar o numărătoare structurală, neevaluată sub zgomot: Scenariul B după eliminarea rotațiilor de unghi exact 0.

Execuțiile parameter-shift măsurate cu `qml.Tracker` sunt identice (41 / exemplu), iar valoarea nominală 1 + 2P = 65 rămâne aceeași.

## 4. Rezultate sub zgomot (validation, medie ± std pe 5 seeds, diferențe pereche)

### Scenariul A: modelul oficial (comparația principală)

| Metrică | Exp. 3 original | Exp. 8 fusion | Diferență (Exp. 8 − Exp. 3) | Seeds cu Exp. 8 mai bun |
|---|---:|---:|---:|---:|
| Porți native | 67 | 47 | −20 (−29.9%) | structural |
| CNOT | 21 | 21 | 0 | structural |
| Adâncime | 31 | 26 | −5 (−16.1%) | structural |
| Locuri de zgomot | 88 | 68 | −20 (−22.7%) | structural |
| AUPRC, p = 0 | 0.6274 ± 0.0280 | 0.6274 ± 0.0280 | 0 | identic |
| AUPRC, p = 0.01 | 0.6226 ± 0.0236 | 0.6236 ± 0.0247 | +0.0010 ± 0.0018 | 3 / 5 |
| **AUPRC, p = 0.05** | 0.5867 ± 0.0385 | **0.5961 ± 0.0387** | **+0.0094 ± 0.0068** | **5 / 5** |
| **Spearman ρ, p = 0.05** | 0.8751 ± 0.1102 | **0.9229 ± 0.0737** | **+0.0478 ± 0.0372** | **5 / 5** |
| MAE noisy–noiseless, p = 0.01 | 0.1425 ± 0.0282 | 0.1192 ± 0.0239 | −0.0233 ± 0.0043 (−16%) | 5 / 5 |
| **MAE noisy–noiseless, p = 0.05** | 0.2940 ± 0.0514 | **0.2807 ± 0.0491** | **−0.0134 ± 0.0045 (−4.5%)** | **5 / 5** |
| F1, p = 0.05, prag Exp. 3 | 0.6744 ± 0.0288 | 0.6213 ± 0.0691 | −0.0531 ± 0.0507 | 1 / 5 |
| F1, p = 0.05, prag reoptimizat (secundar) | 0.6800 | 0.6813 | +0.0012 ± 0.0071 | 3 / 5 |
| Timp simulare zgomot / seed (3 p) | 71.4 ± 4.5 s | 55.0 ± 4.4 s | −16.4 s (−23%) | 5 / 5 |

Δ AUPRC per seed la p = 0.05: +0.018, +0.004, +0.014, +0.009, +0.002.

### Scenariul B: Rot descompus (control)

| Metrică | Exp. 3 original | Exp. 8 fusion | Diferență | Seeds cu Exp. 8 mai bun |
|---|---:|---:|---:|---:|
| Locuri de zgomot | 96 | 108 | +12 (+12.5%) | structural |
| AUPRC, p = 0.05 | 0.5842 ± 0.0406 | 0.5743 ± 0.0415 | −0.0099 ± 0.0100 | 0 / 5 |
| Spearman ρ, p = 0.05 | 0.8406 ± 0.1242 | 0.8263 ± 0.1372 | −0.0143 ± 0.0224 | 2 / 5 |
| MAE, p = 0.05 | 0.2975 ± 0.0521 | 0.3015 ± 0.0530 | +0.0040 ± 0.0022 | 0 / 5 |
| Timp simulare zgomot / seed | 70.3 ± 7.0 s | 84.3 ± 3.2 s | +14.0 s | 0 / 5 |

### Cum interpretăm F1 la pragul Exp. 3

Zgomotul depolarizant contractă ⟨Z₀⟩ spre 0, deci scorurile se adună în jurul valorii 0.5. Pragurile Exp. 3 la p = 0.05 (0.503–0.505) sunt calibrate pe contracția circuitului original. Circuitul fuzionat contractă mai puțin, iar la același prag lasă să treacă mai multe tranzacții legitime (seed 0: 17 → 76 FP).

Cu pragul reoptimizat, F1 este practic egal. Scăderea F1 vine deci din pragul neadaptat la noua scară a scorurilor, nu dintr-o clasificare mai slabă: AUPRC și ρ, care nu depind de prag, cresc la 5/5 seeds.

**Concluzie practică:** un prag ales pentru un circuit sub zgomot nu se transferă la alt circuit, chiar dacă cele două sunt echivalente fără zgomot. Valoarea reoptimizată este optimistă, pentru că pragul a fost ales și evaluat pe același validation.

## 5. De ce câștigul e real, dar mic

Canalul depolarizant pe un qubit e covariant unitar. De aceea, perechea „U₁, D_p, U₂, D_p” este egală cu „U₂U₁, apoi D_q”, unde q = 2p − 4p²/3. Fuziunea înlocuiește D_q cu D_p, deci pe acele 20 de poziții zgomotul se reduce aproximativ la jumătate.

Câștigul rămâne totuși modest, din trei motive:
- Cele 42 de canale ale CNOT-urilor rămân neschimbate (62% din cele 68 de locuri de zgomot rămase).
- O parte din porțile eliminate (ultimul strat, pe qubiții 1–3) se află în afara conului de lumină al lui ⟨Z₀⟩ și oricum nu influențau citirea.
- La p = 0.05 scorurile sunt deja puternic contractate. Reducerea relativă a MAE scade de la −16% la p = 0.01 la −4.5% la p = 0.05.

## 6. Reproducerea Exp. 3 și o capcană numerică

Cu dtype-ul implicit torch float32 (setarea folosită în Exp. 3 și Exp. 6), `default.mixed` diferă de `default.qubit` cu ~7e-8 chiar la p = 0, **inclusiv pentru circuitul original**. Cu float64, diferența scade la ~5e-13 (`simulator_precision.csv`). Exp. 8 rulează ambele circuite în float64.

Față de valorile istorice:

| Mărime | Diferență max |
|---|---:|
| AUPRC | ≤ 3e-9 |
| Praguri F1-max | ≤ 7e-8 |
| Scoruri vs cache-ul Exp. 6 | ≤ 8e-8 |

Toate se încadrează în toleranța `REPRO_TOL = 1e-6`, justificată de diagnosticul de mai sus.

F1 la pragul salvat diferă cu o tranzacție în 12 din 20 de cazuri. Pragul F1-max *este* scorul unei anumite tranzacții, iar deplasarea de ~1e-8 o mută sub prag. Comparația pereche Exp. 8 − Exp. 3 folosește Exp. 3 recalculat în aceleași condiții numerice, deci nu este afectată.

## 7. Cost

| | Exp. 3 | Exp. 8 |
|---|---:|---:|
| Antrenare nouă | 0 | 0 (fără reantrenare) |
| Antrenare, dacă s-ar reantrena (nominal, / seed) | 1 150 500 | 1 150 500 |
| Execuții / exemplu, Tracker | 41 | 41 |
| Inferență validation, 4 p, 5 seeds | 1 139 240 | 1 139 240 |
| Shots nominale (1024 / execuție) | 1 166 581 760 | 1 166 581 760 |
| Porți / adâncime per execuție (A) | 67 / 31 | 47 / 26 |
| Timp simulare zgomot, 5 seeds (A) | 5.95 min | 4.58 min |

Numărul de execuții și de shots **nu se schimbă**: el depinde de parametri, exemple și niveluri p, nu de porți. Se reduc complexitatea fiecărei execuții (porți, adâncime) și timpul de simulare. Pe hardware, asta ar însemna execuții mai scurte, nu mai puține.

## 8. Răspunsuri

1. **Câte rotații ai fuzionat?** 20 de fuziuni (40 de porți devin 20 de `Rot`): 4 perechi H+RZ în feature map și 16 perechi RY+RZ în ansatz. Cele 6 RZ din blocurile ZZ nu au putut fi fuzionate.
2. **Ce identități ai folosit?** `RZ(b)·RY(a) = Rot(0, a, b)` (exactă) și `RZ(θ)·H = i·Rot(π, π/2, θ)` (până la fază globală).
3. **Circuitul ideal produce exact aceleași predicții?** Da. Eroarea maximă e 1.3e-15 pe scoruri și 0 pe AUPRC, pe toată validarea și pe 5 seeds.
4. **Câte porți native au dispărut?** 20 (67 → 47, −30%) în modelul oficial.
5. **Câte locuri de zgomot am eliminat?** 20 (88 → 68, −23%) în Scenariul A. În Scenariul B apar 12 în plus (96 → 108).
6. **Numărul de CNOT a rămas identic?** Da: 21, în aceeași ordine.
7. **La p = 0.05, scorurile sunt mai puțin perturbate?** Da în Scenariul A, la 5/5 seeds: MAE −0.013 (−4.5%), ρ +0.048. Efectul este mic, dar consistent. În Scenariul B sunt ușor mai perturbate.
8. **AUPRC s-a îmbunătățit, a scăzut sau a rămas asemănător?**
   - Scenariul A, p = 0.05: o creștere mică, dar la 5/5 seeds (+0.009 ± 0.007).
   - Scenariul A, p ≤ 0.01: practic neschimbat.
   - Scenariul B: o scădere de aceeași mărime (−0.010, 0/5 seeds).
9. **Cât din avantaj depinde de tratarea lui Rot ca poartă nativă?** Tot avantajul. Dacă `Rot` se descompune generic în RZ·RY·RZ, fuziunea se inversează: +12 locuri de zgomot, AUPRC și MAE mai slabe. Chiar și cu un compilator care elimină `RZ(0)` (B′), câștigul structural scade la 4 locuri de zgomot.
10. **Se reduce costul estimat de execuție sau doar complexitatea circuitului?** Doar complexitatea: porți, adâncime și timpul de simulare (−23% în A). Numărul de execuții și de shots rămâne identic.
11. **Merită păstrată optimizarea în arhitectura finală?** Da, ca rafinare *opțională* în modelul oficial de zgomot. Este gratuită (aceeași funcție, fără reantrenare) și aduce un câștig de robustețe mic, dar consistent. Nu trebuie prezentată ca îmbunătățire majoră și nu înlocuiește contribuția Exp. 3 (+0.21 AUPRC față de baseline). Are două condiții:
    - pragurile trebuie reselectate pe validation pentru circuitul fuzionat;
    - testul a fost deja consultat în Exp. 5, deci orice evaluare pe test a lui Exp. 8 ar fi doar exploratorie.
12. **Care sunt limitele față de hardware-ul cuantic real?**
    - Modelul oficial taxează fiecare poartă de un qubit cu același p și tratează `Rot` ca pe o singură poartă. Pe hardware real, porțile native sunt altele (de ex. RZ virtual, √X, ECR/CZ), iar `Rot` se compilează în 2–3 pulsuri.
    - Pe multe dispozitive, RZ este virtual și fără eroare, așa că o parte din câștig ar dispărea.
    - Erorile reale sunt dominate de porțile de doi qubiți, de citire și de decoerență, pe care fuziunea nu le atinge.
    - Concluzia rămâne deci valabilă **doar** pentru modelul de simulare din task (Scenariul A).

## Verdict

| | Exp. 3 original | Exp. 8 fusion (A, oficial) | Exp. 8 fusion (B, Rot descompus) |
|---|---:|---:|---:|
| Funcție fără zgomot | referință | identică (≤ 1.3e-15) | identică |
| Porți / CNOT / adâncime | 67 / 21 / 31 | **47 / 21 / 26** | 87 / 21 / 36 |
| Locuri de zgomot | 88 (A), 96 (B) | **68** | 108 |
| AUPRC, p = 0.05 | 0.587 (A), 0.584 (B) | **0.596 (+0.009, 5/5)** | 0.574 (−0.010, 0/5) |
| ρ, p = 0.05 | 0.875 (A), 0.841 (B) | **0.923 (+0.048, 5/5)** | 0.826 (−0.014) |
| MAE, p = 0.05 | 0.294 (A), 0.297 (B) | **0.281 (−4.5%)** | 0.301 (+1.3%) |
| Execuții / shots | 1 139 240 / 1.17e9 | identic | identic |

**Verdict obiectiv:** fuziunea rotațiilor este o optimizare **corectă matematic** (echivalență verificată la precizia mașinii) care, **în modelul oficial de zgomot**, elimină 20 de porți și 20 de locuri de zgomot și aduce un câștig de robustețe **mic, dar consistent** (5/5 seeds la p = 0.05). Avantajul depinde însă **în întregime** de convenția că `Rot` este o singură poartă: dacă `Rot` este descompusă în rotații elementare, fuziunea **înrăutățește** circuitul. Nu reduce costul în execuții sau shots. Metricile cu prag necesită reselectarea pragului pentru noul circuit.

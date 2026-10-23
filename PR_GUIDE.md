# Ghid Pull Request – BreaQ 2026 (Quantum AI)

Acest ghid descrie cum deschidem, verificăm și acceptăm Pull Request-urile (PR) în echipa de doi.

## Regula de bază

**Autorul unui PR nu își face merge singur. Celălalt coleg verifică și acceptă PR-ul.**

De ce:
- A doua pereche de ochi prinde greșeli (fișiere urcate din greșeală, cod stricat, chei expuse).
- Amândoi știm ce intră în `main`, deci nimeni nu e luat prin surprindere.
- `main` rămâne mereu o versiune care rulează, iar codul de acolo se publică la final.

Regula e o convenție de echipă, GitHub nu o impune automat. Respectați-o și când vă grăbiți.

---

## 1. Pentru autor: înainte să deschizi PR-ul

- [ ] Ești pe ramura ta (`git branch --show-current`), nu pe `main`.
- [ ] `git pull origin main` și `git merge main` în ramura ta, ca să fii la zi.
- [ ] Codul rulează local (notebook-ul se execută de la cap la coadă, fără erori).
- [ ] `git status` nu arată fișiere nedorite: **nu** trebuie să apară `data/`, `*.csv`, `.venv/`, `.env`, chei sau parole.
- [ ] Outputurile notebook-urilor sunt curățate (`nbstripout` e activ).
- [ ] `git push` pe ramura ta.

Apoi deschizi PR-ul (GitHub → *Compare & pull request*, sau `gh pr create`) către `main`, cu:
- **Titlu scurt** care spune ce face (ex. „Add classical baseline for fraud detection").
- **Descriere de 2-4 rânduri:** ce ai schimbat, cum se rulează, ce rezultate ai (metrici), ce nu e terminat.
- Îi spui colegului pe Discord/WhatsApp că PR-ul e gata.

---

## 2. Pentru reviewer (celălalt coleg): cum verifici și accepți

Se face în ordinea de mai jos. Durează 5-10 minute.

### Pas 1 – Citește descrierea
Înțelege ce vrea PR-ul să facă, înainte să te uiți la cod.

### Pas 2 – Verifică fișierele (tab-ul *Files changed*)
- [ ] **Nu apar** `data/`, `*.csv`, `.venv/`, `.env`, `__pycache__/`, chei API, tokenuri sau parole.
- [ ] Nu s-au modificat fișiere care nu au legătură cu PR-ul (`requirements.txt`, `CLAUDE.md`, `.gitignore`) fără motiv. Dacă `requirements.txt` s-a schimbat, versiunile noi trebuie anunțate echipei.
- [ ] Nu e cod comentat în exces, `print`-uri de debug sau fișiere temporare.
- [ ] Fișierele modificate sunt cele din zona autorului (ca să evitați conflicte).

### Pas 3 – Citește codul
- [ ] E clar ce face fiecare funcție, iar numele sunt sugestive.
- [ ] Codul reutilizabil e în `src/`, nu copiat între notebook-uri.
- [ ] Seed-urile sunt fixate, iar split-ul e stratificat.
- [ ] Preprocesarea (scaler etc.) e fitată doar pe train, fără scurgeri de date.
- [ ] Rezultatele raportează precision, recall, F1 și/sau PR-AUC, nu doar accuracy (datasetul e foarte dezechilibrat).
- [ ] Există o comparație cu un baseline clasic, pe același split.

### Pas 4 – Rulează codul local (recomandat)
```
git fetch
git checkout ramura-autorului
```
Rulează notebook-ul sau scriptul. Dacă nu rulează, **nu accepta**, spune-i autorului ce eroare ai.
Apoi revino pe ramura ta:
```
git checkout ramura-ta
```

### Pas 5 – Lasă feedback
Pe GitHub: *Files changed* → *Review changes* → alege:
- **Approve:** totul e în regulă.
- **Request changes:** trebuie reparat ceva, scrie concret ce și unde.
- **Comment:** doar observații, fără decizie.

Feedback-ul e despre cod, nu despre persoană. Fii concret și scurt.

### Pas 6 – Merge
Dacă ai dat Approve și nu sunt conflicte:
1. Apasă **Merge pull request**.
2. Preferă **Squash and merge** (un singur commit curat în `main`).
3. Confirmă și, dacă ramura nu mai e folosită, apasă **Delete branch**.

Din terminal:
```
gh pr list
gh pr view NUMĂR
gh pr merge NUMĂR --squash
```

---

## 3. După merge (amândoi)

```
git checkout main
git pull
git checkout ramura-ta
git merge main
git push
```
Așa lucrați mereu pe cea mai nouă versiune de cod.

---

## 4. Conflicte

Dacă GitHub spune „This branch has conflicts":
1. **Autorul** rezolvă conflictul local:
   ```
   git checkout ramura-ta
   git merge main
   ```
2. Git marchează zonele cu `<<<<<<<`, `=======`, `>>>>>>>`. Le editezi manual și păstrezi varianta corectă (sau ceri ajutor colegului).
3. Apoi:
   ```
   git add .
   git commit
   git push
   ```
4. PR-ul se actualizează singur, iar reviewer-ul îl poate accepta.

Ca să evitați conflictele: nu lucrați amândoi pe același fișier în același timp.

---

## 5. Reguli de aur

- **Nu se face push direct pe `main`.** Tot ce intră acolo trece printr-un PR.
- **Nu se face merge la propriul PR.** Îl acceptă colegul.
- **Nu se folosește `git push --force`** pe `main` sau pe ramura colegului.
- **Nu se urcă niciodată** datasetul, mediul virtual, chei sau parole.
- PR-urile mici sunt mai ușor de verificat decât cele mari. Deschide-le des, nu o dată la final.
- Un PR neverificat nu se acceptă „pe încredere", chiar și sub presiunea timpului.

## 6. Cazul special: lipsește colegul sau e urgent

Dacă celălalt coleg nu poate verifica imediat (doarme, e în pauză, e blocat):
- Autorul **așteaptă** sau cere confirmare pe chat înainte de merge.
- Dacă e absolut necesar, merge doar cu **acordul explicit al colegului**, scris pe chat (ex. „ok, fă merge, mă uit după").
- Reviewer-ul verifică oricum după, iar dacă găsește probleme, repară într-un PR nou.

## 7. Înainte de predare (cu o oră înainte de prezentare)

- [ ] Toate PR-urile importante sunt în `main`.
- [ ] `main` rulează curat dintr-un clone proaspăt:
  ```
  git clone https://github.com/AlexandruMihaiVrabie/breaq-quantum-ai.git test-clone
  ```
  Instalezi din `requirements.txt`, pui datasetul în `data/` și rulezi notebook-ul principal.
- [ ] README-ul e actualizat (cum se rulează, rezultatele principale).
- [ ] Repo-ul respectă cerința organizatorilor: codul publicat la finalul hackatonului.

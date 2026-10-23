# BreaQ 2026 – Quantum AI (PennyLane)

Hackathon project (24–25 Oct 2026, CAMPUS Research Institute, București), track Quantum AI.
Task: explore a quantum AI pipeline with PennyLane and make it robust to realistic noise.
Team of two. Work is judged on code (published at the end) and a short final presentation.

## Stack
- Python 3.12, PennyLane 0.45.1, torch 2.14.1, scikit-learn 1.9.1, pandas, numpy, scipy, matplotlib
- Dependencies live in `requirements.txt`. Do not bump versions without telling the team.
- Dataset: credit card fraud (OpenML 42175), file `data/creditcard.csv`
  (284,807 rows, 31 columns, target `Class`: 284,315 legit vs 492 fraud).
  `data/` is gitignored. Never commit the CSV.

## Structure
- `src/`: shared reusable code (data loading, models, circuits, metrics)
- `notebooks/`: experiments (one notebook per person, e.g. `alex_*.ipynb`)
- `data/`: local only
- `README.md`: setup and team workflow

## ML rules for this project
- Extremely imbalanced data: never report accuracy alone. Use precision, recall, F1, PR-AUC (average precision), and the confusion matrix.
- Split train/test with `stratify=y` and a fixed `random_state`. Fit scalers and any preprocessing on train only (no leakage).
- Always set seeds (numpy, torch, PennyLane) and say so in the notebook.
- Always compare against a classical baseline (e.g. logistic regression, random forest) on the same split.
- Quantum circuits are slow: start with a small subsample and few qubits, then scale up. Keep the number of qubits, layers and shots in a config block at the top.
- For noise experiments, use `default.mixed` or noise channels (depolarizing, amplitude damping, etc.) and report results at several noise levels.
- Put reusable functions in `src/` and import them; don't copy-paste between notebooks.

## Git workflow
- Never commit or push directly to `main`. Each person works on their own branch (`alex-...`, `coleg-...`) and merges via Pull Request.
- The team runs `git pull --ff-only` before starting a session, and commits small and often and pushes regularly (manually; see the next section for what Claude may do).
- Do not both edit the same file at the same time. Split work by file/module.
- Notebook outputs are stripped by `nbstripout` (installed as a git filter). Do not disable it.
- Never commit: `.venv/`, `data/`, `*.csv`, `.env`, API keys, tokens or passwords.
- Commit messages: short, imperative, no trailers.
- Do NOT add "Co-authored-by", "Generated with Claude" or any similar attribution line to commits or pull requests.

## Git: ce poate și ce nu poate face Claude
- Claude NU rulează `git commit`, `git push`, `git merge`, `git rebase`, `git reset`, `gh pr create` sau `gh pr merge`. Commit-ul, push-ul și PR-urile le facem noi, manual.
- Claude POATE rula `git pull`, dar doar ca `git pull --ff-only`. Dacă pull-ul nu se poate face liniar (conflict sau ramuri divergente), Claude se oprește, ne explică situația și ne lasă pe noi să decidem, nu face merge sau rebase singur.
- Claude poate rula comenzi de citire: `git status`, `git diff`, `git log`, `git branch`, `git show`, `git ls-files`, `git fetch`, `gh pr view`, `gh pr list`, `gh pr diff`.
- Când e momentul unui commit sau push, Claude ne dă comenzile exacte de copiat și un mesaj de commit propus (scurt, fără trailere), dar nu le execută.

## După commit: verificare conform PR_GUIDE.md
Când spunem că am făcut commit, push sau am deschis un PR (sau rulăm `/review-pr`):
1. Claude citește `PR_GUIDE.md` din rădăcina repo-ului.
2. Întreabă dacă vorbește cu **autorul** sau cu **reviewer-ul** (dacă nu e clar) și parcurge checklist-ul potrivit: secțiunea 1 pentru autor, secțiunea 2 pentru reviewer.
3. Rulează doar verificări de citire: `git status`, `git log --oneline main..HEAD`, `git diff --stat main...HEAD`, `git ls-files` (să nu apară `data/`, `*.csv`, `.venv/`, `.env`), și citește fișierele modificate.
4. Ne dă un raport scurt cu OK / ATENȚIE / PROBLEMĂ pentru fiecare punct din ghid, plus ce trebuie să verificăm noi manual (rularea notebook-ului pe mașina noastră, decizia de merge).
5. Reamintește regula de bază: autorul nu își face merge singur, îl acceptă colegul. Claude nu aprobă, nu dă merge și nu modifică cod în timpul review-ului, doar sugerează.

## How to work in this repo (for Claude)
- Ask before running anything that is slow (training on the full dataset, large circuits) or that installs/uninstalls packages.
- Never delete or overwrite files without asking. Never use destructive git commands (`reset --hard`, `push --force`, `clean -fd`) unless explicitly told.
- Explain what a change does in 1–2 sentences. Keep code simple and readable; this is a 24-hour hackathon, not a library.
- Prefer small, runnable steps and show results (metrics, plots) over long explanations.
- Use the project's own virtual environment; don't create new ones.
- Windows (teammate) and macOS (Alex) both in use: avoid OS-specific paths; use `pathlib` and relative paths from the repo root.

## Useful commands
- Activate env: macOS `source .venv/bin/activate`; Windows use the env's `Scripts\activate`
- Install deps: `pip install -r requirements.txt`
- Quick check: `python -c "import pennylane, torch, sklearn; print(pennylane.__version__, torch.__version__, sklearn.__version__)"`

# breaq-hackathon

Proiect pentru hackatonul BreaQ 2026, track Quantum AI (PennyLane): detecția fraudelor cu carduri de credit.

## Structură
- `data/` — datasetul (`creditcard.csv`), NU este în git
- `notebooks/` — notebook-uri Jupyter
- `src/` — cod Python reutilizabil
- `.venv/` — mediul virtual Python 3.12 (local, nu este în git)
- `requirements.txt` — dependențe (`pip freeze`)

## Pregătirea mediului
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
nbstripout --install
```
Kernel Jupyter: `python -m ipykernel install --user --name breaq`.

## Dataset
Credit Card Fraud Detection, OpenML 42175 (sau Kaggle `mlg-ulb/creditcardfraud`).
Datasetul NU este în git: după clone, creează manual folderul `data/` și pune fișierul în `data/creditcard.csv`.

## Cum lucrăm în echipă
- Fiecare lucrează pe ramura lui (ex. `alex-preprocessing`), nu direct pe `main`.
- `git pull` înainte de a începe lucrul.
- Commit și push des.
- Fișiere/notebook-uri diferite pentru fiecare, ca să evităm conflictele.
- Modificările ajung în `main` prin Pull Request-uri.

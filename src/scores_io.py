"""Încărcare scoruri salvate (Exp. 5), etichete și verificarea split-ului comun."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
SPLIT_PATH = ROOT / "split_indices.npz"
DATA_PATH = ROOT / "data" / "creditcard.csv"

SEEDS = [0, 1, 2, 3, 4]
NOISE_LEVELS = [0.0, 0.001, 0.01, 0.05]
EXPECTED_FRAUDS = {"train": 295, "val": 99, "test": 98}
EXPECTED_SIZES = {"train": 170_883, "val": 56_962, "test": 56_962}


def expected_split_sha256():
    config = json.loads((RESULTS_DIR / "result_exp5" / "config.json").read_text())
    return config["split_sha256"]


def check_split():
    """P1: sha256 al split-ului, dimensiuni și număr de fraude pe fiecare set. Încarcă doar coloana Class."""
    sha = hashlib.sha256(SPLIT_PATH.read_bytes()).hexdigest()
    assert sha == expected_split_sha256(), f"split_indices.npz diferă de cel din Exp. 5: {sha}"
    split = np.load(SPLIT_PATH)
    y = pd.read_csv(DATA_PATH, usecols=["Class"])["Class"].to_numpy()
    for name in ["train", "val", "test"]:
        idx = split[name]
        assert len(idx) == EXPECTED_SIZES[name], f"{name}: {len(idx)} tranzacții"
        assert int(y[idx].sum()) == EXPECTED_FRAUDS[name], f"{name}: {int(y[idx].sum())} fraude"
    return sha


def load_labels(part):
    """Etichetele Class pentru 'train', 'val' sau 'test', în ordinea din split_indices.npz."""
    idx = np.load(SPLIT_PATH)[part]
    y = pd.read_csv(DATA_PATH, usecols=["Class"])["Class"].to_numpy()
    return y[idx]


def score_path(model, seed, p):
    """C (regresia logistică) e clasică: are un singur fișier, la p = 0.0, valabil pentru orice p."""
    if model == "C":
        p = 0.0
    return RESULTS_DIR / "result_exp5" / "scores" / f"{model}_seed{seed}_p{p}.npz"


def load_scores(model, seed, p, part="val"):
    """Scorurile unui model ('A', 'B' sau 'C') pe 'val' sau 'test'. Citește doar cheia cerută din .npz."""
    assert part in ("val", "test")
    with np.load(score_path(model, seed, p)) as z:
        return z[part]

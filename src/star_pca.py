"""Exp. 10: STAR ZZ VQC cu două encodere PCA (BALANCED vs FULL-TRAIN).

Funcțiile de circuit, antrenare, zgomot și metrici sunt copiate fidel din notebook-ul Exp. 9
(`notebooks/exp9_sparse_zz_feature_map.ipynb`); singura noutate este `FullTrainPCAEncoder`.
Exp. 9 definește totul în notebook, deci verificarea reproducerii se face în Exp. 10 față de rezultatele salvate.
"""
import hashlib
import time

import numpy as np
import pennylane as qml
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

N_QUBITS = 4
FEATURE_MAP_REPS = 1
ANSATZ_REPS = 3                 # (3 + 1) * 4 * 2 = 32 parametri
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.05
STAR_PAIRS = [(0, 1), (0, 2), (0, 3)]
NATIVE_GATES = {"RX", "RY", "RZ", "Rot", "Hadamard", "CNOT", "CZ"}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def undersample(X_part, y_part, seed):
    rng = np.random.default_rng(seed)
    fraud = np.flatnonzero(y_part == 1)
    legit = rng.choice(np.flatnonzero(y_part == 0), size=len(fraud), replace=False)
    keep = rng.permutation(np.concatenate([fraud, legit]))
    return X_part.iloc[keep], y_part[keep]


class AngleEncoder:
    """Encoderul din Exp. 1-9 (BALANCED PCA): StandardScaler pe Time/Amount, PCA și MinMaxScaler, toate fitate pe setul dat."""
    raw_columns = ["Time", "Amount"]

    def __init__(self, n_components, seed):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=seed)
        self.angles = MinMaxScaler(feature_range=(0, np.pi))

    def _scaled(self, frame):
        out = frame.copy()
        out[self.raw_columns] = self.scaler.transform(frame[self.raw_columns])
        return out.to_numpy()

    def fit(self, frame):
        self.scaler.fit(frame[self.raw_columns])
        self.angles.fit(self.pca.fit_transform(self._scaled(frame)))
        return self

    def transform(self, frame):
        return np.clip(self.angles.transform(self.pca.transform(self._scaled(frame))), 0, np.pi)


class FullTrainPCAEncoder(AngleEncoder):
    """Ca AngleEncoder, dar `PCA.fit` vede tot train-ul. StandardScaler și MinMaxScaler se fitează ca în varianta A
    (pe cele 590 de exemple echilibrate; MinMaxScaler pe proiecțiile PCA-ului propriu ale acestora)."""

    def fit(self, balanced_frame, full_train_frame):
        self.scaler.fit(balanced_frame[self.raw_columns])
        self.pca.fit(self._scaled(full_train_frame))
        self.angles.fit(self.pca.transform(self._scaled(balanced_frame)))
        return self


def unclipped_angles(encoder, frame):
    """Unghiurile înainte de np.clip(0, pi): valorile din afara [0, pi] sunt cele din afara domeniului de fit al MinMaxScaler."""
    return encoder.angles.transform(encoder.pca.transform(encoder._scaled(frame)))


# ---- circuitul STAR ZZ (Exp. 9) ----
def feature_map_star(x):
    for _ in range(FEATURE_MAP_REPS):
        for i in range(N_QUBITS):
            qml.Hadamard(wires=i)
            qml.RZ(2 * x[..., i], wires=i)
        for i, j in STAR_PAIRS:
            qml.CNOT(wires=[i, j])
            qml.RZ(2 * (np.pi - x[..., i]) * (np.pi - x[..., j]), wires=j)
            qml.CNOT(wires=[i, j])


def ansatz(weights):
    for layer in range(ANSATZ_REPS + 1):
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        if layer < ANSATZ_REPS:
            for i in range(N_QUBITS - 1):
                qml.CNOT(wires=[i, i + 1])


def star_circuit(x, weights):
    feature_map_star(x)
    ansatz(weights)
    return qml.expval(qml.PauliZ(0))


_dev = qml.device("default.qubit", wires=N_QUBITS)   # valori de așteptare exacte
_qnode = qml.QNode(star_circuit, _dev, interface="torch", diff_method="backprop")


def model(x, weights):
    """Probabilitatea de fraudă s = (1 - <Z0>) / 2."""
    return (1 - _qnode(x, weights)) / 2


def readout(expval):
    return (1 - expval) / 2


def init_weights(seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.nn.Parameter(2 * np.pi * torch.rand(ANSATZ_REPS + 1, N_QUBITS, 2, generator=generator, dtype=torch.float64))


def predict(Z, weights, chunk=20000):
    X_t = torch.tensor(Z)
    with torch.no_grad():
        return torch.cat([model(X_t[k:k + chunk], weights) for k in range(0, len(X_t), chunk)]).numpy()


def train_vqc_run(Z_tr, y_tr, Z_valid, y_valid, initial_weights, seed, epochs=EPOCHS):
    """Bucla de antrenare din Exp. 1-9: Adam, BCE, mini-batch 32, epoca aleasă după AUPRC pe validation (p = 0)."""
    set_seed(seed)
    shuffle_generator = torch.Generator().manual_seed(seed)
    weights = torch.nn.Parameter(initial_weights.detach().clone())
    optimizer = torch.optim.Adam([weights], lr=LEARNING_RATE)
    loss_fn = torch.nn.BCELoss()
    X_t = torch.tensor(Z_tr)
    y_t = torch.tensor(y_tr, dtype=torch.float64)
    history, best_ap, best_weights, best_epoch = [], -1.0, None, None
    orders = hashlib.sha256()
    adam_steps = 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(X_t), generator=shuffle_generator)
        orders.update(order.numpy().tobytes())
        losses = []
        for k in range(0, len(X_t), BATCH_SIZE):
            idx = order[k:k + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(X_t[idx], weights), y_t[idx])
            loss.backward()
            optimizer.step()
            adam_steps += 1
            losses.append(loss.item())
        val_ap = average_precision_score(y_valid, predict(Z_valid, weights))
        history.append({"epoch": epoch, "train loss": float(np.mean(losses)), "validation AUPRC": float(val_ap)})
        if val_ap > best_ap:
            best_ap, best_weights, best_epoch = val_ap, weights.detach().clone(), epoch
    return {"weights": best_weights, "history": history, "best_epoch": best_epoch, "best_val_ap": float(best_ap),
            "time_train_s": time.perf_counter() - start, "batch_order_sha256": orders.hexdigest(), "adam_steps": adam_steps}


# ---- zgomot (modelul oficial, ca în Exp. 2-9) ----
def noisy_qnode(circuit_fn, p, n_wires=N_QUBITS):
    noisy_device = qml.noise.insert(qml.device("default.mixed", wires=n_wires), qml.DepolarizingChannel, p, position="all")
    qnode = qml.QNode(circuit_fn, noisy_device, interface="torch", diff_method="backprop")
    return qml.transforms.decompose(qnode, gate_set=NATIVE_GATES)


def noisy_predict(Z, weights, p, circuit_fn=star_circuit, n_wires=N_QUBITS, chunk=2000):
    noisy_circuit = noisy_qnode(circuit_fn, p, n_wires)
    X_t = torch.tensor(Z)
    with torch.no_grad():
        return torch.cat([readout(noisy_circuit(X_t[k:k + chunk], weights)) for k in range(0, len(X_t), chunk)]).numpy()


def noise_locations(circuit_fn, *args, n_wires=N_QUBITS):
    qnode = qml.transforms.decompose(qml.QNode(circuit_fn, qml.device("default.qubit", wires=n_wires)), gate_set=NATIVE_GATES)
    return sum(size * count for size, count in qml.specs(qnode)(*args)["resources"].gate_sizes.items())


# ---- metrici ----
def choose_threshold(y_true, scores):
    """Pragul care maximizează F1 pe (y_true, scores)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return thresholds[np.argmax(f1[:-1])]


def evaluate(y_true, scores, threshold):
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {"AUPRC": average_precision_score(y_true, scores), "AUROC": roc_auc_score(y_true, scores),
            "precision": precision_score(y_true, predicted, zero_division=0), "recall": recall_score(y_true, predicted, zero_division=0),
            "F1": f1_score(y_true, predicted, zero_division=0), "MCC": matthews_corrcoef(y_true, predicted),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def recall_target_row(y_true, scores, target):
    """Regula deterministă pentru ținta de recall: dintre pragurile curbei PR cu recall >= target, cel mai mare (cele mai puține alarme).
    Exploratoriu: pragul se alege și se evaluează pe același validation."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    ok = np.flatnonzero(recall[:-1] >= target)
    if len(ok) == 0:
        return None
    threshold = thresholds[ok.max()]
    return {"target recall": target, "threshold": float(threshold), **evaluate(y_true, scores, threshold)}

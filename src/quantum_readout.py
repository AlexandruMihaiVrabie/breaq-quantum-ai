"""Exp. 11: STAR ZZ circuit with a configurable readout observable.

The data, circuit, training and noise functions are copied unchanged from Exp. 9
(notebooks/exp9_sparse_zz_feature_map.ipynb, itself copied from Exp. 3 / the official notebook).
The only new parts are the readout observable as an argument (`make_vqc(pairs, observable)`)
and the causal-cone helpers at the end.
"""
import hashlib
import time

import numpy as np
import pennylane as qml
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ---- fixed, identical to Exp. 9 ----
N_QUBITS = 4
FEATURE_MAP_REPS = 1
ANSATZ_REPS = 3               # (3 + 1) * 4 * 2 = 32 parameters
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.05
NATIVE_GATES = {"RX", "RY", "RZ", "Rot", "Hadamard", "CNOT", "CZ"}
STAR_PAIRS = [(0, 1), (0, 2), (0, 3)]

# ---- the only change in Exp. 11: the measured observable ----
OBSERVABLES = {
    "Z0": lambda: qml.PauliZ(0),
    "Z1": lambda: qml.PauliZ(1),
    "Z2": lambda: qml.PauliZ(2),
    "Z3": lambda: qml.PauliZ(3),
    "Z0Z1": lambda: qml.PauliZ(0) @ qml.PauliZ(1),
}
OBSERVABLE_WIRES = {"Z0": [0], "Z1": [1], "Z2": [2], "Z3": [3], "Z0Z1": [0, 1]}


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


def make_feature_map(pairs):
    """The official ZZ feature map with ZZ interactions only on `pairs` (kept in the given order)."""
    def feature_map_pairs(x):
        for _ in range(FEATURE_MAP_REPS):
            for i in range(N_QUBITS):
                qml.Hadamard(wires=i)
                qml.RZ(2 * x[..., i], wires=i)
            for i, j in pairs:
                qml.CNOT(wires=[i, j])
                qml.RZ(2 * (np.pi - x[..., i]) * (np.pi - x[..., j]), wires=j)
                qml.CNOT(wires=[i, j])
    return feature_map_pairs


def ansatz(weights):
    """EfficientSU2-style ansatz; weights has shape (ANSATZ_REPS + 1, N_QUBITS, 2)."""
    for layer in range(ANSATZ_REPS + 1):
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        if layer < ANSATZ_REPS:
            for i in range(N_QUBITS - 1):
                qml.CNOT(wires=[i, i + 1])


def make_vqc(pairs, observable="Z0"):
    """Exp. 9 `make_vqc(pairs)` with the readout observable as an argument (a key of OBSERVABLES)."""
    fm = make_feature_map(pairs)
    obs = OBSERVABLES[observable]

    def vqc_pairs(x, weights):
        fm(x)
        ansatz(weights)
        return qml.expval(obs())
    return vqc_pairs


def readout(expval):
    return (1 - expval) / 2


def make_model(qnode):
    """fraud_probability = (1 - <O>) / 2 for a QNode returning <O>."""
    return lambda x, weights: readout(qnode(x, weights))


def init_weights(seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.nn.Parameter(2 * np.pi * torch.rand(ANSATZ_REPS + 1, N_QUBITS, 2, generator=generator, dtype=torch.float64))


def predict(Z, weights, model, chunk=20000):
    X_t = torch.tensor(Z)
    with torch.no_grad():
        return torch.cat([model(X_t[k:k + chunk], weights) for k in range(0, len(X_t), chunk)]).numpy()


def train_vqc_run(Z_tr, y_tr, Z_valid, y_valid, initial_weights, seed, model, epochs=EPOCHS):
    """The training loop of Exp. 1-3 / Exp. 9, with the model as an argument."""
    set_seed(seed)
    shuffle_generator = torch.Generator().manual_seed(seed)
    weights = torch.nn.Parameter(initial_weights.detach().clone())   # independent copy
    optimizer = torch.optim.Adam([weights], lr=LEARNING_RATE)
    loss_fn = torch.nn.BCELoss()
    X_t = torch.tensor(Z_tr)
    y_t = torch.tensor(y_tr, dtype=torch.float64)
    history, best_ap, best_weights, best_epoch = [], -1.0, None, None
    orders = hashlib.sha256()
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
            losses.append(loss.item())
        val_ap = average_precision_score(y_valid, predict(Z_valid, weights, model))
        history.append({"epoch": epoch, "train loss": float(np.mean(losses)), "validation AUPRC": float(val_ap)})
        if val_ap > best_ap:
            best_ap, best_weights, best_epoch = val_ap, weights.detach().clone(), epoch
    return {"weights": best_weights, "history": history, "best_epoch": best_epoch, "best_val_ap": float(best_ap),
            "time_train_s": time.perf_counter() - start, "batch_order_sha256": orders.hexdigest()}


def choose_threshold(y_true, scores):
    """The threshold that maximises F1 on (y_true, scores)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return thresholds[np.argmax(f1[:-1])]


def evaluate(y_true, scores, threshold):
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "AUPRC": average_precision_score(y_true, scores),
        "AUROC": roc_auc_score(y_true, scores),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "F1": f1_score(y_true, predicted, zero_division=0),
        "MCC": matthews_corrcoef(y_true, predicted),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


def noisy_qnode(circuit_fn, p, n_wires=N_QUBITS):
    """QNode of circuit_fn under the challenge's noise model: native gates, then a depolarizing channel of
    probability p after every gate, on each of its qubits. Differentiable with backprop."""
    noisy_device = qml.noise.insert(qml.device("default.mixed", wires=n_wires), qml.DepolarizingChannel, p, position="all")
    qnode = qml.QNode(circuit_fn, noisy_device, interface="torch", diff_method="backprop")
    return qml.transforms.decompose(qnode, gate_set=NATIVE_GATES)


def noisy_predict(Z, weights, p, circuit_fn, n_wires=N_QUBITS, chunk=2000):
    """Fraud scores of the transactions Z when circuit_fn runs under depolarizing noise p."""
    noisy_circuit = noisy_qnode(circuit_fn, p, n_wires)
    X_t = torch.tensor(Z)
    with torch.no_grad():
        return torch.cat([readout(noisy_circuit(X_t[k:k + chunk], weights)) for k in range(0, len(X_t), chunk)]).numpy()


def noise_locations(circuit_fn, *args, n_wires=N_QUBITS):
    """Number of depolarizing channels the noise model inserts: one per qubit of every native gate."""
    qnode = qml.transforms.decompose(qml.QNode(circuit_fn, qml.device("default.qubit", wires=n_wires)), gate_set=NATIVE_GATES)
    return sum(size * count for size, count in qml.specs(qnode)(*args)["resources"].gate_sizes.items())


# ---- new in Exp. 11: backward causal (light) cone of the measured observable ----

def tape_ops(circuit_fn, x, weights, n_wires=N_QUBITS):
    """Operations of circuit_fn after decomposition into NATIVE_GATES (numpy inputs: one transaction)."""
    qnode = qml.transforms.decompose(qml.QNode(circuit_fn, qml.device("default.qubit", wires=n_wires)), gate_set=NATIVE_GATES)
    return qml.workflow.construct_tape(qnode)(x, weights).operations


def backward_light_cone(ops, measured_wires):
    """Wire-based backward light cone. Walks the gate list from the measurement backwards, keeping the set S of
    wires that can still influence the measured wires. A gate is in the cone if it touches S (then all its wires
    join S). The depolarizing channels placed right after a gate act on wires whose operator, at that point in
    the Heisenberg picture, is the identity unless the wire is already in S; so a channel counts as in the cone
    only if its wire is in S *before* the gate is added. Returns per-op flags and per-op in-cone channel wires."""
    S = set(measured_wires)
    in_cone, noisy_wires = [False] * len(ops), [[] for _ in ops]
    for k in range(len(ops) - 1, -1, -1):
        wires = list(ops[k].wires)
        noisy_wires[k] = [w for w in wires if w in S]
        if noisy_wires[k]:
            in_cone[k] = True
            S |= set(wires)
    return in_cone, noisy_wires


def selective_noise_qnode(ops, channel_mask, p, measurement, n_wires=N_QUBITS):
    """Runs the fixed op list `ops` on default.mixed, with DepolarizingChannel(p) after op k on each wire w
    for which channel_mask[k] contains w. Used only for the causal-cone checks, on single transactions."""
    @qml.qnode(qml.device("default.mixed", wires=n_wires))
    def circuit():
        for op, wires in zip(ops, channel_mask):
            qml.apply(op)
            for w in wires:
                qml.DepolarizingChannel(p, wires=w)
        return qml.expval(measurement())
    return circuit

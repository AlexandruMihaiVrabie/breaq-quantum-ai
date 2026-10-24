"""Circuitul VQC (Modelul A/B), encoderul de unghiuri și scorarea sub zgomot depolarizant.

Cod mutat din notebook-urile Exp. 2–7, neschimbat numeric. Modelul B: FEATURE_MAP_REPS = 1, ANSATZ_REPS = 3.
"""
import numpy as np
import pennylane as qml
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

N_QUBITS = 4
ANSATZ_REPS = 3                  # (3 + 1) * 4 * 2 = 32 parametri
FEATURE_MAP_REPS_A, FEATURE_MAP_REPS_B = 2, 1
NATIVE_GATES = {"RX", "RY", "RZ", "Rot", "Hadamard", "CNOT", "CZ"}


def undersample(X_part, y_part, seed):
    """Toate fraudele + același număr de legitime trase fără înlocuire, apoi permutate (1:1)."""
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


def make_vqc(feature_map_reps, ansatz_reps=ANSATZ_REPS, n_qubits=N_QUBITS):
    """Returnează funcția circuitului (x, weights) -> <Z0>."""
    def feature_map(x):
        """ZZ feature map. x are N_QUBITS unghiuri în [0, pi], sau un batch (formă (..., N_QUBITS))."""
        for _ in range(feature_map_reps):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(2 * x[..., i], wires=i)
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    qml.CNOT(wires=[i, j])
                    qml.RZ(2 * (np.pi - x[..., i]) * (np.pi - x[..., j]), wires=j)
                    qml.CNOT(wires=[i, j])

    def ansatz(weights):
        """Ansatz tip EfficientSU2; weights are forma (ansatz_reps + 1, n_qubits, 2)."""
        for layer in range(ansatz_reps + 1):
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            if layer < ansatz_reps:
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

    def vqc(x, weights):
        feature_map(x)
        ansatz(weights)
        return qml.expval(qml.PauliZ(0))

    return vqc


def noisy_qnode(circuit_fn, p, n_wires=N_QUBITS):
    """QNode al lui circuit_fn sub modelul de zgomot al competiției: porți native, apoi un canal depolarizant
    de probabilitate p după fiecare poartă, pe fiecare qubit al ei."""
    noisy_device = qml.noise.insert(qml.device("default.mixed", wires=n_wires), qml.DepolarizingChannel, p, position="all")
    qnode = qml.QNode(circuit_fn, noisy_device, interface="torch", diff_method="backprop")
    return qml.transforms.decompose(qnode, gate_set=NATIVE_GATES)


def noisy_predict(Z, weights, p, circuit_fn, readout=lambda expval: (1 - expval) / 2, n_wires=N_QUBITS, chunk=2000):
    """Scorurile de fraudă ale tranzacțiilor Z când circuit_fn rulează sub zgomot depolarizant p."""
    noisy_circuit = noisy_qnode(circuit_fn, p, n_wires)
    X_t = torch.tensor(Z)
    with torch.no_grad():
        return torch.cat([readout(noisy_circuit(X_t[k:k + chunk], weights)) for k in range(0, len(X_t), chunk)]).numpy()

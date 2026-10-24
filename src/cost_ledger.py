"""Cost ledger în execuții de circuit și shots nominale (nu timp de simulare).

Convenția conservatoare a notebook-ului oficial: un pas de gradient costă 1 + 2P execuții (regula parameter-shift),
iar scorarea costă o execuție per tranzacție.
"""

SHOTS_PER_EXECUTION = 1024
N_PARAMS = 32          # model A și B: (ANSATZ_REPS + 1) * N_QUBITS * 2 = 4 * 4 * 2


def training_executions(epochs, n_train, n_params=N_PARAMS):
    return epochs * n_train * (1 + 2 * n_params)


def validation_executions_per_epoch(epochs, n_val):
    """Validare după fiecare epocă (alegerea epocii cu AUPRC maxim)."""
    return epochs * n_val


def scoring_executions(n_transactions, n_passes=1):
    return n_transactions * n_passes


def shots(executions, shots_per_execution=SHOTS_PER_EXECUTION):
    return executions * shots_per_execution


def vqc_ledger(epochs=30, n_train=590, n_val=56_962, n_test=56_962, n_noise_levels=4, n_params=N_PARAMS):
    """Ledger per seed pentru un VQC antrenat o dată: antrenare, validare pe epoci, evaluare finală val și test."""
    parts = {
        "antrenare": training_executions(epochs, n_train, n_params),
        "validare pe epoci": validation_executions_per_epoch(epochs, n_val),
        "evaluare finală validation": scoring_executions(n_val, n_noise_levels),
        "test final": scoring_executions(n_test, n_noise_levels),
    }
    parts["total"] = sum(parts.values())
    return parts

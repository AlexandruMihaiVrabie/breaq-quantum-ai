"""Metrica „buget de alarme": cele mai multe fraude detectate cu cel mult k alarme false."""
import numpy as np
import pandas as pd

BUDGETS = [3, 5, 10, 50]


def operating_points(y_true, scores):
    """TP și FP pentru fiecare prag distinct (predicție: score >= prag), pragul descrescător."""
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], y_true[order]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    last = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]   # ultimul element din fiecare grup de scoruri egale
    return pd.DataFrame({"threshold": s[last], "TP": tp[last], "FP": fp[last]})


def threshold_at_fp_budget(y_true, scores, k):
    """Pragul cu TP maxim sub FP <= k; la egalitate, pragul cel mai mare. None dacă nu există niciun punct fezabil."""
    points = operating_points(y_true, scores)
    cand = points[points["FP"] <= k]
    if cand.empty:
        return None
    cand = cand[cand["TP"] == cand["TP"].max()]
    return float(cand["threshold"].max())


def counts_at(y_true, scores, threshold):
    """TP, FP, FN, TN, precision, recall la pragul dat (predicție: score >= prag)."""
    pred = scores >= threshold
    tp = int(np.sum(pred & (y_true == 1)))
    fp = int(np.sum(pred & (y_true == 0)))
    fn = int(np.sum(~pred & (y_true == 1)))
    tn = int(np.sum(~pred & (y_true == 0)))
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn)}


def alert_budget_row(y_select, scores_select, y_eval, scores_eval, k):
    """Pragul se alege pe (y_select, scores_select) și se aplică neschimbat pe (y_eval, scores_eval)."""
    thr = threshold_at_fp_budget(y_select, scores_select, k)
    if thr is None:
        return {"k": k, "prag": np.nan, "fezabil": False}
    return {"k": k, "prag": thr, "fezabil": True, **counts_at(y_eval, scores_eval, thr)}


def alert_budget_table(y_select, y_eval, get_scores, models, seeds, noise_levels, budgets=BUDGETS):
    """Un rând per (model, seed, p, k). get_scores(model, seed, p) -> (scores_select, scores_eval).
    Pe validation: y_select = y_eval = y_val și get_scores returnează de două ori scorurile de validation."""
    rows = []
    for model in models:
        for seed in seeds:
            for p in noise_levels:
                s_sel, s_eval = get_scores(model, seed, p)
                for k in budgets:
                    rows.append({"model": model, "seed": seed, "p": p,
                                 **alert_budget_row(y_select, s_sel, y_eval, s_eval, k)})
    return pd.DataFrame(rows)


def max_f1_threshold(y_true, scores):
    """Pragul care maximizează F1 pe (y_true, scores); identic cu choose_threshold din Exp. 2–7."""
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return thresholds[np.argmax(f1[:-1])]

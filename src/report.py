"""Agregări pe seeds: medie ± std (ddof=1) și comparații pereche."""
import numpy as np
import pandas as pd


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def fmt(values, digits=3):
    m, s = mean_std(values)
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def paired_difference(df, metric, model_a, model_b, group_cols=("p",), seed_col="seed", model_col="model"):
    """Δ = metric(model_a) − metric(model_b), pereche pe seed, pentru fiecare grup (implicit pe p).
    Returnează medie, std, numărul de seeds cu Δ > 0 și numărul total de seeds."""
    wide = df.pivot_table(index=[*group_cols, seed_col], columns=model_col, values=metric)
    diff = (wide[model_a] - wide[model_b]).rename("delta").reset_index()
    out = []
    for key, g in diff.groupby(list(group_cols)):
        m, s = mean_std(g["delta"])
        out.append({**dict(zip(group_cols, key if isinstance(key, tuple) else (key,))),
                    "delta_mean": m, "delta_std": s, "seeds_delta_pos": int((g["delta"] > 0).sum()), "n_seeds": len(g)})
    return pd.DataFrame(out)


def summarize(df, metrics, group_cols=("model", "p")):
    """Medie ± std pe seeds pentru fiecare metrică, grupat după group_cols."""
    rows = []
    for key, g in df.groupby(list(group_cols)):
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        for metric in metrics:
            row[f"{metric}_mean"], row[f"{metric}_std"] = mean_std(g[metric])
        rows.append(row)
    return pd.DataFrame(rows)

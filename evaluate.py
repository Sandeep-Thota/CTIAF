"""Evaluation metrics for CTIAF (Section 4.6)."""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score


def intent_metrics(y_true, y_prob, thresh: float = 0.5) -> dict:
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= thresh).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if len(set(y_true.tolist())) > 1:
        out["auc_roc"] = roc_auc_score(y_true, y_prob)
    return out


def trust_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    out = {
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "spearman": float(spearmanr(y_true, y_pred).correlation),
    }
    return out


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(0)
    yt = rng.integers(0, 2, 200); yp = np.clip(yt * 0.6 + rng.random(200) * 0.5, 0, 1)
    print("intent:", intent_metrics(yt, yp))
    print("trust :", trust_metrics(rng.random(200), rng.random(200)))

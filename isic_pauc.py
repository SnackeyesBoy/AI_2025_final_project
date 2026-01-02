"""ISIC 2024 pAUC metric (pAUC above a minimum TPR).

Kaggle ISIC 2024 primary metric: pAUC above 80% TPR (min_tpr=0.80).

Definition:
  pAUC = ∫ max(TPR(FPR) - min_tpr, 0) dFPR
This yields a score in [0, 1-min_tpr] (e.g., max 0.2 when min_tpr=0.8).

Optionally, normalize to [0,1] by dividing by (1 - min_tpr).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve


def isic_pauc_above_tpr(
    y_true,
    y_score,
    *,
    min_tpr: float = 0.80,
    normalize: bool = False,
) -> float:
    """Compute ISIC pAUC-above-TPR.

    Args:
        y_true: Iterable of binary labels (0/1).
        y_score: Iterable of prediction scores (higher = more malignant).
        min_tpr: Minimum TPR threshold (default 0.80 for Kaggle ISIC 2024).
        normalize: If True, return pAUC / (1 - min_tpr) in [0,1].

    Returns:
        float: raw pAUC in [0, 1-min_tpr] (or normalized in [0,1]).
               If y_true has <2 classes, returns NaN (cannot define ROC).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if np.unique(y_true).size < 2:
        return float("nan")

    if not (0.0 < min_tpr < 1.0):
        raise ValueError(f"min_tpr must be in (0,1). Got {min_tpr}.")

    fpr, tpr, _ = roc_curve(y_true, y_score)

    # If the ROC never reaches min_tpr, area above min_tpr is 0 by definition.
    if np.all(tpr < min_tpr):
        return 0.0 if not normalize else 0.0

    # Find first index where tpr >= min_tpr (tpr is non-decreasing).
    idx = int(np.searchsorted(tpr, min_tpr, side="left"))

    if idx == 0:
        fpr_trunc = fpr
        tpr_trunc = tpr
    else:
        fpr0, tpr0 = fpr[idx - 1], tpr[idx - 1]
        fpr1, tpr1 = fpr[idx], tpr[idx]

        # Linear interpolation to get point exactly at tpr=min_tpr.
        if tpr1 == tpr0:
            fpr_at_min_tpr = fpr1
        else:
            fpr_at_min_tpr = fpr0 + (min_tpr - tpr0) * (fpr1 - fpr0) / (tpr1 - tpr0)

        fpr_trunc = np.concatenate([[fpr_at_min_tpr], fpr[idx:]])
        tpr_trunc = np.concatenate([[min_tpr], tpr[idx:]])

    # Integrate (tpr - min_tpr) over FPR.
    pauc = float(np.trapz(tpr_trunc - min_tpr, fpr_trunc))
    pauc = max(pauc, 0.0)

    if normalize:
        denom = (1.0 - min_tpr)
        return pauc / denom if denom > 0 else float("nan")
    return pauc

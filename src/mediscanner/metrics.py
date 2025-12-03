from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


@dataclass
class EvaluationResult:
    accuracy: float
    loss: float
    auc: Optional[float]
    report: Dict[str, Dict[str, float]]
    confusion: np.ndarray


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray],
    loss: float,
) -> EvaluationResult:
    accuracy = float(accuracy_score(labels, predictions))
    auc = None
    if probabilities is not None and probabilities.shape[1] > 1:
        # sklearn's roc_auc_score for multiclass expects the number of columns in
        # probabilities to match the number of classes present in y_true. When the
        # validation split misses some classes, this raises a ValueError. Handle
        # gracefully by skipping AUC in that case.
        try:
            unique = np.unique(labels)
            if len(unique) == probabilities.shape[1]:
                auc = float(roc_auc_score(labels, probabilities, multi_class="ovr"))
            else:
                auc = None
        except Exception:
            auc = None
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    confusion = confusion_matrix(labels, predictions)
    return EvaluationResult(
        accuracy=accuracy,
        loss=loss,
        auc=auc,
        report=report,
        confusion=confusion,
    )

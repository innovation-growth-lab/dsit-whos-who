"""Model evaluation utilities."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from typing import Dict


def log_performance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    stage: str,
    metric_prefix: str = "",
) -> Dict:
    """Calculate performance metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        stage: Stage name (e.g., 'validation', 'test')
        metric_prefix: Prefix to add to metric names

    Returns:
        Dictionary containing computed metrics
    """
    metrics = {}

    # Calculate standard metrics
    metrics[f"{metric_prefix}{stage}_accuracy"] = accuracy_score(y_true, y_pred)
    metrics[f"{metric_prefix}{stage}_precision"] = precision_score(y_true, y_pred)
    metrics[f"{metric_prefix}{stage}_recall"] = recall_score(y_true, y_pred)
    metrics[f"{metric_prefix}{stage}_f1"] = f1_score(y_true, y_pred)
    metrics[f"{metric_prefix}{stage}_roc_auc"] = roc_auc_score(y_true, y_pred_proba)
    metrics[f"{metric_prefix}{stage}_average_precision"] = average_precision_score(
        y_true, y_pred_proba
    )

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics[f"{metric_prefix}{stage}_true_negatives"] = int(tn)
    metrics[f"{metric_prefix}{stage}_false_positives"] = int(fp)
    metrics[f"{metric_prefix}{stage}_false_negatives"] = int(fn)
    metrics[f"{metric_prefix}{stage}_true_positives"] = int(tp)

    # Add derived metrics
    total = tn + fp + fn + tp
    metrics[f"{metric_prefix}{stage}_negative_predictive_value"] = (
        tn / (tn + fn) if (tn + fn) > 0 else 0
    )
    metrics[f"{metric_prefix}{stage}_false_discovery_rate"] = (
        fp / (fp + tp) if (fp + tp) > 0 else 0
    )
    metrics[f"{metric_prefix}{stage}_false_omission_rate"] = (
        fn / (fn + tn) if (fn + tn) > 0 else 0
    )

    # Class proportions in predictions
    metrics[f"{metric_prefix}{stage}_predicted_positive_rate"] = (tp + fp) / total
    metrics[f"{metric_prefix}{stage}_actual_positive_rate"] = (tp + fn) / total

    return metrics

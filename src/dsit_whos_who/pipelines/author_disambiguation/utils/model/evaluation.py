"""Model evaluation utilities."""

import logging
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


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


def analyse_model_performance(
    model,
    x_train,
    x_test,
    y_train,
    y_test,
    feature_names,
    model_type: str,
    params: Dict,
) -> None:
    """analyse performance metrics for a single model

    args:
        model: trained model (pipeline or direct)
        x_train: training features
        x_test: test features
        y_train: training labels
        y_test: test labels
        feature_names: list of feature names
        model_type: name of model type for logging
        params: model training parameters
    """
    # get feature importance
    if hasattr(model, "named_steps"):
        importance = model.named_steps["classifier"].feature_importances_
    else:
        importance = model.feature_importances_

    # create feature importance dataframe
    feature_imp = pd.DataFrame(
        {"feature": feature_names, "importance": importance}
    ).sort_values("importance", ascending=False)

    logger.info("\nmodel type: %s", model_type)
    logger.info("parameters:")
    logger.info("\n- test size: %.2f", params["test_size"])
    logger.info("- random seed: %d", params["random_seed"])
    if model_type == "smote_model" and params["smote"]["enabled"]:
        logger.info("- using SMOTE")

    # log top and bottom features
    logger.info("\ntop 10 most important features:")
    logger.info("\n| feature | importance |" "\n|---------|------------|")
    for _, row in feature_imp.head(10).iterrows():
        logger.info("| %-40s | %10.4f |", row["feature"], row["importance"])

    logger.info("\n5 least important features:")
    logger.info("\n| feature | importance |" "\n|---------|------------|")
    for _, row in feature_imp.tail().iterrows():
        logger.info("| %-40s | %10.4f |", row["feature"], row["importance"])

    # get predictions and probabilities
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    # analyse each split
    for split_name, _, y_true, y_pred in [
        ("training", x_train, y_train, train_pred),
        ("test", x_test, y_test, test_pred),
    ]:
        # compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        logger.info("\n%s results:", split_name)
        logger.info("\nconfusion matrix:")
        logger.info(
            "\n| true\\pred | negative | positive |"
            "\n|-----------|----------|----------|"
            "\n| negative  | %8d | %8d |"
            "\n| positive  | %8d | %8d |",
            cm[0, 0],
            cm[0, 1],
            cm[1, 0],
            cm[1, 1],
        )

        # compute raw metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        # compute balanced metrics
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        total = n_pos + n_neg
        pos_weight = total / (2 * n_pos)
        neg_weight = total / (2 * n_neg)
        sample_weights = np.where(y_true == 1, pos_weight, neg_weight)

        balanced_accuracy = accuracy_score(y_true, y_pred, sample_weight=sample_weights)
        balanced_precision = precision_score(
            y_true, y_pred, sample_weight=sample_weights
        )
        balanced_recall = recall_score(y_true, y_pred, sample_weight=sample_weights)
        balanced_f1 = f1_score(y_true, y_pred, sample_weight=sample_weights)

        logger.info(
            "\nraw metrics:"
            "\n- accuracy: %.3f"
            "\n- precision: %.3f"
            "\n- recall: %.3f"
            "\n- f1: %.3f"
            "\n\nbalanced metrics:"
            "\n- accuracy: %.3f"
            "\n- precision: %.3f"
            "\n- recall: %.3f"
            "\n- f1: %.3f"
            "\n- positive examples: %d (%.1f%%)"
            "\n- negative examples: %d (%.1f%%)",
            accuracy,
            precision,
            recall,
            f1,
            balanced_accuracy,
            balanced_precision,
            balanced_recall,
            balanced_f1,
            n_pos,
            100 * n_pos / total,
            n_neg,
            100 * n_neg / total,
        )

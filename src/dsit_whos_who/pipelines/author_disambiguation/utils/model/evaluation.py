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
    logger.info(
        "\ntop 10 most important features:"
        "\n| feature | importance |"
        "\n|---------|------------|"
    )
    for _, row in feature_imp.head(10).iterrows():
        logger.info("| %-40s | %10.4f |", row["feature"], row["importance"])

        logger.info(
            "\n5 least important features:"
            "\n| feature | importance |"
            "\n|---------|------------|"
        )
    for _, row in feature_imp.tail().iterrows():
        logger.info("| %-40s | %10.4f |", row["feature"], row["importance"])
    # get predictions and probabilities
    train_pred_proba = model.predict_proba(x_train)[:, 1]
    test_pred_proba = model.predict_proba(x_test)[:, 1]

    # analyse each split
    for split_name, _, y_true, y_pred_proba in [
        ("training", x_train, y_train, train_pred_proba),
        ("test", x_test, y_test, test_pred_proba),
    ]:
        logger.info("\n%s results:", split_name)

        # evaluate metrics at different thresholds
        thresholds = list(np.linspace(0.1, 0.9, 9)) + [0.95, 0.99]

        logger.info("\nMetrics at different thresholds:")
        logger.info(
            "\n| threshold | accuracy | precision | recall | f1 | balanced_f1 |"
        )
        logger.info(
            "|-----------|----------|-----------|--------|-------|-------------|"
        )

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)

            # compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)

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

            balanced_f1 = f1_score(y_true, y_pred, sample_weight=sample_weights)

            logger.info(
                "| %.2f | %.3f | %.3f | %.3f | %.3f | %.3f |",
                threshold,
                accuracy,
                precision,
                recall,
                f1,
                balanced_f1,
            )

        # Show confusion matrix at 0.5 threshold for reference
        y_pred = (y_pred_proba >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        logger.info("\nConfusion matrix at threshold=0.5:")
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

        # Show class distribution
        logger.info(
            "\nClass distribution:"
            "\n- positive examples: %d (%.1f%%)"
            "\n- negative examples: %d (%.1f%%)",
            n_pos,
            100 * n_pos / total,
            n_neg,
            100 * n_neg / total,
        )


def evaluate_model_performance_on_full_data(
    gtr_persons: pd.DataFrame,
    gtr_projects: pd.DataFrame,
    matched_authors: pd.DataFrame,
    feature_matrix: pd.DataFrame,
):
    """Evaluate model performance on full data."""

    # Overall coverage statistics
    total_persons = len(gtr_persons)
    matchable_persons = feature_matrix["gtr_id"].nunique()
    matched_persons = len(matched_authors["gtr_id"].unique())

    coverage_rate = matched_persons / total_persons
    matchable_coverage = (
        matched_persons / matchable_persons if matchable_persons > 0 else 0
    )

    logger.info(
        "Overall coverage: %d/%d persons matched (%.1f%%)",
        matched_persons,
        total_persons,
        100 * coverage_rate,
    )
    logger.info(
        "Coverage of matchable persons: %d/%d (%.1f%%)",
        matched_persons,
        matchable_persons,
        100 * matchable_coverage,
    )

    # Coverage for persons with projects
    persons_with_projects = gtr_persons[gtr_persons["projects"].apply(len) > 0]
    total_active = len(persons_with_projects)
    matched_active = len(
        matched_authors[
            matched_authors["gtr_id"].isin(persons_with_projects["person_id"])
        ]["gtr_id"].unique()
    )
    active_coverage = matched_active / total_active if total_active > 0 else 0

    logger.info(
        "Coverage for persons with projects: %d/%d (%.1f%%)",
        matched_active,
        total_active,
        100 * active_coverage,
    )

    # Coverage by grant category
    # Explode persons array in projects
    project_persons = pd.DataFrame(
        [
            (project["grant_category"], person["id"])
            for _, project in gtr_projects.iterrows()
            for person in project["persons"]
        ],
        columns=["grant_category", "person_id"],
    )

    # Group by category and compute stats
    category_stats = (
        project_persons.groupby("grant_category")
        .agg(total=("person_id", "nunique"))
        .assign(
            matched=lambda df: df.index.map(
                lambda cat: len(
                    set(
                        project_persons[project_persons["grant_category"] == cat][
                            "person_id"
                        ]
                    )
                    & set(matched_authors["gtr_id"])
                )
            )
        )
    )

    category_stats["coverage_rate"] = (
        category_stats["matched"] / category_stats["total"]
    )

    logger.info("\nCoverage by grant category:")
    for category, row in category_stats.iterrows():
        logger.info(
            "- %s: %d/%d (%.1f%%)",
            category,
            row["matched"],
            row["total"],
            100 * row["coverage_rate"],
        )

    # Coverage by year
    # Convert start_date to year and explode persons
    project_year_persons = pd.DataFrame(
        [
            (pd.to_datetime(project["start_date"]).year, person["id"])
            for _, project in gtr_projects.iterrows()
            if project["start_date"] is not None
            for person in project["persons"]
        ],
        columns=["year", "person_id"],
    )

    # Group by year and compute stats
    year_stats = (
        project_year_persons.groupby("year")
        .agg(total=("person_id", "nunique"))
        .assign(
            matched=lambda df: df.index.map(
                lambda yr: len(
                    set(
                        project_year_persons[project_year_persons["year"] == yr][
                            "person_id"
                        ]
                    )
                    & set(matched_authors["gtr_id"])
                )
            )
        )
    )

    year_stats["coverage_rate"] = year_stats["matched"] / year_stats["total"]

    logger.info("\nCoverage by project start year:")
    for year, row in year_stats.sort_index().iterrows():
        logger.info(
            "- %d: %d/%d (%.1f%%)",
            year,
            row["matched"],
            row["total"],
            100 * row["coverage_rate"],
        )

    # Prepare results dictionary
    results = {
        "overall_coverage": {
            "total_persons": total_persons,
            "matchable_persons": matchable_persons,
            "matched_persons": matched_persons,
            "coverage_rate": coverage_rate,
            "matchable_coverage_rate": matchable_coverage,
        },
        "active_coverage": {
            "total_active": total_active,
            "matched_active": matched_active,
            "active_coverage_rate": active_coverage,
        },
        "grant_category_coverage": category_stats.to_dict("index"),
        "year_coverage": year_stats.to_dict("index"),
    }

    return results

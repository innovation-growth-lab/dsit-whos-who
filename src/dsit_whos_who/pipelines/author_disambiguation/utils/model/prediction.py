"""Model prediction utilities."""

import logging
from typing import Dict
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _prepare_features_for_prediction(
    feature_matrix: pd.DataFrame,
    feature_names: list,
    scaler: StandardScaler,
) -> np.ndarray:
    """Prepare features for prediction.

    Args:
        feature_matrix: DataFrame with features
        feature_names: List of feature names to use
        scaler: Fitted StandardScaler

    Returns:
        Scaled feature matrix
    """
    # Select features and handle missing values (CHECK)
    x = feature_matrix[feature_names].fillna(0)

    # Scale features
    x_scaled = scaler.transform(x)

    return x_scaled


def predict_matches(
    model_dict: Dict,
    feature_matrix: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Predict matches for author pairs.

    Args:
        model_dict: Dictionary containing model and metadata
        feature_matrix: DataFrame containing features
        threshold: Probability threshold for positive predictions (default: 0.5)

    Returns:
        DataFrame with predictions and probabilities, containing at most one match
        per GtR ID (the one with highest probability above threshold)
    """
    model = model_dict["model"]
    scaler = model_dict["scaler"]
    feature_names = model_dict["feature_names"]

    # Prepare features
    x = _prepare_features_for_prediction(feature_matrix, feature_names, scaler)

    # Get probabilities
    probabilities = model.predict_proba(x)[:, 1]

    # Create results DataFrame
    pred_df = pd.DataFrame(
        {
            "gtr_id": feature_matrix["gtr_id"],
            "oa_id": feature_matrix["oa_id"],
            "match_probability": probabilities,
        }
    )

    # For each gtr_id, keep only the match with highest probability above threshold
    results = []
    for gtr_id, group in pred_df.groupby("gtr_id"):
        max_prob_idx = group["match_probability"].idxmax()
        max_prob = group.loc[max_prob_idx, "match_probability"]

        if max_prob >= threshold:
            results.append(
                {
                    "gtr_id": gtr_id,
                    "oa_id": group.loc[max_prob_idx, "oa_id"],
                    "match_probability": max_prob,
                }
            )

    if not results:
        logger.warning("No matches found above threshold %.2f", threshold)
        return pd.DataFrame(columns=["gtr_id", "oa_id", "match_probability"])

    final_df = pd.DataFrame(results)

    logger.info(
        "Found %d matches above threshold %.2f from %d unique GtR IDs",
        len(final_df),
        threshold,
        len(pred_df["gtr_id"].unique()),
    )

    return final_df

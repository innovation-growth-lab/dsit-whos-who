"""Model prediction utilities."""

import logging
from typing import Dict
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def prepare_features_for_prediction(
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
    x = feature_matrix[feature_names]  # .fillna(0)

    # Scale features
    x_scaled = scaler.transform(x)

    return x_scaled


def predict_matches(
    model_dict: Dict,
    feature_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Predict matches for author pairs.

    Args:
        model_dict: Dictionary containing model and metadata
        feature_matrix: DataFrame containing features

    Returns:
        DataFrame with predictions and probabilities
    """
    # Extract components
    model = model_dict["model"]
    scaler = model_dict["scaler"]
    feature_names = model_dict["feature_names"]

    # Prepare features
    x = prepare_features_for_prediction(feature_matrix, feature_names, scaler)

    # Make predictions
    predictions = model.predict(x)
    probabilities = model.predict_proba(x)[:, 1]

    # Create results DataFrame
    results = pd.DataFrame(
        {
            "gtr_id": feature_matrix["gtr_id"],
            "oa_id": feature_matrix["oa_id"],
            "predicted_match": predictions,
            "match_probability": probabilities,
        }
    )

    return results

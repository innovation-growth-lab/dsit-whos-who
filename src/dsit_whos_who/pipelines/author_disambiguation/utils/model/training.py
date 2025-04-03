"""
Model training utilities for author disambiguation.

This module provides functionality for:
- Training data preparation and preprocessing
- Model training with hyperparameter optimisation
- Class imbalance handling via SMOTE or class weights
- Feature importance analysis
- Cross-validation and performance evaluation

The implementation focuses on XGBoost classifiers with standardised features
and robust evaluation metrics.
"""

# pylint: disable=E0402

import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from .evaluation import log_performance_metrics

logger = logging.getLogger(__name__)


def prepare_training_data(
    feature_matrix: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prepare feature matrix for model training.

    Processes:
    - Feature selection and filtering
    - Missing value handling
    - Label extraction
    - Data type standardisation

    Args:
        feature_matrix: Raw features with labels

    Returns:
        Tuple containing:
        - Processed feature matrix
        - Target labels
        - Feature name list
    """
    # Remove ID columns and label
    feature_cols = [
        col
        for col in feature_matrix.columns
        if col not in ["gtr_id", "oa_id", "is_match"]
    ]

    # Handle missing values
    x = feature_matrix[feature_cols].fillna(0)
    y = feature_matrix["is_match"].values

    return x.values, y, feature_cols


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    params: Dict,
    use_smote: bool = True,
    metric_prefix: str = "",
) -> Dict:
    """Train and optimise author disambiguation model.

    Implements:
    - Feature standardisation
    - Class imbalance handling
    - Hyperparameter optimisation via grid search
    - Cross-validation with stratification
    - Feature importance analysis
    - Performance evaluation

    Args:
        x: Standardised feature matrix
        y: Binary match labels
        feature_names: Feature column names
        params: Training configuration
        use_smote: Whether to use SMOTE resampling
        metric_prefix: Metric name prefix

    Returns:
        Dictionary with:
        - Trained model and scaler
        - Best parameters
        - Performance metrics
        - Feature importance
        - Cross-validation results
    """
    # Split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=params["test_size"],
        random_state=params["random_seed"],
        stratify=y,
    )

    # Scale features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Calculate scale_pos_weight for XGBoost (for non-SMOTE option)
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    # Get base parameters from config
    base_params = params["base_params"]

    if use_smote:
        # Create pipeline with SMOTE
        steps = [
            ("smote", SMOTE(random_state=params["random_seed"])),
            (
                "classifier",
                XGBClassifier(
                    random_state=params["random_seed"],
                    tree_method="hist",
                    enable_categorical=False,
                ),
            ),
        ]
        base_model = Pipeline(steps)

        # Create parameter grid for pipeline
        param_grid = {
            "smote__k_neighbors": params["smote"]["k_neighbors"],
        }
        # Add classifier parameters with proper prefix
        for key, value in base_params.items():
            param_grid[f"classifier__{key}"] = value
    else:
        # Let XGBoost handle missing values natively
        base_model = XGBClassifier(
            random_state=params["random_seed"],
            tree_method="hist",
            enable_categorical=False,
            scale_pos_weight=scale_pos_weight,  # handle class imbalance
        )
        param_grid = base_params

    # set up cross-validation
    cv = StratifiedKFold(
        n_splits=params["cv"]["n_splits"],
        shuffle=params["cv"]["shuffle"],
        random_state=params["random_seed"],
    )

    # instantiate GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring=params["grid_search"]["scoring"],
        n_jobs=params["grid_search"]["n_jobs"],
        verbose=params["grid_search"]["verbose"],
    )

    # collect model-specific parameters
    model_params = {
        f"{metric_prefix}resampling_strategy": (
            "SMOTE" if use_smote else "scale_pos_weight"
        ),
        f"{metric_prefix}train_size": len(x_train),
        f"{metric_prefix}test_size": len(x_test),
    }

    # fit GridSearchCV for either class
    logger.info(
        "Starting GridSearchCV with %s", "SMOTE" if use_smote else "scale_pos_weight"
    )
    grid_search.fit(x_train_scaled, y_train)

    # Get best model
    best_model = grid_search.best_estimator_

    # Collect best parameters with prefix
    best_params = {
        f"{metric_prefix}{k}": v for k, v in grid_search.best_params_.items()
    }

    # feature importance (extract from pipeline if using SMOTE!)
    if use_smote:
        feature_importance = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": best_model.named_steps["classifier"].feature_importances_,
            }
        ).sort_values("importance", ascending=False)
    else:
        feature_importance = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": best_model.feature_importances_,
            }
            # https://xgboost.readthedocs.io/en/stable/python/python_api.html
        ).sort_values("importance", ascending=False)

    # evaluate on test set
    test_pred = best_model.predict(x_test_scaled)
    test_pred_proba = best_model.predict_proba(x_test_scaled)[:, 1]
    test_metrics = log_performance_metrics(
        y_test, test_pred, test_pred_proba, "test", metric_prefix=metric_prefix
    )

    return {
        "model": best_model,
        "scaler": scaler,
        "feature_names": feature_names,
        "best_params": grid_search.best_params_,
        "cv_results": grid_search.cv_results_,
        "parameters": {**model_params, **best_params},
        "metrics": test_metrics,
        "feature_importance": feature_importance.to_dict(),
    }

"""Model training utilities."""

# pylint: disable=E0402

import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import mlflow
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
    """Prepare data for training.

    Args:
        feature_matrix: DataFrame with features and labels

    Returns:
        Tuple containing:
        - Feature matrix
        - Target labels
        - List of feature names
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
    """Train model with hyperparameter tuning.

    Args:
        x: Feature matrix
        y: Target labels
        feature_names: List of feature names
        params: Dictionary containing model training parameters
        use_smote: Whether to use SMOTE for resampling
        metric_prefix: Prefix to add to metric names

    Returns:
        Dictionary containing trained model, metrics and metadata
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
                    tree_method="hist",  # For faster training
                    enable_categorical=False,  # All features are numeric
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
        base_model = XGBClassifier(
            random_state=params["random_seed"],
            tree_method="hist",
            enable_categorical=False,
            scale_pos_weight=scale_pos_weight,  # Handle class imbalance
        )
        param_grid = base_params

    # Set up cross-validation
    cv = StratifiedKFold(
        n_splits=params["cv"]["n_splits"],
        shuffle=params["cv"]["shuffle"],
        random_state=params["random_seed"],
    )

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring=params["grid_search"]["scoring"],
        n_jobs=params["grid_search"]["n_jobs"],
        verbose=params["grid_search"]["verbose"],
    )

    # Collect model-specific parameters
    model_params = {
        f"{metric_prefix}resampling_strategy": (
            "SMOTE" if use_smote else "scale_pos_weight"
        ),
        f"{metric_prefix}train_size": len(x_train),
        f"{metric_prefix}test_size": len(x_test),
    }

    # Fit GridSearchCV
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

    # Feature importance (extract from pipeline if using SMOTE)
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
        ).sort_values("importance", ascending=False)

    # Evaluate on test set
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

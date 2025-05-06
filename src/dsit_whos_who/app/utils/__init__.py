"""Utility functions specific to the Streamlit application."""

from .kedro_init import setup_kedro_environment, get_kedro_context
from .cache_utils import (
    persistent_cached_value,
    get_model_cache,
    store_in_model_cache,
    get_from_model_cache,
    clear_model_cache,
)
from .author_cache import (
    cached_search_and_process,
    cached_compute_features,
    load_model_dict,
    load_disambiguation_params,
)

__all__ = [
    # Kedro utilities
    "setup_kedro_environment",
    "get_kedro_context",
    # Cache utilities
    "persistent_cached_value",
    "get_model_cache",
    "store_in_model_cache",
    "get_from_model_cache",
    "clear_model_cache",
    # Author cache utilities
    "cached_search_and_process",
    "cached_compute_features",
    "load_model_dict",
    "load_disambiguation_params",
]

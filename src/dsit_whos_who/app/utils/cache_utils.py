"""Caching utilities for the Streamlit application."""

import logging
import functools
from typing import Any, Callable, Dict, TypeVar
import streamlit as st

T = TypeVar("T")
log = logging.getLogger(__name__)


def persistent_cached_value(
    key: str, ttl: int = None, show_spinner: bool = True, max_entries: int = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for creating persistent cached values in Streamlit.

    This uses st.cache_data with a consistent key to ensure persistence
    across Streamlit reruns while allowing TTL and other cache parameters.

    Args:
        key: Unique key for this cached value in the Streamlit session state
        ttl: Time to live in seconds, None means cache forever
        show_spinner: Whether to display a spinner when computing the value
        max_entries: Maximum number of entries to keep in the cache

    Returns:
        Decorated function that will use cached values when available
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @st.cache_data(ttl=ttl, show_spinner=show_spinner, max_entries=max_entries)
        def cached_func(*args, **kwargs) -> T:
            try:
                log.info("Computing cached value for key: %s", key)
                return func(*args, **kwargs)
            except Exception as e:
                log.error("Error computing cached value for %s: %s", key, e)
                raise

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return cached_func(*args, **kwargs)

        return wrapper

    return decorator


def get_model_cache(model_key: str = "model_cache") -> Dict[str, Any]:
    """Get or create a model cache dictionary in the session state.

    Args:
        model_key: Key to use for the model cache in the session state

    Returns:
        Dictionary to store model-related cached objects
    """
    if model_key not in st.session_state:
        st.session_state[model_key] = {}

    return st.session_state[model_key]


def store_in_model_cache(key: str, value: Any) -> None:
    """Store a value in the model cache.

    Args:
        key: Key to store the value under
        value: Value to store
    """
    model_cache = get_model_cache()
    model_cache[key] = value
    log.info("Stored value in model cache with key: %s", key)


def get_from_model_cache(key: str, default: Any = None) -> Any:
    """Get a value from the model cache.

    Args:
        key: Key to retrieve
        default: Default value to return if key is not in cache

    Returns:
        The cached value or the default
    """
    model_cache = get_model_cache()
    value = model_cache.get(key, default)
    return value


def clear_model_cache() -> None:
    """Clear the entire model cache."""
    model_cache = get_model_cache()
    model_cache.clear()
    log.info("Cleared model cache")

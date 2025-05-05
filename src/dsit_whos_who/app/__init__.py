"""DSIT Who's Who Web Application module."""

from .author_search import AuthorSearchApp
from .utils.kedro_init import setup_kedro_environment, get_kedro_context

__all__ = [
    "AuthorSearchApp",
    "setup_kedro_environment",
    "get_kedro_context",
] 
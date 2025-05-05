"""Services for the Streamlit application (data fetching, processing, etc.)."""

from .author_processing import (
    search_and_extract_features,
    OpenAlexFetcher,
    AuthorProcessor,
    FeatureComputer,
)

__all__ = [
    "search_and_extract_features",
    "OpenAlexFetcher",
    "AuthorProcessor",
    "FeatureComputer",
]

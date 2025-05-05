"""Data models (Pydantic, dataclasses) for the Streamlit app."""

from typing import List, Optional, Any
from pydantic import BaseModel, Field


class AuthorFeatures(BaseModel):
    """Represents the computed features for an OpenAlex author candidate."""

    oa_id: str = Field(..., description="OpenAlex ID")
    gtr_id: str = Field(..., description="Placeholder GtR ID used during computation")
    # add fields for all computed features (name, topic, inst, pub, meta)
    # example:
    # name_similarity: float
    # topic_overlap: float
    # ... more features ...
    feature_placeholder: int = Field(..., description="Remove this placeholder")

    # include some raw OA data for display purposes?
    display_name: Optional[str] = None
    institution: Optional[str] = None  # last known or most prominent?
    orcid: Optional[str] = None
    works_count: Optional[int] = None
    cited_by_count: Optional[int] = None

    class Config:
        """Config for the AuthorFeatures model."""

        extra = "allow"  # allow extra fields if feature set is large/dynamic


# you might also want a model for the parsed openalex data before feature eng.
class OpenAlexAuthorData(BaseModel):
    """Represents the parsed OpenAlex author data before feature engineering."""

    id: str
    orcid: Optional[str]
    display_name: Optional[str]
    # ... add other fields from parse_author_results ...
    affiliations: List[Any] = []  # define more specific model later
    topics: List[Any] = []  # define more specific model later
    works_count: Optional[int] = None
    cited_by_count: Optional[int] = None
    h_index: Optional[int] = None
    i10_index: Optional[int] = None
    counts_by_year: List[Any] = []

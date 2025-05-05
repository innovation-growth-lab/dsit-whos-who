"""Data models (Pydantic, dataclasses) for the Streamlit app."""

from typing import List, Optional, Any
from pydantic import BaseModel, Field

# Example - Define more specific models as needed


class AuthorFeatures(BaseModel):
    """Represents the computed features for an OpenAlex author candidate."""

    oa_id: str = Field(..., description="OpenAlex ID")
    gtr_id: str = Field(..., description="Placeholder GtR ID used during computation")
    # Add fields for all computed features (name, topic, inst, pub, meta)
    # Example:
    # name_similarity: float
    # topic_overlap: float
    # ... more features ...
    feature_placeholder: int = Field(..., description="Remove this placeholder")

    # Include some raw OA data for display purposes?
    display_name: Optional[str] = None
    institution: Optional[str] = None  # Last known or most prominent?
    orcid: Optional[str] = None
    works_count: Optional[int] = None
    cited_by_count: Optional[int] = None

    class Config:
        """Config for the AuthorFeatures model."""

        extra = "allow"  # Allow extra fields if feature set is large/dynamic


# You might also want a model for the parsed OpenAlex data before feature eng.
class OpenAlexAuthorData(BaseModel):
    """Represents the parsed OpenAlex author data before feature engineering."""

    id: str
    orcid: Optional[str]
    display_name: Optional[str]
    # ... add other fields from parse_author_results ...
    affiliations: List[Any] = []  # Define more specific model later
    topics: List[Any] = []  # Define more specific model later
    works_count: Optional[int] = None
    cited_by_count: Optional[int] = None
    h_index: Optional[int] = None
    i10_index: Optional[int] = None
    counts_by_year: List[Any] = []

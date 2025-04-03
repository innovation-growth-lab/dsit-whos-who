"""
Publication overlap features.

This module provides:
- Publication overlap metrics computation between GTR and OpenAlex
- Co-authorship analysis
- Publication count comparison
"""

from typing import List
import numpy as np


def compute_publication_features(
    candidate_id: str, gtr_publications: List[str], project_authors: List[List[str]]
) -> dict:
    """Compute publication overlap features.

    Args:
        gtr_publications: List of publication IDs from GTR projects
        project_authors: List of [author_id, count] pairs from project publications

    Returns:
        Dictionary of publication overlap features:
        - publication_coverage: Author's publications / total project publications
        - author_proportion: Author's publication count / sum of all author counts
    """
    if len(gtr_publications) == 0 or len(project_authors) == 0:
        return {
            "publication_coverage": np.nan,
            "author_proportion": np.nan,
        }

    # Convert project_authors to numpy array if not already
    if not isinstance(project_authors, np.ndarray):
        project_authors = np.array(project_authors)

    # Get author IDs and counts
    author_ids = [x[0] for x in project_authors]
    counts = [int(x[1]) for x in project_authors]

    # find the candidate author's count
    author_mask = np.array(author_ids) == candidate_id
    author_count = np.array(counts)[author_mask].sum() if any(author_mask) else 0

    # Calculate metrics
    total_publications = len(gtr_publications)
    total_author_counts = sum(counts)

    publication_coverage = (
        author_count / total_publications if total_publications > 0 else 0.0
    )
    author_proportion = (
        author_count / total_author_counts if total_author_counts > 0 else 0.0
    )

    return {
        "publication_coverage": float(publication_coverage),
        "author_proportion": float(author_proportion),
    }

"""
Metadata-based feature engineering for author disambiguation.
"""


def compute_metadata_features(
    works_count: int, cited_by_count: int, h_index: int, i10_index: int
) -> dict:
    """Compute metadata features from author metrics.

    Args:
        works_count: Total number of works
        cited_by_count: Total citation count
        h_index: Author's h-index
        i10_index: Author's i10-index

    Returns:
        Dictionary of metadata features
    """
    return {
        "works_count": works_count,
        "cited_by_count": cited_by_count,
        "h_index": h_index,
        "i10_index": i10_index,
        "citations_per_work": cited_by_count / works_count if works_count > 0 else 0.0,
    }

"""Name similarity features."""

from typing import List
import numpy as np
from textdistance import levenshtein, jaro_winkler
from rapidfuzz.fuzz import token_set_ratio


def compute_name_features(
    gtr_name: str, oa_name: str, oa_alternatives: List[str]
) -> dict:
    """Compute name similarity features between GTR and OpenAlex names.

    Args:
        gtr_name: Full name from GTR
        oa_name: Primary display name from OpenAlex
        oa_alternatives: List of alternative names from OpenAlex

    Returns:
        Dictionary of name similarity features
    """
    # Split GTR name
    gtr_parts = gtr_name.lower().split()
    gtr_first = gtr_parts[0] if gtr_parts else ""
    gtr_surname = gtr_parts[-1] if gtr_parts else ""

    features = {}

    # process display name
    display_parts = oa_name.lower().split()
    display_first = display_parts[0] if display_parts else ""
    display_surname = display_parts[-1] if display_parts else ""

    features["display_lev"] = float(
        levenshtein.normalized_similarity(gtr_name, oa_name)
    )
    features["display_jw"] = float(jaro_winkler.similarity(gtr_name, oa_name))
    features["display_token"] = float(token_set_ratio(gtr_name, oa_name) / 100)

    # Binary matches for display name
    features["surname_match"] = int(gtr_surname == display_surname)
    features["first_initial_match"] = int(
        gtr_first[0] == display_first[0] if gtr_first and display_first else 0
    )
    features["full_first_match"] = int(gtr_first == display_first)

    # process alternative names if they exist
    if len(oa_alternatives) > 0:
        alt_lev_scores = []
        alt_jw_scores = []
        alt_token_scores = []

        for name in oa_alternatives:
            alt_lev_scores.append(levenshtein.normalized_similarity(gtr_name, name))
            alt_jw_scores.append(jaro_winkler.similarity(gtr_name, name))
            alt_token_scores.append(token_set_ratio(gtr_name, name) / 100)

        # Mean alternative scores
        features["alt_lev_mean"] = float(np.mean(alt_lev_scores))
        features["alt_jw_mean"] = float(np.mean(alt_jw_scores))
        features["alt_token_mean"] = float(np.mean(alt_token_scores))

        # Max alternative scores
        features["alt_lev_max"] = float(max(alt_lev_scores))
        features["alt_jw_max"] = float(max(alt_jw_scores))
        features["alt_token_max"] = float(max(alt_token_scores))
    else:
        # If no alternatives, set scores to nan
        features["alt_lev_mean"] = np.nan
        features["alt_jw_mean"] = np.nan
        features["alt_token_mean"] = np.nan
        features["alt_lev_max"] = np.nan
        features["alt_jw_max"] = np.nan
        features["alt_token_max"] = np.nan

    return features

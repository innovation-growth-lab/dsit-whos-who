"""Institution similarity features."""

from typing import List
import numpy as np
from rapidfuzz.fuzz import token_set_ratio
from textdistance import jaro_winkler


def compute_institution_features(
    gb_affiliation_proportion: float,
    has_gb_affiliation: bool,
    has_gb_associated: bool,
    gtr_institution: str,
    oa_institutions: List[str],
    associated_institutions: List[str],
) -> dict:
    """Compute institution similarity features.

    Args:
        gtr_institution: Primary institution name from GTR
        oa_institutions: List of institution names from OpenAlex
        associated_institutions: List of associated institution names

    Returns:
        Dictionary of institution similarity features
    """
    features = {}

    if not gtr_institution or len(oa_institutions) == 0:
        return {
            "inst_jw_max": np.nan,
            "inst_token_max": np.nan,
            "inst_child_jw_max": np.nan,
            "inst_child_token_max": np.nan,
            "gb_affiliation_proportion": np.nan,
            "has_gb_affiliation": np.nan,
            "has_gb_associated": np.nan,
        }

    # Main institution similarities
    jw_scores = [
        jaro_winkler.similarity(gtr_institution.lower(), inst.lower())
        for inst in oa_institutions
    ]
    token_scores = [
        token_set_ratio(gtr_institution.lower(), inst.lower()) / 100
        for inst in oa_institutions
    ]

    features["inst_jw_max"] = max(jw_scores) if jw_scores else 0.0
    features["inst_token_max"] = max(token_scores) if token_scores else 0.0

    features["gb_affiliation_proportion"] = gb_affiliation_proportion
    features["has_gb_affiliation"] = float(has_gb_affiliation)

    # Associated institution similarities
    if len(associated_institutions) > 0:
        child_jw_scores = [
            jaro_winkler.similarity(gtr_institution.lower(), inst.lower())
            for inst in associated_institutions
        ]
        child_token_scores = [
            token_set_ratio(gtr_institution.lower(), inst.lower()) / 100
            for inst in associated_institutions
        ]

        features["inst_child_jw_max"] = max(child_jw_scores)
        features["inst_child_token_max"] = max(child_token_scores)
        features["has_gb_associated"] = float(has_gb_associated)
    else:
        features["inst_child_jw_max"] = np.nan
        features["inst_child_token_max"] = np.nan
        features["has_gb_associated"] = np.nan

    return features

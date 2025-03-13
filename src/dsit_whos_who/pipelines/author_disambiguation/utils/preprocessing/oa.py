"""OpenAlex preprocessing utilities."""

from typing import List, Tuple
import logging
import numpy as np
logger = logging.getLogger(__name__)


def process_affiliations(
    affiliations: List,
) -> Tuple[List[str], List[str], bool, float]:
    """Process author affiliations to extract institutions and GB metrics.

    Args:
        affiliations: List of affiliation lists, each containing
            [id, name, country_code, type, years]

    Returns:
        Tuple containing:
        - List of institution names
        - List of institution IDs
        - Boolean indicating if any affiliation is GB
        - Proportion of GB affiliations
    """
    if not affiliations or not isinstance(affiliations, list):
        return [], [], False, 0.0

    # Extract institution names and IDs
    inst_names = [aff[1] for aff in affiliations if len(aff) > 1]
    inst_ids = [aff[0] for aff in affiliations if len(aff) > 0]

    # Calculate GB metrics
    gb_affiliations = [aff[2] == "GB" for aff in affiliations if len(aff) > 2]
    has_gb = any(gb_affiliations)
    gb_proportion = sum(gb_affiliations) / len(affiliations) if affiliations else 0.0

    return inst_names, inst_ids, has_gb, gb_proportion


def get_associated_institutions(
    inst_ids: List[str], institutions_dict: dict
) -> Tuple[List[str], bool]:
    """Get associated institution information for a list of institution IDs.

    Args:
        inst_ids: List of institution IDs
        institutions_dict: Dictionary mapping institution IDs to associated institutions

    Returns:
        Tuple containing:
        - List of associated institution names
        - Boolean indicating if any associated institution is in GB
    """
    associated_names = []
    has_gb_associated = False

    if not isinstance(inst_ids, list):
        return [], False

    for inst_id in inst_ids:
        associated = institutions_dict.get(inst_id)
        if not isinstance(associated, np.ndarray):
            continue
            
        for assoc in associated:
            associated_names.append(assoc[1])
            if assoc[2] == "GB":
                has_gb_associated = True

    return associated_names, has_gb_associated

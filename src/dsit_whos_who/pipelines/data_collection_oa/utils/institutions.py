"""
Institution data processing utilities for OpenAlex API.
Handles institution parsing and relationship mapping.
"""

import logging
from typing import Dict, List
import pandas as pd

logger = logging.getLogger(__name__)


def parse_institution_results(response: List[Dict]) -> List[Dict]:
    """Extract and normalise institution data from OpenAlex API response.

    Processes:
    - Basic institution metadata (ID)
    - Associated institutions and relationships

    Args:
        response: Raw API response containing institution data

    Returns:
        List of normalised institution records
    """
    parsed_response = []
    for institution in response:
        parsed_institution = {
            "id": institution.get("id", "").replace("https://openalex.org/", ""),
            "associated_institutions": [
                [
                    assoc["id"].replace("https://openalex.org/", ""),
                    assoc["display_name"],
                    assoc.get("country_code"),
                    assoc.get("relationship"),
                ]
                for assoc in institution.get("associated_institutions", [])
            ],
        }
        parsed_response.append(parsed_institution)

    return parsed_response


def json_loader_institutions(data: List[List[Dict]]) -> pd.DataFrame:
    """Transform batched institution data into structured DataFrame.

    Args:
        data: Batched institution records from OpenAlex

    Returns:
        DataFrame with normalised institution data
    """
    output = []
    for batch in data:
        df = pd.DataFrame(batch)
        if not df.empty:
            output.append(df)

    return pd.concat(output) if output else pd.DataFrame()

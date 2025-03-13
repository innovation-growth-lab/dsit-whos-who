"""Institution-specific utilities for OpenAlex data collection."""

import logging
from typing import Dict, List
import pandas as pd

logger = logging.getLogger(__name__)


def parse_institution_results(response: List[Dict]) -> List[Dict]:
    """Parses OpenAlex API institution response.

    Args:
        response (List[Dict]): The response from the OpenAlex API.

    Returns:
        List[Dict]: List of dictionaries containing parsed institution information.
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
    """Load institutions JSON data into a DataFrame.

    Args:
        data (List[List[Dict]]): The institutions JSON data in batches.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    output = []
    for batch in data:
        df = pd.DataFrame(batch)
        if not df.empty:
            output.append(df)

    return pd.concat(output) if output else pd.DataFrame()

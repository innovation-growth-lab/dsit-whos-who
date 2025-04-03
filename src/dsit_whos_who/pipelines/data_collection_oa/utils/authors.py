"""
Author data processing utilities for OpenAlex API.
Handles author parsing, name matching, and data transformation.
"""

import logging
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher
import pandas as pd

logger = logging.getLogger(__name__)


def _normalise_name(name: str) -> str:
    """Convert name to lowercase alphanumeric for comparison."""
    return "".join(c.lower() for c in name if c.isalnum())


def _name_similarity(name1: str, name2: str) -> float:
    """Calculate string similarity ratio between two names."""
    return SequenceMatcher(None, _normalise_name(name1), _normalise_name(name2)).ratio()


def _find_best_name_match(display_name: str, gtr_names: List[str]) -> Tuple[str, float]:
    """Match OpenAlex display name to closest GTR name variant.

    Args:
        display_name: Author name from OpenAlex
        gtr_names: Candidate names from GTR database

    Returns:
        Best matching GTR name and similarity score
    """
    if not gtr_names:
        return "", 0.0

    similarities = [(name, _name_similarity(display_name, name)) for name in gtr_names]
    best_match = max(similarities, key=lambda x: x[1])
    return best_match


def parse_author_results(
    response: List[Dict],
    gtr_author_names: Optional[str] = None,
) -> List[Dict]:
    """Extract and normalise author data from OpenAlex API response.

    Processes:
    - Basic author metadata (ID, name, metrics)
    - Institutional affiliations and years
    - Research topics and hierarchies
    - Publication/citation counts by year
    - Name matching with GTR database

    Args:
        response: Raw API response containing author data
        gtr_author_names: Pipe-separated GTR names for matching

    Returns:
        List of normalised author records
    """
    parsed_response = []
    for author in response:
        # Only include important fields
        parsed_author = {
            "id": author.get("id", "").replace("https://openalex.org/", ""),
            "orcid": author.get("orcid")
            and author.get("orcid").replace("https://orcid.org/", ""),
            "display_name": author.get("display_name", ""),
            "display_name_alternatives": author.get("display_name_alternatives", []),
            "works_count": author.get("works_count", 0),
            "cited_by_count": author.get("cited_by_count", 0),
            "h_index": author.get("summary_stats", {}).get("h_index", 0),
            "i10_index": author.get("summary_stats", {}).get("i10_index", 0),
        }

        # Process affiliations
        parsed_author["affiliations"] = [
            [
                aff["institution"]["id"].replace("https://openalex.org/", ""),
                aff["institution"]["display_name"],
                aff["institution"]["country_code"],
                aff["institution"]["type"],
                (
                    ",".join([str(y) for y in aff["years"]])
                    if isinstance(aff["years"], list)
                    else str(aff["years"])
                ),
            ]
            for aff in author.get("affiliations", [])
        ]

        # Process last known institutions
        parsed_author["last_known_institutions"] = [
            [
                inst["id"].replace("https://openalex.org/", ""),
                inst["display_name"],
                inst["country_code"],
                inst["type"],
            ]
            for inst in author.get("last_known_institutions", [])
        ]

        # Process topics
        parsed_author["topics"] = [
            [
                topic["id"].replace("https://openalex.org/", ""),
                topic["display_name"],
                str(topic["count"]),
                topic["subfield"]["id"].replace("https://openalex.org/", ""),
                topic["subfield"]["display_name"],
                topic["field"]["id"].replace("https://openalex.org/", ""),
                topic["field"]["display_name"],
                topic["domain"]["id"].replace("https://openalex.org/", ""),
                topic["domain"]["display_name"],
            ]
            for topic in author.get("topics", [])
        ]

        # Process counts by year at parse time
        parsed_author["counts_by_year"] = [
            [int(year["year"]), int(year["works_count"]), int(year["cited_by_count"])]
            for year in author.get("counts_by_year", [])
        ]

        # Add GTR name matching if we have GTR names
        if gtr_author_names:
            gtr_names = gtr_author_names.split("|")
            display_name = author.get("display_name", "")
            matched_name, similarity = _find_best_name_match(display_name, gtr_names)
            parsed_author["gtr_author_name"] = matched_name
            parsed_author["name_match_score"] = similarity

        parsed_response.append(parsed_author)

    return parsed_response


def json_loader_authors(
    data: List[List[Dict]], include_match_info: bool = False
) -> pd.DataFrame:
    """Transform batched author data into structured DataFrame.

    Args:
        data: Batched author records from OpenAlex
        include_match_info: Whether to retain GTR name matching columns

    Returns:
        DataFrame with normalised author data
    """
    output = []
    for batch in data:
        df = pd.DataFrame(batch)
        if df.empty:
            continue

        # Remove matching columns if not needed
        if not include_match_info:
            df = df.drop(
                columns=["gtr_author_name", "name_match_score"], errors="ignore"
            )

        output.append(df)

    return pd.concat(output) if output else pd.DataFrame()

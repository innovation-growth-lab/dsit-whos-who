"""Author-specific utilities for OpenAlex data collection."""

import logging
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher
import pandas as pd

logger = logging.getLogger(__name__)


def _normalise_name(name: str) -> str:
    """Normalise a name for comparison by lowercasing and removing punctuation."""
    return "".join(c.lower() for c in name if c.isalnum())


def _name_similarity(name1: str, name2: str) -> float:
    """Calculate similarity between two names using SequenceMatcher."""
    return SequenceMatcher(None, _normalise_name(name1), _normalise_name(name2)).ratio()


def _find_best_name_match(display_name: str, gtr_names: List[str]) -> Tuple[str, float]:
    """Find the closest matching GTR name for a given OpenAlex display name.
    
    Args:
        display_name (str): The display name from OpenAlex
        gtr_names (List[str]): List of author names from GTR
        
    Returns:
        Tuple[str, float]: The best matching GTR name and the similarity score
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
    """Parses OpenAlex API author search response and matches results to GTR author names.

    Args:
        response (List[Dict]): The response from the OpenAlex API.
        gtr_author_names (str): Pipe-separated list of GTR author names to match against.

    Returns:
        List[Dict]: List of dictionaries containing parsed author information with GTR matches.
    """
    
    parsed_response = []
    for author in response:

        parsed_author = {
            "id": author.get("id", "").replace("https://openalex.org/", ""),
            "orcid": author.get("orcid") and author.get("orcid").replace("https://orcid.org/", ""),
            "display_name": author.get("display_name", ""),
            "display_name_alternatives": author.get("display_name_alternatives", []),
            "works_count": author.get("works_count", 0),
            "cited_by_count": author.get("cited_by_count", 0),
            "h_index": author.get("summary_stats", {}).get("h_index", 0),
            "i10_index": author.get("summary_stats", {}).get("i10_index", 0),
            "affiliations": [
                {
                    "institution_id": aff["institution"]["id"].replace(
                        "https://openalex.org/", ""
                    ),
                    "institution_name": aff["institution"]["display_name"],
                    "country_code": aff["institution"]["country_code"],
                    "type": aff["institution"]["type"],
                    "years": aff["years"],
                }
                for aff in author.get("affiliations", [])
            ],
            "last_known_institutions": [
                {
                    "institution_id": inst["id"].replace("https://openalex.org/", ""),
                    "institution_name": inst["display_name"],
                    "country_code": inst["country_code"],
                    "type": inst["type"],
                }
                for inst in author.get("last_known_institutions", [])
            ],
            "topics": [
                {
                    "id": topic["id"].replace("https://openalex.org/", ""),
                    "display_name": topic["display_name"],
                    "count": topic["count"],
                    "subfield": {
                        "id": topic["subfield"]["id"].replace(
                            "https://openalex.org/", ""
                        ),
                        "display_name": topic["subfield"]["display_name"],
                    },
                    "field": {
                        "id": topic["field"]["id"].replace("https://openalex.org/", ""),
                        "display_name": topic["field"]["display_name"],
                    },
                    "domain": {
                        "id": topic["domain"]["id"].replace(
                            "https://openalex.org/", ""
                        ),
                        "display_name": topic["domain"]["display_name"],
                    },
                }
                for topic in author.get("topics", [])
            ],
            "counts_by_year": author.get("counts_by_year", []),
        }
        
        # Add GTR name matching if we have GTR names
        if gtr_author_names:
            gtr_names = gtr_author_names.split("|")
            display_name = author.get("display_name", "")
            matched_name, similarity = _find_best_name_match(display_name, gtr_names)
            parsed_author["gtr_author_name"] = matched_name
            parsed_author["name_match_score"] = similarity
        
        parsed_response.append(parsed_author)
    
    return parsed_response


def json_loader_authors(data: List[List[Dict]], include_match_info: bool = False) -> pd.DataFrame:
    """
    Load authors JSON data, transform it into a DataFrame, and wrangle data.

    Args:
        data (List[List[Dict]]): The authors JSON data in batches.
        include_match_info (bool): Whether to include name matching columns.

    Returns:
        pandas.DataFrame: The transformed DataFrame.
    """
    output = []
    for batch in data:
        json_data = [
            {
                k: v
                for k, v in item.items()
                if k
                in [
                    "id",
                    "orcid",
                    "display_name",
                    "display_name_alternatives",
                    "works_count",
                    "cited_by_count",
                    "h_index",
                    "i10_index",
                    "affiliations",
                    "last_known_institutions",
                    "topics",
                    "counts_by_year",
                    "gtr_author_name",
                    "name_match_score",
                ]
            }
            for item in batch
        ]

        df = pd.DataFrame(json_data)
        if df.empty:
            continue

        # Process affiliations
        df["affiliations"] = df["affiliations"].apply(
            lambda x: (
                [
                    [
                        aff["institution_id"],
                        aff["institution_name"],
                        aff["country_code"],
                        aff["type"],
                        (
                            ",".join([str(y) for y in aff["years"]])
                            if isinstance(aff["years"], list)
                            else str(aff["years"])
                        ),
                    ]
                    for aff in x
                ]
                if x
                else None
            )
        )

        # Process last known institutions
        df["last_known_institutions"] = df["last_known_institutions"].apply(
            lambda x: (
                [
                    [
                        str(inst["institution_id"]),
                        str(inst["institution_name"]),
                        str(inst["country_code"]),
                        str(inst["type"]),
                    ]
                    for inst in x
                ]
                if x
                else None
            )
        )

        # Process topics
        df["topics"] = df["topics"].apply(
            lambda x: (
                [
                    [
                        topic["id"],
                        topic["display_name"],
                        str(topic["count"]),
                        topic["subfield"]["id"],
                        topic["subfield"]["display_name"],
                        topic["field"]["id"],
                        topic["field"]["display_name"],
                        topic["domain"]["id"],
                        topic["domain"]["display_name"],
                    ]
                    for topic in x
                ]
                if x
                else None
            )
        )

        # Process counts by year
        df["counts_by_year"] = df["counts_by_year"].apply(
            lambda x: (
                [
                    [
                        int(year["year"]),
                        int(year["works_count"]),
                        int(year["cited_by_count"]),
                    ]
                    for year in x
                ]
                if x
                else None
            )
        )

        # Remove matching columns if not needed
        if not include_match_info:
            df = df.drop(columns=["gtr_author_name", "name_match_score"], errors="ignore")

        output.append(df)

    return pd.concat(output) if output else pd.DataFrame()



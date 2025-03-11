"""Author-specific utilities for OpenAlex data collection."""

import logging
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


def parse_author_results(
    response: List[Dict], keys_to_include: Optional[List[str]] = None
) -> List[Dict]:
    """Parses OpenAlex API author response to retain specified keys or all if keys_to_include
     is None.

    Args:
        response (List[Dict]): The response from the OpenAlex API.
        keys_to_include (Optional[List[str]]): List of keys to include in the
            result. Returns full dictionary if None.

    Returns:
        List[Dict]: A list of dictionaries containing the parsed author information.
    """
    parsed_response = []
    for author in response:
        parsed_author = {
            "id": author.get("id", "").replace("https://openalex.org/", ""),
            "orcid": author.get("orcid", "").replace("https://orcid.org/", ""),
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
        if keys_to_include is not None:
            # Filter the dictionary to only include specified keys
            parsed_author = {
                key: parsed_author[key]
                for key in keys_to_include
                if key in parsed_author
            }
        parsed_response.append(parsed_author)
    return parsed_response


def json_loader_authors(data: List[List[Dict]]) -> pd.DataFrame:
    """
    Load authors JSON data, transform it into a DataFrame, and wrangle data.

    Args:
        data (List[List[Dict]]): The authors JSON data in batches.

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
        output.append(df)

    return pd.concat(output) if output else pd.DataFrame()

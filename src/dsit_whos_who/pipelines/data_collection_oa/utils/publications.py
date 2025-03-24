"""Publications-specific utilities for OpenAlex data collection."""

import logging
from typing import Dict, List, Optional, Sequence
import pandas as pd

logger = logging.getLogger(__name__)


def _revert_abstract_index(abstract_inverted_index: Dict[str, Sequence[int]]) -> str:
    """Reverts the abstract inverted index to the original text.

    Args:
        abstract_inverted_index (Dict[str, Sequence[int]]): The abstract inverted index.

    Returns:
        str: The original text.
    """
    try:
        length_of_text = (
            max(
                [
                    index
                    for sublist in abstract_inverted_index.values()
                    for index in sublist
                ]
            )
            + 1
        )
        recreated_text = [""] * length_of_text

        for word, indices in abstract_inverted_index.items():
            for index in indices:
                recreated_text[index] = word

        return " ".join(recreated_text)
    except (AttributeError, ValueError):
        return ""


def preprocess_publication_doi(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the Gateway to Research publication data to include
    doi values that are compatible with OA filter module.

    Args:
        df (pd.DataFrame): The Gateway to Research publication data.

    Returns:
        pd.DataFrame: The preprocessed publication data.
    """
    if "doi" in df.columns:
        df["doi"] = df["doi"].str.extract(r"(10\..+)")
    return df


def parse_works_results(
    response: List[Dict], keys_to_include: Optional[List[str]] = None
) -> List[Dict]:
    """Parses OpenAlex API works response to retain specified keys or all if keys_to_include
     is None.

    Args:
        response (List[Dict]): The response from the OpenAlex API.
        keys_to_include (Optional[List[str]]): List of keys to include in the
            result. Returns full dictionary if None.

    Returns:
        List[Dict]: A list of dictionaries containing the parsed works information.
    """
    parsed_response = []
    for paper in response:
        parsed_paper = {
            "id": paper.get("id", "").replace("https://openalex.org/", ""),
            "doi": paper.get("doi", ""),
            "title": paper.get("title", ""),
            "publication_date": paper.get("publication_date", ""),
            "fwci": paper.get("fwci", ""),
            "authorships": paper.get("authorships", []),
            "cited_by_count": paper.get("cited_by_count", ""),
            "topics": paper.get("topics", []),
            "referenced_works": paper.get("referenced_works", []),
            "ids": paper.get("ids", []),
            "counts_by_year": paper.get("counts_by_year", []),
            "citation_normalized_percentile": paper.get(
                "citation_normalized_percentile", {}
            ),
        }
        if keys_to_include is not None:
            # Filter the dictionary to only include specified keys
            parsed_paper = {
                key: parsed_paper[key] for key in keys_to_include if key in parsed_paper
            }
        parsed_response.append(parsed_paper)
    return parsed_response


def json_loader_works(data: List[List[Dict]]) -> pd.DataFrame:
    """
    Load works JSON data, transform it into a DataFrame, and wrangle data.

    Args:
        data (List[List[Dict]]): The works JSON data in batches.

    Returns:
        pandas.DataFrame: The transformed DataFrame.
    """
    output = []
    for batch in data:
        # Create DataFrame with all available columns
        df = pd.DataFrame(batch)
        if df.empty:
            continue

        # Only process columns that exist in the DataFrame
        if "ids" in df.columns:
            df["pmid"] = df["ids"].apply(
                lambda x: (
                    x.get("pmid").replace("https://pubmed.ncbi.nlm.nih.gov/", "")
                    if x and x.get("pmid")
                    else None
                )
            )

            df["mag_id"] = df["ids"].apply(
                lambda x: (x.get("mag") if x and x.get("mag") else None)
            )

        if "authorships" in df.columns:
            df["authorships"] = df["authorships"].apply(
                lambda x: (
                    [
                        (
                            (
                                author["author"]["id"].replace(
                                    "https://openalex.org/", ""
                                ),
                                inst["id"].replace("https://openalex.org/", ""),
                                inst["country_code"],
                                author["author_position"],
                            )
                            if author["institutions"]
                            else [
                                author["author"]["id"].replace(
                                    "https://openalex.org/", ""
                                ),
                                "",
                                "",
                                author["author_position"],
                            ]
                        )
                        for author in x
                        for inst in author["institutions"] or [{}]
                    ]
                    if x
                    else None
                )
            )

        if "counts_by_year" in df.columns:
            df["counts_by_year"] = df["counts_by_year"].apply(
                lambda x: (
                    [(year["year"], year["cited_by_count"]) for year in x]
                    if x
                    else None
                )
            )

        if "topics" in df.columns:
            df["topics"] = df["topics"].apply(
                lambda x: (
                    [
                        (
                            topic["id"].replace("https://openalex.org/", ""),
                            topic["display_name"],
                            topic["subfield"]["id"].replace(
                                "https://openalex.org/", ""
                            ),
                            topic["subfield"]["display_name"],
                            topic["field"]["id"].replace("https://openalex.org/", ""),
                            topic["field"]["display_name"],
                            topic["domain"]["id"].replace("https://openalex.org/", ""),
                            topic["domain"]["display_name"],
                        )
                        for topic in x
                    ]
                    if x
                    else None
                )
            )

        if "referenced_works" in df.columns:
            df["referenced_works"] = df["referenced_works"].apply(
                lambda x: (
                    [work.replace("https://openalex.org/", "") for work in x]
                    if x
                    else None
                )
            )

        if "citation_normalized_percentile" in df.columns:
            # keep value key
            df["citation_percentile"] = df[
                "citation_normalized_percentile"
            ].apply(lambda x: (x.get("value", 0) if x else None))

        # Keep all columns that exist in the DataFrame
        available_columns = [
            col
            for col in [
                "id",
                "doi",
                "pmid",
                "mag_id",
                "title",
                "publication_date",
                "cited_by_count",
                "fwci",
                "counts_by_year",
                "authorships",
                "topics",
                "referenced_works",
                "citation_percentile"
            ]
            if col in df.columns
        ]

        df = df[available_columns]
        output.append(df)

    return pd.concat(output) if output else pd.DataFrame()

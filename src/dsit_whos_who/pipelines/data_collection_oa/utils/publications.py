"""
Publication data processing utilities for OpenAlex API.
Handles publication parsing, DOI preprocessing, and data transformation.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


def preprocess_publication_doi(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DOIs from Gateway to Research data for OpenAlex compatibility.

    Args:
        df: Publication data with DOI column

    Returns:
        DataFrame with standardised DOI format
    """
    if "doi" in df.columns:
        df["doi"] = df["doi"].str.extract(r"(10\..+)")
    return df


def parse_works_results(
    response: List[Dict], keys_to_include: Optional[List[str]] = None
) -> List[Dict]:
    """Extract and normalise publication data from OpenAlex API response.

    Args:
        response: Raw API response containing works data
        keys_to_include: Specific fields to keep (all if None)

    Returns:
        List of normalised publication records
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
    """Transform batched OpenAlex works data into structured DataFrame.

    Processes:
    - Publication IDs (PMID, MAG)
    - Author affiliations and positions
    - Citation counts by year
    - Topic hierarchies
    - Referenced works

    Args:
        data: Batched publication records from OpenAlex

    Returns:
        DataFrame with normalised publication data
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
            df["citation_normalized_percentile"] = df[
                "citation_normalized_percentile"
            ].apply(lambda x: x if x else None)

        output.append(df)

    if output:
        return pd.concat(output, ignore_index=True)
    return pd.DataFrame()

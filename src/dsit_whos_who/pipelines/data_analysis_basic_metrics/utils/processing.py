"""
Utility functions for processing collected research data.

This module provides functions for processing and transforming raw data collected
from various sources (OpenAlex, GTR) into structured formats suitable for analysis.
It handles:
- Author metadata processing
- Date standardisation
- Project data aggregation
- Publication metrics computation
- Collaboration network analysis
"""

# pylint: disable=E0402

import logging
from typing import Dict, List
import pandas as pd
import numpy as np
from ...data_collection_oa.utils.authors import json_loader_authors

logger = logging.getLogger(__name__)


def process_author_metadata(
    author_dict: Dict, matched_authors: pd.DataFrame
) -> pd.DataFrame:
    """
    Process metadata for a batch of authors from OpenAlex.

    This function transforms raw author metadata into a structured format and
    enriches it with additional metrics. It:
    - Creates a DataFrame from the raw JSON data
    - Filters to include only authors matched with GTR records
    - Computes citation impact metrics
    - Standardises publication dates and counts

    Args:
        author_dict (Dict): Dictionary containing raw author metadata from OpenAlex
        matched_authors (pd.DataFrame): DataFrame containing matched author IDs
            between GTR and OpenAlex, with columns for 'gtr_id', 'id', and
            'match_probability'

    Returns:
        pd.DataFrame: Processed author metadata with computed metrics including
            citation counts and impact measures
    """
    # create dataframe
    author_df = json_loader_authors(author_dict)

    # filter by matched authors
    author_df = author_df.merge(
        matched_authors[["gtr_id", "id", "match_probability"]],
        on="id",
        how="inner",
    )

    # # get the first year of publication
    # author_df["first_work_year"] = pd.to_numeric(
    #     author_df["counts_by_year"].apply(
    #         lambda x: min(y[0] for y in x) if x else None
    #     ),
    #     errors="coerce",
    # ).astype("Int64")

    # citations per publication
    author_df["citations_per_publication"] = (
        author_df["cited_by_count"] / author_df["works_count"]
    )

    return author_df


def date_cleaner(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise and clean date fields in project data.

    This function processes date fields in project records to ensure consistency
    and handle edge cases. It:
    - Converts string dates to datetime objects
    - Handles missing or invalid dates
    - Resolves project extensions by using extended end dates where available
    - Ensures all dates are in a consistent format

    Args:
        df (pd.DataFrame): DataFrame containing project records with date fields:
            - start_date: Project start date
            - end_date: Original project end date
            - extended_end: Extended project end date (if applicable)

    Returns:
        pd.DataFrame: DataFrame with cleaned and standardised date fields
    """
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df["extended_end"] = pd.to_datetime(df["extended_end"], errors="coerce")

    df["extended_end"] = df["extended_end"].fillna(df["end_date"])
    df["end_date"] = df["extended_end"]

    return df


def process_person_gtr_data(
    persons_projects: pd.DataFrame,
) -> pd.DataFrame:
    """
    Process Gateway to Research (GTR) data for a group of researchers.

    This function aggregates and summarises project information for each researcher
    in the GTR database. It computes:
    - Project timelines and duration metrics
    - Grant categories and funding patterns
    - Project status information
    - Funding source diversity

    The function creates a comprehensive summary of each researcher's funding
    history and project portfolio.

    Args:
        persons_projects (pd.DataFrame): DataFrame containing person-project
            relationships with columns for dates, grant categories, and project
            status

    Returns:
        pd.DataFrame: Aggregated project information per person, including:
            - earliest_start_date: Date of first project
            - latest_end_date: Date of most recent project end
            - project_timeline: Detailed timeline of all projects
            - grant_categories: Distribution of grant types
            - lead_funders: List of funding organisations
            - has_active_project: Current project status
    """
    logger.info("Creating aggregated project information per person")
    return (
        persons_projects.groupby("person_id")
        .agg(
            {
                "start_date": lambda x: (
                    min(x).strftime("%Y-%m-%d") if pd.notnull(min(x)) else None
                ),
                "end_date": lambda x: (
                    max(x).strftime("%Y-%m-%d") if pd.notnull(max(x)) else None
                ),
                "project_id": lambda x: list(
                    [
                        (
                            [
                                str(pid),
                                str(grant),
                                (
                                    start.strftime("%Y-%m-%d")
                                    if pd.notnull(start)
                                    else None
                                ),
                                end.strftime("%Y-%m-%d") if pd.notnull(end) else None,
                            ]
                            if pd.notnull(start) and pd.notnull(end)
                            else []
                        )
                        for pid, grant, start, end in zip(
                            x,
                            persons_projects.loc[x.index, "grant_category"],
                            persons_projects.loc[x.index, "start_date"],
                            persons_projects.loc[x.index, "end_date"],
                        )
                    ]
                ),
                "grant_category": lambda x: [
                    [str(cat), str(count)] for cat, count in x.value_counts().items()
                ],
                "lead_funder": lambda x: [
                    [str(funder), str(count)]
                    for funder, count in x.value_counts().items()
                ],
                "status": lambda x: "True" if "Active" in set(x) else "False",
            }
        )
        .rename(
            columns={
                "start_date": "earliest_start_date",
                "end_date": "latest_end_date",
                "project_id": "project_timeline",
                "grant_category": "grant_categories",
                "lead_funder": "lead_funders",
                "status": "has_active_project",
            }
        )
        .reset_index()
    )


def prepare_final_person_data(
    persons: pd.DataFrame,
    person_summaries: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare the final researcher dataset with comprehensive metrics.

    This function combines personal information with computed metrics to create
    a complete researcher profile. It organises the data into logical sections:
    - Personal identifiers and basic information
    - Current institutional affiliations
    - Project timelines and funding history
    - Research outputs and collaboration networks
    - Grant portfolio analysis

    Args:
        persons (pd.DataFrame): Base DataFrame containing researcher information
        person_summaries (pd.DataFrame): DataFrame containing computed metrics
            and aggregated project information

    Returns:
        pd.DataFrame: Complete researcher profiles with columns organised into
            sections:
            - Personal identifiers (ID, name)
            - Institutional information
            - Project timelines
            - Funding details
            - Research outputs
            - Collaboration networks
    """
    return (
        persons.merge(person_summaries, on="person_id", how="left")
        .drop(columns=["orcid_gtr"])
        .assign(
            has_multiple_funders=lambda x: x["lead_funders"].apply(
                lambda y: "True" if len(y) > 1 else "False"
            ),
            number_grants=lambda x: x["project_id"].apply(
                lambda y: len(y) if isinstance(y, np.ndarray) and y.any() else 0
            ),
        )
        .reindex(
            columns=[
                # Personal identifiers and basic info
                "person_id",
                "first_name",
                "surname",
                # Current institutional affiliation
                "organisation",
                "organisation_name",
                # Project timeline information
                "earliest_start_date",
                "latest_end_date",
                "has_active_project",
                # Funding details
                "number_grants",
                "has_multiple_funders",
                "grant_categories",
                "lead_funders",
                # Detailed project information
                "project_timeline",
                "project_id",
                # Research outputs and collaborations
                "project_publications",
                "project_topics",
                "project_authors",
                "project_oa_ids",
            ]
        )
    )


def process_publication_batch(
    publications_batch: pd.DataFrame, matched_ids: set
) -> List[Dict]:
    """
    Process a batch of publications to extract collaboration metrics.

    This function analyses publication authorships to understand collaboration
    patterns. For each publication, it:
    - Identifies relevant authors from the matched set
    - Maps author locations and institutional affiliations
    - Computes domestic vs international collaboration metrics
    - Tracks unique collaborators and their geographic distribution

    The function processes each publication's authorship data in two passes:
    1. First pass: Collect all authors and their countries
    2. Second pass: Compute collaboration metrics for matched authors

    Args:
        publications_batch (pd.DataFrame): Batch of publications to process,
            containing authorship and affiliation data
        matched_ids (set): Set of author IDs that have been matched between
            OpenAlex and GTR

    Returns:
        List[Dict]: List of processed collaboration data dictionaries, each
            containing:
            - Publication year and impact metrics
            - UK vs international collaboration counts
            - Geographic distribution of collaborators
            - Affiliation patterns
            - Unique collaborator identifiers
    """
    processed_data = []
    for _, pub in publications_batch.iterrows():
        year = pub["year"]
        fwci = pub.get("fwci", np.nan)
        cited_by_count = pub.get("cited_by_count", np.nan)

        if not isinstance(pub["authorships"], np.ndarray):
            continue

        relevant_authors = set()
        author_countries = {}

        # First pass: collect all authors and their countries
        for author_data in pub["authorships"]:
            if not isinstance(author_data, np.ndarray) or len(author_data) < 2:
                continue

            author_id, country = author_data

            # skip if author is already in the list (ie. stick to first country)
            if author_id in author_countries:
                continue

            author_countries[author_id] = country

            # add to relevant authors if they are in the matched list
            if author_id in matched_ids:
                relevant_authors.add(author_id)

        if not relevant_authors:
            continue

        # Second pass: process collaborations for each matched author
        for main_author_id in relevant_authors:
            uk_collabs = 0
            abroad_collabs = 0
            unknown_collabs = 0
            collab_countries = set()
            collab_ids = set()
            uk_affils = 0
            abroad_affils = 0
            unknown_affils = 0
            affiliation_countries = list()

            for collab_id, country in author_countries.items():
                if pd.isna(country):
                    continue

                # if main author, tally up affiliation countries
                if collab_id == main_author_id:
                    if country == "":
                        unknown_affils += 1
                    elif country == "GB":
                        uk_affils += 1
                    else:
                        abroad_affils += 1
                    affiliation_countries.append(country)
                    continue

                # if not main author, tally up collab countries
                if country == "":
                    unknown_collabs += 1
                elif country == "GB":
                    uk_collabs += 1
                    collab_countries.add(country)
                else:
                    abroad_collabs += 1
                    collab_countries.add(country)
                collab_ids.add(collab_id)

            # Convert counts to list of pairs

            processed_data.append(
                {
                    "author_id": main_author_id,
                    "year": year,
                    "affiliation_countries": sorted(affiliation_countries),
                    "n_affils_uk": uk_affils,
                    "n_affils_abroad": abroad_affils,
                    "n_affils_unknown": unknown_affils,
                    "n_collab_uk": uk_collabs,
                    "n_collab_abroad": abroad_collabs,
                    "n_collab_unknown": unknown_collabs,
                    "collab_countries": sorted(list(collab_countries)),
                    "collab_ids": sorted(list(collab_ids)),
                    "fwci": fwci,
                    "cited_by_count": cited_by_count,
                }
            )

    return processed_data

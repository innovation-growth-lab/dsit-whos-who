"""
Utility functions for processing collected data.
"""

# pylint: disable=E0402

import logging
from typing import Dict
import pandas as pd
import numpy as np
from ...data_collection_oa.utils.authors import json_loader_authors

logger = logging.getLogger(__name__)


def process_author_metadata(
    author_dict: Dict, matched_authors: pd.DataFrame
) -> pd.DataFrame:
    """Process metadata for a single author batch."""
    # create dataframe
    author_df = json_loader_authors(author_dict)

    # filter by matched authors
    author_df = author_df.merge(
        matched_authors[["gtr_id", "id", "match_probability"]],
        on="id",
        how="inner",
    )

    # get the first year of publication
    author_df["first_work_year"] = pd.to_numeric(
        author_df["counts_by_year"].apply(
            lambda x: min(y[0] for y in x) if x else None
        ),
        errors="coerce",
    ).astype("Int64")

    # citations per publication
    author_df["citations_per_publication"] = (
        author_df["cited_by_count"] / author_df["works_count"]
    )

    return author_df


def date_cleaner(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dates in the dataframe."""
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df["extended_end"] = pd.to_datetime(df["extended_end"], errors="coerce")

    df["extended_end"] = df["extended_end"].fillna(df["end_date"])
    df["end_date"] = df["extended_end"]

    return df


def process_person_gtr_data(
    persons_projects: pd.DataFrame,
) -> pd.DataFrame:
    """Process GTR data for a group of persons."""
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
    """Prepare the final person data with all metrics."""
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

"""
Utility functions for computing affiliation-related metrics.

This module provides functions for analysing researchers' institutional affiliations
and collaboration patterns. It processes data about:
- Current and historical institutional affiliations
- UK vs international institutional relationships
- Collaboration networks and geographic distribution
- Career mobility patterns
"""

# pylint: disable=E0402

import logging
from collections import Counter
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def process_last_institution(row: pd.Series) -> bool:
    """
    Determine whether a researcher's most recent institution is UK-based.

    This function analyses the most recent institutional affiliation in a researcher's
    record to determine if they are currently (or were most recently) based at a UK
    institution. This information is useful for understanding researcher mobility and
    current geographic distribution.

    Args:
        row (pd.Series): Row containing 'last_known_institutions' field with an array
            of institution records, where each record contains country information

    Returns:
        bool: True if the most recent institution is in the UK (GB),
            False if abroad, NaN if unknown or no data available
    """
    if (
        not isinstance(row["last_known_institutions"], np.ndarray)
        or len(row["last_known_institutions"]) == 0
    ):
        return np.nan

    # Get the first institution (most recent)
    last_inst = row["last_known_institutions"][0]
    if not isinstance(last_inst, np.ndarray):
        return np.nan

    country = last_inst[2]
    if pd.isna(country):
        return np.nan

    return country == "GB"


def process_collaborations(publications_group: pd.Series) -> dict:
    """
    Analyse collaboration patterns for an individual researcher.

    This function processes a researcher's publication record to understand their
    collaboration patterns before and after receiving their first research grant.
    It analyses:
    - Geographic distribution of collaborators
    - Number of unique collaborators
    - Balance between domestic and international collaborations
    - Temporal changes in collaboration patterns

    Args:
        publications_group (pd.Series): Group of publications for an author,
            containing collaboration data and the date of their first grant
            ('earliest_start_date')

    Returns:
        dict: Dictionary containing collaboration metrics:
            - collab_countries_before/after: Lists of [country, count] pairs
            - unique_collabs_before/after: Number of unique collaborators
            - total_collabs_before/after: Total number of collaborations
            - foreign_collab_fraction_before/after: Proportion of international collabs
            - collab_countries_list_before/after: Lists of unique countries
    """
    if pd.isnull(publications_group["earliest_start_date"].iloc[0]):
        return {
            "collab_countries_before": [],
            "collab_countries_after": [],
            "unique_collabs_before": np.nan,
            "unique_collabs_after": np.nan,
            "total_collabs_before": np.nan,
            "total_collabs_after": np.nan,
            "foreign_collab_fraction_before": np.nan,
            "foreign_collab_fraction_after": np.nan,
            "collab_countries_list_before": [],
            "collab_countries_list_after": [],
        }

    ref_year = pd.to_datetime(
        publications_group["earliest_start_date"].iloc[0] + "-01-01"
    ).year

    # Split into before/after
    before = publications_group[publications_group["year"] < ref_year]
    after = publications_group[publications_group["year"] >= ref_year]

    # Process before period
    uk_before = before["n_collab_uk"].sum()
    abroad_before = before["n_collab_abroad"].sum()
    total_before = int(uk_before + abroad_before)
    countries_before = Counter(
        country
        for countries in before["unique_collab_countries"]
        for country in countries
        if country != "GB"
    )
    collabs_before = set(
        collab_id
        for collab_ids in before["unique_collab_ids"]
        for collab_id in collab_ids
    )

    # Process after period
    uk_after = after["n_collab_uk"].sum()
    abroad_after = after["n_collab_abroad"].sum()
    total_after = int(uk_after + abroad_after)
    countries_after = Counter(
        country
        for countries in after["unique_collab_countries"]
        for country in countries
        if country != "GB"
    )
    collabs_after = set(
        collab_id
        for collab_ids in after["unique_collab_ids"]
        for collab_id in collab_ids
    )

    # Calculate fractions
    foreign_fraction_before = (
        float(round(abroad_before / total_before, 3)) if total_before > 0 else np.nan
    )
    foreign_fraction_after = (
        float(round(abroad_after / total_after, 3)) if total_after > 0 else np.nan
    )

    # Convert counters to list of pairs
    countries_before_list = [[k, str(v)] for k, v in countries_before.items()]
    countries_after_list = [[k, str(v)] for k, v in countries_after.items()]

    return {
        "collab_countries_before": countries_before_list,
        "collab_countries_after": countries_after_list,
        "unique_collabs_before": len(collabs_before),
        "unique_collabs_after": len(collabs_after),
        "total_collabs_before": total_before,
        "total_collabs_after": total_after,
        "foreign_collab_fraction_before": foreign_fraction_before,
        "foreign_collab_fraction_after": foreign_fraction_after,
        "collab_countries_list_before": sorted(countries_before.keys()),
        "collab_countries_list_after": sorted(countries_after.keys()),
    }


def compile_affiliations_by_author(publications: pd.DataFrame) -> dict:
    """
    Pre-aggregate institutional affiliation data by author.

    This function processes publication records to create a year-by-year summary of
    each author's institutional affiliations. It tracks:
    - UK vs non-UK institutional affiliations
    - Geographic distribution of affiliations
    - Temporal patterns in institutional relationships

    The pre-aggregation improves performance for subsequent analyses by avoiding
    repeated filtering operations.

    Args:
        publications (pd.DataFrame): DataFrame containing publication records with
            columns for author_id, year, and affiliation information including
            country codes and counts

    Returns:
        dict: Dictionary mapping author_ids to lists of yearly affiliation data,
            where each year's data contains:
            - year: Publication year
            - n_uk: Number of UK affiliations
            - n_abroad: Number of non-UK affiliations
            - countries: List of unique countries (excluding UK)
    """
    # Group by author and year, aggregate the counts and countries
    agg_data = (
        publications.groupby(["author_id", "year"])
        .agg(
            {
                "n_affil_uk": "sum",
                "n_affil_abroad": "sum",
                "affiliation_countries_abroad": lambda x: set().union(
                    *[
                        set(countries) if isinstance(countries, list) else set()
                        for countries in x
                    ]
                ),
            }
        )
        .reset_index()
    )

    # Convert to dictionary for fast lookup
    author_data = {}
    for _, row in agg_data.iterrows():
        author_id = row["author_id"]
        if author_id not in author_data:
            author_data[author_id] = []

        author_data[author_id].append(
            {
                "year": row["year"],
                "n_uk": row["n_affil_uk"] if not pd.isna(row["n_affil_uk"]) else 0,
                "n_abroad": (
                    row["n_affil_abroad"] if not pd.isna(row["n_affil_abroad"]) else 0
                ),
                "countries": (
                    list(row["affiliation_countries_abroad"])
                    if isinstance(row["affiliation_countries_abroad"], set)
                    else []
                ),
            }
        )

    return author_data


def process_affiliations(publications_group: pd.Series) -> dict:
    """
    Analyse institutional affiliation patterns before and after first grant.

    This function examines a researcher's institutional affiliations to understand
    their geographic mobility and international experience. It analyses patterns
    both before and after their first research grant to identify changes in:
    - International research experience
    - Geographic diversity of affiliations
    - Balance between UK and international institutional relationships

    Args:
        publications_group (pd.Series): Group of publications for an author,
            containing affiliation data and the date of their first grant
            ('earliest_start_date')

    Returns:
        dict: Dictionary containing affiliation metrics:
            - abroad_experience_before/after: Boolean indicating international experience
            - countries_before/after: Lists of countries (excluding UK) where the
                researcher has held affiliations
            - abroad_fraction_before/after: Proportion of non-UK affiliations
    """
    if pd.isnull(publications_group["earliest_start_date"].iloc[0]):
        return {
            "abroad_experience_before": np.nan,
            "abroad_experience_after": np.nan,
            "countries_before": [],
            "countries_after": [],
            "abroad_fraction_before": np.nan,
            "abroad_fraction_after": np.nan,
        }

    ref_year = pd.to_datetime(publications_group["earliest_start_date"]).iloc[0].year

    # Split into before/after
    before = publications_group[publications_group["year"] < ref_year]
    after = publications_group[publications_group["year"] >= ref_year]

    # Process before period
    uk_before = before["n_affils_uk"].sum()
    abroad_before = before["n_affils_abroad"].sum()
    countries_before = set(
        country
        for countries in before["affiliation_countries"]
        for country in countries
        if country != "GB" and country != ""
    )

    # Process after period
    uk_after = after["n_affils_uk"].sum()
    abroad_after = after["n_affils_abroad"].sum()
    countries_after = set(
        country
        for countries in after["affiliation_countries"]
        for country in countries
        if country != "GB" and country != ""
    )

    # Calculate fractions
    total_before = uk_before + abroad_before
    total_after = uk_after + abroad_after

    abroad_fraction_before = float(
        round(abroad_before / total_before, 3) if total_before > 0 else np.nan
    )
    abroad_fraction_after = float(
        round(abroad_after / total_after, 3) if total_after > 0 else np.nan
    )

    return {
        "abroad_experience_before": (
            bool(countries_before) if total_before > 0 else np.nan
        ),
        "abroad_experience_after": bool(countries_after) if total_after > 0 else np.nan,
        "countries_before": sorted(list(countries_before)),
        "countries_after": sorted(list(countries_after)),
        "abroad_fraction_before": abroad_fraction_before,
        "abroad_fraction_after": abroad_fraction_after,
    }

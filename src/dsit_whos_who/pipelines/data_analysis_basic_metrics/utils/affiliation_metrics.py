"""
Utility functions for computing affiliation-related metrics.
"""

# pylint: disable=E0402

import logging
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from .publication_metrics import calculate_uk_fraction

logger = logging.getLogger(__name__)


def process_last_institution(row: pd.Series) -> bool:
    """
    Check if the last known institution is UK-based.

    Args:
        row (pd.Series): Row containing last_known_institutions

    Returns:
        bool: True if last institution is in UK, False if abroad, NaN if unknown
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


def process_collaborations(row: pd.Series, publications: pd.DataFrame) -> dict:
    """
    Process publication collaborations for an author.

    Args:
        row (pd.Series): Row containing author ID and earliest_start_date
        publications (pd.DataFrame): DataFrame with publication data

    Returns:
        dict: Dictionary containing collaboration metrics
    """
    if pd.isnull(row["earliest_start_date"]):
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

    # Get reference date
    ref_year = pd.to_datetime(row["earliest_start_date"]).year
    author_id = row["id"]

    # Get author's publications
    author_pubs = publications[publications["author_id"] == author_id]

    # Split into before and after
    pubs_before = author_pubs[author_pubs["year"] < ref_year]
    pubs_after = author_pubs[author_pubs["year"] >= ref_year]

    # Process before period
    collabs_before = set()
    countries_before = Counter()
    total_collabs_before = 0
    foreign_collabs_before = 0

    for _, pub in pubs_before.iterrows():
        collabs_before.update(pub["collab_ids"])
        countries_before.update(pub["countries_abroad"])
        total_collabs_before += pub["n_collab_uk"] + pub["n_collab_abroad"]
        foreign_collabs_before += pub["n_collab_abroad"]

    # Process after period
    collabs_after = set()
    countries_after = Counter()
    total_collabs_after = 0
    foreign_collabs_after = 0

    for _, pub in pubs_after.iterrows():
        collabs_after.update(pub["collab_ids"])
        countries_after.update(pub["countries_abroad"])
        total_collabs_after += pub["n_collab_uk"] + pub["n_collab_abroad"]
        foreign_collabs_after += pub["n_collab_abroad"]

    # Calculate fractions and means
    foreign_fraction_before = (
        round(foreign_collabs_before / total_collabs_before, 3)
        if total_collabs_before > 0
        else np.nan
    )
    foreign_fraction_after = (
        round(foreign_collabs_after / total_collabs_after, 3)
        if total_collabs_after > 0
        else np.nan
    )

    # Convert sets to lists for JSON serialisation
    countries_before_list = [[k, str(v)] for k, v in countries_before.items()]
    countries_after_list = [[k, str(v)] for k, v in countries_after.items()]

    return {
        "collab_countries_before": countries_before_list,
        "collab_countries_after": countries_after_list,
        "unique_collabs_before": len(collabs_before),
        "unique_collabs_after": len(collabs_after),
        "total_collabs_before": total_collabs_before,
        "total_collabs_after": total_collabs_after,
        "foreign_collab_fraction_before": foreign_fraction_before,
        "foreign_collab_fraction_after": foreign_fraction_after,
        "collab_countries_list_before": sorted(countries_before.keys()),
        "collab_countries_list_after": sorted(countries_after.keys()),
    }


def process_affiliations(row: pd.Series) -> dict:
    """
    Process affiliations to determine UK vs abroad experience before and after first grant.

    Args:
        row (pd.Series): Row containing affiliations and earliest_start_date

    Returns:
        dict: Dictionary containing affiliation metrics
    """
    if pd.isnull(row["earliest_start_date"]) or not isinstance(
        row["affiliations"], np.ndarray
    ):
        return {
            "abroad_experience_before": np.nan,
            "abroad_experience_after": np.nan,
            "countries_before": [],
            "countries_after": [],
            "abroad_fraction_before": np.nan,
            "abroad_fraction_after": np.nan,
        }

    ref_date = pd.to_datetime(f"{row['earliest_start_date']}-01-01")
    yearly_affiliations_before = defaultdict(lambda: {"uk": 0, "abroad": 0})
    yearly_affiliations_after = defaultdict(lambda: {"uk": 0, "abroad": 0})
    countries_before = set()
    countries_after = set()

    for affiliation in row["affiliations"]:
        if len(affiliation) >= 5:
            country = affiliation[2]
            if pd.isna(country):
                continue

            try:
                years = {
                    int(y)
                    for y in affiliation[4].split(",")
                    if y.strip() and 1980 <= int(y) <= 2026
                }
            except (ValueError, TypeError):
                continue

            for year in years:
                # Compare year directly instead of converting to datetime
                if year < ref_date.year:  # if before the first grant
                    if country == "GB":
                        yearly_affiliations_before[year]["uk"] += 1
                    else:
                        yearly_affiliations_before[year]["abroad"] += 1
                        countries_before.add(country)
                else:  # if after or during the first grant year
                    if country == "GB":
                        yearly_affiliations_after[year]["uk"] += 1
                    else:
                        yearly_affiliations_after[year]["abroad"] += 1
                        countries_after.add(country)

    uk_fraction_before = calculate_uk_fraction(yearly_affiliations_before)
    uk_fraction_after = calculate_uk_fraction(yearly_affiliations_after)

    # Convert NaN to NaN for abroad fractions
    abroad_fraction_before = (
        np.nan if np.isnan(uk_fraction_before) else round(1 - uk_fraction_before, 3)
    )
    abroad_fraction_after = (
        np.nan if np.isnan(uk_fraction_after) else round(1 - uk_fraction_after, 3)
    )

    # Set experience to NaN if we have no affiliations in the period
    has_affiliations_before = bool(yearly_affiliations_before)
    has_affiliations_after = bool(yearly_affiliations_after)

    return {
        "abroad_experience_before": (
            bool(countries_before) if has_affiliations_before else np.nan
        ),
        "abroad_experience_after": (
            bool(countries_after) if has_affiliations_after else np.nan
        ),
        "countries_before": sorted(list(filter(None, countries_before))),
        "countries_after": sorted(list(filter(None, countries_after))),
        "abroad_fraction_before": abroad_fraction_before,
        "abroad_fraction_after": abroad_fraction_after,
    }

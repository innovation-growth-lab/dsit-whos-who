"""
Utility functions for computing basic metrics.
"""

import logging
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
logger = logging.getLogger(__name__)


def compute_academic_age(row: pd.Series) -> int:
    """
    Compute academic age (years between first publication and first grant).

    Args:
        row (pd.Series): Row containing first_work_year and earliest_start_date

    Returns:
        int: Academic age in years, or None if dates are missing
    """
    if pd.isnull(row["earliest_start_date"]):
        return None

    if pd.isnull(row["first_work_year"]):
        return None
    return (
        pd.to_datetime(f"{row['earliest_start_date']}-01-01").year
        - pd.to_datetime(f"{row['first_work_year']}-01-01").year
    )


def _process_last_institution(row: pd.Series) -> bool:
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


def add_international_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add international experience metrics to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with affiliations information

    Returns:
        pd.DataFrame: DataFrame with added international metrics
    """
    # Apply the affiliation processing
    affiliation_metrics = df.progress_apply(_process_affiliations, axis=1)

    # Add the computed metrics to the dataframe
    metrics = [
        "abroad_experience_before",
        "abroad_experience_after",
        "countries_before",
        "countries_after",
        "abroad_fraction_before",
        "abroad_fraction_after",
    ]

    # Create a new dataframe with the metrics and explicitly assign back to original df
    for metric in metrics:
        df[metric] = affiliation_metrics.apply(
            lambda x: x[metric]  # pylint: disable=W0640
        )

    # Add last known institution info
    df["last_known_institution_uk"] = df.progress_apply(
        _process_last_institution, axis=1
    )

    return df


def _calculate_uk_fraction(yearly_data):
    """
    Calculate the fraction of UK affiliations over total affiliations.
    Returns NaN if there are no affiliations in the period.

    Args:
        yearly_data (dict): Dictionary of years with UK and abroad affiliation counts

    Returns:
        float: Fraction of UK affiliations, NaN if no affiliations exist
    """
    if not yearly_data:
        return np.nan

    total_uk = sum(year["uk"] for year in yearly_data.values())
    total_abroad = sum(year["abroad"] for year in yearly_data.values())
    total = total_uk + total_abroad

    return round(total_uk / total if total > 0 else np.nan, 3)


def _process_affiliations(row: pd.Series) -> dict:
    """
    Process affiliations for a single author to compute international experience metrics.

    Args:
        row (pd.Series): Row containing affiliations and earliest_start_date

    Returns:
        dict: Dictionary containing computed metrics about international experience
    """
    if not isinstance(row["affiliations"], np.ndarray):
        return {
            "abroad_experience_before": np.nan,
            "abroad_experience_after": np.nan,
            "countries_before": [],
            "countries_after": [],
            "abroad_fraction_before": np.nan,
            "abroad_fraction_after": np.nan,
        }

    # Get reference date
    ref_date = pd.to_datetime(row["earliest_start_date"])
    if pd.isnull(ref_date):
        return {
            "abroad_experience_before": np.nan,
            "abroad_experience_after": np.nan,
            "countries_before": [],
            "countries_after": [],
            "abroad_fraction_before": np.nan,
            "abroad_fraction_after": np.nan,
        }

    # Track yearly affiliations before and after
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
                    if y.strip() and 2002 <= int(y) <= 2026
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

    uk_fraction_before = _calculate_uk_fraction(yearly_affiliations_before)
    uk_fraction_after = _calculate_uk_fraction(yearly_affiliations_after)

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

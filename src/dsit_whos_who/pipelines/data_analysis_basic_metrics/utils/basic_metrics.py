"""
Utility functions for computing basic metrics.
"""

import logging
from collections import defaultdict, Counter
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


def _process_collaborations(row: pd.Series, publications: pd.DataFrame) -> dict:
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

    # Convert sets to lists for JSON serialization
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


def add_international_metrics(
    df: pd.DataFrame, publications: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Add international experience metrics to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with affiliations information
        publications (pd.DataFrame, optional): DataFrame with publication data

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

    logger.info("Processing publication collaborations...")
    collab_metrics = df.progress_apply(
        lambda x: _process_collaborations(x, publications), axis=1
    )

    collab_fields = [
        "collab_countries_before",
        "collab_countries_after",
        "unique_collabs_before",
        "unique_collabs_after",
        "total_collabs_before",
        "total_collabs_after",
        "foreign_collab_fraction_before",
        "foreign_collab_fraction_after",
        "collab_countries_list_before",
        "collab_countries_list_after",
    ]

    for field in collab_fields:
        df[field] = collab_metrics.apply(lambda x: x[field])  # pylint: disable=W0640

    # make int vars be Int64 to accomodate missing
    df["total_collabs_before"] = df["total_collabs_before"].astype("Int64")
    df["total_collabs_after"] = df["total_collabs_after"].astype("Int64")
    df["unique_collabs_before"] = df["unique_collabs_before"].astype("Int64")
    df["unique_collabs_after"] = df["unique_collabs_after"].astype("Int64")

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


def _redefine_counts_by_year(publications: pd.DataFrame) -> pd.DataFrame:
    """
    Recreate the counts_by_year data combining cited_by_count, n_pubs, and year
    to create a new column with a list of three elements. This is necessary because
    the author data only has data from 2012 onwards.
    """
    publications["counts_by_year"] = publications.apply(
        lambda row: [row["year"], row["n_pubs"], row["cited_by_count"]], axis=1
    )

    # create the author group'd version of the counts_by_year data
    author_grouped = publications.groupby("author_id")
    author_grouped["counts_by_year"] = author_grouped["counts_by_year"].apply(
        lambda x: x.tolist()
    )

    return author_grouped


def _process_counts_by_year(row: pd.Series) -> dict:
    """
    Process publication and citation counts from counts_by_year data.

    Args:
        row (pd.Series): Row containing counts_by_year and earliest_start_date

    Returns:
        dict: Dictionary containing publication and citation metrics
    """
    if pd.isnull(row["earliest_start_date"]) or not isinstance(
        row["counts_by_year"], np.ndarray
    ):
        return {
            "n_pubs_before": np.nan,
            "n_pubs_after": np.nan,
            "total_citations_before": np.nan,
            "total_citations_after": np.nan,
            "mean_citations_before": np.nan,
            "mean_citations_after": np.nan,
            "citations_per_pub_before": np.nan,
            "citations_per_pub_after": np.nan,
        }

    ref_year = pd.to_datetime(row["earliest_start_date"]).year
    pubs_before = []
    pubs_after = []
    citations_before = []
    citations_after = []

    for year_data in row["counts_by_year"]:
        if not isinstance(year_data, np.ndarray) or len(year_data) < 3:
            continue

        try:
            year = int(year_data[0])
            works = int(year_data[1])
            citations = int(year_data[2])
        except (ValueError, TypeError):
            continue

        if year < ref_year:
            pubs_before.append(works)
            citations_before.append(citations)
        else:
            pubs_after.append(works)
            citations_after.append(citations)

    total_pubs_before = sum(pubs_before) if pubs_before else 0
    total_pubs_after = sum(pubs_after) if pubs_after else 0
    total_citations_before = sum(citations_before) if citations_before else 0
    total_citations_after = sum(citations_after) if citations_after else 0

    return {
        "n_pubs_before": total_pubs_before if total_pubs_before > 0 else np.nan,
        "n_pubs_after": total_pubs_after if total_pubs_after > 0 else np.nan,
        "total_citations_before": (
            total_citations_before if total_citations_before > 0 else np.nan
        ),
        "total_citations_after": (
            total_citations_after if total_citations_after > 0 else np.nan
        ),
        "mean_citations_before": (
            round(np.mean(citations_before), 3) if citations_before else np.nan
        ),
        "mean_citations_after": (
            round(np.mean(citations_after), 3) if citations_after else np.nan
        ),
        "citations_per_pub_before": (
            round(total_citations_before / total_pubs_before, 3)
            if total_pubs_before > 0
            else np.nan
        ),
        "citations_per_pub_after": (
            round(total_citations_after / total_pubs_after, 3)
            if total_pubs_after > 0
            else np.nan
        ),
    }


def _process_fwci(row: pd.Series, pubs_df: pd.DataFrame) -> tuple:
    """
    Process FWCI metrics from publication data.

    Args:
        row (pd.Series): Row containing author ID and earliest_start_date
        pubs_df (pd.DataFrame): DataFrame with publication FWCI data

    Returns:
        tuple: (mean_fwci_before, mean_fwci_after)
    """
    if pd.isnull(row["earliest_start_date"]):
        return np.nan, np.nan

    ref_year = pd.to_datetime(row["earliest_start_date"]).year
    author_pubs = pubs_df[pubs_df["author_id"] == row["id"]]

    fwci_before = author_pubs[author_pubs["year"] < ref_year]["fwci"]
    fwci_after = author_pubs[author_pubs["year"] >= ref_year]["fwci"]

    mean_fwci_before = round(fwci_before.mean(), 3) if not fwci_before.empty else np.nan
    mean_fwci_after = round(fwci_after.mean(), 3) if not fwci_after.empty else np.nan

    return mean_fwci_before, mean_fwci_after


def add_publication_metrics(
    df: pd.DataFrame,
    publications: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add publication metrics including FWCI, citations, and publication counts before and after first grant.

    Args:
        df (pd.DataFrame): DataFrame with counts_by_year and earliest_start_date
        publications (pd.DataFrame): DataFrame with yearly FWCI data

    Returns:
        pd.DataFrame: DataFrame with added publication metrics
    """
    logger.info("Processing publication metrics...")
    counts_by_year = _redefine_counts_by_year(publications)

    # replace the counts_by_year column in the df with the author group'd version
    df["counts_by_year"] = df["author_id"].map(counts_by_year["counts_by_year"])

    # Process counts_by_year data
    logger.info("Processing citation and publication counts...")
    metrics = df.progress_apply(_process_counts_by_year, axis=1)

    for field in [
        "n_pubs_before",
        "n_pubs_after",
        "total_citations_before",
        "total_citations_after",
        "mean_citations_before",
        "mean_citations_after",
        "citations_per_pub_before",
        "citations_per_pub_after",
    ]:
        df[field] = metrics.apply(lambda x: x[field])

    # Process FWCI data
    logger.info("Processing FWCI metrics...")
    fwci_metrics = df.progress_apply(lambda x: _process_fwci(x, publications), axis=1)
    df["mean_fwci_before"] = fwci_metrics.apply(lambda x: x[0])
    df["mean_fwci_after"] = fwci_metrics.apply(lambda x: x[1])

    # Convert integer fields to Int64 to handle missing values
    int_fields = [
        "n_pubs_before",
        "n_pubs_after",
        "total_citations_before",
        "total_citations_after",
    ]
    for field in int_fields:
        df[field] = df[field].astype("Int64")

    return df

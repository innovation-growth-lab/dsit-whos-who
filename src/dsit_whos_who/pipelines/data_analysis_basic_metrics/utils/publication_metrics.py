"""
Utility functions for computing publication-related metrics.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_uk_fraction(yearly_data):
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


def compile_citations_by_year(publications: pd.DataFrame) -> pd.DataFrame:
    """
    Recreate the citations_by_year data combining cited_by_count, n_pubs, and year
    to create a new column with a list of three elements.

    NB. This was thought necessary because the author data only has data from 2012
    onwards, but turns out the work data also has data from 2012 onwards only.
    """
    publications["citations_by_year"] = publications.apply(
        lambda row: [row["year"], row["n_pubs"], row["cited_by_count"]], axis=1
    )

    # rename author_id to id
    publications = publications.rename(columns={"author_id": "id"})

    # create the author group'd version of the counts_by_year data
    author_grouped = publications.groupby("id").agg(
        {"citations_by_year": lambda x: x.tolist()}
    )

    return author_grouped


def process_counts_by_year(row: pd.Series) -> dict:
    """
    Process publication and citation counts from both counts_by_year and citations_by_year data.

    Uses two different approaches:
    1. citations_by_year: Citations attributed to publication year (1980-)
    2. counts_by_year: Citations counted in the year they occur (2012-)

    Args:
        row (pd.Series): Row containing counts_by_year, citations_by_year and earliest_start_date

    Returns:
        dict: Dictionary containing publication and citation metrics
    """
    if pd.isnull(row["earliest_start_date"]) or (
        not isinstance(row["counts_by_year"], np.ndarray)
        and not isinstance(row["citations_by_year"], np.ndarray)
    ):
        return {
            "n_pubs_before": np.nan,  # these use citations_by_year (1980-)
            "n_pubs_after": np.nan,
            "total_citations_pubyear_before": np.nan,
            "total_citations_pubyear_after": np.nan,
            "mean_citations_pubyear_before": np.nan,
            "mean_citations_pubyear_after": np.nan,
            "citations_per_pub_pubyear_before": np.nan,
            "citations_per_pub_pubyear_after": np.nan,
            "mean_annual_citations_before": np.nan,  # these use counts_by_year (2012-)
            "mean_annual_citations_after": np.nan,
            "mean_annual_citations_pp_before": np.nan,
            "mean_annual_citations_pp_after": np.nan,
        }

    ref_year = pd.to_datetime(row["earliest_start_date"]).year

    # Process citations_by_year (citations attributed to publication year)
    pubs_before = []
    pubs_after = []
    citations_pubyear_before = []
    citations_pubyear_after = []

    if isinstance(row["citations_by_year"], np.ndarray):
        for year_data in row["citations_by_year"]:
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
                citations_pubyear_before.append(citations)
            else:
                pubs_after.append(works)
                citations_pubyear_after.append(citations)

    # Process counts_by_year (citations counted in year they occur)
    years_before = []
    years_after = []
    annual_citations_before = []
    annual_citations_after = []
    annual_pubs_before = []
    annual_pubs_after = []

    if isinstance(row["counts_by_year"], np.ndarray):
        for year_data in row["counts_by_year"]:
            if not isinstance(year_data, np.ndarray) or len(year_data) < 3:
                continue

            try:
                year = int(year_data[0])
                n_pubs = int(year_data[1])
                citations = int(year_data[2])
            except (ValueError, TypeError):
                continue

            if year < ref_year:
                years_before.append(year)
                annual_citations_before.append(citations)
                annual_pubs_before.append(n_pubs)
            else:
                years_after.append(year)
                annual_citations_after.append(citations)
                annual_pubs_after.append(n_pubs)

    # Calculate metrics for citations_by_year
    total_pubs_before = sum(pubs_before) if pubs_before else 0
    total_pubs_after = sum(pubs_after) if pubs_after else 0
    total_citations_pubyear_before = (
        sum(citations_pubyear_before) if citations_pubyear_before else 0
    )
    total_citations_pubyear_after = (
        sum(citations_pubyear_after) if citations_pubyear_after else 0
    )

    # Calculate metrics for counts_by_year
    mean_annual_citations_before = (
        round(np.mean(annual_citations_before), 3)
        if annual_citations_before
        else np.nan
    )
    mean_annual_citations_after = (
        round(np.mean(annual_citations_after), 3) if annual_citations_after else np.nan
    )

    mean_annual_pubs_before = (
        np.mean(annual_pubs_before) if annual_pubs_before else np.nan
    )
    mean_annual_pubs_after = np.mean(annual_pubs_after) if annual_pubs_after else np.nan

    return {
        # metrics using citations_by_year (1980-)
        "n_pubs_before": total_pubs_before if total_pubs_before > 0 else np.nan,
        "n_pubs_after": total_pubs_after if total_pubs_after > 0 else np.nan,
        "total_citations_pubyear_before": (
            total_citations_pubyear_before
            if total_citations_pubyear_before > 0
            else np.nan
        ),
        "total_citations_pubyear_after": (
            total_citations_pubyear_after
            if total_citations_pubyear_after > 0
            else np.nan
        ),
        "mean_citations_pubyear_before": (
            round(np.mean(citations_pubyear_before), 3)
            if citations_pubyear_before
            else np.nan
        ),
        "mean_citations_pubyear_after": (
            round(np.mean(citations_pubyear_after), 3)
            if citations_pubyear_after
            else np.nan
        ),
        "citations_per_pub_pubyear_before": (
            round(total_citations_pubyear_before / total_pubs_before, 3)
            if total_pubs_before > 0
            else np.nan
        ),
        "citations_per_pub_pubyear_after": (
            round(total_citations_pubyear_after / total_pubs_after, 3)
            if total_pubs_after > 0
            else np.nan
        ),
        # metrics using counts_by_year (2012-)
        "mean_annual_citations_before": mean_annual_citations_before,
        "mean_annual_citations_after": mean_annual_citations_after,
        "mean_annual_citations_pp_before": (
            round(mean_annual_citations_before / mean_annual_pubs_before, 3)
            if mean_annual_pubs_before and mean_annual_citations_before is not np.nan
            else np.nan
        ),
        "mean_annual_citations_pp_after": (
            round(mean_annual_citations_after / mean_annual_pubs_after, 3)
            if mean_annual_pubs_after and mean_annual_citations_after is not np.nan
            else np.nan
        ),
    }


def process_fwci(row: pd.Series, pubs_df: pd.DataFrame) -> tuple:
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
"""
Utility functions for computing publication-related metrics.

This module provides specialised functions for analysing publication data and computing
various publication-related metrics, including:
- Citation impact metrics
- Publication counts and temporal patterns
- Field-Weighted Citation Impact (FWCI)
- UK vs international publication patterns
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compile_counts_by_pubyear(publications: pd.DataFrame) -> pd.DataFrame:
    """
    Compile publication counts and citation data by publication year.

    This function reorganises publication data to create year-wise summaries of
    publication counts and citation impact. It combines cited_by_count, publication
    counts, and year data into a structured format for further analysis.

    Note: While this was initially implemented to handle historical data pre-2012,
    the current OpenAlex data only extends back to 2012.

    Args:
        publications (pd.DataFrame): DataFrame containing publication records with
            columns for year, publication counts, and citation counts

    Returns:
        pd.DataFrame: Author-grouped DataFrame with a 'counts_by_pubyear' column
            containing lists of [year, n_pubs, cited_by_count] for each publication
    """
    publications["counts_by_pubyear"] = publications.apply(
        lambda row: [row["year"], row["n_pubs"], row["cited_by_count"]], axis=1
    )

    # rename author_id to id
    publications = publications.rename(columns={"author_id": "id"})

    # create the author group'd version of the counts_by_year data
    author_grouped = publications.groupby("id").agg(
        {"counts_by_pubyear": lambda x: x.tolist()}
    )

    return author_grouped


def process_counts_by_year(row: pd.Series) -> dict:
    """
    Process publication and citation counts using multiple counting approaches.

    This function analyses publication and citation data using two distinct
    methodologies:
    1. Publication Year Attribution (1980 onwards):
       - Citations are attributed to the year of publication
       - Provides insight into the long-term impact of publications
       - Better for career-long analysis

    2. Citation Year Counting (2012 onwards):
       - Citations are counted in the year they occur
       - Better reflects current impact and citation patterns
       - More suitable for recent impact analysis

    Args:
        row (pd.Series): Row containing:
            - counts_by_year: Array of [year, n_pubs, citations] for citation years
            - counts_by_pubyear: Array of [year, n_pubs, citations] for pub years
            - earliest_start_date: Date of first research grant

    Returns:
        dict: Dictionary containing various publication and citation metrics:
            Publication Year Metrics (1980-):
            - n_pubs_before/after: Publication counts
            - total_citations_pubyear_before/after: Total citations
            - mean_citations_pubyear_before/after: Mean citations per year
            - citations_pp_pubyear_before/after: Citations per publication

            Citation Year Metrics (2012-):
            - mean_citations_before/after: Mean annual citations
            - citations_pp_before/after: Citations per publication per year
    """
    if pd.isnull(row["earliest_start_date"]) or (
        not isinstance(row["counts_by_year"], np.ndarray)
        and not isinstance(row["counts_by_pubyear"], np.ndarray)
    ):
        return {
            "n_pubs_before": np.nan,  # these use counts_by_pubyear (1980-)
            "n_pubs_after": np.nan,
            "total_citations_pubyear_before": np.nan,
            "total_citations_pubyear_after": np.nan,
            "mean_citations_pubyear_before": np.nan,
            "mean_citations_pubyear_after": np.nan,
            "citations_pp_pubyear_before": np.nan,
            "citations_pp_pubyear_after": np.nan,
            "mean_citations_before": np.nan,  # these use counts_by_year (2012-)
            "mean_citations_after": np.nan,
            "citations_pp_before": np.nan,
            "citations_pp_after": np.nan,
        }

    ref_year = pd.to_datetime(row["earliest_start_date"]).year

    # Process counts_by_pubyear (citations attributed to publication year)
    pubs_before = []
    pubs_after = []
    citations_pubyear_before = []
    citations_pubyear_after = []

    if isinstance(row["counts_by_pubyear"], list):
        for year_data in row["counts_by_pubyear"]:
            if not isinstance(year_data, list) or len(year_data) < 3:
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

    # Calculate metrics for counts_by_pubyear
    total_pubs_before = sum(pubs_before) if pubs_before else 0
    total_pubs_after = sum(pubs_after) if pubs_after else 0
    total_citations_pubyear_before = (
        sum(citations_pubyear_before) if citations_pubyear_before else 0
    )
    total_citations_pubyear_after = (
        sum(citations_pubyear_after) if citations_pubyear_after else 0
    )

    # Calculate metrics for counts_by_year
    mean_citations_before = (
        round(np.nanmean(annual_citations_before), 3)
        if annual_citations_before
        else np.nan
    )
    mean_citations_after = (
        round(np.nanmean(annual_citations_after), 3)
        if annual_citations_after
        else np.nan
    )

    mean_annual_pubs_before = (
        np.nanmean(annual_pubs_before) if annual_pubs_before else np.nan
    )
    mean_annual_pubs_after = (
        np.nanmean(annual_pubs_after) if annual_pubs_after else np.nan
    )

    return {
        # metrics using counts_by_pubyear (1980-)
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
            round(np.nanmean(citations_pubyear_before), 3)
            if citations_pubyear_before
            else np.nan
        ),
        "mean_citations_pubyear_after": (
            round(np.nanmean(citations_pubyear_after), 3)
            if citations_pubyear_after
            else np.nan
        ),
        "citations_pp_pubyear_before": (
            round(total_citations_pubyear_before / total_pubs_before, 3)
            if total_pubs_before > 0
            else np.nan
        ),
        "citations_pp_pubyear_after": (
            round(total_citations_pubyear_after / total_pubs_after, 3)
            if total_pubs_after > 0
            else np.nan
        ),
        # metrics using counts_by_year (2012-)
        "mean_citations_before": mean_citations_before,
        "mean_citations_after": mean_citations_after,
        "citations_pp_before": (
            round(mean_citations_before / mean_annual_pubs_before, 3)
            if mean_annual_pubs_before and mean_citations_before is not np.nan
            else np.nan
        ),
        "citations_pp_after": (
            round(mean_citations_after / mean_annual_pubs_after, 3)
            if mean_annual_pubs_after and mean_citations_after is not np.nan
            else np.nan
        ),
    }


def compile_fwci_by_author(publications: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-aggregate Field-Weighted Citation Impact (FWCI) data by author.

    This function processes publication-level FWCI data to create author-specific
    summaries. FWCI is a measure that indicates how the number of citations
    received by a publication compares to the average number of citations received
    by similar publications.

    Args:
        publications (pd.DataFrame): DataFrame containing publication records with
            columns for author_id, year, and fwci

    Returns:
        dict: Dictionary mapping author_ids to lists of [year, fwci] pairs,
            representing their FWCI scores across different years
    """
    # Group FWCI data by author and year
    fwci_data = publications.groupby(["author_id", "year"])["fwci"].mean().reset_index()

    # Create a dictionary mapping author_id to their yearly FWCI data
    author_fwci = (
        fwci_data.groupby("author_id")
        .apply(lambda x: x[["year", "fwci"]].values.tolist())
        .to_dict()
    )

    return author_fwci


def process_fwci(row: pd.Series, author_fwci: dict) -> tuple:
    """
    Process Field-Weighted Citation Impact (FWCI) metrics from pre-aggregated data.

    This function calculates average FWCI scores for periods before and after a
    researcher's first grant. FWCI is normalised by field, publication type, and
    publication year, making it suitable for cross-field comparisons.

    Args:
        row (pd.Series): Row containing researcher data including earliest_start_date
        author_fwci (dict): Dictionary mapping author IDs to lists of [year, fwci]
            pairs from pre-aggregated FWCI data

    Returns:
        tuple: (mean_fwci_before, mean_fwci_after) containing average FWCI scores
            for periods before and after the researcher's first grant
    """
    if pd.isnull(row["earliest_start_date"]):
        return np.nan, np.nan

    ref_year = pd.to_datetime(row["earliest_start_date"]).year

    # Get author's FWCI data
    author_data = author_fwci.get(row["id"], [])

    fwci_before = []
    fwci_after = []

    # Split FWCI values into before/after
    for year, fwci in author_data:
        if year < ref_year:
            fwci_before.append(fwci)
        else:
            fwci_after.append(fwci)
    mean_fwci_before = (
        round(np.nanmean(fwci_before), 3) if len(fwci_before) > 0 else np.nan
    )
    mean_fwci_after = (
        round(np.nanmean(fwci_after), 3) if len(fwci_after) > 0 else np.nan
    )

    return mean_fwci_before, mean_fwci_after

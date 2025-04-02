"""
Utility functions for computing basic metrics.
"""

# pylint: disable=E0402

import logging
import pandas as pd
from tqdm import tqdm
from .publication_metrics import (
    compile_counts_by_pubyear,
    process_counts_by_year,
    process_fwci,
    compile_fwci_by_author,
)
from .affiliation_metrics import (
    process_last_institution,
    process_collaborations,
    process_affiliations,
)

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
    if pd.isnull(row["earliest_start_date"]) or pd.isnull(row["first_work_year"]):
        return None
    return (
        pd.to_datetime(f"{row['earliest_start_date']}-01-01").year
        - pd.to_datetime(f"{row['first_work_year']}-01-01").year
    )


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
    # Merge reference dates to publications
    publications = publications.merge(
        df[["id", "earliest_start_date"]],
        left_on="author_id",
        right_on="id",
        how="left",
    )

    logger.info("Processing affiliation metrics...")
    affiliation_metrics = publications.groupby("author_id").progress_apply(
        process_affiliations
    )

    logger.info("Processing collaboration metrics...")
    collab_metrics = publications.groupby("author_id").progress_apply(
        process_collaborations
    )

    # Map metrics back to original dataframe
    affiliation_fields = [
        "abroad_experience_before",
        "abroad_experience_after",
        "countries_before",
        "countries_after",
        "abroad_fraction_before",
        "abroad_fraction_after",
    ]

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

    # Map affiliation metrics
    for field in affiliation_fields:
        df[field] = df["id"].map(affiliation_metrics.apply(lambda x: x[field]))

    # Map collaboration metrics
    for field in collab_fields:
        df[field] = df["id"].map(collab_metrics.apply(lambda x: x[field]))

    # Add last known institution info
    df["last_known_institution_uk"] = df.progress_apply(
        process_last_institution, axis=1
    )

    # Convert integer fields to Int64 to handle missing values
    int_fields = [
        "total_collabs_before",
        "total_collabs_after",
        "unique_collabs_before",
        "unique_collabs_after",
    ]
    df[int_fields] = df[int_fields].astype("Int64")

    return df


def add_publication_metrics(
    df: pd.DataFrame,
    publications: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add publication metrics including FWCI, citations, and publication counts
    before and after first grant.

    Args:
        df (pd.DataFrame): DataFrame with counts_by_year and earliest_start_date
        publications (pd.DataFrame): DataFrame with yearly FWCI data

    Returns:
        pd.DataFrame: DataFrame with added publication metrics
    """
    logger.info("Processing publication metrics...")
    counts_by_publication_year = compile_counts_by_pubyear(publications)

    # replace the counts_by_year column in the df with the author group'd version
    df["counts_by_pubyear"] = df["id"].map(
        counts_by_publication_year["counts_by_pubyear"]
    )

    # get first publication year for each author and map to main dataframe
    df["first_work_year"] = (
        df["id"].map(publications.groupby("author_id")["year"].min()).astype("Int64")
    )

    # Process counts_by_year data
    logger.info("Processing citation and publication counts...")
    metrics = df.progress_apply(process_counts_by_year, axis=1)

    for field in [
        "n_pubs_before",
        "n_pubs_after",
        "total_citations_pubyear_before",
        "total_citations_pubyear_after",
        "mean_citations_pubyear_before",
        "mean_citations_pubyear_after",
        "citations_pp_pubyear_before",
        "citations_pp_pubyear_after",
        "mean_citations_before",
        "mean_citations_after",
        "citations_pp_before",
        "citations_pp_after",
    ]:
        df[field] = metrics.apply(lambda x: x[field])  # pylint: disable=W0640

    # Pre-aggregate FWCI data
    logger.info("Pre-aggregating FWCI data...")
    author_fwci = compile_fwci_by_author(publications)

    # Process FWCI data
    logger.info("Processing FWCI metrics...")
    fwci_metrics = df.progress_apply(lambda x: process_fwci(x, author_fwci), axis=1)
    df["mean_fwci_before"] = fwci_metrics.apply(lambda x: x[0])
    df["mean_fwci_after"] = fwci_metrics.apply(lambda x: x[1])

    # Convert integer fields to Int64 to handle missing values
    int_fields = [
        "n_pubs_before",
        "n_pubs_after",
        "total_citations_pubyear_before",
        "total_citations_pubyear_after",
    ]
    for field in int_fields:
        df[field] = df[field].astype("Int64")

    return df

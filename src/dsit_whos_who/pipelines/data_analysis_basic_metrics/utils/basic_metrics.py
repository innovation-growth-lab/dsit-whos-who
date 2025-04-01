"""
Utility functions for computing basic metrics.
"""

# pylint: disable=E0402

import logging
import pandas as pd
from tqdm import tqdm
from .publication_metrics import (
    compile_citations_by_year,
    process_counts_by_year,
    process_fwci,
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
    # Apply the affiliation processing
    affiliation_metrics = df.progress_apply(process_affiliations, axis=1)

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
        process_last_institution, axis=1
    )

    logger.info("Processing publication collaborations...")
    collab_metrics = df.progress_apply(
        lambda x: process_collaborations(x, publications), axis=1
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
    counts_by_publication_year = compile_citations_by_year(publications)

    # replace the counts_by_year column in the df with the author group'd version
    df["citations_by_year"] = df["id"].map(
        counts_by_publication_year["citations_by_year"]
    )

    # Get first publication year for each author and map to main dataframe
    df["first_work_year"] = df["id"].map(publications.groupby("id")["year"].min())

    # Process counts_by_year data
    logger.info("Processing citation and publication counts...")
    metrics = df.progress_apply(process_counts_by_year, axis=1)

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
    fwci_metrics = df.progress_apply(lambda x: process_fwci(x, publications), axis=1)
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


# oa work only has up to 2012 as well.
# so we rely on the poorer, all citations go to publication year approach. This also requires changing the way the first_work_year is computed.

# To do:
# - rerun the collection to get (again) the cited_by_count to create the "true" citations_by_year data.
# - preprocess authors without the first_work_year (this goes only up to 2012).
# - run the above, use the "true" citations_by_year data to create metrics, but still report the counts_by_year data.

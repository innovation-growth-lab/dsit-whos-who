"""
Utility functions for computing complex metrics.

This module provides functions for calculating and analysing complex bibliometric metrics
for academic authors, including:

- Disruption indices that measure how much a paper disrupts vs consolidates its research field
- Diversity metrics that assess the breadth and variety of an author's research portfolio
- Before/after analysis to compare metrics before and after key career events like first funding

The module handles:
1. Preprocessing and merging disruption indices with publication data
2. Computing weighted and unweighted disruption metrics at author-year level
3. Processing diversity metrics including variety, evenness and disparity
4. Calculating aggregate metrics for pre/post analysis around reference timepoints

The functions expect standardised input DataFrames containing publication records, author
information, and precomputed metrics. They output processed metrics suitable for further
statistical analysis.
"""

import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

logger = logging.getLogger(__name__)


def preprocess_disruption_to_merge_with_publications(
    disruption_indices: pd.DataFrame,
    publications: pd.DataFrame,
    basic_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Preprocess disruption indices data to merge with publications data.

    This function takes disruption indices, publications and authors data and:
    1. Formats disruption indices IDs to match publication IDs
    2. Merges disruption indices with publications
    3. Filters for valid disruption index status
    4. Extracts author IDs from publication authorships
    5. Processes publication dates into years
    6. Filters for authors present in the provided authors DataFrame
    7. Prepares data for author-year level disruption metric calculations

    Args:
        disruption_indices (pd.DataFrame): DataFrame containing disruption indices with columns:
            id, n_f, n_b, total, di_status, disruption_index
        publications (pd.DataFrame): DataFrame containing publication data with columns:
            id, authorships, fwci, publication_date
        basic_metrics (pd.DataFrame): DataFrame containing author data with column:
            oa_id (OpenAlex author IDs)

    Returns:
        pd.DataFrame: Processed DataFrame containing publication-author level data with columns:
            author, year, disruption_index, fwci, disruption_index_weighted, author_year_disruption
    """
    logger.info("Preprocess disruption to merge with publications")
    disruption_indices["id"] = "W" + disruption_indices["id"].astype(str)
    # make n_f, n_b and total into an int format that accepts nulls
    disruption_indices["n_f"] = disruption_indices["n_f"].astype(pd.Int64Dtype())
    disruption_indices["n_b"] = disruption_indices["n_b"].astype(pd.Int64Dtype())
    disruption_indices["total"] = disruption_indices["total"].astype(pd.Int64Dtype())

    # merge on id
    publications = publications.merge(disruption_indices, on="id", how="left")

    # keep if di_status is valid
    publications = publications[publications["di_status"] == "valid"]

    logger.info("Work out the authors from publications")
    # get a list of authors from authorships
    publications["authorships"] = publications["authorships"].apply(
        lambda x: [author[0] for author in x]
    )

    # extract year from publication_date
    publications["year"] = pd.to_datetime(publications["publication_date"]).dt.year

    # keep relevant cols
    disruption_publications = publications[
        ["id", "authorships", "fwci", "year", "disruption_index", "n_f", "n_b", "total"]
    ]

    # explode authorships
    disruption_publications = disruption_publications.explode("authorships")

    # keep publications with authors in authors["oa_id"]
    disruption_publications = disruption_publications[
        disruption_publications["authorships"].isin(basic_metrics["oa_id"])
    ]

    # keep unique id, authorships
    disruption_publications = disruption_publications.drop_duplicates(
        subset=["id", "authorships"]
    )

    logger.info("Calculating author-year level disruption metrics")

    # fill fwci nan with 0
    disruption_publications["fwci_weights"] = disruption_publications["fwci"].fillna(
        0.01
    )
    # Calculate mean disruption index
    author_year_disruption = (
        disruption_publications.groupby(["authorships", "year"])
        .agg({"disruption_index": "mean", "fwci": "mean"})
        .reset_index()
    )

    # Calculate weighted mean disruption index, falling back to simple mean when weights sum to zero
    weighted_metrics = (
        disruption_publications.groupby(["authorships", "year"])
        .progress_apply(_weighted_mean_with_fallback)
        .reset_index()
    )
    weighted_metrics.columns = ["authorships", "year", "disruption_index_weighted"]

    logger.info("Merge weighted metrics")
    author_year_metrics = author_year_disruption.merge(
        weighted_metrics, on=["authorships", "year"], how="left"
    )

    # Rename columns
    author_year_metrics.columns = [
        "author",
        "year",
        "disruption_index",
        "fwci",
        "disruption_index_weighted",
    ]

    # sort by author and year
    author_year_metrics = author_year_metrics.sort_values(["author", "year"])

    # create a column that is a list of the author's id, year, and disruption_index_mean
    author_year_metrics["author_year_disruption"] = author_year_metrics.apply(
        lambda row: [
            str(row["year"]),
            str(round(row["disruption_index"], 2)),
            str(round(row["disruption_index_weighted"], 2)),
        ],
        axis=1,
    )

    return author_year_metrics


def _weighted_mean_with_fallback(group):
    try:
        return np.average(group["disruption_index"], weights=group["fwci_weights"])
    except ZeroDivisionError:
        return np.nan


def process_disruption_metrics(author_disruption: pd.Series, ref_year: int) -> dict:
    """
    Process disruption metrics for an author before and after their first funding.

    Args:
        author_disruption (pd.Series): Series containing author's yearly disruption metrics
        ref_year (int): Reference year (first funding year)

    Returns:
        dict: Dictionary containing mean disruption metrics before and after first funding
    """
    if pd.isna(ref_year):
        return {
            "mean_disruption_before": np.nan,
            "mean_disruption_after": np.nan,
            "mean_weighted_disruption_before": np.nan,
            "mean_weighted_disruption_after": np.nan,
        }

    metrics_before = []
    metrics_after = []
    weighted_before = []
    weighted_after = []

    for year_data in author_disruption:
        try:
            year = int(year_data[0])
            disruption = float(year_data[1])
            weighted_disruption = float(year_data[2])

            if year < ref_year:
                metrics_before.append(disruption)
                weighted_before.append(weighted_disruption)
            else:
                metrics_after.append(disruption)
                weighted_after.append(weighted_disruption)
        except (ValueError, IndexError):
            continue

    return {
        "mean_disruption_before": (
            round(np.nanmean(metrics_before), 3) if metrics_before else np.nan
        ),
        "mean_disruption_after": (
            round(np.nanmean(metrics_after), 3) if metrics_after else np.nan
        ),
        "mean_weighted_disruption_before": (
            round(np.nanmean(weighted_before), 3) if weighted_before else np.nan
        ),
        "mean_weighted_disruption_after": (
            round(np.nanmean(weighted_after), 3) if weighted_after else np.nan
        ),
    }


def process_diversity_metrics(author_diversity: pd.Series, ref_year: int) -> dict:
    """
    Process diversity metrics for an author before and after their first funding.

    Args:
        author_diversity (pd.Series): Series containing author's yearly diversity metrics
        ref_year (int): Reference year (first funding year)

    Returns:
        dict: Dictionary containing mean diversity metrics before and after first funding
    """
    if pd.isna(ref_year):
        return {
            "mean_variety_before": np.nan,
            "mean_variety_after": np.nan,
            "mean_evenness_before": np.nan,
            "mean_evenness_after": np.nan,
            "mean_disparity_before": np.nan,
            "mean_disparity_after": np.nan,
        }

    variety_before = []
    variety_after = []
    evenness_before = []
    evenness_after = []
    disparity_before = []
    disparity_after = []

    for year_data in author_diversity:
        try:
            year = int(year_data[0])
            variety = float(year_data[1])
            evenness = float(year_data[2])
            disparity = float(year_data[3])

            if year < ref_year:
                variety_before.append(variety)
                evenness_before.append(evenness)
                disparity_before.append(disparity)
            else:
                variety_after.append(variety)
                evenness_after.append(evenness)
                disparity_after.append(disparity)
        except (ValueError, IndexError):
            continue

    return {
        "mean_variety_before": (
            round(np.nanmean(variety_before), 3) if variety_before else np.nan
        ),
        "mean_variety_after": (
            round(np.nanmean(variety_after), 3) if variety_after else np.nan
        ),
        "mean_evenness_before": (
            round(np.nanmean(evenness_before), 3) if evenness_before else np.nan
        ),
        "mean_evenness_after": (
            round(np.nanmean(evenness_after), 3) if evenness_after else np.nan
        ),
        "mean_disparity_before": (
            round(np.nanmean(disparity_before), 3) if disparity_before else np.nan
        ),
        "mean_disparity_after": (
            round(np.nanmean(disparity_after), 3) if disparity_after else np.nan
        ),
    }


def compute_before_after_metrics(
    author_disruption: pd.DataFrame,
    author_diversity: pd.DataFrame,
    author_earliest_year: pd.Series,
) -> pd.DataFrame:
    """
    Compute mean metrics before and after first funding for each author.

    Args:
        author_disruption (pd.DataFrame): DataFrame with author disruption metrics
        author_diversity (pd.DataFrame): DataFrame with author diversity metrics
        author_earliest_year (pd.Series): Series with author's first funding year

    Returns:
        pd.DataFrame: DataFrame with mean metrics before and after first funding
    """
    logger.info("Computing before/after metrics for each author...")

    results = []
    for author in tqdm(author_earliest_year.index):
        ref_year = author_earliest_year[author]

        # if ref_year is length >1, pick the smallest
        if isinstance(ref_year, pd.Series):
            ref_year = ref_year.min()

        # Process disruption metrics if available
        disruption_metrics = {}
        if author in author_disruption.index:
            disruption_metrics = process_disruption_metrics(
                author_disruption.loc[author]["author_year_disruption"], ref_year
            )

        # Process diversity metrics if available
        diversity_metrics = {}
        if author in author_diversity.index:
            diversity_metrics = process_diversity_metrics(
                author_diversity.loc[author]["author_year_diversity"], ref_year
            )

        # Combine metrics
        combined_metrics = {
            "author": author,
            "first_funding_year": ref_year,
            **disruption_metrics,
            **diversity_metrics,
        }
        results.append(combined_metrics)

    return pd.DataFrame(results)

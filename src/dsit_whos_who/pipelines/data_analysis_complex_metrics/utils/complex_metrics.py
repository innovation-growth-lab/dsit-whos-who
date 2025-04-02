import logging
import numpy as np
import pandas as pd

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
        authors (pd.DataFrame): DataFrame containing author data with column:
            oa_id (OpenAlex author IDs)

    Returns:
        pd.DataFrame: Processed DataFrame containing publication-author level data with columns:
            id, authorships, fwci, year, disruption_index, n_f, n_b, total
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
        .apply(_weighted_mean_with_fallback)
        .reset_index()
    )
    weighted_metrics.columns = ["authorships", "year", "disruption_index_weighted"]

    # Merge weighted metrics
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

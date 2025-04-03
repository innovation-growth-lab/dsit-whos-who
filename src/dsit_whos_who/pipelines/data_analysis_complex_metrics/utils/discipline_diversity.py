"""
This module provides utility functions for computing discipline diversity metrics for academic
authors.

The module implements functions to analyse the diversity of an author's research portfolio across
different scientific disciplines and subfields. It calculates three key components of diversity:

1. Variety - The number of unique disciplines/subfields an author has published in
2. Evenness - How uniformly distributed the author's publications are across disciplines
3. Disparity - How different or distant the disciplines are from each other

The implementation follows established bibliometric approaches:
- Uses the Leydesdorff, Wagner & Bornmann (2019) framework for measuring diversity
- Implements the Kvålseth-Jost measure for evenness as recommended by Rousseau (2023)
- Calculates disparity based on semantic distances between discipline embeddings

Key Functions:
- create_author_and_year_subfield_frequency: Aggregates publication counts by author, year and
    subfield
- calculate_disparity: Computes the average semantic distance between disciplines
- weight_function: Applies temporal weighting to account for changes over time
- calculate_diversity_components: Calculates the three diversity components for each author

The module expects standardised input DataFrames containing publication records with discipline
classifications and pre-computed discipline embedding distances. It outputs processed diversity
metrics suitable for further analysis.
"""

import logging
import re
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def filter_single_list(topic, level):
    """
    Parse out the level-th position from nested topic classification lists.

    Args:
        topic: List containing nested topic classifications
        level: Integer specifying which level to extract

    Returns:
        int: The numeric ID at the specified level, or np.nan if not found
    """
    matches = re.findall(r"\d+", topic[level])
    return int(matches[0]) if matches else np.nan


def _compute_frequency_arrays(topic_counts, author_counts, topic_to_col, n_topics):
    """
    Efficiently compute (n_topics,) dimensional frequency arrays for all author-year combinations.

    Args:
        topic_counts (pd.DataFrame): DataFrame with columns: author, year, topic_id, frequency.
        author_counts (pd.DataFrame): DataFrame with columns: author, year.
        topic_to_col (dict): Mapping of topic IDs to column indices.
        n_topics (int): Total number of unique topics.

    Returns:
        np.ndarray: Array of shape (n_author_years, n_topics), where each row is a topic frequency array.
    """
    # map author-year combinations to row indices
    author_year_map = {
        tuple(row): i for i, row in author_counts[["author", "year"]].iterrows()
    }
    topic_counts["row_index"] = topic_counts.apply(
        lambda row: author_year_map[(row["author"], row["year"])], axis=1
    )

    # map topic IDs to column indices
    topic_counts["col_index"] = topic_counts["subfield_ids"].map(topic_to_col)

    # initialise an empty array for all frequencies
    frequency_matrix = np.zeros((len(author_counts), n_topics), dtype=int)

    # populate the matrix
    for _, row in topic_counts.iterrows():
        frequency_matrix[row["row_index"], row["col_index"]] += row["frequency"]

    # convert each row into a list of frequencies
    return list(frequency_matrix)


def create_author_and_year_subfield_frequency(
    data: pd.DataFrame, cwts_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregates taxonomy subfields by author and year, and adds publication counts.

    Args:
        data (pd.DataFrame): Input DataFrame with columns 'id', 'author', 'publication_date',
            and 'subfield_ids'
        cwts_data (pd.DataFrame): DataFrame containing the CWTS classification system data

    Returns:
        pd.DataFrame: DataFrame with author, year, frequency arrays and publication counts
    """

    topic_to_col = {topic: i for i, topic in enumerate(sorted(cwts_data))}

    # aggregate counts for authors
    author_counts = (
        data.groupby(["author", "year"]).size().reset_index(name="publications")
    )

    # flatten topics and create frequency arrays
    flattened_topics = data.explode("subfield_ids").dropna(subset=["subfield_ids"])
    topic_counts = (
        flattened_topics.groupby(["author", "year", "subfield_ids"])
        .size()
        .reset_index(name="frequency")
    )

    # create (n_topics,) frequency arrays
    frequency_arrays = _compute_frequency_arrays(
        topic_counts, author_counts, topic_to_col, len(cwts_data)
    )
    author_counts["frequency"] = frequency_arrays

    return author_counts


def calculate_disparity(x_row: np.array, d: np.array) -> float:
    """
    Calculates the disparity between elements in the given array.

    Args:
        x_row (np.array): Array of frequencies for each topic
        d (np.array): Matrix of pairwise distances between topics

    Returns:
        float: The calculated disparity value between 0 and 1
    """
    non_unit_indices = np.where(x_row >= 1)[0]
    num_non_unit = len(non_unit_indices)
    if num_non_unit <= 1:
        return 0.0

    disparity_sum = 0.0
    for i in range(num_non_unit):
        for j in range(i + 1, num_non_unit):
            disparity_sum += d[non_unit_indices[i], non_unit_indices[j]]
    return disparity_sum / ((num_non_unit * (num_non_unit - 1)) / 2)


def weight_function(delta_year, alpha=1):
    """
    Compute weight based on the time difference between publications.

    Args:
        delta_year (int): The difference between publication years
        alpha (float): Smoothing factor (higher = steeper weight dropoff)

    Returns:
        float: Weight between 0 and 1
    """
    return 1 / (1 + alpha * abs(delta_year))


def calculate_diversity_components(
    data: pd.DataFrame, disparity_matrix: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate diversity components based on the given data and disparity matrix.

    The diversity measure builds from Leydesdorff, Wagner, and Bornmann (2019) and consists
    of three components:

    - Variety: The number of unique topics an author has published on
    - Evenness: The distribution of publications across topics
    - Disparity: The diversity of topics an author has published on

    The implementation follows Rousseau's (2023) suggestion to use the Kvålseth-Jost measure for
    evenness, which is a generalisation of the Gini coefficient presented by Jost (2006) and
    included in the meta discussion paper by Chao and Ricotta (2023).

    Args:
        data (pd.DataFrame): DataFrame containing frequency arrays by author and year
        disparity_matrix (pd.DataFrame): Matrix of pairwise distances between topics

    Returns:
        pd.DataFrame: Original data with added columns for variety, evenness and disparity
    """
    x_matrix = data[["frequency"]].to_numpy()
    x_matrix = np.vstack(x_matrix[:, 0])
    disparity_matrix = disparity_matrix.to_numpy()
    data.drop(columns=["frequency"], inplace=True)

    # compute variety
    logger.info("Calculating variety")
    n = x_matrix.shape[1]
    nx = np.sum(x_matrix >= 1, axis=1)
    variety = nx / n

    # can we mask < 1 values to be 0?
    x_matrix = np.where(x_matrix < 1, 0, x_matrix)

    # compute eveness using the Kvålseth-Jost measure for each row
    logger.info("Calculating evenness")
    q = 2
    with np.errstate(divide="ignore", invalid="ignore"):
        p_matrix = x_matrix / np.sum(x_matrix, axis=1, keepdims=True)
        evenness = np.sum(p_matrix**q, axis=1) ** (1 / (1 - q)) - 1
        evenness = np.nan_to_num(evenness / (nx - 1), nan=0.0)

    # compute disparity
    logger.info("Calculating disparity")
    disparity = np.array(
        [calculate_disparity(row, disparity_matrix) for row in x_matrix]
    )

    logger.info("Diversity components calculated")
    diversity_components = data.copy()
    diversity_components["variety"] = variety
    diversity_components["evenness"] = evenness
    diversity_components["disparity"] = disparity

    return diversity_components

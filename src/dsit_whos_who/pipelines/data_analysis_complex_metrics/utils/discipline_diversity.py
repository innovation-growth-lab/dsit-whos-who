"""
This script provides utility functions for computing topic embeddings, distance
matrices, and diversity components for a given dataset.

Functions:
    - compute_distance_matrix(embeddings: np.ndarray, ids: list) -> pd.DataFrame:
        Computes the pairwise distance matrix between embeddings and normalises
        the matrix.
    - _filter_single_list(topic, level):
        Extracts the specified level from a nested list of topics.
    - _compute_frequency_arrays(topic_counts: pd.DataFrame, author_counts:
        pd.DataFrame, topic_to_col: dict, n_topics: int) -> list:
        Efficiently computes topic frequency arrays for author-year combinations.
    - create_author_and_year_frequency(data: pd.DataFrame, level: int, cwts_data:
        pd.DataFrame) -> pd.DataFrame:
        Aggregates taxonomy data by author and year and creates topic frequency
        arrays.
    - calculate_disparity(x_row: np.array, d: np.array) -> float:
        Calculates disparity as the average distance between elements in the
        given array based on a disparity matrix.

"""

import re
import pandas as pd
import numpy as np


def filter_single_list(topic, level):
    """Util function to parse out the "level"th position of nested lists"""
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
        df (pd.DataFrame): Input DataFrame with columns 'id', 'author', 'publication_date',
            and 'topics', where 'topics' is a list of dictionaries with keys 'topic', 'subfield',
            'field', and 'domain'.

    Returns:
        pd.DataFrame: DataFrame with 'author_id', 'year', 'topics', 'yearly_publication_count',
            and 'total_publication_count' aggregated.
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
        x_row (np.array): The input array.
        d (np.array): The disparity matrix.

    Returns:
        float: The calculated disparity.

    """
    non_zero_indices = np.nonzero(x_row)[0]
    num_non_zero = len(non_zero_indices)
    if num_non_zero <= 1:
        return 0.0

    disparity_sum = 0.0
    for i in range(num_non_zero):
        for j in range(i + 1, num_non_zero):
            disparity_sum += d[non_zero_indices[i], non_zero_indices[j]]
    return disparity_sum / ((num_non_zero * (num_non_zero - 1)) / 2)

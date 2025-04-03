"""
Utility functions for working with embeddings and computing distance matrices.

This module provides functions for processing and analysing embeddings data, including:

- Computing normalised distance matrices between embeddings
- Aggregating embeddings by groups and calculating distances
- Handling high-dimensional embedding vectors efficiently

The module handles:
1. Computing pairwise Euclidean distances between embedding vectors
2. Normalising distance matrices to [0,1] range
3. Aggregating embeddings by specified groupings
4. Converting between different matrix formats

The functions expect standardised input arrays containing embedding vectors and 
corresponding IDs. They output processed distance matrices suitable for further analysis.
"""

# pylint: disable=E0402

import logging
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)


def compute_distance_matrix(embeddings: np.ndarray, ids: list) -> pd.DataFrame:
    """
    Compute the distance matrix between embeddings and return a normalised matrix.

    Parameters:
        embeddings (numpy.ndarray): An array of shape (n_samples, n_features) containing the
            embeddings vectors.
        ids (list): A list of length n_samples containing the unique identifiers corresponding 
            to each embedding.

    Returns:
        pd.DataFrame: A DataFrame of shape (n_samples, n_samples) containing the normalised
            distance matrix with values scaled to [0,1].
    """
    distance_matrix = squareform(pdist(embeddings, "euclidean"))
    min_value = np.min(distance_matrix[distance_matrix >= 0])
    max_value = np.max(distance_matrix)
    normalised_matrix = (distance_matrix - min_value) / (max_value - min_value)
    np.fill_diagonal(normalised_matrix, 0)
    return pd.DataFrame(normalised_matrix, index=ids, columns=ids)


def aggregate_embeddings_and_compute_matrix(
    data: pd.DataFrame, group_by_col: str, embeddings_col: str
) -> pd.DataFrame:
    """
    Aggregates embeddings by group and computes a distance matrix based on the aggregated embeddings.

    Parameters:
        data (pandas.DataFrame): The input DataFrame containing the embeddings data.
        group_by_col (str): The column name to group the data by.
        embeddings_col (str): The column name containing the embeddings vectors.

    Returns:
        pd.DataFrame: A distance matrix computed from the mean embeddings of each group.
    """
    grouped_data = data.groupby(group_by_col)[embeddings_col].apply(
        lambda x: np.mean(np.vstack(x), axis=0)
    )
    aggregated_embeddings = grouped_data.tolist()
    ids = grouped_data.index.tolist()
    return compute_distance_matrix(np.array(aggregated_embeddings), ids)

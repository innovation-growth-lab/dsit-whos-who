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
            embeddings.
        ids (list): A list of length n_samples containing the IDs corresponding to each
            embedding.

    Returns:
        pd.DataFrame: A DataFrame of shape (n_samples, n_samples) containing the normalised
            distance matrix.
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
    Aggregates embeddings and computes a distance matrix based on the aggregated embeddings.

    Args:
        data (pandas.DataFrame): The input data containing the embeddings.
        group_by_col (str): The column to group the data by.
        embeddings_col (str): The column containing the embeddings.

    Returns:
        pd.DataFrame: The distance matrix based on the aggregated embeddings.
    """
    grouped_data = data.groupby(group_by_col)[embeddings_col].apply(
        lambda x: np.mean(np.vstack(x), axis=0)
    )
    aggregated_embeddings = grouped_data.tolist()
    ids = grouped_data.index.tolist()
    return compute_distance_matrix(np.array(aggregated_embeddings), ids)


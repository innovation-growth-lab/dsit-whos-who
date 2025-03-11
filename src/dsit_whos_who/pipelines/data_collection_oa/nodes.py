"""
Tmplat.e
"""

import logging
from typing import List, Dict, Union, Callable
import pandas as pd
from joblib import Parallel, delayed
from kedro.io import AbstractDataset
from .utils.common import fetch_openalex_objects, preprocess_ids
from .utils.authors import json_loader_authors
from .utils.publications import json_loader_works


logger = logging.getLogger(__name__)


def preprocess_publication_doi(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the Gateway to Research publication data to include
    doi values that are compatible with OA filter module.

    Args:
        df (pd.DataFrame): The Gateway to Research publication data.

    Returns:
        pd.DataFrame: The preprocessed publication data.
    """
    if "doi" in df.columns:
        df["doi"] = df["doi"].str.extract(r"(10\..+)")
    return df


def create_list_doi_inputs(df: pd.DataFrame, **kwargs) -> list:
    """Create a list of doi values from the Gateway to Research publication data.

    Args:
        df (pd.DataFrame): The Gateway to Research publication data.

    Returns:
        list: A list of doi values.
    """
    doi_singleton_list = df[df["doi"].notnull()]["doi"].drop_duplicates().tolist()

    # concatenate doi values to create group querise
    doi_list = preprocess_ids(doi_singleton_list, kwargs.get("grouped", True))

    return doi_list


def create_list_orcid_inputs(df: pd.DataFrame, **kwargs) -> list:
    """Create a list of orcid values from the Gateway to Research author data.

    Args:
        df (pd.DataFrame): The Gateway to Research author data.

    Returns:
        list: A list of orcid values.
    """
    orcid_singleton_list = (
        df[df["orcid_id"].notnull()]["orcid_id"].drop_duplicates().tolist()
    )

    # concatenate orcid values to create group querise
    orcid_list = preprocess_ids(orcid_singleton_list, kwargs.get("grouped", True))

    return orcid_list


def fetch_openalex(
    ids: Union[List[str], List[List[str]]],
    mails: List[str],
    perpage: int,
    filter_criteria: Union[str, List[str]],
    parallel_jobs: int = 8,
    endpoint: str = "works",
) -> Dict[str, List[Callable]]:
    """
    Fetches objects from OpenAlex based on the provided processed_ids, mailto, perpage,
    filter_criteria, and parallel_jobs.

    Args:
        ids (Union[List[str], List[List[str]]]): The processed IDs to fetch.
        mails (List[str]): The email address to use for fetching.
        perpage (int): The number of objects to fetch per page.
        filter_criteria (Union[str, List[str]]): The filter criteria to apply.
        parallel_jobs (int, optional): The number of parallel jobs to use. Defaults to 8.
        endpoint (str, optional): The OpenAlex endpoint to query. Defaults to "works".

    Returns:
        Dict[str, List[Callable]]: A dictionary containing the fetched objects, grouped by chunks.
    """
    # slice oa_ids
    oa_id_chunks = [ids[i : i + 80] for i in range(0, len(ids), 80)]
    logger.info("Slicing data. Number of oa_id_chunks: %s", len(oa_id_chunks))
    return {
        f"s{str(i)}": lambda chunk=chunk: Parallel(n_jobs=parallel_jobs, verbose=10)(
            delayed(fetch_openalex_objects)(oa_id, mails, perpage, filter_criteria, endpoint)
            for oa_id in chunk
        )
        for i, chunk in enumerate(oa_id_chunks)
    }


def concatenate_openalex(
    data: Dict[str, AbstractDataset],
    endpoint: str = "authors",
) -> pd.DataFrame:
    """
    Load the partitioned JSON dataset, iterate transforms, return dataframe.

    Args:
        data (Dict[str, AbstractDataset]): The partitioned JSON dataset.
        endpoint (str): The OpenAlex endpoint type (works, authors, etc).

    Returns:
        pd.DataFrame: The concatenated OpenAlex dataset.
    """
    outputs = []
    for i, (key, batch_loader) in enumerate(data.items()):
        data_batch = batch_loader()
        if endpoint == "authors":
            df_batch = json_loader_authors(data_batch)
        else:
            df_batch = json_loader_works(data_batch)
        outputs.append(df_batch)
        logger.info("Loaded %s. Progress: %s/%s", key, i + 1, len(data))
    outputs = pd.concat(outputs)

    return outputs

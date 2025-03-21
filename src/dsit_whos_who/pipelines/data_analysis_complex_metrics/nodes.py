import logging
from typing import Dict, List, Union
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from kedro.io import AbstractDataset

from ..data_collection_oa.utils import preprocess_ids
from ..data_collection_oa.nodes import fetch_openalex_objects
from .utils.cd_index import process_works_batch

logger = logging.getLogger(__name__)


def create_list_cited_ids(works: pd.DataFrame) -> list:
    """
    Create a list of OpenAlex work IDs from a DataFrame of works.

    Args:
        authors (pd.DataFrame): A DataFrame containing authors information.

    Returns:
        list: A list of OpenAlex author IDs.
    """
    logger.info("Starting to create list of OpenAlex author IDs...")

    # create unique list
    oa_ids = list(set(works[works["id"].notnull()]["id"].drop_duplicates().tolist()))

    logger.info("Found %s unique OpenAlex IDs", len(oa_ids))

    # concatenate doi values to create group queries
    oa_list = preprocess_ids(oa_ids, True)

    logger.info("Finished preprocessing OpenAlex IDs")
    return oa_list


def fetch_openalex_work_citations(
    ids: Union[List[str], List[List[str]]],
    mails: List[str],
    perpage: int,
    filter_criteria: Union[str, List[str]],
    parallel_jobs: int = 8,
    endpoint: str = "works",
    **kwargs,
) -> pd.DataFrame:
    """
    Fetches and processes works from OpenAlex.
    """
    logger.info(
        "Beginning to fetch %s OpenAlex records from %s endpoint", len(ids), endpoint
    )

    # slice oa_ids
    oa_id_chunks = [ids[i : i + 500] for i in range(0, len(ids), 500)]
    logger.info("Created %s chunks of up to 40 IDs each", len(oa_id_chunks))

    final_data = []
    seen_ids = set()

    for i, chunk in enumerate(oa_id_chunks, 1):
        logger.info(
            "Processing chunk %s of %s (%s%% complete)",
            i,
            len(oa_id_chunks),
            f"{(i/len(oa_id_chunks)*100):.1f}",
        )

        data = Parallel(n_jobs=parallel_jobs, verbose=10)(
            delayed(fetch_openalex_objects)(
                oa_id, mails, perpage, filter_criteria, endpoint, **kwargs
            )
            for oa_id in chunk
        )

        df_batch = process_works_batch(data, seen_ids)

        yield {f"works_{i}": df_batch}
"""
Utility functions for collecting data from various sources.
"""

# pylint: disable=E0402

import logging
from typing import Dict, List, Union
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ...data_collection_oa.utils.publications import json_loader_works
from ...data_collection_oa.nodes import fetch_openalex_objects

logger = logging.getLogger(__name__)

def fetch_openalex_works(
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
    oa_id_chunks = [ids[i : i + 40] for i in range(0, len(ids), 40)]
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
        if not df_batch.empty:
            final_data.append(df_batch)
            seen_ids.update(df_batch["id"].tolist())

    # Concatenate all processed dataframes
    final_df = pd.concat(final_data) if final_data else pd.DataFrame()
    logger.info("Completed processing with %s total unique papers", len(final_df))
    return final_df


def process_works_batch(data: List[Dict], seen_ids: set) -> pd.DataFrame:
    """Process a batch of works data."""
    logger.debug("Converting fetched data to DataFrame")
    df_batch = json_loader_works(data)

    if df_batch.empty:
        return df_batch

    # create list of authorships
    logger.debug("Processing authorships")
    df_batch["authorships"] = df_batch["authorships"].apply(
        lambda x: [[author[0], author[2]] for author in x]
    )

    # create list of topics
    logger.debug("Processing topics")
    df_batch["topics"] = df_batch["topics"].apply(
        lambda x: (
            [
                [
                    topic[0].replace("T", ""),
                    topic[2].replace("subfields/", ""),
                    topic[4].replace("fields/", ""),
                    topic[6].replace("domains/", ""),
                ]
                for topic in x
                if topic is not None
            ]
            if x
            else []
        )
    )

    # clean and filter the dataframe
    logger.debug("Cleaning and filtering data")
    df_batch = (
        df_batch.drop_duplicates(subset=["id"])
        .loc[lambda x: x["publication_date"] >= "1980-01-01"]
        .assign(
            fwci=lambda x: pd.to_numeric(
                x["fwci"].replace("", np.nan).infer_objects(copy=False)
            )
        )
        .reset_index(drop=True)
    )

    # filter out papers we've already seen
    if not df_batch.empty:
        original_count = len(df_batch)
        df_batch = df_batch[~df_batch["id"].isin(seen_ids)]
        logger.info(
            "Found %s new unique papers (filtered %s duplicates)",
            len(df_batch),
            original_count - len(df_batch),
        )

    return df_batch

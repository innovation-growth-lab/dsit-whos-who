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


logger = logging.getLogger(__name__)


def process_author_sampling(
    author_id: str, author_papers: pd.DataFrame
) -> pd.DataFrame:
    """
    Process sampling for a single author.

    Args:
        author_id (str): The OpenAlex ID of the author
        author_papers (pd.DataFrame): DataFrame containing all papers for this author

    Returns:
        pd.DataFrame: Sampled papers for this author
    """
    # stratify by year and fwci bin
    samples_per_stratum = max(
        1, int(100 / (len(author_papers["year"].unique()) * 4))
    )  # distributes 100 samples evenly across all year and fwci stratas

    author_samples = []
    for year in author_papers["year"].unique():
        year_papers = author_papers[author_papers["year"] == year]
        for fwci_bin in ["low", "medium-low", "medium-high", "high"]:
            stratum = year_papers[year_papers["fwci_bin"] == fwci_bin]
            if not stratum.empty:
                # sample papers from this stratum
                sampled = stratum.sample(n=min(samples_per_stratum, len(stratum)))
                author_samples.append(sampled)

    if author_samples:
        author_sampled_papers = pd.concat(author_samples)
        # Cap at 100 papers per author if we got more
        if len(author_sampled_papers) > 100:
            author_sampled_papers = author_sampled_papers.sample(n=100)
        return author_sampled_papers
    return pd.DataFrame()


def process_works_batch(data: List[Dict], seen_ids: set) -> pd.DataFrame:
    """Process a batch of works data."""
    logger.debug("Converting fetched data to DataFrame")
    df_batch = json_loader_works(data)

    if df_batch.empty:
        return df_batch

    # clean and filter the dataframe
    logger.debug("Cleaning and filtering data")
    df_batch = df_batch.drop_duplicates(subset=["id"]).reset_index(drop=True)

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

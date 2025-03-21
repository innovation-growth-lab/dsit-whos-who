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

def process_works_batch(data: List[Dict], seen_ids: set) -> pd.DataFrame:
    """Process a batch of works data."""
    logger.debug("Converting fetched data to DataFrame")
    df_batch = json_loader_works(data)

    if df_batch.empty:
        return df_batch

    # clean and filter the dataframe
    logger.debug("Cleaning and filtering data")
    df_batch = (
        df_batch.drop_duplicates(subset=["id"])
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

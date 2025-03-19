"""
This module contains the nodes for the data analysis disciplinarity pipeline.
"""

import logging
from typing import Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
from kedro.io import AbstractDataset

from ..data_collection_oa.utils import preprocess_ids
from ..data_collection_oa.utils.publications import json_loader_works

logger = logging.getLogger(__name__)


def create_list_oa_author_ids(authors: pd.DataFrame) -> list:
    """
    Create a list of OpenAlex author IDs from a DataFrame of persons.

    Args:
        authors (pd.DataFrame): A DataFrame containing authors information.

    Returns:
        list: A list of OpenAlex author IDs.
    """
    # create unique list
    oa_ids = list(
        set(authors[authors["oa_id"].notnull()]["oa_id"].drop_duplicates().tolist())
    )

    # concatenate doi values to create group querise
    oa_list = preprocess_ids(oa_ids, True)

    return oa_list


def concatenate_openalex_publications(
    data: Dict[str, AbstractDataset], gtr_publications: pd.DataFrame
) -> pd.DataFrame:
    """
    Load the partitioned JSON dataset, iterate transforms, return dataframe.
    Removes duplicate papers based on OpenAlex ID.
    Limits papers per author to 250, except for papers that appear in gtr_publications.

    Args:
        data (Dict[str, AbstractDataset]): The partitioned JSON dataset.
        gtr_publications (pd.DataFrame): GTR publications data to preserve regardless of limits.

    Returns:
        pd.DataFrame: The concatenated OpenAlex dataset with duplicates removed and author
            limits applied.
    """

    gtr_paper_ids = (
        set(gtr_publications["id"].dropna().unique())
        if "id" in gtr_publications.columns
        else set()
    )
    logger.info("%d GTR papers to preserve", len(gtr_paper_ids))

    outputs = []
    seen_ids = set()
    author_paper_counts = {}  # Track papers per author
    total_papers_processed = 0
    total_papers_kept = 0
    for i, (key, batch_loader) in enumerate(data.items()):
        logger.info("Processing batch %s (%d/%d)", key, i + 1, len(data))
        data_batch = batch_loader()
        df_batch = json_loader_works(data_batch)

        # create list of authorships
        df_batch["authorships"] = df_batch["authorships"].apply(
            lambda x: [author[0] for author in x]
        )

        # create list of topics
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
        df_batch = (
            df_batch.drop(columns=["doi"])
            .drop_duplicates(subset=["id"])
            .loc[lambda x: x["publication_date"] >= "2002-01-01"]
            .assign(
                fwci=lambda x: pd.to_numeric(
                    x["fwci"].replace("", np.nan).infer_objects(copy=False)
                )
            )
            .reset_index(drop=True)
        )

        # filter out papers we've already seen based on id
        if not df_batch.empty:
            df_batch = df_batch[~df_batch["id"].isin(seen_ids)]
            batch_size_before = len(df_batch)
            df_batch = _filter_papers_by_author_limit(
                df_batch, gtr_paper_ids, author_paper_counts
            )
            batch_size_after = len(df_batch)
            seen_ids.update(df_batch["id"].tolist())

            total_papers_processed += batch_size_before
            total_papers_kept += batch_size_after

            logger.info(
                "Batch %s: Processed %d papers, kept %d (%.1f%%). "
                "Running totals: processed %d, kept %d (%.1f%%)",
                key,
                batch_size_before,
                batch_size_after,
                (
                    100 * batch_size_after / batch_size_before
                    if batch_size_before > 0
                    else 0
                ),
                total_papers_processed,
                total_papers_kept,
                (
                    100 * total_papers_kept / total_papers_processed
                    if total_papers_processed > 0
                    else 0
                ),
            )

        if not df_batch.empty:
            outputs.append(df_batch)

    outputs = pd.concat(outputs) if outputs else pd.DataFrame()

    logger.info(
        "Final statistics: Processed %d papers, kept %d (%.1f%%)."
        "Current number of tracked authors: %d",
        total_papers_processed,
        total_papers_kept,
        (
            100 * total_papers_kept / total_papers_processed
            if total_papers_processed > 0
            else 0
        ),
        len(author_paper_counts),
    )

    return outputs


def _filter_papers_by_author_limit(
    df: pd.DataFrame, gtr_ids: set, author_counts: dict
) -> pd.DataFrame:
    """Filter papers based on author limits and GTR inclusion.

    Args:
        df: DataFrame containing papers to filter
        gtr_ids: Set of GTR paper IDs to always include
        author_counts: Dictionary tracking paper counts per author

    Returns:
        Filtered DataFrame
    """
    papers_to_keep = []
    gtr_papers = 0
    limited_papers = 0

    # Use tqdm for progress tracking
    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Filtering papers", leave=False
    ):
        # Always keep GTR papers
        if row["id"] in gtr_ids:
            papers_to_keep.append(True)
            gtr_papers += 1
            # Update author counts for GTR papers
            for author_id in row["authorships"]:
                if author_id in author_counts:
                    author_counts[author_id] += 1
            continue

        # Check if any author has reached the limit
        author_limit_reached = False
        for author_id in row["authorships"]:
            current_count = author_counts.get(author_id, 0)
            if current_count >= 250:
                author_limit_reached = True
                limited_papers += 1
                break

        if not author_limit_reached:
            # Update counts for all authors of this paper
            for author_id in row["authorships"]:
                author_counts[author_id] = author_counts.get(author_id, 0) + 1
            papers_to_keep.append(True)
        else:
            papers_to_keep.append(False)

    logger.debug(
        "Paper filtering details: GTR papers: %d, Limited papers: %d, Total papers: %d",
        gtr_papers,
        limited_papers,
        len(df),
    )

    return df[papers_to_keep]

"""
Utility functions for calculating and processing disruption indices.

This module implements the Wu and Yan (2019) disruption index, which is a variant of
the Funk & Owen-Smith (2017) CD-index. The disruption index measures how much a paper
disrupts the status quo by introducing new ideas that change the direction of research.

The Wu and Yan (2019) disruption index is calculated as:
    DI = (n_f - n_b) / (n_f + n_b)

Where:
- n_f is the number of papers citing the focal paper but not its references
- n_b is the number of papers citing both the focal paper and its references

A positive DI (closer to 1) indicates a disruptive paper that changes the direction of research,
while a negative DI (closer to -1) indicates a consolidating paper that reinforces existing
research.

References:
- Wu & Yan (2019): https://doi.org/10.48550/arXiv.1905.03461
- Bornmann et al. (2020): https://doi.org/10.1162/qss_a_00068

The decision to use the Wu and Yan (2019) disruption index was made because:
- It overcomes a limitation of OpenAlex' forward citation collection, which requires
    inefficient crawling through citations and millions of API calls (which runs into rate limits).
- It performs surprisingly well when compared with labelled data, as highlighted in Leibel and
    Bornmann (2023): https://doi.org/10.48550/arXiv.2308.02383. In particular, see Table 5.

"""

# pylint: disable=E0402

import logging
from typing import Dict, List
import numpy as np
import pandas as pd
from tqdm import tqdm
from ...data_collection_oa.utils.publications import json_loader_works


logger = logging.getLogger(__name__)


def process_chunk(
    chunk_authors: List[str], works_exploded: pd.DataFrame
) -> pd.DataFrame:
    """
    Process a chunk of authors.

    Args:
        chunk_authors (List[str]): List of author IDs to process
        works_exploded (pd.DataFrame): DataFrame containing all works with exploded authorships

    Returns:
        pd.DataFrame: Combined sampled papers for all authors in the chunk
    """
    chunk_results = []
    for author_id in chunk_authors:
        # Get papers for this author
        author_papers = works_exploded[works_exploded["author_id"] == author_id].copy()
        if not author_papers.empty:
            result = process_author_sampling(author_papers)
            if not result.empty:
                chunk_results.append(result)
    return pd.concat(chunk_results) if chunk_results else pd.DataFrame()


def process_author_sampling(author_papers: pd.DataFrame) -> pd.DataFrame:
    """
    Process sampling for a single author.

    Args:
        author_id (str): The OpenAlex ID of the author
        author_papers (pd.DataFrame): DataFrame containing all papers for this author

    Returns:
        pd.DataFrame: Sampled papers for this author
    """
    # stratify by year
    samples_per_stratum = max(
        1, int(50 / (len(author_papers["year"].unique())))
    )  # distributes 100 samples evenly across all year stratas

    author_samples = []
    for year in author_papers["year"].unique():
        year_papers = author_papers[author_papers["year"] == year]
        if not year_papers.empty:
            # sample papers from this year stratum
            sampled = year_papers.sample(
                n=min(samples_per_stratum, len(year_papers)), weights="fwci_quantile"
            )
            author_samples.append(sampled)

    if author_samples:
        author_sampled_papers = pd.concat(author_samples)
        # Cap at 100 papers per author if we got more
        if len(author_sampled_papers) > 50:
            author_sampled_papers = author_sampled_papers.sample(
                n=50, weights="fwci_quantile"
            )
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


def process_disruption_indices(
    focal_papers: pd.DataFrame,
    citing_papers_dataset: Dict[str, callable],
) -> pd.DataFrame:
    """
    Process disruption indices for all focal papers using efficient set operations.
    Uses batch-level set operations to minimizse iterations and maximise efficiency.

    Args:
        focal_papers (pd.DataFrame): DataFrame containing focal papers with 'id' and
            'referenced_works' columns
        citing_papers_dataset (Dict[str, callable]): Dictionary of partition loaders
            for citing papers

    Returns:
        pd.DataFrame: DataFrame with focal paper IDs and their disruption indices
    """
    logger.info("Processing disruption indices for %d focal papers", len(focal_papers))

    # Convert focal papers and their references to sets once
    focal_ids = focal_papers["id"].values
    focal_id_to_idx = {id_: idx for idx, id_ in enumerate(focal_ids)}
    focal_ids_set = set(focal_ids)
    focal_refs_sets = [set(refs) for refs in focal_papers["referenced_works"].values]

    # instantiate count arrays
    n_f = np.zeros(len(focal_ids), dtype=np.int32)
    n_b = np.zeros(len(focal_ids), dtype=np.int32)

    seen_citing_ids = set()
    for key, loader in tqdm(
        citing_papers_dataset.items(), total=len(citing_papers_dataset)
    ):
        batch = loader()

        unseen_batch = batch[~batch["id"].isin(seen_citing_ids)]

        tqdm.write(
            f"Batch {key}: Processing {len(unseen_batch)} papers, "
            f"filtered {len(batch) - len(unseen_batch)} duplicates"
        )

        for refs in unseen_batch["referenced_works"].values:
            refs_set = set(refs)

            # Find which focal papers this citing paper references
            cited_focal_papers = refs_set & focal_ids_set
            if not cited_focal_papers:
                continue

            # For each cited focal paper, check if any of its references are cited
            for focal_id in cited_focal_papers:
                focal_idx = focal_id_to_idx[focal_id]
                if focal_refs_sets[focal_idx] & refs_set:
                    n_b[focal_idx] += 1
                else:
                    n_f[focal_idx] += 1

        seen_citing_ids.update(unseen_batch["id"].values)

    # Create results DataFrame
    result_data = {"id": focal_ids, "n_f": n_f, "n_b": n_b, "total": n_f + n_b}

    combined_results = pd.DataFrame(result_data)

    # Calculate disruption index using vectorised operations
    mask = combined_results["total"] > 0
    combined_results["disruption_index"] = np.nan
    combined_results.loc[mask, "disruption_index"] = (
        combined_results.loc[mask, "n_f"] - combined_results.loc[mask, "n_b"]
    ) / combined_results.loc[mask, "total"]

    combined_results["di_status"] = np.where(
        combined_results["total"] > 0, "valid", "calculation_failed"
    )

    logger.info("Calculated disruption indices for %d papers", len(combined_results))

    # Calculate summary statistics by status
    status_counts = combined_results["di_status"].value_counts()
    for status, count in status_counts.items():
        logger.info(
            "%s: %d papers (%.1f%%)", status, count, count / len(combined_results) * 100
        )

    # Calculate statistics for valid disruption indices
    valid_dis = combined_results[combined_results["di_status"] == "valid"][
        "disruption_index"
    ]

    logger.info("Valid disruption indices statistics (%d papers):", len(valid_dis))
    logger.info("  Mean: %.3f", valid_dis.mean())
    logger.info("  Median: %.3f", valid_dis.median())
    logger.info("  Min: %.3f", valid_dis.min())
    logger.info("  Max: %.3f", valid_dis.max())
    logger.info(
        "  Papers with positive DI: %d (%.1f%%)",
        (valid_dis > 0).sum(),
        (valid_dis > 0).sum() / len(valid_dis) * 100,
    )
    logger.info(
        "  Papers with negative DI: %d (%.1f%%)",
        (valid_dis < 0).sum(),
        (valid_dis < 0).sum() / len(valid_dis) * 100,
    )
    logger.info(
        "  Papers with neutral DI: %d (%.1f%%)",
        (valid_dis == 0).sum(),
        (valid_dis == 0).sum() / len(valid_dis) * 100,
    )

    return combined_results[
        ["id", "di_status", "disruption_index", "n_f", "n_b", "total"]
    ]

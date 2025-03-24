import logging
from typing import Dict, List, Union, Generator
import pandas as pd
from joblib import Parallel, delayed

from ..data_collection_oa.utils import preprocess_ids
from ..data_collection_oa.nodes import fetch_openalex_objects
from .utils.cd_index import process_works_batch, process_author_sampling

logger = logging.getLogger(__name__)


def create_cited_work_ids(
    works: pd.DataFrame, authors: pd.DataFrame, n_jobs: int = 8
) -> pd.DataFrame:
    """
    Create a stratified sample of OpenAlex work IDs from a DataFrame of works.
    The sampling is done per author, considering publication year and citation impact.

    Args:
        works (pd.DataFrame): DataFrame containing work information including authorships and FWCI
        authors (pd.DataFrame): DataFrame containing author information with oa_id
        n_jobs (int): Number of jobs for parallel processing. Default is 8.

    Returns:
        pd.DataFrame: A DataFrame with the sampled papers
    """
    logger.info("Starting stratified sampling of papers per author...")

    # year and non-null fwci
    works["year"] = pd.to_datetime(works["publication_date"]).dt.year
    works["fwci"] = works["fwci"].fillna(-1)

    # create bins for strat. on fwci
    works["fwci_bin"] = pd.qcut(
        works["fwci"], q=4, labels=["low", "medium-low", "medium-high", "high"]
    )

    # extract author ids from authorships and explode
    logger.info("Exploding authorships to create author-paper pairs...")
    works["author_id"] = works["authorships"].apply(
        lambda x: [auth_id for auth_id, _ in x]
    )

    # select only relevant columns
    works = works[["id", "year", "fwci_bin", "author_id"]]

    works_exploded = works.explode("author_id")

    # filter to only include authors we care about
    author_ids = set(authors["oa_id"].dropna())
    works_exploded = works_exploded[works_exploded["author_id"].isin(author_ids)]

    logger.info("Processing %d unique authors", len(author_ids))

    # Pre-split data for each author
    logger.info("Preparing data for parallel processing...")
    author_data = {
        author_id: works_exploded[works_exploded["author_id"] == author_id].copy()
        for author_id in works_exploded["author_id"].unique()
    }

    # Process authors in parallel
    logger.info("Starting parallel processing of authors...")
    sampled_papers = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_author_sampling)(papers) for papers in author_data.values()
    )

    # Combine results
    papers_df = pd.concat([df for df in sampled_papers if not df.empty])

    logger.info("Sampled %d papers across all authors", len(papers_df))
    return papers_df


def create_list_ids(works: pd.DataFrame) -> List[str]:
    """
    Create a list of OpenAlex IDs from a DataFrame of works.
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
) -> Generator[Dict[str, pd.DataFrame], None, None]:
    """
    Fetches and processes works from OpenAlex.
    """
    logger.info(
        "Beginning to fetch %s OpenAlex records from %s endpoint", len(ids), endpoint
    )

    # slice oa_ids
    oa_id_chunks = [ids[i : i + 500] for i in range(0, len(ids), 500)]
    logger.info("Created %s chunks of up to 40 IDs each", len(oa_id_chunks))

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

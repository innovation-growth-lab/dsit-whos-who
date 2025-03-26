import logging
from typing import Dict, List, Union, Generator
import pandas as pd
from joblib import Parallel, delayed

from ..data_collection_oa.utils import preprocess_ids
from ..data_collection_oa.nodes import fetch_openalex_objects
from .utils.cd_index import process_works_batch, process_chunk

logger = logging.getLogger(__name__)


def sample_cited_work_ids(
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

    works["year"] = pd.to_datetime(works["publication_date"]).dt.year
    works["fwci"] = works["fwci"].fillna(-1)

    # extract author ids from authorships and explode
    logger.info("Exploding authorships to create author-paper pairs...")
    works["author_id"] = works["authorships"].apply(
        lambda x: [auth_id for auth_id, _ in x]
    )

    # drop if referenced_works is None (can't compute DI)
    works = works[works["referenced_works"].notna()]

    works = works.copy()
    # create quantile column, with highest fwci getting highest value (4)
    works["fwci_quantile"] = (
        pd.qcut(works["fwci"], q=5, labels=False, duplicates="drop") + 1
    )

    # select only relevant columns
    works_filt = works[["id", "year", "author_id", "fwci_quantile"]]
    works = works[["id", "referenced_works"]]
    # explode the author_id column
    works_exploded = works_filt.explode("author_id")

    # filter to only include authors we care about
    author_ids = set(authors["oa_id"].dropna())
    works_exploded = works_exploded[works_exploded["author_id"].isin(author_ids)]
    author_ids = list(works_exploded["author_id"].unique())

    logger.info("Processing %d unique authors", len(author_ids))

    # Create chunks of author IDs for parallel processing
    author_chunks = [author_ids[i : i + 500] for i in range(0, len(author_ids), 500)]

    logger.info(
        "Created %d chunks of approximately %d authors each",
        len(author_chunks),
        500,
    )

    logger.info("Starting parallel processing of chunks...")
    chunk_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_chunk)(chunk, works_exploded) for chunk in author_chunks
    )

    concat_papers = pd.concat(chunk_results).reset_index(drop=True)
    logger.info("Sampled %d papers across all authors", len(concat_papers))

    # drop fwci_quantile and year
    concat_papers = concat_papers.drop(columns=["fwci_quantile", "year"])

    # merge back the referenced_works
    concat_papers = concat_papers.merge(works, on="id", how="left")

    return concat_papers


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


def fetch_author_work_citations(
    ids: Union[List[str], List[List[str]]],
    mails: List[str],
    perpage: int,
    filter_criteria: Union[str, List[str]],
    parallel_jobs: int = 8,
    endpoint: str = "works",
    **kwargs,
) -> Generator[Dict[str, pd.DataFrame], None, None]:
    """
    Fetches and processes works from OpenAlex who cite focal works.
    """
    logger.info(
        "Beginning to fetch %s OpenAlex records from %s endpoint", len(ids), endpoint
    )

    # slice oa_ids
    oa_id_chunks = [ids[i : i + 200] for i in range(0, len(ids), 200)]
    logger.info("Created %s chunks of up to 50 IDs each", len(oa_id_chunks))

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

        # convert ID column - remove 'W' prefix and convert to int (efficiency)
        df_batch["id"] = df_batch["id"].str.replace("W", "").astype(int)

        if "referenced_works" in df_batch.columns:
            # process referenced_works column - remove 'W' prefix and convert to int for each list
            df_batch["referenced_works"] = df_batch["referenced_works"].apply(
                lambda x: (
                    [int(ref.replace("W", "")) for ref in x]
                    if isinstance(x, list)
                    else x
                )
            )

        seen_ids.update(df_batch["id"].tolist())

        yield {f"works_{i}": df_batch}


def refactor_reference_works(works: pd.DataFrame) -> List[str]:
    """
    Fetch reference works from OpenAlex.
    """
    logger.info("Starting to fetch reference works from OpenAlex...")

    # keep only the referenced_works column
    works = works[["referenced_works"]]

    # explode the referenced_works column
    works = works.explode("referenced_works")

    # count these, remove top 3% frequency (~ Deng & Zeng, 2023)
    works["count"] = works.groupby("referenced_works")["referenced_works"].transform(
        "count"
    )

    # drop duplicates
    works = works.drop_duplicates(subset=["referenced_works"]).reset_index(drop=True)

    # prune
    works = works[works["count"] <= works["count"].quantile(0.97)]
    works = works.drop(columns=["count"])

    # rename column to id
    works = works.rename(columns={"referenced_works": "id"})

    return works


def fetch_author_work_references(
    ids: Union[List[str], List[List[str]]],
    mails: List[str],
    perpage: int,
    filter_criteria: Union[str, List[str]],
    parallel_jobs: int = 8,
    endpoint: str = "works",
    **kwargs,
) -> Generator[Dict[str, pd.DataFrame], None, None]:
    """
    Fetches and processes works from OpenAlex who cite focal works.
    """
    logger.info(
        "Beginning to fetch %s OpenAlex records from %s endpoint", len(ids), endpoint
    )

    # slice oa_ids
    oa_id_chunks = [ids[i : i + 200] for i in range(0, len(ids), 200)]
    logger.info("Created %s chunks of up to 50 IDs each", len(oa_id_chunks))

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

        # clean the referenced works,
        cleaned_works = []
        for batch_return in data:
            df = pd.DataFrame(batch_return)
            df["id"] = df["id"].str.replace("W", "").astype(int)
            df["referenced_works"] = df["referenced_works"].apply(
                lambda x: (
                    [int(ref.replace("https://openalex.org/W", "")) for ref in x]
                    if isinstance(x, list)
                    else x
                )
            )
            cleaned_works.append(df)
        cleaned_works = pd.concat(cleaned_works)

        # find the corresponding id in the chunk, after "|" splitting the ids, remove W
        chunk_ids = set(
            [int(id_part.replace("W", "")) for id in chunk for id_part in id.split("|")]
        )

        # Use set intersection instead of list comprehension
        cleaned_works["reference_id"] = cleaned_works["referenced_works"].apply(
            lambda x: list(chunk_ids.intersection(x))
        )

        # Filter out empty lists before exploding to reduce size
        cleaned_works = cleaned_works[cleaned_works["reference_id"].apply(len) > 0]

        # Now explode and group on smaller dataset
        cleaned_works_e = cleaned_works.explode("reference_id")
        cleaned_works_agg = (
            cleaned_works_e.groupby("reference_id").agg(ids=("id", list)).reset_index()
        )

        # relabel reference_id to id, and ids to citing_ids
        cleaned_works_agg = cleaned_works_agg.rename(
            columns={"reference_id": "id", "ids": "citing_ids"}
        )

        yield {f"works_{i}": cleaned_works_agg}

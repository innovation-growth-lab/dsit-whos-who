"""
This module contains nodes for calculating complex metrics for author analysis.

The module implements functions to:
1. Sample papers from authors' publication records using stratified sampling
2. Fetch citation data from OpenAlex for sampled papers and their references
3. Calculate disruption indices based on Wu and Yan (2019) methodology

The disruption index measures how much a paper disrupts vs. consolidates its research field
by checking whether papers that cite it also cite its references or not.
"""

import logging
from typing import Dict, List, Union, Generator
import pandas as pd
from joblib import Parallel, delayed
from sentence_transformers import SentenceTransformer

from ..data_collection_oa.utils import preprocess_ids
from ..data_collection_oa.nodes import fetch_openalex_objects
from .utils.cd_index import (
    process_works_batch,
    process_chunk,
    process_disruption_indices,
)
from .utils.embeddings import compute_distance_matrix
from .utils.discipline_diversity import (
    create_author_and_year_subfield_frequency,
    filter_single_list,
)

logger = logging.getLogger(__name__)


def sample_cited_work_ids(
    works: pd.DataFrame, authors: pd.DataFrame, n_jobs: int = 12
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
        "Beginning to fetch %s OpenAlex works from %s endpoint", len(ids), endpoint
    )

    # slice oa_ids
    oa_id_chunks = [ids[i : i + 200] for i in range(0, len(ids), 200)]
    logger.info("Created %s chunks of up to 50 IDs each", len(oa_id_chunks))

    seen_ids = set()
    for i, chunk in enumerate(oa_id_chunks, 1):
        if i <= 110:
            continue
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
            lambda x: list(chunk_ids.intersection(x))  # pylint: disable=W0640
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


def calculate_disruption_indices(
    focal_papers: pd.DataFrame,
    citing_papers_dataset: Dict[str, callable],
) -> pd.DataFrame:
    """
    Calculate disruption indices for a set of focal papers.

    This function implements the Wu and Yan (2019) disruption index, which ignores papers
    bypassing the focal paper. The formula is:

    DI = (n_f - n_b) / (n_f + n_b)

    Where:
    - n_f is the number of papers citing the focal but not its references
    - n_b is the number of papers citing both the focal and its references

    Args:
        focal_papers (pd.DataFrame): DataFrame containing focal papers with 'id' and
            'referenced_works' columns
        citing_papers_dataset (Dict[str, callable]): Dictionary of partition loaders
            for citing papers
        n_jobs (int, optional): Number of parallel jobs. Defaults to 8.

    Returns:
        pd.DataFrame: DataFrame with focal paper IDs and their disruption indices
    """
    logger.info(
        "Starting disruption index calculation for %d papers", len(focal_papers)
    )

    if "referenced_works" not in focal_papers.columns:
        logger.error("Referenced works column not found in focal papers dataset")
        return pd.DataFrame(columns=["id", "disruption_index"])

    focal_papers = focal_papers[["id", "referenced_works"]]
    # Copy the dataframe to avoid modifying the original
    focal_papers = focal_papers.copy()

    # drop referenced_works = None
    focal_papers = focal_papers[focal_papers["referenced_works"].notna()]

    # harmonise the ids and referenced_works
    focal_papers["id"] = focal_papers["id"].str.replace("W", "").astype(int)
    focal_papers["referenced_works"] = focal_papers["referenced_works"].apply(
        lambda x: [int(ref.replace("W", "")) for ref in x]
    )

    # Log statistics about focal papers
    ref_counts = focal_papers["referenced_works"].apply(len)
    logger.info("Focal papers statistics:")
    logger.info(" - Total papers: %d", len(focal_papers))
    logger.info(" - Mean references per paper: %.1f", ref_counts.mean())
    logger.info(" - Max references: %d", ref_counts.max())

    # Process the disruption indices
    return process_disruption_indices(focal_papers, citing_papers_dataset)


def compute_subfield_embeddings(
    cwts_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute topic embeddings and distance matrices for topics, subfields, fields, and domains.

    Args:
        cwts_data (pd.DataFrame): The input dataframe containing the CWTS data.

    Returns:
        Tuple: A tuple containing the topic distance matrix, subfield distance matrix,
        field distance matrix, and domain distance matrix.
    """
    encoder = SentenceTransformer("sentence-transformers/allenai-specter")

    cwts_data = cwts_data.copy()[cwts_data["level"] == 2]

    # retrieve id (last item after splitting on ">" id_path)
    cwts_data["subfield_id"] = (
        cwts_data["id_path"].apply(lambda x: x.split("> ")[-1]).astype(int)
    )

    logger.info("Computing embeddings for topics")
    cwts_data["subfield_embeddings"] = cwts_data["label"].apply(encoder.encode)

    logger.info("Computing distance matrix for subfields")
    subfield_distance_matrix = compute_distance_matrix(
        cwts_data["subfield_embeddings"].tolist(), cwts_data["subfield_id"].tolist()
    )

    return subfield_distance_matrix


def create_author_aggregates(
    authors_data: pd.DataFrame, authors: pd.DataFrame, cwts_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Create aggregates of author data based on a specified taxonomy level.

    Args:
        authors_data (AbstractDataset): A dataset containing author data.
        level (int): The taxonomy level to aggregate the data on.
        cwts_data (list): List of unique topic IDs.

    Returns:
        pd.DataFrame: DataFrame with columns: author, year, publications,
            total_publications, frequency. The "frequency" column contains
            (n_topics,) dimensional arrays of topic frequencies.
    """
    # split authors
    authors_data["authorships"] = authors_data["authorships"].apply(
        lambda x: [author[0] for author in x]
    )

    # get year from str "YYYY-MM-DD" changing to datetime
    authors_data["year"] = pd.to_datetime(authors_data["publication_date"]).dt.year

    # extract list of subfields
    authors_data["subfield_ids"] = authors_data["topics"].apply(
        lambda x: [filter_single_list(topic, 1) for topic in x]
    )

    # keep relevant cols: authorships, year, subfield_id
    authors_data = authors_data[["authorships", "year", "subfield_ids"]]

    logger.info("Exploding authorships")
    authors_data = authors_data.explode("authorships")

    # rename authorships to author
    authors_data.rename(columns={"authorships": "author"}, inplace=True)

    # keep rows only if author in authors["oa_id"]
    authors_data = authors_data[authors_data["author"].isin(authors["oa_id"])]

    logger.info("Creating author and year frequency data")
    author_frequencies = create_author_and_year_subfield_frequency(
        authors_data, cwts_data
    )

    author_frequencies["total_publications"] = author_frequencies.groupby("author")[
        "publications"
    ].transform("sum")

    return author_frequencies

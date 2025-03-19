"""
This module contains the nodes for the data analysis basic metrics pipeline.
"""

import logging
import pandas as pd
from kedro.io import AbstractDataset

from .utils.collecting import fetch_openalex_works
from .utils.processing import (
    process_author_metadata,
    date_cleaner,
    process_person_gtr_data,
    prepare_final_person_data,
)
from ..data_collection_oa.utils import preprocess_ids
from ..author_disambiguation.utils.preprocessing.gtr import preprocess_gtr_persons

logger = logging.getLogger(__name__)


def create_list_oa_author_ids(authors: pd.DataFrame) -> list:
    """
    Create a list of OpenAlex author IDs from a DataFrame of persons.

    Args:
        authors (pd.DataFrame): A DataFrame containing authors information.

    Returns:
        list: A list of OpenAlex author IDs.
    """
    logger.info("Starting to create list of OpenAlex author IDs...")

    # create unique list
    oa_ids = list(
        set(authors[authors["oa_id"].notnull()]["oa_id"].drop_duplicates().tolist())
    )

    logger.info("Found %s unique OpenAlex IDs", len(oa_ids))

    # concatenate doi values to create group queries
    oa_list = preprocess_ids(oa_ids, True)

    logger.info("Finished preprocessing OpenAlex IDs")
    return oa_list


def fetch_openalex_matched_author_works(
    ids: list,
    mails: list,
    perpage: int,
    filter_criteria: str,
    **kwargs,
) -> pd.DataFrame:
    """Node for fetching OpenAlex works."""
    return fetch_openalex_works(ids, mails, perpage, filter_criteria, **kwargs)


def process_matched_author_metadata(
    author_loaders: AbstractDataset,
    matched_authors: pd.DataFrame,
) -> pd.DataFrame:
    """Node for processing author metadata.

    This function processes metadata for authors who have been matched between OpenAlex and 
        GTR.
    It performs the following steps:
    1. Iterates through author data loaded from OpenAlex
    2. For each batch, processes the metadata to extract key metrics like:
        - First year of publication
        - Citations per publication
    3. Filters authors to only include those matched with GTR
    4. Combines all processed batches into a single DataFrame

    Args:
        author_loaders (AbstractDataset): Dataset containing OpenAlex author metadata
        matched_authors (pd.DataFrame): DataFrame containing matched author IDs between GTR 
            and OpenAlex

    Returns:
        pd.DataFrame: Processed author metadata containing publication metrics and citation 
            information
    """
    author_data = []
    for key, author_loader in author_loaders.items():
        logger.info("Processing author metadata for %s", key)
        author_df = process_author_metadata(author_loader(), matched_authors)
        author_data.append(author_df)

    author_df = pd.concat(author_data)
    logger.info("Completed processing author metadata for %s authors", len(author_df))
    return author_df


def process_matched_person_gtr_data(
    persons: pd.DataFrame,
    projects: pd.DataFrame,
    matched_authors: pd.DataFrame,
) -> pd.DataFrame:
    """Node for processing GTR data for matched persons.

    This function processes Gateway to Research (GTR) data for persons who have been matched
    to OpenAlex authors. It performs the following steps:
    1. Preprocesses and filters persons data to only include matched authors
    2. Merges person-project relationships with project metadata
    3. Cleans project dates and processes timeline information
    4. Creates aggregated summaries of project data per person

    Args:
        persons (pd.DataFrame): DataFrame containing person data from GTR
        projects (pd.DataFrame): DataFrame containing project data from GTR
        matched_authors (pd.DataFrame): DataFrame containing matched author IDs between GTR and OpenAlex

    Returns:
        pd.DataFrame: Processed person data containing project timelines, grant categories,
            and funding information aggregated by person
    """
    # filter persons
    persons = persons[persons["person_id"].isin(matched_authors["gtr_id"])].reset_index(
        drop=True
    )

    # Process projects data
    persons_projects = persons.explode("project_id")
    persons_projects = persons_projects.merge(
        projects.drop_duplicates(subset=["project_id"]),
        on="project_id",
        how="left",
    )
    persons_projects = date_cleaner(persons_projects)

    # Get person summaries
    person_summaries = process_person_gtr_data(persons_projects)
    logger.info(
        "Completed processing GTR data for %s persons with project information",
        len(person_summaries),
    )

    return prepare_final_person_data(persons, person_summaries)

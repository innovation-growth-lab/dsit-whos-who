"""
This module contains the nodes for the data analysis basic metrics pipeline.
"""

import logging
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from kedro.io import AbstractDataset

from .utils.collecting import fetch_openalex_works
from .utils.processing import (
    process_author_metadata,
    date_cleaner,
    process_person_gtr_data,
    prepare_final_person_data,
    process_publication_batch,
)
from .utils.basic_metrics import (
    compute_academic_age,
    add_international_metrics,
)
from ..data_collection_oa.utils import preprocess_ids

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


def process_matched_author_works(
    publications: pd.DataFrame,
    matched_authors: pd.DataFrame,
    n_jobs: int = 8,
    batch_size: int = 1000,
) -> pd.DataFrame:
    """Process publication data to create a summary of collaborations per author per year.

    This function creates an intermediate DataFrame that summarises collaboration metrics
    for each matched author in each year of their publications. This makes subsequent
    analysis more efficient by pre-aggregating the data.

    Args:
        publications (pd.DataFrame): DataFrame containing publication data with authorships
            information
        matched_authors (pd.DataFrame): DataFrame containing matched author IDs between GTR
            and OpenAlex
        n_jobs (int, optional): Number of parallel jobs to run. Defaults to 8.
        batch_size (int, optional): Size of publication batches to process. Defaults to 1000.

    Returns:
        pd.DataFrame: DataFrame with columns:
            - author_id: OpenAlex ID of the author
            - year: Publication year
            - n_collab_uk: Number of UK-based collaborators
            - n_collab_abroad: Number of non-UK collaborators
            - unknown_collabs: Number of collaborators with unknown country
            - countries_abroad: List of unique foreign countries
    """
    logger.info("Processing publication data for collaboration metrics...")

    matched_ids = set(matched_authors["id"].unique())
    logger.info("Processing collaborations for %d matched authors", len(matched_ids))

    publications["year"] = pd.to_datetime(publications["publication_date"]).dt.year

    n_batches = (len(publications) + batch_size - 1) // batch_size
    publication_batches = np.array_split(publications, n_batches)

    logger.info(
        "Processing %d publications in %d batches using %d jobs",
        len(publications),
        n_batches,
        n_jobs,
    )

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_publication_batch)(batch, matched_ids)
        for batch in publication_batches
    )

    processed_data = [item for sublist in results for item in sublist]

    result_df = pd.DataFrame(processed_data)

    if len(result_df) == 0:
        logger.warning("No collaboration data found for matched authors")
        return pd.DataFrame(
            columns=[
                "author_id",
                "year",
                "n_collab_uk",
                "n_collab_abroad",
                "n_collab_unknown",
                "countries_abroad",
                "collab_ids",
            ]
        )

    result_df = (
        result_df.groupby(["author_id", "year"])
        .agg(
            {
                "n_collab_uk": "sum",
                "n_collab_abroad": "sum",
                "n_collab_unknown": "sum",
                "countries_abroad": lambda x: sorted(
                    list(set(country for countries in x for country in countries))
                ),
                "collab_ids": lambda x: sorted(
                    list(set(id for ids in x for id in ids))
                ),
            }
        )
        .reset_index()
    )

    logger.info(
        "Processed collaboration metrics for %d author-year combinations",
        len(result_df),
    )

    return result_df


def compute_basic_metrics(
    author_data: pd.DataFrame,
    person_data: pd.DataFrame,
    publications: pd.DataFrame,
    n_jobs: int = 8,
) -> pd.DataFrame:
    """
    Compute basic metrics from author and person data.

    Args:
        author_data (pd.DataFrame): OpenAlex author data with affiliations and publication history
        person_data (pd.DataFrame): GTR person data with project information
        publications (pd.DataFrame): Publication data with collaboration information
        n_jobs (int, optional): Number of parallel jobs to run. Defaults to 8.

    Returns:
        pd.DataFrame: Combined data with computed metrics
    """
    author_data = author_data.drop_duplicates(subset=["id", "gtr_id"])

    logger.info("Computing basic metrics for %s authors", len(author_data))

    merged_data = author_data.merge(
        person_data, left_on="gtr_id", right_on="person_id", how="inner", validate="1:1"
    )
    logger.info("Merged data contains %s records", len(merged_data))

    # Compute academic age
    merged_data["academic_age_at_first_grant"] = merged_data.apply(
        compute_academic_age, axis=1
    ).astype("Int64")

    # Add international experience metrics
    merged_data = add_international_metrics(merged_data, publications)

    # Drop duplicate columns and reorder
    merged_data = merged_data.drop(columns=["gtr_id", "person_id"])

    # Reorder columns thematically
    column_order = [
        # Personal identifiers
        "id",  # OpenAlex ID
        "orcid",  # ORCID
        "display_name",  # Full name
        "first_name",  # First name from GTR
        "surname",  # Surname from GTR
        "display_name_alternatives",  # Alternative names
        # Current institutional information
        "organisation",  # Current organisation ID
        "organisation_name",  # Current organisation name
        "last_known_institution_uk",  # Whether last known institution is in UK
        "last_known_institutions",  # List of last known institutions
        # Academic metrics
        "works_count",  # Total number of works
        "cited_by_count",  # Total citations
        "citations_per_publication",  # Average citations per publication
        "h_index",  # H-index
        "i10_index",  # i10-index
        "first_work_year",  # First publication year
        "academic_age_at_first_grant",  # Academic age when receiving first grant
        # Grant information
        "earliest_start_date",  # First grant start date
        "latest_end_date",  # Last grant end date
        "has_active_project",  # Whether has active projects
        "number_grants",  # Total number of grants
        "has_multiple_funders",  # Whether has multiple funders
        "grant_categories",  # List of grant categories
        "lead_funders",  # List of lead funders
        "project_timeline",  # Detailed project timeline
        # Research profile
        "topics",  # Research topics
        "affiliations",  # Historical affiliations
        "counts_by_year",  # Publication counts by year
        # Project outputs
        "project_publications",  # Project-linked publications
        "project_topics",  # Project-specific topics
        "project_authors",  # Project collaborators
        "project_oa_ids",  # Project OpenAlex IDs
        # International experience
        "abroad_experience_before",  # International experience before first grant
        "abroad_experience_after",  # International experience after first grant
        "countries_before",  # Countries worked in before first grant
        "countries_after",  # Countries worked in after first grant
        "abroad_fraction_before",  # Fraction of time abroad before first grant
        "abroad_fraction_after",  # Fraction of time abroad after first grant
        # Collaboration metrics
        "unique_collabs_before",  # Unique collaborators before first grant
        "unique_collabs_after",  # Unique collaborators after first grant
        "total_collabs_before",  # Total collaborations before first grant
        "total_collabs_after",  # Total collaborations after first grant
        "foreign_collab_fraction_before",  # Fraction of foreign collabs before
        "foreign_collab_fraction_after",  # Fraction of foreign collabs after
        "collab_countries_before",  # Collaboration countries before with counts
        "collab_countries_after",  # Collaboration countries after with counts
        "collab_countries_list_before",  # List of collaboration countries before
        "collab_countries_list_after",  # List of collaboration countries after
    ]

    # Reindex with only existing columns (in case some are missing)
    existing_columns = [col for col in column_order if col in merged_data.columns]
    merged_data = merged_data.reindex(columns=existing_columns)

    logger.info("Completed computing basic metrics")
    return merged_data

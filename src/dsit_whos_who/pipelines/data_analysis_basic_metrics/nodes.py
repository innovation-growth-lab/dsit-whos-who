"""
This module contains the nodes for the data analysis basic metrics pipeline. Each node
performs a specific data processing task for analysing researcher metrics, including
publication data, collaboration patterns, and academic impact measures.
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
    add_publication_metrics,
)
from ..data_collection_oa.utils import preprocess_ids

logger = logging.getLogger(__name__)


def create_list_oa_author_ids(authors: pd.DataFrame) -> list:
    """
    Create a list of OpenAlex author IDs from a DataFrame of persons.

    This function processes a DataFrame of authors to extract unique OpenAlex IDs and
    prepares them for batch processing. It removes duplicates and formats the IDs
    appropriately for API queries.

    Args:
        authors (pd.DataFrame): A DataFrame containing author information with an 'oa_id'
            column representing OpenAlex identifiers.

    Returns:
        list: A preprocessed list of unique OpenAlex author IDs ready for API queries.
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
    """
    Fetch publication works data from OpenAlex for matched authors.

    This node retrieves detailed publication information from the OpenAlex API for a list
    of matched author IDs. It handles pagination and applies specified filtering criteria
    to the results.

    Args:
        ids (list): List of OpenAlex author IDs to fetch works for.
        mails (list): List of email addresses for API authentication.
        perpage (int): Number of results to return per page.
        filter_criteria (str): Filter string to apply to the OpenAlex API query.
        **kwargs: Additional keyword arguments to pass to the API query.

    Returns:
        pd.DataFrame: DataFrame containing the fetched publication works data.
    """
    return fetch_openalex_works(ids, mails, perpage, filter_criteria, **kwargs)


def process_matched_author_metadata(
    author_loaders: AbstractDataset,
    matched_authors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Process metadata for authors matched between OpenAlex and GTR.

    This node processes metadata for authors who have been matched between OpenAlex and
    Gateway to Research (GTR). It performs the following steps:
    1. Iterates through author data loaded from OpenAlex
    2. For each batch, processes the metadata to extract key metrics such as:
        - First year of publication
        - Citations per publication
        - Publication impact measures
    3. Filters authors to include only those matched with GTR
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
    """
    Process GTR data for matched persons.

    This node processes Gateway to Research (GTR) data for persons who have been matched
    to OpenAlex authors. It performs the following steps:
    1. Preprocesses and filters persons data to include only matched authors
    2. Merges person-project relationships with project metadata
    3. Cleans project dates and processes timeline information
    4. Creates aggregated summaries of project data per person, including:
        - Project timelines
        - Grant categories
        - Funding information
        - Collaboration patterns

    Args:
        persons (pd.DataFrame): DataFrame containing person data from GTR
        projects (pd.DataFrame): DataFrame containing project data from GTR
        matched_authors (pd.DataFrame): DataFrame containing matched author IDs between
            GTR and OpenAlex

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
    """
    Process publication data to create a summary of collaborations per author per year.

    This function creates an intermediate DataFrame that summarises collaboration metrics
    for each matched author in each year of their publications. This makes subsequent
    analysis more efficient by pre-aggregating the data. The function processes:
        - UK and international collaborations
        - Temporal collaboration patterns
        - Geographic distribution of collaborators
        - Publication impact metrics

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
            - collaboration_metrics: Various metrics about collaboration patterns
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
                "affiliation_countries",
                "n_affils_uk",
                "n_affils_abroad",
                "n_affils_unknown",
                "fwci",
                "cited_by_count",
                "n_pubs",
                "n_collab_uk",
                "n_collab_abroad",
                "n_collab_unknown",
                "collab_countries",
                "collab_ids",
            ]
        )

    result_df = (
        result_df.groupby(["author_id", "year"])
        .agg(
            {
                "affiliation_countries": lambda x: sorted(
                    list(set(country for countries in x for country in countries))
                ),
                "n_affils_uk": "sum",
                "n_affils_abroad": "sum",
                "n_affils_unknown": "sum",
                "fwci": "mean",
                "cited_by_count": "sum",
                "n_collab_uk": "sum",
                "n_collab_abroad": "sum",
                "n_collab_unknown": "sum",
                "collab_countries": lambda x: sorted(
                    list(set(country for countries in x for country in countries))
                ),
                "collab_ids": lambda x: sorted(
                    list(set(id for ids in x for id in ids))
                ),
                "author_id": "size",  # Count number of rows per group
            }
        )
        .rename(
            columns={
                "author_id": "n_pubs",
                "collab_countries": "unique_collab_countries",
                "collab_ids": "unique_collab_ids",
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
) -> pd.DataFrame:
    """
    Compute basic metrics for matched authors.

    This function combines data from multiple sources to compute comprehensive metrics
    for each matched author. It processes:
        - Publication metrics (counts, citations, impact)
        - Collaboration patterns (domestic and international)
        - Career trajectory metrics
        - Research funding patterns
        - Academic age and career stage indicators

    Args:
        author_data (pd.DataFrame): Processed author metadata from OpenAlex
        person_data (pd.DataFrame): Processed person data from GTR
        publications (pd.DataFrame): Processed publication data with collaboration info

    Returns:
        pd.DataFrame: DataFrame containing comprehensive metrics for each matched author,
            including publication impact, collaboration patterns, and career indicators
    """
    author_data = author_data.drop_duplicates(subset=["id", "gtr_id"])

    logger.info("Computing basic metrics for %s authors", len(author_data))
    merged_data = author_data.merge(
        person_data, left_on="gtr_id", right_on="person_id", how="inner", validate="1:1"
    )
    logger.info("Merged data contains %s records", len(merged_data))

    # Add publication metrics
    merged_data = add_publication_metrics(merged_data, publications)

    # Compute academic age
    merged_data["academic_age_at_first_grant"] = merged_data.apply(
        compute_academic_age, axis=1
    ).astype("Int64")

    # Add international experience metrics
    merged_data = add_international_metrics(merged_data, publications)

    # Drop person_id
    merged_data = merged_data.drop(columns=["gtr_id"])

    # Rename id to oa_id
    merged_data = merged_data.rename(
        columns={
            "id": "oa_id",
            "project_publications": "gtr_project_publications",
            "project_id": "gtr_project_id",
            "project_timeline": "gtr_project_timeline",
            "project_topics": "gtr_project_topics",
            "project_authors": "gtr_project_oa_authors",
            "project_oa_ids": "gtr_project_oa_ids",
            "organisation": "gtr_organisation",
            "organisation_name": "gtr_organisation_name",
            "person_id": "gtr_person_id",
        }
    )

    # Reorder columns thematically
    column_order = [
        # Personal identifiers and basic info
        "oa_id",  # OpenAlex ID
        "orcid",  # ORCID ID
        "display_name",  # Full name from OpenAlex
        "display_name_alternatives",  # Alternative names
        "first_name",  # First name from GTR
        "surname",  # Surname from GTR
        "gtr_person_id",  # GTR person ID
        "match_probability",  # Probability of correct matching
        # Current institutional information
        "gtr_organisation",  # Current GTR organisation ID
        "gtr_organisation_name",  # Current GTR organisation name
        "last_known_institutions",  # List of last known institutions
        "last_known_institution_uk",  # Whether last known institution is in UK
        # Academic profile and metrics
        "works_count",  # Total number of works
        "cited_by_count",  # Total citations
        "citations_per_publication",  # Average citations per publication
        "h_index",  # H-index
        "i10_index",  # i10-index
        "first_work_year",  # First publication year
        "academic_age_at_first_grant",  # Academic age when receiving first grant
        "topics",  # Research topics
        "affiliations",  # Historical affiliations
        "counts_by_year",  # Publication counts by year
        "counts_by_pubyear",  # Publication counts by publication year
        # Grant information
        "earliest_start_date",  # First grant start date
        "latest_end_date",  # Last grant end date
        "has_active_project",  # Whether has active projects
        "number_grants",  # Total number of grants
        "has_multiple_funders",  # Whether has multiple funders
        "grant_categories",  # List of grant categories
        "lead_funders",  # List of lead funders
        "gtr_project_timeline",  # Detailed project timeline
        "gtr_project_id",  # GTR project IDs
        "gtr_project_publications",  # Project-linked publications
        "gtr_project_topics",  # Project-specific topics
        "gtr_project_oa_authors",  # Project OpenAlex authors
        "gtr_project_oa_ids",  # Project OpenAlex IDs
        # Publication metrics before/after first grant
        "n_pubs_before",  # Number of publications before
        "n_pubs_after",  # Number of publications after
        "total_citations_pubyear_before",  # Total citations by pub year before
        "total_citations_pubyear_after",  # Total citations by pub year after
        "mean_citations_pubyear_before",  # Mean citations by pub year before
        "mean_citations_pubyear_after",  # Mean citations by pub year after
        "citations_pp_pubyear_before",  # Citations per pub by pub year before
        "citations_pp_pubyear_after",  # Citations per pub by pub year after
        "mean_citations_before",  # Mean citations before
        "mean_citations_after",  # Mean citations after
        "citations_pp_before",  # Citations per pub before
        "citations_pp_after",  # Citations per pub after
        "mean_fwci_before",  # Mean FWCI before
        "mean_fwci_after",  # Mean FWCI after
        # International experience metrics
        "abroad_experience_before",  # Had international experience before
        "abroad_experience_after",  # Had international experience after
        "countries_before",  # Countries worked in before
        "countries_after",  # Countries worked in after
        "abroad_fraction_before",  # Fraction of time abroad before
        "abroad_fraction_after",  # Fraction of time abroad after
        # Collaboration metrics
        "collab_countries_before",  # Collaboration countries with counts before
        "collab_countries_after",  # Collaboration countries with counts after
        "collab_countries_list_before",  # List of collaboration countries before
        "collab_countries_list_after",  # List of collaboration countries after
        "unique_collabs_before",  # Unique collaborators before
        "unique_collabs_after",  # Unique collaborators after
        "total_collabs_before",  # Total collaborations before
        "total_collabs_after",  # Total collaborations after
        "foreign_collab_fraction_before",  # Fraction of foreign collabs before
        "foreign_collab_fraction_after",  # Fraction of foreign collabs after
    ]

    # Reindex with only existing columns (in case some are missing)
    existing_columns = [col for col in column_order if col in merged_data.columns]
    merged_data = merged_data.reindex(columns=existing_columns)

    logger.info("Completed computing basic metrics")
    return merged_data

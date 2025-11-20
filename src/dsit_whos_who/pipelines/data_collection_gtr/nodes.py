"""
Gateway to Research (GtR) data preprocessing pipeline.

This module provides comprehensive functionality for fetching and preprocessing data
from the Gateway to Research API. It handles:
- Organisation, fund, publication, project, and person data preprocessing
- API request management with retry logic
- Parallel data fetching and processing
- Data transformation and standardisation

The module is structured around the GtRDataPreprocessor class, which provides
specialised methods for each data type, ensuring consistent and efficient
data processing across all GtR endpoints.
"""

import logging
import random
import time
import datetime
from typing import Dict, Union, Generator, Optional, Callable
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
from kedro.io import AbstractDataset
from .utils import (
    api_config,
    extract_main_address,
    extract_value_from_nested_dict,
    extract_date,
    transform_nested_dict,
)

logger = logging.getLogger(__name__)


class GtRDataPreprocessor:
    """
    Preprocessor for Gateway to Research data types.

    Provides specialised methods for transforming raw API data into standardised
    formats. Handles:
    - Organisations: Address extraction and standardisation
    - Funds: Currency and amount normalisation
    - Publications: Date standardisation and metadata extraction
    - Projects: Research topic classification and timeline extraction
    - Persons: Role and affiliation processing
    """

    def __init__(self) -> None:
        """Initialise preprocessor with type-specific methods."""
        self.methods = {
            "organisations": self._preprocess_organisations,
            "funds": self._preprocess_funds,
            "publications": self._preprocess_publications,
            "projects": self._preprocess_projects,
            "persons": self._preprocess_persons,
        }

    def _preprocess_organisations(self, org_df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw organisation data into standardised format.

        Processes:
        - Main address extraction and formatting
        - Address component standardisation
        - Removal of redundant link data

        Args:
            org_df: Raw organisation data from GtR API

        Returns:
            DataFrame with standardised organisation data
        """
        address_columns = org_df["addresses"].apply(extract_main_address)
        address_columns = address_columns.drop("created", axis=1).add_prefix("address_")
        org_df = org_df.drop("addresses", axis=1).join(address_columns)
        org_df = org_df.drop(columns=["links"])
        return org_df

    def _preprocess_funds(self, funds_df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw funding data into standardised format.

        Processes:
        - Currency amount extraction (GBP)
        - Value standardisation
        - Removal of redundant link data

        Args:
            funds_df: Raw funding data from GtR API

        Returns:
            DataFrame with standardised funding data
        """
        funds_df["value"] = funds_df["valuePounds"].apply(lambda x: x["amount"])
        funds_df = funds_df.drop("valuePounds", axis=1)
        funds_df = funds_df.drop(columns=["links"])
        return funds_df

    def _preprocess_publications(self, publications_df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw publication data into standardised format.

        Processes:
        - Project ID extraction
        - Publication date standardisation
        - Column renaming and selection
        - Author metadata extraction

        Args:
            publications_df: Raw publication data from GtR API

        Returns:
            DataFrame with standardised publication data
        """
        # extract project_id
        publications_df["project_id"] = publications_df["links"].apply(
            lambda x: extract_value_from_nested_dict(
                data=x,
                outer_key="link",
                inner_key="rel",
                inner_value="PROJECT",
                extract_key="href",
            )
        )

        # create publication_date from datePublished (milliseconds)
        publications_df["publication_date"] = publications_df["datePublished"].apply(
            lambda x: (
                datetime.datetime.fromtimestamp(x / 1000).strftime("%Y-%m-%d")
                if np.isfinite(x)
                else np.nan
            )
        )

        # rename cols
        publications_df = publications_df.rename(
            columns={
                "id": "outcome_id",
                "journalTitle": "journal_title",
                "publicationUrl": "publication_url",
            }
        )

        return publications_df[
            [
                "project_id",
                "outcome_id",
                "pubMedId",
                "isbn",
                "issn",
                "doi",
                "title",
                "chapterTitle",
                "type",
                "publication_date",
                "journal_title",
                "publication_url",
                "author",
            ]
        ]

    def _preprocess_projects(self, projects_df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw project data into standardised format.

        Processes:
        - Research subject and topic extraction
        - Person role identification
        - Timeline standardisation (start/end dates)
        - Publication linkage
        - Nested data structure flattening

        Args:
            projects_df: Raw project data from GtR API

        Returns:
            DataFrame with standardised project data
        """
        columns_to_transform = {
            "identifiers": ["value", "type"],
            "researchSubjects": ["id", "text", "percentage"],
            "researchTopics": ["id", "text", "percentage"],
        }

        for col, keys in columns_to_transform.items():
            projects_df = transform_nested_dict(projects_df, col, keys)

        # extract person information from links
        projects_df["persons"] = projects_df["links"].apply(
            lambda x: [
                {
                    "id": item.get("href", "").replace(
                        "http://gtr.ukri.org/gtr/apipersons/", ""
                    ),
                    "role": item.get("rel", ""),
                }
                for item in x["link"]
                if "apipersons" in item["href"]
            ]
        )

        # extract start and end dates from FUND, FURTHER_FUNDING
        projects_df["start_date"] = projects_df["links"].apply(
            lambda x: extract_date(x, "start", "FUND")
        )
        projects_df["end_date"] = projects_df["links"].apply(
            lambda x: extract_date(x, "end", "FUND")
        )
        projects_df["extended_end"] = projects_df["links"].apply(
            lambda x: extract_date(x, "end", "FURTHER_FUNDING")
        )

        projects_df["publications"] = projects_df["links"].apply(
            lambda x: [
                item["href"].replace(
                    "http://gtr.ukri.org/gtr/api/outcomes/publications/", ""
                )
                for item in x["link"]
                if item["rel"] == "PUBLICATION"
            ]
        )

        # rename cols
        projects_df = projects_df.rename(
            columns={
                "grantCategory": "grant_category",
                "abstractText": "abstract_text",
                "techAbstractText": "tech_abstract_text",
                "potentialImpact": "potential_impact",
                "leadFunder": "lead_funder",
                "leadOrganisationDepartment": "lead_org_department",
                "researchTopics": "research_topics",
                "researchSubjects": "research_subjects",
                "id": "project_id",
            }
        )

        return projects_df[
            [
                "project_id",
                "identifiers",
                "title",
                "abstract_text",
                "tech_abstract_text",
                "potential_impact",
                "status",
                "grant_category",
                "lead_funder",
                "lead_org_department",
                "research_topics",
                "research_subjects",
                "publications",
                "persons",
                "start_date",
                "end_date",
                "extended_end",
            ]
        ]

    def _preprocess_persons(self, persons_df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw person data into standardised format.

        Processes:
        - Name standardisation
        - Role extraction
        - Project linkage
        - Organisation affiliation processing

        Args:
            persons_df: Raw person data from GtR API

        Returns:
            DataFrame with standardised person data
        """
        persons_df["projects"] = persons_df["links"].apply(
            lambda x: [
                {
                    "id": item["href"].replace(
                        "http://gtr.ukri.org/gtr/apiprojects/", ""
                    ),
                    "role": item["rel"],
                }
                for item in x["link"]
                if "apiprojects" in item["href"]
            ]
        )

        persons_df["organisations"] = persons_df["links"].apply(
            lambda x: [
                {
                    "id": item["href"].replace(
                        "http://gtr.ukri.org/gtr/apiorganisations/", ""
                    ),
                    "role": item["rel"],
                }
                for item in x["link"]
                if "apiorganisations" in item["href"]
            ]
        )

        persons_df = persons_df.rename(
            columns={
                "id": "person_id",
                "firstName": "first_name",
                "otherNames": "other_names",
                "orcidId": "orcid_id",
            }
        )

        return persons_df[
            [
                "person_id",
                "orcid_id",
                "first_name",
                "surname",
                "other_names",
                "email",
                "projects",
                "organisations",
            ]
        ]


def fetch_gtr_data(
    parameters: Dict[str, Union[str, int]], url_endpoint: str, **kwargs
) -> Generator[Dict[str, pd.DataFrame], None, None]:
    """Fetch and preprocess data from Gateway to Research API.

    Implements:
    - Parallel data fetching with retry logic
    - Rate limiting and backoff
    - Response validation and error handling
    - Data preprocessing via GtRDataPreprocessor

    Args:
        parameters: API configuration parameters
        url_endpoint: Target API endpoint
        **kwargs: Additional configuration options

    Yields:
        Dictionary containing preprocessed data chunks
    """
    config = api_config(parameters, url_endpoint)

    page = 1
    total_pages = 1
    preprocessor = GtRDataPreprocessor()
    endpoint_type = url_endpoint.split("/")[-1]

    while page <= total_pages:
        page_data = []
        url = f"{url_endpoint}?p={page}&s={config['page_size']}"

        session = requests.Session()
        retries = Retry(
            total=config["max_retries"], backoff_factor=config["backoff_factor"]
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))

        try:
            response = session.get(url, headers=config["headers"], timeout=30)
            response.raise_for_status()
            data = response.json()

            if "totalPages" in data and page == 1:
                logger.info("Total pages: %s", data["totalPages"])
                total_pages = 2 if kwargs.get("test_mode") else data["totalPages"]

            if config["key"] in data:
                items = data[config["key"]]
                if not items:
                    logger.info("No more data to fetch. Exiting loop.")
                    break

                for item in items:
                    item["page_fetched_from"] = page  # add page info
                    page_data.append(item)
            else:
                logger.error(
                    "No '%s' key found in the response. Response: %s",
                    config["key"],
                    response.json(),
                )
                time.sleep(random.uniform(10, 60))
                continue

        except (ValueError, requests.RequestException) as e:
            logger.error("Failed to fetch or decode response: %s", e)
            time.sleep(random.uniform(10, 180))
            continue

        logger.info("Fetched page %s / %s", page, total_pages)
        page += 1
        time.sleep(random.uniform(0.3, 1))  # Respect web etiquette

        # preprocess before save
        if page_data:
            page_df = pd.DataFrame(page_data)
            page_df = preprocessor.methods[endpoint_type](page_df)
            yield {f"p{page-1}": page_df}


def concatenate_endpoint(
    abstract_dict: Union[AbstractDataset, Dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Merge preprocessed data chunks into unified dataset.

    Handles:
    - Chunk validation and error checking
    - Memory-efficient concatenation
    - Publication-specific data loading
    - Index standardisation

    Args:
        abstract_dict: Dictionary of data chunks or Kedro dataset

    Returns:
        Unified DataFrame with all preprocessed data
    """
    publications = Parallel(n_jobs=-1, verbose=10)(
        delayed(_load_publication)(key, load_function)
        for key, load_function in abstract_dict.items()
    )

    # Filter out None values
    publications = [p for p in publications if p is not None]

    # Check if publications is a list of dictionary entries
    if publications and all(isinstance(item, dict) for item in publications):
        # Create a dataframe directly from the list of dictionaries
        publication_data = pd.DataFrame(publications)
    else:
        # Concatenate all dataframes into a single dataframe
        publication_data = pd.concat(publications, ignore_index=True)

    return publication_data


def _load_publication(
    key: str, load_function: Union[Callable, pd.DataFrame]
) -> Optional[pd.DataFrame]:
    """Load and validate publication data chunk.

    Args:
        key: Chunk identifier
        load_function: Data loading function or DataFrame

    Returns:
        Validated publication DataFrame or None if invalid
    """
    try:
        if callable(load_function):
            df = load_function()
        else:
            df = load_function
        return df
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Failed to load DataFrame for %s: %s", key, e)
        return None

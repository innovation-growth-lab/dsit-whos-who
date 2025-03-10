"""
This script provides functionality for preprocessing GtR (Gateway to Research)
data. It includes a class `GtRDataPreprocessor` with methods to preprocess
different types of GtR data such as organisations, funds, publications, and
projects. Additionally, it includes utility functions for API configuration
and data extraction.

Classes:
    GtRDataPreprocessor: A class for preprocessing various types of GtR data.

Functions:
    fetch_gtr_data(parameters, endpoint, **kwargs): Fetches data from the GtR
        API and preprocesses it.
    concatenate_endpoint(abstract_dict): Concatenates DataFrames from a single
        endpoint into a single DataFrame.
"""

import logging
import random
import time
import datetime
from typing import Dict, Union, Generator
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
    transform_nested_dict,
)

logger = logging.getLogger(__name__)


class GtRDataPreprocessor:
    """
    Class for preprocessing GtR data.

    This class provides methods to preprocess different types of GtR data, such as organisations,
    funds, publications, and projects.
    """

    def __init__(self) -> None:
        self.methods = {
            "organisations": self._preprocess_organisations,
            "funds": self._preprocess_funds,
            "publications": self._preprocess_publications,
            "projects": self._preprocess_projects,
            "persons": self._preprocess_persons,
        }

    def _preprocess_organisations(self, org_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the organisations data.

        It extracts the main address and drops the "links" column.

        Args:
            org_df (pd.DataFrame): The organisations data.

        Returns:
            pd.DataFrame: The preprocessed data.
        """
        address_columns = org_df["addresses"].apply(extract_main_address)
        address_columns = address_columns.drop("created", axis=1).add_prefix("address_")
        org_df = org_df.drop("addresses", axis=1).join(address_columns)
        org_df = org_df.drop(columns=["links"])
        return org_df

    def _preprocess_funds(self, funds_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the funds data.

        Extracts the value in pound (ie. {'currencyCode': 'GBP', 'amount': 283590})
        for each row and drops the "links" column.

        Args:
            funds_df (pd.DataFrame): The funds data.

        Returns:
            pd.DataFrame: The preprocessed data.
        """
        funds_df["value"] = funds_df["valuePounds"].apply(lambda x: x["amount"])
        funds_df = funds_df.drop("valuePounds", axis=1)
        funds_df = funds_df.drop(columns=["links"])
        return funds_df

    def _preprocess_publications(self, publications_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the publications DataFrame by extracting project_id, renaming
        columns, and selecting specific columns.

        Args:
            publications_df (pd.DataFrame): The input DataFrame containing publications data.

        Returns:
            pd.DataFrame: The preprocessed DataFrame with selected columns.

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

        # create publication_date from datePublished (miliseconds)
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
        """Preprocess the projects data."""
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
            lambda x: _extract_date(x, "start", "FUND")
        )
        projects_df["end_date"] = projects_df["links"].apply(
            lambda x: _extract_date(x, "end", "FUND")
        )
        projects_df["extended_end"] = projects_df["links"].apply(
            lambda x: _extract_date(x, "end", "FURTHER_FUNDING")
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
        """Preprocess the persons data."""

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
) -> Generator[pd.DataFrame, None, None]:
    """Fetch data from the GtR API.

    Args:
        parameters (Dict[str, Union[str, int]]): Parameters for the API request.
        url_endpoint (str): The endpoint to fetch data from.

    Returns:
        List[Dict[str, Any]]: The fetched data.
    """
    config = api_config(parameters, url_endpoint)

    page = 1
    total_pages = 1
    preprocessor = GtRDataPreprocessor()

    while page <= total_pages:
        page_data = []
        url = f"{url_endpoint}?p={page}&s={config['page_size']}"
        session = requests.Session()
        retries = Retry(
            total=config["max_retries"], backoff_factor=config["backoff_factor"]
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))
        response = session.get(url, headers=config["headers"], timeout=30)
        try:
            data = response.json()
            if "totalPages" in data and page == 1:
                logger.info("Total pages: %s", data["totalPages"])
                total_pages = (
                    2 if kwargs.get("test_mode") is True else data["totalPages"]
                )
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
        except ValueError as e:
            logger.error("Failed to decode JSON response: %s", e)
            time.sleep(random.uniform(10, 180))
            continue

        logger.info("Fetched page %s / %s", page, total_pages)
        page += 1
        time.sleep(random.uniform(0.3, 1))  # [HACK] Respect web etiquette

        # preprocess before save
        page_df = pd.DataFrame(page_data)
        preprocessor = GtRDataPreprocessor()
        page_df = preprocessor.methods[url_endpoint.split("/")[-1]](page_df)
        yield {f"p{page}": page_df}


def concatenate_endpoint(
    abstract_dict: Union[AbstractDataset, Dict[str, Dict[str, str]]],
) -> pd.DataFrame:
    """
    Concatenate DataFrames from a single endpoint into a single DataFrame.

    Args:
        abstract_dict (AbstractDataset): A dictionary where the keys are
            the endpoint names and the values are functions that load the
            JSON dictionaries for each endpoint.

    Returns:
        pd.DataFrame: The concatenated DataFrame.
    """
    publications = Parallel(n_jobs=-1, verbose=10)(
        delayed(_load_publication)(key, load_function)
        for key, load_function in abstract_dict.items()
    )

    # check if publications is a list of dictionary entries
    if all(isinstance(item, dict) for item in publications):
        # create a dataframe directly from the list of dictionaries
        publication_data = pd.DataFrame(publications)
    else:
        # concatenate all dataframes into a single dataframe
        publication_data = pd.concat(publications, ignore_index=True)

    return publication_data


def _load_publication(key, load_function):
    try:
        try:
            df = load_function()
        except TypeError:
            df = load_function
        return df
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Failed to load DataFrame for %s: %s", key, e)
        return None


def _extract_date(links, extract_key, inner_value):
    timestamp = extract_value_from_nested_dict(
        data=links,
        outer_key="link",
        inner_key="rel",
        inner_value=inner_value,
        extract_key=extract_key,
        split_on_slash=False,
    )
    try:
        if timestamp:
            return datetime.datetime.fromtimestamp(timestamp / 1000).strftime(
                "%Y-%m-%d"
            )
    except (TypeError, ValueError):
        pass
    return None

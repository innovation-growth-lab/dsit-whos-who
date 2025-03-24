"""Common utilities for OpenAlex data collection."""

import logging
import random
from typing import Iterator, List, Dict, Union, Generator, Optional
import time
import requests
from requests.adapters import HTTPAdapter, Retry

from .publications import parse_works_results
from .authors import parse_author_results
from .institutions import parse_institution_results

logger = logging.getLogger(__name__)


def preprocess_ids(
    ids: Union[str, List[str], Dict[str, str]], grouped: bool = True
) -> List[str]:
    """Preprocesses ids to ensure they are in the correct format."""
    if isinstance(ids, str):
        ids = [ids]
    if isinstance(ids, dict):
        ids = list(ids.values())
    if grouped:
        ids = list(_chunk_oa_ids(ids))
    return ids


def _chunk_oa_ids(ids: List[str], chunk_size: int = 50) -> Generator[str, None, None]:
    """Yield successive chunk_size-sized chunks from ids."""
    for i in range(0, len(ids), chunk_size):
        yield "|".join(ids[i : i + chunk_size])


def openalex_generator(
    mails: List[str],
    perpage: str,
    oa_id: Union[str, List[str]],
    filter_criteria: Union[str, List[str]],
    session: requests.Session,
    endpoint: str = "works",
    sample_size: int = -1,
    select_variables: Optional[List[str]] = None,
) -> Iterator[list]:
    """Creates a generator that yields a list of objects from the OpenAlex API based on a
    given ID.

    Args:
        mails (List[str]): The email address to use for the API.
        perpage (str): The number of results to return per page.
        oa_id (Union[str, List[str]): The ID to use for the API.
        filter_criteria (Union[str, List[str]]): The filter criteria to use for the API.
        session (requests.Session): The requests session to use.
        endpoint (str): The OpenAlex endpoint to query (works, authors, institutions, etc).
        sample_size (int): Number of random samples to return. -1 means no sampling.
        select_variables (Optional[List[str]]): List of variables to select in the API response.

    Yields:
        Iterator[list]: A generator that yields a list of objects from the OpenAlex API.
    """
    cursor = "*"
    assert isinstance(
        filter_criteria, type(oa_id)
    ), "filter_criteria and oa_id must be of the same type."

    # multiple filter criteria
    if isinstance(filter_criteria, list) and isinstance(oa_id, list):
        filter_string = ",".join(
            [f"{criteria}:{id_}" for criteria, id_ in zip(filter_criteria, oa_id)]
        )
    else:
        filter_string = f"{filter_criteria}:{oa_id}"

    mailto = random.choice(mails)

    if sample_size == -1:
        cursor_url = (
            f"https://api.openalex.org/{endpoint}?filter={filter_string}"
            f"&mailto={mailto}&per-page={perpage}"
        )
        if select_variables is not None:
            cursor_url += f"&select={','.join(select_variables)}"
        cursor_url += "&cursor={}"

        try:
            # make a call to estimate total number of results
            response = session.get(cursor_url.format(cursor), timeout=20)
            data = response.json()

            while response.status_code == 429:  # needs testing (try with 200)
                logger.info("Waiting for 30 seconds...")
                time.sleep(30)
                response = session.get(cursor_url.format(cursor), timeout=20)
                data = response.json()

            logger.info("Fetching data for %s", oa_id[:50])
            total_results = data["meta"]["count"]
            num_calls = total_results // int(perpage) + 1
            logger.info("Total results: %s, in %s calls", total_results, num_calls)
            while cursor:
                response = session.get(cursor_url.format(cursor), timeout=20)
                data = response.json()
                results = data.get("results")
                cursor = data["meta"].get("next_cursor", False)
                yield results

        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error fetching data for %s: %s", oa_id, e)
            yield []
    else:  # OA does not accept cursor pagination with samples.
        cursor_url = (
            f"https://api.openalex.org/{endpoint}?filter={filter_string}&seed=123"
            f"&mailto={mailto}&per-page={perpage}&sample={sample_size}"
        )
        if select_variables is not None:
            cursor_url += f"&select={','.join(select_variables)}"
        cursor_url += "&page={}"

        try:
            # make a call to estimate total number of results
            response = session.get(cursor_url.format(1), timeout=20)
            data = response.json()

            while response.status_code == 429:  # needs testing (try with 200)
                logger.info("Waiting for 1 hour...")
                time.sleep(30)
                response = session.get(cursor_url.format(1), timeout=20)
                data = response.json()

            logger.info("Fetching data for %s", oa_id[:50])
            total_results = data["meta"]["count"]
            num_calls = total_results // int(perpage) + 1
            logger.info("Total results: %s, in %s calls", total_results, num_calls)
            for page in range(1, num_calls + 1):
                response = session.get(cursor_url.format(page), timeout=20)
                data = response.json()
                results = data.get("results")
                yield results

        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error fetching data for %s: %s", oa_id, e)
            yield []


def fetch_openalex_objects(
    oa_id: Union[str, List[str]],
    mails: List[str],
    perpage: str,
    filter_criteria: Union[str, List[str]],
    endpoint: str = "works",
    **kwargs,
) -> List[dict]:
    """Fetches objects from OpenAlex API based on ID and endpoint type.

    Args:
        oa_id (Union[str, List[str]]): The ID(s) to query.
        mails (List[str]): List of email addresses for API usage.
        perpage (str): Number of results per page.
        filter_criteria (Union[str, List[str]]): Filter criteria for the query.
        endpoint (str): OpenAlex endpoint (works, authors, institutions, etc).
        **kwargs: Additional arguments including:
            - sample_size: Number of random samples to return
            - keys_to_include: List of keys to include in results
            - gtr_author_names: For author search, the original GTR names to match against

    Returns:
        List[dict]: List of fetched objects from OpenAlex.
    """
    assert isinstance(
        filter_criteria, type(oa_id)
    ), "filter_criteria and oa_id must be of the same type."
    objects_for_id = []
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.3)
    session.mount("https://", HTTPAdapter(max_retries=retries))
    for page, objects in enumerate(
        openalex_generator(
            mails,
            perpage,
            oa_id,
            filter_criteria,
            session,
            endpoint=endpoint,
            sample_size=kwargs.get("sample_size", -1),
            select_variables=kwargs.get("select_variables", None),
        )
    ):
        if endpoint == "authors" and filter_criteria == "orcid":
            parsed_objects = parse_author_results(objects)
        elif endpoint == "authors" and filter_criteria == "display_name.search":
            parsed_objects = parse_author_results(objects, oa_id)
        elif endpoint == "works":
            parsed_objects = parse_works_results(
                objects, kwargs.get("keys_to_include", None)
            )
        elif endpoint == "institutions":
            parsed_objects = parse_institution_results(objects)
        else:
            raise ValueError(f"Parsing for endpoint '{endpoint}' not implemented yet")

        objects_for_id.extend(parsed_objects)
        logger.info(
            "Fetching page %s. Total objects collected: %s",
            page,
            len(objects_for_id),
        )

    return objects_for_id

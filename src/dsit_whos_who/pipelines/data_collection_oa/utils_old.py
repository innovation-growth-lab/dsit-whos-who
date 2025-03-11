import logging
import random
from typing import Iterator, List, Dict, Sequence, Union, Generator, Optional
import time
import requests
from requests.adapters import HTTPAdapter, Retry
import pandas as pd

logger = logging.getLogger(__name__)


def _revert_abstract_index(abstract_inverted_index: Dict[str, Sequence[int]]) -> str:
    """Reverts the abstract inverted index to the original text.

    Args:
        abstract_inverted_index (Dict[str, Sequence[int]]): The abstract inverted index.

    Returns:
        str: The original text.
    """
    try:
        length_of_text = (
            max(
                [
                    index
                    for sublist in abstract_inverted_index.values()
                    for index in sublist
                ]
            )
            + 1
        )
        recreated_text = [""] * length_of_text

        for word, indices in abstract_inverted_index.items():
            for index in indices:
                recreated_text[index] = word

        return " ".join(recreated_text)
    except (AttributeError, ValueError):
        return ""


def _parse_author_results(
    response: List[Dict], keys_to_include: Optional[List[str]] = None
) -> List[Dict]:
    """Parses OpenAlex API author response to retain specified keys or all if keys_to_include
     is None.

    Args:
        response (List[Dict]): The response from the OpenAlex API.
        keys_to_include (Optional[List[str]]): List of keys to include in the
            result. Returns full dictionary if None.

    Returns:
        List[Dict]: A list of dictionaries containing the parsed author information.
    """
    parsed_response = []
    for author in response:
        parsed_author = {
            "id": author.get("id", "").replace("https://openalex.org/", ""),
            "orcid": author.get("orcid", "").replace("https://orcid.org/", ""),
            "display_name": author.get("display_name", ""),
            "display_name_alternatives": author.get("display_name_alternatives", []),
            "works_count": author.get("works_count", 0),
            "cited_by_count": author.get("cited_by_count", 0),
            "h_index": author.get("summary_stats", {}).get("h_index", 0),
            "i10_index": author.get("summary_stats", {}).get("i10_index", 0),
            "affiliations": [
                {
                    "institution_id": aff["institution"]["id"].replace(
                        "https://openalex.org/", ""
                    ),
                    "institution_name": aff["institution"]["display_name"],
                    "country_code": aff["institution"]["country_code"],
                    "type": aff["institution"]["type"],
                    "years": aff["years"],
                }
                for aff in author.get("affiliations", [])
            ],
            "last_known_institutions": [
                {
                    "institution_id": inst["id"].replace("https://openalex.org/", ""),
                    "institution_name": inst["display_name"],
                    "country_code": inst["country_code"],
                    "type": inst["type"],
                }
                for inst in author.get("last_known_institutions", [])
            ],
            "topics": [
                {
                    "id": topic["id"].replace("https://openalex.org/", ""),
                    "display_name": topic["display_name"],
                    "count": topic["count"],
                    "subfield": {
                        "id": topic["subfield"]["id"].replace(
                            "https://openalex.org/", ""
                        ),
                        "display_name": topic["subfield"]["display_name"],
                    },
                    "field": {
                        "id": topic["field"]["id"].replace("https://openalex.org/", ""),
                        "display_name": topic["field"]["display_name"],
                    },
                    "domain": {
                        "id": topic["domain"]["id"].replace(
                            "https://openalex.org/", ""
                        ),
                        "display_name": topic["domain"]["display_name"],
                    },
                }
                for topic in author.get("topics", [])
            ],
            "counts_by_year": author.get("counts_by_year", []),
        }
        if keys_to_include is not None:
            # Filter the dictionary to only include specified keys
            parsed_author = {
                key: parsed_author[key]
                for key in keys_to_include
                if key in parsed_author
            }
        parsed_response.append(parsed_author)
    return parsed_response


def _parse_works_results(
    response: List[Dict], keys_to_include: Optional[List[str]] = None
) -> List[Dict]:
    """Parses OpenAlex API works response to retain specified keys or all if keys_to_include
     is None.

    Args:
        response (List[Dict]): The response from the OpenAlex API.
        keys_to_include (Optional[List[str]]): List of keys to include in the
            result. Returns full dictionary if None.

    Returns:
        List[Dict]: A list of dictionaries containing the parsed works information.
    """
    parsed_response = []
    for paper in response:
        parsed_paper = {
            "id": paper.get("id", "").replace("https://openalex.org/", ""),
            "doi": paper.get("doi", ""),
            "title": paper.get("title", ""),
            "publication_date": paper.get("publication_date", ""),
            "abstract": _revert_abstract_index(
                paper.get("abstract_inverted_index", {})
            ),
            "fwci": paper.get("fwci", ""),
            "citation_normalized_percentile": paper.get(
                "citation_normalized_percentile", []
            ),
            "authorships": paper.get("authorships", []),
            "cited_by_count": paper.get("cited_by_count", ""),
            "concepts": paper.get("concepts", []),
            "mesh_terms": paper.get("mesh", []),
            "topics": paper.get("topics", []),
            "grants": paper.get("grants", []),
            "referenced_works": paper.get("referenced_works", []),
            "ids": paper.get("ids", []),
            "counts_by_year": paper.get("counts_by_year", []),
        }
        if keys_to_include is not None:
            # Filter the dictionary to only include specified keys
            parsed_paper = {
                key: parsed_paper[key] for key in keys_to_include if key in parsed_paper
            }
        parsed_response.append(parsed_paper)
    return parsed_response


def _parse_results(
    response: List[Dict],
    keys_to_include: Optional[List[str]] = None,
    endpoint: str = "works",
) -> List[Dict]:
    """Parses OpenAlex API response based on the endpoint type.

    Args:
        response (List[Dict]): The response from the OpenAlex API.
        keys_to_include (Optional[List[str]]): List of keys to include in the
            result. Returns full dictionary if None.
        endpoint (str): The OpenAlex endpoint being queried (works, authors, etc).

    Returns:
        List[Dict]: A list of dictionaries containing the parsed information.
    """
    if endpoint == "authors":
        return _parse_author_results(response, keys_to_include)
    elif endpoint == "works":
        return _parse_works_results(response, keys_to_include)
    else:
        raise ValueError(f"Parsing for endpoint '{endpoint}' not implemented yet")


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


def _openalex_generator(
    mails: List[str],
    perpage: str,
    oa_id: Union[str, List[str]],
    filter_criteria: Union[str, List[str]],
    session: requests.Session,
    endpoint: str = "works",
    sample_size: int = -1,
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
            f"&mailto={mailto}&per-page={perpage}&cursor={{}}"
        )

        try:
            # make a call to estimate total number of results
            response = session.get(cursor_url.format(cursor), timeout=20)
            data = response.json()

            while response.status_code == 429:  # needs testing (try with 200)
                logger.info("Waiting for 1 hour...")
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
            f"&mailto={mailto}&per-page={perpage}&sample={sample_size}&page={{}}"
        )

        try:
            # make a call to estimate total number of results
            response = session.get(cursor_url.format(1), timeout=20)
            data = response.json()

            while response.status_code == 429:  # needs testing (try with 200)
                logger.info("Waiting for 1 hour...")
                time.sleep(3600)
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
        **kwargs: Additional arguments including sample_size and keys_to_include.

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
        _openalex_generator(
            mails,
            perpage,
            oa_id,
            filter_criteria,
            session,
            endpoint=endpoint,
            sample_size=kwargs.get("sample_size", -1),
        )
    ):
        objects_for_id.extend(
            _parse_results(
                objects, kwargs.get("keys_to_include", None), endpoint=endpoint
            )
        )
        logger.info(
            "Fetching page %s. Total objects collected: %s",
            page,
            len(objects_for_id),
        )

    return objects_for_id


def json_loader(
    data: Dict[str, Union[str, List[str]]], endpoint: str = "works"
) -> pd.DataFrame:
    """
    Load JSON data, transform it into a DataFrame, and wrangle data based on endpoint type.

    Args:
        data (Dict[str, Union[str, List[str]]]): The JSON data.
        endpoint (str): The OpenAlex endpoint type (works, authors, etc).

    Returns:
        pandas.DataFrame: The transformed DataFrame.
    """
    output = []

    for batch in data:
        if endpoint == "works":
            json_data = [
                {
                    k: v
                    for k, v in item.items()
                    if k
                    in [
                        "id",
                        "ids",
                        "doi",
                        "title",
                        "publication_date",
                        "cited_by_count",
                        "fwci",
                        "citation_normalized_percentile",
                        "counts_by_year",
                        "authorships",
                        "topics",
                        "concepts",
                        "grants",
                    ]
                }
                for item in batch
            ]

            df = pd.DataFrame(json_data)
            if df.empty:
                continue

            df["pmid"] = df["ids"].apply(
                lambda x: (
                    x.get("pmid").replace("https://pubmed.ncbi.nlm.nih.gov/", "")
                    if x and x.get("pmid")
                    else None
                )
            )

            df["mag_id"] = df["ids"].apply(
                lambda x: (x.get("mag") if x and x.get("mag") else None)
            )

            # break atuhorship nested dictionary jsons, create triplets of authorship
            df["authorships"] = df["authorships"].apply(
                lambda x: (
                    [
                        (
                            (
                                author["author"]["id"].replace(
                                    "https://openalex.org/", ""
                                ),
                                inst["id"].replace("https://openalex.org/", ""),
                                inst["country_code"],
                                author["author_position"],
                            )
                            if author["institutions"]
                            else [
                                author["author"]["id"].replace(
                                    "https://openalex.org/", ""
                                ),
                                "",
                                "",
                                author["author_position"],
                            ]
                        )
                        for author in x
                        for inst in author["institutions"] or [{}]
                    ]
                    if x
                    else None
                )
            )

            # create tuples from counts by year, if available
            df["counts_by_year"] = df["counts_by_year"].apply(
                lambda x: (
                    [(year["year"], year["cited_by_count"]) for year in x]
                    if x
                    else None
                )
            )

            # create a list of topics
            df["topics"] = df["topics"].apply(
                lambda x: (
                    [
                        (
                            topic["id"].replace("https://openalex.org/", ""),
                            topic["display_name"],
                            topic["subfield"]["id"].replace(
                                "https://openalex.org/", ""
                            ),
                            topic["subfield"]["display_name"],
                            topic["field"]["id"].replace("https://openalex.org/", ""),
                            topic["field"]["display_name"],
                            topic["domain"]["id"].replace("https://openalex.org/", ""),
                            topic["domain"]["display_name"],
                        )
                        for topic in x
                    ]
                    if x
                    else None
                )
            )

            # extract concepts
            df["concepts"] = df["concepts"].apply(
                lambda x: (
                    [
                        (
                            concept["id"].replace("https://openalex.org/", ""),
                            concept["display_name"],
                        )
                        for concept in x
                    ]
                    if x
                    else None
                )
            )

            # process grants, getting triplets out of "funder", "funder_display_name", and "award_id"
            df["grants"] = df["grants"].apply(
                lambda x: (
                    [
                        (
                            grant.get("funder", {}).replace(
                                "https://openalex.org/", ""
                            ),
                            grant.get("funder_display_name"),
                            grant.get("award_id"),
                        )
                        for grant in x
                    ]
                    if x
                    else None
                )
            )

            df = df[
                [
                    "id",
                    "doi",
                    "pmid",
                    "mag_id",
                    "title",
                    "publication_date",
                    "cited_by_count",
                    "counts_by_year",
                    "authorships",
                    "topics",
                    "concepts",
                    "grants",
                ]
            ]

        elif endpoint == "authors":
            json_data = [
                {
                    k: v
                    for k, v in item.items()
                    if k
                    in [
                        "id",
                        "orcid",
                        "display_name",
                        "display_name_alternatives",
                        "works_count",
                        "cited_by_count",
                        "h_index",
                        "i10_index",
                        "affiliations",
                        "last_known_institutions",
                        "topics",
                        "counts_by_year",
                    ]
                }
                for item in batch
            ]

            df = pd.DataFrame(json_data)
            if df.empty:
                continue

            # Process affiliations
            df["affiliations"] = df["affiliations"].apply(
                lambda x: (
                    [
                        [
                            aff["institution_id"],
                            aff["institution_name"],
                            aff["country_code"],
                            aff["type"],
                            (
                                ",".join([str(y) for y in aff["years"]])
                                if isinstance(aff["years"], list)
                                else str(aff["years"])
                            ),
                        ]
                        for aff in x
                    ]
                    if x
                    else None
                )
            )

            # Process last known institutions
            df["last_known_institutions"] = df["last_known_institutions"].apply(
                lambda x: (
                    [
                        [
                            str(inst["institution_id"]),
                            str(inst["institution_name"]),
                            str(inst["country_code"]),
                            str(inst["type"]),
                        ]
                        for inst in x
                    ]
                    if x
                    else None
                )
            )

            # Process topics
            df["topics"] = df["topics"].apply(
                lambda x: (
                    [
                        [
                            topic["id"],
                            topic["display_name"],
                            str(topic["count"]),
                            topic["subfield"]["id"],
                            topic["subfield"]["display_name"],
                            topic["field"]["id"],
                            topic["field"]["display_name"],
                            topic["domain"]["id"],
                            topic["domain"]["display_name"],
                        ]
                        for topic in x
                    ]
                    if x
                    else None
                )
            )

            # Process counts by year
            df["counts_by_year"] = df["counts_by_year"].apply(
                lambda x: (
                    [
                        [
                            int(year["year"]),
                            int(year["works_count"]),
                            int(year["cited_by_count"]),
                        ]
                        for year in x
                    ]
                    if x
                    else None
                )
            )

        else:
            raise ValueError(
                f"JSON loading for endpoint '{endpoint}' not implemented yet"
            )

        # append to output
        output.append(df)

    df = pd.concat(output)

    return df

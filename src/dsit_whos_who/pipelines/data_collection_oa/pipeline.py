"""
This pipeline fetches data from the GtR API and preprocesses it into a format
that can be used by the rest of the project.

Pipelines:
    - data_collection_gtr:
        Fetches and preprocesses data from the GtR API.

Dependencies:
    - Kedro
    - pandas
    - requests
    - logging

Usage:
    Run the pipeline to fetch and preprocess data from the GtR API.

Command Line Example:
    ```
    kedro run --pipeline data_collection_gtr
    ```
    Alternatively, you can run this pipeline for a single endpoint:
    ```
    kedro run --pipeline data_collection_gtr --tags projects
    ```

Note:
    In regards to the use of namespaces, note that these are appended as
    prefixes to the outputs of the nodes in the pipeline.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    create_list_doi_inputs,
    create_list_orcid_inputs,
    create_list_author_names_inputs,
    fetch_openalex,
    concatenate_openalex,
    extract_institution_ids,
)


def create_pipeline(**kwargs) -> Pipeline:  # pylint: disable=W0613
    """Pipeline for data collection.

    Returns:
        Pipeline: The data collection pipeline.
    """
    orcid_pipeline = pipeline(
        [
            node(
                func=create_list_orcid_inputs,
                inputs="gtr.data_collection.persons.intermediate",
                outputs="oa.data_collection.gtr.orcid.list",
                name="create_nested_orcid_list",
            ),
            node(
                func=fetch_openalex,
                inputs={
                    "perpage": "params:oa.data_collection.api.perpage",
                    "mails": "params:oa.data_collection.api.mails",
                    "ids": "oa.data_collection.gtr.orcid.list",
                    "filter_criteria": "params:oa.data_collection.filter_orcid",
                    "parallel_jobs": "params:oa.data_collection.n_jobs",
                    "endpoint": "params:oa.data_collection.authors_endpoint",
                },
                outputs="oa.data_collection.orcid.raw",
                name="fetch_orcid",
            ),
            node(
                func=concatenate_openalex,
                inputs={
                    "data": "oa.data_collection.orcid.raw",
                    "endpoint": "params:oa.data_collection.authors_endpoint",
                },
                outputs="oa.data_collection.orcid.intermediate",
                name="concatenate_orcid",
            ),
        ],
        tags="fetch_orcid",
    )

    author_search_pipeline = pipeline(
        [
            node(
                func=create_list_author_names_inputs,
                inputs="gtr.data_collection.persons.intermediate",
                outputs="oa.data_collection.gtr.author_search.list",
            ),
            node(
                func=fetch_openalex,
                inputs={
                    "perpage": "params:oa.data_collection.api.perpage",
                    "mails": "params:oa.data_collection.api.mails",
                    "ids": "oa.data_collection.gtr.author_search.list",
                    "filter_criteria": "params:oa.data_collection.filter_author_search",
                    "parallel_jobs": "params:oa.data_collection.n_jobs",
                    "endpoint": "params:oa.data_collection.authors_endpoint",
                },
                outputs="oa.data_collection.author_search.raw",
                name="fetch_author_names",
            ),
        ],
        tags="fetch_author_names",
    )

    publications_pipeline = pipeline(
        [
            node(
                func=create_list_doi_inputs,
                inputs="gtr.data_collection.publications.intermediate",
                outputs="oa.data_collection.publications.list",
            ),
            node(
                fetch_openalex,
                inputs={
                    "perpage": "params:oa.data_collection.api.perpage",
                    "mails": "params:oa.data_collection.api.mails",
                    "ids": "oa.data_collection.publications.list",
                    "filter_criteria": "params:oa.data_collection.filter_doi",
                    "parallel_jobs": "params:oa.data_collection.n_jobs",
                    "endpoint": "params:oa.data_collection.publications_endpoint",
                },
                outputs="oa.data_collection.publications.raw",
                name="fetch_doi",
            ),
            node(
                concatenate_openalex,
                inputs={
                    "data": "oa.data_collection.publications.raw",
                    "endpoint": "params:oa.data_collection.publications_endpoint",
                },
                outputs="oa.data_collection.publications.intermediate",
                name="concatenate_publications",
            ),
        ],
        tags="fetch_doi_publications",
    )

    institutions_pipeline = pipeline(
        [
            node(
                func=extract_institution_ids,
                inputs="oa.data_collection.author_search.raw",
                outputs="oa.data_collection.institutions.list",
                name="extract_institution_ids",
            ),
            node(
                func=fetch_openalex,
                inputs={
                    "perpage": "params:oa.data_collection.api.perpage",
                    "mails": "params:oa.data_collection.api.mails",
                    "ids": "oa.data_collection.institutions.list",
                    "filter_criteria": "params:oa.data_collection.filter_oa",
                    "parallel_jobs": "params:oa.data_collection.n_jobs",
                    "endpoint": "params:oa.data_collection.institutions_endpoint",
                },
                outputs="oa.data_collection.institutions.raw",
                name="fetch_institutions",
            ),
            node(
                func=concatenate_openalex,
                inputs={
                    "data": "oa.data_collection.institutions.raw",
                    "endpoint": "params:oa.data_collection.institutions_endpoint",
                },
                outputs="oa.data_collection.institutions.intermediate",
                name="concatenate_institutions",
            ),
        ],
        tags="fetch_institutions",
    )

    return (
        orcid_pipeline
        + author_search_pipeline
        + publications_pipeline
        + institutions_pipeline
    )

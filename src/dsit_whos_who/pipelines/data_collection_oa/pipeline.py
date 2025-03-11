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
    create_list_orcid_inputs,
    fetch_openalex,
    concatenate_openalex,
)


def create_pipeline(**kwargs) -> Pipeline:  # pylint: disable=W0613
    """Pipeline for data collection.

    Returns:
        Pipeline: The data collection pipeline.
    """
    return pipeline(
        [
            node(
                func=create_list_orcid_inputs,
                inputs="gtr.data_collection.persons.intermediate",
                outputs="oa.data_collection.gtr.orcid_list",
                name="create_nested_orcid_list",
            ),
            node(
                func=fetch_openalex,
                inputs={
                    "perpage": "params:oa.data_collection.api.perpage",
                    "mails": "params:oa.data_collection.api.mails",
                    "ids": "oa.data_collection.gtr.orcid_list",
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

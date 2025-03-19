"""
Command Line Example:
    ```
    kedro run --pipeline data_analysis_basic_metrics
    ```
    Alternatively, you can run this pipeline for a single endpoint:
    ```
    kedro run --pipeline data_analysis_basic_metrics --tags projects
    ```
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    create_list_oa_author_ids,
    fetch_openalex_matched_author_works,
    process_matched_author_metadata,
    process_matched_person_gtr_data,
)


def create_pipeline(**kwargs) -> Pipeline:  # pylint: disable=W0613
    """Pipeline for data collection.

    Returns:
        Pipeline: The data collection pipeline.
    """
    collection_pipeline = pipeline(
        [
            node(
                func=create_list_oa_author_ids,
                inputs="ad.matched_authors.primary",
                outputs="analysis.basic_metrics.oa.list",
                name="create_oa_author_ids",
            ),
            node(
                func=fetch_openalex_matched_author_works,
                inputs={
                    "perpage": "params:basic_metrics.oa.api.perpage",
                    "mails": "params:basic_metrics.oa.api.mails",
                    "ids": "analysis.basic_metrics.oa.list",
                    "filter_criteria": "params:basic_metrics.oa.filter",
                    "parallel_jobs": "params:basic_metrics.oa.n_jobs",
                    "endpoint": "params:basic_metrics.oa.publications_endpoint",
                    "keys_to_include": "params:basic_metrics.oa.keys_to_include",
                },
                outputs="analysis.basic_metrics.oa.raw",
                name="fetch_openalex_matched_author_works",
            ),
        ]
    )

    author_processing_pipeline = pipeline(
        [
            node(
                func=process_matched_author_metadata,
                inputs={
                    "author_loaders": "oa.data_collection.author_search.raw",
                    "matched_authors": "ad.matched_authors.primary",
                },
                outputs="analysis.basic_metrics.author_metadata.intermediate",
                name="process_matched_author_metadata",
            ),
            node(
                func=process_matched_person_gtr_data,
                inputs={
                    "persons": "ad.aggregated_persons.intermediate",
                    "projects": "gtr.data_collection.projects.intermediate",
                    "matched_authors": "ad.matched_authors.primary",
                },
                outputs="analysis.basic_metrics.gtr_data.intermediate",
                name="process_matched_person_gtr_data",
            ),
        ]
    )
    return collection_pipeline + author_processing_pipeline

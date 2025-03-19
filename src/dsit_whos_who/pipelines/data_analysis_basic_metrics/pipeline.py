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
from .nodes import create_list_oa_author_ids, fetch_openalex_matched_author_works


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
                name="fetch_oa_publications",
            )
        ]
    )

    return collection_pipeline

"""
Command Line Example:
    ```
    kedro run --pipeline data_analysis_complex_metrics
    ```
    Alternatively, you can run this pipeline for a single endpoint:
    ```
    kedro run --pipeline data_analysis_complex_metrics --tags projects
    ```
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import create_cited_work_ids, create_list_ids, fetch_openalex_work_citations


def create_pipeline(**kwargs) -> Pipeline:  # pylint: disable=W0613
    """Pipeline for data collection.

    Returns:
        Pipeline: The data collection pipeline.
    """
    collection_pipeline = pipeline(
        [
            node(
                func=create_cited_work_ids,
                inputs={
                    "works": "analysis.basic_metrics.publications.filtered",
                    "authors": "ad.matched_authors.primary",
                },
                outputs="analysis.complex_metrics.selected_author_papers.raw",
                name="create_oa_author_work_ids",
            ),
            node(
                func=create_list_ids,
                inputs="analysis.complex_metrics.selected_author_papers.raw",
                outputs="analysis.complex_metrics.oa.list",
                name="create_oa_cited_list",
            ),
            node(
                func=fetch_openalex_work_citations,
                inputs={
                    "perpage": "params:complex_metrics.oa.api.perpage",
                    "mails": "params:complex_metrics.oa.api.mails",
                    "ids": "analysis.complex_metrics.oa.list",
                    "filter_criteria": "params:complex_metrics.oa.filter",
                    "parallel_jobs": "params:complex_metrics.oa.n_jobs",
                    "endpoint": "params:complex_metrics.oa.publications_endpoint",
                    "keys_to_include": "params:complex_metrics.oa.keys_to_include",
                },
                outputs="analysis.complex_metrics.publications.raw",
                name="fetch_openalex_work_citations",
            ),
        ],
        tags="cites_collection_pipeline",
    )

    return collection_pipeline

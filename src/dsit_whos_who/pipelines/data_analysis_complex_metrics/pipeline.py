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
from .nodes import (
    sample_cited_work_ids,
    create_list_ids,
    fetch_author_work_citations,
    fetch_reference_works,
)


def create_pipeline(**kwargs) -> Pipeline:  # pylint: disable=W0613
    """Pipeline for data collection.

    Returns:
        Pipeline: The data collection pipeline.
    """
    sample_pipeline = pipeline(
        [
            node(
                func=sample_cited_work_ids,
                inputs={
                    "works": "analysis.basic_metrics.publications.filtered",
                    "authors": "ad.matched_authors.primary",
                },
                outputs="analysis.complex_metrics.publications.sampled",
                name="sample_cited_work_ids",
                tags="collect_sample",
            ),
        ]
    )

    focal_collection_pipeline = pipeline(
        [
            node(
                func=create_list_ids,
                inputs="analysis.complex_metrics.publications.sampled",
                outputs="analysis.complex_metrics.oa.list",
                name="create_oa_cited_list",
                tags="collect_focal",
            ),
            node(
                func=fetch_author_work_citations,
                inputs={
                    "perpage": "params:complex_metrics.oa.api.perpage",
                    "mails": "params:complex_metrics.oa.api.mails",
                    "ids": "analysis.complex_metrics.oa.list",
                    "filter_criteria": "params:complex_metrics.oa.filter",
                    "parallel_jobs": "params:complex_metrics.oa.n_jobs",
                    "endpoint": "params:complex_metrics.oa.publications_endpoint",
                    "select_variables": "params:complex_metrics.oa.select_variables.focal",
                },
                outputs="analysis.complex_metrics.focal_publications.raw",
                name="fetch_openalex_focal_work_citations",
                tags="collect_focal",
            ),
        ],
        tags="cites_collection_pipeline",
    )

    reference_collection_pipeline = pipeline(
        [
            node(
                func=fetch_reference_works,
                inputs="analysis.complex_metrics.publications.sampled",
                outputs="analysis.complex_metrics.reference_works",
                name="fetch_reference_works",
                tags="fetch_reference",
            ),
            node(
                func=create_list_ids,
                inputs="analysis.complex_metrics.reference_works",
                outputs="analysis.complex_metrics.oa.reference_list",
                name="create_oa_reference_list",
            ),
            node(
                func=fetch_author_work_citations,
                inputs={
                    "perpage": "params:complex_metrics.oa.api.perpage",
                    "mails": "params:complex_metrics.oa.api.mails",
                    "ids": "analysis.complex_metrics.oa.reference_list",
                    "filter_criteria": "params:complex_metrics.oa.filter",
                    "parallel_jobs": "params:complex_metrics.oa.n_jobs",
                    "endpoint": "params:complex_metrics.oa.publications_endpoint",
                    "select_variables": "params:complex_metrics.oa.select_variables.reference",
                },
                outputs="analysis.complex_metrics.reference_publications.raw",
                name="fetch_openalex_reference_work_citations",
            ),
        ],
        tags="reference_collection_pipeline",
    )

    return sample_pipeline + focal_collection_pipeline + reference_collection_pipeline

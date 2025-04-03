"""
Pipeline for calculating complex bibliometric metrics for author analysis.

This pipeline implements the calculation of complex metrics including:
1. Disruption indices that measure how papers disrupt vs consolidate research fields
2. Discipline diversity metrics that assess breadth and variety of research portfolios
3. Combined metrics that integrate disruption and diversity measures

The pipeline consists of several stages:
1. Sampling papers from authors' publication records using stratified sampling
2. Collecting citation data from OpenAlex for sampled papers and their references
3. Computing disruption indices based on citation patterns
4. Calculating discipline diversity metrics using CWTS field classifications
5. Combining metrics to create composite measures

Command Line Usage:
    Run full pipeline:
    ```
    kedro run --pipeline data_analysis_complex_metrics
    ```
    
    Run single component:
    ```
    kedro run --pipeline data_analysis_complex_metrics --tags projects
    ```
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    sample_cited_work_ids,
    create_list_ids,
    fetch_author_work_citations,
    refactor_reference_works,
    fetch_author_work_references,
    calculate_disruption_indices,
    compute_subfield_embeddings,
    create_author_aggregates,
    cumulative_author_aggregates,
    calculate_author_diversity,
    compute_complex_metrics,
)


def create_pipeline(**kwargs) -> Pipeline:  # pylint: disable=W0613
    """Create the data analysis pipeline for complex metrics.

    Returns:
        Pipeline: A Kedro pipeline containing nodes for:
            - Sampling papers for analysis
            - Collecting citation data
            - Computing disruption indices
            - Calculating discipline diversity
            - Combining metrics into final measures
    """
    sample_pipeline = pipeline(
        [
            node(
                func=sample_cited_work_ids,
                inputs={
                    "works": "analysis.basic_metrics.publications.intermediate",
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
                    "select_variables": "params:complex_metrics.oa.select_variables",
                    "keys_to_include": "params:complex_metrics.oa.keys_to_include",
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
                func=refactor_reference_works,
                inputs="analysis.complex_metrics.publications.sampled",
                outputs="analysis.complex_metrics.reference_works",
                name="refactor_reference_works",
            ),
            node(
                func=create_list_ids,
                inputs="analysis.complex_metrics.reference_works",
                outputs="analysis.complex_metrics.oa.reference_list",
                name="create_oa_reference_list",
                tags="collect_reference",
            ),
            node(
                func=fetch_author_work_references,
                inputs={
                    "perpage": "params:complex_metrics.oa.api.perpage",
                    "mails": "params:complex_metrics.oa.api.mails",
                    "ids": "analysis.complex_metrics.oa.reference_list",
                    "filter_criteria": "params:complex_metrics.oa.filter",
                    "parallel_jobs": "params:complex_metrics.oa.n_jobs",
                    "endpoint": "params:complex_metrics.oa.publications_endpoint",
                    "select_variables": "params:complex_metrics.oa.select_variables",
                    "keys_to_include": "params:complex_metrics.oa.keys_to_include",
                },
                outputs="analysis.complex_metrics.reference_publications.raw",
                name="fetch_openalex_reference_work_citations",
                tags="collect_reference",
            ),
        ],
        tags="reference_collection_pipeline",
    )

    disruption_index_pipeline = pipeline(
        [
            node(
                func=calculate_disruption_indices,
                inputs={
                    "sample_ids": "analysis.complex_metrics.publications.sampled_ids",
                    "focal_papers": "analysis.basic_metrics.publications.intermediate",
                    "citing_papers_dataset": "analysis.complex_metrics.focal_publications.raw",
                },
                outputs="analysis.complex_metrics.disruption_indices.intermediate",
                name="calculate_disruption_indices",
            ),
        ],
    )

    discipline_diversity_pipeline = pipeline(
        [
            node(
                func=compute_subfield_embeddings,
                inputs={"cwts_data": "cwts.taxonomy"},
                outputs="analysis.complex_metrics.cwts.subfield.distance_matrix",
                name="compute_subfield_embeddings",
            ),
            node(
                func=create_author_aggregates,
                inputs={
                    "authors_data": "analysis.complex_metrics.publications.topics",
                    "authors": "ad.matched_authors.primary",
                    "cwts_data": "analysis.complex_metrics.cwts.subfield.distance_matrix",
                },
                outputs="analysis.complex_metrics.author_topic_aggregates.intermediate",
                name="create_author_topic_aggregates",
            ),
            node(
                func=cumulative_author_aggregates,
                inputs={
                    "author_topics": "analysis.complex_metrics.author_topic_aggregates.intermediate"
                },
                outputs="analysis.complex_metrics.cumul_topic_aggs.intermediate",
                name="create_cumulative_author_aggregates",
            ),
            node(
                func=calculate_author_diversity,
                inputs={
                    "author_frequencies": "analysis.complex_metrics.cumul_topic_aggs.intermediate",
                    "disparity_matrix": "analysis.complex_metrics.cwts.subfield.distance_matrix",
                },
                outputs="analysis.complex_metrics.author_diversity.intermediate",
                name="calculate_author_diversity",
            ),
        ],
        tags="discipline_diversity_pipeline",
    )

    compute_metrics_pipeline = pipeline(
        [
            node(
                func=compute_complex_metrics,
                inputs={
                    "basic_metrics": "analysis.basic_metrics.primary",
                    "publications": "analysis.basic_metrics.publications.intermediate",
                    "disruption_indices": "analysis.complex_metrics.disruption_indices.intermediate",  # pylint: disable=C0301
                    "author_diversity": "analysis.complex_metrics.author_diversity.intermediate",
                },
                outputs="analysis.complex_metrics.primary",
                name="compute_complex_metrics",
            ),
        ]
    )
    return (
        sample_pipeline
        + focal_collection_pipeline
        + reference_collection_pipeline
        + disruption_index_pipeline
        + discipline_diversity_pipeline
        + compute_metrics_pipeline
    )

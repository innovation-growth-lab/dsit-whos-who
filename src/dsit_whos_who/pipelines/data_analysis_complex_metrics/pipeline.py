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
    refactor_reference_works,
    fetch_author_work_references,
    calculate_disruption_indices,
    compute_subfield_embeddings,
    create_author_aggregates,
    cumulative_author_aggregates,
    calculate_author_diversity,
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
                    "focal_papers": "analysis.basic_metrics.publications.filtered",
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
                outputs="analysis.complex_metrics.author_diversity",
                name="calculate_author_diversity",
            ),
        ],
    )

    # author_aggregates_pipeline = pipeline(
    #     [
    #         node(
    #             func=create_author_aggregates,
    #             inputs={
    #                 "authors_data": "authors.oa_dataset.raw",
    #                 "level": f"params:tm.levels.{level}",
    #                 "cwts_data": f"cwts.topics.{level}.distance_matrix",
    #             },
    #             outputs=f"authors.{level}.aggregates.intermediate",
    #             name=f"create_author_aggregates_{level}",
    #         )
    #         for level in ["subfield", "field", "domain"]
    #     ],
    # )

    # author_cumulative_aggregates_pipeline = pipeline(
    #     [
    #         node(
    #             func=cumulative_author_aggregates,
    #             inputs={"author_topics": f"authors.{level}.aggregates.intermediate"},
    #             outputs=f"authors.{level}.cumulative_aggregates.intermediate",
    #             name=f"create_cumulative_author_aggregates_{level}",
    #         )
    #         for level in ["subfield", "field", "domain"]
    #     ],
    # )

    # calculate_diversity_scores_pipeline = pipeline(
    #     [
    #         node(
    #             func=calculate_paper_diversity,
    #             inputs={
    #                 "publications": "oa.publications.gtr.primary",
    #                 "disparity_matrix": f"cwts.topics.{level}.distance_matrix",
    #                 "cwts_data": f"cwts.topics.{level}.distance_matrix",
    #                 "level": f"params:tm.levels.{level}",
    #             },
    #             outputs=f"publications.{level}.paper_diversity_scores.intermediate",
    #             name=f"calculate_paper_diversity_{level}",
    #         )
    #         for level in ["subfield", "field", "domain"]
    #     ]
    #     + [
    #         node(
    #             func=calculate_coauthor_diversity,
    #             inputs={
    #                 "publications": "oa.publications.gtr.primary",
    #                 "author_topics": f"authors.{level}.cumulative_aggregates.intermediate",
    #                 "disparity_matrix": f"cwts.topics.{level}.distance_matrix",
    #             },
    #             outputs=f"publications.{level}.coauthor_diversity_scores.intermediate",
    #             name=f"calculate_coauthor_diversity_{level}",
    #         )
    #         for level in ["subfield", "field", "domain"]
    #     ],
    #     tags="calculate_discipline_diversity_metrics",
    # )

    return (
        sample_pipeline
        + focal_collection_pipeline
        + reference_collection_pipeline
        + disruption_index_pipeline
        + discipline_diversity_pipeline
    )

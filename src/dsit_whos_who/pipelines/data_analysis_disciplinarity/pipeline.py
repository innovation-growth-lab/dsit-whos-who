"""
Command Line Example:
    ```
    kedro run --pipeline data_analysis_disciplinarity
    ```
    Alternatively, you can run this pipeline for a single endpoint:
    ```
    kedro run --pipeline data_analysis_disciplinarity --tags projects
    ```
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import create_list_oa_author_ids, concatenate_openalex_publications
from ..data_collection_oa.nodes import fetch_openalex


def create_pipeline(**kwargs) -> Pipeline:  # pylint: disable=W0613
    """Pipeline for data collection.

    Returns:
        Pipeline: The data collection pipeline.
    """
    # collection_pipeline = pipeline(
    #     [
    #         node(
    #             func=create_list_oa_author_ids,
    #             inputs="ad.matched_authors.primary",
    #             outputs="analysis.disciplinarity.oa.list",
    #             name="create_oa_author_ids",
    #         ),
    #         node(
    #             func=fetch_openalex,
    #             inputs={
    #                 "perpage": "params:disciplinarity.oa.api.perpage",
    #                 "mails": "params:disciplinarity.oa.api.mails",
    #                 "ids": "analysis.disciplinarity.oa.list",
    #                 "filter_criteria": "params:disciplinarity.oa.filter",
    #                 "parallel_jobs": "params:disciplinarity.oa.n_jobs",
    #                 "endpoint": "params:disciplinarity.oa.publications_endpoint",
    #                 "keys_to_include": "params:disciplinarity.oa.keys_to_include",
    #             },
    #             outputs="analysis.disciplinarity.oa.raw",
    #             name="fetch_oa_publications",
    #         ),
    #         node(
    #             func=concatenate_openalex_publications,
    #             inputs={
    #                 "data": "analysis.disciplinarity.oa.raw",
    #                 "gtr_publications": "oa.data_collection.publications.intermediate",
    #             },
    #             outputs="analysis.disciplinarity.oa.intermediate",
    #             name="concatenate_openalex_publications",
    #         ),
    #     ]
    # )

    return pipeline([])

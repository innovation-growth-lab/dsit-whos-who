"""
Data Analysis Basic Metrics Pipeline

This module defines the pipeline for computing basic metrics from OpenAlex and GTR data.
The pipeline processes author metadata, publication data, and project information to
generate comprehensive metrics about researchers' academic impact and collaboration patterns.

Command Line Examples:
    Run the complete pipeline:
    ```
    kedro run --pipeline data_analysis_basic_metrics
    ```

    Run a specific pipeline segment:
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
    process_matched_author_works,
    compute_basic_metrics,
)


def create_pipeline(**kwargs) -> Pipeline:  # pylint: disable=W0613
    """
    Create the data analysis basic metrics pipeline.

    This function constructs a Kedro pipeline that processes researcher data through
    three main stages:
    1. Collection Pipeline:
       - Creates lists of OpenAlex author IDs
       - Fetches publication works data for matched authors

    2. Processing Pipeline:
       - Processes author metadata from OpenAlex
       - Processes person data from GTR
       - Analyses publication data for collaboration patterns

    3. Metrics Computation:
       - Combines processed data to compute comprehensive metrics
       - Generates final analysis outputs

    Returns:
        Pipeline: A complete Kedro pipeline that processes author data and computes
            basic metrics for researcher analysis.
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
                outputs="analysis.basic_metrics.publications.intermediate",
                name="fetch_openalex_matched_author_works",
            ),
        ],
        tags="collection_pipeline_basic_metrics",
    )

    processing_pipeline = pipeline(
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
            node(
                func=process_matched_author_works,
                inputs={
                    "publications": "analysis.basic_metrics.publications.intermediate",
                    "matched_authors": "ad.matched_authors.primary",
                    "n_jobs": "params:basic_metrics.oa.n_jobs",
                    "batch_size": "params:basic_metrics.oa.batch_size",
                },
                outputs="analysis.basic_metrics.processed_publications.intermediate",
                name="process_matched_author_works",
            ),
        ]
    )

    compute_metrics_pipeline = pipeline(
        [
            node(
                func=compute_basic_metrics,
                inputs={
                    "author_data": "analysis.basic_metrics.author_metadata.intermediate",
                    "person_data": "analysis.basic_metrics.gtr_data.intermediate",
                    "publications": "analysis.basic_metrics.processed_publications.intermediate",
                },
                outputs="analysis.basic_metrics.primary",
                name="compute_basic_metrics",
            ),
        ]
    )
    return collection_pipeline + processing_pipeline + compute_metrics_pipeline

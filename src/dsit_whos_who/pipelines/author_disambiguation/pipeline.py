"""Pipeline for author disambiguation."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    aggregate_author_information,
    create_feature_matrix,
    train_disambiguation_model,
    predict_author_matches,
    evaluate_model_performance,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the author disambiguation pipeline.

    Returns:
        A Pipeline object containing all the author disambiguation nodes.
    """
    return pipeline(
        [
            node(
                aggregate_author_information,
                inputs={
                    "gtr_persons": "gtr.data_collection.persons.intermediate",
                    "gtr_projects": "gtr.data_collection.projects.intermediate",
                    "gtr_organisations": "gtr.data_collection.organisations.intermediate",
                    "gtr_project_topics": "projects.cwts.topics",
                    "gtr_publications": "gtr.data_collection.publications.intermediate",
                    "cwts_taxonomy": "cwts.taxonomy",
                    "oa_publications": "oa.data_collection.publications.intermediate",
                },
                outputs="author_disambiguation.aggregated_authors",
                name="aggregate_author_information",
            ),
            # node(
            #     func=create_feature_matrix,
            #     inputs={
            #         "gtr_authors": "gtr.data_collection.persons.intermediate",
            #         "oa_candidates": "oa.data_collection.author_search.intermediate",
            #     },
            #     outputs="author_disambiguation.features",
            #     name="create_feature_matrix",
            # ),
            # node(
            #     func=train_disambiguation_model,
            #     inputs={
            #         "feature_matrix": "author_disambiguation.features",
            #         "orcid_labels": "author_disambiguation.orcid_labels",
            #     },
            #     outputs="author_disambiguation.model",
            #     name="train_model",
            # ),
            # node(
            #     func=predict_author_matches,
            #     inputs={
            #         "model": "author_disambiguation.model",
            #         "feature_matrix": "author_disambiguation.features",
            #     },
            #     outputs="author_disambiguation.predictions",
            #     name="predict_matches",
            # ),
            # node(
            #     func=evaluate_model_performance,
            #     inputs={
            #         "predictions": "author_disambiguation.predictions",
            #         "ground_truth": "author_disambiguation.ground_truth",
            #     },
            #     outputs="author_disambiguation.metrics",
            #     name="evaluate_performance",
            # ),
        ]
    )

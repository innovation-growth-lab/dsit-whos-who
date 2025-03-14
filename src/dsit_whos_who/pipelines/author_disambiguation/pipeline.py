"""Pipeline for author disambiguation."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import aggregate_person_information, preprocess_oa_candidates


def create_pipeline(**kwargs) -> Pipeline:  # pylint: disable=W0613
    """Create the author disambiguation pipeline.

    Returns:
        A Pipeline object containing all the author disambiguation nodes.
    """
    return pipeline(
        [
            node(
                aggregate_person_information,
                inputs={
                    "gtr_persons": "gtr.data_collection.persons.intermediate",
                    "gtr_projects": "gtr.data_collection.projects.intermediate",
                    "gtr_organisations": "gtr.data_collection.organisations.intermediate",
                    "gtr_project_topics": "projects.cwts.topics",
                    "gtr_publications": "gtr.data_collection.publications.intermediate",
                    "cwts_taxonomy": "cwts.taxonomy",
                    "oa_publications": "oa.data_collection.publications.intermediate",
                },
                outputs="author_disambiguation.aggregated_persons.intermediate",
                name="aggregate_person_information",
            ),
            node(
                preprocess_oa_candidates,
                inputs={
                    "oa_candidates": "oa.data_collection.author_search.raw",
                    "institutions": "oa.data_collection.institutions.intermediate",
                },
                outputs="author_disambiguation.preprocessed_oa_candidates.intermediate",
                name="preprocess_oa_candidates",
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

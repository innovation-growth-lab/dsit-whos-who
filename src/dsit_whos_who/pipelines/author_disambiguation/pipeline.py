"""Pipeline for author disambiguation."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    aggregate_person_information,
    preprocess_oa_candidates,
    merge_candidates_with_gtr,
    create_feature_matrix,
)


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
                outputs="ad.aggregated_persons.intermediate",
                name="aggregate_person_information",
            ),
            node(
                preprocess_oa_candidates,
                inputs={
                    "oa_candidates": "oa.data_collection.author_search.raw",
                    "institutions": "oa.data_collection.institutions.intermediate",
                },
                outputs="ad.preprocessed_oa_candidates.raw.ptd",
                name="preprocess_oa_candidates",
            ),
            node(
                merge_candidates_with_gtr,
                inputs={
                    "gtr_persons": "ad.aggregated_persons.intermediate",
                    "oa_candidates": "ad.preprocessed_oa_candidates.raw.ptd",
                    "orcid_match": "params:global.true",
                },
                outputs="ad.orcid_labelled_persons.raw.ptd",
                name="merge_and_filter_by_orcid",
            ),
            node(
                func=create_feature_matrix,
                inputs={
                    "input_data": "ad.orcid_labelled_persons.raw.ptd"
                },
                outputs="ad.feature_matrix.intermediate",
                name="create_feature_matrix",
            ),
            # node(
            #     func=train_disambiguation_model,
            #     inputs={
            #         "feature_matrix": "ad.features",
            #         "orcid_labels": "ad.orcid_labels",
            #     },
            #     outputs="ad.model",
            #     name="train_model",
            # ),
            # node(
            #     func=predict_author_matches,
            #     inputs={
            #         "model": "ad.model",
            #         "feature_matrix": "ad.features",
            #     },
            #     outputs="ad.predictions",
            #     name="predict_matches",
            # ),
            # node(
            #     func=evaluate_model_performance,
            #     inputs={
            #         "predictions": "ad.predictions",
            #         "ground_truth": "ad.ground_truth",
            #     },
            #     outputs="ad.metrics",
            #     name="evaluate_performance",
            # ),
        ]
    )

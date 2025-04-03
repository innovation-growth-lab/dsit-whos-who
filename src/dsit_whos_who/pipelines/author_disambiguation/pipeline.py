"""
Author disambiguation pipeline orchestration.

This module defines the structure and flow of the author disambiguation process:

Core Pipelines:
- Preprocessing: Data aggregation and standardisation
- Model Training: Feature engineering and model optimisation
- Model Validation: Performance analysis and quality checks
- Prediction: Author matching and result validation
- Coverage Analysis: Match rate and data quality assessment

The pipeline is designed for modular execution, allowing selective running of
specific stages using tags.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    aggregate_person_information,
    preprocess_oa_candidates,
    merge_candidates_with_gtr,
    create_feature_matrix,
    train_disambiguation_model,
    check_model_performance,
    predict_author_matches,
    get_matched_authors,
    check_prediction_coverage,
)


def create_pipeline(**kwargs) -> Pipeline:  # pylint: disable=W0613
    """Create modular author disambiguation pipeline.

    Constructs pipeline with five main stages:
    1. Data Preprocessing
       - Author information aggregation
       - OpenAlex candidate processing

    2. Model Training
       - ORCID-based pair creation
       - Feature engineering
       - Model training and optimisation

    3. Model Validation
       - Performance metrics analysis
       - Feature importance evaluation
       - Error pattern analysis

    4. Prediction
       - Non-ORCID pair processing
       - Match prediction and filtering
       - Result aggregation

    5. Coverage Analysis
       - Match rate assessment
       - Data quality validation
       - Coverage statistics

    Returns:
        Complete disambiguation pipeline with tagged components
    """
    preprocess_pipeline = pipeline(
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
        ]
    )

    model_training_pipeline = pipeline(
        [
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
                inputs={"input_data": "ad.orcid_labelled_persons.raw.ptd"},
                outputs="ad.orcid_labelled_feature_matrix.intermediate",
                name="create_orcid_labelled_feature_matrix",
            ),
            node(
                func=train_disambiguation_model,
                inputs={
                    "feature_matrix": "ad.orcid_labelled_feature_matrix.intermediate",
                    "model_training": "params:model_training",
                },
                outputs="ad.model.training",
                name="train_model",
            ),
        ],
        tags="model_training",
    )

    model_checks_pipeline = pipeline(
        [
            node(
                func=check_model_performance,
                inputs={
                    "feature_matrix": "ad.orcid_labelled_feature_matrix.intermediate",
                    "model_versions": "ad.model.training.ptd",
                    "params": "params:model_training",
                    "lite": "params:global.true",
                },
                outputs=None,
                name="check_model_performance",
            ),
        ]
    )

    prediction_pipeline = pipeline(
        [
            node(
                merge_candidates_with_gtr,
                inputs={
                    "gtr_persons": "ad.aggregated_persons.intermediate",
                    "oa_candidates": "ad.preprocessed_oa_candidates.raw.ptd",
                    "orcid_match": "params:global.false",
                },
                outputs="ad.non_orcid_labelled_persons.raw.ptd",
                name="merge_candidates_with_gtr",
            ),
            node(
                func=create_feature_matrix,
                inputs={"input_data": "ad.non_orcid_labelled_persons.raw.ptd"},
                outputs="ad.non_orcid_labelled_feature_matrix.intermediate",
                name="create_non_orcid_labelled_feature_matrix",
            ),
            node(
                func=predict_author_matches,
                inputs={
                    "model_dict": "ad.model.choice",
                    "feature_matrix": "ad.non_orcid_labelled_feature_matrix.intermediate",
                    "params": "params:model_prediction",
                },
                outputs="ad.predictions.intermediate",
                name="predict_author_matches",
            ),
            node(
                func=get_matched_authors,
                inputs={
                    "predictions": "ad.predictions.intermediate",
                    "merged_candidates": "ad.preprocessed_oa_candidates.raw.ptd",
                },
                outputs="ad.matched_authors.primary",
                name="get_matched_authors",
            ),
        ],
        tags="model_prediction",
    )

    prediction_checks_pipeline = pipeline(
        [
            node(
                func=check_prediction_coverage,
                inputs={
                    "matched_authors": "ad.matched_authors.primary",
                    "gtr_persons": "gtr.data_collection.persons.intermediate",
                    "gtr_projects": "gtr.data_collection.projects.intermediate",
                    "feature_matrix": "ad.non_orcid_labelled_feature_matrix.intermediate",
                },
                outputs="ad.coverage_analysis.tmp",
                name="check_prediction_coverage",
            ),
        ]
    )

    return (
        preprocess_pipeline
        + model_training_pipeline
        + model_checks_pipeline
        + prediction_pipeline
        + prediction_checks_pipeline
    )

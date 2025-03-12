"""Nodes for author disambiguation pipeline."""

import logging
from typing import Dict
import pandas as pd
from kedro.io import AbstractDataset
from .utils.preprocessing.gtr import (
    preprocess_gtr_persons,
    preprocess_gtr_projects,
    preprocess_gtr_topics,
    preprocess_gtr_publications,
    map_project_info,
)

logger = logging.getLogger(__name__)


def aggregate_author_information(
    gtr_persons: pd.DataFrame,
    gtr_projects: pd.DataFrame,
    gtr_project_topics: pd.DataFrame,
    gtr_organisations: pd.DataFrame,
    gtr_publications: pd.DataFrame,
    cwts_taxonomy: pd.DataFrame,
    oa_publications: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate comprehensive author information from multiple data sources.

    Creates a consolidated view of each author's activities by combining:
    - Personal information (name, ORCID, organization)
    - Project participation
    - Research topics
    - Publication records
    - Institutional affiliations

    Args:
        gtr_persons: Author information from Gateway to Research
        gtr_projects: Project data including dates and basic metadata
        gtr_project_topics: Topic classifications for projects
        gtr_organisations: Organization reference data
        gtr_publications: Publication records from GtR
        cwts_taxonomy: CWTS topic classification taxonomy
        oa_publications: Publication records from OpenAlex

    Returns:
        DataFrame with aggregated author information, indexed by person_id, containing:
        - Personal identifiers (name)
        - Organizational affiliations
        - List of projects with dates
        - Aggregated publications
        - Consolidated research topics
        - Associated OpenAlex publication IDs
    """
    # preprocess input data
    gtr_persons = preprocess_gtr_persons(gtr_persons)
    projects_dict = preprocess_gtr_projects(gtr_projects)
    topics_dict = preprocess_gtr_topics(gtr_project_topics, cwts_taxonomy)
    publications_dict = preprocess_gtr_publications(gtr_publications, oa_publications)

    # # filter for persons with ORCID matches and explode projects
    # gtr_labelled_candidates = (
    #     gtr_persons[gtr_persons["orcid_gtr"].isin(orcid_authors["orcid"])]
    #     .copy()
    #     .explode("projects")
    # )

    # map project publications data
    gtr_persons["project_authors"] = gtr_persons["projects"].map(
        lambda x: publications_dict[x]["authors"] if x in publications_dict else []
    )
    gtr_persons["project_oa_ids"] = gtr_persons["projects"].map(
        lambda x: publications_dict[x]["id"] if x in publications_dict else []
    )

    # map project info to each person-project combination
    project_info = [
        {
            "person_id": row["person_id"],
            "first_name": row["first_name"],
            "surname": row["surname"],
            "orcid_gtr": row["orcid_gtr"],
            "organisation": row["organisation"],
            "project_id": info["project_id"],
            "project_start": info["start_date"],
            "project_end": info["end_date"],
            "project_publications": info["publications"],
            "project_topics": info["topics"],
            "project_authors": row["project_authors"],
            "project_oa_ids": row["project_oa_ids"],
        }
        for _, row in gtr_persons.iterrows()
        if (info := map_project_info(row["projects"], projects_dict, topics_dict))
    ]

    # convert to DataFrame and aggregate by person
    result = pd.DataFrame(project_info)
    agg_dict = {
        col: "first"
        for col in ["person_id", "orcid_gtr", "first_name", "surname", "organisation"]
    }
    agg_dict.update(
        {
            col: list
            for col in [
                "project_id",
                "project_publications",
                "project_topics",
                "project_authors",
                "project_oa_ids",
            ]
        }
    )

    agg_result = result.groupby("person_id").agg(agg_dict)

    # Flatten nested lists
    agg_result["project_publications"] = agg_result["project_publications"].apply(
        lambda x: list(set().union(*x))
    )
    agg_result["project_topics"] = agg_result["project_topics"].apply(
        lambda x: [item for sublist in x for item in sublist]
    )

    # add organisation name
    organisation_names = gtr_organisations.set_index("id")["name"].to_dict()
    agg_result["organisation_name"] = agg_result["organisation"].map(organisation_names)

    return agg_result


def create_feature_matrix(
    gtr_authors: pd.DataFrame, oa_candidates: pd.DataFrame
) -> pd.DataFrame:
    """Create feature matrix for author pairs.

    Args:
        gtr_authors: DataFrame containing GtR author information
        oa_candidates: DataFrame containing OpenAlex author candidates

    Returns:
        DataFrame containing computed features for each author pair
    """
    logger.info("Computing features for %d author pairs", len(gtr_authors))

    return gtr_authors


def train_disambiguation_model(
    feature_matrix: pd.DataFrame, orcid_labels: pd.DataFrame
) -> Dict:
    """Train the disambiguation model using ORCID matches as ground truth.

    Args:
        feature_matrix: DataFrame containing computed features
        orcid_labels: DataFrame containing known ORCID matches

    Returns:
        Trained model and associated metadata
    """
    logger.info("Training disambiguation model")

    return feature_matrix


def predict_author_matches(model: Dict, feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """Predict matches for all author pairs.

    Args:
        model: Trained model and metadata
        feature_matrix: DataFrame containing features for prediction

    Returns:
        DataFrame containing predicted matches and confidence scores
    """
    logger.info("Predicting matches for %d pairs", len(feature_matrix))

    return model


def evaluate_model_performance(
    predictions: pd.DataFrame, ground_truth: pd.DataFrame
) -> Dict:
    """Evaluate model performance using held-out data.

    Args:
        predictions: DataFrame containing predicted matches
        ground_truth: DataFrame containing known correct matches

    Returns:
        Dictionary containing evaluation metrics
    """
    logger.info("Evaluating model performance")

    return predictions

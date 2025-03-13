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
    flatten_and_aggregate_authors,
)
from .utils.preprocessing.oa import process_affiliations, get_associated_institutions

logger = logging.getLogger(__name__)


def aggregate_person_information(
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
    - Personal information (name, ORCID, organisation)
    - Project participation
    - Research topics
    - Publication records
    - Institutional affiliations

    Args:
        gtr_persons: Author information from Gateway to Research
        gtr_projects: Project data including dates and basic metadata
        gtr_project_topics: Topic classifications for projects
        gtr_organisations: Organisation reference data
        gtr_publications: Publication records from GtR
        cwts_taxonomy: CWTS topic classification taxonomy
        oa_publications: Publication records from OpenAlex

    Returns:
        DataFrame with aggregated author information, indexed by person_id, containing:
        - Personal identifiers (name)
        - Organisational affiliations
        - List of projects with dates
        - Aggregated publications
        - Consolidated research topics
        - Associated OpenAlex publication IDs
    """
    logger.info(
        "Starting person information aggregation for %d persons", len(gtr_persons)
    )

    # Preprocess input data
    logger.info("Preprocessing GtR persons data")
    gtr_persons = preprocess_gtr_persons(gtr_persons)

    logger.info("Preprocessing GtR projects data")
    projects_dict = preprocess_gtr_projects(gtr_projects)
    logger.info("Processed %d projects", len(projects_dict))

    logger.info("Preprocessing GtR topics data")
    topics_dict = preprocess_gtr_topics(gtr_project_topics, cwts_taxonomy)
    logger.info("Processed topics for %d projects", len(topics_dict))

    logger.info("Preprocessing GtR publications data")
    publications_dict = preprocess_gtr_publications(gtr_publications, oa_publications)
    logger.info("Processed publications for %d projects", len(publications_dict))

    # project-wise data
    gtr_persons_projects = gtr_persons.explode("projects")

    # Map project publications data
    logger.info("Mapping project publications to persons")
    gtr_persons_projects["project_authors"] = gtr_persons_projects["projects"].map(
        lambda x: publications_dict[x]["authors"] if x in publications_dict else []
    )
    gtr_persons_projects["project_oa_ids"] = gtr_persons_projects["projects"].map(
        lambda x: publications_dict[x]["id"] if x in publications_dict else []
    )

    # Map project info to each person-project combination
    logger.info("Creating detailed project information for each person")
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
        for _, row in gtr_persons_projects.iterrows()
        if (info := map_project_info(row["projects"], projects_dict, topics_dict))
    ]

    # Convert to DataFrame and aggregate by person
    logger.info("Converting to DataFrame and aggregating by person")
    result = pd.DataFrame(project_info)
    logger.info(
        "Created initial DataFrame with %d person-project combinations", len(result)
    )

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
    logger.info("Aggregated data for %d unique persons", len(agg_result))

    # Flatten nested lists
    logger.info("Flattening nested publication and topic lists")
    agg_result["project_publications"] = agg_result["project_publications"].apply(
        lambda x: list(set().union(*x))
    )
    agg_result["project_topics"] = agg_result["project_topics"].apply(
        lambda x: [item for sublist in x for item in sublist]
    )

    # flatten authors
    agg_result["project_authors"] = agg_result["project_authors"].apply(
        flatten_and_aggregate_authors
    )

    # Add organisation name
    logger.info("Adding organisation names")
    organisation_names = gtr_organisations.set_index("id")["name"].to_dict()
    agg_result["organisation_name"] = agg_result["organisation"].map(organisation_names)

    # Log summary statistics
    logger.info(
        "Aggregation complete. Summary statistics:"
        "\n- Total persons: %d"
        "\n- Average projects per person: %.2f"
        "\n- Average publications per person: %.2f"
        "\n- Persons with ORCID: %d",
        len(agg_result),
        agg_result["project_id"].apply(len).mean(),
        agg_result["project_publications"].apply(len).mean(),
        agg_result["orcid_gtr"].notna().sum(),
    )

    return agg_result


def preprocess_oa_candidates(
    oa_candidates: AbstractDataset, institutions: pd.DataFrame
) -> pd.DataFrame:
    """Preprocess OpenAlex author candidates.

    Args:
        oa_candidates: Dataset containing OpenAlex author candidates
        institutions: DataFrame with institution IDs and their associated institutions

    Returns:
        DataFrame containing all OpenAlex author information plus:
        - List of institution names
        - List of associated institution names
        - GB affiliation indicators and proportions
    """
    logger.info("Starting preprocessing of OpenAlex candidates")

    institutions_dict = institutions.set_index("id")[
        "associated_institutions"
    ].to_dict()

    processed_batches = []
    for key, batch_loader in oa_candidates.items():
        logger.info("Processing batch %s", key)
        candidates = batch_loader()

        # flatten and convert to dataframe
        candidates = [item for sublist in candidates for item in sublist]
        candidate_batch_df = pd.DataFrame(candidates)

        # process affiliations
        affiliations_processed = candidate_batch_df["affiliations"].apply(
            process_affiliations
        )

        # add new columns while preserving original data
        candidate_batch_df["institution_names"] = affiliations_processed.apply(
            lambda x: x[0]
        )
        inst_ids = affiliations_processed.apply(lambda x: x[1])
        candidate_batch_df["has_gb_affiliation"] = affiliations_processed.apply(
            lambda x: x[2]
        )
        candidate_batch_df["gb_affiliation_proportion"] = affiliations_processed.apply(
            lambda x: x[3]
        )

        # process associated institutions
        associated_processed = inst_ids.apply(
            lambda x: get_associated_institutions(x, institutions_dict)
        )
        candidate_batch_df["associated_institution_names"] = associated_processed.apply(
            lambda x: x[0]
        )
        candidate_batch_df["has_gb_associated"] = associated_processed.apply(
            lambda x: x[1]
        )

        # drop the columns we used to create our features
        candidate_batch_df = candidate_batch_df.drop(
            columns=["affiliations", "last_known_institutions"], errors="ignore"
        )

        processed_batches.append(candidate_batch_df)
        logger.info("Processed batch %s", key)

    # Combine all processed batches
    result = pd.concat(processed_batches, ignore_index=True)

    return result


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

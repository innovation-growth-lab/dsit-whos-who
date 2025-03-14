"""Nodes for author disambiguation pipeline."""

import logging
from typing import Dict, Iterator
import pandas as pd
import numpy as np
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
from .utils.feature_engineering.compute_features import compute_all_features

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
) -> Iterator[pd.DataFrame]:
    """Preprocess OpenAlex author candidates.

    Args:
        oa_candidates: Dataset containing OpenAlex author candidates
        institutions: DataFrame with institution IDs and their associated institutions

    Returns:
        Iterator yielding DataFrames for each batch of OpenAlex author information plus:
        - List of institution names
        - List of associated institution names
        - GB affiliation indicators and proportions
    """
    logger.info("Starting preprocessing of OpenAlex candidates")

    institutions_dict = institutions.set_index("id")[
        "associated_institutions"
    ].to_dict()

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
            columns=["affiliations", "last_known_institutions", "counts_by_year"],
            errors="ignore",
        )

        yield {key: candidate_batch_df}


def clean_name(name: str) -> str:
    """Clean name by removing special characters."""
    return name.translate(str.maketrans("", "", ",.;:"))


def merge_candidates_with_gtr(
    gtr_persons: pd.DataFrame, oa_candidates: dict, orcid_match: bool = True
) -> Iterator[pd.DataFrame]:
    """Merge and filter by ORCID.

    Args:
        gtr_persons: DataFrame containing GtR person information
        oa_candidates: Dictionary of loader functions for OpenAlex candidate data

    Returns:
        Iterator yielding DataFrames of merged and filtered results
    """
    if orcid_match:
        logger.info("Merging and filtering by ORCID")
        gtr_filtered = gtr_persons[gtr_persons["orcid_gtr"].notna()].copy()
        gtr_filtered["full_name"] = (
            gtr_filtered["first_name"] + " " + gtr_filtered["surname"]
        ).apply(clean_name)

    for key, batch_loader in oa_candidates.items():
        logger.info("Processing batch %s", key)
        candidates_batch = batch_loader()

        # Clean candidate names
        candidates_batch["gtr_author_name"] = candidates_batch["gtr_author_name"].apply(
            clean_name
        )

        # sort by works_count, cited_by_count
        candidates_batch = candidates_batch.sort_values(
            by=["works_count", "cited_by_count"], ascending=False
        )

        # get first institution_name
        candidates_batch["first_institution_name"] = candidates_batch[
            "institution_names"
        ].apply(lambda x: x[0] if isinstance(x, np.ndarray) and len(x) > 0 else None)

        # drop duplicates
        candidates_batch = candidates_batch.drop_duplicates(
            subset=["display_name", "gtr_author_name", "orcid"]
        )

        # merge on full name
        batch_merged = gtr_filtered.merge(
            candidates_batch,
            left_on="full_name",
            right_on="gtr_author_name",
            how="inner",
        )

        n_matched_authors = len(batch_merged["gtr_author_name"].unique())
        logger.info(
            "Found %d unique GTR authors with candidate matches in this batch",
            n_matched_authors,
        )

        if orcid_match:
            # create binary labels based on ORCID match
            batch_merged["is_match"] = (
                batch_merged["orcid_gtr"] == batch_merged["orcid"]
            ).astype(int)

            # drop duplicates prioritising is_match=1
            batch_merged = batch_merged.sort_values(
                "is_match", ascending=False
            ).drop_duplicates(
                subset=["display_name", "gtr_author_name", "first_institution_name"]
            )

            # print value counts of is_match grouped by author
            logger.info("Value counts of matches by GTR author:")
            match_counts = batch_merged.groupby("gtr_author_name")["is_match"].sum()
            logger.info("\n%s", match_counts.value_counts())

            # [NOTE] we did lose some 2k when search orcids. Now, roughly 1/5 researchers don't have
            # ORCIDs that match (ie. the name matches multiple times, but not a single candidate
            # has the same ORCID as the GTR author - one is outdated, or wrong)

            # [NOTE]2: we also have authors who are of course duplicates on OA, with often
            # a different name

            # keep only authors with at least one match when grouped by gtr_author_name
            batch_merged = batch_merged.groupby("gtr_author_name").filter(
                lambda x: x["is_match"].sum() > 0
            )

        yield {key: batch_merged}


def create_feature_matrix(
    input_data: AbstractDataset,
) -> Iterator[pd.DataFrame]:
    """Create feature matrix for author pairs.

    Args:
        input_data: Partitioned dataset containing matched GTR-OA pairs

    Returns:
        Iterator yielding DataFrames with computed features for each batch
    """
    aggregated_features = []
    logger.info("Starting feature computation")
    for key, batch_loader in input_data.items():
        logger.info("Processing batch %s", key)
        batch_df = batch_loader()

        # compute features for the batch
        features = compute_all_features(batch_df)

        aggregated_features.append(features)

    return pd.concat(aggregated_features)


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

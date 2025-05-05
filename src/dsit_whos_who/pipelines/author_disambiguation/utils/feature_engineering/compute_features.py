"""
Feature computation orchestration for author disambiguation.
"""

# pylint: disable=E0402

import pandas as pd
from tqdm import tqdm
from .name_features import compute_name_features
from .topic_features import compute_topic_features
from .institution_features import compute_institution_features
from .publication_features import compute_publication_features
from .metadata_features import compute_metadata_features


def compute_all_features(batch_df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features for a batch of author pairs.

    Args:
        batch_df: DataFrame containing matched GTR-OA pairs

    Returns:
        DataFrame with computed features
    """
    features = []
    for _, row in tqdm(
        batch_df.iterrows(), total=len(batch_df), desc="Computing features for pairs"
    ):
        pair_features = {}

        # compute name features
        name_feats = compute_name_features(
            row["gtr_author_name"], # author (from GtR, via the OA search)
            row["display_name"], # author (from OpenAlex)
            row.get("display_name_alternatives", []), # author (from OpenAlex)
        )

        # compute topic features
        topic_feats = compute_topic_features(
            row["project_topics"], # person (from GtR)
            row["topics"], # author (from OpenAlex)
            len(row["project_topics"]),
            row["works_count"], # author (from OpenAlex)
        )

        # compute institution features
        inst_feats = compute_institution_features(
            row["gb_affiliation_proportion"], # author (from OpenAlex)
            row["has_gb_affiliation"], # author (from OpenAlex)
            row["has_gb_associated"], # author (from OpenAlex)
            row["organisation_name"], # organisation (from GtR)
            row["institution_names"], # institution (from OpenAlex)
            row["associated_institution_names"], # institution (from OpenAlex)
        )

        # compute publication coverage features (of candidate author)
        pub_feats = compute_publication_features(
            row["id"], # author (from OpenAlex)
            row["project_publications"], # publication (from GtR)
            row["project_authors"], # publication (from GtR)
        )

        # compute metadata features
        meta_feats = compute_metadata_features(
            row["works_count"], # author (from OpenAlex)
            row["cited_by_count"], # author (from OpenAlex)
            row["h_index"], # author (from OpenAlex)
            row["i10_index"], # author (from OpenAlex)
        )

        # add id features
        pair_features["gtr_id"] = row["person_id"]
        pair_features["oa_id"] = row["id"]
        if "is_match" in row.index:
            pair_features["is_match"] = row["is_match"]

        # Combine all features
        pair_features.update(name_feats)
        pair_features.update(topic_feats)
        pair_features.update(inst_feats)
        pair_features.update(pub_feats)
        pair_features.update(meta_feats)

        features.append(pair_features)

    return pd.DataFrame(features)

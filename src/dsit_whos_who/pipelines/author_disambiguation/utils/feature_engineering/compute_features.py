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
            row["gtr_author_name"],
            row["display_name"],
            row.get("display_name_alternatives", []),
        )

        # compute topic features
        topic_feats = compute_topic_features(
            row["project_topics"],
            row["topics"],
            len(row["project_topics"]),
            row["works_count"],
        )

        # compute institution features
        inst_feats = compute_institution_features(
            row["gb_affiliation_proportion"],
            row["has_gb_affiliation"],
            row["has_gb_associated"],
            row["organisation_name"],
            row["institution_names"],
            row["associated_institution_names"],
        )

        # compute publication features
        pub_feats = compute_publication_features(
            row["id"], row["project_publications"], row["project_authors"]
        )

        # compute metadata features
        meta_feats = compute_metadata_features(
            row["works_count"], row["cited_by_count"], row["h_index"], row["i10_index"]
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

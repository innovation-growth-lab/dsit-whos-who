"""Topic similarity features."""

from typing import List, Dict
import numpy as np
from scipy.spatial.distance import jensenshannon
from collections import Counter


def compute_topic_features(
    gtr_topics: List[np.ndarray],
    oa_topics: List[np.ndarray],
    gtr_project_count: int,
    oa_work_count: int,
) -> Dict[str, float]:
    """Compute topic similarity features between GTR and OpenAlex topics.

    Args:
        gtr_topics: List of GTR topic arrays
        oa_topics: List of OpenAlex topic arrays
        gtr_project_count: Number of GTR projects (for normalisation)
        oa_work_count: Number of OA works (for normalisation)

    Returns:
        Dictionary of topic similarity features at each taxonomic level
    """
    features = {}

    # Group topics by level
    gtr_by_level = _group_gtr_topics_by_level(gtr_topics)
    oa_by_level = _group_oa_topics_by_level(oa_topics)

    for level in ["domain", "field", "subfield", "topic"]:
        gtr_dist = gtr_by_level[level]
        oa_dist = oa_by_level[level]

        if not gtr_dist or not oa_dist:
            features.update(
                {
                    f"{level}_jaccard": 0.0,
                    f"{level}_cosine": 0.0,
                    f"{level}_js_div": 1.0,
                    f"{level}_containment": 0.0,
                }
            )
            continue

        # Get all unique IDs at this level
        all_ids = sorted(set(gtr_dist) | set(oa_dist))

        # Create normalised count vectors
        gtr_vec = np.array([gtr_dist[id_] for id_ in all_ids], dtype=float)
        oa_vec = np.array([oa_dist[id_] for id_ in all_ids], dtype=float)

        # Normalise by project/work counts
        gtr_norm = gtr_vec / gtr_project_count if gtr_project_count > 0 else gtr_vec
        oa_norm = oa_vec / oa_work_count if oa_work_count > 0 else oa_vec

        # Compute metrics
        features[f"{level}_jaccard"] = len(set(gtr_dist) & set(oa_dist)) / len(
            set(gtr_dist) | set(oa_dist)
        )
        features[f"{level}_cosine"] = float(
            np.dot(gtr_norm, oa_norm)
            / (np.linalg.norm(gtr_norm) * np.linalg.norm(oa_norm))
            if np.any(gtr_norm) and np.any(oa_norm)
            else 0.0
        )
        features[f"{level}_js_div"] = float(
            jensenshannon(gtr_norm, oa_norm)
            if np.any(gtr_norm) and np.any(oa_norm)
            else 1.0
        )
        features[f"{level}_containment"] = (
            len(set(gtr_dist) & set(oa_dist)) / len(set(gtr_dist)) if gtr_dist else 0.0
        )

    return features


def _group_gtr_topics_by_level(topics: List[np.ndarray]) -> Dict[str, Counter]:
    """Group GTR topics by taxonomic level with counts.

    Args:
        topics: List of numpy arrays, each containing:
            [topic_id, topic_name, subfield_id, subfield_name,
             field_id, field_name, domain_id, domain_name]
    """
    result = {
        "domain": Counter(),
        "field": Counter(),
        "subfield": Counter(),
        "topic": Counter(),
    }

    for topic_array in topics:
        if not isinstance(topic_array, np.ndarray) or len(topic_array) != 8:
            continue

        # Extract IDs at each level, counting duplicates
        if topic_array[0]:  # Topic level
            result["topic"][topic_array[0]] += 1
        if topic_array[2]:  # Subfield level
            result["subfield"][topic_array[2]] += 1
        if topic_array[4]:  # Field level
            result["field"][topic_array[4]] += 1
        if topic_array[6]:  # Domain level
            result["domain"][topic_array[6]] += 1

    return result


def _group_oa_topics_by_level(topics: List[np.ndarray]) -> Dict[str, Counter]:
    """Group OpenAlex topics by taxonomic level with counts.

    Args:
        topics: List of numpy arrays, each containing:
            [topic_id, topic_name, count, subfield_id, subfield_name,
             field_id, field_name, domain_id, domain_name]
    """
    result = {
        "domain": Counter(),
        "field": Counter(),
        "subfield": Counter(),
        "topic": Counter(),
    }

    for topic_array in topics:
        if not isinstance(topic_array, np.ndarray) or len(topic_array) != 9:
            continue

        try:
            count = int(topic_array[2])
            # Clean IDs by removing prefixes
            topic_id = topic_array[0]
            subfield_id = topic_array[3].replace("subfields/", "")
            field_id = topic_array[5].replace("fields/", "")
            domain_id = topic_array[7].replace("domains/", "")

            # Add counts at each level
            result["topic"][topic_id] += count
            result["subfield"][subfield_id] += count
            result["field"][field_id] += count
            result["domain"][domain_id] += count

        except (ValueError, TypeError):
            continue

    return result


def _compute_jaccard(gtr_dist: Dict, oa_dist: Dict) -> float:
    """Compute weighted Jaccard similarity."""
    intersection = sum(
        min(gtr_dist.get(k, 0), oa_dist.get(k, 0)) for k in set(gtr_dist) | set(oa_dist)
    )
    union = sum(
        max(gtr_dist.get(k, 0), oa_dist.get(k, 0)) for k in set(gtr_dist) | set(oa_dist)
    )
    return intersection / union if union > 0 else 0.0


def _compute_cosine(gtr_dist: Dict, oa_dist: Dict) -> float:
    """Compute cosine similarity between normalised distributions."""
    keys = set(gtr_dist) | set(oa_dist)
    gtr_vec = np.array([gtr_dist.get(k, 0) for k in keys])
    oa_vec = np.array([oa_dist.get(k, 0) for k in keys])

    # Normalise
    gtr_norm = gtr_vec / np.sum(gtr_vec) if np.sum(gtr_vec) > 0 else gtr_vec
    oa_norm = oa_vec / np.sum(oa_vec) if np.sum(oa_vec) > 0 else oa_vec

    dot_product = np.dot(gtr_norm, oa_norm)
    norms = np.linalg.norm(gtr_norm) * np.linalg.norm(oa_norm)

    return dot_product / norms if norms > 0 else 0.0


def _compute_jensen_shannon(gtr_dist: Dict, oa_dist: Dict) -> float:
    """Compute Jensen-Shannon divergence between normalised distributions."""
    keys = sorted(set(gtr_dist) | set(oa_dist))
    gtr_vec = np.array([gtr_dist.get(k, 0) for k in keys])
    oa_vec = np.array([oa_dist.get(k, 0) for k in keys])

    # Normalise
    gtr_norm = gtr_vec / np.sum(gtr_vec) if np.sum(gtr_vec) > 0 else gtr_vec
    oa_norm = oa_vec / np.sum(oa_vec) if np.sum(oa_vec) > 0 else oa_vec

    return jensenshannon(gtr_norm, oa_norm, base=2)


def _compute_coverage(gtr_dist: Dict, oa_dist: Dict) -> float:
    """Compute topic coverage ratio."""
    gtr_topics = set(gtr_dist.keys())
    oa_topics = set(oa_dist.keys())

    return len(gtr_topics & oa_topics) / len(gtr_topics) if gtr_topics else 0.0

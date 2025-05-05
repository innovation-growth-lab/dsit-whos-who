"""Service layer for fetching and processing author data for the Streamlit app."""

# pylint: skip-file
import logging
import pandas as pd
from kedro.io import DataCatalog
from typing import List, Dict, Any
import requests
import time

# import the existing OpenAlex fetcher
from dsit_whos_who.pipelines.data_collection_oa.utils.common import (
    fetch_openalex_objects,
)

from dsit_whos_who.pipelines.author_disambiguation.utils.preprocessing.oa import (
    process_affiliations,
    get_associated_institutions,
)

from dsit_whos_who.pipelines.author_disambiguation.utils.feature_engineering.compute_features import (
    compute_all_features,
)

from ..models.author import OpenAlexAuthorData

log = logging.getLogger(__name__)


def search_and_extract_features(
    name: str, institution: str, catalog: DataCatalog
) -> List[str]:
    """Searches OpenAlex for authors, processes results, and computes features.

    Args:
        name: Author name input by the user.
        institution: Institution name input by the user.
        catalog: Kedro DataCatalog for loading credentials or static data.

    Returns:
        A list of dictionaries, each containing computed features for a candidate OA author.
    """
    log.info(f"Starting author search for Name: '{name}', Institution: '{institution}'")

    params = catalog.load("params:oa.data_collection")

    # fetch candidate authors from OpenAlex
    author_results = fetch_openalex_objects(
        oa_id=name,
        mails=params["api"]["mails"],
        perpage=params["api"]["perpage"],
        filter_criteria=params["filter_author_search"],
        endpoint=params["authors_endpoint"],
    )

    # extract candidate institutions
    institution_ids = set()
    for author in author_results:
        for affiliation in author.get("affiliations", []):
            if isinstance(affiliation, list) and affiliation:
                inst_id = affiliation[0]
                if inst_id:
                    institution_ids.add(inst_id)

    # create OR-syntax chunks of 50 IDs
    institution_list = []
    ids_list = list(institution_ids)
    for i in range(0, len(ids_list), 50):
        chunk = ids_list[i : i + 50]
        institution_list.append("|".join(chunk))

    # fetch candidate institutions
    institution_results = []
    for chunk in institution_list:
        institution_results.extend(
            fetch_openalex_objects(
                oa_id=chunk,
                mails=params["api"]["mails"],
                perpage=params["api"]["perpage"],
                filter_criteria=params["filter_oa"],
                endpoint=params["institutions_endpoint"],
            )
        )

    institution_results = pd.DataFrame(institution_results)
    institutions_dict = institution_results.set_index("id")[
        "associated_institutions"
    ].to_dict()

    # process candidates
    candidate_df = pd.DataFrame(author_results)
    affiliations_processed = candidate_df["affiliations"].apply(process_affiliations)
    # add new columns while preserving original data
    candidate_df["institution_names"] = affiliations_processed.apply(lambda x: x[0])
    inst_ids = affiliations_processed.apply(lambda x: x[1])
    candidate_df["has_gb_affiliation"] = affiliations_processed.apply(lambda x: x[2])
    candidate_df["gb_affiliation_proportion"] = affiliations_processed.apply(
        lambda x: x[3]
    )

    # process associated institutions
    associated_processed = inst_ids.apply(
        lambda x: get_associated_institutions(x, institutions_dict)
    )
    candidate_df["associated_institution_names"] = associated_processed.apply(
        lambda x: x[0]
    )
    candidate_df["has_gb_associated"] = associated_processed.apply(lambda x: x[1])

    # drop the columns we used to create our features
    candidate_df = candidate_df.drop(
        columns=["affiliations", "last_known_institutions", "counts_by_year"],
        errors="ignore",
    )

    # add gtr_author_name
    candidate_df["gtr_author_name"] = name

    # add institution name
    candidate_df["organisation_name"] = institution

    # [TEMP] add project_topics, project_publications, project_authors as empty lists
    candidate_df["project_topics"] = [[] for _ in range(len(candidate_df))]
    candidate_df["project_publications"] = [[] for _ in range(len(candidate_df))]
    candidate_df["project_authors"] = [[] for _ in range(len(candidate_df))]

    # [TEMP] add placeholder person_id
    candidate_df["person_id"] = "USER_INPUT_" + name.replace(" ", "_")

    feature_matrix = compute_all_features(candidate_df)

    return feature_matrix

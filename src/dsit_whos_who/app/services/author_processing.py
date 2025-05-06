"""Service layer for fetching and processing author data for the Streamlit app."""

# pylint: skip-file
import logging
import pandas as pd
from kedro.io import DataCatalog
from typing import List, Dict, Any, Set

# import the existing openalex fetcher
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


class OpenAlexFetcher:
    """Class for fetching data from OpenAlex API."""

    def __init__(self, params: Dict[str, Any]):
        """Initialise with OpenAlex API parameters.

        Args:
            params: Dictionary containing OpenAlex API parameters.
        """
        self.params = params

    def fetch_authors(self, name: str) -> List[Dict[str, Any]]:
        """Fetch author data from OpenAlex based on name.

        Args:
            name: Author name to search for.

        Returns:
            List of author data dictionaries from OpenAlex.
        """
        return fetch_openalex_objects(
            oa_id=name,
            mails=self.params["api"]["mails"],
            perpage=self.params["api"]["perpage"],
            filter_criteria=self.params["filter_author_search"],
            endpoint=self.params["authors_endpoint"],
        )

    def fetch_institutions(self, institution_ids: Set[str]) -> List[Dict[str, Any]]:
        """Fetch institution data from OpenAlex based on IDs.

        Args:
            institution_ids: Set of institution IDs to fetch.

        Returns:
            List of institution data dictionaries from OpenAlex.
        """
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
                    mails=self.params["api"]["mails"],
                    perpage=self.params["api"]["perpage"],
                    filter_criteria=self.params["filter_oa"],
                    endpoint=self.params["institutions_endpoint"],
                )
            )

        return institution_results


class AuthorProcessor:
    """Class for processing author data from OpenAlex."""

    def __init__(self, name: str, institution: str):
        """Initialise with search parameters.

        Args:
            name: Author name input by the user.
            institution: Institution name input by the user.
        """
        self.name = name
        self.institution = institution

    def extract_institution_ids(self, authors: List[Dict[str, Any]]) -> Set[str]:
        """Extract institution IDs from author data.

        Args:
            authors: List of author data dictionaries.

        Returns:
            Set of institution IDs.
        """
        institution_ids = set()
        for author in authors:
            for affiliation in author.get("affiliations", []):
                if isinstance(affiliation, list) and affiliation:
                    inst_id = affiliation[0]
                    if inst_id:
                        institution_ids.add(inst_id)
        return institution_ids

    def process_candidates(
        self, authors: List[Dict[str, Any]], institutions_dict: Dict[str, Any]
    ) -> pd.DataFrame:
        """Process author candidates and compute features.

        Args:
            authors: List of author data dictionaries.
            institutions_dict: Dictionary mapping institution IDs to associated institutions.

        Returns:
            DataFrame with processed author data.
        """
        # process candidates
        candidate_df = pd.DataFrame(authors)

        # process affiliations
        affiliations_processed = candidate_df["affiliations"].apply(
            process_affiliations
        )
        candidate_df["institution_names"] = affiliations_processed.apply(lambda x: x[0])
        inst_ids = affiliations_processed.apply(lambda x: x[1])
        candidate_df["has_gb_affiliation"] = affiliations_processed.apply(
            lambda x: x[2]
        )
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
        candidate_df["gtr_author_name"] = self.name

        # add institution name
        candidate_df["organisation_name"] = self.institution

        # [TEMP] add project_topics, project_publications, project_authors as empty lists
        candidate_df["project_topics"] = [[] for _ in range(len(candidate_df))]
        candidate_df["project_publications"] = [[] for _ in range(len(candidate_df))]
        candidate_df["project_authors"] = [[] for _ in range(len(candidate_df))]

        # [TEMP] add placeholder person_id
        candidate_df["person_id"] = "USER_INPUT_" + self.name.replace(" ", "_")

        return candidate_df


class FeatureComputer:
    """Class for computing features from processed author data."""

    @staticmethod
    def compute_features(candidate_df: pd.DataFrame) -> pd.DataFrame:
        """Compute features from processed author data.

        Args:
            candidate_df: DataFrame with processed author data.

        Returns:
            DataFrame with computed features.
        """
        return compute_all_features(candidate_df)


def search_and_process(
    name: str, institution: str, catalog: DataCatalog
) -> pd.DataFrame:
    """Searches OpenAlex for authors, processes results, and computes features.

    Args:
        name: Author name input by the user.
        institution: Institution name input by the user.
        catalog: Kedro DataCatalog for loading credentials or static data.

    Returns:
        A DataFrame containing computed features for candidate OA authors.
    """
    log.info(f"Starting author search for Name: '{name}', Institution: '{institution}'")

    # load parameters
    params = catalog.load("params:oa.data_collection")

    # create instances
    fetcher = OpenAlexFetcher(params)
    processor = AuthorProcessor(name, institution)

    # fetch author data
    author_results = fetcher.fetch_authors(name)

    # extract institution IDs
    institution_ids = processor.extract_institution_ids(author_results)

    # fetch institution data
    institution_results = fetcher.fetch_institutions(institution_ids)

    # create institutions dictionary
    institution_results_df = pd.DataFrame(institution_results)
    if not institution_results_df.empty:
        institutions_dict = institution_results_df.set_index("id")[
            "associated_institutions"
        ].to_dict()
    else:
        institutions_dict = {}

    # process candidates
    candidate_df = processor.process_candidates(author_results, institutions_dict)

    return candidate_df


def compute_features(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """Compute features from processed author data.

    Args:
        candidate_df: DataFrame with processed author data.

    Returns:
        DataFrame with computed features.
    """
    return FeatureComputer.compute_features(candidate_df)

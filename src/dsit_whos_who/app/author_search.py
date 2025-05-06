"""Main Streamlit application script for DSIT Who's Who Author Search."""

# pylint: disable=W0718

import logging
import pandas as pd
import streamlit as st

from dsit_whos_who.app.utils.kedro_init import (
    setup_kedro_environment,
    get_kedro_context,
)
from dsit_whos_who.app.utils.cache_utils import (
    get_from_model_cache,
    store_in_model_cache,
)
from dsit_whos_who.app.utils.author_cache import (
    cached_search_and_process,
    cached_compute_features,
    load_model_dict,
    load_disambiguation_params,
)
from dsit_whos_who.pipelines.author_disambiguation.nodes import predict_author_matches

# configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


class AuthorSearchApp:
    """Streamlit application for author search and feature extraction."""

    def __init__(self):
        """Initialise the application and setup the Kedro environment."""
        # setup kedro environment
        if not setup_kedro_environment():
            st.error("Failed to initialise Kedro. Please check the logs.")
            st.stop()

        # configure the streamlit page
        st.set_page_config(page_title="DSIT Who's Who - Author Search", layout="wide")

        # initialise kedro resources
        self._initialise_kedro_resources()

    def _initialise_kedro_resources(self):
        """Initialise Kedro resources including context, catalog and model."""
        try:
            # get kedro context and catalog
            _, self.catalog = get_kedro_context(env="local")

            # try to load the model and parameters for author matching
            try:
                # use cached values if available
                model_dict = get_from_model_cache("ad_model_dict")
                disambiguation_params = get_from_model_cache("disambiguation_params")

                # if not in cache, load and store them
                if model_dict is None:
                    model_dict = load_model_dict()
                    store_in_model_cache("ad_model_dict", model_dict)

                if disambiguation_params is None:
                    disambiguation_params = load_disambiguation_params()
                    store_in_model_cache("disambiguation_params", disambiguation_params)

                self.model_dict = model_dict
                self.disambiguation_params = disambiguation_params
                self.has_model = True

                log.info("Successfully loaded author disambiguation model")
            except Exception as model_err:
                log.warning("Could not load author disambiguation model: %s", model_err)
                self.has_model = False

        except Exception as e:
            log.error("Failed to initialise Kedro resources: %s", e, exc_info=True)
            st.error(f"Failed to initialise resources: {e}")
            st.stop()

    def run(self):
        """Run the Streamlit application."""
        st.title("DSIT Who's Who - Author Search")
        st.markdown(
            "Enter an author's name and their institution to search OpenAlex and find potential matches."
        )

        # display the search form
        self._display_search_form()

    def _display_search_form(self):
        """Display the author search form."""
        with st.form(key="author_search_form"):
            st.subheader("Search Criteria")
            author_name_input = st.text_input(
                "Author Name *", placeholder="e.g., Anouk L'Hermitte"
            )
            institution_name_input = st.text_input(
                "Institution Name *", placeholder="e.g., Imperial College London"
            )

            # add matching options if model is available
            if self.has_model:
                st.subheader("Matching Options")
                col1, col2 = st.columns(2)
                with col1:
                    threshold = st.slider(
                        "Match Threshold",
                        min_value=0.1,
                        max_value=0.9,
                        value=0.5,
                        step=0.05,
                        help="Minimum probability threshold for considering a match",
                    )

                with col2:
                    model_choice = st.selectbox(
                        "Model Type",
                        options=["smote_model", "class_weights_model"],
                        index=0,
                        help="Type of model to use for predictions",
                    )

                enable_matching = st.checkbox(
                    "Run Author Disambiguation",
                    value=True,
                    help="Apply the author disambiguation model to find likely matches",
                )
            else:
                threshold = 0.5
                model_choice = "smote_model"
                enable_matching = False

            submitted = st.form_submit_button("Search Authors")

        # handle form submission
        if submitted:
            self._handle_search_submission(
                author_name_input,
                institution_name_input,
                threshold,
                model_choice,
                enable_matching,
            )
        else:
            st.markdown("_Enter search criteria above and click 'Search Authors'._")

    def _handle_search_submission(
        self,
        author_name: str,
        institution_name: str,
        threshold: float,
        model_choice: str,
        enable_matching: bool,
    ):
        """Handle the search form submission.

        Args:
            author_name: The author name entered by the user.
            institution_name: The institution name entered by the user.
            threshold: Probability threshold for matches.
            model_choice: Type of model to use.
            enable_matching: Whether to enable author disambiguation.
        """
        if not author_name or not institution_name:
            st.warning("Please enter both Author Name and Institution Name.")
            return

        st.info(f"Searching for '{author_name}' at '{institution_name}'...")
        results_placeholder = st.empty()
        results_placeholder.markdown("_(Running search and feature extraction...)_")

        try:
            with st.spinner("Initialising Kedro and searching..."):
                # step 1: search and process - using external cached function
                candidate_df = cached_search_and_process(
                    name=author_name, institution=institution_name
                )

                # step 2: compute features - using external cached function
                # Convert DataFrame to dict for hashing
                candidate_dict = candidate_df.to_dict()
                feature_matrix = cached_compute_features(candidate_dict)

                if feature_matrix.empty:
                    results_placeholder.empty()
                    st.warning(
                        "No potential author candidates found. Please try with a different name or institution."
                    )
                    return

                # step 3: apply author disambiguation
                match_predictions = None
                if enable_matching and self.has_model:
                    results_placeholder.markdown(
                        "_(Applying author disambiguation model...)_"
                    )
                    log.info("Predicting author matches")
                    prediction_params = {
                        "model_choice": model_choice,
                        "threshold": threshold,
                    }

                    match_predictions = predict_author_matches(
                        model_dict=self.model_dict,
                        feature_matrix=feature_matrix,
                        params=prediction_params,
                    )

            results_placeholder.empty()  # clear the placeholder message

            # display results
            if enable_matching and self.has_model:
                self._display_match_results(match_predictions, candidate_df)
            else:
                self._display_search_results(candidate_df)

        except Exception as e:
            results_placeholder.empty()
            log.error("Error during search execution: %s", e, exc_info=True)
            st.error(f"An unexpected error occurred: {e}")

    def _display_search_results(self, feature_matrix: pd.DataFrame):
        """Display the author search results without disambiguation.

        Args:
            feature_matrix: DataFrame containing the search results.
        """
        st.success(
            f"Found {len(feature_matrix)} potential authors matching the criteria."
        )

        # show candidates
        st.subheader("Author Candidates")
        self._show_candidate_details(feature_matrix)

    def _display_match_results(
        self, match_predictions: pd.DataFrame, feature_matrix: pd.DataFrame
    ):
        """Display the author matching results.

        Args:
            match_predictions: DataFrame containing the match predictions.
            feature_matrix: Original feature matrix with candidate details.
        """
        if match_predictions.empty:
            st.info(
                "No matches found that meet the threshold criteria. Try lowering the threshold "
                "or search with a different name/institution."
            )

            # show the top candidates anyway
            st.subheader("Top Candidates (Below Threshold)")
            self._show_candidate_details(feature_matrix.head(5))

        else:
            st.success(
                f"Found {len(match_predictions)} potential matches above the threshold."
            )

            # join with feature_matrix to get more details
            match_details = pd.merge(
                match_predictions,
                feature_matrix,
                left_on="oa_id",
                right_on="id",
                how="left",
            )

            # display the matched results
            st.subheader("Matched Authors")
            self._show_match_details(match_details)

    def _show_match_details(self, match_details: pd.DataFrame):
        """Show detailed information about matched authors.

        Args:
            match_details: DataFrame with match details and features.
        """
        # format the display columns
        display_cols = [
            "display_name",
            "oa_id",
            "match_probability",
            "institution_names",
            "works_count",
            "cited_by_count",
        ]

        if all(col in match_details.columns for col in display_cols):
            display_df = match_details[display_cols].copy()

            # format probability as percentage
            display_df["match_probability"] = display_df["match_probability"].apply(
                lambda x: f"{x:.1%}"
            )

            # rename columns for display
            display_df = display_df.rename(
                columns={
                    "display_name": "Author Name",
                    "oa_id": "OpenAlex ID",
                    "match_probability": "Match Score",
                    "institution_names": "Institutions",
                    "works_count": "Works Count",
                    "cited_by_count": "Citations",
                }
            )

            st.dataframe(display_df, use_container_width=True)

            # show full details for the top match
            if not match_details.empty:
                st.subheader(
                    f"Details for Top Match: {match_details.iloc[0]['display_name']}"
                )
                self._show_author_profile(match_details.iloc[0])
        else:
            st.dataframe(match_details, use_container_width=True)

    def _show_candidate_details(self, candidates: pd.DataFrame):
        """Show details for author candidates.

        Args:
            candidates: DataFrame with author candidates.
        """
        if candidates.empty:
            return

        # format the display columns - keep it simple
        display_cols = [
            "display_name",
            "id",
            "institution_names",
            "works_count",
            "cited_by_count",
        ]

        if all(col in candidates.columns for col in display_cols):
            display_df = candidates[display_cols].copy()

            # rename columns for display
            display_df = display_df.rename(
                columns={
                    "display_name": "Author Name",
                    "id": "OpenAlex ID",
                    "institution_names": "Institutions",
                    "works_count": "Works Count",
                    "cited_by_count": "Citations",
                }
            )

            st.dataframe(display_df, use_container_width=True)

    def _show_author_profile(self, author: pd.Series):
        """Show detailed profile for a single author.

        Args:
            author: Series with author details.
        """
        # create two columns layout
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Author Information")
            st.markdown(f"**Name:** {author.get('display_name', 'N/A')}")
            st.markdown(f"**ORCID:** {author.get('orcid', 'N/A')}")
            st.markdown(f"**Works Count:** {author.get('works_count', 'N/A')}")
            st.markdown(f"**Citations:** {author.get('cited_by_count', 'N/A')}")

            # add OpenAlex link
            author_id = author.get("id") or author.get("oa_id", "")
            if author_id:
                oa_url = f"https://openalex.org/authors/{author_id.split('/')[-1]}"
                st.markdown(f"[View on OpenAlex]({oa_url})")

        with col2:
            st.markdown("### Institutions")
            institutions = author.get("institution_names", [])
            if institutions:
                for inst in institutions:
                    st.markdown(f"- {inst}")
            else:
                st.markdown("No institution information available")

            # show topics if available
            st.markdown("### Research Topics")
            topics = author.get("x_topics", [])
            if topics:
                for topic in topics[:5]:  # limit to top 5
                    st.markdown(f"- {topic}")
            else:
                st.markdown("No topic information available")


# main entry point
if __name__ == "__main__":
    app = AuthorSearchApp()
    app.run()

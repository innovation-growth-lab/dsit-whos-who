"""Main Streamlit application script for DSIT Who's Who Author Search."""

# pylint: disable=W0718

import logging
import pandas as pd
import streamlit as st

from dsit_whos_who.app.utils.kedro_init import (
    setup_kedro_environment,
    get_kedro_context,
)
from dsit_whos_who.app.services.author_processing import search_and_extract_features

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

    def run(self):
        """Run the Streamlit application."""
        st.title("DSIT Who's Who - Author Search")
        st.markdown(
            "Enter an author's name and their institution to search OpenAlex and compute features."
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

            submitted = st.form_submit_button("Search Authors")

        # handle form submission
        if submitted:
            self._handle_search_submission(author_name_input, institution_name_input)
        else:
            st.markdown("_Enter search criteria above and click 'Search Authors'._")

    def _handle_search_submission(self, author_name: str, institution_name: str):
        """Handle the search form submission.

        Args:
            author_name: The author name entered by the user.
            institution_name: The institution name entered by the user.
        """
        if not author_name or not institution_name:
            st.warning("Please enter both Author Name and Institution Name.")
            return

        st.info(f"Searching for '{author_name}' at '{institution_name}'...")
        results_placeholder = st.empty()
        results_placeholder.markdown("_(Running search and feature extraction...)_")

        try:
            with st.spinner("Initialising Kedro and searching..."):
                # get kedro context and catalog
                _, catalog = get_kedro_context(env="local")

                # call the backend service function
                search_results = search_and_extract_features(
                    name=author_name,
                    institution=institution_name,
                    catalog=catalog,
                )

            results_placeholder.empty()  # clear the placeholder message

            self._display_search_results(search_results)

        except Exception as e:
            results_placeholder.empty()
            log.error("Error during search execution: %s", e, exc_info=True)
            st.error(f"An unexpected error occurred: {e}")

    def _display_search_results(self, search_results: pd.DataFrame):
        """Display the search results.

        Args:
            search_results: DataFrame containing the search results.
        """
        if search_results.shape[0] > 0:
            st.success(
                f"Found {len(search_results)} potential matches and computed features."
            )
            # display results as a dataframe
            st.dataframe(search_results)
        else:
            st.info(
                "No potential matches found in OpenAlex based on the provided criteria, "
                "or no features could be computed."
            )


# main entry point
if __name__ == "__main__":
    app = AuthorSearchApp()
    app.run()

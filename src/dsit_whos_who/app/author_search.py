"""Main Streamlit application script for DSIT Who's Who Author Search."""

import logging
import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# --- Kedro Setup ---
# Add the project root to the Python path to allow importing dsit_whos_who
# Adjust the relative path ../../ if app.py is moved
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# check src dir also in the path
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

# set KEDRO_PROJECT_PATH environment variable
if "KEDRO_PROJECT_PATH" not in os.environ:
    os.environ["KEDRO_PROJECT_PATH"] = str(project_root)

try:
    from kedro.framework.startup import bootstrap_project
    from kedro.framework.session import KedroSession
except ImportError:
    st.error("Kedro is not installed or not found in the Python path. ")
    st.stop()

try:
    # ensure the project is configured before importing project modules
    bootstrap_project(project_root)
    from dsit_whos_who.app.services.author_processing import search_and_extract_features
except ImportError as e:
    st.error(f"Could not import project components: {e}. ")
    st.stop()
except Exception as e:  # pylint: disable=W0718
    st.error(f"An error occurred during Kedro/project initialisation: {e}")
    st.stop()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# --- Streamlit App UI ---
st.set_page_config(page_title="DSIT Who's Who - Author Search", layout="wide")

st.title("DSIT Who's Who - Author Search")
st.markdown(
    "Enter an author's name and their institution to search OpenAlex and compute features."
)

# --- Search Form ---
with st.form(key="author_search_form"):
    st.subheader("Search Criteria")
    author_name_input = st.text_input(
        "Author Name *", placeholder="e.g., Anouk L'Hermitte"
    )
    institution_name_input = st.text_input(
        "Institution Name *", placeholder="e.g., Imperial College London"
    )

    submitted = st.form_submit_button("Search Authors")

# --- Search Execution and Results --- #
if submitted:
    if not author_name_input or not institution_name_input:
        st.warning("Please enter both Author Name and Institution Name.")
    else:
        st.info(f"Searching for '{author_name_input}' at '{institution_name_input}'...")
        results_placeholder = st.empty()
        results_placeholder.markdown("_(Running search and feature extraction...)_")

        try:
            with st.spinner("Initialising Kedro and searching..."):
                # Create a Kedro session to load the catalog
                # Assuming the default 'local' environment
                with KedroSession.create(
                    project_path=project_root, env="local"
                ) as session:
                    context = session.load_context()
                    catalog = context.catalog

                    # Call the backend service function
                    search_results = (  # pylint: disable=C0103
                        search_and_extract_features(
                            name=author_name_input,
                            institution=institution_name_input,
                            catalog=catalog,
                        )
                    )

            results_placeholder.empty()  # Clear the placeholder message

            if search_results.shape[0] > 0:
                st.success(
                    f"Found {len(search_results)} potential matches and computed features."
                )
                # Display results (simple display for now, refine later)
                st.dataframe(pd.DataFrame(search_results))
                # st.write(search_results) # Alternative raw display
            else:
                st.info(
                    "No potential matches found in OpenAlex based on the provided criteria, "
                    "or no features could be computed."
                )

        except Exception as e:  # pylint: disable=W0718
            results_placeholder.empty()  # Clear the placeholder message
            log.error("Error during search execution: %s", e, exc_info=True)
            st.error(f"An unexpected error occurred: {e}")

else:
    st.markdown("_Enter search criteria above and click 'Search Authors'._")

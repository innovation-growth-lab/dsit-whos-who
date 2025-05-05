"""Kedro initialisation utilities for Streamlit application."""

# pylint: disable=W0718

import logging
import os
import sys
from pathlib import Path
from typing import Tuple

from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession
from kedro.framework.context import KedroContext
from kedro.io import DataCatalog

log = logging.getLogger(__name__)
project_root = Path(__file__).resolve().parents[4]


def setup_kedro_environment() -> bool:
    """Set up the Kedro environment and ensure all paths are correctly configured.

    Returns:
        bool: True if setup was successful, False otherwise.
    """
    # add the project root to the python path to allow importing dsit_whos_who
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
        # ensure the project is configured
        bootstrap_project(project_root)
        return True
    except Exception as e:
        log.error("Failed to bootstrap Kedro project: %s", e)
        return False


def get_kedro_context(env: str = "local") -> Tuple[KedroContext, DataCatalog]:
    """Initialise a Kedro session and return the context and catalog.

    Args:
        env: Kedro environment name, defaults to "local".

    Returns:
        Tuple containing the Kedro context and catalog.

    Raises:
        RuntimeError: If the Kedro session cannot be created or loaded.
    """
    try:
        with KedroSession.create(project_path=project_root, env=env) as session:
            kcontext = session.load_context()
            kcatalog = kcontext.catalog
            return kcontext, kcatalog
    except Exception as e:
        log.error("Failed to create Kedro session: %s", e)
        raise RuntimeError(f"Failed to initialise Kedro: {e}") from e


# example usage (for testing or direct script execution):
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log.info("Attempting to load local Kedro catalog...")
    try:
        context, catalog = get_kedro_context(env="local")
        log.info("Successfully loaded catalog. Found %d datasets.", len(catalog.list()))
        # example: try loading credentials
        try:
            creds = catalog.load("openalex_credentials")
            log.info("Successfully loaded openalex_credentials: %s", creds)
        except Exception as load_err:
            log.warning("Could not load openalex_credentials: %s", load_err)
    except Exception as e:
        log.error("Failed to get catalog: %s", e)

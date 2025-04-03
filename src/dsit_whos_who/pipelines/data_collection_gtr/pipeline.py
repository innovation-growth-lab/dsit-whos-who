"""
Gateway to Research data collection pipeline.

This pipeline orchestrates the collection and preprocessing of data from the
Gateway to Research (GtR) API. It provides:

Core Functionality:
- Parallel data fetching from multiple GtR endpoints
- Standardised data preprocessing for each endpoint type
- Configurable test mode for development
- Namespace management for output organisation

Available Endpoints:
- Projects: Research project metadata and relationships
- Publications: Research outputs and citations
- Persons: Researcher profiles and affiliations
- Organisations: Institution details and addresses
- Funds: Grant funding details and amounts

Usage:
    Run complete pipeline:
    ```bash
    kedro run --pipeline data_collection_gtr
    ```

    Run specific endpoint:
    ```bash
    kedro run --pipeline data_collection_gtr --tags gtr_projects
    ```

Note:
    Namespace prefixes are automatically applied to node outputs for
    organisational clarity.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import fetch_gtr_data, concatenate_endpoint
from ...settings import GTR_ENDPOINTS


def create_pipeline(**kwargs) -> Pipeline:  # pylint: disable=W0613
    """Create data collection pipeline for GtR endpoints.

    Constructs a modular pipeline that:
    - Fetches raw data from configured GtR endpoints
    - Preprocesses data into standardised formats
    - Organises outputs using namespaced structure

    Returns:
        Pipeline with endpoint-specific nodes and tags
    """
    template_pipeline = pipeline(
        [
            node(
                func=fetch_gtr_data,
                inputs={
                    "parameters": "params:param_requests",
                    "url_endpoint": "params:url_endpoint",
                    "test_mode": "params:test_mode",
                },
                outputs="raw",
                name="fetch_gtr_data",
            ),
            node(
                func=concatenate_endpoint,
                inputs="raw",
                outputs="intermediate",
                name="concatenate_endpoint",
            ),
        ]
    )
    pipelines = []
    for endpoint in GTR_ENDPOINTS:
        pipelines.append(
            pipeline(
                template_pipeline,
                namespace=f"gtr.data_collection.{endpoint}",
                tags=[f"gtr_{endpoint}"],
            )
        )
    return sum(pipelines)

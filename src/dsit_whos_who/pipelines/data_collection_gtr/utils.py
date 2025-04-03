"""
Utility functions for Gateway to Research data processing.

This module provides core utilities for handling GtR API data structures and
transformations. It includes:

Core Functions:
- API configuration management
- Address extraction and standardisation
- Date parsing and normalisation
- Nested dictionary transformation
- Value extraction from complex data structures

The utilities focus on maintaining consistent data formats and efficient
processing of the hierarchical JSON structures returned by the GtR API.

Functions:
    api_config(parameters, endpoint): Constructs the API configuration dictionary from parameters.
    extract_main_address(addresses): Extracts the main address from a list of addresses.
    extract_date(links, date_type, link_type): Extracts dates from link dictionaries.
    extract_value_from_nested_dict(data, outer_key, inner_key, inner_value, extract_key):
        Extracts a value from a nested dictionary structure.
    transform_nested_dict(df, parent_col, inner_keys): Transforms columns containing nested
        dictionaries into flattened columns.

Dependencies:
    - pandas
    - datetime
    - typing
"""

from typing import Dict, List, Any, Optional
import datetime
import pandas as pd


def api_config(parameters: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
    """Create API configuration for Gateway to Research endpoint.

    Processes:
    - Base URL configuration
    - Header setup
    - Pagination settings
    - Retry policy configuration

    Args:
        parameters: Raw configuration parameters
        endpoint: Target API endpoint

    Returns:
        Standardised API configuration dictionary
    """
    gtr_config = parameters["gtr_config"]
    key = endpoint.split("/")[-1].rstrip("s")

    base_url, headers, page_size = (
        gtr_config["base_url"],
        gtr_config["headers"],
        gtr_config["page_size"],
    )
    max_retries, backoff_factor = (
        parameters["max_retries"],
        parameters["backoff_factor"],
    )

    return {
        "key": key,
        "base_url": base_url,
        "headers": headers,
        "page_size": page_size,
        "max_retries": max_retries,
        "backoff_factor": backoff_factor,
    }


def extract_main_address(addresses: List[Dict[str, Any]]) -> pd.Series:
    """Extract primary address from organisation address list.

    Processes:
    - Main address identification
    - Address component extraction
    - Standardised field formatting

    Example structure:
    ```json
    {
        "addresses": [{
            "id": 123,
            "created": "2018-01-01T00:00:00Z",
            "postCode": "SW1A 1AA",
            "region": "London",
            "type": "MAIN_ADDRESS"
        }]
    }
    ```

    Args:
        addresses: List of address dictionaries

    Returns:
        Series with standardised address fields
    """
    address = addresses["address"]
    main_address = next(
        (addr for addr in address if addr["type"] == "MAIN_ADDRESS"), None
    )
    return (
        pd.Series(main_address)
        if main_address
        else pd.Series([None] * 4, index=["id", "created", "postCode", "region"])
    )


def extract_value_from_nested_dict(
    data: List[Dict[str, Any]],
    outer_key: str,
    inner_key: str,
    inner_value: Any,
    extract_key: str,
    split_on_slash: bool = True,
) -> Any:
    """Extract values from nested dictionary structures.

    Processes:
    - Nested dictionary traversal
    - Value matching and extraction
    - Optional path splitting
    - Multiple match handling

    Args:
        data: Source data structure
        outer_key: Top-level dictionary key
        inner_key: Nested dictionary key
        inner_value: Value to match
        extract_key: Key of value to extract
        split_on_slash: Whether to split extracted paths

    Returns:
        Extracted value or maximum value if multiple matches
    """
    matched_dicts = [item for item in data[outer_key] if item[inner_key] == inner_value]

    if not matched_dicts:
        return None

    if split_on_slash:
        values = [
            d[extract_key].split("/")[-1] for d in matched_dicts if d.get(extract_key)
        ]
        return max(values) if values else None

    values = [d[extract_key] for d in matched_dicts if d.get(extract_key)]
    return max(values) if values else None


def extract_date(
    links: Dict[str, Any], extract_key: str, inner_value: str
) -> Optional[str]:
    """Extract and standardise dates from link structures.

    Processes:
    - Date field identification
    - Timestamp conversion
    - Format standardisation
    - Error handling

    Args:
        links: Link data structure
        extract_key: Target date field
        inner_value: Link type identifier

    Returns:
        Standardised date string (YYYY-MM-DD) or None
    """
    timestamp = extract_value_from_nested_dict(
        data=links,
        outer_key="link",
        inner_key="rel",
        inner_value=inner_value,
        extract_key=extract_key,
        split_on_slash=False,
    )
    try:
        if timestamp:
            return datetime.datetime.fromtimestamp(timestamp / 1000).strftime(
                "%Y-%m-%d"
            )
    except (TypeError, ValueError):
        pass
    return None


def transform_nested_dict(
    df: pd.DataFrame, parent_col: str, inner_keys: List[str]
) -> pd.DataFrame:
    """Transform nested dictionary columns into flattened structure.

    Processes:
    - Dictionary structure flattening
    - Key filtering and extraction
    - List comprehension optimisation
    - Memory usage optimisation

    Args:
        df: Source DataFrame
        parent_col: Column containing nested dictionaries
        inner_keys: Keys to extract from nested structure

    Returns:
        DataFrame with flattened dictionary columns
    """

    def _extract_dict(row):
        # Extract the nested list/dict under the parent key
        nested_items = row.get(parent_col, [])[parent_col[:-1]]
        return [
            {key: item[key] for key in item if key in inner_keys}
            for item in nested_items
        ]

    df[parent_col] = df.apply(_extract_dict, axis=1)
    return df

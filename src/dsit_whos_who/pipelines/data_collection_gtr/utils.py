"""
Utility functions for the GtR data collection pipeline.

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
    """
    Constructs the API configuration dictionary for a given endpoint.

    Args:
        parameters (Dict[str, Any]): A dictionary containing the parameters for
            the API configuration.
        endpoint (str): The endpoint URL.

    Returns:
        Dict[str, Any]: The API configuration dictionary.

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
    """Extract the main address from the addresses list. This resolves the
    issue of having multiple addresses per organisation. For example:

    "addresses": [
        {
            "id": 123,
            "created": "2018-01-01T00:00:00Z",
            "postCode": "SW1A 1AA",
            "region": "London",
            "type": "MAIN_ADDRESS"
        },
        {
            ...
        }
    ]

    Args:
        addresses (List[Dict[str, Any]]): The addresses list.

    Returns:
        pd.Series: The main address.
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
    """Extracts a value from a nested dictionary based on the given keys and value.
    If multiple matches are found, returns the maximum value.

    Args:
        data (List[Dict[str, Any]]): The list of dictionaries.
        outer_key (str): The key of the outer dictionary.
        inner_key (str): The key of the inner dictionary.
        inner_value (Any): The value to match in the inner dictionary.
        extract_key (str): The key of the value to extract.
        split_on_slash (bool): Whether to split the extracted value on slashes.

    Returns:
        Any: The maximum extracted value if multiple matches found, otherwise the single match.
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
    """
    Extract a date from links data.

    Args:
        links: The links data.
        extract_key: The key to extract.
        inner_value: The inner value to match.

    Returns:
        The extracted date as a string in YYYY-MM-DD format or None if extraction fails.
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
    """
    Transforms a column containing nested dictionaries into an inner dict.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        parent_col (str): The name of the column containing the nested dictionaries.
        inner_keys (List[str]): The keys in the nested dictionaries to be extracted.

    Returns:
        pd.DataFrame: The DataFrame with the transformed column.
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

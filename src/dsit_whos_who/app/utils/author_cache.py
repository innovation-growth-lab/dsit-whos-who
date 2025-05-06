"""Caching utilities specific to author search and disambiguation."""

import logging
import pandas as pd
import streamlit as st

from dsit_whos_who.app.utils.kedro_init import get_kedro_context
from dsit_whos_who.app.utils.cache_utils import persistent_cached_value
from dsit_whos_who.app.services.author_processing import (
    search_and_process,
    compute_features,
)

# configure logging
log = logging.getLogger(__name__)


@st.cache_data(ttl=600)
def cached_search_and_process(name, institution, catalog_name="base"):
    """Cached version of search_and_process to avoid repeated API calls.
    
    This is defined outside any class to avoid hashing issues with 'self'
    and complex objects like the catalog.
    
    Args:
        name: the author name to search for
        institution: the institution name to search for
        catalog_name: the name of the kedro catalog environment to use
        
    Returns:
        dataframe with search results
    """
    log.info("performing cached search for name: %s at institution: %s", name, institution)
    # get a fresh catalog for each call
    _, catalog = get_kedro_context(env=catalog_name)
    return search_and_process(name=name, institution=institution, catalog=catalog)


@st.cache_data(ttl=600)
def cached_compute_features(candidate_df_dict):
    """Cached version of compute_features to avoid repeated computation.
    
    This is defined outside class context to avoid hashing issues with 'self'.
    note: we convert the dataframe to a dict to make it hashable.
    
    Args:
        candidate_df_dict: dictionary representation of dataframe with candidate authors
        
    Returns:
        feature matrix for the candidates
    """
    log.info("computing features for candidates")
    # convert dict back to dataframe
    candidate_df = pd.DataFrame.from_dict(candidate_df_dict)
    return compute_features(candidate_df)


@persistent_cached_value(key="model_dict", ttl=3600)
def load_model_dict(catalog_name="base"):
    """Load the model dictionary with persistent caching.
    
    Args:
        catalog_name: the name of the kedro catalog environment
        
    Returns:
        the loaded model dictionary
    """
    log.info("loading model dictionary from catalog")
    _, catalog = get_kedro_context(env=catalog_name)
    return catalog.load("ad.model.choice")


@persistent_cached_value(key="disambiguation_params", ttl=3600)
def load_disambiguation_params(catalog_name="base"):
    """Load the disambiguation parameters with persistent caching.
    
    Args:
        catalog_name: the name of the kedro catalog environment
        
    Returns:
        the loaded disambiguation parameters
    """
    log.info("loading disambiguation parameters from catalog")
    _, catalog = get_kedro_context(env=catalog_name)
    return catalog.load("params:model_prediction") 
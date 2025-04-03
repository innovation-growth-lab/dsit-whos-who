# OpenAlex Data Collection Pipeline

This pipeline fetches and processes data from the OpenAlex API, collecting information about authors, publications, and institutions related to GTR (Gateway to Research) data.

## Pipeline Overview

The pipeline consists of four main sub-pipelines:

1. **ORCID Pipeline** (`fetch_orcid`)
   - Creates list of ORCID IDs from GTR person data
   - Fetches author data from OpenAlex using ORCID identifiers
   - Processes and concatenates the results

2. **Author Search Pipeline** (`fetch_author_names`)
   - Creates list of author names from GTR person data
   - Searches OpenAlex for matching authors
   - Used for author disambiguation and matching

3. **Publications Pipeline** (`fetch_doi_publications`)
   - Creates list of DOIs from GTR publication data
   - Fetches detailed publication data from OpenAlex
   - Processes and concatenates publication records

4. **Institutions Pipeline** (`fetch_institutions`)
   - Extracts institution IDs from author search results
   - Fetches detailed institution data from OpenAlex
   - Processes and concatenates institution records

## Implementation Notes

- API requests are batched and processed in parallel for efficiency
- Rate limiting is handled through email registration with OpenAlex
- Results are partitioned into chunks of 40 queries each
- Missing or invalid identifiers are logged but don't halt the pipeline
- Data is stored in intermediate formats for subsequent processing

## Pipeline Structure

```python
def create_pipeline(**kwargs) -> Pipeline:
    return (
        orcid_pipeline           # Fetch author data via ORCID
        + author_search_pipeline # Search authors by name
        + publications_pipeline  # Fetch publication data
        + institutions_pipeline  # Fetch institution data
    )
```

## Running the Pipeline

Run the complete pipeline:
```bash
kedro run --pipeline data_collection_oa
```

Run specific sub-pipelines using tags:
```bash
# Fetch only ORCID data
kedro run --pipeline data_collection_oa --tags fetch_orcid

# Fetch only publication data
kedro run --pipeline data_collection_oa --tags fetch_doi_publications

# Fetch only institution data
kedro run --pipeline data_collection_oa --tags fetch_institutions

# Fetch only author search data
kedro run --pipeline data_collection_oa --tags fetch_author_names
```

## Dependencies

- OpenAlex API access (free, but email registration recommended)
- Python packages:
  - kedro
  - pandas
  - requests
  - joblib (for parallel processing)

## Notes

- All dates are in ISO format (YYYY-MM-DD)
- Country codes follow ISO 3166-1 alpha-2 standard
- Missing values are handled gracefully by the pipeline 
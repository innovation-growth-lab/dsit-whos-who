# Basic metrics analysis

This pipeline processes and analyses basic metrics for researchers, combining data from OpenAlex and Gateway to Research (GTR).

## Pipeline overview

### 1. Data integration and preprocessing
- `process_matched_author_metadata`: Processes OpenAlex author metadata and matches with GTR IDs
- `process_matched_person_gtr_data`: Processes GTR person data and project information
- `process_matched_author_works`: Creates yearly collaboration summaries from publication data

### 2. Metric computation

The `compute_basic_metrics` function orchestrates the computation of all metrics through the following steps:

#### Author metadata processing
- `process_author_metadata`: Extracts first publication year and citation metrics
- `compute_academic_age`: Calculates years between first publication and first grant

#### Grant processing
- `process_person_gtr_data`: Aggregates grant categories, funders, and project timelines
- `prepare_final_person_data`: Creates summary statistics of grant activity

#### International metrics
- `_process_affiliations`: Analyses institutional affiliations and time spent abroad
- `_process_last_institution`: Determines if current institution is in the UK
- `_calculate_uk_fraction`: Computes fraction of time in UK vs abroad

#### Collaboration metrics
- `_process_collaborations`: Analyses collaboration patterns before/after first grant
- `process_publication_batch`: Processes publication batches for collaboration counting
- Computes unique collaborators, international collaborations, and country-level statistics

## Data dictionary

### Personal identifiers and basic info
| Variable | Description | Type |
|----------|-------------|------|
| oa_id | OpenAlex unique identifier | string |
| orcid | ORCID identifier | string |
| display_name | Full name from OpenAlex | string |
| display_name_alternatives | Alternative names | list[string] |
| first_name | First name from GTR | string |
| surname | Surname from GTR | string |
| gtr_person_id | GTR person ID | string |
| match_probability | Probability of correct matching | float |

### Current institutional information
| Variable | Description | Type |
|----------|-------------|------|
| gtr_organisation | Current GTR organisation ID | string |
| gtr_organisation_name | Current GTR organisation name | string |
| last_known_institutions | List of last known institutions | list[list] |
| last_known_institution_uk | Whether last known institution is in UK | boolean |

### Academic profile and metrics
| Variable | Description | Type |
|----------|-------------|------|
| works_count | Total number of works | integer |
| cited_by_count | Total citations | integer |
| citations_per_publication | Average citations per publication | float |
| h_index | H-index | integer |
| i10_index | i10-index | integer |
| first_work_year | First publication year | integer |
| academic_age_at_first_grant | Academic age when receiving first grant | integer |
| topics | Research topics | list[list] |
| affiliations | Historical affiliations | list[list] |
| counts_by_year | Publication counts by year (OpenAlex, limited to 2012) | list[list] |
| counts_by_pubyear | Publication counts by publication year | list[list] |

### Citation counting methods
The pipeline uses two different approaches to count citations:

1. **Publication year based (counts_by_pubyear)**
   - Citations are attributed to the publication year of the cited paper
   - Used for most before/after calculations in the pipeline
   - Complete coverage of the time period
   - Less granular view of citation impact over time
   - Format: `[[year, {citations: X, works: Y}], ...]`

2. **Citation year based (counts_by_year)**
   - Citations are counted in the year they occur
   - Only available from OpenAlex up to 2012
   - More accurate representation of when impact occurs
   - Limited usefulness for before/after comparisons due to 2012 cutoff
   - Format: `[[year, {citations: X, works: Y}], ...]`

Due to the 2012 limitation in OpenAlex's citation-year data and the fact that many beneficiaries received their first UKRI funding before 2012, we primarily use publication-year based metrics (counts_by_pubyear) for before/after comparisons to avoid temporal distortions. Both metrics are retained for transparency and different analytical needs.

### Grant information
| Variable | Description | Type |
|----------|-------------|------|
| earliest_start_date | First grant start date | date |
| latest_end_date | Last grant end date | date |
| has_active_project | Whether has active projects | boolean |
| number_grants | Total number of grants | integer |
| has_multiple_funders | Whether has multiple funders | boolean |
| grant_categories | List of grant categories | list[list] |
| lead_funders | List of lead funders | list[list] |
| gtr_project_timeline | Detailed project timeline | list[list] |
| gtr_project_id | GTR project IDs | list[string] |
| gtr_project_publications | Project-linked publications | list[string] |
| gtr_project_topics | Project-specific topics | list[list] |
| gtr_project_oa_authors | Project OpenAlex authors | list[string] |
| gtr_project_oa_ids | Project OpenAlex IDs | list[string] |

### Publication metrics before/after first grant
| Variable | Description | Type |
|----------|-------------|------|
| n_pubs_before | Number of publications before | integer |
| n_pubs_after | Number of publications after | integer |
| total_citations_pubyear_before | Total citations by pub year before | integer |
| total_citations_pubyear_after | Total citations by pub year after | integer |
| mean_citations_pubyear_before | Mean citations by pub year before | float |
| mean_citations_pubyear_after | Mean citations by pub year after | float |
| citations_pp_pubyear_before | Citations per pub by pub year before | float |
| citations_pp_pubyear_after | Citations per pub by pub year after | float |
| mean_citations_before | Mean citations before | float |
| mean_citations_after | Mean citations after | float |
| citations_pp_before | Citations per pub before | float |
| citations_pp_after | Citations per pub after | float |
| mean_fwci_before | Mean FWCI before | float |
| mean_fwci_after | Mean FWCI after | float |

### International experience metrics
| Variable | Description | Type |
|----------|-------------|------|
| abroad_experience_before | Had international experience before | boolean |
| abroad_experience_after | Had international experience after | boolean |
| countries_before | Countries worked in before | list[string] |
| countries_after | Countries worked in after | list[string] |
| abroad_fraction_before | Fraction of time abroad before | float |
| abroad_fraction_after | Fraction of time abroad after | float |

### Collaboration metrics
| Variable | Description | Type |
|----------|-------------|------|
| collab_countries_before | Collaboration countries with counts before | list[list] |
| collab_countries_after | Collaboration countries with counts after | list[list] |
| collab_countries_list_before | List of collaboration countries before | list[string] |
| collab_countries_list_after | List of collaboration countries after | list[string] |
| unique_collabs_before | Unique collaborators before | integer |
| unique_collabs_after | Unique collaborators after | integer |
| total_collabs_before | Total collaborations before | integer |
| total_collabs_after | Total collaborations after | integer |
| foreign_collab_fraction_before | Fraction of foreign collabs before | float |
| foreign_collab_fraction_after | Fraction of foreign collabs after | float |

## Notes

- All dates are in ISO format (YYYY-MM-DD)
- Country codes follow the ISO 3166-1 alpha-2 standard
- List[list] entries typically contain pairs of [item, count] or [item, metadata]
- Missing values are represented as:
  - NaN for numeric fields
  - None/null for string fields
  - Empty lists for array fields
  - pd.NA for nullable integer fields (Int64)
  - Empty strings for optional string fields

## Pipeline Structure

The pipeline consists of three main stages:

1. **Collection Pipeline** (`collection_pipeline_basic_metrics`)
   - Creates list of OpenAlex author IDs from matched authors
   - Fetches publication data for matched authors from OpenAlex
   - Processes and cleans raw publication data

2. **Processing Pipeline**
   - Processes author metadata from OpenAlex
   - Processes person and project data from GTR
   - Creates detailed publication and collaboration summaries

3. **Metric Computation Pipeline**
   - Combines all processed data
   - Computes before/after metrics relative to first grant
   - Generates final dataset with all metrics

## Running the Pipeline

Run the complete pipeline using:
```bash
kedro run --pipeline data_analysis_basic_metrics
```

Run specific stages using tags:
```bash
# Run only the data collection
kedro run --pipeline data_analysis_basic_metrics --tags collection_pipeline_basic_metrics
```
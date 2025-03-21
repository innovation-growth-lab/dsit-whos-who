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

### Personal identifiers
| Variable | Description | Type |
|----------|-------------|------|
| id | OpenAlex unique identifier | string |
| orcid | ORCID identifier | string |
| display_name | Author's full name | string |
| first_name | Author's first name from GTR | string |
| surname | Author's surname from GTR | string |
| display_name_alternatives | List of alternative name spellings | list[string] |

### Current institutional information
| Variable | Description | Type |
|----------|-------------|------|
| organisation | Current organisation identifier from GTR | string |
| organisation_name | Current organisation name | string |
| last_known_institution_uk | Whether the last known institution is in the UK | boolean |
| last_known_institutions | List of last known institutions with details | list[list] |

### Academic metrics
| Variable | Description | Type |
|----------|-------------|------|
| works_count | Total number of publications | integer |
| cited_by_count | Total number of citations | integer |
| citations_per_publication | Average citations per publication | float |
| h_index | H-index value | integer |
| i10_index | Number of publications with at least 10 citations | integer |
| first_work_year | Year of first publication | integer |
| academic_age_at_first_grant | Years between first publication and first grant | integer |

### Grant information
| Variable | Description | Type |
|----------|-------------|------|
| earliest_start_date | Start date of first grant | date |
| latest_end_date | End date of last grant | date |
| has_active_project | Whether researcher has active projects | boolean |
| number_grants | Total number of grants | integer |
| has_multiple_funders | Whether grants come from multiple funders | boolean |
| grant_categories | List of grant categories with counts | list[list] |
| lead_funders | List of lead funders with counts | list[list] |
| project_timeline | Detailed timeline of all projects | list[list] |

### Research profile
| Variable | Description | Type |
|----------|-------------|------|
| topics | Research topics with confidence scores | list[list] |
| affiliations | Historical institutional affiliations | list[list] |
| counts_by_year | Publication counts per year | list[list] |

### Project outputs
| Variable | Description | Type |
|----------|-------------|------|
| project_publications | Publications linked to projects | list[string] |
| project_topics | Topics associated with projects | list[list] |
| project_authors | Project collaborators with counts | list[list] |
| project_oa_ids | OpenAlex IDs of project outputs | list[list] |

### International experience
| Variable | Description | Type |
|----------|-------------|------|
| abroad_experience_before | International experience before first grant | boolean |
| abroad_experience_after | International experience after first grant | boolean |
| countries_before | Countries worked in before first grant | list[string] |
| countries_after | Countries worked in after first grant | list[string] |
| abroad_fraction_before | Fraction of time spent abroad before first grant | float |
| abroad_fraction_after | Fraction of time spent abroad after first grant | float |

### Collaboration metrics
| Variable | Description | Type |
|----------|-------------|------|
| unique_collabs_before | Number of unique collaborators before first grant | integer |
| unique_collabs_after | Number of unique collaborators after first grant | integer |
| total_collabs_before | Total number of collaborations before first grant | integer |
| total_collabs_after | Total number of collaborations after first grant | integer |
| foreign_collab_fraction_before | Fraction of foreign collaborations before first grant | float |
| foreign_collab_fraction_after | Fraction of foreign collaborations after first grant | float |
| collab_countries_before | Countries of collaborators before first grant with counts | list[list] |
| collab_countries_after | Countries of collaborators after first grant with counts | list[list] |
| collab_countries_list_before | List of unique collaboration countries before first grant | list[string] |
| collab_countries_list_after | List of unique collaboration countries after first grant | list[string] |

## Notes

- All dates are in ISO format (YYYY-MM-DD)
- Country codes follow the ISO 3166-1 alpha-2 standard
- Boolean values are represented as strings ("True"/"False")
- List[list] typically contains pairs of [item, count] or [item, score]
- Missing values are represented as NaN for numeric fields and empty lists for array fields 
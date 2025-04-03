# Who's Who in UK Research

A data pipeline for linking Gateway to Research (GtR) and OpenAlex data to analyse research impact at the person level.

## Project structure

The project consists of five main pipeline blocks that process data from collection through to metric computation.

### 1. Gateway to Research data collection
The pipeline fetches data from the GtR API with a primary focus on project data. It extracts research topics, subjects, and linked publications while implementing retry mechanisms and rate limiting for robust data collection.

### 2. OpenAlex data collection
Data collection from OpenAlex occurs through four interconnected sub-pipelines. The first two focus on author data, collecting through ORCID identifiers and name-based searches. The remaining pipelines retrieve publication data through DOIs and collect institution information. The system processes queries in parallel batches of 40 and implements rate limiting through email registration.

### 3. Author disambiguation
The matching between GtR persons and OpenAlex authors uses a machine learning approach based on five feature categories: name similarity, institution matching, topic similarity, publication overlap, and author metadata. At a threshold of 0.80, the system achieves precision of 0.920 and recall of 0.937.

The pipeline successfully links 67.6% of matchable GtR authors (85,444 out of 126,304). Coverage varies by grant type, with higher rates in academic research grants (Fellowships: 82.9%, Research Grants: 79.0%) and lower rates in industry-focused schemes (Collaborative R&D: 13.7%, Feasibility Studies: 15.1%).

### 4. Basic metrics analysis
The basic metrics cover four dimensions: academic profile (publications, citations, indices), grant information and funding patterns, international experience, and collaboration patterns. All metrics are computed before and after first funding to enable temporal analysis. The pipeline implements two citation counting approaches: a complete publication-year based method and a citation-year based method limited to 2012.

### 5. Complex metrics analysis
The pipeline implements two sets of bibliometric indicators. The first is the Wu & Yan variant of the disruption index, which measures how research affects subsequent directions through citation patterns. The second implements the Leydesdorff framework for discipline diversity, measuring variety in topic coverage, evenness in distribution, and disparity through cognitive distance.

## Running the pipeline

Each component can be run independently:

```bash
# GtR data collection
kedro run --pipeline data_collection_gtr

# OpenAlex data collection
kedro run --pipeline data_collection_oa

# Author disambiguation
kedro run --pipeline author_disambiguation

# Basic metrics
kedro run --pipeline data_analysis_basic_metrics

# Complex metrics
kedro run --pipeline data_analysis_complex_metrics
```

## Dependencies

- Python 3.8+
- Kedro
- pandas
- scikit-learn
- torch (for SPECTER embeddings)
- requests
- joblib

## References

- Wu & Yan (2019). [https://doi.org/10.48550/arXiv.1905.03461]
- Leydesdorff, Wagner, & Bornmann (2019) [https://doi.org/10.1016/j.joi.2018.12.006]
- Leibel & Bornmann (2023) [https://doi.org/10.48550/arXiv.2308.02383]

# Who's Who in UK Research

A data pipeline for linking Gateway to Research (GtR) and OpenAlex data to analyse research impact at the individual researcher level.

```mermaid
---
title: DSIT Who's Who Data Pipeline Overview
config:
  theme: base
  themeVariables:
    fontSize: 16px
    mainBkg: '#FFFFFF'
    primaryColor: '#f8f8f8'
    primaryTextColor: '#333'
    primaryBorderColor: '#ccc'
    lineColor: '#555'
    nodeBorder: '#999'
    textColor: '#333'
---
graph TD
    classDef api fill:#cfe2f3,stroke:#333,stroke-width:1px,color:#333;
    classDef data fill:#e2dcf7,stroke:#5a4f91,stroke-width:1.5px,color:#333;
    classDef action fill:#d9ead3,stroke:#5b824d,stroke-width:1.5px,color:#333;
    classDef methodology fill:#fceecb,stroke:#b48c34,stroke-width:1px,stroke-dasharray: 3 3,color:#444,shape:hexagon;
    classDef outputNode fill:#fff2cc,stroke:#c49b00,stroke-width:1.5px,color:#333;
    classDef pipelineTitle fill:#e0e0e0,stroke:#333,stroke-width:1px,color:#000,font-weight:bold;

    subgraph External APIs
        direction LR  # Changed direction
            GTR_API[GtR API]:::api;
            OA_API[OpenAlex API]:::api;
    end

    subgraph P1 [1.GtR Data Collection]
        direction TB
            P1_Title[data_collection_gtr]:::pipelineTitle;
            P1_Fetch[Fetch GtR Data<br>Projects Focus]:::action;
            P1_Out[GtR Data & Identifiers]:::data;

            GTR_API --> P1_Fetch;
            P1_Fetch --> P1_Out;
    end

    subgraph P2 [2.OpenAlex Data Collection]
        direction TB
            P2_Title[data_collection_oa]:::pipelineTitle;
            P2_Fetch[Fetch OA Data<br>using GtR IDs]:::action;
            P2_Out[OA Data<br>Authors, Pubs, Inst]:::data;

            OA_API --> P2_Fetch;
            P1_Out -- GtR Identifiers --> P2_Fetch;
            P2_Fetch --> P2_Out;
    end

    subgraph P3 [3.Author Disambiguation]
        direction TB
            P3_Title[author_disambiguation]:::pipelineTitle;
            P3_Match[Match GtR <=> OA Authors]:::action;
            P3_Method[ML Model<br>Name, Inst, Topics, Pubs]:::methodology;
            P3_Out[Matched Author Pairs]:::data;

            P1_Out -- GtR Person/Project Data --> P3_Match;
            P2_Out -- OA Author Candidates --> P3_Match;
            P3_Match -- Uses --> P3_Method;
            P3_Match --> P3_Out;
    end

    subgraph P4 [4.Basic Metrics Analysis]
        direction TB
            P4_Title[data_analysis_basic_metrics]:::pipelineTitle;
            P4_Calc["Calculate Basic Metrics (Pubs, Cites, Collabs, Intl.)"]:::action;
            P4_Method[Temporal Analysis<br>Before/After Grant]:::methodology;
            P4_Out[Basic Metrics Dataset]:::data;

            P3_Out -- Matched Pairs --> P4_Calc;
            P1_Out -- GtR Grant Info --> P4_Calc;
            P2_Out -- OA Details --> P4_Calc;
            P4_Calc -- Uses --> P4_Method;
            P4_Calc --> P4_Out;
    end

    subgraph P5 [5.Complex Metrics Analysis]
        direction TB
            P5_Title[data_analysis_complex_metrics]:::pipelineTitle;
            P5_Calc_Disrupt[Calculate Disruption Index<br>Wu & Yan Variant]:::action;
            P5_Calc_Diverse[Calculate Discipline Diversity<br>Leydesdorff Framework]:::action;
            P5_Method[Based on Sampled Pubs<br>+ SPECTER Embeddings]:::methodology;
            P5_Out[Complex Metrics Dataset]:::data;

            P4_Out -- Grant Timing --> P5_Calc_Disrupt & P5_Calc_Diverse;
            P2_Out -- Sampled OA Data/Topics --> P5_Calc_Disrupt & P5_Calc_Diverse;
            P5_Calc_Disrupt -- Uses --> P5_Method;
            P5_Calc_Diverse -- Uses --> P5_Method;
            P5_Calc_Disrupt --> P5_Out;
            P5_Calc_Diverse --> P5_Out;
    end

    subgraph Final Output
        direction LR # Changed direction
            Final_Combine[Combine Metrics]:::action;
            Final_Dataset[Final Analysis Dataset]:::outputNode;

            P4_Out -- Basic Metrics --> Final_Combine;
            P5_Out -- Complex Metrics --> Final_Combine;
            Final_Combine --> Final_Dataset;
    end

    P1 --> P2;
    P1 --> P3;
    P2 --> P3;
    P3 --> P4;
    P1 --> P4;
    P2 --> P4;
    P4 --> P5;
    P2 --> P5;
    P4 --> Final_Combine;
    P5 --> Final_Combine;

    style P1 font-size:28px;
    style P1_Title font-size:28px;
    style P1_Fetch fill:#d9ead3,stroke:#5b824d,stroke-width:1.5px,color:#333,font-size:28px;
    style P1_Out fill:#e2dcf7,stroke:#5a4f91,stroke-width:1.5px,color:#333,font-size:28px;
    style P2_Title font-size:28px;
    style P2_Fetch fill:#d9ead3,stroke:#5b824d,stroke-width:1.5px,color:#333,font-size:28px;
    style P2_Out fill:#e2dcf7,stroke:#5a4f91,stroke-width:1.5px,color:#333,font-size:28px;
    style P3_Title font-size:28px;
    style P3_Match fill:#d9ead3,stroke:#5b824d,stroke-width:1.5px,color:#333,font-size:28px;
    style P3_Method fill:#fceecb,stroke:#b48c34,stroke-width:1px,stroke-dasharray: 3 3,color:#444,shape:hexagon,font-size:28px;
    style P3_Out fill:#e2dcf7,stroke:#5a4f91,stroke-width:1.5px,color:#333,font-size:28px;
    style P4_Title font-size:28px;
    style P4_Calc fill:#d9ead3,stroke:#5b824d,stroke-width:1.5px,color:#333,font-size:28px;
    style P4_Method fill:#fceecb,stroke:#b48c34,stroke-width:1px,stroke-dasharray: 3 3,color:#444,shape:hexagon,font-size:28px;
    style P4_Out fill:#e2dcf7,stroke:#5a4f91,stroke-width:1.5px,color:#333,font-size:28px;
    style P5_Title font-size:28px;
    style P5_Calc_Disrupt fill:#d9ead3,stroke:#5b824d,stroke-width:1.5px,color:#333,font-size:28px;
    style P5_Calc_Diverse fill:#d9ead3,stroke:#5b824d,stroke-width:1.5px,color:#333,font-size:28px;
    style P5_Method fill:#fceecb,stroke:#b48c34,stroke-width:1px,stroke-dasharray: 3 3,color:#444,shape:hexagon,font-size:28px;
    style P5_Out fill:#e2dcf7,stroke:#5a4f91,stroke-width:1.5px,color:#333,font-size:28px;
    style Final_Combine fill:#d9ead3,stroke:#5b824d,stroke-width:1.5px,color:#333,font-size:28px;
    style Final_Dataset fill:#fff2cc,stroke:#c49b00,stroke-width:1.5px,color:#333,font-size:28px;
    style GTR_API fill:#cfe2f3,stroke:#333,stroke-width:1px,color:#333,font-size:28px;
    style OA_API fill:#cfe2f3,stroke:#333,stroke-width:1px,color:#333,font-size:28px;
```

## Installation

To set up the project environment, install the required dependencies:

```bash
pip install -r requirements.txt
```

Configuration settings for pipelines, APIs, and parameters are managed within the `conf/` directory.

## Pipelines Overview

The project is structured around five core Kedro pipelines that sequentially process data from initial collection through to complex metric computation:

### 1. Gateway to Research Data Collection (`data_collection_gtr`)
This pipeline fetches data primarily concerning research projects from the UKRI Gateway to Research (GtR) API. It extracts project details, research topics, subjects, funding information, and linked publications. The process includes robust error handling, retry mechanisms, and rate limiting.

### 2. OpenAlex Data Collection (`data_collection_oa`)
This pipeline collects complementary data from the OpenAlex API. It uses identifiers gathered from GtR (ORCID iDs, author names, publication DOIs) to fetch corresponding author profiles, publication records, and institutional details. Data retrieval is optimised through parallel processing and batching.

### 3. Author Disambiguation (`author_disambiguation`)
This pipeline matches researchers identified in GtR with their likely profiles in OpenAlex. It employs a machine learning model trained on features including name similarity, institutional affiliation, research topics, publication history, and author metadata. Strategies like SMOTE or class weighting address the inherent class imbalance. The model aims to link GtR individuals to their OpenAlex counterparts accurately.

### 4. Basic Metrics Analysis (`data_analysis_basic_metrics`)
This pipeline calculates fundamental bibliometric and career indicators for the matched researchers. It computes metrics such as publication counts, citation impact (using both publication-year and citation-year attribution), h-index, i10-index, academic age, international affiliations, and collaboration patterns. Metrics are often calculated separately for periods before and after the researcher's first recorded GtR grant, enabling temporal analysis.

### 5. Complex Metrics Analysis (`data_analysis_complex_metrics`)
This pipeline computes more advanced bibliometric indicators requiring significant computation:
*   **Disruption Index:** Implements the Wu & Yan (2019) variant to measure whether a paper's citations indicate a shift in the research direction (disruptive) or consolidation of existing work. Author-level indices are calculated as both unweighted and Field-Weighted Citation Impact (FWCI) weighted averages. Due to computational demands, this calculation uses a stratified sample of each author's publications.
*   **Discipline Diversity:** Following Leydesdorff et al. (2019), this calculates diversity based on the variety (breadth of topics), evenness (distribution across topics), and disparity (cognitive distance between topics, using SPECTER embeddings).

## Running the Pipelines

Each pipeline can be executed independently using Kedro commands:

```bash
# GtR data collection
kedro run --pipeline data_collection_gtr

# OpenAlex data collection
kedro run --pipeline data_collection_oa

# Author disambiguation
kedro run --pipeline author_disambiguation

# Basic metrics analysis
kedro run --pipeline data_analysis_basic_metrics

# Complex metrics analysis
kedro run --pipeline data_analysis_complex_metrics
```

You can also run the complete sequence of pipelines if needed, though this may take significant time depending on data volume and API constraints. Specific sub-parts of pipelines can often be run using tags (refer to individual pipeline READMEs for details).

## Dependencies

Key dependencies include:
- Python 3.8+
- Kedro
- pandas
- scikit-learn
- PyTorch (for SPECTER model embeddings)
- requests
- joblib
- openalex-api
- fuzzywuzzy
- python-Levenshtein

Please refer to `requirements.txt` and `pyproject.toml` for a complete list of dependencies and version specifications.

## References

- Wu, L., Wang, D., & Evans, J. A. (2019). Large teams develop and small teams disrupt science and technology. *Nature*, *566*(7744), 378-382. ([arXiv version](https://doi.org/10.48550/arXiv.1905.03461))
- Leydesdorff, L., Wagner, C. S., & Bornmann, L. (2019). The European Union challenge for innovation studies: A decomposition analysis of the Rate of Return on Investment (ROI) in the case of the Framework Programmes (FP6 and FP7). *Journal of Informetrics*, *13*(1), 339-350. ([https://doi.org/10.1016/j.joi.2018.12.006](https://doi.org/10.1016/j.joi.2018.12.006))
- Leibel, E., & Bornmann, L. (2023). Disruption indices: A critical examination and prospects for a research program. *Quantitative Science Studies*, *4*(3), 519-541. ([arXiv version](https://doi.org/10.48550/arXiv.2308.02383))

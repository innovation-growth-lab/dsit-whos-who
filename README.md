# Who's Who in UK Research

## Connecting funded projects with authorship data for new metrics

### Overview

This project aims to create data suitable for mapping the impacts of UKRI-funded research at the person-level. We develop a methodology to disambiguate and link persons named in UKRI-funded projects listed in the Gateway to Research (GtR) database with open publication authorship data. Additionally, we outline metrics that can be used to understand the characteristics of funding recipients and their research outputs.

### Project Objectives

1. Develop a robust person disambiguation and linking methodology
2. Create a linked dataset connecting GtR person IDs with OpenAlex and ORCID IDs
3. Implement and evaluate metrics for studying the research impact at the person level
4. Deliver scalable, reusable code following open-source principles

### Methodology

#### Person Disambiguation and Linking

Our approach to linking and disambiguation leverages multiple data points:

- **Initial Matching**: For each person in GtR with a unique ID, we perform author searches using OpenAlex's API to generate match candidates
- **Filtering Criteria**:
  - Research topic overlap between OpenAlex author topics and GtR project descriptions
  - Institutional affiliation matching between GtR and OpenAlex records
  - Time window restrictions for considering institutional affiliations
- **Refinement Process**:
  - If no matches are found, we relax topic matching requirements
  - If multiple equivalent matches are found, we tighten criteria by comparing semantic similarity of publication abstracts to project descriptions

#### Evaluation

- Creation of a gold standard dataset using ORCID IDs present in both GtR and OpenAlex
- Implementation of model accuracy metrics to monitor performance
- Validation across research domains and publication activity levels

### Metrics Development

#### Basic Metrics
- Number of papers and citations by author
- Number of collaborators by author
- Other simple aggregations of available metadata

#### Advanced Metrics
We will implement 2-3 of the following advanced metrics:

1. **Disciplinary Diversity**
   - Variety of disciplines in which authors publish
   - Balance between publishing behavior across disciplines
   - Disparity between disciplines

2. **Collaborator Centrality Measures**
   - Network centrality measures to assess individuals' positions in collaboration networks

3. **UKRI Reliance**
   - Estimation of the fraction of publications linked to UKRI funding compared to other sources

4. **Consolidation-Disruption Index**
   - Variant of the CD index to compare distribution of disruptive research

5. **Research Alignment**
   - Semantic similarity between project and publication abstracts to assess adherence to original research plans

### Expected Outcomes

- **Linked Dataset**: A dataset linking person IDs in GtR data to OpenAlex IDs and ORCID IDs
- **Scalable Code**: Python-written, user-friendly code following Open Source principles and Nesta's code writing guidelines
- **Documentation**: Accessible notebooks detailing methodologies, code functionalities, and troubleshooting tips

### Project Timeline

- Project Start: June 25, 2024
- [Additional timeline milestones to be determined]

### Contributors

[To be added]

### License

[To be determined]

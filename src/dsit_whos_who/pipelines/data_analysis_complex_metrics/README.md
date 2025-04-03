# Complex Metrics

This module implements the calculation of complex bibliometric indicators, specifically focusing on discipline diversity and research disruption. The pipeline processes publication data to generate author-level metrics over time.

## Pipeline Structure

The pipeline consists of several sequential sub-pipelines:

1. **Sample Pipeline**
   - Performs stratified sampling of papers per author
   - Samples up to 50 papers per author, distributed across publication years
   - Uses FWCI quantiles to weight sampling toward higher-impact papers

2. **Focal Collection Pipeline**
   - Creates lists of sampled paper IDs for API queries
   - Fetches citation data for sampled papers from OpenAlex
   - Processes and cleans citation data for disruption index calculation

3. **Reference Collection Pipeline**
   - Extracts and processes reference works from sampled papers
   - Fetches metadata for referenced papers from OpenAlex
   - Prunes references to remove highly-cited papers (top 3%)

4. **Disruption Index Pipeline**
   - Calculates disruption indices for sampled papers
   - Processes citation networks using efficient set operations
   - Computes paper-level and author-level disruption metrics

5. **Discipline Diversity Pipeline**
   - Computes subfield embeddings using SPECTER model
   - Creates author-level topic aggregates over time
   - Calculates variety, evenness, and disparity metrics

6. **Metric Computation Pipeline**
   - Combines disruption and diversity metrics
   - Computes before/after funding comparisons
   - Merges with basic bibliometric indicators

## Methodological Approach

### Disruption Index

We implement the Wu and Yan (2019) variant of the disruption index, which measures how a paper affects the direction of subsequent research. For each focal paper i, the disruption index is calculated as:

$$
DI = \frac{n_f - n_b}{n_f + n_b}
$$

Where:
- $n_f$: number of papers citing the focal paper but not its references
- $n_b$: number of papers citing both the focal paper and its references

The index ranges from -1 to 1:
- Values closer to 1 indicate disruptive papers that redirect research
- Values closer to -1 indicate consolidating papers that reinforce existing research paths
- Values near 0 suggest papers with balanced influence

When aggregating disruption indices to the author level, we compute both unweighted and FWCI-weighted averages:

1. **Unweighted Average**: Simple mean of disruption indices across all papers by an author
2. **FWCI-Weighted Average**: Mean weighted by each paper's Field-Weighted Citation Impact

The FWCI weighting helps account for field-specific citation patterns and gives more importance to papers with higher relative impact. For example:
- A paper with FWCI=2 (cited twice as much as expected) gets twice the weight of a paper with FWCI=1
- Papers with no citations (FWCI=NaN) receive a minimum weight of 0.01 to avoid complete exclusion
- This weighting ensures that highly impactful papers (relative to their field) have more influence on the author's overall disruption score

The Wu and Yan (2019) disruption index was selected for two key reasons:

1. **Technical feasibility**: The formula avoids the need to crawl through forward citations in OpenAlex, which would require millions of API calls and hit rate limits. Instead, it uses available reference work data from the focal papers, as well as reference work data from papers citing the focal papers (but ignores other works citing focal papers' references but not citing the focal paper).

2. **Empirical validation**: Recent research by Leibel and Bornmann (2023) demonstrates the index performs remarkably well when compared against manually labelled data of disruptive papers. Their analysis (Table 5) shows strong correlation between the Wu-Yan index and expert assessments of research disruption, and performs better than the more complex index that considers $n_r$. 

### Discipline Diversity

The discipline diversity measurement follows Leydesdorff, Wagner, and Bornmann (2019), incorporating three components:

1. **Variety**: Proportion of unique topics an author has published on
   $$
   \text{Variety} = \frac{nc}{N}
   $$
   - Normalised by the total number of possible topics
   - Range: [0,1]


2. **Evenness**: Distribution balance across topics using the Kvålseth-Jost measure
   $$
   \text{Evenness} = 1 - \text{Gini}
   $$
   - Accounts for publication frequency in each topic
   - Range: [0,1], where 1 indicates perfectly even distribution

3. **Disparity**: Average cognitive distance between published topics
   $$
   \text{Disparity} = \frac{\sum d_{ij}}{nc \times (nc - 1)}
   $$
   - Computed using semantic embeddings from the SPECTER model
   - Based on pairwise distances in the embedding space

The overall diversity metric is then calculated as the product of these three components:

$$
\text{Diversity} = \text{Variety} \times \text{Evenness} \times \text{Disparity}
$$

This multiplicative approach ensures that high diversity requires good performance across all three dimensions, as proposed by Leydesdorff et al.

## Output Variables

The pipeline produces two main types of metrics: annual time series data and aggregated before/after metrics relative to an author's first funding year.

### Annual Time Series Data

For each author, we maintain detailed yearly time series stored in nested lists:

**Disruption Metrics (`author_year_disruption`):**
- Year
- Simple mean disruption index (unweighted average of paper-level disruption indices)
- FWCI-weighted mean disruption index (weighted by field-weighted citation impact)

**Diversity Metrics (`author_year_diversity`):**
- Year
- Variety (proportion of unique topics covered, range [0,1])
- Evenness (balance of publication distribution using Kvålseth-Jost measure, range [0,1])
- Disparity (average cognitive distance between topics using SPECTER embeddings)

### Before/After Funding Metrics

For each author, we compute aggregate metrics before and after their first funding year:

**Disruption Metrics:**
- `mean_disruption_before`/`mean_disruption_after`: Average of unweighted disruption indices
- `mean_weighted_disruption_before`/`mean_weighted_disruption_after`: Average of FWCI-weighted disruption indices

**Diversity Metrics:**
- `mean_variety_before`/`mean_variety_after`: Average proportion of unique topics covered
- `mean_evenness_before`/`mean_evenness_after`: Average balance in topic distribution
- `mean_disparity_before`/`mean_disparity_after`: Average cognitive distance between topics

All metrics are computed at the author level and merged with basic bibliometric indicators (e.g., publication counts, citation metrics) to provide a comprehensive view of research impact and evolution over time.

## Implementation Details

### Author Sampling Strategy
The pipeline uses a sophisticated sampling approach to select papers for disruption index calculation:
1. Papers are stratified by publication year to ensure temporal coverage
2. Within each year stratum, papers are sampled with probability proportional to their FWCI quantile
3. Up to 50 papers are sampled per author, distributed evenly across years
4. Papers without references are excluded as they cannot contribute to disruption indices

### Technical Implementation Notes
- All before/after metrics are computed relative to an author's first funding year
- Missing values (NaN) are used when there is insufficient data for a period
- Annual metrics are stored as nested lists to maintain the complete time series
- FWCI-weighted disruption indices use a minimum weight of 0.01 for papers with no citations
- Semantic embeddings are generated using the allenai-specter model

## Running Pipelines

Run the complete pipeline using:
```bash
kedro run --pipeline data_analysis_complex_metrics
```

Run specific sub-pipelines using tags:
```bash
# Run only the sample collection
kedro run --pipeline data_analysis_complex_metrics --tags collect_sample

# Run only the disruption index calculation
kedro run --pipeline data_analysis_complex_metrics --tags disruption_index_pipeline

# Run only the discipline diversity calculation
kedro run --pipeline data_analysis_complex_metrics --tags discipline_diversity_pipeline
```

## References

- Wu & Yan (2019). [https://doi.org/10.48550/arXiv.1905.03461]
- Leydesdorff, Wagner, & Bornmann (2019) [https://doi.org/10.1016/j.joi.2018.12.006]
- Leibel & Bornmann (2023) [https://doi.org/10.48550/arXiv.2308.02383]
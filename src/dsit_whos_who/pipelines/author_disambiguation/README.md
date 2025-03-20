# Author disambiguation pipeline

The **Author disambiguation pipeline** matches researchers between Gateway to Research (GtR) and OpenAlex using a machine learning approach. The pipeline implements both SMOTE and class weights strategies to handle the significant class imbalance in the training data.

## Pipeline overview

### 1. Data collection and preprocessing
- Aggregates comprehensive author information from GtR:
  - Personal details (name, ORCID)
  - Institutional affiliations
  - Project participation
  - Research topics
  - Publication records
- Processes OpenAlex author candidates:
  - Institutional affiliations
  - Publication metrics
  - Name variants
  - GB affiliation indicators

### 2. Candidate generation
- Initial matching based on name similarity
- ORCID-based validation for training set
- Handles multiple name variants and institutional changes
- Processes ~5.7M potential author pairs

### 3. Feature engineering

The pipeline computes five categories of features for each author pair:

#### 1. Name similarity features
- **Direct name comparisons**
  - `display_lev`: Levenshtein similarity between GTR and OA display names
  - `display_jw`: Jaro-Winkler similarity between names
  - `display_token`: Token set ratio between names
- **Name component matches**
  - `surname_match`: Exact surname match (binary)
  - `first_initial_match`: First initial match (binary)
  - `full_first_match`: Full first name match (binary)
- **Alternative name comparisons**
  - `alt_lev_mean/max`: Mean/Max Levenshtein similarity with alternative names
  - `alt_jw_mean/max`: Mean/Max Jaro-Winkler with alternatives
  - `alt_token_mean/max`: Mean/Max token set ratio with alternatives

#### 2. Institution features
- **Direct institution comparisons**
  - `inst_jw_max`: Maximum Jaro-Winkler similarity between institutions
  - `inst_token_max`: Maximum token set ratio between institutions
- **Associated institution metrics**
  - `inst_child_jw_max`: Max Jaro-Winkler with associated institutions
  - `inst_child_token_max`: Max token ratio with associated institutions
- **GB affiliation indicators**
  - `gb_affiliation_proportion`: Proportion of GB affiliations
  - `has_gb_affiliation`: Has any GB affiliation (binary)
  - `has_gb_associated`: Has GB associated institution (binary)

#### 3. Topic similarity features
Computed at four taxonomic levels (domain, field, subfield, topic):
- **Overlap metrics** (for each level)
  - `{level}_jaccard`: Jaccard similarity between topic sets
  - `{level}_cosine`: Cosine similarity between normalised topic vectors
  - `{level}_js_div`: Jensen-Shannon divergence between distributions
  - `{level}_containment`: Topic containment ratio

#### 4. Publication features
- **Coverage metrics**
  - `publication_coverage`: Project publications with candidate author / total project publications
  - `author_proportion`: Project publications with candidate author / total author counts

#### 5. Author metadata features
- **Publication metrics**
  - `works_count`: Total number of works
  - `cited_by_count`: Total citation count
  - `h_index`: Author's h-index
  - `i10_index`: Author's i10-index
  - `citations_per_work`: Average citations per work

### 4. Model training

#### Dataset characteristics
- Training set: 538,049 pairs
  - Positive class (matches): 13,040 (2.4%)
  - Negative class (non-matches): 525,009 (97.6%)
- Test set: 59,784 pairs
  - Positive class (matches): 1,449 (2.4%)
  - Negative class (non-matches): 58,335 (97.6%)
#### Model performance at optimal thresholds

##### SMOTE model (Optimal threshold = 0.70)
Test set confusion matrix:
| true\pred | negative | positive |
|-----------|----------|----------|
| negative | 58,243 | 92 |
| positive | 113 | 1,336 |

Performance metrics:
- Accuracy: 0.997
- Precision: 0.936
- Recall: 0.922
- F1: 0.929
- Balanced F1: 0.959

##### Class weights model (Optimal threshold = 0.80) 
Test set confusion matrix:
| true\pred | negative | positive |
|-----------|----------|----------|
| negative | 58,217 | 118 |
| positive | 92 | 1,357 |

Performance metrics:
- Accuracy: 0.996
- Precision: 0.920
- Recall: 0.937
- F1: 0.928
- Balanced F1: 0.966

### 5. Production predictions

#### Prediction results
- Total candidate pairs evaluated: 5,675,026
- Unique GtR IDs processed: 126,304
- Matches found (threshold 0.80): 85,444
- Final matched author pairs: 83,091
- Match rate: 67.6% of GtR authors matched

#### Threshold selection
Based on a handful thresholds, we select 0.80 for production to balance precision and recall:

#### SMOTE model - Test set performance

| Threshold | Accuracy | Precision | Recall | F1 | Balanced F1 |
|-----------|----------|-----------|--------|-----|-------------|
| 0.10 | 0.990 | 0.714 | 0.973 | 0.824 | 0.982 |
| 0.20 | 0.993 | 0.799 | 0.965 | 0.874 | 0.979 |
| 0.30 | 0.994 | 0.829 | 0.956 | 0.888 | 0.975 |
| 0.40 | 0.995 | 0.856 | 0.944 | 0.898 | 0.969 |
| 0.50 | 0.996 | 0.885 | 0.936 | 0.910 | 0.965 |
| 0.60 | 0.997 | 0.927 | 0.929 | 0.928 | 0.962 |
| 0.70 | 0.997 | 0.936 | 0.922 | 0.929 | 0.959 |
| 0.80 | 0.997 | 0.953 | 0.905 | 0.928 | 0.949 |
| 0.90 | 0.996 | 0.963 | 0.870 | 0.914 | 0.930 |
| 0.95 | 0.995 | 0.973 | 0.831 | 0.896 | 0.907 |
| 0.99 | 0.992 | 0.987 | 0.676 | 0.803 | 0.807 |

#### Class weights model - Test set performance

| Threshold | Accuracy | Precision | Recall | F1 | Balanced F1 |
|-----------|----------|-----------|--------|-----|-------------|
| 0.10 | 0.988 | 0.670 | 0.978 | 0.795 | 0.983 |
| 0.20 | 0.991 | 0.750 | 0.972 | 0.847 | 0.982 |
| 0.30 | 0.993 | 0.790 | 0.968 | 0.870 | 0.981 |
| 0.40 | 0.994 | 0.819 | 0.962 | 0.885 | 0.978 |
| 0.50 | 0.994 | 0.839 | 0.955 | 0.893 | 0.975 |
| 0.60 | 0.996 | 0.886 | 0.952 | 0.918 | 0.974 |
| 0.70 | 0.996 | 0.899 | 0.943 | 0.921 | 0.970 |
| 0.80 | 0.996 | 0.920 | 0.937 | 0.928 | 0.966 |
| 0.90 | 0.997 | 0.936 | 0.923 | 0.929 | 0.959 |
| 0.95 | 0.997 | 0.956 | 0.905 | 0.929 | 0.949 |
| 0.99 | 0.996 | 0.971 | 0.849 | 0.906 | 0.918 |

#### Feature importance rankings

| Feature - SMOTE | Importance | Feature - Class W. | Importance |
|---------|------------|---------|------------|
| inst_token_max | 0.3887 | alt_lev_max | 0.4787 |
| alt_lev_max | 0.3499 | inst_token_max | 0.3074 |
| author_proportion | 0.0520 | publication_coverage | 0.0269 |
| publication_coverage | 0.0426 | alt_jw_max | 0.0243 |
| inst_jw_max | 0.0263 | inst_jw_max | 0.0202 |
| display_lev | 0.0166 | topic_containment | 0.0168 |
| subfield_containment | 0.0116 | subfield_containment | 0.0149 |
| topic_js_div | 0.0103 | author_proportion | 0.0136 |
| alt_jw_max | 0.0098 | full_first_match | 0.0079 |
| surname_match | 0.0096 | topic_js_div | 0.0070 |

Key observations:
1. Both models heavily rely on institutional affiliation matching (`inst_token_max`) and name similarity (`alt_lev_max`)
2. Publication overlap metrics (`author_proportion`, `publication_coverage`) are moderately important
3. Topic similarity features have relatively low importance
4. Citation metrics (h-index, i10-index) are among the least important features

### 6. Coverage analysis

#### Overall coverage statistics
- Total GtR persons: 140,245
- Matchable persons (at least one name candidate): 126,304
- Matched persons: 85,444
- Overall coverage rate: 60.9%
- Coverage of matchable persons: 67.6%
- Coverage of active researchers (with projects): 68,329/99,945 (68.4%)

#### Coverage by grant category
Highest coverage rates:
- Fellowship: 82.9% (7,710/9,303)
- Research and Innovation: 82.8% (3,339/4,034)
- Intramural: 82.8% (1,330/1,606)
- Institute Project: 79.0% (595/753)
- Research Grant: 79.0% (55,881/70,748)
- Training Grant: 70.7% (5,285/7,477)
- Studentship: 56.3% (33,829/60,074)
- EU-Funded: 47.9% (1,062/2,218)

Notable categories with lower coverage:
- Collaborative R&D: 13.7% (1,088/7,921)
- Feasibility Studies: 15.1% (556/3,689)
- Small Business Research Initiative: 16.9% (136/805)

#### Temporal coverage analysis
1. Higher coverage rates for academic research grants compared to industry-focused schemes
2. Strong performance in matching established researchers with longer track records
3. Declining coverage for recent years, possibly due to:
   - Publication lag in OpenAlex
   - Less established publication records for newer researchers
   - Ongoing project updates in GtR

## Pipeline components

### Key nodes
1. **`aggregate_person_information`**
   - Consolidates author information from GtR
   - Processes project participation and research topics
   - Links publications across databases

2. **`preprocess_oa_candidates`**
   - Processes OpenAlex author candidates
   - Handles institutional affiliations
   - Computes GB affiliation indicators

3. **`merge_candidates_with_gtr`**
   - Matches authors using name similarity
   - Creates training dataset with ORCID validation
   - Handles batched processing for large datasets

4. **`create_feature_matrix`**
   - Engineers features for author matching
   - Computes similarity metrics:
     - Name variations
     - Institutional affiliations
     - Research topics
     - Publication patterns

5. **`train_disambiguation_model`**
   - Implements both SMOTE and class weights approaches
   - Handles cross-validation and hyperparameter tuning
   - Logs model metrics and feature importance

6. **`predict_author_matches`**
   - Applies trained model to new author pairs
   - Uses probability threshold of 0.80
   - Returns highest confidence match per GtR author

## Configuration

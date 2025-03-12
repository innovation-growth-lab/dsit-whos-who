# Plan for Person Disambiguation and Linking

Based on your project context and current situation, I'll outline a comprehensive plan for connecting GtR persons to OpenAlex author IDs. Let's break this down into phases:

## Phase 1: Data Collection and Preparation (Already in Progress)

- ✅ Collect GtR projects with person links (PI_PER, COI_PER, etc.)
- ✅ Extract person IDs and their roles in projects
- ✅ Collect GtR publications linked to projects
- ✅ Collect GtR person information

# Refined Person Disambiguation Plan

## Data Collection
- Extract GtR persons with roles from projects
- Collect person metadata (name, ORCID, institutions)
- Link projects to publications

## Matching Strategy

### 1. ORCID Direct Matching
- Match GtR persons with ORCIDs to OpenAlex authors
- Create gold standard dataset for training/validation

### 2. Feature-Based Matching for Others
For each candidate pair (GtR person, OpenAlex author):

#### Feature Extraction
- **Name Similarity**:
  - Levenshtein distance (normalized)
  - Jaro-Winkler similarity
  - Exact match on last name + first initial

- **Institutional Overlap**:
  - Max similarity between GtR and OpenAlex institutions
  - Mean similarity across all institution pairs
  - Binary match indicator (any/none)

- **Topic Similarity**:
  - Overlap between GtR project topics and OpenAlex author topics
  - Hierarchical weighting (level 1 topics > level 2 topics)
  - Cosine similarity of topic vectors

- **Publication Evidence**:
  - Ratio: (matched publications with author) / (total GtR publications)
  - Binary indicator of any publication match

### 3. Model-Based Ranking
- Train XGBoost classifier on gold standard (ORCID matches)
- Features: all similarity metrics above
- Output: probability of correct match
- Rank candidates by probability

### 4. Confidence Thresholds
- High confidence (>0.8): Accept match
- Medium (0.5-0.8): Accept if top candidate significantly outranks others
- Low (<0.5): Flag for manual review or leave unmatched

## Evaluation
- Precision/recall on held-out ORCID matches
- Manual validation of sample across confidence levels
- Confusion matrix analysis for error patterns

# To dos
- [X] Get ORCID people
- [X] Search based on name 
- [ ] For each candidate (person ID in GtR data), create a series of similarity values, based on:
    - [ ] Name distance (mean, max)
    - [ ] Topic overlap (mean, across top two levels)
    - [ ] Institution (mean, max)
    - [ ] Past activity (optional, based on past projects with the "person" having associated research papers where the candidate name also appears (after matching paper to OA db))
        - [ ] Number of instances where a past project with a publication has the candidate in the publication's OA data / number of publications identified through past projects who should "bore" the person's name
- [ ] Majority rule
- [ ] XGBoost (yes, because we already have the candidates!)

```yaml
    0000-0003-1376-8409:
      "display_name": "Adrian L. Harris",
      "display_name_alternatives": [
        "A. L Harris",
        "A. L. Harris",
        "Adrian. Harris",
        "Austin Harris",
        "AdrianL. Harris",
        "Adrian L. Harris",
        "AL. Harris",
        "Al Harris",
        "Alex L. Harris",
        "and Adrian L. Harris",
        "A. Harris",
        "Harris Al",
        "Harris"
      ]

    0000-0002-5304-9372:
      "display_name": "I. W. Harry",
      "display_name_alternatives": [
        "Ian W. Harry",
        "G. Harry",
        "Ian William Harry",
        "Gregory Michael Harry",
        "I. W. Harry",
        "I. Harry",
        "Ian Harry",
        "Gregory Harry",
        "I. W Harry",
        "G. M. Harry",
        "Gregory M. Harry",
        "Gregg Harry",
        "G. M Harry"
      ]
```

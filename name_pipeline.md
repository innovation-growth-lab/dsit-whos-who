# Author Disambiguation and Linking: Methodology

## Overview
Links Gateway to Research (GtR) authors to OpenAlex profiles using supervised machine learning with features derived from names, topics, institutions, publications, and author metrics.

## Data Dimensions and Features

### 1. Name Features
- Levenshtein distance between GtR name and OA display name
- Jaro-Winkler similarity between GtR name and OA display name
- Token set ratio between GtR name and OA display name
- Binary surname match
- Binary first initial match
- Binary full first name match
- Mean and max Levenshtein scores across alternative names
- Mean and max Jaro-Winkler scores across alternative names
- Mean and max token set ratios across alternative names

### 2. Topic Features
Computed at each level (domain, field, subfield, topic):
- Jaccard similarity between topic sets
- Cosine similarity between normalized topic frequency vectors
- Jensen-Shannon divergence between topic distributions
- Topic containment ratio (proportion of GtR topics in OA topics)

### 3. Institution Features
- Maximum Jaro-Winkler similarity between GtR and OA institutions
- Maximum token set ratio between GtR and OA institutions
- Maximum Jaro-Winkler similarity with associated institutions
- Maximum token set ratio with associated institutions
- Proportion of GB affiliations
- Binary indicator for any GB affiliation
- Binary indicator for any GB associated institution

### 4. Publication Features
- Publication coverage (author's publications / total project publications)
- Author proportion (author's publication count / sum of all author counts)
- Normalized proportion (average contribution per publication)

### 5. Metadata Features
- Total works count
- Total citation count
- H-index
- i10-index
- Citations per work ratio

## Training Data
- Uses ~16,800 authors with ORCID identifiers as ground truth
- Binary classification: 1 for correct match, 0 for incorrect match
- Validation on held-out ORCID-identified authors

## Technical Implementation
- Case-insensitive string comparisons throughout
- Topic similarities normalized by project/work counts
- Institution matching considers both direct and associated affiliations
- Publication overlap based on author contribution counts
- All features computed in batches with progress tracking



# OLd

### 1. Author Names

**Data Structure**
- GtR side: Single canonical name string
- OpenAlex side: Primary display name and array of name variants
```
Example OpenAlex entry:
Display name: "Adrian L. Harris"
Alternatives: ["A. L Harris", "A. L. Harris", "Adrian. Harris", "AL. Harris", ...]
```

**Similarity Metrics** (computed for all name variants, yielding mean and maximum)
1. Levenshtein distance (normalised)
2. Jaro-Winkler distance
3. Set token ratio
4. Binary surname match
5. Binary first initial match
6. Binary full first name match

### 2. Research Topics

**Data Structure**
- GtR side: Topics associated with research projects, hierarchically organised. Extracted from our prior work.
- OpenAlex side: Topics derived from publications, with counts and proportional shares
```
Example topic entry:
{
    "id": "T10184",
    "display_name": "Plant Molecular Biology Research",
    "count": 105,
    "subfield": {"id": "S1110", "display_name": "Plant Science"},
    "field": {"id": "F11", "display_name": "Agricultural and Biological Sciences"},
    "domain": {"id": "D1", "display_name": "Life Sciences"}
}
```

**Similarity Metrics** (computed at each taxonomic level: domain, field, subfield, topic)
1. Jaccard Similarity (intersection defined as minimum count, union as maximum count)
2. Cosine similarity (frequencies normalised by project count and publication count respectively)
3. Jensen-Shannon divergence
4. Topic coverage ratio (proportion of GtR topics present in candidate's publication topics) - Containment.

### 3. Institutional Affiliations

**Data Structure**
- GtR side: Primary institution name
- OpenAlex side: Array of institutional affiliations with temporal metadata
```
Example OpenAlex affiliation:
["I40120149", "University of Oxford", "GB", "funder", "2025,2024,2023,..."]
```

**Similarity Metrics**
1. Jaro-Winkler similarity
2. Set token ratio
3. Location-based match
4. (Optional) Institution child organisation JW/STR (max)

### 4. Publication Evidence

**Data Structure**
- GtR side: Publications linked to projects
- OpenAlex side: Author's publication record

**Metric**
1. Publication overlap ratio:
   - Numerator: Count of project publications where candidate appears as author
   - Denominator: Total count of publications in author's GtR projects

### (5.) Metadata
- works count, citations, i10 index, h index.

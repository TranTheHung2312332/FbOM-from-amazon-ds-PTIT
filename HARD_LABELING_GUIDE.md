# Hard Labeling Pipeline Guide

## Overview
This guide explains how to use the hard labeling pipeline for extracting aspect-opinion-sentiment (AOS) triplets from office product reviews.

## Pipeline Workflow

```
1. Load Data
   ↓
2. Sample Sentences (~1M per category)
   ↓
3. Generate Embeddings (sentence-transformers/all-MiniLM-L6-v2)
   ↓
4. Cluster into 100 clusters (K-means)
   ↓
5. Select 8 sentences per cluster
   ↓
6. Extract AOS Triplets (LLM-based)
   ↓
7. Validate & Generate Statistics
   ↓
8. Export Results
```

## Installation Requirements

### 1. Install Required Python Packages
```bash
pip install pandas numpy scikit-learn
pip install sentence-transformers
pip install anthropic  # For Claude API (optional, for LLM labeling)
```

### 2. Set Environment Variables (if using Claude)
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Data Format

### Input Data
- **Source**: `data/processed/Office_Products_Cleaned_v2.parquet`
- **Columns**: `parent_asin`, `review_id`, `sentence_id`, `sentence_text`, `rating`

### Output Data
- **Format**: Parquet files
- **Location**: `data/processed/labeled_data_{category}.parquet`
- **Columns**: 
  - `parent_asin`: Product category ID
  - `sentence_id`: Unique sentence identifier
  - `sentence_text`: The review sentence
  - `rating`: Product rating (1-5)
  - `triplets`: List of AOS triplets
  - `category_name`: Category name

## Triplet Format

Each triplet is a dictionary with three fields:

```json
{
  "aspect": "battery life",
  "opinion": "amazing",
  "sentiment": 2
}
```

**Sentiment Values**:
- `0` = Negative (poor, bad, terrible, etc.)
- `1` = Neutral (okay, fine, average, etc.)
- `2` = Positive (great, amazing, excellent, etc.)

## Output Reports

### Category-Specific Reports
- **Location**: `outputs/reports/hard_labeling_report_{category_name}.txt`
- **Contents**:
  - Dataset statistics (total sentences, coverage)
  - Triplet statistics
  - Sentiment distribution
  - Top 10 aspects and opinions

### Combined Report
- **Location**: `outputs/reports/hard_labeling_report_all_categories.txt`
- **Contents**: Overall statistics across all categories

## Report Statistics Explained

### Key Metrics
| Metric | Definition |
|--------|-----------|
| **Total sentences annotated** | Number of sentences selected for labeling |
| **Sentences with triplets** | Count of sentences with at least one AOS triplet |
| **Sentences without triplets** | Count of sentences with no valid triplets |
| **Coverage** | % of sentences with at least one triplet |
| **Total triplets** | Total count of all extracted triplets |
| **Avg triplets per sentence** | Average number of triplets per labeled sentence |
| **Sentences with multiple triplets** | Count of sentences with > 1 triplet |

### Sentiment Distribution
Shows the breakdown of extracted triplets by sentiment:
- **Negative**: Critical/negative opinions
- **Neutral**: Factual/neutral observations
- **Positive**: Positive/complimentary opinions

### Top Aspects & Opinions
Lists the most frequently mentioned aspects and opinions in the dataset, helping identify key product features discussed in reviews.

## Usage Example

### Running the Pipeline
1. Open `notebooks/hard_labeling_pipeline.ipynb` in Jupyter
2. Configure your ANTHROPIC_API_KEY (optional, for Claude LLM)
3. Run cells sequentially from top to bottom
4. Monitor progress messages and statistics at each step

### Processing Time Estimates
| Step | Approx. Time |
|------|-------------|
| Load data | < 1 minute |
| Sample sentences | < 5 minutes |
| Embedding generation | 10-30 minutes (depends on sentence count) |
| Clustering | 5-15 minutes |
| Sentence selection | < 1 minute |
| LLM labeling | 30 mins - 2 hours (depends on API rate limits) |
| Statistics & export | < 5 minutes |

**Total**: 1-3 hours depending on data size and API availability

## Advanced Options

### Customizing Clustering
Edit the notebook to change clustering parameters:
```python
NUM_CLUSTERS = 100  # Number of clusters
SENTENCES_PER_CLUSTER = 8  # Sentences per cluster
```

### Manual Annotation (Alternative to LLM)
Instead of using Claude LLM, you can manually annotate sentences:

1. Skip the LLM labeling cells
2. Use the provided annotation interface to label sentences
3. Store triplets in the format specified above
4. Continue with validation and export

### Using Different Embedding Models
To use a different model, modify:
```python
model = SentenceTransformer('your-model-name-here')
```

Available alternatives:
- `sentence-transformers/all-mpnet-base-v2` (larger, more accurate)
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingual)

## Quality Assurance

### Data Quality Checks
The pipeline includes validation for:
- Triplet format correctness
- Sentiment value constraints (0, 1, 2)
- Aspect and opinion non-empty strings
- Duplicate triplet detection (in statistics)

### Improving Quality
1. **Review LLM Outputs**: Sample extracted triplets to verify accuracy
2. **Adjust Prompts**: Modify the Claude prompt in cell 7 for different extraction criteria
3. **Manual Review**: Cross-check LLM results with human review on subset

## Troubleshooting

### Issue: Claude API Returns Errors
**Solution**: Check your `ANTHROPIC_API_KEY` and API rate limits

### Issue: Out of Memory During Embedding
**Solution**: Reduce batch size or process categories separately:
```python
batch_size = 16  # Reduce from 32
```

### Issue: Poor Clustering Results
**Solution**: Increase number of K-means initializations:
```python
kmeans = KMeans(n_clusters=NUM_CLUSTERS, n_init=20)
```

### Issue: No Triplets Extracted
**Solution**: 
1. Check sentence text quality
2. Verify Claude prompt is appropriate
3. Try manual annotation as alternative

## Contact & Support

For questions or issues with the pipeline:
1. Check generated reports for detailed statistics
2. Review notebook comments for detailed explanations
3. Adjust parameters and retry based on results

## Citation

If using this pipeline, please acknowledge:
- **Embedding Model**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Clustering**: scikit-learn K-means implementation
- **LLM**: Claude 3.5 Sonnet (Anthropic)

---

**Last Updated**: April 17, 2026
**Pipeline Version**: 1.0

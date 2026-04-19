"""
Utility functions for working with hard-labeled triplet data.

This module provides helper functions for:
- Loading and validating labeled datasets
- Analyzing triplet statistics
- Exporting data in different formats
- Quality assurance checks
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict


def load_labeled_data(filepath: str) -> pd.DataFrame:
    """Load labeled data from parquet file.
    
    Args:
        filepath: Path to the parquet file
        
    Returns:
        DataFrame with labeled triplets
    """
    df = pd.read_parquet(filepath)
    
    # Convert JSON strings to list of dicts if needed
    if isinstance(df['triplets'].iloc[0], str):
        df['triplets'] = df['triplets'].apply(json.loads)
    
    return df


def validate_triplets(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate all triplets in dataset.
    
    Args:
        df: DataFrame with triplets column
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'total_rows': len(df),
        'valid_rows': 0,
        'invalid_rows': 0,
        'errors': []
    }
    
    for idx, row in df.iterrows():
        triplets = row['triplets']
        
        if not isinstance(triplets, list):
            results['invalid_rows'] += 1
            results['errors'].append({
                'row': idx,
                'error': 'triplets is not a list',
                'value': str(triplets)[:100]
            })
            continue
        
        valid = True
        for triplet in triplets:
            if not isinstance(triplet, dict):
                valid = False
                break
            if not all(k in triplet for k in ['aspect', 'opinion', 'sentiment']):
                valid = False
                break
            if triplet['sentiment'] not in [0, 1, 2]:
                valid = False
                break
        
        if valid:
            results['valid_rows'] += 1
        else:
            results['invalid_rows'] += 1
    
    return results


def get_triplet_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive triplet statistics.
    
    Args:
        df: DataFrame with triplets column
        
    Returns:
        Dictionary with detailed statistics
    """
    stats = {
        'total_sentences': len(df),
        'sentences_with_triplets': 0,
        'sentences_without_triplets': 0,
        'total_triplets': 0,
        'sentiment_distribution': {0: 0, 1: 0, 2: 0},
        'aspect_frequency': Counter(),
        'opinion_frequency': Counter(),
        'sentences_with_multiple_triplets': 0,
        'triplets_per_sentence_distribution': defaultdict(int),
        'max_triplets_per_sentence': 0,
        'avg_triplets_per_sentence': 0.0
    }
    
    triplet_counts = [0]
    
    for triplets in df['triplets']:
        count = len(triplets) if isinstance(triplets, list) else 0
        triplet_counts.append(count)
        
        if count > 0:
            stats['sentences_with_triplets'] += 1
        else:
            stats['sentences_without_triplets'] += 1
        
        stats['total_triplets'] += count
        stats['triplets_per_sentence_distribution'][count] += 1
        
        if count > 1:
            stats['sentences_with_multiple_triplets'] += 1
        
        if count > stats['max_triplets_per_sentence']:
            stats['max_triplets_per_sentence'] = count
        
        for triplet in (triplets if isinstance(triplets, list) else []):
            if isinstance(triplet, dict):
                stats['sentiment_distribution'][triplet.get('sentiment', 1)] += 1
                stats['aspect_frequency'][triplet.get('aspect', 'unknown')] += 1
                stats['opinion_frequency'][triplet.get('opinion', 'unknown')] += 1
    
    # Convert counters to dicts
    stats['aspect_frequency'] = dict(stats['aspect_frequency'].most_common(20))
    stats['opinion_frequency'] = dict(stats['opinion_frequency'].most_common(20))
    stats['triplets_per_sentence_distribution'] = dict(stats['triplets_per_sentence_distribution'])
    
    if stats['sentences_with_triplets'] > 0:
        stats['avg_triplets_per_sentence'] = stats['total_triplets'] / stats['sentences_with_triplets']
    
    return stats


def export_to_json(df: pd.DataFrame, output_path: str, limit: int = None) -> None:
    """Export labeled data to JSON format.
    
    Args:
        df: DataFrame with triplets
        output_path: Path to save JSON file
        limit: Maximum number of rows to export (None for all)
    """
    if limit:
        df = df.head(limit)
    
    records = []
    for _, row in df.iterrows():
        record = {
            'parent_asin': row['parent_asin'],
            'sentence_id': row['sentence_id'],
            'sentence_text': row['sentence_text'],
            'rating': row['rating'],
            'triplets': row['triplets'] if isinstance(row['triplets'], list) else json.loads(row['triplets']),
            'category_name': row.get('category_name', 'unknown')
        }
        records.append(record)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    
    print(f"Exported {len(records)} records to {output_path}")


def export_to_csv(df: pd.DataFrame, output_path: str, limit: int = None) -> None:
    """Export labeled data to CSV format.
    
    Args:
        df: DataFrame with triplets
        output_path: Path to save CSV file
        limit: Maximum number of rows to export (None for all)
    """
    if limit:
        df = df.head(limit)
    
    df_export = df.copy()
    
    # Convert triplets to JSON string
    df_export['triplets'] = df_export['triplets'].apply(
        lambda x: json.dumps(x) if isinstance(x, list) else x
    )
    
    df_export.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Exported {len(df_export)} records to {output_path}")


def filter_by_sentiment(df: pd.DataFrame, sentiment: int) -> pd.DataFrame:
    """Filter sentences containing triplets with specific sentiment.
    
    Args:
        df: DataFrame with triplets
        sentiment: 0 (negative), 1 (neutral), or 2 (positive)
        
    Returns:
        Filtered DataFrame
    """
    def has_sentiment(triplets, target_sentiment):
        if not isinstance(triplets, list):
            return False
        return any(t.get('sentiment') == target_sentiment for t in triplets)
    
    return df[df['triplets'].apply(lambda x: has_sentiment(x, sentiment))]


def filter_by_aspect(df: pd.DataFrame, aspect: str, case_sensitive: bool = False) -> pd.DataFrame:
    """Filter sentences containing specific aspect.
    
    Args:
        df: DataFrame with triplets
        aspect: Aspect to search for
        case_sensitive: Whether to use case-sensitive matching
        
    Returns:
        Filtered DataFrame
    """
    def has_aspect(triplets, target_aspect, case_sensitive):
        if not isinstance(triplets, list):
            return False
        
        for t in triplets:
            a = t.get('aspect', '')
            if case_sensitive:
                if a == target_aspect:
                    return True
            else:
                if a.lower() == target_aspect.lower():
                    return True
        return False
    
    return df[df['triplets'].apply(lambda x: has_aspect(x, aspect, case_sensitive))]


def get_aspect_sentiment_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Create aspect-sentiment co-occurrence matrix.
    
    Args:
        df: DataFrame with triplets
        
    Returns:
        DataFrame with aspects as rows, sentiment as columns
    """
    matrix = defaultdict(lambda: {0: 0, 1: 0, 2: 0})
    
    for triplets in df['triplets']:
        if isinstance(triplets, list):
            for triplet in triplets:
                aspect = triplet.get('aspect', 'unknown')
                sentiment = triplet.get('sentiment', 1)
                matrix[aspect][sentiment] += 1
    
    return pd.DataFrame(matrix).fillna(0).astype(int)


def generate_qa_report(df: pd.DataFrame, output_path: str = None) -> Dict[str, Any]:
    """Generate quality assurance report.
    
    Args:
        df: DataFrame with triplets
        output_path: Optional path to save report as text file
        
    Returns:
        Dictionary with QA metrics
    """
    validation = validate_triplets(df)
    statistics = get_triplet_statistics(df)
    
    report = {
        'validation': validation,
        'statistics': statistics,
        'quality_metrics': {
            'data_completeness': 100 * validation['valid_rows'] / validation['total_rows'],
            'annotation_coverage': 100 * statistics['sentences_with_triplets'] / statistics['total_sentences'],
            'avg_triplets': statistics['avg_triplets_per_sentence'],
            'max_triplets': statistics['max_triplets_per_sentence']
        }
    }
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("TRIPLET DATA QUALITY ASSURANCE REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("VALIDATION RESULTS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total rows: {validation['total_rows']}\n")
            f.write(f"Valid rows: {validation['valid_rows']}\n")
            f.write(f"Invalid rows: {validation['invalid_rows']}\n")
            f.write(f"Completeness: {report['quality_metrics']['data_completeness']:.1f}%\n\n")
            
            f.write("TRIPLET STATISTICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total sentences: {statistics['total_sentences']}\n")
            f.write(f"Sentences with triplets: {statistics['sentences_with_triplets']}\n")
            f.write(f"Sentences without triplets: {statistics['sentences_without_triplets']}\n")
            f.write(f"Coverage: {report['quality_metrics']['annotation_coverage']:.1f}%\n")
            f.write(f"Total triplets: {statistics['total_triplets']}\n")
            f.write(f"Avg triplets per sentence: {statistics['avg_triplets_per_sentence']:.2f}\n")
            f.write(f"Max triplets per sentence: {statistics['max_triplets_per_sentence']}\n\n")
            
            f.write("SENTIMENT DISTRIBUTION\n")
            f.write("-" * 70 + "\n")
            sentiments = statistics['sentiment_distribution']
            total = sum(sentiments.values())
            f.write(f"Negative (0): {sentiments[0]} ({100*sentiments[0]/max(total,1):.1f}%)\n")
            f.write(f"Neutral (1):  {sentiments[1]} ({100*sentiments[1]/max(total,1):.1f}%)\n")
            f.write(f"Positive (2): {sentiments[2]} ({100*sentiments[2]/max(total,1):.1f}%)\n\n")
            
            f.write("TOP ASPECTS\n")
            f.write("-" * 70 + "\n")
            for aspect, count in list(statistics['aspect_frequency'].items())[:10]:
                f.write(f"  {aspect}: {count}\n")
        
        print(f"QA report saved to {output_path}")
    
    return report


if __name__ == "__main__":
    # Example usage
    print("Labeled data utilities module")
    print("Import and use functions from this module in your scripts")

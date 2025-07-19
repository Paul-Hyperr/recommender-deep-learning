#!/usr/bin/env python3
"""
Evaluate popularity baseline metrics for comparison with trained models.
Calculates Purchase Recall@10, Purchase Recall@20, Purchase MRR, and Item Coverage@20
using the most popular items as recommendations.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.session_processor2 import SessionProcessor2
from utils.package_processor import PackageProcessor
from utils.training_utils import (
    filter_by_min_session_length,
    filter_items_by_frequency,
    time_based_split_year
)


def calculate_popularity_metrics(test_samples, popular_items, total_valid_packages, k_values=[10, 20]):
    """
    Calculate metrics for popularity baseline
    
    Args:
        test_samples: List of test samples with purchase targets
        popular_items: List of most popular package IDs (ordered by popularity)
        total_valid_packages: Total number of valid packages in the filtered dataset
        k_values: List of k values for recall calculation
        
    Returns:
        Dict with metrics
    """
    metrics = {}
    
    # Filter only purchase samples for evaluation
    purchase_samples = [s for s in test_samples if s.get('is_purchase', False)]
    
    if not purchase_samples:
        print("Warning: No purchase samples found in test set!")
        return metrics
    
    print(f"Evaluating on {len(purchase_samples):,} purchase samples")
    
    # Calculate metrics for each k
    for k in k_values:
        top_k_items = popular_items[:k]
        
        # Calculate recall@k
        hits = 0
        for sample in purchase_samples:
            target = sample.get('purchased_package')
            if target in top_k_items:
                hits += 1
        
        recall = hits / len(purchase_samples)
        metrics[f'purchase_recall@{k}'] = recall
        print(f"  Purchase Recall@{k}: {recall*100:.2f}% ({hits}/{len(purchase_samples)})")
    
    # Calculate MRR (Mean Reciprocal Rank)
    reciprocal_ranks = []
    for sample in purchase_samples:
        target = sample.get('purchased_package')
        try:
            rank = popular_items.index(target) + 1  # 1-indexed
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            # Target not in popular items list
            reciprocal_ranks.append(0.0)
    
    mrr = np.mean(reciprocal_ranks)
    metrics['purchase_mrr'] = mrr
    print(f"  Purchase MRR: {mrr:.4f}")
    
    # Calculate Item Coverage@20 (how many unique items are recommended)
    # Coverage should be based on all valid packages, not just purchased ones
    coverage_at_20 = min(20, len(popular_items))
    coverage_ratio = coverage_at_20 / total_valid_packages if total_valid_packages > 0 else 0
    
    metrics['item_coverage@20'] = coverage_ratio
    print(f"  Item Coverage@20: {coverage_ratio*100:.2f}% ({coverage_at_20}/{total_valid_packages})")
    print(f"\n📊 Total valid packages in catalog: {total_valid_packages:,}")
    print(f"📊 Unique purchased packages in training: {len(popular_items):,}")
    
    return metrics


def get_popularity_ranking(train_samples):
    """
    Get popularity ranking based on purchase frequency in training data
    
    Args:
        train_samples: List of training samples
        
    Returns:
        List of package IDs ordered by popularity (most popular first)
    """
    # Count purchase frequencies
    purchase_counts = Counter()
    
    for sample in train_samples:
        if sample.get('is_purchase', False):
            package_id = sample.get('purchased_package')
            if package_id is not None:
                purchase_counts[package_id] += 1
    
    # Sort by count (descending)
    popular_items = [item for item, count in purchase_counts.most_common()]
    
    print(f"Found {len(popular_items):,} unique purchased items")
    if popular_items:
        top_item_count = purchase_counts[popular_items[0]]
        print(f"Most popular item purchased {top_item_count:,} times")
        
        # Show top 10
        print("Top 10 most popular items:")
        for i, (item, count) in enumerate(purchase_counts.most_common(10)):
            print(f"  {i+1}. Item {item}: {count} purchases")
    
    return popular_items


def evaluate_dataset(dataset_name, event_data_path):
    """
    Evaluate popularity baseline for a specific dataset
    
    Args:
        dataset_name: Name of dataset ('2months' or '13months')
        event_data_path: Path to event data file
        
    Returns:
        Dict with metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Popularity Baseline: {dataset_name}")
    print(f"Data: {event_data_path}")
    print(f"{'='*60}")
    
    # Data paths
    package_data_path = "data/feed.parquet"
    
    # Initialize processors
    print("Initializing data processors...")
    package_processor = PackageProcessor(
        feed_data_path=package_data_path,
        cache_dir='data/cache',
        load_coordinates=False,  # Faster loading
        load_embeddings=False,   # Not needed for popularity baseline
        use_reduced_embeddings=False
    )
    
    session_processor = SessionProcessor2(
        event_data_path=event_data_path,
        cache_dir='data/cache',
        session_timeout_hours=30,
        min_interactions=8,
        max_sessions_per_user=20,
        max_samples_per_user=10
    )
    
    # Load and process data
    print("Loading and processing data...")
    package_processor.load_data()
    package_processor.create_mappings()
    
    session_processor.load_data()
    session_processor.create_mappings()
    
    # Prepare training samples
    print("Preparing training samples...")
    training_samples = session_processor.prepare_enhanced_training_data()
    
    # Apply same filtering as in training scripts
    quality_samples = filter_by_min_session_length(training_samples, min_session_length=2)
    filtered_samples, valid_packages = filter_items_by_frequency(quality_samples, min_frequency=5)
    
    purchase_count = sum(1 for s in filtered_samples if s.get('is_purchase', False))
    print(f"After filtering: {len(filtered_samples):,} samples with {purchase_count:,} purchases")
    
    # Split data temporally (same as training scripts)
    print("Splitting data temporally...")
    train_samples, test_samples, split_date = time_based_split_year(
        filtered_samples, train_ratio=0.91
    )
    
    print(f"Train samples: {len(train_samples):,}")
    print(f"Test samples: {len(test_samples):,}")
    print(f"Split date: {split_date}")
    
    # Get popularity ranking from training data
    print("\nCalculating item popularity from training data...")
    popular_items = get_popularity_ranking(train_samples)
    
    # Evaluate popularity baseline
    print(f"\nEvaluating popularity baseline metrics...")
    total_valid_packages = len(valid_packages)
    metrics = calculate_popularity_metrics(test_samples, popular_items, total_valid_packages, k_values=[10, 20])
    
    # Add dataset info
    metrics['dataset'] = dataset_name
    metrics['event_data_path'] = event_data_path
    metrics['split_date'] = str(split_date)
    metrics['train_samples'] = len(train_samples)
    metrics['test_samples'] = len(test_samples)
    metrics['total_popular_items'] = len(popular_items)
    
    return metrics


def main():
    """Main function to evaluate popularity baselines for both datasets"""
    
    # Dataset configurations
    datasets = {
        '2months': 'data/bookit_events_2_months.parquet',
        '13months': 'data/13_months_new2_clean.parquet'
    }
    
    all_results = {}
    
    # Evaluate each dataset
    for dataset_name, event_data_path in datasets.items():
        # Check if file exists
        if not os.path.exists(event_data_path):
            print(f"⚠️  Skipping {dataset_name}: File not found - {event_data_path}")
            continue
        
        try:
            metrics = evaluate_dataset(dataset_name, event_data_path)
            all_results[dataset_name] = metrics
            
            print(f"\n✅ {dataset_name} Results:")
            print(f"   Purchase Recall@10: {metrics.get('purchase_recall@10', 0)*100:.2f}%")
            print(f"   Purchase Recall@20: {metrics.get('purchase_recall@20', 0)*100:.2f}%")
            print(f"   Purchase MRR: {metrics.get('purchase_mrr', 0):.4f}")
            print(f"   Item Coverage@20: {metrics.get('item_coverage@20', 0)*100:.2f}%")
            
        except Exception as e:
            print(f"❌ Error evaluating {dataset_name}: {e}")
            continue
    
    # Save results
    output_dir = 'output/popularity_baseline'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results as JSON
    results_file = os.path.join(output_dir, 'popularity_baseline_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Create summary table
    if all_results:
        print(f"\n{'='*80}")
        print("POPULARITY BASELINE SUMMARY")
        print(f"{'='*80}")
        
        # Create comparison table
        summary_data = []
        for dataset_name, metrics in all_results.items():
            summary_data.append({
                'Dataset': dataset_name,
                'Train Samples': f"{metrics.get('train_samples', 0):,}",
                'Test Samples': f"{metrics.get('test_samples', 0):,}",
                'Popular Items': f"{metrics.get('total_popular_items', 0):,}",
                'Recall@10 (%)': f"{metrics.get('purchase_recall@10', 0)*100:.2f}",
                'Recall@20 (%)': f"{metrics.get('purchase_recall@20', 0)*100:.2f}",
                'MRR': f"{metrics.get('purchase_mrr', 0):.4f}",
                'Coverage@20 (%)': f"{metrics.get('item_coverage@20', 0)*100:.2f}"
            })
        
        # Convert to DataFrame for nice formatting
        import pandas as pd
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
        
        # Save summary as CSV
        csv_file = os.path.join(output_dir, 'popularity_baseline_summary.csv')
        df.to_csv(csv_file, index=False)
        print(f"\n💾 Summary table saved to: {csv_file}")
    
    print(f"\n✅ Popularity baseline evaluation complete!")
    print(f"   Results saved in: {output_dir}/")


if __name__ == "__main__":
    main()
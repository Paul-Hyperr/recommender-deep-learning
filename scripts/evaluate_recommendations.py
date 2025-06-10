#!/usr/bin/env python3
"""
Evaluate NATR Recommendations - Consistent with Training Evaluation

This script evaluates recommendations from utils/recommendations.py using the same
methodology as training evaluation for consistent and reliable metrics.

Usage:
    # Basic evaluation
    python evaluate_recommendations.py --model-info output/model_info_2months/model_info.json
    
    # With availability filtering
    python evaluate_recommendations.py --model-info output/model_info_2months/model_info.json \
        --availability-data data/availability.parquet --min-available-days 5
        
    # Fast evaluation on sample
    python evaluate_recommendations.py --model-info output/model_info_2months/model_info.json \
        --sample-size 1000 --purchases-only
"""

import argparse
import json
import sys
import os
import numpy as np
import torch
import time
from typing import Dict, List, Optional
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from utils.recommendations import NATRRecommender
from utils.unified_metrics import UnifiedMetricsTracker
from utils.training_utils import (
    extract_test_events_from_parquet, create_dataloaders, 
    filter_by_min_session_length, filter_items_by_frequency,
    identify_event_types, analyze_data_distribution, time_based_split_year
)
from utils.session_processor import SessionProcessor
from utils.package_processor import PackageProcessor


class RecommendationEvaluator:
    """
    Recommendation evaluator that uses the same methodology as training evaluation
    to ensure consistent and comparable metrics.
    """
    
    def __init__(self, model_info_path: str, availability_data_path: Optional[str] = None, exclude_prev_purchases: bool = False, exclude_recent_months: Optional[int] = None):
        """
        Initialize evaluator
        
        Args:
            model_info_path: Path to model_info.json file
            availability_data_path: Optional path to availability data
            exclude_prev_purchases: Whether to exclude previously bought packages from recommendations
            exclude_recent_months: If set, only exclude purchases from the last N months instead of all history
        """
        self.model_info_path = model_info_path
        self.availability_data_path = availability_data_path
        self.exclude_prev_purchases = exclude_prev_purchases
        self.exclude_recent_months = exclude_recent_months
        
        # Load model info
        with open(model_info_path, 'r') as f:
            self.model_info = json.load(f)
        
        print(f"Evaluating model: {os.path.basename(model_info_path)}")
        print(f"Training strategy: {self.model_info.get('training_strategy', 'standard')}")
        
        # Get train ratio and split date
        self.train_ratio = self.model_info.get('train_ratio', 0.91)
        self.split_date = self.model_info.get('split_date', None)
        
        print(f"Using train ratio: {self.train_ratio} (test ratio: {1-self.train_ratio:.2f})")
        if self.split_date:
            print(f"Split date: {self.split_date}")
        
        # Initialize recommender
        print("\nInitializing recommender...")
        print(f"Exclude previous purchases: {self.exclude_prev_purchases}")
        self.recommender = NATRRecommender(
            model_info_path=model_info_path,
            availability_data_path=availability_data_path,
            exclude_prev_purchases=self.exclude_prev_purchases,
            exclude_recent_months=self.exclude_recent_months
        )
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() 
                                  else 'mps' if torch.backends.mps.is_available() 
                                  else 'cpu')
        self.recommender.model = self.recommender.model.to(self.device)
        print(f"Using device: {self.device}")
    
    def prepare_test_data(self, sample_size: Optional[int] = None, purchases_only: bool = False):
        """
        Prepare test data using the same methodology as training
        
        Args:
            sample_size: Optional limit on number of test samples
            purchases_only: If True, only evaluate on purchase samples
        """
        print("\nPreparing test data using training methodology...")
        
        # Data paths
        original_event_data = self.model_info.get('event_data_path', 'data/bookit_events_2_months.parquet')
        package_data_path = self.model_info.get('package_data_path', 'data/feed.parquet')
        
        # Initialize processors EXACTLY like training
        print("Initializing processors...")
        package_processor = PackageProcessor(
            feed_data_path=package_data_path,
            cache_dir='data/cache',
            load_coordinates=True,
            load_embeddings=True,
            api_key=os.environ.get("OPENAI_API_KEY"),
            embedding_model='text-embedding-3-small',
            use_reduced_embeddings=True
        )
        
        session_processor = SessionProcessor(
            event_data_path=original_event_data,
            cache_dir='data/cache',
            min_interactions=5,
            max_sessions_per_user=20,
            max_samples_per_user=10
        )
        
        # Load and process data exactly like training
        print("Loading data...")
        package_processor.load_data()
        session_processor.load_data()
        
        print("Creating mappings...")
        package_processor.create_mappings()
        session_processor.create_mappings()
        
        print("Extracting sessions...")
        session_processor.extract_sessions()
        
        print("Preparing samples...")
        all_samples = session_processor.prepare_enhanced_training_data()
        
        # Identify event types
        event_to_idx = session_processor.get_idx_mappings()['event_to_idx']
        all_samples = identify_event_types(all_samples, event_to_idx)
        
        # Apply same filters as training
        print("Applying training filters...")
        quality_samples = filter_by_min_session_length(all_samples, min_session_length=2)
        
        # Use valid packages from model if available
        valid_packages = set(self.model_info.get('valid_packages', []))
        if valid_packages:
            print(f"Using {len(valid_packages)} valid packages from training")
            filtered_samples = []
            for sample in quality_samples:
                if str(sample.get('purchased_package', '')) in valid_packages:
                    filtered_samples.append(sample)
            print(f"Samples after package filtering: {len(filtered_samples)}")
        else:
            filtered_samples, valid_packages = filter_items_by_frequency(quality_samples, min_frequency=5)
        
        # Time-based split using same ratio as training
        train_samples, test_samples, split_date = time_based_split_year(filtered_samples, train_ratio=self.train_ratio)
        
        print(f"Test samples: {len(test_samples):,}")
        
        # Filter to purchases only if requested
        if purchases_only:
            test_samples = [s for s in test_samples if s.get('is_purchase', False)]
            print(f"Purchase-only samples: {len(test_samples):,}")
        
        # Sample if requested
        if sample_size and sample_size < len(test_samples):
            if purchases_only:
                # Random sample of purchases
                import random
                random.seed(42)
                test_samples = random.sample(test_samples, sample_size)
            else:
                # Prioritize purchase samples in sampling
                purchase_samples = [s for s in test_samples if s.get('is_purchase', False)]
                non_purchase_samples = [s for s in test_samples if not s.get('is_purchase', False)]
                
                if len(purchase_samples) >= sample_size:
                    test_samples = purchase_samples[:sample_size]
                else:
                    remaining = sample_size - len(purchase_samples)
                    test_samples = purchase_samples + non_purchase_samples[:remaining]
            
            print(f"Sampled to {len(test_samples):,} test samples")
        
        # Analyze distribution
        analyze_data_distribution(test_samples, "Test")
        
        # Identify cold start users (users in test but not in train)
        train_users = set(sample['user_id'] for sample in train_samples)
        test_users = set(sample['user_id'] for sample in test_samples)
        cold_start_users = test_users - train_users
        
        print(f"\nCold start analysis:")
        print(f"  Train users: {len(train_users):,}")
        print(f"  Test users: {len(test_users):,}")
        print(f"  Cold start users: {len(cold_start_users):,} ({len(cold_start_users)/len(test_users)*100:.1f}%)")
        
        # Mark cold start samples in test set
        for sample in test_samples:
            sample['is_cold_start'] = sample['user_id'] in cold_start_users
        
        # Count cold start purchases
        cold_start_purchases = sum(1 for s in test_samples if s.get('is_cold_start', False) and s.get('is_purchase', False))
        warm_purchases = sum(1 for s in test_samples if not s.get('is_cold_start', False) and s.get('is_purchase', False))
        print(f"  Cold start purchases: {cold_start_purchases:,}")
        print(f"  Warm user purchases: {warm_purchases:,}")
        
        # Update user mappings to include all users
        all_users = set()
        for sample in train_samples + test_samples:
            all_users.add(sample['user_id'])
        
        unmapped_users = all_users - set(session_processor.user_to_idx.keys())
        if unmapped_users:
            print(f"Found {len(unmapped_users)} unmapped users, adding to mappings...")
            max_idx = max(session_processor.user_to_idx.values()) if session_processor.user_to_idx else 0
            for user_id in unmapped_users:
                max_idx += 1
                session_processor.user_to_idx[user_id] = max_idx
        
        # Create dataloader exactly like training
        print("Creating test dataloader...")
        dummy_train_samples = test_samples[:100] if len(test_samples) > 100 else test_samples
        
        train_loader, test_loader = create_dataloaders(
            train_samples=dummy_train_samples,
            test_samples=test_samples,
            package_processor=package_processor,
            session_processor=session_processor,
            batch_size=64,
            num_workers=0,
            use_weighted_sampling=False
        )
        
        print(f"Test dataloader: {len(test_loader.dataset):,} samples ({len(test_loader):,} batches)")
        
        return test_loader, test_samples
    
    def evaluate_batch_with_exclusion(self, test_loader, test_samples: List[Dict], k_values: List[int] = [10, 20, 50]) -> Dict:
        """
        Fast batch-based evaluation with purchase exclusion post-processing
        
        Args:
            test_loader: DataLoader for batch processing
            test_samples: List of test samples for exclusion info
            k_values: List of k values for recall@k metrics
            
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\nRunning fast batch evaluation with purchase exclusion (k={k_values})...")
        
        # First, do batch prediction to get all scores efficiently
        print("Step 1: Getting batch predictions...")
        all_predictions = []
        all_targets = []
        all_user_ids = []
        
        model = self.recommender.model
        model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Batch prediction")):
                # Move batch to device
                batch = self._move_batch_to_device(batch, self.device)
                
                # Forward pass
                outputs = model(batch)
                predictions = outputs['predictions']  # [batch_size, num_packages]
                targets = batch['purchased']['package_ids']  # [batch_size]
                user_ids = batch['user_id']  # [batch_size]
                
                # Store results
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
                all_user_ids.append(user_ids.cpu())
        
        # Combine all results
        all_predictions = torch.cat(all_predictions, dim=0)  # [total_samples, num_packages]
        all_targets = torch.cat(all_targets, dim=0)  # [total_samples]
        all_user_ids = torch.cat(all_user_ids, dim=0)  # [total_samples]
        
        print(f"Step 2: Post-processing with purchase exclusion for {len(all_predictions)} samples...")
        
        # Build user ID to sample mapping for fast lookup
        user_to_samples = {}
        for sample in test_samples:
            user_id = sample['user_id']
            if user_id not in user_to_samples:
                user_to_samples[user_id] = sample
        
        # Track exclusion statistics
        total_exclusions = 0
        exclusion_ranks = []
        
        # Process each prediction with purchase exclusion
        results = {f'recall@{k}': [] for k in k_values}
        results['mrr'] = []
        
        for i in tqdm(range(len(all_predictions)), desc="Processing exclusions"):
            user_id_tensor = all_user_ids[i].item()
            target_idx = all_targets[i].item()
            predictions = all_predictions[i]  # [num_packages]
            
            # Convert user_idx back to user_id string
            user_idx_to_id = {v: k for k, v in self.recommender.session_processor.user_to_idx.items()}
            user_id = user_idx_to_id.get(user_id_tensor, str(user_id_tensor))
            
            # Get user's previous purchases
            purchased_packages = self.recommender.get_user_purchase_history(user_id)
            
            if self.exclude_prev_purchases and purchased_packages:
                # Get package indices for purchased packages
                package_to_idx = self.recommender.session_processor.package_to_idx
                purchased_indices = []
                for pkg_id in purchased_packages:
                    pkg_idx = package_to_idx.get(str(pkg_id))
                    if pkg_idx is not None:
                        purchased_indices.append(pkg_idx)
                
                if purchased_indices:
                    # Get original ranks of purchased packages before exclusion
                    sorted_indices = torch.argsort(predictions, descending=True)
                    for pkg_idx in purchased_indices:
                        # Find rank of this purchased package (1-indexed)
                        rank_pos = (sorted_indices == pkg_idx).nonzero(as_tuple=True)[0]
                        if len(rank_pos) > 0:
                            rank = rank_pos[0].item() + 1
                            exclusion_ranks.append(rank)
                    
                    total_exclusions += len(purchased_indices)
                    
                    # Exclude purchased packages by setting their scores to -inf
                    predictions_excluded = predictions.clone()
                    predictions_excluded[purchased_indices] = float('-inf')
                    predictions = predictions_excluded
            
            # Calculate metrics with (potentially) excluded predictions
            sorted_scores, sorted_indices = torch.sort(predictions, descending=True)
            
            # Calculate recall@k for each k
            for k in k_values:
                top_k_indices = sorted_indices[:k]
                hit = 1.0 if target_idx in top_k_indices else 0.0
                results[f'recall@{k}'].append(hit)
            
            # Calculate MRR
            try:
                rank_pos = (sorted_indices == target_idx).nonzero(as_tuple=True)[0]
                if len(rank_pos) > 0:
                    rank = rank_pos[0].item() + 1  # 1-indexed
                    mrr = 1.0 / rank
                else:
                    mrr = 0.0
            except:
                mrr = 0.0
            
            results['mrr'].append(mrr)
        
        # Update recommender statistics
        if self.exclude_prev_purchases:
            self.recommender.total_purchase_exclusions += total_exclusions
            self.recommender.test_user_purchase_exclusions += total_exclusions
            self.recommender.excluded_purchase_ranks.extend(exclusion_ranks)
            self.recommender.test_users_processed += len([s for s in test_samples if self.recommender.get_user_purchase_history(s['user_id'])])
        
        # Calculate final metrics
        final_metrics = {}
        for k in k_values:
            recall = np.mean(results[f'recall@{k}'])
            final_metrics[f'purchase_recall@{k}'] = recall
        
        mrr = np.mean(results['mrr'])
        final_metrics['purchase_mrr'] = mrr
        
        # Add counts
        final_metrics['counts'] = {
            'total': len(all_predictions),
            'successful': len(all_predictions),
            'purchase': len(all_predictions),
            'exclusions': total_exclusions
        }
        
        print(f"Fast evaluation completed: {len(all_predictions)} samples, {total_exclusions} exclusions")
        
        return final_metrics

    def evaluate_individual_based(self, test_samples: List[Dict], k_values: List[int] = [10, 20, 50]) -> Dict:
        """
        Evaluate using individual recommend() calls to properly handle purchase exclusion
        
        Args:
            test_samples: List of test samples
            k_values: List of k values for recall@k metrics
            
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\nRunning individual-based evaluation (k={k_values}) with exclude_prev_purchases={self.exclude_prev_purchases}...")
        
        # Track results manually
        results = {f'recall@{k}': [] for k in k_values}
        results['mrr'] = []
        
        total_samples = len(test_samples)
        successful_evaluations = 0
        
        for i, sample in enumerate(tqdm(test_samples, desc="Individual evaluation")):
            user_id = sample['user_id']
            target_package = str(sample['purchased_package'])
            
            try:
                # Get recommendations using the recommend method (which respects exclude_prev_purchases)
                max_k = max(k_values)
                recommendations = self.recommender.recommend(
                    user_id=user_id,
                    top_k=max_k,
                    exclude_purchased=self.exclude_prev_purchases,
                    is_test_user=True  # This user has purchases in test data
                )
                
                # Extract package IDs from recommendations
                if recommendations and len(recommendations) > 0:
                    if isinstance(recommendations[0], dict):
                        rec_package_ids = [str(rec['package_id']) for rec in recommendations]
                    else:
                        rec_package_ids = [str(rec) for rec in recommendations]
                else:
                    rec_package_ids = []
                
                # Calculate recall@k for each k
                for k in k_values:
                    top_k_packages = rec_package_ids[:k]
                    hit = 1.0 if target_package in top_k_packages else 0.0
                    results[f'recall@{k}'].append(hit)
                
                # Calculate MRR
                try:
                    rank = rec_package_ids.index(target_package) + 1  # 1-indexed
                    mrr = 1.0 / rank
                except ValueError:
                    mrr = 0.0  # Target not in recommendations
                
                results['mrr'].append(mrr)
                successful_evaluations += 1
                
            except Exception as e:
                if i < 5:  # Only print first few errors
                    print(f"Error evaluating sample {i} (user {user_id}): {e}")
                
                # Add zeros for failed samples
                for k in k_values:
                    results[f'recall@{k}'].append(0.0)
                results['mrr'].append(0.0)
        
        # Calculate final metrics
        final_metrics = {}
        for k in k_values:
            recall = np.mean(results[f'recall@{k}'])
            final_metrics[f'purchase_recall@{k}'] = recall
        
        mrr = np.mean(results['mrr'])
        final_metrics['purchase_mrr'] = mrr
        
        # Add counts
        final_metrics['counts'] = {
            'total': total_samples,
            'successful': successful_evaluations,
            'purchase': total_samples  # All samples are purchases in this method
        }
        
        print(f"Individual evaluation completed: {successful_evaluations}/{total_samples} successful")
        
        return final_metrics
    
    def evaluate_batch_based(self, test_loader, k_values: List[int] = [10, 20, 50]) -> Dict:
        """
        Evaluate using batch-based approach exactly like training evaluation
        
        Args:
            test_loader: DataLoader for test data
            k_values: List of k values for recall@k metrics
            
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\nRunning batch-based evaluation (k={k_values})...")
        
        # Use the same UnifiedMetricsTracker as training
        metrics_tracker = UnifiedMetricsTracker(k_values=k_values)
        
        model = self.recommender.model
        model.eval()
        
        total_samples = 0
        
        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc="Evaluating")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                batch = self._move_batch_to_device(batch, self.device)
                
                # Forward pass
                try:
                    outputs = model(batch)
                    predictions = outputs['predictions']
                    targets = batch['purchased']['package_ids']
                    
                    # Get event indicators (same as training)
                    is_purchase = batch.get('is_purchase', torch.zeros_like(targets, dtype=torch.bool))
                    has_checkout = batch.get('has_checkout', torch.zeros_like(targets, dtype=torch.bool))
                    has_add_to_cart = batch.get('has_add_to_cart', torch.zeros_like(targets, dtype=torch.bool))
                    
                    # Get inclusive flags for individual event recalls
                    has_checkout_inclusive = batch.get('has_checkout_inclusive', has_checkout)
                    has_add_to_cart_inclusive = batch.get('has_add_to_cart_inclusive', has_add_to_cart)
                    
                    # Get cold start flag
                    is_cold_start = batch.get('is_cold_start', torch.zeros_like(targets, dtype=torch.bool))
                    
                    # Update metrics tracker (same as training)
                    metrics_tracker.update(
                        predictions=predictions,
                        targets=targets,
                        is_purchase=is_purchase,
                        has_checkout=has_checkout,
                        has_add_to_cart=has_add_to_cart,
                        has_checkout_inclusive=has_checkout_inclusive,
                        has_add_to_cart_inclusive=has_add_to_cart_inclusive,
                        is_cold_start=is_cold_start
                    )
                    
                    total_samples += targets.size(0)
                    
                    # Debug info for first batch
                    if batch_idx == 0:
                        print(f"Debug - First batch: {targets.size(0)} samples")
                        print(f"Debug - Purchase samples: {is_purchase.sum().item()}")
                        print(f"Debug - Checkout samples: {has_checkout.sum().item()}")
                        print(f"Debug - Add-to-cart samples: {has_add_to_cart.sum().item()}")
                
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    continue
                
                # Memory cleanup for MPS
                if self.device.type == 'mps' and (batch_idx + 1) % 10 == 0:
                    torch.mps.empty_cache()
        
        # Compute final metrics
        metrics = metrics_tracker.compute()
        
        print(f"Evaluation completed on {total_samples:,} samples")
        
        return metrics
    
    def _move_batch_to_device(self, batch: Dict, device: torch.device) -> Dict:
        """Move batch data to specified device"""
        moved_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(device)
            elif isinstance(value, dict):
                moved_batch[key] = self._move_batch_to_device(value, device)
            else:
                moved_batch[key] = value
        
        return moved_batch
    
    def run_evaluation(self, 
                      k_values: List[int] = [10, 20, 50],
                      sample_size: Optional[int] = None,
                      purchases_only: bool = False,
                      save_results: bool = True) -> Dict:
        """
        Run complete evaluation pipeline
        
        Args:
            k_values: List of k values for metrics
            sample_size: Optional sample size limit
            purchases_only: If True, only evaluate purchase samples
            save_results: If True, save results to JSON file
            
        Returns:
            Dictionary of evaluation results
        """
        start_time = time.time()
        
        # Prepare test data
        test_loader, test_samples = self.prepare_test_data(sample_size, purchases_only)
        
        # Choose evaluation method based on whether we need to exclude previous purchases
        if self.exclude_prev_purchases:
            # Use fast batch-based evaluation with post-processing for purchase exclusion
            print("Using fast batch evaluation with purchase exclusion...")
            purchase_samples = [s for s in test_samples if s.get('is_purchase', False)]
            if not purchase_samples:
                purchase_samples = test_samples  # Fallback if no purchase flag
            metrics = self.evaluate_batch_with_exclusion(test_loader, purchase_samples, k_values)
        else:
            # Use fast batch-based evaluation
            print("Using batch-based evaluation (no purchase exclusion)...")
            metrics = self.evaluate_batch_based(test_loader, k_values)
        
        # Add metadata
        evaluation_info = {
            'model_info_path': self.model_info_path,
            'training_strategy': self.model_info.get('training_strategy', 'standard'),
            'train_ratio': self.train_ratio,
            'split_date': self.split_date,
            'evaluation_settings': {
                'k_values': k_values,
                'sample_size': sample_size,
                'purchases_only': purchases_only,
                'exclude_prev_purchases': self.exclude_prev_purchases,
                'test_samples': len(test_samples),
                'availability_filtering': self.availability_data_path is not None
            },
            'metrics': metrics,
            'evaluation_time': time.time() - start_time,
            'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Print results
        self._print_results(evaluation_info)
        
        # Print final exclusion summary if we excluded previous purchases
        if self.exclude_prev_purchases:
            self.recommender.print_final_exclusion_summary()
        
        # Save results
        if save_results:
            output_file = f"evaluation_results_{int(time.time())}.json"
            with open(output_file, 'w') as f:
                json.dump(evaluation_info, f, indent=2)
            print(f"\nResults saved to: {output_file}")
        
        return evaluation_info
    
    def _print_results(self, results: Dict):
        """Print evaluation results in a clean format"""
        metrics = results['metrics']
        settings = results['evaluation_settings']
        
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        print(f"Model: {os.path.basename(results['model_info_path'])}")
        print(f"Strategy: {results['training_strategy']}")
        print(f"Test samples: {settings['test_samples']:,}")
        print(f"Evaluation time: {results['evaluation_time']:.1f}s")
        
        print(f"\nCore Metrics:")
        for k in settings['k_values']:
            if f'purchase_recall@{k}' in metrics:
                print(f"  Purchase Recall@{k}: {metrics[f'purchase_recall@{k}']*100:.2f}%")
        
        if 'purchase_mrr' in metrics:
            print(f"  Purchase MRR: {metrics['purchase_mrr']:.4f}")
        
        print(f"\nDetailed Event Metrics:")
        for k in settings['k_values']:
            if f'checkout_recall@{k}' in metrics:
                print(f"  Checkout Recall@{k}: {metrics[f'checkout_recall@{k}']*100:.2f}%")
            if f'item_coverage@{k}' in metrics:
                print(f"  Item Coverage@{k}: {metrics[f'item_coverage@{k}']*100:.2f}%")
        
        # Cold start vs warm user performance
        has_cold_start_metrics = any(f'cold_start_purchase_recall@{k}' in metrics for k in settings['k_values'])
        if has_cold_start_metrics:
            print(f"\nCold Start vs Warm User Performance:")
            for k in settings['k_values']:
                if f'cold_start_purchase_recall@{k}' in metrics:
                    print(f"  Cold Start Purchase Recall@{k}: {metrics[f'cold_start_purchase_recall@{k}']*100:.2f}%")
                if f'warm_purchase_recall@{k}' in metrics:
                    print(f"  Warm User Purchase Recall@{k}: {metrics[f'warm_purchase_recall@{k}']*100:.2f}%")
            
            if 'cold_start_purchase_mrr' in metrics:
                print(f"  Cold Start Purchase MRR: {metrics['cold_start_purchase_mrr']:.4f}")
            if 'warm_purchase_mrr' in metrics:
                print(f"  Warm User Purchase MRR: {metrics['warm_purchase_mrr']:.4f}")
        
        print(f"\nSample Counts:")
        counts = metrics.get('counts', {})
        if 'total' in counts:
            print(f"  Total: {counts['total']:,}")
        if 'purchase' in counts:
            print(f"  Purchase: {counts['purchase']:,}")
        if 'checkout' in counts:
            print(f"  Checkout: {counts['checkout']:,}")
        
        # Cold start vs warm user counts
        if 'cold_start_purchase' in counts or 'warm_purchase' in counts:
            print(f"\nUser Type Breakdown:")
            if 'cold_start_purchase' in counts:
                print(f"  Cold Start Purchases: {counts['cold_start_purchase']:,}")
            if 'warm_purchase' in counts:
                print(f"  Warm User Purchases: {counts['warm_purchase']:,}")
            if 'cold_start_checkout' in counts:
                print(f"  Cold Start Checkouts: {counts['cold_start_checkout']:,}")
            if 'warm_checkout' in counts:
                print(f"  Warm User Checkouts: {counts['warm_checkout']:,}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate NATR recommendations with training-consistent methodology'
    )
    
    parser.add_argument('--model-info', type=str, required=True,
                        help='Path to model_info.json file')
    parser.add_argument('--availability-data', type=str,
                        help='Path to availability parquet file')
    parser.add_argument('--k-values', type=int, nargs='+', default=[10, 20, 50],
                        help='K values for recall@k metrics (default: 10 20 50)')
    parser.add_argument('--sample-size', type=int,
                        help='Limit evaluation to N samples for faster testing')
    parser.add_argument('--purchases-only', action='store_true',
                        help='Only evaluate on purchase samples (filters test data)')
    parser.add_argument('--exclude-prev-purchases', action='store_true',
                        help='Exclude previously bought packages from recommendations')
    parser.add_argument('--exclude-recent-months', type=int,
                        help='Only exclude purchases from the last N months instead of all history')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to file')
    
    args = parser.parse_args()
    
    # Validate model info file
    if not os.path.exists(args.model_info):
        print(f"Error: Model info file not found: {args.model_info}")
        return 1
    
    # Validate availability data if provided
    if args.availability_data and not os.path.exists(args.availability_data):
        print(f"Error: Availability data file not found: {args.availability_data}")
        return 1
    
    try:
        # Initialize evaluator
        evaluator = RecommendationEvaluator(
            model_info_path=args.model_info,
            availability_data_path=args.availability_data,
            exclude_prev_purchases=args.exclude_prev_purchases,
            exclude_recent_months=args.exclude_recent_months
        )
        
        # Run evaluation
        results = evaluator.run_evaluation(
            k_values=args.k_values,
            sample_size=args.sample_size,
            purchases_only=args.purchases_only,
            save_results=not args.no_save
        )
        
        return 0
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
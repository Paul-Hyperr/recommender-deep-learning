#!/usr/bin/env python3
"""
Independent Model Performance Verification Script

This script provides a completely different approach to evaluate the model performance
to verify the 46.18% Purchase Recall@20 result from evaluate_recommendations.py.

Key differences from evaluate_recommendations.py:
1. Direct model loading and inference (no NATRRecommender wrapper)
2. Manual data processing and batching
3. Simple, transparent metric calculation
4. Detailed sample-by-sample analysis
5. Different random sampling approach
6. Cross-validation with multiple random seeds

Usage:
    python verify_model_performance.py --model-info output/model_info/model_info_pretrain_finetune_46_18.json
"""

import argparse
import json
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import time
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.natr import NATR, NATRConfig
from utils.session_processor import SessionProcessor
from utils.package_processor import PackageProcessor
from utils.training_utils import (
    filter_by_min_session_length, filter_items_by_frequency,
    identify_event_types, time_based_split_year
)


class IndependentModelVerifier:
    """
    Independent model performance verifier using a different approach
    """
    
    def __init__(self, model_info_path: str):
        """Initialize verifier"""
        self.model_info_path = model_info_path
        
        # Load model info
        with open(model_info_path, 'r') as f:
            self.model_info = json.load(f)
        
        print(f"🔍 Independent Verification of: {os.path.basename(model_info_path)}")
        print(f"Training strategy: {self.model_info.get('training_strategy', 'standard')}")
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() 
                                  else 'mps' if torch.backends.mps.is_available() 
                                  else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.model = None
        self.package_processor = None
        self.session_processor = None
        self.test_samples = None
        
    def load_model_directly(self):
        """Load model directly from checkpoint"""
        print("\n📦 Loading model directly from checkpoint...")
        
        checkpoint_path = self.model_info['checkpoint_path']
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create config
        config_dict = self.model_info['config']
        config = NATRConfig(**config_dict)
        
        # Create and load model
        self.model = NATR(config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Model loaded successfully")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def prepare_data_independently(self, sample_size: Optional[int] = None):
        """Prepare test data using independent approach"""
        print("\n📊 Preparing test data independently...")
        
        # Data paths
        event_data_path = self.model_info.get('event_data_path', 'data/bookit_events_data_13_months.parquet')
        package_data_path = self.model_info.get('package_data_path', 'data/feed.parquet')
        
        # Initialize processors
        print("Initializing processors...")
        self.package_processor = PackageProcessor(
            feed_data_path=package_data_path,
            cache_dir='data/cache',
            load_coordinates=True,
            load_embeddings=True,
            api_key=os.environ.get("OPENAI_API_KEY"),
            embedding_model='text-embedding-3-small',
            use_reduced_embeddings=True
        )
        
        self.session_processor = SessionProcessor(
            event_data_path=event_data_path,
            cache_dir='data/cache',
            min_interactions=5,
            max_sessions_per_user=20,
            max_samples_per_user=10
        )
        
        # Load data
        print("Loading package and session data...")
        self.package_processor.load_data()
        self.session_processor.load_data()
        
        print("Creating mappings...")
        self.package_processor.create_mappings()
        self.session_processor.create_mappings()
        
        print("Extracting sessions...")
        self.session_processor.extract_sessions()
        
        # Prepare samples
        print("Preparing training samples...")
        all_samples = self.session_processor.prepare_enhanced_training_data()
        
        # Apply same processing as training
        event_to_idx = self.session_processor.get_idx_mappings()['event_to_idx']
        all_samples = identify_event_types(all_samples, event_to_idx)
        
        # Filter by session length
        quality_samples = filter_by_min_session_length(all_samples, min_session_length=2)
        
        # Use valid packages from training
        valid_packages = set(self.model_info.get('valid_packages', []))
        if valid_packages:
            print(f"Filtering to {len(valid_packages)} valid packages from training")
            filtered_samples = [s for s in quality_samples 
                              if str(s.get('purchased_package', '')) in valid_packages]
        else:
            filtered_samples, _ = filter_items_by_frequency(quality_samples, min_frequency=5)
        
        # Time-based split
        train_ratio = self.model_info.get('train_ratio', 0.91)
        train_samples, test_samples, split_date = time_based_split_year(
            filtered_samples, train_ratio=train_ratio
        )
        
        # Focus on purchases only
        purchase_samples = [s for s in test_samples if s.get('is_purchase', False)]
        print(f"Found {len(purchase_samples):,} purchase samples in test set")
        
        # Sample if requested
        if sample_size and sample_size < len(purchase_samples):
            # Use different random sampling approach
            np.random.seed(12345)  # Different seed than main evaluation
            sampled_indices = np.random.choice(len(purchase_samples), sample_size, replace=False)
            purchase_samples = [purchase_samples[i] for i in sampled_indices]
            print(f"Randomly sampled {len(purchase_samples):,} purchases")
        
        self.test_samples = purchase_samples
        print(f"✅ Prepared {len(self.test_samples):,} test samples")
        
    def create_batch_manually(self, samples: List[Dict], batch_size: int = 32) -> List[Dict]:
        """Create batches manually without using DataLoader"""
        print(f"Creating batches manually (batch_size={batch_size})...")
        
        batches = []
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i+batch_size]
            batch = self._samples_to_batch(batch_samples)
            batches.append(batch)
        
        print(f"Created {len(batches)} batches")
        return batches
    
    def _samples_to_batch(self, samples: List[Dict]) -> Dict:
        """Convert samples to batch format manually"""
        batch_size = len(samples)
        max_short_term = 10
        max_long_term = 20
        
        # Initialize batch tensors
        batch = {
            'user_id': torch.zeros(batch_size, dtype=torch.long),
            'short_term': {
                'package_ids': torch.zeros(batch_size, max_short_term, dtype=torch.long),
                'title_embeddings': torch.zeros(batch_size, max_short_term, 1536),
                'coordinates': torch.zeros(batch_size, max_short_term, 2),
                'country_ids': torch.zeros(batch_size, max_short_term, dtype=torch.long),
                'category_ids': torch.zeros(batch_size, max_short_term, dtype=torch.long),
                'theme_ids': torch.zeros(batch_size, max_short_term, dtype=torch.long),
                'prices': torch.zeros(batch_size, max_short_term),
                'event_types': torch.zeros(batch_size, max_short_term, dtype=torch.long),
            },
            'long_term': {
                'package_ids': torch.zeros(batch_size, max_long_term, dtype=torch.long),
                'title_embeddings': torch.zeros(batch_size, max_long_term, 1536),
                'coordinates': torch.zeros(batch_size, max_long_term, 2),
                'country_ids': torch.zeros(batch_size, max_long_term, dtype=torch.long),
                'category_ids': torch.zeros(batch_size, max_long_term, dtype=torch.long),
                'theme_ids': torch.zeros(batch_size, max_long_term, dtype=torch.long),
                'prices': torch.zeros(batch_size, max_long_term),
                'event_types': torch.zeros(batch_size, max_long_term, dtype=torch.long),
            },
            'purchased': {
                'package_ids': torch.zeros(batch_size, dtype=torch.long),
                'title_embeddings': torch.zeros(batch_size, 1536),
                'coordinates': torch.zeros(batch_size, 2),
                'country_ids': torch.zeros(batch_size, dtype=torch.long),
                'category_ids': torch.zeros(batch_size, dtype=torch.long),
                'theme_ids': torch.zeros(batch_size, dtype=torch.long),
                'prices': torch.zeros(batch_size),
            },
            'is_purchase': torch.ones(batch_size, dtype=torch.bool),  # All are purchases
        }
        
        # Fill batch data
        user_to_idx = self.session_processor.user_to_idx
        package_to_idx = self.session_processor.package_to_idx
        
        for i, sample in enumerate(samples):
            # User ID
            user_id = sample['user_id']
            batch['user_id'][i] = user_to_idx.get(user_id, 0)
            
            # Short-term history
            short_term_packages = sample.get('short_term_packages', [])[-max_short_term:]
            short_term_events = sample.get('short_term_events', [])[-max_short_term:]
            
            for j, (pkg_id, event_id) in enumerate(zip(short_term_packages, short_term_events)):
                if j >= max_short_term:
                    break
                
                pkg_idx = package_to_idx.get(str(pkg_id), 0)
                batch['short_term']['package_ids'][i, j] = pkg_idx
                batch['short_term']['event_types'][i, j] = event_id
                
                # Get package features
                features = self.package_processor.get_package_features(str(pkg_id))
                if features:
                    if features['title_embedding'] is not None:
                        batch['short_term']['title_embeddings'][i, j] = torch.from_numpy(features['title_embedding'])
                    if features['latitude'] is not None and features['longitude'] is not None:
                        batch['short_term']['coordinates'][i, j] = torch.tensor([features['latitude'], features['longitude']])
                    batch['short_term']['country_ids'][i, j] = features['country_idx']
                    batch['short_term']['category_ids'][i, j] = features['category_idx']
                    batch['short_term']['theme_ids'][i, j] = features['theme_idx']
                    batch['short_term']['prices'][i, j] = features['price']
            
            # Long-term history
            long_term_packages = sample.get('long_term_packages', [])[-max_long_term:]
            long_term_events = sample.get('long_term_events', [])[-max_long_term:]
            
            for j, (pkg_id, event_id) in enumerate(zip(long_term_packages, long_term_events)):
                if j >= max_long_term:
                    break
                
                pkg_idx = package_to_idx.get(str(pkg_id), 0)
                batch['long_term']['package_ids'][i, j] = pkg_idx
                batch['long_term']['event_types'][i, j] = event_id
                
                # Get package features
                features = self.package_processor.get_package_features(str(pkg_id))
                if features:
                    if features['title_embedding'] is not None:
                        batch['long_term']['title_embeddings'][i, j] = torch.from_numpy(features['title_embedding'])
                    if features['latitude'] is not None and features['longitude'] is not None:
                        batch['long_term']['coordinates'][i, j] = torch.tensor([features['latitude'], features['longitude']])
                    batch['long_term']['country_ids'][i, j] = features['country_idx']
                    batch['long_term']['category_ids'][i, j] = features['category_idx']
                    batch['long_term']['theme_ids'][i, j] = features['theme_idx']
                    batch['long_term']['prices'][i, j] = features['price']
            
            # Purchased package
            purchased_id = str(sample['purchased_package'])
            batch['purchased']['package_ids'][i] = package_to_idx.get(purchased_id, 0)
            
            features = self.package_processor.get_package_features(purchased_id)
            if features:
                if features['title_embedding'] is not None:
                    batch['purchased']['title_embeddings'][i] = torch.from_numpy(features['title_embedding'])
                if features['latitude'] is not None and features['longitude'] is not None:
                    batch['purchased']['coordinates'][i] = torch.tensor([features['latitude'], features['longitude']])
                batch['purchased']['country_ids'][i] = features['country_idx']
                batch['purchased']['category_ids'][i] = features['category_idx']
                batch['purchased']['theme_ids'][i] = features['theme_idx']
                batch['purchased']['prices'][i] = features['price']
        
        return batch
    
    def evaluate_with_detailed_tracking(self, k_values: List[int] = [10, 20, 50]) -> Dict:
        """Evaluate with detailed sample-by-sample tracking"""
        print(f"\n🎯 Running detailed evaluation (k={k_values})...")
        
        # Create batches
        batches = self.create_batch_manually(self.test_samples, batch_size=64)
        
        # Track results
        all_predictions = []
        all_targets = []
        sample_details = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
                # Move to device
                batch = self._move_to_device(batch)
                
                # Forward pass
                outputs = self.model(batch)
                predictions = outputs['predictions']  # [batch_size, num_packages]
                targets = batch['purchased']['package_ids']  # [batch_size]
                
                # Store for later analysis
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
                
                # Track sample details
                for i in range(targets.size(0)):
                    sample_idx = batch_idx * 64 + i
                    if sample_idx < len(self.test_samples):
                        sample_details.append({
                            'sample_idx': sample_idx,
                            'user_id': self.test_samples[sample_idx]['user_id'],
                            'purchased_package': self.test_samples[sample_idx]['purchased_package'],
                            'target_idx': targets[i].item(),
                            'prediction_scores': predictions[i].cpu().numpy()
                        })
        
        # Combine all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)  # [total_samples, num_packages]
        all_targets = torch.cat(all_targets, dim=0)  # [total_samples]
        
        print(f"Processed {len(all_predictions)} samples")
        
        # Calculate metrics manually
        metrics = self._calculate_metrics_manually(all_predictions, all_targets, k_values)
        
        # Add detailed analysis
        metrics['sample_details'] = sample_details[:100]  # Store first 100 for inspection
        metrics['total_samples'] = len(all_predictions)
        
        return metrics
    
    def _calculate_metrics_manually(self, predictions: torch.Tensor, targets: torch.Tensor, k_values: List[int]) -> Dict:
        """Calculate metrics manually with transparency"""
        print("📊 Calculating metrics manually...")
        
        metrics = {}
        total_samples = len(targets)
        
        # Get top-k predictions for each sample
        for k in k_values:
            print(f"Calculating Recall@{k}...")
            
            # Get top-k indices for each sample
            _, top_k_indices = torch.topk(predictions, k, dim=1)  # [total_samples, k]
            
            # Check if target is in top-k for each sample
            targets_expanded = targets.unsqueeze(1).expand(-1, k)  # [total_samples, k]
            hits = (top_k_indices == targets_expanded).any(dim=1)  # [total_samples]
            
            # Calculate recall
            recall = hits.float().mean().item()
            metrics[f'purchase_recall@{k}'] = recall
            
            print(f"  Recall@{k}: {recall*100:.2f}% ({hits.sum().item()}/{total_samples})")
        
        # Calculate MRR manually
        print("Calculating MRR...")
        mrr_sum = 0.0
        
        for i in range(total_samples):
            target_idx = targets[i].item()
            sample_predictions = predictions[i]
            
            # Sort predictions in descending order
            sorted_indices = torch.argsort(sample_predictions, descending=True)
            
            # Find rank of target (1-indexed)
            rank = (sorted_indices == target_idx).nonzero(as_tuple=True)[0]
            if len(rank) > 0:
                rank = rank[0].item() + 1  # Convert to 1-indexed
                mrr_sum += 1.0 / rank
        
        mrr = mrr_sum / total_samples
        metrics['purchase_mrr'] = mrr
        print(f"  MRR: {mrr:.4f}")
        
        return metrics
    
    def _move_to_device(self, batch: Dict) -> Dict:
        """Move batch to device recursively"""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {key: self._move_to_device(value) for key, value in batch.items()}
        else:
            return batch
    
    def cross_validate_with_seeds(self, num_seeds: int = 5, sample_size: int = 1000) -> Dict:
        """Cross-validate results with different random seeds"""
        print(f"\n🔄 Cross-validating with {num_seeds} different random seeds...")
        
        results = []
        
        for seed in range(num_seeds):
            print(f"\nSeed {seed + 1}/{num_seeds}: {seed * 111}")
            
            # Set seeds
            np.random.seed(seed * 111)
            torch.manual_seed(seed * 111)
            random.seed(seed * 111)
            
            # Sample test data with this seed
            sampled_indices = np.random.choice(len(self.test_samples), min(sample_size, len(self.test_samples)), replace=False)
            seed_samples = [self.test_samples[i] for i in sampled_indices]
            
            # Temporarily replace test samples
            original_samples = self.test_samples
            self.test_samples = seed_samples
            
            # Evaluate
            metrics = self.evaluate_with_detailed_tracking([20])
            results.append(metrics['purchase_recall@20'])
            
            print(f"  Seed {seed * 111}: Recall@20 = {metrics['purchase_recall@20']*100:.2f}%")
            
            # Restore original samples
            self.test_samples = original_samples
        
        # Calculate statistics
        mean_recall = np.mean(results)
        std_recall = np.std(results)
        
        return {
            'mean_recall@20': mean_recall,
            'std_recall@20': std_recall,
            'individual_results': results,
            'confidence_interval_95': (mean_recall - 1.96*std_recall, mean_recall + 1.96*std_recall)
        }
    
    def run_verification(self, sample_size: Optional[int] = None, cross_validate: bool = True) -> Dict:
        """Run complete independent verification"""
        start_time = time.time()
        
        print("🚀 Starting Independent Model Verification")
        print("=" * 80)
        
        # Load model
        self.load_model_directly()
        
        # Prepare data
        self.prepare_data_independently(sample_size)
        
        # Main evaluation
        main_results = self.evaluate_with_detailed_tracking([10, 20, 50])
        
        # Cross-validation if requested
        if cross_validate and len(self.test_samples) > 1000:
            cv_results = self.cross_validate_with_seeds(num_seeds=5, sample_size=1000)
            main_results['cross_validation'] = cv_results
        
        # Summary
        verification_results = {
            'verification_time': time.time() - start_time,
            'model_info_path': self.model_info_path,
            'total_test_samples': len(self.test_samples),
            'device': str(self.device),
            'main_results': main_results,
            'verification_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self._print_verification_summary(verification_results)
        
        return verification_results
    
    def _print_verification_summary(self, results: Dict):
        """Print verification summary"""
        main = results['main_results']
        
        print("\n" + "=" * 80)
        print("🎯 INDEPENDENT VERIFICATION RESULTS")
        print("=" * 80)
        
        print(f"Model: {os.path.basename(results['model_info_path'])}")
        print(f"Test samples: {results['total_test_samples']:,}")
        print(f"Device: {results['device']}")
        print(f"Verification time: {results['verification_time']:.1f}s")
        
        print(f"\n📊 Core Metrics:")
        for k in [10, 20, 50]:
            if f'purchase_recall@{k}' in main:
                recall = main[f'purchase_recall@{k}']
                print(f"  Purchase Recall@{k}: {recall*100:.2f}%")
        
        if 'purchase_mrr' in main:
            print(f"  Purchase MRR: {main['purchase_mrr']:.4f}")
        
        # Cross-validation results
        if 'cross_validation' in main:
            cv = main['cross_validation']
            print(f"\n🔄 Cross-Validation (5 seeds, 1000 samples each):")
            print(f"  Mean Recall@20: {cv['mean_recall@20']*100:.2f}% ± {cv['std_recall@20']*100:.2f}%")
            print(f"  95% CI: [{cv['confidence_interval_95'][0]*100:.2f}%, {cv['confidence_interval_95'][1]*100:.2f}%]")
            print(f"  Individual results: {[f'{r*100:.1f}%' for r in cv['individual_results']]}")
        
        # Compare with original evaluation
        expected_recall_20 = 46.18
        actual_recall_20 = main.get('purchase_recall@20', 0) * 100
        difference = actual_recall_20 - expected_recall_20
        
        print(f"\n✅ Verification vs Original:")
        print(f"  Original Recall@20: {expected_recall_20:.2f}%")
        print(f"  Verified Recall@20: {actual_recall_20:.2f}%")
        print(f"  Difference: {difference:+.2f}%")
        
        if abs(difference) < 2.0:
            print(f"  Status: ✅ VERIFIED (within 2%)")
        elif abs(difference) < 5.0:
            print(f"  Status: ⚠️  CLOSE (within 5%)")
        else:
            print(f"  Status: ❌ SIGNIFICANT DIFFERENCE (>5%)")


def main():
    parser = argparse.ArgumentParser(
        description='Independent verification of model performance'
    )
    parser.add_argument('--model-info', type=str, required=True,
                        help='Path to model_info.json file')
    parser.add_argument('--sample-size', type=int,
                        help='Limit to N samples for faster verification')
    parser.add_argument('--no-cross-validate', action='store_true',
                        help='Skip cross-validation')
    parser.add_argument('--save-results', type=str,
                        help='Save detailed results to JSON file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_info):
        print(f"Error: Model info file not found: {args.model_info}")
        return 1
    
    try:
        # Run verification
        verifier = IndependentModelVerifier(args.model_info)
        results = verifier.run_verification(
            sample_size=args.sample_size,
            cross_validate=not args.no_cross_validate
        )
        
        # Save results if requested
        if args.save_results:
            with open(args.save_results, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nDetailed results saved to: {args.save_results}")
        
        return 0
        
    except Exception as e:
        print(f"\nError during verification: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
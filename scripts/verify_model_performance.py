#!/usr/bin/env python3
"""
Verify NATR Enhanced Model Performance on Unseen Test Data

This script evaluates how well the model ranks unseen purchases in the test data,
without availability filtering or purchase exclusion. It also provides warm/cold
start analysis and detailed examples of 3 warm-start users.
"""

import argparse
import json
import sys
import os
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Tuple
from tqdm import tqdm
from collections import defaultdict
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.natr_enhanced import NATREnhanced, NATRConfig
from utils.unified_metrics import UnifiedMetricsTracker
from utils.training_utils import (
    extract_test_events_from_parquet, create_dataloaders, 
    filter_by_min_session_length, filter_items_by_frequency,
    identify_event_types, analyze_data_distribution, time_based_split_year
)
from utils.session_processor2 import SessionProcessor2
from utils.package_processor import PackageProcessor


class ModelPerformanceVerifier:
    """Verify model performance on unseen test data with detailed analysis"""
    
    def __init__(self, model_info_path: str):
        """Initialize verifier"""
        self.model_info_path = model_info_path
        
        # Load model info
        with open(model_info_path, 'r') as f:
            self.model_info = json.load(f)
        
        print(f"Verifying model: {os.path.basename(model_info_path)}")
        print(f"Model type: {self.model_info.get('model_type', 'unknown')}")
        print(f"Training strategy: {self.model_info.get('training_strategy', 'standard')}")
        print(f"Best Purchase Recall@20: {self.model_info.get('best_purchase_recall@20', 0)*100:.2f}%")
        
        # Get split info
        self.train_ratio = self.model_info.get('train_ratio', 0.91)
        self.split_date = self.model_info.get('split_date', None)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() 
                                  else 'mps' if torch.backends.mps.is_available() 
                                  else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load the enhanced model directly from checkpoint
        print("\nLoading enhanced model from checkpoint...")
        checkpoint_path = 'checkpoints/natr_enhanced/finetuned_model.pth'
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
        # Create model config from model_info
        config = NATRConfig(
            num_users=self.model_info['config']['num_users'],
            num_packages=self.model_info['config']['num_packages'],
            num_countries=self.model_info['config']['num_countries'],
            num_categories=self.model_info['config']['num_categories'],
            num_themes=self.model_info['config']['num_themes'],
            title_embedding_dim=self.model_info['config']['title_embedding_dim'],
            hidden_dim=self.model_info['config']['hidden_dim'],
            embedding_dim=self.model_info['config']['embedding_dim'],
            user_embedding_dim=self.model_info['config']['user_embedding_dim'],
            dropout=self.model_info['config']['dropout'],
            max_short_term=self.model_info['config']['max_short_term'],
            max_long_term=self.model_info['config']['max_long_term']
        )
        
        # Initialize and load model
        self.model = NATREnhanced(config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully from {checkpoint_path}")
        print(f"Model performance from checkpoint: Recall@20: {checkpoint.get('best_recall_20', 'N/A')}")
    
    def prepare_test_data(self) -> Tuple[List[Dict], List[Dict], Dict, Dict]:
        """Prepare test data efficiently by reusing cached data when possible"""
        print("\nPreparing test data (checking for cached data)...")
        
        # Data paths
        event_data = self.model_info.get('event_data_path', 'data/13_months_new2_clean.parquet')
        package_data_path = self.model_info.get('package_data_path', 'data/feed.parquet')
        
        # Initialize processors exactly like training script
        print("Initializing data processors...")
        package_processor = PackageProcessor(
            feed_data_path=package_data_path,
            cache_dir='data/cache',
            load_coordinates=True,
            load_embeddings=True,
            api_key=os.environ.get("OPENAI_API_KEY"),
            embedding_model='text-embedding-3-small',
            use_reduced_embeddings=True
        )
        
        session_processor = SessionProcessor2(
            event_data_path=event_data,
            cache_dir='data/cache',
            session_timeout_hours=30,
            min_interactions=8,  # Match training script
            max_sessions_per_user=20,
            max_samples_per_user=10
        )
        
        # Load and process data (will use cache if available)
        print("Loading and processing data (using cache if available)...")
        package_processor.load_data()
        session_processor.load_data()
        
        package_processor.create_mappings()
        session_processor.create_mappings()
        
        session_processor.extract_sessions()
        
        # Check if we can reuse processed samples from model training
        print("Preparing training samples...")
        all_samples = session_processor.prepare_enhanced_training_data()
        
        # Identify event types
        event_to_idx = session_processor.get_idx_mappings()['event_to_idx']
        all_samples = identify_event_types(all_samples, event_to_idx)
        
        # Apply same filters as training
        print("Applying training filters...")
        quality_samples = filter_by_min_session_length(all_samples, min_session_length=2)
        
        # Use exact same valid packages from training
        valid_packages = set(self.model_info.get('valid_packages', []))
        if valid_packages:
            print(f"Using {len(valid_packages)} valid packages from training")
            filtered_samples = []
            for sample in quality_samples:
                if str(sample.get('purchased_package', '')) in valid_packages:
                    filtered_samples.append(sample)
        else:
            print("No valid packages in model_info, filtering by frequency...")
            filtered_samples, valid_packages = filter_items_by_frequency(quality_samples, min_frequency=5)
        
        # Use exact same split as training
        print("Creating train/test split...")
        train_samples, test_samples, split_date = time_based_split_year(filtered_samples, train_ratio=self.train_ratio)
        
        # Verify split date matches training
        expected_split = self.model_info.get('split_date')
        if expected_split and split_date != expected_split:
            print(f"Warning: Split date mismatch! Expected: {expected_split}, Got: {split_date}")
        
        print(f"\nData split verification:")
        print(f"  Split date: {split_date} (expected: {expected_split})")
        print(f"  Train samples: {len(train_samples):,}")
        print(f"  Test samples: {len(test_samples):,}")
        
        # Get only purchase samples from test set
        test_purchases = [s for s in test_samples if s.get('is_purchase', False)]
        print(f"  Test purchases: {len(test_purchases):,}")
        
        # Identify warm/cold start users
        train_users = set(sample['user_id'] for sample in train_samples)
        test_user_purchases = defaultdict(list)
        for sample in test_purchases:
            test_user_purchases[sample['user_id']].append(sample)
        
        warm_users = {}
        cold_users = {}
        
        for user_id, purchases in test_user_purchases.items():
            if user_id in train_users:
                warm_users[user_id] = purchases
            else:
                cold_users[user_id] = purchases
        
        print(f"\nUser analysis:")
        print(f"  Warm start users (in train): {len(warm_users):,}")
        print(f"  Cold start users (new): {len(cold_users):,}")
        print(f"  Warm start purchases: {sum(len(p) for p in warm_users.values()):,}")
        print(f"  Cold start purchases: {sum(len(p) for p in cold_users.values()):,}")
        
        # Store processors for later use
        self.package_processor = package_processor
        self.session_processor = session_processor
        
        return train_samples, test_purchases, warm_users, cold_users
    
    def get_model_recommendations(self, user_id: str, top_k: int = 100) -> List[Dict]:
        """Get recommendations from the loaded model"""
        try:
            # Get user index
            user_idx = self.session_processor.user_to_idx.get(user_id)
            if user_idx is None:
                return []
            
            # Create a dummy batch for this user
            # We'll use the last session data for this user if available
            user_samples = [s for s in self.all_test_samples if s['user_id'] == user_id]
            if not user_samples:
                return []
            
            # Use the most recent sample for this user
            sample = user_samples[-1]
            
            # Create batch
            from utils.training_utils import create_dataloaders, move_batch_to_device
            
            # Create a single-sample dataset for inference
            dummy_train_samples = [sample]  # Just for dataloader creation
            test_samples = [sample]
            
            _, test_loader = create_dataloaders(
                train_samples=dummy_train_samples,
                test_samples=test_samples,
                package_processor=self.package_processor,
                session_processor=self.session_processor,
                batch_size=1,
                num_workers=0,
                use_weighted_sampling=False
            )
            
            # Get predictions
            with torch.no_grad():
                for batch in test_loader:
                    batch = move_batch_to_device(batch, self.device)
                    outputs = self.model(batch)
                    predictions = outputs['predictions'][0]  # First (and only) sample
                    
                    # Get top k predictions
                    scores, indices = torch.topk(predictions, k=min(top_k, len(predictions)))
                    
                    # Convert to package IDs
                    idx_to_package = {v: k for k, v in self.session_processor.package_to_idx.items()}
                    recommendations = []
                    
                    for score, idx in zip(scores.cpu().numpy(), indices.cpu().numpy()):
                        package_id = idx_to_package.get(idx)
                        if package_id:
                            recommendations.append({
                                'package_id': package_id,
                                'score': float(score)
                            })
                    
                    return recommendations
            
        except Exception as e:
            print(f"Error getting recommendations for user {user_id}: {e}")
            return []
        
        return []

    def evaluate_ranking_performance(self, test_purchases: List[Dict], 
                                   k_values: List[int] = [1, 5, 10, 20, 50]) -> Dict:
        """Evaluate how well the model ranks unseen purchases using efficient batch processing"""
        print(f"\nEvaluating ranking performance on {len(test_purchases)} purchases...")
        
        # Create ONE dataloader for all test samples - much more efficient!
        print("Creating test dataloader for batch evaluation...")
        dummy_train_samples = test_purchases[:10]  # Minimal train set
        
        from utils.training_utils import create_dataloaders, move_batch_to_device
        
        _, test_loader = create_dataloaders(
            train_samples=dummy_train_samples,
            test_samples=test_purchases,
            package_processor=self.package_processor,
            session_processor=self.session_processor,
            batch_size=32,  # Process in reasonable batches
            num_workers=0,
            use_weighted_sampling=False
        )
        
        print(f"Created dataloader with {len(test_loader)} batches")
        
        # Batch process all samples
        all_predictions = []
        all_targets = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Batch evaluation")):
                batch = move_batch_to_device(batch, self.device)
                
                # Get model predictions
                outputs = self.model(batch)
                predictions = outputs['predictions']  # [batch_size, num_packages]
                targets = batch['purchased']['package_ids']  # [batch_size]
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Combine all results
        all_predictions = torch.cat(all_predictions, dim=0)  # [total_samples, num_packages]
        all_targets = torch.cat(all_targets, dim=0)  # [total_samples]
        
        print(f"Processing {len(all_predictions)} predictions...")
        
        # Calculate metrics for each sample
        results = {f'hit@{k}': [] for k in k_values}
        results['rank'] = []
        results['mrr'] = []
        
        # Get package ID mapping
        idx_to_package = {v: k for k, v in self.session_processor.package_to_idx.items()}
        
        for i in tqdm(range(len(all_predictions)), desc="Computing metrics"):
            predictions = all_predictions[i]  # [num_packages]
            target_idx = all_targets[i].item()
            
            # Get top 100 predictions
            scores, indices = torch.topk(predictions, k=min(100, len(predictions)))
            
            # Convert to package IDs and find target rank
            rec_package_ids = []
            for idx in indices:
                package_id = idx_to_package.get(idx.item())
                if package_id:
                    rec_package_ids.append(str(package_id))
            
            # Get target package ID
            target_package = idx_to_package.get(target_idx)
            if target_package:
                target_package = str(target_package)
            
            # Find rank of target
            if target_package and target_package in rec_package_ids:
                rank = rec_package_ids.index(target_package) + 1
                results['rank'].append(rank)
                results['mrr'].append(1.0 / rank)
                
                # Calculate hits at different k values
                for k in k_values:
                    hit = 1.0 if rank <= k else 0.0
                    results[f'hit@{k}'].append(hit)
            else:
                # Not in top 100
                results['rank'].append(101)  # Beyond top 100
                results['mrr'].append(0.0)
                for k in k_values:
                    results[f'hit@{k}'].append(0.0)
        
        # Calculate metrics
        metrics = {}
        for k in k_values:
            metrics[f'recall@{k}'] = np.mean(results[f'hit@{k}']) * 100
        
        metrics['mrr'] = np.mean(results['mrr'])
        metrics['mean_rank'] = np.mean(results['rank'])
        metrics['median_rank'] = np.median(results['rank'])
        
        # Rank distribution
        ranks = results['rank']
        metrics['rank_distribution'] = {
            'top_1': sum(1 for r in ranks if r == 1) / len(ranks) * 100,
            'top_5': sum(1 for r in ranks if r <= 5) / len(ranks) * 100,
            'top_10': sum(1 for r in ranks if r <= 10) / len(ranks) * 100,
            'top_20': sum(1 for r in ranks if r <= 20) / len(ranks) * 100,
            'top_50': sum(1 for r in ranks if r <= 50) / len(ranks) * 100,
            'beyond_100': sum(1 for r in ranks if r > 100) / len(ranks) * 100
        }
        
        return metrics
    
    def analyze_warm_cold_performance(self, warm_users: Dict, cold_users: Dict,
                                    k_values: List[int] = [1, 5, 10, 20, 50]) -> Dict:
        """Separate analysis for warm and cold start users"""
        print("\nAnalyzing warm vs cold start performance...")
        
        # Warm start analysis
        warm_purchases = []
        for user_purchases in warm_users.values():
            warm_purchases.extend(user_purchases)
        
        print(f"\nEvaluating {len(warm_purchases)} warm start purchases...")
        warm_metrics = self.evaluate_ranking_performance(warm_purchases, k_values)
        
        # Cold start analysis
        cold_purchases = []
        for user_purchases in cold_users.values():
            cold_purchases.extend(user_purchases)
        
        print(f"\nEvaluating {len(cold_purchases)} cold start purchases...")
        cold_metrics = self.evaluate_ranking_performance(cold_purchases, k_values)
        
        return {
            'warm_start': warm_metrics,
            'cold_start': cold_metrics
        }
    
    def analyze_example_users(self, warm_users: Dict, train_samples: List[Dict], 
                            num_examples: int = 3) -> List[Dict]:
        """Detailed analysis of example warm start users"""
        print(f"\nAnalyzing {num_examples} example warm start users...")
        
        # Select users with multiple test purchases for interesting examples
        eligible_users = [(user_id, purchases) for user_id, purchases in warm_users.items() 
                         if len(purchases) >= 2]
        
        if len(eligible_users) < num_examples:
            # Fall back to any warm users
            eligible_users = list(warm_users.items())
        
        # Random sample
        random.seed(42)
        selected_users = random.sample(eligible_users, min(num_examples, len(eligible_users)))
        
        examples = []
        
        for user_id, test_purchases in selected_users:
            print(f"\n--- Analyzing user {user_id} ---")
            
            # Get user's training history
            train_history = [s for s in train_samples if s['user_id'] == user_id]
            
            # Extract purchased packages from training
            train_packages = set()
            for sample in train_history:
                if sample.get('is_purchase', False):
                    train_packages.add(str(sample['purchased_package']))
            
            # Get package names
            package_idx_to_id = {v: k for k, v in self.session_processor.package_to_idx.items()}
            package_names = {}
            if hasattr(self.package_processor, 'feed_df'):
                for pkg_id in list(train_packages) + [str(p['purchased_package']) for p in test_purchases]:
                    if pkg_id in self.package_processor.feed_df.index:
                        package_names[pkg_id] = self.package_processor.feed_df.loc[pkg_id, 'title']
            
            user_analysis = {
                'user_id': user_id,
                'train_purchases': len(train_packages),
                'train_interactions': len(train_history),
                'test_purchases': len(test_purchases),
                'train_packages': list(train_packages)[:5],  # First 5 for brevity
                'recommendations_analysis': []
            }
            
            # Analyze each test purchase
            for i, test_purchase in enumerate(test_purchases[:3]):  # Max 3 purchases per user
                target_package = str(test_purchase['purchased_package'])
                target_name = package_names.get(target_package, f"Package {target_package}")
                
                # For user examples, we'll get recommendations more efficiently
                # by just using the model directly on this sample
                try:
                    from utils.training_utils import create_dataloaders, move_batch_to_device
                    idx_to_package = {v: k for k, v in self.session_processor.package_to_idx.items()}
                    
                    # Create a mini batch for this one sample
                    single_sample = [test_purchase]
                    _, mini_loader = create_dataloaders(
                        train_samples=single_sample,
                        test_samples=single_sample,
                        package_processor=self.package_processor,
                        session_processor=self.session_processor,
                        batch_size=1,
                        num_workers=0,
                        use_weighted_sampling=False
                    )
                    
                    # Get predictions
                    recommendations = []
                    with torch.no_grad():
                        for batch in mini_loader:
                            batch = move_batch_to_device(batch, self.device)
                            outputs = self.model(batch)
                            predictions = outputs['predictions'][0]  # First sample
                            
                            # Get top 20
                            scores, indices = torch.topk(predictions, k=20)
                            
                            # Convert to recommendations
                            for score, idx in zip(scores.cpu().numpy(), indices.cpu().numpy()):
                                package_id = idx_to_package.get(idx)
                                if package_id:
                                    recommendations.append({
                                        'package_id': package_id,
                                        'score': float(score)
                                    })
                            break
                except Exception as e:
                    print(f"Error getting recommendations for user {user_id}: {e}")
                    recommendations = []
                
                # Find rank
                rec_ids = [str(rec['package_id']) for rec in recommendations]
                if target_package in rec_ids:
                    rank = rec_ids.index(target_package) + 1
                else:
                    rank = ">20"
                
                # Get top 5 recommendations with names
                top_5_recs = []
                for j, rec in enumerate(recommendations[:5]):
                    pkg_id = str(rec['package_id'])
                    pkg_name = package_names.get(pkg_id, f"Package {pkg_id}")
                    top_5_recs.append({
                        'rank': j + 1,
                        'package_id': pkg_id,
                        'name': pkg_name,
                        'score': float(rec['score'])
                    })
                
                purchase_analysis = {
                    'target_package': target_package,
                    'target_name': target_name,
                    'rank': rank,
                    'purchase_date': test_purchase.get('session_start', 'Unknown'),
                    'top_5_recommendations': top_5_recs
                }
                
                user_analysis['recommendations_analysis'].append(purchase_analysis)
            
            examples.append(user_analysis)
            
            # Print summary
            print(f"  Training: {len(train_packages)} purchases, {len(train_history)} interactions")
            print(f"  Test: {len(test_purchases)} purchases")
            for i, analysis in enumerate(user_analysis['recommendations_analysis']):
                print(f"  Purchase {i+1}: '{analysis['target_name']}' - Rank: {analysis['rank']}")
        
        return examples
    
    def run_verification(self) -> Dict:
        """Run complete verification pipeline"""
        # Prepare data
        train_samples, test_purchases, warm_users, cold_users = self.prepare_test_data()
        
        # Overall performance
        print("\n" + "="*80)
        print("OVERALL PERFORMANCE ON TEST PURCHASES")
        print("="*80)
        overall_metrics = self.evaluate_ranking_performance(test_purchases)
        
        # Warm vs Cold analysis
        print("\n" + "="*80)
        print("WARM VS COLD START ANALYSIS")
        print("="*80)
        warm_cold_metrics = self.analyze_warm_cold_performance(warm_users, cold_users)
        
        # Example users
        print("\n" + "="*80)
        print("DETAILED USER EXAMPLES")
        print("="*80)
        examples = self.analyze_example_users(warm_users, train_samples)
        
        # Compile results
        results = {
            'model_info_path': self.model_info_path,
            'model_type': self.model_info.get('model_type', 'unknown'),
            'overall_performance': overall_metrics,
            'warm_cold_comparison': warm_cold_metrics,
            'user_examples': examples,
            'test_statistics': {
                'total_test_purchases': len(test_purchases),
                'warm_start_users': len(warm_users),
                'cold_start_users': len(cold_users),
                'warm_start_purchases': sum(len(p) for p in warm_users.values()),
                'cold_start_purchases': sum(len(p) for p in cold_users.values())
            }
        }
        
        # Print final summary
        self._print_summary(results)
        
        # Save results
        output_file = 'verification_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {output_file}")
        
        return results
    
    def _print_summary(self, results: Dict):
        """Print results summary"""
        print("\n" + "="*80)
        print("VERIFICATION SUMMARY")
        print("="*80)
        
        overall = results['overall_performance']
        print("\nOverall Performance:")
        print(f"  Recall@1: {overall['recall@1']:.1f}%")
        print(f"  Recall@5: {overall['recall@5']:.1f}%")
        print(f"  Recall@10: {overall['recall@10']:.1f}%")
        print(f"  Recall@20: {overall['recall@20']:.1f}%")
        print(f"  MRR: {overall['mrr']:.4f}")
        print(f"  Mean Rank: {overall['mean_rank']:.1f}")
        print(f"  Median Rank: {overall['median_rank']:.0f}")
        
        print("\nRank Distribution:")
        dist = overall['rank_distribution']
        print(f"  Top 1: {dist['top_1']:.1f}%")
        print(f"  Top 5: {dist['top_5']:.1f}%")
        print(f"  Top 10: {dist['top_10']:.1f}%")
        print(f"  Top 20: {dist['top_20']:.1f}%")
        print(f"  Beyond 100: {dist['beyond_100']:.1f}%")
        
        warm_cold = results['warm_cold_comparison']
        print("\nWarm vs Cold Start:")
        print(f"  Warm Recall@20: {warm_cold['warm_start']['recall@20']:.1f}%")
        print(f"  Cold Recall@20: {warm_cold['cold_start']['recall@20']:.1f}%")
        print(f"  Warm MRR: {warm_cold['warm_start']['mrr']:.4f}")
        print(f"  Cold MRR: {warm_cold['cold_start']['mrr']:.4f}")
        
        stats = results['test_statistics']
        print("\nTest Data Statistics:")
        print(f"  Total purchases: {stats['total_test_purchases']:,}")
        print(f"  Warm users: {stats['warm_start_users']:,} ({stats['warm_start_purchases']:,} purchases)")
        print(f"  Cold users: {stats['cold_start_users']:,} ({stats['cold_start_purchases']:,} purchases)")


def main():
    parser = argparse.ArgumentParser(description='Verify NATR model performance on unseen test data')
    parser.add_argument('--model-info', type=str, 
                       default='output/model_info/model_info_enhanced_pretrain_finetune.json',
                       help='Path to model_info.json file')
    
    args = parser.parse_args()
    
    # Validate file exists
    if not os.path.exists(args.model_info):
        print(f"Error: Model info file not found: {args.model_info}")
        return 1
    
    try:
        # Run verification
        verifier = ModelPerformanceVerifier(args.model_info)
        results = verifier.run_verification()
        
        return 0
        
    except Exception as e:
        print(f"\nError during verification: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
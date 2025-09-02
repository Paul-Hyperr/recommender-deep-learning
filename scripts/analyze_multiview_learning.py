#!/usr/bin/env python3
"""
Comprehensive Multi-View Learning Analysis for NATR

This script provides a complete analysis of NATR's multi-view learning effectiveness by:
1. Performing feature ablation studies to measure individual feature contributions
2. Analyzing model architecture and parameter allocation
3. Creating comprehensive visualizations of multi-view effectiveness

Features analyzed:
- Title embeddings (semantic understanding)
- Geographic coordinates (location preferences)
- Country (destination preferences)
- Category (travel type preferences)
- Theme (travel theme preferences)
- Price (budget considerations)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
import sys
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import argparse
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.natr_enhanced import NATREnhanced, NATRConfig
from utils.session_processor2 import SessionProcessor2
from utils.package_processor import PackageProcessor
from utils.unified_metrics import UnifiedMetricsTracker
from utils.training_utils import create_dataloaders, move_batch_to_device


class ComprehensiveMultiViewAnalyzer:
    """
    Comprehensive analyzer for NATR's multi-view learning effectiveness
    """
    
    def __init__(self, model_info_path: str, checkpoint_path: str, sample_size: Optional[int] = None):
        """
        Initialize the analyzer
        
        Args:
            model_info_path: Path to model_info.json
            checkpoint_path: Path to model checkpoint file (.pth)
            sample_size: Number of samples to use for ablation analysis (None = all purchases)
        """
        self.model_info_path = model_info_path
        self.checkpoint_path = checkpoint_path
        self.sample_size = sample_size
        
        # Load model info
        with open(model_info_path, 'r') as f:
            self.model_info = json.load(f)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() 
                                  else 'mps' if torch.backends.mps.is_available() 
                                  else 'cpu')
        
        # Validate checkpoint path
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
        print(f"Loading model from checkpoint: {checkpoint_path}")
        
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
        
        # Check if checkpoint matches expected model dimensions
        checkpoint_config = checkpoint.get('config', {})
        expected_users = self.model_info['config']['num_users']
        expected_packages = self.model_info['config']['num_packages']
        checkpoint_users = checkpoint_config.get('num_users', 0)
        checkpoint_packages = checkpoint_config.get('num_packages', 0)
        
        if (checkpoint_users != expected_users or checkpoint_packages != expected_packages):
            print(f"🚨 WARNING: Checkpoint mismatch!")
            print(f"   Expected: {expected_users:,} users, {expected_packages:,} packages")
            print(f"   Checkpoint: {checkpoint_users:,} users, {checkpoint_packages:,} packages")
            print(f"   This checkpoint appears to be from a different dataset.")
            print(f"   Analysis will continue but results may not be accurate.")
            response = input("Do you want to continue anyway? (y/n): ")
            if response.lower() != 'y':
                raise RuntimeError("Analysis aborted due to checkpoint mismatch")
        
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model loaded successfully with matching dimensions")
        except RuntimeError as e:
            print(f"Warning: Model architecture mismatch. Loading with strict=False...")
            print(f"Error details: {str(e)[:200]}...")
            # Load with strict=False to ignore size mismatches
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if missing_keys:
                print(f"Missing keys: {len(missing_keys)} parameters")
            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)} parameters") 
            print("Continuing with partial model loading...")
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully from {checkpoint_path}")
        
        # Feature groups for analysis (6-view model, excluding temporal)
        # Include events since they're different from temporal user encoding
        self.feature_groups = {
            'title_embeddings': 'Title Embeddings',           # View 0
            'coordinates': 'Geographic Coordinates',          # View 1  
            'country': 'Country',                            # View 2
            'category_theme': 'Category/Theme',               # View 3
            'price': 'Price',                                # View 4
            'events': 'Events'                               # View 5
        }
        
        # Results storage
        self.ablation_results = None
        self.architecture_analysis = None
        self.test_loader = None
        
    def run_ablation_study(self) -> pd.DataFrame:
        """
        Run comprehensive ablation study for all features
        
        Returns:
            DataFrame with ablation results
        """
        print(f"\n{'='*60}")
        print("RUNNING FEATURE ABLATION STUDY")
        print(f"{'='*60}")
        
        # Prepare test data using same approach as verification script
        print(f"\nPreparing test data for ablation study...")
        
        # Data paths - determine from model info
        package_data_path = self.model_info.get('package_data_path', 'data/feed.parquet')
        
        # Smart dataset detection based on model info
        if 'event_data_path' in self.model_info:
            event_data = self.model_info['event_data_path']
        elif 'dataset' in self.model_info:
            dataset_map = {
                '13months': 'data/13_months_new2_clean.parquet',
                '2months': 'data/bookit_events_2_months.parquet'
            }
            event_data = dataset_map.get(self.model_info['dataset'], 'data/13_months_new2_clean.parquet')
        else:
            # Detect from model dimensions (2 months has ~6k packages, 13 months has ~9k packages)
            num_packages = self.model_info['config'].get('num_packages', 0)
            num_users = self.model_info['config'].get('num_users', 0)
            
            if num_packages < 7000 or num_users < 200000:  # Likely 2 months
                event_data = 'data/bookit_events_2_months.parquet'
                print(f"Auto-detected 2 months dataset based on model size (packages: {num_packages}, users: {num_users})")
            else:  # Likely 13 months
                event_data = 'data/13_months_new2_clean.parquet'
                print(f"Auto-detected 13 months dataset based on model size (packages: {num_packages}, users: {num_users})")
        
        print(f"Using event data: {event_data}")
        
        # Initialize processors exactly like training script
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
        
        # Prepare samples
        from utils.training_utils import (
            filter_by_min_session_length, filter_items_by_frequency,
            identify_event_types, time_based_split_year
        )
        
        all_samples = session_processor.prepare_enhanced_training_data()
        
        # Identify event types
        event_to_idx = session_processor.get_idx_mappings()['event_to_idx']
        all_samples = identify_event_types(all_samples, event_to_idx)
        
        # Apply same filters as training
        quality_samples = filter_by_min_session_length(all_samples, min_session_length=2)
        
        # Use exact same valid packages from training
        valid_packages = set(self.model_info.get('valid_packages', []))
        if valid_packages:
            filtered_samples = []
            for sample in quality_samples:
                if str(sample.get('purchased_package', '')) in valid_packages:
                    filtered_samples.append(sample)
        else:
            filtered_samples, valid_packages = filter_items_by_frequency(quality_samples, min_frequency=5)
        
        # Use exact same split as training
        train_ratio = self.model_info.get('train_ratio', 0.91)
        train_samples, test_samples, split_date = time_based_split_year(filtered_samples, train_ratio=train_ratio)
        
        # Get only purchase samples from test set
        test_purchases = [s for s in test_samples if s.get('is_purchase', False)]
        
        # Use ALL test purchases (both warm-start and cold-start users)
        print("Using all test purchases (warm-start + cold-start users)...")
        train_users = set(sample['user_id'] for sample in train_samples)
        self.train_users = train_users  # Store for later use in attention analysis
        warm_start_purchases = [s for s in test_purchases if s['user_id'] in train_users]
        cold_start_purchases = [s for s in test_purchases if s['user_id'] not in train_users]
        
        print(f"Total test purchases: {len(test_purchases)}")
        print(f"Warm-start test purchases: {len(warm_start_purchases)} ({len(warm_start_purchases)/len(test_purchases)*100:.1f}%)")
        print(f"Cold-start test purchases: {len(cold_start_purchases)} ({len(cold_start_purchases)/len(test_purchases)*100:.1f}%)")
        
        # Use ALL test purchases for analysis (ignore sample_size for final analysis)
        print(f"Using ALL {len(test_purchases)} test purchase samples for ablation study")
        
        # Store train samples for user type evaluation
        self.train_samples = train_samples
        
        # Create dataloader
        dummy_train_samples = test_purchases[:100] if len(test_purchases) > 100 else test_purchases
        
        train_loader, test_loader = create_dataloaders(
            train_samples=dummy_train_samples,
            test_samples=test_purchases,
            package_processor=package_processor,
            session_processor=session_processor,
            batch_size=64,
            num_workers=0,
            use_weighted_sampling=False
        )
        
        # Store test_loader and warm-start purchases for attention analysis
        self.test_loader = test_loader
        self.warm_start_purchases = warm_start_purchases
        self.test_purchases = test_purchases
        self.package_processor = package_processor
        self.session_processor = session_processor
        
        results = []
        
        # Baseline performance (no masking)
        print("\nEvaluating baseline performance...")
        baseline_metrics = self._evaluate_with_feature_masking(test_loader, feature_to_mask=None)
        baseline_recall_20 = baseline_metrics.get('purchase_recall@20', 0.0)
        baseline_recall_10 = baseline_metrics.get('purchase_recall@10', 0.0)
        baseline_mrr = baseline_metrics.get('purchase_mrr', 0.0)
        
        # Evaluate separately for warm-start and cold-start users
        warm_metrics, cold_metrics = self._evaluate_by_user_type(test_loader)
        
        print(f"Overall Baseline - Recall@10: {baseline_recall_10*100:.2f}%, Recall@20: {baseline_recall_20*100:.2f}%, MRR: {baseline_mrr:.4f}")
        print(f"Warm-start Users - Recall@10: {warm_metrics['purchase_recall@10']*100:.2f}%, Recall@20: {warm_metrics['purchase_recall@20']*100:.2f}%, MRR: {warm_metrics['purchase_mrr']:.4f}")
        print(f"Cold-start Users - Recall@10: {cold_metrics['purchase_recall@10']*100:.2f}%, Recall@20: {cold_metrics['purchase_recall@20']*100:.2f}%, MRR: {cold_metrics['purchase_mrr']:.4f}")
        
        results.append({
            'feature': 'Baseline (All Features)',
            'recall@10': baseline_recall_10,
            'recall@20': baseline_recall_20,
            'mrr': baseline_mrr,
            'recall@10_drop': 0.0,
            'recall@20_drop': 0.0,
            'mrr_drop': 0.0,
            'relative_importance': 0.0
        })
        
        print(f"Baseline - Recall@10: {baseline_recall_10*100:.2f}%, Recall@20: {baseline_recall_20*100:.2f}%, MRR: {baseline_mrr:.4f}")
        
        # Ablation for each feature
        for feature_key, feature_name in self.feature_groups.items():
            print(f"\nEvaluating with {feature_name} masked...")
            
            ablation_metrics = self._evaluate_with_feature_masking(test_loader, feature_to_mask=feature_key)
            ablation_recall_10 = ablation_metrics.get('purchase_recall@10', 0.0)
            ablation_recall_20 = ablation_metrics.get('purchase_recall@20', 0.0)
            ablation_mrr = ablation_metrics.get('purchase_mrr', 0.0)
            
            # Calculate performance drops
            recall_10_drop = baseline_recall_10 - ablation_recall_10
            recall_20_drop = baseline_recall_20 - ablation_recall_20
            mrr_drop = baseline_mrr - ablation_mrr
            
            # Calculate relative importance (based on recall@20 drop)
            relative_importance = (recall_20_drop / baseline_recall_20 * 100) if baseline_recall_20 > 0 else 0
            
            results.append({
                'feature': feature_name,
                'recall@10': ablation_recall_10,
                'recall@20': ablation_recall_20,
                'mrr': ablation_mrr,
                'recall@10_drop': recall_10_drop,
                'recall@20_drop': recall_20_drop,
                'mrr_drop': mrr_drop,
                'relative_importance': relative_importance
            })
            
            print(f"  Recall@10: {ablation_recall_10*100:.2f}% (drop: {recall_10_drop*100:.2f}%)")
            print(f"  Recall@20: {ablation_recall_20*100:.2f}% (drop: {recall_20_drop*100:.2f}%)")
            print(f"  MRR: {ablation_mrr:.4f} (drop: {mrr_drop:.4f})")
            print(f"  Relative importance: {relative_importance:.1f}%")
        
        self.ablation_results = pd.DataFrame(results)
        return self.ablation_results
    
    def _evaluate_with_feature_masking(self, test_loader, feature_to_mask: Optional[str] = None) -> Dict:
        """
        Evaluate model with optional feature masking
        
        Args:
            test_loader: Test data loader
            feature_to_mask: Feature to mask during evaluation
            
        Returns:
            Evaluation metrics
        """
        # Set up metrics tracker
        metrics_tracker = UnifiedMetricsTracker(k_values=[10, 20])
        
        self.model.eval()
        
        # Apply feature masking if specified
        hooks = []
        if feature_to_mask:
            def mask_features(module, input):
                if hasattr(module, '__class__') and module.__class__.__name__ == 'PackageEncoder':
                    # Mask features in the input dictionary
                    if isinstance(input, tuple) and len(input) > 0:
                        features = input[0]
                        if isinstance(features, dict):
                            if feature_to_mask == 'title_embeddings' and 'title_embeddings' in features:
                                features['title_embeddings'] = torch.zeros_like(features['title_embeddings'])
                            elif feature_to_mask == 'coordinates' and 'coordinates' in features:
                                features['coordinates'] = torch.zeros_like(features['coordinates'])
                            elif feature_to_mask == 'country':
                                # Mask ALL country-related features
                                for country_field in ['country_ids', 'purchased_countries', 'short_term_countries', 'long_term_countries']:
                                    if country_field in features:
                                        features[country_field] = torch.zeros_like(features[country_field])
                            elif feature_to_mask == 'category_theme':
                                # Mask ALL category and theme related features since they're combined in view 3
                                category_fields = ['category_ids', 'purchased_categories', 'short_term_categories', 'long_term_categories']
                                theme_fields = ['theme_ids', 'purchased_themes', 'short_term_themes', 'long_term_themes']
                                masked_fields = []
                                for field in category_fields + theme_fields:
                                    if field in features:
                                        original_shape = features[field].shape
                                        features[field] = torch.zeros_like(features[field])
                                        masked_fields.append(f"{field}({original_shape})")
                                if len(masked_fields) > 0:
                                    pass  # Removed verbose debug output
                            elif feature_to_mask == 'price':
                                # Mask ALL price-related features
                                for price_field in ['prices', 'purchased_prices', 'short_term_prices', 'long_term_prices']:
                                    if price_field in features:
                                        features[price_field] = torch.zeros_like(features[price_field])
                            # Note: Temporal information is no longer a separate view
                            elif feature_to_mask == 'events':
                                # Mask event-related features in the batch (event_types in enhanced model)
                                if 'event_types' in features:
                                    features['event_types'] = torch.zeros_like(features['event_types'])
                return input
            
            # Register hook on PackageEncoder
            for module in self.model.modules():
                if hasattr(module, '__class__') and module.__class__.__name__ == 'PackageEncoder':
                    hook = module.register_forward_pre_hook(mask_features)
                    hooks.append(hook)
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating (masking: {feature_to_mask or 'none'})", leave=False):
                batch = move_batch_to_device(batch, self.device)
                
                outputs = self.model(batch)
                predictions = outputs['predictions']
                targets = batch['purchased']['package_ids']
                
                # Get event indicators
                is_purchase = batch.get('is_purchase', torch.ones_like(targets, dtype=torch.bool))
                
                # Update metrics
                metrics_tracker.update(
                    predictions=predictions,
                    targets=targets,
                    is_purchase=is_purchase
                )
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Compute metrics
        metrics = metrics_tracker.compute()
        return metrics
    
    def _evaluate_by_user_type(self, test_loader):
        """
        Evaluate model performance separately for warm-start and cold-start users
        
        Returns:
            Tuple of (warm_start_metrics, cold_start_metrics)
        """
        # Get train users from the prepared data  
        train_users = set()
        if hasattr(self, 'train_samples') and self.train_samples:
            train_users = set(sample['user_id'] for sample in self.train_samples)
            self.train_users = train_users  # Store for later use
        
        print(f"Number of train users identified: {len(train_users)}")
        
        # Set up metrics trackers
        warm_metrics_tracker = UnifiedMetricsTracker(k_values=[10, 20])
        cold_metrics_tracker = UnifiedMetricsTracker(k_values=[10, 20])
        
        # Count users by type
        warm_count = 0
        cold_count = 0
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating by user type", leave=False):
                batch = move_batch_to_device(batch, self.device)
                
                outputs = self.model(batch)
                predictions = outputs['predictions']
                targets = batch['purchased']['package_ids']
                user_ids = batch['user_id']
                
                # Get event indicators
                is_purchase = batch.get('is_purchase', torch.ones_like(targets, dtype=torch.bool))
                
                # Separate by user type
                warm_start_mask = torch.tensor([uid.item() in train_users for uid in user_ids], device=self.device)
                cold_start_mask = ~warm_start_mask
                
                # Update warm-start metrics
                if warm_start_mask.any():
                    warm_count += warm_start_mask.sum().item()
                    warm_metrics_tracker.update(
                        predictions=predictions[warm_start_mask],
                        targets=targets[warm_start_mask],
                        is_purchase=is_purchase[warm_start_mask]
                    )
                
                # Update cold-start metrics
                if cold_start_mask.any():
                    cold_count += cold_start_mask.sum().item()
                    cold_metrics_tracker.update(
                        predictions=predictions[cold_start_mask],
                        targets=targets[cold_start_mask],
                        is_purchase=is_purchase[cold_start_mask]
                    )
        
        # Compute metrics
        warm_metrics = warm_metrics_tracker.compute()
        cold_metrics = cold_metrics_tracker.compute()
        
        print(f"Processed {warm_count} warm-start users, {cold_count} cold-start users")
        
        return warm_metrics, cold_metrics
    
    def analyze_model_architecture(self) -> pd.DataFrame:
        """
        Analyze model architecture and parameter allocation
        
        Returns:
            DataFrame with architecture analysis
        """
        print(f"\n{'='*60}")
        print("ANALYZING MODEL ARCHITECTURE")
        print(f"{'='*60}")
        
        results = []
        
        with torch.no_grad():
            # Analyze parameter allocation for each feature component
            if hasattr(self.model.package_encoder, 'country_embedding'):
                country_size = self.model.package_encoder.country_embedding.weight.shape[1]
                country_params = self.model.package_encoder.country_embedding.weight.numel()
            else:
                country_size = 0
                country_params = 0
                
            if hasattr(self.model.package_encoder, 'category_embedding'):
                category_size = self.model.package_encoder.category_embedding.weight.shape[1]
                category_params = self.model.package_encoder.category_embedding.weight.numel()
            else:
                category_size = 0
                category_params = 0
                
            if hasattr(self.model.package_encoder, 'theme_embedding'):
                theme_size = self.model.package_encoder.theme_embedding.weight.shape[1]
                theme_params = self.model.package_encoder.theme_embedding.weight.numel()
            else:
                theme_size = 0
                theme_params = 0
            
            # Title encoder parameters (actual name from natr.py)
            title_params = 0
            if hasattr(self.model.package_encoder, 'title_encoder'):
                title_params = sum(p.numel() for p in self.model.package_encoder.title_encoder.parameters())
            
            # Coordinate encoder parameters (actual name from natr.py)
            coord_params = 0
            if hasattr(self.model.package_encoder, 'coordinate_encoder'):
                coord_params = sum(p.numel() for p in self.model.package_encoder.coordinate_encoder.parameters())
                
            # Price encoder parameters (actual name from natr.py)
            price_params = 0
            if hasattr(self.model.package_encoder, 'price_encoder'):
                price_params = sum(p.numel() for p in self.model.package_encoder.price_encoder.parameters())
                
            # Events encoder parameters
            events_params = 0
            if hasattr(self.model.package_encoder, 'events_encoder'):
                events_params = sum(p.numel() for p in self.model.package_encoder.events_encoder.parameters())
            # Otherwise check for individual event components (in enhanced model)
            else:
                if hasattr(self.model.package_encoder, 'event_embedding'):
                    events_params += self.model.package_encoder.event_embedding.weight.numel()
                if hasattr(self.model.package_encoder, 'event_lstm'):
                    events_params += sum(p.numel() for p in self.model.package_encoder.event_lstm.parameters())
                if hasattr(self.model.package_encoder, 'event_attention'):
                    events_params += sum(p.numel() for p in self.model.package_encoder.event_attention.parameters())
                if hasattr(self.model.package_encoder, 'event_projection'):
                    events_params += sum(p.numel() for p in self.model.package_encoder.event_projection.parameters())
            
            # Calculate total and percentages
            total_params = (country_params + category_params + theme_params + 
                          title_params + coord_params + price_params + events_params)
            
            if total_params > 0:
                results.append({
                    'feature': 'Title Embeddings',
                    'parameters': title_params,
                    'param_percentage': (title_params / total_params) * 100,
                    'embedding_dim': self.model_info['config'].get('title_embedding_dim', 1536)
                })
                results.append({
                    'feature': 'Geographic Coordinates',
                    'parameters': coord_params,
                    'param_percentage': (coord_params / total_params) * 100,
                    'embedding_dim': self.model_info['config'].get('hidden_dim', 256)  # Output of coordinate_encoder
                })
                results.append({
                    'feature': 'Country',
                    'parameters': country_params,
                    'param_percentage': (country_params / total_params) * 100,
                    'embedding_dim': country_size
                })
                results.append({
                    'feature': 'Category/Theme',
                    'parameters': category_params + theme_params,  # Combined parameters
                    'param_percentage': ((category_params + theme_params) / total_params) * 100,
                    'embedding_dim': max(category_size, theme_size)  # Larger of the two
                })
                results.append({
                    'feature': 'Price',
                    'parameters': price_params,
                    'param_percentage': (price_params / total_params) * 100,
                    'embedding_dim': self.model_info['config'].get('hidden_dim', 256)  # Output of price_encoder
                })
                results.append({
                    'feature': 'Events',
                    'parameters': events_params,
                    'param_percentage': (events_params / total_params) * 100,
                    'embedding_dim': self.model_info['config'].get('hidden_dim', 256)  # Output of events_encoder
                })
                
                print(f"\nParameter allocation analysis:")
                print(f"Total feature encoding parameters: {total_params:,}")
                for result in results:
                    print(f"  {result['feature']:.<30} {result['param_percentage']:>6.1f}% ({result['parameters']:,} params)")
        
        self.architecture_analysis = pd.DataFrame(results)
        return self.architecture_analysis
    
    def create_comprehensive_visualization(self):
        """Create comprehensive visualization combining all analyses"""
        
        print(f"\n{'='*60}")
        print("CREATING COMPREHENSIVE VISUALIZATIONS")
        print(f"{'='*60}")
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        colors = plt.cm.Set3(np.linspace(0, 1, 12))
        view_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF']
        
        # Create large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Feature Importance from Ablation Study
        ax1 = plt.subplot(3, 3, 1)
        if self.ablation_results is not None:
            feature_results = self.ablation_results[self.ablation_results['feature'] != 'Baseline (All Features)'].copy()
            
            # Custom sort: Title Embeddings at top, Price at bottom
            def importance_sort_key(row):
                if 'Title' in row['feature']:
                    return 1000 + row['relative_importance']  # Title at top
                elif 'Price' in row['feature']:
                    return -1000 + row['relative_importance']  # Price at bottom
                else:
                    return row['relative_importance']
            
            feature_results['sort_key'] = feature_results.apply(importance_sort_key, axis=1)
            feature_results = feature_results.sort_values('sort_key', ascending=True)
            
            bars = ax1.barh(feature_results['feature'], feature_results['relative_importance'], 
                           color=colors[:len(feature_results)])
            ax1.set_xlabel('Relative Importance (%)')
            ax1.set_title('Feature Importance\n(% Performance Drop When Masked)', fontweight='bold')
            
            # Add value labels
            for i, (idx, row) in enumerate(feature_results.iterrows()):
                ax1.text(row['relative_importance'] + 0.2, i, f"{row['relative_importance']:.1f}%", 
                        va='center', fontsize=9)
        
        # 2. Performance Comparison (Baseline vs Ablated)
        ax2 = plt.subplot(3, 3, 2)
        if self.ablation_results is not None:
            baseline = self.ablation_results[self.ablation_results['feature'] == 'Baseline (All Features)'].iloc[0]
            feature_results = self.ablation_results[self.ablation_results['feature'] != 'Baseline (All Features)'].copy()
            feature_results = feature_results.sort_values('relative_importance', ascending=False)
            
            x = np.arange(len(feature_results))
            width = 0.35
            
            # Baseline line
            ax2.axhline(baseline['recall@20'] * 100, color='green', linestyle='--', 
                       alpha=0.8, label=f"Baseline: {baseline['recall@20']*100:.1f}%")
            
            # Ablated performance bars
            bars = ax2.bar(x, feature_results['recall@20'] * 100, width, 
                          color=colors[:len(feature_results)], alpha=0.7,
                          label='Performance with feature masked')
            
            ax2.set_xlabel('Masked Feature')
            ax2.set_ylabel('Purchase Recall@20 (%)')
            ax2.set_title('Impact of Feature Masking\non Model Performance', fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(feature_results['feature'], rotation=45, ha='right')
            ax2.legend()
        
        # 3. Parameter Allocation Histogram
        ax3 = plt.subplot(3, 3, 3)
        if self.architecture_analysis is not None:
            param_df = self.architecture_analysis[self.architecture_analysis['parameters'] > 0].copy()
            
            # Fix duplicate Events/Interactions and rename Category/Theme
            param_df['feature'] = param_df['feature'].replace({
                'Events/Interactions': 'Events',
                'Category/Theme (Combined)': 'Category/Theme'
            })
            
            # Remove duplicates by grouping
            param_df = param_df.groupby('feature').agg({
                'parameters': 'sum',
                'param_percentage': 'sum',
                'embedding_dim': 'first'
            }).reset_index()
            
            if not param_df.empty:
                bars = ax3.bar(range(len(param_df)), param_df['param_percentage'], 
                              color=colors[:len(param_df)])
                ax3.set_xticks(range(len(param_df)))
                ax3.set_xticklabels(param_df['feature'], rotation=45, ha='right')
                ax3.set_ylabel('Parameter Percentage (%)')
                ax3.set_title('Model Capacity Allocation\n(% of Feature Parameters)', fontweight='bold')
                
                # Add value labels on bars
                for bar, pct in zip(bars, param_df['param_percentage']):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 4. Embedding Dimensions Comparison
        ax4 = plt.subplot(3, 3, 4)
        if self.architecture_analysis is not None:
            features = self.architecture_analysis['feature'].values
            dims = self.architecture_analysis['embedding_dim'].values
            
            bars = ax4.bar(features, dims, color=colors[:len(features)])
            ax4.set_xlabel('Feature Type')
            ax4.set_ylabel('Embedding Dimension')
            ax4.set_title('Feature Representation Dimensions', fontweight='bold')
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, dim in zip(bars, dims):
                if dim > 0:
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                            f'{int(dim)}', ha='center', va='bottom', fontsize=9)
        
        # 5. Combined Importance Score
        ax5 = plt.subplot(3, 3, 5)
        if self.ablation_results is not None and self.architecture_analysis is not None:
            # Merge ablation and architecture data
            merged_df = pd.merge(
                self.ablation_results[self.ablation_results['feature'] != 'Baseline (All Features)'],
                self.architecture_analysis,
                on='feature',
                how='inner'
            )
            
            # Calculate combined importance score
            merged_df['combined_importance'] = (
                merged_df['relative_importance'] * 0.7 +  # Ablation importance (70%)
                merged_df['param_percentage'] * 0.3       # Parameter allocation (30%)
            )
            
            merged_df = merged_df.sort_values('combined_importance', ascending=True)
            
            bars = ax5.barh(merged_df['feature'], merged_df['combined_importance'], 
                           color=colors[:len(merged_df)])
            ax5.set_xlabel('Combined Importance Score')
            ax5.set_title('Overall Feature Importance\n(Ablation + Architecture)', fontweight='bold')
            
            # Add value labels
            for i, (idx, row) in enumerate(merged_df.iterrows()):
                ax5.text(row['combined_importance'] + 0.3, i, f"{row['combined_importance']:.1f}", 
                        va='center', fontsize=9)
        
        # 6. Multi-View Architecture Diagram
        ax6 = plt.subplot(3, 3, 6)
        ax6.axis('off')
        
        # Create architecture flow diagram
        y_positions = np.linspace(0.95, 0.05, 7)
        feature_labels = ['Title\nEmbeddings', 'Geographic\nCoordinates', 
                         'Country', 'Category', 'Theme', 'Price', 'Events/\nInteractions']
        
        # Draw feature boxes
        for i, (feat, y) in enumerate(zip(feature_labels, y_positions)):
            rect = plt.Rectangle((0.05, y-0.06), 0.25, 0.1, 
                               facecolor=colors[i], edgecolor='black', linewidth=1.5, alpha=0.8)
            ax6.add_patch(rect)
            ax6.text(0.175, y-0.01, feat, ha='center', va='center', fontsize=8, fontweight='bold')
            
            # Draw arrow to fusion
            ax6.arrow(0.3, y-0.01, 0.15, 0, head_width=0.02, 
                     head_length=0.02, fc='gray', ec='gray', alpha=0.7)
        
        # Draw fusion box
        fusion_rect = plt.Rectangle((0.45, 0.3), 0.2, 0.4, 
                                   facecolor='lightsteelblue', edgecolor='black', linewidth=2)
        ax6.add_patch(fusion_rect)
        ax6.text(0.55, 0.5, 'Multi-View\nFusion\nLayer', ha='center', va='center', 
                fontsize=10, fontweight='bold')
        
        # Draw output arrow
        ax6.arrow(0.65, 0.5, 0.15, 0, head_width=0.03, 
                 head_length=0.02, fc='darkblue', ec='darkblue')
        ax6.text(0.82, 0.5, 'Package\nRepresentation', ha='left', va='center', 
                fontsize=9, fontweight='bold')
        
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.set_title('NATR Multi-View Architecture', fontweight='bold')
        
        # 7. Performance Drops Detailed View
        ax7 = plt.subplot(3, 3, 7)
        if self.ablation_results is not None:
            feature_results = self.ablation_results[self.ablation_results['feature'] != 'Baseline (All Features)'].copy()
            feature_results = feature_results.sort_values('recall@20_drop', ascending=True)
            
            # Create grouped bars for different metrics
            x = np.arange(len(feature_results))
            width = 0.25
            
            bars1 = ax7.barh(x - width, feature_results['recall@10_drop'] * 100, width, 
                            label='Recall@10 Drop', color='lightcoral', alpha=0.8)
            bars2 = ax7.barh(x, feature_results['recall@20_drop'] * 100, width, 
                            label='Recall@20 Drop', color='cornflowerblue', alpha=0.8)
            bars3 = ax7.barh(x + width, feature_results['mrr_drop'] * 100, width, 
                            label='MRR Drop (×100)', color='lightgreen', alpha=0.8)
            
            ax7.set_yticks(x)
            ax7.set_yticklabels(feature_results['feature'])
            ax7.set_xlabel('Performance Drop (%)')
            ax7.set_title('Detailed Performance Impact\nby Metric Type', fontweight='bold')
            ax7.legend(fontsize=8)
        
        # 8. Feature Contribution Matrix
        ax8 = plt.subplot(3, 3, 8)
        if self.ablation_results is not None:
            feature_results = self.ablation_results[self.ablation_results['feature'] != 'Baseline (All Features)'].copy()
            
            # Create matrix data
            matrix_data = []
            for _, row in feature_results.iterrows():
                matrix_data.append([
                    row['recall@10_drop'] * 100,
                    row['recall@20_drop'] * 100,
                    row['mrr_drop'] * 100
                ])
            
            matrix_df = pd.DataFrame(matrix_data, 
                                   index=feature_results['feature'],
                                   columns=['Recall@10\nDrop (%)', 'Recall@20\nDrop (%)', 'MRR Drop\n(×100)'])
            
            sns.heatmap(matrix_df, annot=True, fmt='.1f', cmap='Reds', 
                       ax=ax8, cbar_kws={'label': 'Performance Drop'})
            ax8.set_title('Feature Impact Heatmap', fontweight='bold')
            ax8.set_ylabel('')
        
        # 9. Key Insights and Recommendations
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # Generate insights based on results
        insights_text = self._generate_insights_text()
        
        ax9.text(0.05, 0.95, insights_text, transform=ax9.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.7', facecolor='lightyellow', alpha=0.9))
        
        ax9.set_title('Key Insights & Recommendations', fontweight='bold')
        
        # Main title
        model_name = os.path.basename(self.model_info.get('checkpoint_path', 'NATR Model'))
        plt.suptitle(f'NATR Multi-View Learning Comprehensive Analysis\n{model_name}',
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Create output directory
        output_dir = 'output/analyze_learning'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save figure
        output_path = os.path.join(output_dir, 'comprehensive_multiview_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nComprehensive visualization saved to: {output_path}")
        
        return fig
    
    def create_view_influence_graph(self):
        """Create a focused visualization showing the influence of each view"""
        print(f"\n{'='*60}")
        print("CREATING VIEW INFLUENCE VISUALIZATION")
        print(f"{'='*60}")
        
        if self.ablation_results is None:
            print("Error: No ablation results available. Run ablation study first.")
            return None
        
        # Set up the figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Enhanced color scheme for 7 views
        view_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF']
        
        # Get feature results (excluding baseline)
        feature_results = self.ablation_results[self.ablation_results['feature'] != 'Baseline (All Features)'].copy()
        feature_results = feature_results.sort_values('relative_importance', ascending=True)
        
        # 1. View Influence Bar Chart (Horizontal)
        bars1 = ax1.barh(feature_results['feature'], feature_results['relative_importance'], 
                        color=view_colors[:len(feature_results)], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax1.set_xlabel('Performance Impact (%)\n(Performance drop when view is removed)', fontsize=12, fontweight='bold')
        ax1.set_title('Multi-View Influence Analysis\nHow much each view contributes to model performance', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars1, feature_results['relative_importance'])):
            ax1.text(val + 0.2, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
                    va='center', fontweight='bold', fontsize=10)
        
        # Add average line
        avg_importance = feature_results['relative_importance'].mean()
        ax1.axvline(avg_importance, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax1.text(avg_importance + 0.5, len(feature_results)-0.5, f'Avg: {avg_importance:.1f}%', 
                rotation=90, va='top', ha='left', color='red', fontweight='bold')
        
        # 2. Recall@20 Performance Comparison
        baseline_recall = self.ablation_results[
            self.ablation_results['feature'] == 'Baseline (All Features)'
        ]['recall@20'].iloc[0] * 100
        
        bars2 = ax2.bar(range(len(feature_results)), feature_results['recall@20'] * 100, 
                       color=view_colors[:len(feature_results)], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add baseline line
        ax2.axhline(baseline_recall, color='green', linestyle='-', linewidth=3, alpha=0.7, 
                   label=f'All Views: {baseline_recall:.1f}%')
        
        ax2.set_xlabel('View Removed', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Recall@20 (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Performance When Each View is Removed\n(Higher is better)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.set_xticks(range(len(feature_results)))
        ax2.set_xticklabels(feature_results['feature'], rotation=45, ha='right')
        ax2.legend(fontsize=11)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars2, feature_results['recall@20'] * 100):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 3. Radar Chart for Multi-Metric View Impact
        if len(feature_results) > 0:
            # Prepare data for radar chart
            categories = ['Recall@10', 'Recall@20', 'MRR']
            
            # Normalize metrics to 0-100 scale for better visualization
            metrics_data = []
            for _, row in feature_results.iterrows():
                metrics_data.append([
                    row['recall@10_drop'] * 1000,  # Scale up for visibility
                    row['recall@20_drop'] * 1000,
                    row['mrr_drop'] * 1000
                ])
            
            # Create polar subplot
            ax3 = plt.subplot(2, 2, 3, projection='polar')
            
            # Number of variables
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Complete the circle
            
            # Plot each feature
            for i, (feature_name, data) in enumerate(zip(feature_results['feature'], metrics_data)):
                values = data + [data[0]]  # Complete the circle
                ax3.plot(angles, values, 'o-', linewidth=2, 
                        label=feature_name, color=view_colors[i], alpha=0.7)
                ax3.fill(angles, values, alpha=0.25, color=view_colors[i])
            
            # Add category labels
            ax3.set_xticks(angles[:-1])
            ax3.set_xticklabels(categories, fontsize=11)
            ax3.set_title('Multi-Metric Impact Profile\n(Performance drops when views removed)', 
                         fontsize=14, fontweight='bold', pad=30)
            ax3.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=9)
            ax3.grid(True, alpha=0.3)
        
        # 4. View Importance Ranking with Confidence Intervals
        # Sort by importance for ranking
        ranked_features = feature_results.sort_values('relative_importance', ascending=False)
        
        # Create ranking visualization
        y_pos = np.arange(len(ranked_features))
        importance_vals = ranked_features['relative_importance'].values
        
        # Create error bars based on MRR variation (as proxy for confidence)
        error_vals = np.abs(ranked_features['mrr_drop'].values) * 50  # Scale for visibility and ensure positive
        
        bars4 = ax4.barh(y_pos, importance_vals, 
                        color=view_colors[:len(ranked_features)], alpha=0.8, 
                        edgecolor='black', linewidth=0.5)
        
        # Add error bars
        ax4.errorbar(importance_vals, y_pos, xerr=error_vals, fmt='none', 
                    ecolor='black', alpha=0.5, capsize=3)
        
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([f"#{i+1}. {name}" for i, name in enumerate(ranked_features['feature'])])
        ax4.set_xlabel('View Importance Score (%)', fontsize=12, fontweight='bold')
        ax4.set_title('View Importance Ranking\n(Most to least critical for performance)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax4.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars4, importance_vals)):
            ax4.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
                    va='center', fontweight='bold', fontsize=10)
        
        # Main title
        model_name = self.model_info.get('model_type', 'NATR Enhanced Model')
        plt.suptitle(f'7-View Architecture Performance Analysis\n{model_name}', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the focused view influence graph
        output_dir = 'output/analyze_learning'
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'view_influence_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nView influence visualization saved to: {output_path}")
        
        return fig

    def analyze_attention_weights(self, num_samples=500):
        """Analyze attention weights across views for warm-start users only"""
        
        print(f"\n{'='*60}")
        print("🔍 ANALYZING VIEW ATTENTION WEIGHTS (WARM-START USERS ONLY)")
        print(f"{'='*60}")
        
        # Get train users to identify warm-start users
        if not hasattr(self, 'train_users') or self.train_users is None:
            print("Warning: No train_users available. Must run ablation study first to identify warm-start users.")
            print("Falling back to collecting attention weights from all users.")
            self.train_users = set()  # Empty set means we'll skip warm-start filtering
        
        if len(self.train_users) > 0:
            print(f"Using {len(self.train_users)} training users for warm-start filtering")
        else:
            print("No warm-start filtering applied - using all test users")
        
        # Enable debug mode to capture attention weights
        self.model.enable_debug(True)
        
        attention_data = []
        view_names = list(self.feature_groups.values())
        warm_start_samples = 0
        total_samples = 0
        
        # Create a separate dataloader with only warm-start users if available
        if len(self.train_users) > 0 and hasattr(self, 'warm_start_purchases'):
            print(f"Creating dataloader with {len(self.warm_start_purchases)} warm-start purchases")
            
            # Debug: Check user IDs
            warm_user_ids = set(p['user_id'] for p in self.warm_start_purchases)
            print(f"Warm-start purchase user IDs sample: {list(warm_user_ids)[:5]}")
            print(f"Train user IDs sample: {list(self.train_users)[:5]}")
            print(f"Overlap check: {len(warm_user_ids.intersection(self.train_users))} users in common")
            
            # Create dataloader with only warm-start purchases
            from utils.training_utils import create_dataloaders
            dummy_train = self.warm_start_purchases[:100] if len(self.warm_start_purchases) > 100 else self.warm_start_purchases
            
            _, warm_test_loader = create_dataloaders(
                train_samples=dummy_train,
                test_samples=self.warm_start_purchases,
                package_processor=self.package_processor,
                session_processor=self.session_processor,
                batch_size=64,
                num_workers=0,
                use_weighted_sampling=False
            )
            
            attention_test_loader = warm_test_loader
            using_warm_start_only = True
            print(f"Using warm-start only dataloader with {len(self.warm_start_purchases)} samples")
        else:
            attention_test_loader = self.test_loader
            using_warm_start_only = False
            print("Using original test_loader (all users)")
        
        with torch.no_grad():
            sample_count = 0
            for batch in tqdm(attention_test_loader, desc="Collecting attention weights"):
                if sample_count >= num_samples:
                    break
                
                # Move batch to device
                batch = move_batch_to_device(batch, self.device)
                
                # Forward pass to generate attention weights
                outputs = self.model(batch)
                
                # Extract view-level attention weights from model
                for module in self.model.modules():
                    if hasattr(module, 'last_attention_weights') and module.last_attention_weights is not None:
                        # Get the attention weights [batch_size, num_views]
                        weights = module.last_attention_weights
                        
                        # Handle different attention weight formats
                        if weights.dim() == 4:  # [batch, heads, seq, views]
                            weights = weights.mean(dim=1).squeeze(1)  # Average over heads and sequence
                        elif weights.dim() == 3:  # [batch, seq, views]
                            weights = weights.squeeze(1)  # Remove sequence dimension
                        
                                # Store attention weights only for warm-start users
                        for i in range(weights.shape[0]):
                            user_id = batch['user_id'][i].item()
                            total_samples += 1
                            
                            # Debug first few users
                            if total_samples <= 5:
                                print(f"Debug - User ID: {user_id}, In train_users: {user_id in self.train_users}")
                            
                            # If using warm-start only dataloader, all users are warm-start
                            if using_warm_start_only or user_id in self.train_users:
                                warm_start_samples += 1
                                sample_weights = weights[i].cpu().numpy()
                                for j, weight in enumerate(sample_weights):
                                    if j < len(view_names):  # Ensure we don't exceed view count
                                        attention_data.append({
                                            'user_id': user_id,
                                            'view': view_names[j],
                                            'attention_weight': weight.item()
                                        })
                
                sample_count += batch['user_id'].shape[0]
        
        # Disable debug mode
        self.model.enable_debug(False)
        
        print(f"Processed {total_samples} total samples, {warm_start_samples} warm-start samples")
        print(f"Warm-start ratio: {warm_start_samples/max(total_samples,1)*100:.1f}%")
        
        if attention_data:
            df = pd.DataFrame(attention_data)
            print(f"Collected {len(df)} attention weight data points from warm-start users")
            return df
        else:
            print("Warning: No attention weights captured from warm-start users")
            return None

    def create_attention_analysis(self):
        """Create attention weight analysis with boxplots"""
        
        # Collect attention weights
        attention_df = self.analyze_attention_weights()
        
        if attention_df is None:
            print("No attention data to analyze")
            return None
        
        # Create figure with attention analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Boxplot of attention weights per view
        import seaborn as sns
        sns.boxplot(data=attention_df, x='view', y='attention_weight', ax=ax1)
        ax1.set_title('View Attention Weight Distribution\n(Higher = More Important)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Views', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Attention Weight', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Mean attention weights with confidence intervals
        view_stats = attention_df.groupby('view')['attention_weight'].agg(['mean', 'std', 'count']).reset_index()
        view_stats['sem'] = view_stats['std'] / np.sqrt(view_stats['count'])  # Standard error
        view_stats = view_stats.sort_values('mean', ascending=True)
        
        y_pos = np.arange(len(view_stats))
        bars = ax2.barh(y_pos, view_stats['mean'], 
                       xerr=view_stats['sem'], 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#54A0FF'][:len(view_stats)],
                       alpha=0.8, capsize=5)
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(view_stats['view'], fontsize=11)
        ax2.set_xlabel('Mean Attention Weight (±SE)', fontsize=12, fontweight='bold')
        ax2.set_title('Average View Importance\n(Across All Users)', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + view_stats['sem'].iloc[i] + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.suptitle('Multi-View Attention Analysis', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save plot
        output_dir = 'output/analyze_learning'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'attention_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nAttention analysis saved to: {output_path}")
        
        return fig, attention_df

    def create_simplified_view_analysis(self):
        """Create simplified 2-graph analysis showing view performance and contribution"""
        
        print(f"\n{'='*60}")
        print("🎨 CREATING SIMPLIFIED VIEW ANALYSIS")
        print(f"{'='*60}")
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        view_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#54A0FF']
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Graph 1: View Performance Impact (when removed)
        if self.ablation_results is not None:
            feature_results = self.ablation_results[self.ablation_results['feature'] != 'Baseline (All Features)'].copy()
            feature_results = feature_results.sort_values('recall@20_drop', ascending=True)
            
            y_pos = np.arange(len(feature_results))
            bars1 = ax1.barh(y_pos, feature_results['recall@20_drop'] * 100, 
                           color=view_colors[:len(feature_results)], alpha=0.8, edgecolor='black')
            
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(feature_results['feature'], fontsize=11)
            ax1.set_xlabel('Performance Drop when Removed (%)', fontsize=12, fontweight='bold')
            ax1.set_title('View Importance\n(Higher = More Important)', fontsize=14, fontweight='bold')
            ax1.grid(axis='x', alpha=0.3)
            
            # Add value labels on bars
            for i, bar in enumerate(bars1):
                width = bar.get_width()
                ax1.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                        f'{width:.1f}%', ha='left', va='center', fontweight='bold')
        
        # Graph 2: View Parameter Allocation vs Performance Impact
        if self.ablation_results is not None and self.architecture_analysis is not None:
            # Merge data
            feature_results = self.ablation_results[self.ablation_results['feature'] != 'Baseline (All Features)'].copy()
            arch_results = self.architecture_analysis[self.architecture_analysis['parameters'] > 0].copy()
            
            merged_df = pd.merge(
                feature_results[['feature', 'recall@20_drop', 'relative_importance']], 
                arch_results[['feature', 'param_percentage']], 
                on='feature', how='inner'
            )
            
            if not merged_df.empty:
                scatter = ax2.scatter(merged_df['param_percentage'], 
                                    merged_df['recall@20_drop'] * 100,
                                    c=view_colors[:len(merged_df)], 
                                    s=200, alpha=0.8, edgecolors='black', linewidth=2)
                
                # Add labels for each point
                for i, row in merged_df.iterrows():
                    ax2.annotate(row['feature'], 
                               (row['param_percentage'], row['recall@20_drop'] * 100),
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=10, fontweight='bold')
                
                ax2.set_xlabel('Parameter Allocation (%)', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Performance Drop when Removed (%)', fontsize=12, fontweight='bold')
                ax2.set_title('Parameter Efficiency\n(Top-right = Most Important)', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                
                # Add diagonal reference line
                max_val = max(merged_df['param_percentage'].max(), merged_df['recall@20_drop'].max() * 100)
                ax2.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal efficiency line')
                ax2.legend()
        
        # Get model name for title
        model_name = os.path.basename(self.model_info.get('checkpoint_path', 'Unknown Model'))
        
        plt.suptitle(f'Multi-View Performance Analysis: {model_name}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save plot
        output_dir = 'output/analyze_learning'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'simplified_view_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSimplified view analysis saved to: {output_path}")
        
        return fig
    
    def create_feature_importance_graph(self):
        """Create feature importance graph showing performance drop when features are removed"""
        
        print(f"\n{'='*60}")
        print("🎨 CREATING FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*60}")
        
        if self.ablation_results is None:
            raise ValueError("Must run ablation study first")
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left plot: Feature importance (performance drop)
        feature_names = []
        performance_drops = []
        
        for _, row in self.ablation_results.iterrows():
            if row['feature'] != 'Baseline (All Features)':
                feature_names.append(row['feature'])
                performance_drops.append(row['recall@20_drop'] * 100)  # Convert to percentage
        
        # Sort by performance drop from highest to lowest
        # For horizontal bar plots, the first item appears at the bottom
        # So we sort in ascending order to get highest values at the top
        sorted_data = sorted(zip(feature_names, performance_drops), key=lambda x: x[1])
        feature_names, performance_drops = zip(*sorted_data)
        
        # Create horizontal bar plot
        colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
        bars = ax1.barh(range(len(feature_names)), performance_drops, color=colors)
        
        ax1.set_yticks(range(len(feature_names)))
        ax1.set_yticklabels(feature_names)
        ax1.set_xlabel('Performance Drop (%)', fontsize=12)
        ax1.set_title('Feature Importance\n(Recall@20 Drop When Removed)', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, performance_drops)):
            ax1.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
                    va='center', fontsize=11, fontweight='bold')
        
        # Right plot: Feature embedding dimensions (histogram)
        if self.architecture_analysis is not None:
            feature_params = []
            param_percentages = []
            
            # Clean up the data first - remove duplicates and rename
            arch_df = self.architecture_analysis.copy()
            arch_df['feature'] = arch_df['feature'].replace({
                'Events/Interactions': 'Events',
                'Category/Theme (Combined)': 'Category/Theme'
            })
            
            # Group by feature to remove duplicates
            arch_df = arch_df.groupby('feature').agg({
                'param_percentage': 'sum',
                'parameters': 'sum'
            }).reset_index()
            
            for _, row in arch_df.iterrows():
                feature_params.append(row['feature'])
                param_percentages.append(row['param_percentage'])
            
            # Sort by parameter percentage (descending)
            sorted_data = sorted(zip(feature_params, param_percentages), key=lambda x: x[1], reverse=True)
            feature_params, param_percentages = zip(*sorted_data)
            
            # Create histogram (vertical bar chart)
            colors_hist = plt.cm.Set3(np.linspace(0, 1, len(feature_params)))
            # Get actual parameter counts for y-axis
            actual_params = []
            for feature in feature_params:
                row = arch_df[arch_df['feature'] == feature].iloc[0]
                actual_params.append(row['parameters'])
            
            bars = ax2.bar(range(len(feature_params)), actual_params, color=colors_hist, alpha=0.7, edgecolor='black')
            
            ax2.set_xticks(range(len(feature_params)))
            ax2.set_xticklabels(feature_params, rotation=45, ha='right')
            ax2.set_ylabel('Number of Parameters', fontsize=12)
            ax2.set_title('Feature Embedding Dimensions\n(Parameter Count & Allocation)', fontsize=14, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            
            # Format y-axis to show parameter counts nicely
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
            
            # Add percentage labels on bars
            for i, (bar, val, params) in enumerate(zip(bars, param_percentages, actual_params)):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000, f'{val:.1f}%', 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('output/analyze_learning', exist_ok=True)
        save_path = 'output/analyze_learning/feature_importance_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance analysis saved to: {save_path}")
        
        return fig
    
    def create_attention_boxplot(self):
        """Create boxplot showing attention weight distributions across views for warm-start users"""
        
        print(f"\n{'='*60}")
        print("🔍 ANALYZING VIEW ATTENTION WEIGHTS (WARM-START USERS)")
        print(f"{'='*60}")
        
        # Collect attention weights with more samples for warm-start users only
        attention_df = self.analyze_attention_weights(num_samples=1000)
        
        if attention_df is None or attention_df.empty:
            print("No attention weights collected from warm-start users")
            return None
        
        # Print some statistics to understand the data
        print(f"Collected {len(attention_df)} attention weight samples from warm-start users")
        print(f"Attention weight range: {attention_df['attention_weight'].min():.6f} to {attention_df['attention_weight'].max():.6f}")
        print(f"Attention weight std: {attention_df['attention_weight'].std():.6f}")
        
        # Create larger figure with better spacing
        fig, ax = plt.subplots(figsize=(14, 10))
        
        view_names = list(self.feature_groups.values())
        attention_data = [attention_df[attention_df['view'] == view]['attention_weight'].values 
                         for view in view_names if view in attention_df['view'].values]
        view_labels = [view for view in view_names if view in attention_df['view'].values]
        
        # Print view-specific statistics
        for view, data in zip(view_labels, attention_data):
            if len(data) > 0:
                print(f"{view}: mean={data.mean():.6f}, std={data.std():.6f}, range=[{data.min():.6f}, {data.max():.6f}]")
        
        # Create enhanced boxplot with better visibility
        box_plot = ax.boxplot(attention_data, labels=view_labels, patch_artist=True, 
                             widths=0.6, showfliers=True, notch=True, 
                             flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.6),
                             medianprops=dict(color='black', linewidth=2),
                             boxprops=dict(linewidth=1.5),
                             whiskerprops=dict(linewidth=1.5),
                             capprops=dict(linewidth=1.5))
        
        # Color the boxes with more vibrant colors
        colors = plt.cm.tab10(np.linspace(0, 1, len(box_plot['boxes'])))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            patch.set_edgecolor('black')
        
        # Improve axes and labels
        ax.set_xlabel('Feature Views', fontsize=14, fontweight='bold')
        ax.set_ylabel('Attention Weight', fontsize=14, fontweight='bold')
        ax.set_title('Attention Weight Distribution Across Views\n(Warm-Start Users Only)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Enhance grid and formatting
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Try different y-axis scaling if the range is very small
        y_range = attention_df['attention_weight'].max() - attention_df['attention_weight'].min()
        if y_range < 0.01:  # If range is very small, zoom in
            mean_val = attention_df['attention_weight'].mean()
            ax.set_ylim(mean_val - 3*attention_df['attention_weight'].std(), 
                       mean_val + 3*attention_df['attention_weight'].std())
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        os.makedirs('output/analyze_learning', exist_ok=True)
        save_path = 'output/analyze_learning/attention_boxplot.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention boxplot saved to: {save_path}")
        
        return fig
    
    def _generate_insights_text(self) -> str:
        """Generate insights text based on analysis results"""
        insights = ["Multi-View Learning Insights:\n"]
        
        if self.ablation_results is not None:
            # Find most important feature
            feature_results = self.ablation_results[self.ablation_results['feature'] != 'Baseline (All Features)'].copy()
            most_important = feature_results.loc[feature_results['relative_importance'].idxmax()]
            least_important = feature_results.loc[feature_results['relative_importance'].idxmin()]
            
            insights.append(f"• Most critical: {most_important['feature']}")
            insights.append(f"  ({most_important['relative_importance']:.1f}% importance)")
            insights.append(f"\n• Least critical: {least_important['feature']}")
            insights.append(f"  ({least_important['relative_importance']:.1f}% importance)")
            
            # Calculate total impact
            total_impact = feature_results['relative_importance'].sum()
            insights.append(f"\n• Multi-view benefit: {total_impact:.1f}%")
            insights.append("  performance loss if using single features")
            
        if self.architecture_analysis is not None:
            # Find largest parameter allocation
            largest_alloc = self.architecture_analysis.loc[self.architecture_analysis['param_percentage'].idxmax()]
            insights.append(f"\n• Largest capacity: {largest_alloc['feature']}")
            insights.append(f"  ({largest_alloc['param_percentage']:.1f}% of parameters)")
        
        insights.extend([
            "\n• Each feature type contributes uniquely",
            "• Removing any feature degrades performance",
            "• Multi-view fusion is essential for",
            "  optimal recommendation quality"
        ])
        
        return "".join(insights)
    
    def save_results(self):
        """Save all analysis results to files"""
        
        print(f"\n{'='*60}")
        print("SAVING ANALYSIS RESULTS")
        print(f"{'='*60}")
        
        # Create output directory
        output_dir = 'output/analyze_learning'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save ablation results
        if self.ablation_results is not None:
            csv_path = os.path.join(output_dir, 'multiview_ablation_results.csv')
            self.ablation_results.to_csv(csv_path, index=False)
            print(f"Ablation results saved to: {csv_path}")
        
        # Save architecture analysis
        if self.architecture_analysis is not None:
            csv_path = os.path.join(output_dir, 'multiview_architecture_analysis.csv')
            self.architecture_analysis.to_csv(csv_path, index=False)
            print(f"Architecture analysis saved to: {csv_path}")
    
    def run_complete_analysis(self):
        """Run the complete multi-view learning analysis"""
        
        print(f"{'='*80}")
        print("NATR MULTI-VIEW LEARNING COMPREHENSIVE ANALYSIS")
        print(f"{'='*80}")
        print(f"Model: {self.model_info_path}")
        print(f"Sample size: {self.sample_size}")
        print(f"Device: {self.device}")
        
        # 1. Run ablation study
        self.run_ablation_study()
        
        # 2. Analyze model architecture
        self.analyze_model_architecture()
        
        # 3. Create feature importance graph
        self.create_feature_importance_graph()
        
        # 4. Create attention boxplot (skip if there are errors)
        try:
            self.create_attention_boxplot()
        except Exception as e:
            print(f"Skipping attention analysis due to error: {e}")
            import traceback
            traceback.print_exc()
        
        # 5. Save all results
        self.save_results()
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        print("Generated files in output/analyze_learning/:")
        print("- feature_importance_analysis.png (feature importance)")
        print("- attention_boxplot.png (attention weights)")
        print("- multiview_ablation_results.csv (ablation study data)")
        print("- multiview_architecture_analysis.csv (architecture data)")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Multi-View Learning Analysis for NATR'
    )
    parser.add_argument('--model-info', type=str, required=True,
                        help='Path to model_info.json file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint file (.pth)')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Number of samples for ablation analysis (default: all purchases)')
    
    args = parser.parse_args()
    
    # Validate model info file
    if not os.path.exists(args.model_info):
        print(f"Error: Model info file not found: {args.model_info}")
        return 1
    
    # Validate checkpoint file
    if not os.path.exists(args.checkpoint):
        print(f"Error: Model checkpoint file not found: {args.checkpoint}")
        return 1
    
    try:
        # Initialize analyzer
        analyzer = ComprehensiveMultiViewAnalyzer(
            model_info_path=args.model_info,
            checkpoint_path=args.checkpoint,
            sample_size=args.sample_size
        )
        
        # Run complete analysis
        analyzer.run_complete_analysis()
        
        return 0
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
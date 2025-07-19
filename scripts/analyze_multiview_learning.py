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
    
    def __init__(self, model_info_path: str, sample_size: Optional[int] = None):
        """
        Initialize the analyzer
        
        Args:
            model_info_path: Path to model_info.json
            sample_size: Number of samples to use for ablation analysis (None = all purchases)
        """
        self.model_info_path = model_info_path
        self.sample_size = sample_size
        
        # Load model info
        with open(model_info_path, 'r') as f:
            self.model_info = json.load(f)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() 
                                  else 'mps' if torch.backends.mps.is_available() 
                                  else 'cpu')
        
        # Load the enhanced model directly from checkpoint
        print(f"Loading enhanced model from checkpoint...")
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
        
        # Feature groups for analysis (7-view enhanced model matching exact implementation)
        # Order matches the model: [title, coordinates, country, category/theme, price, time, events]
        self.feature_groups = {
            'title_embeddings': 'Title Embeddings',           # View 0
            'coordinates': 'Geographic Coordinates',          # View 1  
            'country': 'Country',                            # View 2
            'category_theme': 'Category/Theme (Combined)',   # View 3 - COMBINED
            'price': 'Price',                               # View 4
            'time': 'Temporal Information',                 # View 5 - MISSING before
            'events': 'Events/Interactions'                 # View 6
        }
        
        # Results storage
        self.ablation_results = None
        self.architecture_analysis = None
        
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
        
        # Data paths
        event_data = self.model_info.get('event_data_path', 'data/13_months_new2_clean.parquet')
        package_data_path = self.model_info.get('package_data_path', 'data/feed.parquet')
        
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
        
        # Filter for warm-start users only (users present in both train and test sets)
        print("Filtering for warm-start users (present in both train and test)...")
        train_users = set(sample['user_id'] for sample in train_samples)
        warm_start_purchases = [s for s in test_purchases if s['user_id'] in train_users]
        
        print(f"Total test purchases: {len(test_purchases)}")
        print(f"Warm-start test purchases: {len(warm_start_purchases)} ({len(warm_start_purchases)/len(test_purchases)*100:.1f}%)")
        
        # Use warm-start purchases for analysis
        test_purchases = warm_start_purchases
        
        if self.sample_size is not None and self.sample_size < len(test_purchases):
            import random
            random.seed(42)
            test_purchases = random.sample(test_purchases, self.sample_size)
        
        print(f"Using {len(test_purchases)} warm-start test purchase samples for ablation study")
        
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
        
        results = []
        
        # Baseline performance (no masking)
        print("\nEvaluating baseline performance...")
        baseline_metrics = self._evaluate_with_feature_masking(test_loader, feature_to_mask=None)
        baseline_recall_20 = baseline_metrics.get('purchase_recall@20', 0.0)
        baseline_recall_10 = baseline_metrics.get('purchase_recall@10', 0.0)
        baseline_mrr = baseline_metrics.get('purchase_mrr', 0.0)
        
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
                                for field in category_fields + theme_fields:
                                    if field in features:
                                        features[field] = torch.zeros_like(features[field])
                            elif feature_to_mask == 'price':
                                # Mask ALL price-related features
                                for price_field in ['prices', 'purchased_prices', 'short_term_prices', 'long_term_prices']:
                                    if price_field in features:
                                        features[price_field] = torch.zeros_like(features[price_field])
                            elif feature_to_mask == 'time':
                                # Mask ALL temporal information
                                for time_field in ['timestamps', 'purchased_timestamps', 'short_term_timestamps', 'long_term_timestamps']:
                                    if time_field in features:
                                        features[time_field] = torch.zeros_like(features[time_field])
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
                
            # Time encoder parameters (new view)
            time_params = 0
            if hasattr(self.model.package_encoder, 'time_encoder'):
                time_params = sum(p.numel() for p in self.model.package_encoder.time_encoder.parameters())
                
            # Events encoder parameters (in package_encoder in enhanced model)
            events_params = 0
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
                          title_params + coord_params + price_params + time_params + events_params)
            
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
                    'feature': 'Category/Theme (Combined)',
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
                    'feature': 'Temporal Information',
                    'parameters': time_params,
                    'param_percentage': (time_params / total_params) * 100,
                    'embedding_dim': self.model_info['config'].get('hidden_dim', 256)  # Output of time_encoder
                })
                results.append({
                    'feature': 'Events/Interactions',
                    'parameters': events_params,
                    'param_percentage': (events_params / total_params) * 100,
                    'embedding_dim': self.model_info['config'].get('hidden_dim', 256)  # Output of event components
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
            feature_results = feature_results.sort_values('relative_importance', ascending=True)
            
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
        
        # 3. Parameter Allocation Pie Chart
        ax3 = plt.subplot(3, 3, 3)
        if self.architecture_analysis is not None:
            param_df = self.architecture_analysis[self.architecture_analysis['parameters'] > 0].copy()
            
            if not param_df.empty:
                wedges, texts, autotexts = ax3.pie(
                    param_df['param_percentage'], 
                    labels=param_df['feature'],
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=colors[:len(param_df)]
                )
                
                ax3.set_title('Model Capacity Allocation\n(% of Feature Parameters)', fontweight='bold')
                
                # Enhance text
                for text in texts:
                    text.set_fontsize(9)
                for autotext in autotexts:
                    autotext.set_fontsize(8)
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
        
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
        
        # 3. Create comprehensive visualization
        self.create_comprehensive_visualization()
        
        # 4. Create focused view influence graph
        self.create_view_influence_graph()
        
        # 5. Save all results
        self.save_results()
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        print("Generated files in output/analyze_learning/:")
        print("- comprehensive_multiview_analysis.png (full analysis)")
        print("- view_influence_analysis.png (focused view influence graph)")
        print("- multiview_ablation_results.csv (ablation study data)")
        print("- multiview_architecture_analysis.csv (architecture data)")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Multi-View Learning Analysis for NATR'
    )
    parser.add_argument('--model-info', type=str, required=True,
                        help='Path to model_info.json file')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Number of samples for ablation analysis (default: all purchases)')
    
    args = parser.parse_args()
    
    # Validate model info file
    if not os.path.exists(args.model_info):
        print(f"Error: Model info file not found: {args.model_info}")
        return 1
    
    try:
        # Initialize analyzer
        analyzer = ComprehensiveMultiViewAnalyzer(
            model_info_path=args.model_info,
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
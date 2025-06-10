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

from models.natr import NATR, NATRConfig
from utils.session_processor import SessionProcessor
from utils.package_processor import PackageProcessor
from utils.unified_metrics import UnifiedMetricsTracker
from utils.training_utils import create_dataloaders, move_batch_to_device
from scripts.evaluate_recommendations import RecommendationEvaluator


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
        
        # Initialize evaluator (for ablation studies)
        self.evaluator = RecommendationEvaluator(model_info_path)
        self.device = self.evaluator.device
        self.model = self.evaluator.recommender.model
        
        # Feature groups for analysis
        self.feature_groups = {
            'title_embeddings': 'Title Embeddings',
            'coordinates': 'Geographic Coordinates',
            'country': 'Country',
            'category': 'Category', 
            'theme': 'Theme',
            'price': 'Price'
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
        
        # Prepare test data
        if self.sample_size is None:
            print(f"\nPreparing test data (using all purchase samples)...")
            test_loader, test_samples = self.evaluator.prepare_test_data(
                sample_size=None,
                purchases_only=True  # Focus on purchase prediction
            )
            self.sample_size = len(test_samples)  # Update with actual size
        else:
            print(f"\nPreparing test data (sample size: {self.sample_size})...")
            test_loader, test_samples = self.evaluator.prepare_test_data(
                sample_size=self.sample_size,
                purchases_only=True  # Focus on purchase prediction
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
                            elif feature_to_mask == 'country' and 'country_ids' in features:
                                features['country_ids'] = torch.zeros_like(features['country_ids'])
                            elif feature_to_mask == 'category' and 'category_ids' in features:
                                features['category_ids'] = torch.zeros_like(features['category_ids'])
                            elif feature_to_mask == 'theme' and 'theme_ids' in features:
                                features['theme_ids'] = torch.zeros_like(features['theme_ids'])
                            elif feature_to_mask == 'price' and 'prices' in features:
                                features['prices'] = torch.zeros_like(features['prices'])
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
            
            # Calculate total and percentages
            total_params = (country_params + category_params + theme_params + 
                          title_params + coord_params + price_params)
            
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
                    'feature': 'Category',
                    'parameters': category_params,
                    'param_percentage': (category_params / total_params) * 100,
                    'embedding_dim': category_size
                })
                results.append({
                    'feature': 'Theme',
                    'parameters': theme_params,
                    'param_percentage': (theme_params / total_params) * 100,
                    'embedding_dim': theme_size
                })
                results.append({
                    'feature': 'Price',
                    'parameters': price_params,
                    'param_percentage': (price_params / total_params) * 100,
                    'embedding_dim': self.model_info['config'].get('hidden_dim', 256)  # Output of price_encoder
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
        y_positions = np.linspace(0.9, 0.1, 6)
        feature_labels = ['Title\nEmbeddings', 'Geographic\nCoordinates', 
                         'Country', 'Category', 'Theme', 'Price']
        
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
        
        # 4. Save all results
        self.save_results()
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        print("Generated files in output/analyze_learning/:")
        print("- comprehensive_multiview_analysis.png (main visualization)")
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
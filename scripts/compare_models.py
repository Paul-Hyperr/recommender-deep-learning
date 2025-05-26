#!/usr/bin/env python3
"""
Model Comparison Script using Unified Metrics

This script loads all trained NATR models and evaluates them using the 
comprehensive UnifiedMetricsTracker to provide detailed performance comparisons.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary components  
from models.natr import NATR, NATRConfig
from utils.session_processor import SessionProcessor
from utils.package_processor import PackageProcessor
from utils.unified_metrics import UnifiedMetricsTracker
from utils.training_utils import move_batch_to_device, create_dataloaders


class ModelComparator:
    """
    Compare multiple trained NATR models using unified metrics
    """
    
    def __init__(self, device: torch.device, k_values: List[int] = [10, 20]):
        """
        Initialize the model comparator
        
        Args:
            device: Device to run evaluation on
            k_values: List of k values for recall@k metrics
        """
        self.device = device
        self.k_values = k_values
        self.models = {}
        self.model_configs = {}
        self.results = {}
        
        # Initialize unified metrics tracker
        self.metrics_tracker = UnifiedMetricsTracker(
            k_values=k_values,
            event_weights={
                'purchase': 1.0,      # Full weight for purchases
                'checkout': 0.6,      # Consistent weight for checkout events
                'add_to_cart': 0.2,   # Consistent weight for add-to-cart
                'view': 0.05          # Very minimal weight for view events
            }
        )
    
    def load_model(self, model_name: str, checkpoint_path: str, config_path: Optional[str] = None):
        """
        Load a trained model from checkpoint
        
        Args:
            model_name: Name identifier for the model
            checkpoint_path: Path to the model checkpoint
            config_path: Optional path to model config (if separate from checkpoint)
        """
        print(f"Loading model: {model_name}")
        
        # Load checkpoint
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Get config from checkpoint or separate file
            if 'config' in checkpoint:
                config_dict = checkpoint['config']
            elif config_path and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    data = json.load(f)
                    # Extract config from model_info JSON structure
                    config_dict = data.get('config', data)
            else:
                print(f"Warning: Could not find config for {model_name}, using default config")
                # Create a default config for models without embedded config
                config_dict = {
                    'num_users': 751729,
                    'num_packages': 8773, 
                    'num_countries': 78,
                    'num_categories': 9,
                    'num_themes': 99,
                    'title_embedding_dim': 1536,
                    'hidden_dim': 128,
                    'embedding_dim': 128,
                    'user_embedding_dim': 128,
                    'dropout': 0.1
                }
            
            # Create model config
            config = NATRConfig(**config_dict)
            
            # Create and load model
            model = NATR(config)
            
            # Handle backward compatibility for BatchNorm -> LayerNorm change
            state_dict = checkpoint['model_state_dict']
            
            # Remove BatchNorm keys that don't exist in current LayerNorm model
            keys_to_remove = [k for k in state_dict.keys() if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k]
            for key in keys_to_remove:
                print(f"Removing incompatible key: {key}")
                del state_dict[key]
            
            # Try to load with dimension adjustments if needed
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    print(f"Attempting to load with dimension adjustments...")
                    # Load with strict=False and warn about mismatches
                    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                    if missing_keys:
                        print(f"Missing keys: {missing_keys}")
                    if unexpected_keys:
                        print(f"Unexpected keys: {unexpected_keys}")
                    print(f"Warning: Model loaded with dimension mismatches. This may affect performance.")
                else:
                    raise e
            
            model.to(self.device)
            model.eval()
            
            # Store model and config
            self.models[model_name] = model
            self.model_configs[model_name] = {
                'config': config_dict,
                'checkpoint_info': {
                    'epoch': checkpoint.get('epoch', 'Unknown'),
                    'timestamp': checkpoint.get('timestamp', 'Unknown'),
                    'best_metric': checkpoint.get('best_purchase_recall', checkpoint.get('best_intent_recall', 'Unknown'))
                }
            }
            
            print(f"✓ Successfully loaded {model_name}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {str(e)}")
            return False
    
    def auto_discover_models(self, checkpoints_dir: str = "checkpoints/natr"):
        """
        Automatically discover and load trained models from checkpoints directory
        
        Args:
            checkpoints_dir: Directory containing model checkpoints
        """
        print(f"Auto-discovering models in {checkpoints_dir}...")
        
        if not os.path.exists(checkpoints_dir):
            print(f"Checkpoints directory not found: {checkpoints_dir}")
            return
        
        # Model checkpoint mappings with optional config files
        model_mappings = {
            'Base NATR': ('best_model.pth', 'model_info.json'),
            'Contrastive NATR': ('contrastive_best_model.pth', 'model_info_contrastive.json'),
            'Pretrain-Finetune NATR': ('finetuned_model.pth', 'model_info_pretrain_finetune.json'),
            'Checkout-Enhanced NATR': ('checkout_best_model.pth', 'model_info_checkout_enhanced.json')
        }
        
        # Try to load each model
        for model_name, (checkpoint_file, config_file) in model_mappings.items():
            checkpoint_path = os.path.join(checkpoints_dir, checkpoint_file)
            # Look for config file in the project root
            config_path = config_file if os.path.exists(config_file) else None
            self.load_model(model_name, checkpoint_path, config_path)
        
        print(f"Loaded {len(self.models)} models successfully")
    
    def evaluate_model(self, model_name: str, test_loader, max_batches: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate a single model using unified metrics with cold/warm start analysis
        
        Args:
            model_name: Name of the model to evaluate
            test_loader: DataLoader for test data
            max_batches: Maximum number of batches to evaluate (None for all)
            
        Returns:
            Dictionary of evaluation metrics including cold/warm start breakdown
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return {}
        
        model = self.models[model_name]
        model.eval()
        
        # Initialize separate metrics trackers for cold/warm start
        overall_tracker = UnifiedMetricsTracker(
            k_values=self.k_values,
            event_weights=self.metrics_tracker.event_weights
        )
        cold_start_tracker = UnifiedMetricsTracker(
            k_values=self.k_values,
            event_weights=self.metrics_tracker.event_weights
        )
        warm_start_tracker = UnifiedMetricsTracker(
            k_values=self.k_values,
            event_weights=self.metrics_tracker.event_weights
        )
        
        if max_batches:
            print(f"Evaluating {model_name} with cold/warm start analysis (limited to {max_batches} batches)...")
        else:
            print(f"Evaluating {model_name} with cold/warm start analysis...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # Break if we've reached the maximum number of batches
                if max_batches and batch_idx >= max_batches:
                    break
                # Move batch to device
                batch = move_batch_to_device(batch, self.device)
                
                # Forward pass
                try:
                    outputs = model(batch)
                    predictions = outputs['predictions']
                    targets = batch['purchased']['package_ids']
                    
                    # Get pre-computed event indicators from batch (these are now consistent across all models)
                    is_purchase = batch.get('is_purchase', torch.zeros_like(targets, dtype=torch.bool))
                    has_checkout = batch.get('has_checkout', torch.zeros_like(targets, dtype=torch.bool))
                    has_add_to_cart = batch.get('has_add_to_cart', torch.zeros_like(targets, dtype=torch.bool))
                    
                    # Get inclusive flags for individual event recalls (count all sessions with the event)
                    has_checkout_inclusive = batch.get('has_checkout_inclusive', torch.zeros_like(targets, dtype=torch.bool))
                    has_add_to_cart_inclusive = batch.get('has_add_to_cart_inclusive', torch.zeros_like(targets, dtype=torch.bool))
                    
                    # Get pre-computed cold/warm start classification
                    is_cold_start = batch.get('is_cold_start', torch.zeros_like(targets, dtype=torch.bool))
                    cold_start_mask = is_cold_start
                    warm_start_mask = ~is_cold_start
                    
                    # Debug print for first batch to verify consistent counts (only if needed)
                    if batch_idx == 0 and max_batches and max_batches <= 20:  # Only debug in small test runs
                        print(f"DEBUG - Batch keys: {list(batch.keys())}")
                        print(f"DEBUG - Purchase samples: {is_purchase.sum().item()}/{len(is_purchase)}")
                        print(f"DEBUG - Checkout samples: {has_checkout.sum().item()}/{len(has_checkout)}")
                        print(f"DEBUG - AddToCart samples: {has_add_to_cart.sum().item()}/{len(has_add_to_cart)}")
                        print(f"DEBUG - Cold start samples: {cold_start_mask.sum().item()}/{len(cold_start_mask)}")
                        print(f"DEBUG - Warm start samples: {warm_start_mask.sum().item()}/{len(warm_start_mask)}")
                        
                        # Debug: Check if targets are in top-k for purchase samples
                        if is_purchase.any():
                            purchase_indices = is_purchase.nonzero(as_tuple=True)[0]
                            _, top_10 = torch.topk(predictions, 10, dim=1)
                            _, top_20 = torch.topk(predictions, 20, dim=1)
                            
                            for i in purchase_indices[:3]:  # Check first 3 purchase samples
                                target = targets[i].item()
                                top10_items = top_10[i].tolist()
                                top20_items = top_20[i].tolist()
                                in_top10 = target in top10_items
                                in_top20 = target in top20_items
                                print(f"DEBUG - Purchase sample {i}: target={target}, in_top10={in_top10}, in_top20={in_top20}")
                                if in_top20 and not in_top10:
                                    rank = top20_items.index(target) + 1
                                    print(f"DEBUG - Target ranked at position {rank} (between 11-20)")
                        
                        # Debug: Check cold start session lengths
                        cold_indices = cold_start_mask.nonzero(as_tuple=True)[0]
                        if len(cold_indices) > 0:
                            print(f"DEBUG - Cold start sample session lengths:")
                            for i in cold_indices[:3]:
                                short_term_pkgs = batch['short_term']['package_ids'][i]
                                session_length = (short_term_pkgs != 0).sum().item()
                                print(f"DEBUG - Cold start sample {i}: session_length={session_length}")
                        
                        print(f"DEBUG - Available keys: has_checkout={batch.get('has_checkout', 'MISSING')}")
                        print(f"DEBUG - Available keys: has_add_to_cart={batch.get('has_add_to_cart', 'MISSING')}")
                        print(f"DEBUG - Available keys: is_cold_start={batch.get('is_cold_start', 'MISSING')}")
                    
                    # Update overall metrics
                    overall_tracker.update(
                        predictions=predictions,
                        targets=targets,
                        is_purchase=is_purchase,
                        has_checkout=has_checkout,
                        has_add_to_cart=has_add_to_cart,
                        has_checkout_inclusive=has_checkout_inclusive,
                        has_add_to_cart_inclusive=has_add_to_cart_inclusive
                    )
                    
                    # Update cold start metrics
                    if cold_start_mask.any():
                        cold_indices = cold_start_mask.nonzero(as_tuple=True)[0]
                        if len(cold_indices) > 0:
                            cold_start_tracker.update(
                                predictions=predictions[cold_indices],
                                targets=targets[cold_indices],
                                is_purchase=is_purchase[cold_indices],
                                has_checkout=has_checkout[cold_indices],
                                has_add_to_cart=has_add_to_cart[cold_indices],
                                has_checkout_inclusive=has_checkout_inclusive[cold_indices],
                                has_add_to_cart_inclusive=has_add_to_cart_inclusive[cold_indices]
                            )
                    
                    # Update warm start metrics
                    if warm_start_mask.any():
                        warm_indices = warm_start_mask.nonzero(as_tuple=True)[0]
                        if len(warm_indices) > 0:
                            warm_start_tracker.update(
                                predictions=predictions[warm_indices],
                                targets=targets[warm_indices],
                                is_purchase=is_purchase[warm_indices],
                                has_checkout=has_checkout[warm_indices],
                                has_add_to_cart=has_add_to_cart[warm_indices],
                                has_checkout_inclusive=has_checkout_inclusive[warm_indices],
                                has_add_to_cart_inclusive=has_add_to_cart_inclusive[warm_indices]
                            )
                    
                except Exception as e:
                    print(f"Error evaluating batch {batch_idx} for {model_name}: {str(e)}")
                    continue
                
                # Periodic cleanup for MPS
                if self.device.type == 'mps' and (batch_idx + 1) % 10 == 0:
                    torch.mps.empty_cache()
        
        # Compute final metrics
        overall_metrics = overall_tracker.compute()
        cold_start_metrics = cold_start_tracker.compute()
        warm_start_metrics = warm_start_tracker.compute()
        
        # Combine metrics with prefixes
        combined_metrics = {}
        for key, value in overall_metrics.items():
            combined_metrics[key] = value
        
        for key, value in cold_start_metrics.items():
            combined_metrics[f'cold_start_{key}'] = value
            
        for key, value in warm_start_metrics.items():
            combined_metrics[f'warm_start_{key}'] = value
        
        # Add cold/warm start sample counts
        combined_metrics['cold_start_samples'] = cold_start_tracker.counts['total']
        combined_metrics['warm_start_samples'] = warm_start_tracker.counts['total']
        
        # Add model-specific information
        combined_metrics['model_name'] = model_name
        combined_metrics['model_config'] = self.model_configs[model_name]
        
        return combined_metrics
    
    def evaluate_all_models(self, test_loader, max_batches: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all loaded models
        
        Args:
            test_loader: DataLoader for test data
            max_batches: Maximum number of batches to evaluate per model (None for all)
            
        Returns:
            Dictionary mapping model names to their evaluation results
        """
        print(f"\nEvaluating {len(self.models)} models...")
        
        results = {}
        
        for model_name in self.models.keys():
            print(f"\n--- Evaluating {model_name} ---")
            model_results = self.evaluate_model(model_name, test_loader, max_batches)
            results[model_name] = model_results
            
            # Print quick summary - focused on user's key metrics including cold/warm start
            if model_results:
                # Print both @10 and @20 metrics
                print(f"Overall Recall@10: {model_results.get('recall@10', 0)*100:.2f}% | Recall@20: {model_results.get('recall@20', 0)*100:.2f}%")
                print(f"Purchase Recall@10: {model_results.get('purchase_recall@10', 0)*100:.2f}% | Recall@20: {model_results.get('purchase_recall@20', 0)*100:.2f}%")
                print(f"InitiateCheckout Recall@10: {model_results.get('checkout_recall@10', 0)*100:.2f}% | Recall@20: {model_results.get('checkout_recall@20', 0)*100:.2f}%")
                print(f"AddToCart Recall@10: {model_results.get('add_to_cart_recall@10', 0)*100:.2f}% | Recall@20: {model_results.get('add_to_cart_recall@20', 0)*100:.2f}%")
                print(f"Weighted Recall@10: {model_results.get('weighted_recall@10', 0)*100:.2f}% | Recall@20: {model_results.get('weighted_recall@20', 0)*100:.2f}%")
                
                # Debug: Show actual counts to understand why some metrics are 0
                print(f"Sample counts - Total: {model_results.get('total', 0):,}, Purchase: {model_results.get('purchase', 0):,}, Checkout: {model_results.get('checkout', 0):,}, AddToCart: {model_results.get('add_to_cart', 0):,}")
                print(f"Cold Start Samples: {model_results.get('cold_start_samples', 0):,} | Warm Start Samples: {model_results.get('warm_start_samples', 0):,}")
                print(f"Cold Start Weighted Recall@10: {model_results.get('cold_start_weighted_recall@10', 0)*100:.2f}%")
                print(f"Warm Start Weighted Recall@10: {model_results.get('warm_start_weighted_recall@10', 0)*100:.2f}%")
        
        self.results = results
        return results
    
    def generate_metrics_table(self) -> str:
        """
        Generate a clean, formatted table with key evaluation metrics
        
        Returns:
            String containing the formatted metrics table
        """
        if not self.results:
            return "No evaluation results available."
        
        # Define the metrics we want in the table
        table_metrics = [
            ('recall@10', 'Overall R@10'),
            ('recall@20', 'Overall R@20'),
            ('purchase_recall@10', 'Purchase R@10'),
            ('purchase_recall@20', 'Purchase R@20'),
            ('checkout_recall@10', 'Checkout R@10'),
            ('checkout_recall@20', 'Checkout R@20'),
            ('add_to_cart_recall@10', 'AddCart R@10'),
            ('add_to_cart_recall@20', 'AddCart R@20'),
            ('weighted_recall@10', 'Weighted R@10'),
            ('weighted_recall@20', 'Weighted R@20'),
            ('cold_start_weighted_recall@10', 'Cold R@10'),
            ('warm_start_weighted_recall@10', 'Warm R@10'),
        ]
        
        model_names = list(self.results.keys())
        
        # Calculate column widths
        model_name_width = max(len(name) for name in model_names) + 2
        metric_width = 12
        
        # Create table header
        table_lines = []
        table_lines.append("=" * 120)
        table_lines.append("MODEL EVALUATION METRICS COMPARISON TABLE")
        table_lines.append("=" * 120)
        
        # Header row
        header = f"{'Model':<{model_name_width}}"
        for _, metric_display in table_metrics:
            header += f"{metric_display:>{metric_width}}"
        table_lines.append(header)
        table_lines.append("-" * len(header))
        
        # Data rows
        for model_name in model_names:
            results = self.results[model_name]
            row = f"{model_name:<{model_name_width}}"
            
            for metric_key, _ in table_metrics:
                value = results.get(metric_key, 0)
                formatted_value = f"{value*100:.1f}%"
                row += f"{formatted_value:>{metric_width}}"
            
            table_lines.append(row)
        
        # Add cold/warm start sample counts
        table_lines.append("-" * len(header))
        first_model = list(self.results.values())[0]
        cold_samples = first_model.get('cold_start_samples', 0)
        warm_samples = first_model.get('warm_start_samples', 0)
        total_samples = first_model.get('total', 0)
        
        table_lines.append(f"{'Sample Distribution:':<{model_name_width}}")
        table_lines.append(f"{'Total Samples:':<{model_name_width}}{total_samples:,}")
        table_lines.append(f"{'Cold Start:':<{model_name_width}}{cold_samples:,} ({(cold_samples/total_samples)*100:.1f}%)")
        table_lines.append(f"{'Warm Start:':<{model_name_width}}{warm_samples:,} ({(warm_samples/total_samples)*100:.1f}%)")
        
        table_lines.append("=" * 120)
        
        return "\n".join(table_lines)
    
    def generate_comparison_report(self, output_file: str = None) -> str:
        """
        Generate a comprehensive comparison report
        
        Args:
            output_file: Optional file path to save the report
            
        Returns:
            String containing the formatted report
        """
        if not self.results:
            return "No evaluation results available. Run evaluate_all_models first."
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_lines = [
            "=" * 80,
            "NATR Model Comparison Report",
            "=" * 80,
            f"Generated: {timestamp}",
            f"Device: {self.device}",
            f"Number of models evaluated: {len(self.results)}",
            "",
        ]
        
        # Model overview
        report_lines.extend([
            "Model Overview:",
            "-" * 40,
        ])
        
        for model_name, results in self.results.items():
            config_info = results.get('model_config', {})
            checkpoint_info = config_info.get('checkpoint_info', {})
            
            report_lines.extend([
                f"{model_name}:",
                f"  - Total parameters: {config_info.get('config', {}).get('total_params', 'Unknown')}",
                f"  - Training epoch: {checkpoint_info.get('epoch', 'Unknown')}",
                f"  - Best training metric: {checkpoint_info.get('best_metric', 'Unknown')}",
                f"  - Model timestamp: {checkpoint_info.get('timestamp', 'Unknown')}",
                ""
            ])
        
        # Performance comparison table
        report_lines.extend([
            "Performance Comparison:",
            "=" * 80,
        ])
        
        # Key metrics to compare - focused on user's requirements
        key_metrics = [
            ('recall@10', 'Overall Recall@10', '%.2f%%'),
            ('recall@20', 'Overall Recall@20', '%.2f%%'),
            ('purchase_recall@10', 'Purchase Recall@10', '%.2f%%'),
            ('purchase_recall@20', 'Purchase Recall@20', '%.2f%%'),
            ('checkout_recall@10', 'InitiateCheckout Recall@10', '%.2f%%'),
            ('checkout_recall@20', 'InitiateCheckout Recall@20', '%.2f%%'),
            ('add_to_cart_recall@10', 'AddToCart Recall@10', '%.2f%%'),
            ('add_to_cart_recall@20', 'AddToCart Recall@20', '%.2f%%'),
            ('weighted_recall@10', 'Weighted Recall@10', '%.2f%%'),
            ('weighted_recall@20', 'Weighted Recall@20', '%.2f%%'),
            ('cold_start_weighted_recall@10', 'Cold Start Weighted@10', '%.2f%%'),
            ('warm_start_weighted_recall@10', 'Warm Start Weighted@10', '%.2f%%'),
        ]
        
        # Create comparison table
        model_names = list(self.results.keys())
        
        # Header
        header = f"{'Metric':<25}"
        for name in model_names:
            header += f"{name:<20}"
        report_lines.append(header)
        report_lines.append("-" * len(header))
        
        # Metrics rows
        for metric_key, metric_name, format_str in key_metrics:
            row = f"{metric_name:<25}"
            
            for model_name in model_names:
                value = self.results[model_name].get(metric_key, 0)
                
                # Format percentage metrics
                if '%%' in format_str:
                    formatted_value = format_str % (value * 100)
                else:
                    formatted_value = format_str % value
                
                row += f"{formatted_value:<20}"
            
            report_lines.append(row)
        
        # Sample counts including cold/warm start breakdown
        report_lines.extend([
            "",
            "Sample Counts & Cold/Warm Start Distribution:",
            "-" * 60,
        ])
        
        # Assume all models evaluated on same test set, so use first model's counts
        first_model = list(self.results.values())[0]
        total_samples = first_model.get('total', 0)
        cold_start_samples = first_model.get('cold_start_samples', 0)
        warm_start_samples = first_model.get('warm_start_samples', 0)
        
        # Overall sample distribution
        report_lines.extend([
            f"Total samples: {int(total_samples):,}",
            f"Cold start samples: {int(cold_start_samples):,} ({(cold_start_samples/total_samples)*100:.1f}%)",
            f"Warm start samples: {int(warm_start_samples):,} ({(warm_start_samples/total_samples)*100:.1f}%)",
            ""
        ])
        
        # Event type breakdown
        count_metrics = ['purchase', 'checkout', 'add_to_cart', 'intent', 'view']
        for count_type in count_metrics:
            count_value = first_model.get(count_type, 0)
            if isinstance(count_value, (int, float)) and count_value > 0:
                percentage = (count_value / total_samples) * 100
                report_lines.append(f"{count_type.title()} samples: {int(count_value):,} ({percentage:.1f}%)")
        
        # Rankings
        report_lines.extend([
            "",
            "Model Rankings:",
            "=" * 80,
        ])
        
        ranking_metrics = [
            ('recall@10', 'Overall Recall@10'),
            ('purchase_recall@10', 'Purchase Recall@10'),
            ('checkout_recall@10', 'InitiateCheckout Recall@10'),
            ('add_to_cart_recall@10', 'AddToCart Recall@10'),
            ('weighted_recall@10', 'Weighted Recall@10'),
            ('weighted_recall@20', 'Weighted Recall@20'),
        ]
        
        for metric_key, metric_name in ranking_metrics:
            # Sort models by this metric
            sorted_models = sorted(
                self.results.items(),
                key=lambda x: x[1].get(metric_key, 0),
                reverse=True
            )
            
            report_lines.append(f"\nRanking by {metric_name}:")
            for rank, (model_name, results) in enumerate(sorted_models, 1):
                value = results.get(metric_key, 0)
                if '%%' in str(metric_name):
                    formatted_value = f"{value*100:.2f}%"
                else:
                    formatted_value = f"{value:.4f}"
                report_lines.append(f"  {rank}. {model_name}: {formatted_value}")
        
        # Cold Start vs Warm Start Analysis
        report_lines.extend([
            "",
            "Cold Start vs Warm Start Analysis:",
            "=" * 80,
        ])
        
        # Create cold/warm start comparison table
        cold_warm_metrics = [
            ('cold_start_weighted_recall@10', 'warm_start_weighted_recall@10', 'Weighted Recall@10'),
            ('cold_start_purchase_recall@10', 'warm_start_purchase_recall@10', 'Purchase Recall@10'),
            ('cold_start_checkout_recall@10', 'warm_start_checkout_recall@10', 'InitiateCheckout Recall@10'),
            ('cold_start_add_to_cart_recall@10', 'warm_start_add_to_cart_recall@10', 'AddToCart Recall@10'),
        ]
        
        # Header for cold/warm comparison
        header = f"{'Model':<25}{'Metric':<25}{'Cold Start':<15}{'Warm Start':<15}{'Difference':<15}"
        report_lines.append(header)
        report_lines.append("-" * len(header))
        
        for model_name, results in self.results.items():
            for cold_metric, warm_metric, metric_name in cold_warm_metrics:
                cold_value = results.get(cold_metric, 0) * 100
                warm_value = results.get(warm_metric, 0) * 100
                difference = warm_value - cold_value
                
                row = f"{model_name:<25}{metric_name:<25}{cold_value:.2f}%{'':<10}{warm_value:.2f}%{'':<10}{difference:+.2f}%"
                report_lines.append(row)
                model_name = ""  # Don't repeat model name for subsequent metrics
        
        # Performance gap analysis
        report_lines.extend([
            "",
            "Cold/Warm Start Performance Gap Analysis:",
            "-" * 50,
        ])
        
        for model_name, results in self.results.items():
            cold_weighted = results.get('cold_start_weighted_recall@10', 0) * 100
            warm_weighted = results.get('warm_start_weighted_recall@10', 0) * 100
            gap = warm_weighted - cold_weighted
            
            if gap > 5:
                gap_desc = "Large gap (>5%)"
            elif gap > 2:
                gap_desc = "Medium gap (2-5%)"
            elif gap > 0:
                gap_desc = "Small gap (<2%)"
            else:
                gap_desc = "Cold start performs better"
            
            report_lines.append(f"{model_name}: {gap_desc} - Warm start is {gap:+.2f}% better")
        
        # Additional insights
        report_lines.extend([
            "",
            "Key Insights:",
            "=" * 80,
        ])
        
        # Find best performing model overall
        best_weighted_model = max(
            self.results.items(),
            key=lambda x: x[1].get('weighted_recall@10', 0)
        )
        
        best_purchase_model = max(
            self.results.items(),
            key=lambda x: x[1].get('purchase_recall@10', 0)
        )
        
        # Find best cold/warm start performers
        best_cold_start_model = max(
            self.results.items(),
            key=lambda x: x[1].get('cold_start_weighted_recall@10', 0)
        )
        
        best_warm_start_model = max(
            self.results.items(),
            key=lambda x: x[1].get('warm_start_weighted_recall@10', 0)
        )
        
        report_lines.extend([
            f"• Best overall model (Weighted Recall@10): {best_weighted_model[0]}",
            f"  - Weighted Recall@10: {best_weighted_model[1].get('weighted_recall@10', 0)*100:.2f}%",
            "",
            f"• Best purchase prediction model: {best_purchase_model[0]}",
            f"  - Purchase Recall@10: {best_purchase_model[1].get('purchase_recall@10', 0)*100:.2f}%",
            "",
            f"• Best cold start model: {best_cold_start_model[0]}",
            f"  - Cold Start Weighted Recall@10: {best_cold_start_model[1].get('cold_start_weighted_recall@10', 0)*100:.2f}%",
            "",
            f"• Best warm start model: {best_warm_start_model[0]}",
            f"  - Warm Start Weighted Recall@10: {best_warm_start_model[1].get('warm_start_weighted_recall@10', 0)*100:.2f}%",
            "",
        ])
        
        # Performance analysis
        if len(self.results) >= 2:
            purchase_recalls = [r.get('purchase_recall@10', 0) for r in self.results.values()]
            weighted_recalls = [r.get('weighted_recall@10', 0) for r in self.results.values()]
            
            report_lines.extend([
                "Performance Statistics:",
                f"  - Purchase Recall@10 range: {min(purchase_recalls)*100:.2f}% - {max(purchase_recalls)*100:.2f}%",
                f"  - Weighted Recall@10 range: {min(weighted_recalls)*100:.2f}% - {max(weighted_recalls)*100:.2f}%",
                f"  - Average Purchase Recall@10: {np.mean(purchase_recalls)*100:.2f}%",
                f"  - Average Weighted Recall@10: {np.mean(weighted_recalls)*100:.2f}%",
                "",
            ])
        
        # Footer
        report_lines.extend([
            "=" * 80,
            "End of Report",
            "=" * 80,
        ])
        
        # Join all lines
        full_report = "\n".join(report_lines)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(full_report)
            print(f"\nReport saved to: {output_file}")
        
        return full_report
    
    def export_results_to_csv(self, output_file: str):
        """
        Export results to CSV for further analysis
        
        Args:
            output_file: Path to save CSV file
        """
        if not self.results:
            print("No results to export")
            return
        
        # Prepare data for CSV
        csv_data = []
        
        for model_name, results in self.results.items():
            row = {'model_name': model_name}
            
            # Add all numeric metrics
            for key, value in results.items():
                if isinstance(value, (int, float)) and key != 'model_name':
                    row[key] = value
            
            csv_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(output_file, index=False)
        print(f"Results exported to CSV: {output_file}")


def precompute_event_flags(test_samples: List[Dict], event_to_idx: Dict[str, int], train_samples: List[Dict] = None) -> List[Dict]:
    """
    Pre-compute event flags for all test samples to ensure consistency across model evaluations
    
    Args:
        test_samples: List of test samples
        event_to_idx: Mapping of event names to indices
        train_samples: List of training samples (for cold start classification)
        
    Returns:
        List of samples with pre-computed event flags
    """
    print("Pre-computing event flags for consistent evaluation...")
    print(f"Using event_to_idx mapping: {event_to_idx}")
    
    # Get event indices
    checkout_idx = event_to_idx.get('InitiateCheckout', 3)
    add_to_cart_idx = event_to_idx.get('AddToCart', 2)
    print(f"Looking for checkout events with index: {checkout_idx}")
    print(f"Looking for add-to-cart events with index: {add_to_cart_idx}")
    
    purchase_count = 0
    checkout_count = 0
    add_to_cart_count = 0
    
    # Debug counters for raw detection
    raw_purchase_count = 0
    raw_checkout_count = 0
    raw_add_to_cart_count = 0
    
    # Build set of training users for cold start classification
    training_users = set()
    if train_samples:
        print("Building training user set for cold start classification...")
        for sample in train_samples:
            training_users.add(sample.get('user_id'))
        print(f"Found {len(training_users):,} unique users in training set")
    
    # Process each sample
    for sample in test_samples:
        # Check events in short-term sequence
        short_term_events = sample.get('short_term_events', [])
        
        # Purchase flag is already available (highest priority)
        is_purchase = sample.get('is_purchase', False)
        if is_purchase:
            raw_purchase_count += 1
        
        # Detect checkout events
        has_raw_checkout = any(event == checkout_idx for event in short_term_events)
        if has_raw_checkout:
            raw_checkout_count += 1
        
        # Detect add-to-cart events  
        has_raw_add_to_cart = any(event == add_to_cart_idx for event in short_term_events)
        if has_raw_add_to_cart:
            raw_add_to_cart_count += 1
        
        # Set inclusive flags (for individual event recalls)
        # These count ALL occurrences regardless of hierarchy
        sample['has_checkout_inclusive'] = has_raw_checkout
        sample['has_add_to_cart_inclusive'] = has_raw_add_to_cart
        
        # Set hierarchical flags (for weighted recall - no double counting)
        # Only the highest priority event should be marked as True
        if is_purchase:
            # Purchase overrides all other events
            sample['has_checkout'] = False
            sample['has_add_to_cart'] = False
            purchase_count += 1
        elif has_raw_checkout:
            # Checkout overrides AddToCart but not Purchase
            sample['has_checkout'] = True
            sample['has_add_to_cart'] = False
            checkout_count += 1
        elif has_raw_add_to_cart:
            # AddToCart only if no higher priority events
            sample['has_checkout'] = False
            sample['has_add_to_cart'] = True
            add_to_cart_count += 1
        else:
            # No special events - just viewing
            sample['has_checkout'] = False
            sample['has_add_to_cart'] = False
        
        # Pre-compute cold/warm start classification based on training set presence
        # Cold start = user not seen in training set (due to temporal split)
        user_id = sample.get('user_id')
        if training_users:
            sample['is_cold_start'] = user_id not in training_users
        else:
            # Fallback if no training samples provided
            sample['is_cold_start'] = False
    
    print(f"Raw event detection in test set (for individual recalls):")
    print(f"  Sessions with Purchase: {raw_purchase_count:,}")
    print(f"  Sessions with Checkout: {raw_checkout_count:,}")
    print(f"  Sessions with Add-to-cart: {raw_add_to_cart_count:,}")
    
    print(f"Hierarchical event distribution (for weighted recall):")
    print(f"  Pure Purchase sessions: {purchase_count:,}")
    print(f"  Pure Checkout sessions: {checkout_count:,}")
    print(f"  Pure Add-to-cart sessions: {add_to_cart_count:,}")
    print(f"  Total samples: {len(test_samples):,}")
    
    # Report cold start statistics
    if training_users:
        cold_start_count = sum(1 for sample in test_samples if sample.get('is_cold_start', False))
        warm_start_count = len(test_samples) - cold_start_count
        print(f"Cold/Warm start distribution:")
        print(f"  Cold start users (not in training): {cold_start_count:,} ({cold_start_count/len(test_samples)*100:.1f}%)")
        print(f"  Warm start users (in training): {warm_start_count:,} ({warm_start_count/len(test_samples)*100:.1f}%)")
    
    return test_samples


def main():
    """Main function to run model comparison"""
    parser = argparse.ArgumentParser(description='Compare NATR model performance using unified metrics')
    parser.add_argument('--checkpoints-dir', type=str, default='checkpoints/natr',
                        help='Directory containing model checkpoints')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output file for the comparison report (default: output/training_comparison/)')
    parser.add_argument('--csv-export', type=str, default=None,
                        help='Export results to CSV file (default: output/training_comparison/)')
    parser.add_argument('--k-values', type=int, nargs='+', default=[5, 10, 20],
                        help='K values for recall@k metrics')
    parser.add_argument('--test-mode', action='store_true',
                        help='Use small test dataset for quick evaluation')
    parser.add_argument('--max-batches', type=int, default=None,
                        help='Maximum number of batches to evaluate per model (for quick testing)')
    
    args = parser.parse_args()
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Set test mode environment variable if requested
    if args.test_mode:
        os.environ["NATR_TEST_MODE"] = "1"
        os.environ["NATR_MAX_SAMPLES"] = "5000"  # Smaller for quick comparison
        print("*** TEST MODE: Using small dataset subset ***")
    
    # Initialize data processors (needed for evaluation)
    print("\nInitializing data processors...")
    
    package_processor = PackageProcessor(
        data_path="data/feed.parquet",
        cache_dir='data/cache',
        load_coordinates=True,
        load_embeddings=True,
        api_key=os.environ.get("OPENAI_API_KEY"),
        embedding_model='text-embedding-3-small',
        use_reduced_embeddings=True
    )
    
    session_processor = SessionProcessor(
        data_path="data/bookit_events_data_13_months.parquet",
        cache_dir='data/cache',
        min_interactions=10,
        max_sessions_per_user=20,
        max_samples_per_user=10
    )
    
    # Load and process data (reuse existing processing)
    try:
        print("Loading data...")
        package_processor.load_data()
        session_processor.load_data()
        package_processor.create_mappings()
        session_processor.create_mappings()
        session_processor.extract_sessions()
        
        # Prepare samples
        from utils.training_utils import limit_samples_for_testing, filter_by_min_session_length, filter_items_by_frequency, time_based_split_year
        
        samples = session_processor.prepare_enhanced_training_data()
        samples = limit_samples_for_testing(samples)  # Apply test mode limiting if enabled
        
        # Apply same filters as training
        quality_samples = filter_by_min_session_length(samples, min_session_length=3)
        filtered_samples, valid_packages = filter_items_by_frequency(quality_samples, min_frequency=50)
        
        # Use the test split
        train_samples, test_samples = time_based_split_year(filtered_samples, train_ratio=0.93)
        
        print(f"Test dataset: {len(test_samples):,} samples")
        
        # Pre-compute event flags for consistent evaluation across all models
        event_to_idx = session_processor.get_idx_mappings()['event_to_idx']
        test_samples = precompute_event_flags(test_samples, event_to_idx, train_samples)
        
        # Create test dataloader
        from utils.training_utils import create_dataloaders
        _, test_loader = create_dataloaders(
            train_samples, test_samples, package_processor, session_processor,
            batch_size=64, num_workers=0, use_weighted_sampling=False
        )
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Initialize model comparator
    comparator = ModelComparator(device=device, k_values=args.k_values)
    
    # Auto-discover and load models
    comparator.auto_discover_models(args.checkpoints_dir)
    
    if not comparator.models:
        print("No models found to compare. Make sure you have trained models in the checkpoints directory.")
        return
    
    # Evaluate all models
    results = comparator.evaluate_all_models(test_loader, max_batches=args.max_batches)
    
    # Generate and display metrics table
    print("\n" + "="*50)
    print("DISPLAYING METRICS TABLE")
    print("="*50)
    metrics_table = comparator.generate_metrics_table()
    print(metrics_table)
    
    # Create output directory
    output_dir = "output/training_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate comprehensive report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.output_file:
        # If user provides path, respect it but ensure it's in the output directory
        if not args.output_file.startswith(output_dir):
            output_file = os.path.join(output_dir, os.path.basename(args.output_file))
        else:
            output_file = args.output_file
    else:
        output_file = os.path.join(output_dir, f"model_comparison_report_{timestamp}.txt")
    
    # Save metrics table to separate file
    table_file = output_file.replace('.txt', '_metrics_table.txt')
    with open(table_file, 'w') as f:
        f.write(metrics_table)
    
    # Generate and save comprehensive report
    report = comparator.generate_comparison_report(output_file)
    
    # Export to CSV
    if args.csv_export:
        # If user provides path, respect it but ensure it's in the output directory
        if not args.csv_export.startswith(output_dir):
            csv_file = os.path.join(output_dir, os.path.basename(args.csv_export))
        else:
            csv_file = args.csv_export
    else:
        # Default CSV export in output directory
        csv_file = os.path.join(output_dir, f"model_comparison_results_{timestamp}.csv")
    
    comparator.export_results_to_csv(csv_file)
    
    print(f"\nComparison complete!")
    print(f"📊 Metrics table saved to: {table_file}")
    print(f"📄 Full report saved to: {output_file}")
    print(f"📈 CSV results saved to: {csv_file}")
    print(f"📁 All outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
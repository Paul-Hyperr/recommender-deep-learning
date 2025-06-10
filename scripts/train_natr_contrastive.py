"""
NATR training script with contrastive learning and event-based margins
Optimized for more organic learning of event importance hierarchies
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import sys
import time
import json
import glob
from datetime import datetime
from collections import defaultdict, Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.backends.cudnn as cudnn

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure tqdm to avoid terminal display issues
os.environ['TQDM_DISABLE'] = '0'  # Ensure tqdm is enabled
os.environ['TQDM_POSITION'] = '0'  # Set default position

# Import tqdm and configure it to handle terminal clearing better
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tqdm")

# Enable optimization settings
cudnn.benchmark = True
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')

# Import all necessary components
from models.natr import NATR, NATRConfig
from utils.session_processor import SessionProcessor
from utils.package_processor import PackageProcessor
from utils.loss_functions import EnhancedEventMetrics
import torch.nn.functional as F


class PurchaseFocusedLoss(nn.Module):
    """
    Loss function that ONLY optimizes for purchase prediction with comprehensive anti-overfitting measures
    Prevents popularity bias and encourages item coverage diversity
    """
    def __init__(self, k=20, margin=1.0, regularization_weight=0.01, item_frequencies=None):
        super().__init__()
        self.k = k
        self.margin = margin
        self.regularization_weight = regularization_weight
        
        # Store item frequencies for popularity bias prevention
        if item_frequencies is not None:
            # Normalize frequencies to create popularity weights
            total_freq = item_frequencies.sum()
            self.register_buffer('popularity_weights', item_frequencies / total_freq)
        else:
            self.popularity_weights = None
            
        # Track which items are being predicted to measure coverage
        self.predicted_items = set()
        
    def forward(self, predictions, targets, is_purchase, has_checkout=None, has_add_to_cart=None):
        """
        Ranking loss ONLY for purchase samples + comprehensive overfitting prevention
        """
        device = predictions.device
        
        if targets.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # ONLY optimize for purchase samples - let model learn event influence organically
        purchase_mask = is_purchase
        
        if not purchase_mask.any():
            # No purchases in batch - use small regularization loss only
            return torch.tensor(self.regularization_weight, device=device, requires_grad=True)
        
        # Get purchase samples only
        purchase_indices = purchase_mask.nonzero(as_tuple=True)[0]
        purchase_predictions = predictions[purchase_indices]
        purchase_targets = targets[purchase_indices].long()  # Convert to int64 for gather
        
        # Get target scores for purchase samples
        target_scores = torch.gather(purchase_predictions, 1, purchase_targets.unsqueeze(1)).squeeze(1)
        
        # Get K-th highest scores as thresholds
        top_k_scores, _ = torch.topk(purchase_predictions, self.k, dim=1)
        kth_scores = top_k_scores[:, -1]  # K-th highest score
        
        # Ranking loss: purchase targets should be in top-K
        purchase_loss = F.relu(kth_scores - target_scores + self.margin)
        
        # === COMPREHENSIVE ANTI-OVERFITTING REGULARIZATION ===
        
        # 1. Prediction diversity: prevent all predictions becoming similar
        pred_std = torch.std(predictions, dim=1).mean()
        diversity_loss = F.relu(0.5 - pred_std)  # Encourage std > 0.5
        
        # 2. Item coverage: penalize if only predicting same items repeatedly
        with torch.no_grad():
            # Track top predicted items to measure coverage
            _, top_items = torch.topk(predictions, self.k, dim=1)
            for batch_top_items in top_items:
                for item in batch_top_items:
                    self.predicted_items.add(item.item())
        
        # Coverage regularization: encourage diverse item predictions
        num_unique_predictions = len(self.predicted_items)
        total_items = predictions.size(1)
        coverage_ratio = num_unique_predictions / total_items
        coverage_loss = F.relu(torch.tensor(0.3 - coverage_ratio, device=device))  # Encourage >30% item coverage
        
        # 3. Popularity bias prevention: penalize over-relying on popular items
        popularity_loss = torch.tensor(0.0, device=device)
        if self.popularity_weights is not None:
            # Get top-k predicted items and their popularity weights
            _, top_predicted = torch.topk(predictions, self.k, dim=1)
            
            # Calculate average popularity of predicted items
            batch_popularity = torch.tensor(0.0, device=device)
            for batch_idx in range(top_predicted.size(0)):
                item_popularities = self.popularity_weights[top_predicted[batch_idx]]
                batch_popularity += item_popularities.mean()
            
            avg_popularity = batch_popularity / top_predicted.size(0)
            # Penalize if average popularity is too high (>0.1 = top 10% of items)
            popularity_loss = F.relu(avg_popularity - 0.1)
        
        # 4. Reset prediction tracking periodically to avoid memory growth
        if len(self.predicted_items) > 10000:  # Reset every ~10k items
            self.predicted_items.clear()
        
        # Combine all losses
        total_loss = (purchase_loss.mean() + 
                     self.regularization_weight * diversity_loss +
                     self.regularization_weight * coverage_loss +
                     self.regularization_weight * popularity_loss)
        
        return total_loss
from utils.unified_metrics import UnifiedMetricsTracker
from utils.memory_utils import (
    detect_device, create_memory_config, MemoryOptimizer, AMPManager, 
    GradientAccumulator, get_optimal_batch_size, print_memory_stats,
    create_optimizer_with_memory_optimizations, get_model_size
)

# Import consolidated utilities
from utils.training_utils import (
    filter_by_min_session_length,
    filter_items_by_frequency,
    time_based_split_year,
    identify_event_types,
    ensure_balanced_test_set,
    move_batch_to_device,
    train_epoch,
    evaluate,
    create_dataloaders,
    analyze_data_distribution,
    set_performance_mode,
    clear_memory,
    limit_samples_for_testing
)

# Import the enhanced evaluation function from train_natr.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_natr import evaluate_with_unified_metrics


def analyze_learned_event_weights(model, test_loader, device, max_batches=10):
    """
    Analyze what event influence weights the model has learned
    by comparing prediction scores for different event types
    """
    model.eval()
    
    event_scores = {
        'purchase': [],
        'checkout': [], 
        'addtocart': [],
        'view': []
    }
    
    with torch.no_grad():
        batch_count = 0
        for batch in test_loader:
            if batch_count >= max_batches:
                break
                
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            predictions = outputs['predictions']
            
            # Get event type flags
            is_purchase = batch.get('is_purchase', torch.zeros(len(predictions), dtype=torch.bool))
            has_checkout = batch.get('has_checkout', torch.zeros(len(predictions), dtype=torch.bool))
            has_add_to_cart = batch.get('has_add_to_cart', torch.zeros(len(predictions), dtype=torch.bool))
            
            # Get target scores for each event type
            targets = batch['purchased']['package_ids']
            
            for i in range(len(predictions)):
                if i < len(targets):
                    target_idx = targets[i].item()
                    if 0 <= target_idx < predictions.size(1):
                        target_score = predictions[i, target_idx].item()
                        
                        # Categorize by event type (exclusive categories)
                        if is_purchase[i]:
                            event_scores['purchase'].append(target_score)
                        elif has_checkout[i]:
                            event_scores['checkout'].append(target_score)
                        elif has_add_to_cart[i]:
                            event_scores['addtocart'].append(target_score)
                        else:
                            event_scores['view'].append(target_score)
            
            batch_count += 1
    
    # Calculate average scores and relative weights
    avg_scores = {}
    for event_type, scores in event_scores.items():
        if scores:
            avg_scores[event_type] = sum(scores) / len(scores)
        else:
            avg_scores[event_type] = 0.0
    
    # Calculate relative influence weights (normalize to purchase = 100%)
    purchase_score = avg_scores.get('purchase', 1.0)
    if purchase_score > 0:
        relative_weights = {
            event_type: (score / purchase_score) * 100 
            for event_type, score in avg_scores.items()
        }
    else:
        relative_weights = {event_type: 0.0 for event_type in avg_scores.keys()}
    
    return avg_scores, relative_weights, event_scores


class ContrastiveAwareDataset(torch.utils.data.Dataset):
    """
    Extended dataset that handles different event types for contrastive learning
    This dataset adds event type flags to model input
    """
    
    def __init__(self, samples, package_processor, user_to_idx, package_to_idx, event_to_idx,
                max_short_term=10, max_long_term=20, use_cache=True, prefetch_features=True):
        """Initialize the dataset with contrastive learning support"""
        from utils.package_processor import TravelPackageDataset
        
        # Create base dataset
        self.base_dataset = TravelPackageDataset(
            samples, package_processor, user_to_idx, package_to_idx, event_to_idx,
            max_short_term, max_long_term, use_cache, prefetch_features
        )
        
        # Keep reference to samples for event flags
        self.samples = samples
    
    def __len__(self):
        """Return the length of the dataset"""
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        """Enhanced getitem that passes event type flags to the model"""
        # Get base sample
        sample = self.base_dataset[idx]
        
        # Add event type flags
        sample['is_purchase'] = self.samples[idx].get('is_purchase', False)
        sample['has_checkout'] = self.samples[idx].get('has_checkout', False)
        sample['has_add_to_cart'] = self.samples[idx].get('has_add_to_cart', False)
        
        return sample


def create_contrastive_dataloaders(train_samples, test_samples, package_processor, session_processor, 
                                  batch_size=32, num_workers=0, use_weighted_sampling=True):
    """Create optimized dataloaders for training and testing with contrastive awareness"""
    
    # Get mappings
    user_to_idx = session_processor.get_idx_mappings()['user_to_idx']
    package_to_idx = session_processor.get_idx_mappings()['package_to_idx']
    event_to_idx = session_processor.get_idx_mappings()['event_to_idx']
    
    # Create datasets
    print("Creating training dataset...")
    train_dataset = ContrastiveAwareDataset(
        train_samples,
        package_processor,
        user_to_idx,
        package_to_idx,
        event_to_idx,
        max_short_term=10,
        max_long_term=20,
        use_cache=True,
        prefetch_features=True  # Prefetch training features for speed
    )
    
    print("Creating test dataset...")
    test_dataset = ContrastiveAwareDataset(
        test_samples,
        package_processor,
        user_to_idx,
        package_to_idx,
        event_to_idx,
        max_short_term=10,
        max_long_term=20,
        use_cache=True,
        prefetch_features=True  # Now prefetching test features too for speed
    )
    
    # Create weighted sampler for training if requested
    sampler = None
    if use_weighted_sampling:
        # Calculate event type counts
        purchase_count = sum(1 for s in train_samples if s.get('is_purchase', False))
        checkout_count = sum(1 for s in train_samples if not s.get('is_purchase', False) 
                            and s.get('has_checkout', False))
        add_to_cart_count = sum(1 for s in train_samples if not s.get('is_purchase', False) 
                              and not s.get('has_checkout', False) and s.get('has_add_to_cart', False))
        view_count = len(train_samples) - purchase_count - checkout_count - add_to_cart_count
        
        # Set weights for different sample types
        purchase_boost = 10.0    # High weight for actual purchases (consistent with other scripts)
        checkout_boost = 5.0     # Medium-high weight for checkouts
        add_to_cart_boost = 2.0  # Medium weight for add-to-cart
        view_weight = 0.5        # Base weight for regular browsing
        
        # Calculate weights for each sample based on event type hierarchy
        sample_weights = []
        for sample in train_samples:
            if sample.get('is_purchase', False):
                # Purchase samples get highest weight
                weight = purchase_boost
            elif sample.get('has_checkout', False):
                # Checkout samples get medium-high weight
                weight = checkout_boost
            elif sample.get('has_add_to_cart', False):
                # Add-to-cart samples get medium weight
                weight = add_to_cart_boost
            else:
                # Regular browsing samples get base weight
                weight = view_weight
            
            sample_weights.append(weight)
        
        print(f"Using weighted sampling:")
        print(f"  Purchase samples boost: {purchase_boost}")
        print(f"  Checkout samples boost: {checkout_boost}")
        print(f"  Add-to-cart samples boost: {add_to_cart_boost}")
        print(f"  View samples weight: {view_weight}")
        
        # Create sampler - sample with replacement to ensure important events are seen frequently
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_samples),
            replacement=True
        )
    
    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),  # Only shuffle if not using sampler
        sampler=sampler,            # Use weighted sampler if created
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() or torch.backends.mps.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=3 if num_workers > 0 else None,  # Increased prefetch factor
        drop_last=True,  # Drop last incomplete batch for stable training
        generator=torch.Generator().manual_seed(42)  # Consistent shuffling
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,  # Larger batches for evaluation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() or torch.backends.mps.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=3 if num_workers > 0 else None  # Increased prefetch factor
    )
    
    return train_loader, test_loader


def evaluate_with_enhanced_metrics(model, test_loader, device, k_values=[5, 10, 20]):
    """Evaluate model with enhanced contrastive-specific metrics for different event types"""
    model.eval()
    
    # Initialize metrics accumulator
    metrics_acc = defaultdict(float)
    event_type_counts = defaultdict(int)
    
    # MPS-specific optimization
    is_mps = device.type == 'mps'
    
    with torch.no_grad():
        # Handle MPS garbage collection
        if is_mps:
            clear_memory(device)
        
        # Use smaller batches for evaluation on MPS to avoid OOM
        progress_bar = tqdm(test_loader, desc="Evaluating",
                           mininterval=2.0,   # Update at most every 2 seconds
                           miniters=5,        # Update after at least 5 iterations
                           ncols=80,          # Fixed width to prevent wrapping
                           leave=False,       # Don't keep evaluation progress bar
                           position=1)        # Position below training bar
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = move_batch_to_device(batch, device)
            
            # Forward pass
            outputs = model(batch)
            predictions = outputs['predictions']
            targets = batch['purchased']['package_ids']
            
            # Get event type flags
            is_purchase = batch.get('is_purchase', torch.zeros_like(targets, dtype=torch.bool))
            has_checkout = batch.get('has_checkout', torch.zeros_like(targets, dtype=torch.bool))
            has_add_to_cart = batch.get('has_add_to_cart', torch.zeros_like(targets, dtype=torch.bool))
            # Get inclusive flags for individual event recall calculations
            has_checkout_inclusive = batch.get('has_checkout_inclusive', torch.zeros_like(targets, dtype=torch.bool))
            has_add_to_cart_inclusive = batch.get('has_add_to_cart_inclusive', torch.zeros_like(targets, dtype=torch.bool))
            
            # Calculate comprehensive metrics using the EnhancedEventMetrics specialized implementation
            batch_metrics = EnhancedEventMetrics.calculate_metrics(
                predictions, targets, is_purchase, has_checkout, has_add_to_cart, k_values,
                has_checkout_inclusive=has_checkout_inclusive, has_add_to_cart_inclusive=has_add_to_cart_inclusive
            )
            
            # Accumulate counts
            batch_size = len(targets)
            event_type_counts['total'] += batch_size
            event_type_counts['purchase'] += batch_metrics.get('purchase_count', 0)
            event_type_counts['checkout'] += batch_metrics.get('checkout_count', 0)
            event_type_counts['add_to_cart'] += batch_metrics.get('add_to_cart_count', 0)
            event_type_counts['intent'] += batch_metrics.get('intent_count', 0)  # Purchase + Checkout
            
            # Accumulate metrics - weighted by count
            for k in k_values:
                # Overall recall
                metrics_acc[f'recall@{k}'] += batch_metrics['recall@k'][k] * batch_size
                
                # Event-specific recall
                if 'purchase_count' in batch_metrics and batch_metrics['purchase_count'] > 0:
                    metrics_acc[f'purchase_recall@{k}'] += batch_metrics['purchase_recall@k'][k] * batch_metrics['purchase_count']
                
                if 'checkout_count' in batch_metrics and batch_metrics['checkout_count'] > 0:
                    metrics_acc[f'checkout_recall@{k}'] += batch_metrics['checkout_recall@k'][k] * batch_metrics['checkout_count']
                
                if 'add_to_cart_count' in batch_metrics and batch_metrics['add_to_cart_count'] > 0:
                    metrics_acc[f'add_to_cart_recall@{k}'] += batch_metrics['add_to_cart_recall@k'][k] * batch_metrics['add_to_cart_count']
                
                if 'intent_count' in batch_metrics and batch_metrics['intent_count'] > 0:
                    metrics_acc[f'intent_recall@{k}'] += batch_metrics['intent_recall@k'][k] * batch_metrics['intent_count']
            
            # MRR metrics
            metrics_acc['overall_mrr'] += batch_metrics['overall_mrr'] * batch_size
            
            if 'purchase_count' in batch_metrics and batch_metrics['purchase_count'] > 0:
                metrics_acc['purchase_mrr'] += batch_metrics['purchase_mrr'] * batch_metrics['purchase_count']
            
            if 'checkout_count' in batch_metrics and batch_metrics['checkout_count'] > 0:
                metrics_acc['checkout_mrr'] += batch_metrics['checkout_mrr'] * batch_metrics['checkout_count']
            
            if 'add_to_cart_count' in batch_metrics and batch_metrics['add_to_cart_count'] > 0:
                metrics_acc['add_to_cart_mrr'] += batch_metrics['add_to_cart_mrr'] * batch_metrics['add_to_cart_count']
            
            if 'intent_count' in batch_metrics and batch_metrics['intent_count'] > 0:
                metrics_acc['intent_mrr'] += batch_metrics['intent_mrr'] * batch_metrics['intent_count']
            
            # Periodic memory cleanup for MPS
            if is_mps and (batch_idx + 1) % 10 == 0:
                clear_memory(device)
                
                # Show memory usage in progress bar if available
                if hasattr(torch.mps, 'current_allocated_memory'):
                    mem_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
                    progress_bar.set_postfix({'memory_mb': f"{mem_mb:.0f}"})
    
    # Final cleanup for MPS
    if is_mps:
        clear_memory(device)
    
    # Normalize accumulated metrics
    metrics = {}
    
    # Recall@k metrics
    metrics['recall@k'] = {}
    for k in k_values:
        metrics['recall@k'][k] = metrics_acc[f'recall@{k}'] / event_type_counts['total']
    
    # Purchase recall@k
    if event_type_counts['purchase'] > 0:
        metrics['purchase_recall@k'] = {}
        for k in k_values:
            metrics['purchase_recall@k'][k] = metrics_acc[f'purchase_recall@{k}'] / event_type_counts['purchase']
    
    # Checkout recall@k
    if event_type_counts['checkout'] > 0:
        metrics['checkout_recall@k'] = {}
        for k in k_values:
            metrics['checkout_recall@k'][k] = metrics_acc[f'checkout_recall@{k}'] / event_type_counts['checkout']
    
    # Add-to-cart recall@k
    if event_type_counts['add_to_cart'] > 0:
        metrics['add_to_cart_recall@k'] = {}
        for k in k_values:
            metrics['add_to_cart_recall@k'][k] = metrics_acc[f'add_to_cart_recall@{k}'] / event_type_counts['add_to_cart']
    
    # Intent (purchase + checkout) recall@k
    if event_type_counts['intent'] > 0:
        metrics['intent_recall@k'] = {}
        for k in k_values:
            metrics['intent_recall@k'][k] = metrics_acc[f'intent_recall@{k}'] / event_type_counts['intent']
    
    # MRR metrics
    metrics['overall_mrr'] = metrics_acc['overall_mrr'] / event_type_counts['total']
    
    if event_type_counts['purchase'] > 0:
        metrics['purchase_mrr'] = metrics_acc['purchase_mrr'] / event_type_counts['purchase']
    
    if event_type_counts['checkout'] > 0:
        metrics['checkout_mrr'] = metrics_acc['checkout_mrr'] / event_type_counts['checkout']
    
    if event_type_counts['add_to_cart'] > 0:
        metrics['add_to_cart_mrr'] = metrics_acc['add_to_cart_mrr'] / event_type_counts['add_to_cart']
    
    if event_type_counts['intent'] > 0:
        metrics['intent_mrr'] = metrics_acc['intent_mrr'] / event_type_counts['intent']
    
    # Add counts
    metrics.update(event_type_counts)
    
    return metrics


def get_contrastive_config(mode="balanced"):
    """Get contrastive learning specific parameters based on performance mode"""
    config = {}
    
    # Special mode for Apple Silicon (M1/M2/M3)
    is_mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    
    if mode == "apple_silicon" or (mode == "balanced" and is_mps):
        # Contrastive margin parameters optimized for Apple Silicon
        config.update({
            # Contrastive margin parameters (negative margins boost logits)
            "temperature": 0.07,                 # Sharp learning for event combinations
            "purchase_margin": -0.8,             # Strong negative boost for purchases
            "checkout_margin": -0.3,             # Medium negative boost for checkouts
            "add_to_cart_margin": 0.0,           # Neutral margin for add-to-cart
            "view_margin": 0.2,                  # Small penalty for views
            "hard_negative_ratio": 0.7           # Proportion of hard negatives to keep
        })
    elif mode == "fastest":
        # Contrastive margin parameters - less aggressive for faster convergence
        config.update({
            "temperature": 0.15,                 # Moderate temperature
            "purchase_margin": -0.5,             # Moderate negative boost for purchases
            "checkout_margin": -0.2,             # Small negative boost for checkouts
            "add_to_cart_margin": 0.0,           # No margin for add-to-cart
            "view_margin": 0.1,                  # Smaller penalty for views
            "hard_negative_ratio": 0.5           # Fewer hard negatives for faster training
        })
    elif mode == "accurate":
        # Contrastive margin parameters - more aggressive for better accuracy
        config.update({
            "temperature": 0.05,                 # Very sharp learning
            "purchase_margin": -1.0,             # Very strong negative boost for purchases
            "checkout_margin": -0.4,             # Strong negative boost for checkouts
            "add_to_cart_margin": 0.0,           # No margin for add-to-cart
            "view_margin": 0.3,                  # Standard penalty for views
            "hard_negative_ratio": 0.8           # More hard negatives for better discrimination
        })
    else:  # "balanced" mode (non-MPS)
        # Contrastive margin parameters - balanced configuration
        config.update({
            "temperature": 0.07,                 # Sharp learning for event combinations
            "purchase_margin": -0.8,             # Strong negative boost for purchases
            "checkout_margin": -0.3,             # Medium negative boost for checkouts
            "add_to_cart_margin": 0.0,           # No margin for add-to-cart
            "view_margin": 0.2,                  # Small penalty for views
            "hard_negative_ratio": 0.7           # Standard hard negative ratio
        })
    
    return config


def main(performance_config=None, dataset="13months", event_data_path=None):
    """Main function implementing contrastive learning with event-based margins
    
    Args:
        performance_config: Performance configuration dict
        dataset (str): Dataset to use - '13months' or '2months'
        event_data_path (str): Optional explicit path to event data file
    """
    # Set device - for Apple Silicon (M1/M2/M3), we can use MPS
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        # MPS-specific optimizations
        clear_memory(device)
        print("Apple MPS device detected - applying Metal-specific optimizations")
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Delete package_features.pkl cache to force regeneration with new embedding model
    package_features_cache = "data/cache/package_features.pkl"
    if os.path.exists(package_features_cache):
        print(f"Removing package features cache to use text-embedding-3-small (1536 dimensions)")
        os.remove(package_features_cache)
        
    # Also clear any dataset caches to ensure they use the new embeddings
    for cache_file in glob.glob("data/cache/dataset_*.pkl"):
        print(f"Removing dataset cache: {cache_file}")
        os.remove(cache_file)
    
    # Use default performance config if none provided
    if performance_config is None:
        performance_config = set_performance_mode("balanced")
    
    # Add contrastive specific parameters
    contrastive_config = get_contrastive_config(
        "apple_silicon" if device.type == "mps" else performance_config.get("mode", "balanced")
    )
    performance_config.update(contrastive_config)
    
    # Data paths
    package_data_path = "data/feed.parquet"
    
    # Determine event data path
    if event_data_path is None:
        # Map dataset selection to file path
        dataset_map = {
            '13months': 'data/bookit_events_data_13_months.parquet',
            '2months': 'data/bookit_events_2_months.parquet'
        }
        
        event_data_path = dataset_map.get(dataset)
        if not event_data_path:
            raise ValueError(f"Unknown dataset: {dataset}. Use '13months' or '2months'")
        
        # Check if file exists
        if not os.path.exists(event_data_path):
            raise FileNotFoundError(f"Event data file not found: {event_data_path}")
    
    print(f"Using event data: {event_data_path}")
    
    # Initialize processors
    print("\nInitializing data processors...")
    package_processor = PackageProcessor(
        feed_data_path=package_data_path,  # Updated parameter name
        cache_dir='data/cache',
        load_coordinates=True,
        load_embeddings=True,
        api_key=os.environ.get("OPENAI_API_KEY"),
        embedding_model='text-embedding-3-small',  # Use small model for lower dimensionality
        use_reduced_embeddings=performance_config.get("use_reduced_embeddings", True)
    )
    
    session_processor = SessionProcessor(
        event_data_path=event_data_path,  # Updated parameter name
        cache_dir='data/cache',
        min_interactions=5,
        max_sessions_per_user=20,
        max_samples_per_user=10
    )
    
    # Apply performance configuration to training parameters
    # Set base batch size according to device
    if device.type == 'cpu':
        base_batch_size = 32
        num_workers = 0
    elif device.type == 'mps':
        # Apple Silicon optimized settings with increased batch size
        base_batch_size = 128  # Increased from 64 to 128 for better utilization
        num_workers = performance_config.get("num_workers", 0)  # Default to 0 for MPS to avoid multiprocessing issues
    else:  # cuda
        base_batch_size = 128
        num_workers = 4
    
    # If direct batch_size is specified, use it, otherwise calculate from multiplier
    if "batch_size" in performance_config:
        batch_size = performance_config["batch_size"]
    else:
        batch_size = int(base_batch_size * performance_config.get("batch_size_multiplier", 1.0))
    learning_rate = performance_config.get("learning_rate", 0.001)
    num_epochs = performance_config.get("epochs", 30)
    accumulation_steps = performance_config.get("accumulation_steps", 2)
    
    print(f"\nTraining parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Gradient accumulation steps: {accumulation_steps}")
    print(f"  Workers: {num_workers}")
    
    # Report contrastive parameters
    print(f"\nContrastive learning parameters:")
    print(f"  Temperature: {performance_config.get('temperature', 0.1)}")
    print(f"  Purchase margin: {performance_config.get('purchase_margin', 0.0)}")
    print(f"  Checkout margin: {performance_config.get('checkout_margin', 0.3)}")
    print(f"  Add-to-cart margin: {performance_config.get('add_to_cart_margin', 0.6)}")
    print(f"  View margin: {performance_config.get('view_margin', 1.0)}")
    print(f"  Hard negative ratio: {performance_config.get('hard_negative_ratio', 0.7)}")
    
    # Load and process data
    print("\n1. Loading package data...")
    package_processor.load_data()
    
    print("\n2. Loading event data...")
    session_processor.load_data()
    
    print("\n3. Creating mappings...")
    package_processor.create_mappings()
    session_processor.create_mappings()
    
    print("\n4. Extracting user sessions...")
    session_processor.extract_sessions()
    
    print("\n5. Preparing training samples...")
    samples = session_processor.prepare_enhanced_training_data()
    
    # Limit samples for testing if needed
    samples = limit_samples_for_testing(samples)
    
    # Enhance samples with event type identification
    print("\n6. Enhancing samples with event types...")
    event_to_idx = session_processor.get_idx_mappings()['event_to_idx']
    enhanced_samples = identify_event_types(samples, event_to_idx)
    
    print("\n7. Applying data filters...")
    
    # Apply session length filtering - quality improves with longer sessions
    quality_samples = filter_by_min_session_length(enhanced_samples, min_session_length=2)  # Reduced from 3
    
    # Filter items by frequency - focus on packages with sufficient data
    filtered_samples, valid_packages = filter_items_by_frequency(quality_samples, min_frequency=5)  # Reduced from 10
    
    # Validate that we have different event types
    purchase_count = sum(1 for s in filtered_samples if s.get('is_purchase', False))
    checkout_count = sum(1 for s in filtered_samples if not s.get('is_purchase', False) and s.get('has_checkout', False))
    add_to_cart_count = sum(1 for s in filtered_samples if not s.get('is_purchase', False) 
                          and not s.get('has_checkout', False) and s.get('has_add_to_cart', False))
    
    if purchase_count == 0:
        raise ValueError("No purchase events found in the filtered samples. Cannot train the model.")
    
    print(f"Purchase events: {purchase_count} ({purchase_count / len(filtered_samples) * 100:.2f}% of all events)")
    print(f"Checkout events: {checkout_count} ({checkout_count / len(filtered_samples) * 100:.2f}% of all events)")
    print(f"Add-to-cart events: {add_to_cart_count} ({add_to_cart_count / len(filtered_samples) * 100:.2f}% of all events)")
    print(f"Combined purchase signals: {purchase_count + checkout_count + add_to_cart_count} " + 
          f"({(purchase_count + checkout_count + add_to_cart_count) / len(filtered_samples) * 100:.2f}% of all events)")
    
    # Time-based split
    train_samples, test_samples, split_date = time_based_split_year(filtered_samples, train_ratio=0.91)
    
    # Ensure evaluation set has enough events of each type for meaningful metrics
    train_samples, test_samples = ensure_balanced_test_set(
        train_samples, test_samples, min_purchases=50
    )
    
    # IMPORTANT: Update user mappings to include ALL users from both train and test sets
    print("\n7b. Updating user mappings to include test users...")
    all_users = set()
    for sample in train_samples + test_samples:
        all_users.add(sample['user_id'])
    
    # Check if we have unmapped users
    unmapped_users = all_users - set(session_processor.user_to_idx.keys())
    if unmapped_users:
        print(f"Found {len(unmapped_users)} unmapped users (likely from test set)")
        # Add them to the mapping
        max_idx = max(session_processor.user_to_idx.values()) if session_processor.user_to_idx else 0
        for user_id in unmapped_users:
            max_idx += 1
            session_processor.user_to_idx[user_id] = max_idx
        print(f"Updated user mappings. Total users: {len(session_processor.user_to_idx)}")
    
    # Analyze distributions
    analyze_data_distribution(train_samples, "Train")
    analyze_data_distribution(test_samples, "Test")
    
    # Create datasets and dataloaders with contrastive-specific dataset
    print("\n8. Creating dataloaders...")
    train_loader, test_loader = create_contrastive_dataloaders(
        train_samples, test_samples, package_processor, session_processor,
        batch_size=batch_size, num_workers=num_workers, use_weighted_sampling=True
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset):,} samples ({len(train_loader):,} batches)")
    print(f"  Test: {len(test_loader.dataset):,} samples ({len(test_loader):,} batches)")
    
    # Get model dimensions
    print("\n9. Creating model...")
    
    # Create config with performance parameters
    hidden_dim = performance_config.get("hidden_dim", 256)
    embedding_dim = performance_config.get("embedding_dim", 256)
    dropout = performance_config.get("dropout", 0.2)
    
    # Get actual dimensions from mappings
    num_users = max(session_processor.user_to_idx.values()) + 1
    num_packages = max(session_processor.package_to_idx.values()) + 1
    
    # Optionally limit package count for faster training
    max_package_count = performance_config.get("max_package_count", None)
    if max_package_count is not None and max_package_count < num_packages:
        print(f"Limiting package count from {num_packages:,} to {max_package_count:,}")
        num_packages = max_package_count
    
    num_countries = max(package_processor.country_to_idx.values()) + 1
    num_categories = max(package_processor.category_to_idx.values()) + 1
    num_themes = max(package_processor.theme_to_idx.values()) + 1
    
    # Detect actual embedding dimension from the data
    # Get the first package's title embedding from the prepared tensors
    package_tensors = package_processor.prepare_package_tensors()
    actual_embedding_dim = package_tensors['title_embeddings'].shape[1]
    print(f"Detected title embedding dimension: {actual_embedding_dim}")
    
    # Create model config with detected embedding dimension
    config = NATRConfig(
        num_users=num_users,
        num_packages=num_packages,
        num_countries=num_countries,
        num_categories=num_categories,
        num_themes=num_themes,
        title_embedding_dim=actual_embedding_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        user_embedding_dim=embedding_dim,
        dropout=dropout,
        max_short_term=10,
        max_long_term=20
    )
    
    # Create model
    model = NATR(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Initialize popularity bias based on item frequencies
    print("Initializing popularity bias...")
    # Calculate item frequencies from training data
    item_counts = Counter()
    for sample in train_samples:
        # Count all packages in sessions
        for pkg in sample.get('short_term_packages', []):
            if pkg is not None:
                item_counts[pkg] += 1
        for pkg in sample.get('long_term_packages', []):
            if pkg is not None:
                item_counts[pkg] += 1
        # Count purchased package
        purchased = sample.get('purchased_package')
        if purchased is not None:
            item_counts[purchased] += 1
    
    # Create frequency tensor for all packages
    item_frequencies = torch.zeros(config.num_packages, device=device)
    for package_id_str, count in item_counts.items():
        # Convert package ID to index
        package_idx = session_processor.package_to_idx.get(str(package_id_str), 0)
        if package_idx < config.num_packages:
            item_frequencies[package_idx] = count
    
    # Skip popularity initialization to test if it hurts contrastive learning on large data
    # model.initialize_popularity(item_frequencies)  # DISABLED for testing
    print(f"Skipping popularity initialization - testing pure contrastive learning")
    
    # Initialize optimizer with enhanced user learning (multi-rate optimization)
    print("Creating custom optimizer with enhanced user learning...")
    
    # Separate parameters for different learning rates
    user_params = []
    user_transform_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'user_encoder.user_embedding' in name:
            user_params.append(param)
        elif 'user_transform' in name:
            user_transform_params.append(param)
        else:
            other_params.append(param)
    
    # Create optimizer with different learning rates for enhanced user learning
    user_lr = learning_rate * 2.0  # 2x learning rate for user embeddings
    user_transform_lr = learning_rate * 10.0  # 10x learning rate for user transform
    
    optimizer = optim.AdamW([
        {'params': other_params, 'lr': learning_rate, 'weight_decay': 1e-4},
        {'params': user_params, 'lr': user_lr, 'weight_decay': 1e-5},  # Less regularization for users
        {'params': user_transform_params, 'lr': user_transform_lr, 'weight_decay': 1e-5}
    ], eps=1e-8)
    
    print(f"  Base learning rate: {learning_rate}")
    print(f"  User embedding learning rate: {user_lr}")
    print(f"  User transform learning rate: {user_transform_lr}")
    print(f"  User embedding params: {sum(p.numel() for p in user_params)}")
    print(f"  User transform params: {sum(p.numel() for p in user_transform_params)}")
    print(f"  Other params: {sum(p.numel() for p in other_params)}")
    
    # Use a simple step scheduler with no warmup
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',           # Optimize for maximizing recall
        factor=0.5,           # Halve the learning rate when plateauing
        patience=2,           # Wait 2 epochs before reducing
        threshold=0.0001,     # Minimum improvement to count as progress
        min_lr=1e-6           # Minimum learning rate
    )
    
    # Use PurchaseFocusedLoss with comprehensive anti-overfitting measures
    loss_fn = PurchaseFocusedLoss(
        k=20,                     # Optimize for top-20 performance (matches evaluation)
        margin=1.0,               # Ranking margin for purchase samples
        regularization_weight=0.01,  # Anti-overfitting regularization
        item_frequencies=item_frequencies  # For popularity bias prevention
    )
    print(f"Using PurchaseFocusedLoss - Pure purchase optimization with comprehensive overfitting prevention:")
    print(f"  Target: ONLY purchase samples optimized for top-20 ranking")
    print(f"  NO hardcoded weights for checkout/addtocart/view events")
    print(f"  Model learns event influence organically through purchase examples")
    print(f"  🛡️  Anti-overfitting measures:")
    print(f"    • Prediction diversity: Prevents similar predictions across features")
    print(f"    • Item coverage: Encourages >30% of catalog to be recommended")  
    print(f"    • Popularity bias prevention: Penalizes over-recommending popular items")
    print(f"  → Loss decreases ⟺ Purchase Recall@20 increases + High item coverage")
    print(f"  → Model discovers: which events predict purchases while maintaining diversity")
    
    # Print model configuration for debugging
    print(f"  Title embedding dimension: {config.title_embedding_dim}")
    print(f"  Hidden dimension: {config.hidden_dim}")
    print(f"  Embedding dimension: {config.embedding_dim}")
    print(f"  User embedding dimension: {config.user_embedding_dim}")
    
    # Compile model if enabled and available
    if performance_config.get("compile_model", False) and hasattr(torch, 'compile'):
        if device.type == 'cuda':
            print("Compiling model for faster execution using CUDA...")
            model = torch.compile(model)
        elif device.type == 'cpu' and sys.platform != 'darwin':
            # Compile for CPU (except on macOS where it might be unstable)
            print("Compiling model for faster execution using CPU...")
            model = torch.compile(model, backend="inductor")
        else:
            print("Model compilation not supported on this device")
    # Model compilation check complete
    
    # Training loop
    # Get performance parameters
    use_amp = performance_config.get("use_amp", False) and device.type == 'cuda'
    early_stopping_patience = performance_config.get("patience", 5)
    
    print("\n10. Starting training...")
    print(f"Training for {num_epochs} epochs with contrastive margins")
    print(f"Early stopping patience: {early_stopping_patience} epochs")
    
    # Track the metric we care about most - Purchase Recall@20 with MRR as tiebreaker
    best_purchase_recall_20 = 0
    best_purchase_mrr = 0  # Track purchase MRR for tiebreaking
    epochs_without_improvement = 0
    history = {'train_loss': [], 'test_metrics': []}
    
    for epoch in range(num_epochs):
        print(f"\n==== Epoch {epoch+1}/{num_epochs} ====")
        
        # Train for one epoch
        start_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch,
            accumulation_steps=accumulation_steps,
            use_amp=use_amp
        )
        train_time = time.time() - start_time
        
        # Evaluate with unified metrics (includes item coverage)
        start_time = time.time()
        # Create simple memory optimizer for evaluation
        memory_config = create_memory_config(device.type, "balanced")
        memory_optimizer = MemoryOptimizer(device, memory_config)
        test_metrics = evaluate_with_unified_metrics(model, test_loader, device, memory_optimizer, k_values=[5, 10, 20])
        eval_time = time.time() - start_time
        
        # Calculate combined recall metric @20 (purchase + checkout + add-to-cart), weighed by importance
        # Purchase gets full weight, checkout gets 0.6 weight, add-to-cart gets 0.2 weight
        purchase_recall_20 = test_metrics.get('purchase_recall@20', 0)
        checkout_recall_20 = test_metrics.get('checkout_recall@20', 0)
        cart_recall_20 = test_metrics.get('add_to_cart_recall@20', 0)
        
        # Also get @10 for display
        purchase_recall_10 = test_metrics.get('purchase_recall@10', 0)
        
        # Get counts from the counts dictionary
        counts = test_metrics.get('counts', {})
        purchase_count = counts.get('purchase', 0)
        checkout_count = counts.get('checkout', 0)
        cart_count = counts.get('add_to_cart', 0)
        total_important = purchase_count + checkout_count + cart_count
        
        # Focus only on purchase metrics (simplified for purchase-focused training)
        
        # Update scheduler with Purchase Recall@20 metric
        scheduler.step(purchase_recall_20)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['test_metrics'].append(test_metrics)
        
        # Print metrics
        print(f"\nTiming: Train={train_time:.1f}s, Eval={eval_time:.1f}s")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"\nTest Metrics:")
        print(f"  Purchase Recall@10: {purchase_recall_10*100:.2f}%")
        print(f"  Purchase Recall@20: {purchase_recall_20*100:.2f}%")
        
        if checkout_count > 0:
            print(f"  InitiateCheckout Recall@20: {checkout_recall_20*100:.2f}%")
        
        purchase_mrr = test_metrics.get('purchase_mrr', 0.0)
        print(f"  Purchase MRR: {purchase_mrr:.4f}")
        
        if 'item_coverage@20' in test_metrics:
            print(f"  Item Coverage@20: {test_metrics['item_coverage@20']*100:.2f}%")
        
        # Analyze learned event weights every few epochs
        if (epoch + 1) % 3 == 0:  # Every 3 epochs
            print(f"\n🧠 Analyzing Learned Event Influence Weights (Epoch {epoch+1}):")
            try:
                _, relative_weights, _ = analyze_learned_event_weights(model, test_loader, device, max_batches=5)
                
                print(f"  📊 Event Influence on Purchase Prediction:")
                print(f"    Purchase events: {relative_weights.get('purchase', 0):.0f}% (baseline=100%)")
                print(f"    Checkout events: {relative_weights.get('checkout', 0):.0f}% (influence on purchase)")
                print(f"    AddToCart events: {relative_weights.get('addtocart', 0):.0f}% (influence on purchase)")
                print(f"    ViewContent events: {relative_weights.get('view', 0):.0f}% (influence on purchase)")
                
                # Show if model is learning the expected hierarchy
                checkout_ratio = relative_weights.get('checkout', 0) / max(relative_weights.get('view', 1), 1)
                print(f"  🎯 Learned Ratios:")
                print(f"    Checkout vs View influence: {checkout_ratio:.1f}x stronger")
                
                if relative_weights.get('view', 0) < 10:  # Less than 10% influence
                    print(f"    ✅ ViewContent correctly learned as low influence ({relative_weights.get('view', 0):.1f}%)")
                else:
                    print(f"    ⚠️  ViewContent influence higher than expected ({relative_weights.get('view', 0):.1f}%)")
                    
            except Exception as e:
                print(f"  ⚠️  Could not analyze event weights: {e}")
        
        # Save best model based on Purchase Recall@20 with MRR as tiebreaker
        epsilon = 1e-6  # tolerance for considering recalls as similar
        current_purchase_mrr = test_metrics.get('purchase_mrr', 0.0)
        
        improvement = False
        if purchase_recall_20 > best_purchase_recall_20 + epsilon:
            # Clear improvement in Purchase Recall@20
            improvement = True
        elif abs(purchase_recall_20 - best_purchase_recall_20) <= epsilon:
            # Purchase Recall@20 is effectively the same, use MRR as tiebreaker
            if current_purchase_mrr > best_purchase_mrr:
                improvement = True
        
        if improvement:
            best_purchase_recall_20 = purchase_recall_20
            best_purchase_mrr = current_purchase_mrr
            epochs_without_improvement = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_purchase_recall_20': best_purchase_recall_20,
                'best_purchase_mrr': best_purchase_mrr,
                'config': config.__dict__,
                'valid_packages': list(valid_packages),
                'history': history,
                'performance_config': performance_config,
                'loss_type': 'PurchaseFocusedLoss',
                'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            }
            
            # Create checkpoints directory
            os.makedirs('checkpoints/natr', exist_ok=True)
            
            # Save to both a versioned file and the best model file
            torch.save(checkpoint, f"checkpoints/natr/contrastive_model_epoch_{epoch+1}.pth")
            torch.save(checkpoint, 'checkpoints/natr/contrastive_best_model.pth')
            print(f"  ✓ New best model saved! Purchase Recall@20: {best_purchase_recall_20*100:.2f}%, Purchase MRR: {best_purchase_mrr:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epochs")
            
            # Save periodic checkpoint every 5 epochs for recovery purposes
            if (epoch + 1) % 5 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_purchase_recall_20': best_purchase_recall_20,
                    'best_purchase_mrr': best_purchase_mrr,
                    'config': config.__dict__,
                    'valid_packages': list(valid_packages),
                    'history': history,
                    'performance_config': performance_config,
                    'margins': loss_fn.get_margins(),
                    'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                }
                torch.save(checkpoint, f"checkpoints/natr/contrastive_checkpoint_epoch_{epoch+1}.pth")
                print(f"  ✓ Periodic checkpoint saved at epoch {epoch+1}")
            
            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping after {epoch+1} epochs without improvement")
                break
    
    # Final summary
    print("\n11. Training complete!")
    print(f"Best Purchase Recall@20: {best_purchase_recall_20*100:.2f}%")
    print(f"Best Purchase MRR: {best_purchase_mrr:.4f}")
    
    # Save model info
    model_info = {
        'config': config.__dict__,
        'valid_packages': list(valid_packages),
        'best_purchase_recall_20': best_purchase_recall_20,
        'best_purchase_mrr': best_purchase_mrr,
        'checkpoint_path': 'checkpoints/natr/contrastive_best_model.pth',
        'package_data_path': package_data_path,
        'event_data_path': event_data_path,
        'performance_config': performance_config,
        'loss_function': 'PurchaseFocusedLoss',
        'training_date': datetime.now().strftime("%Y-%m-%d"),
        'package_count': len(valid_packages),
        'user_count': num_users,
        'training_strategy': 'contrastive_learning_enhanced',
        'primary_metric': 'purchase_recall@20',
        'tiebreaker_metric': 'purchase_mrr',
        'split_date': split_date,  # Save the train/test split date
        'train_ratio': 0.91  # Save the train ratio used
    }
    
    # Save model info to appropriate directory based on dataset
    output_dir = 'output/model_info' if args.dataset == '13months' else 'output/model_info_2months'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'model_info_contrastive.json'), 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("\nFiles saved:")
    print("  - checkpoints/natr/contrastive_best_model.pth")
    print("  - checkpoints/natr/contrastive_model_epoch_X.pth")
    print("  - model_info_contrastive.json")


if __name__ == "__main__":
    import sys
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train NATR model with contrastive learning')
    parser.add_argument('--mode', type=str, default='balanced', 
                        choices=['fastest', 'balanced', 'accurate', 'apple_silicon'],
                        help='Performance mode (default: balanced)')
    parser.add_argument('--dataset', type=str, default='13months',
                        choices=['13months', '2months'],
                        help='Which dataset to use: 13months or 2months (default: 13months)')
    parser.add_argument('--event-data', type=str, default=None,
                        help='Path to event data file (overrides --dataset)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size (default: determined by mode)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (default: determined by mode)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of data loading workers (default: determined by mode)')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Temperature for contrastive learning (default: determined by mode)')
    parser.add_argument('--purchase-margin', type=float, default=None,
                        help='Margin for purchase events (default: determined by mode)')
    parser.add_argument('--checkout-margin', type=float, default=None,
                        help='Margin for checkout events (default: determined by mode)')
    parser.add_argument('--cart-margin', type=float, default=None,
                        help='Margin for add-to-cart events (default: determined by mode)')
    parser.add_argument('--view-margin', type=float, default=None,
                        help='Margin for view events (default: determined by mode)')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Override learning rate (default: determined by mode)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check for Apple Silicon and recommend the mode if not explicitly set
    is_mps = torch.backends.mps.is_available()
    if is_mps and args.mode == 'balanced':
        print("Apple Silicon (MPS) device detected! Using optimized Apple Silicon settings.")
        print("For best performance, consider using '--mode apple_silicon'")
    
    # Set performance configuration
    performance_config = set_performance_mode(args.mode)
    performance_config["mode"] = args.mode  # Store the mode for contrastive config
    
    # Override with specific command-line arguments if provided
    if args.batch_size is not None:
        performance_config["batch_size"] = args.batch_size
        performance_config["batch_size_multiplier"] = 1.0
        print(f"Overriding batch size to {args.batch_size}")
    
    if args.workers is not None:
        performance_config["num_workers"] = args.workers
        print(f"Overriding number of workers to {args.workers}")
    
    if args.epochs is not None:
        performance_config["epochs"] = args.epochs
        print(f"Overriding epochs to {args.epochs}")
    
    if args.temperature is not None:
        performance_config["temperature"] = args.temperature
        print(f"Overriding temperature to {args.temperature}")
    
    if args.purchase_margin is not None:
        performance_config["purchase_margin"] = args.purchase_margin
        print(f"Overriding purchase margin to {args.purchase_margin}")
    
    if args.checkout_margin is not None:
        performance_config["checkout_margin"] = args.checkout_margin
        print(f"Overriding checkout margin to {args.checkout_margin}")
    
    if args.cart_margin is not None:
        performance_config["add_to_cart_margin"] = args.cart_margin
        print(f"Overriding add-to-cart margin to {args.cart_margin}")
    
    if args.view_margin is not None:
        performance_config["view_margin"] = args.view_margin
        print(f"Overriding view margin to {args.view_margin}")
    
    if args.learning_rate is not None:
        performance_config["learning_rate"] = args.learning_rate
        print(f"Overriding learning rate to {args.learning_rate}")
    
    # Start training with contrastive learning
    main(performance_config, dataset=args.dataset, event_data_path=args.event_data)
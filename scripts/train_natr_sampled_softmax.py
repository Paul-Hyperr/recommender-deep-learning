"""
Enhanced NATR training with Adaptive Sampled Softmax loss
Uses SessionProcessor2, NATREnhanced model with 6 views, and recent popularity boost

Key updates from working train_natr_enhanced_pre_finetune_fixed.py:
- SessionProcessor2 with purchase timestamp adjustment (00:00:00 → 23:59:59)
- NATREnhanced model with 6 views including events as separate view
- User price bias for personalized price sensitivity
- Recent popularity boost using last 20 days of training data
- Intent-based sampling preserving all purchases
- Enhanced user learning with different learning rates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import sys
import time
import json
import glob

from datetime import datetime
from tqdm import tqdm
from collections import defaultdict, Counter
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enable optimization settings
cudnn.benchmark = True
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')

# Import all necessary components
from models.natr_enhanced import NATREnhanced, NATRConfig
from utils.session_processor2 import SessionProcessor2
from utils.package_processor import PackageProcessor, TravelPackageDataset
from utils.unified_metrics import UnifiedMetricsTracker, MetricsTracker, EnhancedEventMetrics
from utils.recent_popularity import get_recent_popularity_from_samples
from utils.memory_utils import (
    detect_device, create_memory_config, MemoryOptimizer, AMPManager, 
    GradientAccumulator, get_optimal_batch_size, print_memory_stats,
    create_optimizer_with_memory_optimizations, get_model_size
)
from utils.training_utils import (
    filter_by_min_session_length, 
    filter_items_by_frequency, 
    time_based_split_year,
    analyze_data_distribution,
    move_batch_to_device,
    train_epoch,
    evaluate,
    create_dataloaders,
    set_performance_mode,
    limit_samples_for_testing,
    identify_event_types
)

# Import evaluation function and create_intent_samples_enhanced
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_natr_enhanced_pre_finetune_fixed import create_intent_samples_enhanced

# Import UnifiedMetricsTracker directly
from utils.unified_metrics import UnifiedMetricsTracker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('train_natr_sampled_softmax')


def evaluate_with_unified_metrics(model, test_loader, device, memory_optimizer, k_values=[5, 10, 20]):
    """Simple evaluation function using UnifiedMetricsTracker"""
    model.eval()
    
    # Initialize unified metrics tracker
    metrics_tracker = UnifiedMetricsTracker(k_values=k_values)
    
    with torch.no_grad():
        # Clear memory before evaluation
        memory_optimizer.before_epoch(0)
        
        progress_bar = tqdm(test_loader, desc="Evaluating")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Prepare batch and memory optimization
            memory_optimizer.before_batch(batch_idx)
            
            # Move batch to device with optimization
            batch = memory_optimizer.optimize_batch(batch)
            
            # Forward pass
            outputs = model(batch)
            predictions = outputs['predictions']
            
            
            targets = batch['purchased']['package_ids']
            
            # Ensure targets are long tensors
            if targets.dtype != torch.long:
                targets = targets.long()
            
            # Get event type indicators
            is_purchase = batch.get('is_purchase', torch.zeros_like(targets, dtype=torch.bool))
            has_checkout = batch.get('has_checkout', torch.zeros_like(targets, dtype=torch.bool))
            has_add_to_cart = batch.get('has_add_to_cart', torch.zeros_like(targets, dtype=torch.bool))
            
            # Update metrics tracker
            metrics_tracker.update(
                predictions=predictions,
                targets=targets,
                is_purchase=is_purchase,
                has_checkout=has_checkout,
                has_add_to_cart=has_add_to_cart
            )
            
            # Memory optimization after batch
            memory_optimizer.after_batch(batch_idx)
            
    
    # Clean up memory after evaluation
    memory_optimizer.after_epoch(0)
    
    # Compute metrics
    metrics = metrics_tracker.compute()
    
    return metrics


class UserBehaviorStats:
    """Track user behavior statistics for adaptive weighting"""
    def __init__(self):
        self.user_stats = defaultdict(lambda: {
            'views': 0,
            'carts': 0,
            'checkouts': 0,
            'purchases': 0,
            'total_events': 0
        })
    
    def update_from_samples(self, samples):
        """Calculate user statistics from training samples"""
        for sample in samples:
            user_id = sample['user_id']
            stats = self.user_stats[user_id]
            
            # Count events in short-term
            for event_type in sample.get('short_term_events', []):
                stats['total_events'] += 1
                if event_type == 1:  # ViewContent
                    stats['views'] += 1
                elif event_type == 2:  # AddToCart
                    stats['carts'] += 1
                elif event_type == 3:  # InitiateCheckout
                    stats['checkouts'] += 1
                elif event_type == 4:  # Purchase
                    stats['purchases'] += 1
            
            # Count final event
            if sample.get('is_purchase', False):
                stats['purchases'] += 1
                stats['total_events'] += 1
            elif sample.get('intent_level') == 'checkout':
                stats['checkouts'] += 1
                stats['total_events'] += 1
            elif sample.get('intent_level') == 'cart':
                stats['carts'] += 1
                stats['total_events'] += 1
            else:
                stats['views'] += 1
                stats['total_events'] += 1
    
    def get_user_weights(self, user_id, base_weights):
        """Get adaptive weights for a user based on their behavior"""
        stats = self.user_stats[user_id]
        
        if stats['total_events'] == 0:
            return base_weights
        
        # Calculate conversion rates
        checkout_to_purchase = stats['purchases'] / max(stats['checkouts'], 1) if stats['checkouts'] > 0 else 0
        cart_to_purchase = stats['purchases'] / max(stats['carts'], 1) if stats['carts'] > 0 else 0
        avg_views_per_purchase = stats['views'] / max(stats['purchases'], 1)
        
        # Adaptive weights based on user behavior
        weights = base_weights.copy()
        
        # Adjust checkout weight based on conversion rate
        if checkout_to_purchase > 0.7:  # High converter
            weights['checkout'] *= 1.2
        elif checkout_to_purchase < 0.3:  # Low converter
            weights['checkout'] *= 0.8
        
        # Adjust cart weight similarly
        if cart_to_purchase > 0.5:
            weights['cart'] *= 1.2
        elif cart_to_purchase < 0.1:
            weights['cart'] *= 0.7
        
        # Adjust view weight based on browsing behavior
        if avg_views_per_purchase < 10:  # Decisive buyer
            weights['multi_view'] *= 2.0
        elif avg_views_per_purchase > 50:  # Heavy browser
            weights['multi_view'] *= 0.5
        
        return weights


class SampledSoftmaxLoss(nn.Module):
    """
    Adaptive Sampled Softmax loss with user normalization
    L(ŷ) = -∑(x_j ∈ K) y_j log(ŷ_j)
    
    Where K is a sampled subset containing positive and negative samples
    """
    def __init__(self, num_items, num_negative_samples=100, temperature=1.0, 
                 item_frequencies=None, base_weights=None, skip_single_views=True):
        super().__init__()
        self.num_items = num_items
        self.num_negative_samples = num_negative_samples
        self.temperature = temperature
        self.skip_single_views = skip_single_views
        
        # Base weights for different intent levels
        self.base_weights = base_weights or {
            'purchase': 1.0,
            'checkout': 0.15,
            'cart': 0.08,
            'multi_view': 0.01,
            'single_view': 0.001
        }
        
        # Create log-uniform distribution for negative sampling
        if item_frequencies is not None and item_frequencies.sum() > 0:
            # Use actual item frequencies for log-uniform sampling
            freq_array = item_frequencies.cpu().numpy()
            # Add small epsilon to avoid log(0)
            freq_array = freq_array + 1e-10
            # Create log-uniform probabilities
            log_probs = np.log(freq_array)
            log_probs = log_probs - log_probs.max()  # Normalize for numerical stability
            self.sampling_probs = torch.from_numpy(np.exp(log_probs) / np.exp(log_probs).sum())
        else:
            # Uniform sampling if no frequencies provided
            self.sampling_probs = torch.ones(num_items) / num_items
    
    def normalize_weights_per_user(self, batch_data, soft_labels):
        """Normalize weights so each user contributes equally"""
        device = soft_labels.device
        batch_size = len(batch_data['user_id'])
        
        # Group samples by user
        user_indices = defaultdict(list)
        for idx in range(batch_size):
            user_id = batch_data['user_id'][idx].item()
            user_indices[user_id].append(idx)
        
        # Normalize per user
        normalized_labels = soft_labels.clone()
        
        for user_id, indices in user_indices.items():
            if len(indices) > 0:
                # Get this user's weights
                user_weights = soft_labels[indices]
                weight_sum = user_weights.sum()
                
                if weight_sum > 0:
                    # Normalize so user contributes total weight = 1.0
                    normalized_labels[indices] = user_weights / weight_sum
                    
                    # Scale to maintain purchase importance
                    # Find if user has any purchases
                    has_purchase = False
                    for idx in indices:
                        if batch_data.get('is_purchase', [False] * batch_size)[idx]:
                            has_purchase = True
                            break
                    
                    # If user has purchase, scale to maintain its relative importance
                    if has_purchase:
                        max_weight = user_weights.max()
                        normalized_labels[indices] *= max_weight
        
        return normalized_labels
            
    def forward(self, predictions, batch_data, user_behavior_stats=None):
        """
        Args:
            predictions: Model output logits [batch_size, num_items]
            batch_data: Dictionary containing all batch information
            user_behavior_stats: Optional UserBehaviorStats for adaptive weighting
        """
        batch_size = predictions.size(0)
        device = predictions.device
        
        # Extract targets and metadata
        targets = batch_data['purchased']['package_ids']
        if targets.dtype != torch.long:
            targets = targets.long()
        
        # Move sampling probabilities to device if needed
        if self.sampling_probs.device != device:
            self.sampling_probs = self.sampling_probs.to(device)
        
        # Get intent levels and event flags
        is_purchase = batch_data.get('is_purchase', torch.zeros(batch_size, dtype=torch.bool, device=device))
        intent_levels = batch_data.get('intent_level', ['ViewContent'] * batch_size)
        # Also check for target_intent which is used in the working script
        target_intents = batch_data.get('target_intent', intent_levels)
        
        # Create soft labels based on intent
        soft_labels = torch.zeros(batch_size, dtype=torch.float, device=device)
        
        for i in range(batch_size):
            # Skip single views if configured
            if self.skip_single_views and (i < len(intent_levels) and intent_levels[i] == 'single_view'):
                soft_labels[i] = 0.0
                continue
            
            # Get base weight for this intent level
            if is_purchase[i]:
                intent = 'purchase'
            else:
                # Use target_intent from the working script if available
                if i < len(target_intents) and target_intents[i] in ['InitiateCheckout', 'AddToCart']:
                    intent = target_intents[i]
                else:
                    intent = intent_levels[i] if i < len(intent_levels) else 'ViewContent'
            
            # Get user-specific weights if available
            if user_behavior_stats is not None:
                user_id = batch_data['user_id'][i].item()
                weights = user_behavior_stats.get_user_weights(user_id, self.base_weights)
            else:
                weights = self.base_weights
            
            soft_labels[i] = weights.get(intent, self.base_weights['multi_view'])
        
        # Normalize weights per user
        soft_labels = self.normalize_weights_per_user(batch_data, soft_labels)
        
        # Calculate loss only for non-zero weights
        total_loss = torch.tensor(0.0, device=device)
        num_valid = 0
        
        for i in range(batch_size):
            if soft_labels[i] > 0:
                target = targets[i]
                
                # Sample negative items using log-uniform distribution
                # Exclude the positive target from negative samples
                mask = torch.ones(self.num_items, dtype=torch.bool, device=device)
                mask[target] = False
                
                # Get sampling probabilities excluding the target
                masked_probs = self.sampling_probs.clone()
                masked_probs[target] = 0
                masked_probs = masked_probs / masked_probs.sum()  # Renormalize
                
                # Sample negative indices
                try:
                    negative_indices = torch.multinomial(
                        masked_probs, 
                        min(self.num_negative_samples, self.num_items - 1),
                        replacement=False
                    )
                except:
                    # Fallback to random sampling if multinomial fails
                    all_indices = torch.arange(self.num_items, device=device)
                    valid_indices = all_indices[mask]
                    perm = torch.randperm(len(valid_indices), device=device)
                    negative_indices = valid_indices[perm[:min(self.num_negative_samples, len(valid_indices))]]
                
                # Combine positive and negative indices
                # K = {positive sample} ∪ {negative samples}
                sampled_indices = torch.cat([target.unsqueeze(0), negative_indices])
                
                # Get predictions for sampled items only
                sampled_predictions = predictions[i, sampled_indices]
                
                # Apply temperature scaling
                sampled_predictions = sampled_predictions / self.temperature
                
                # Compute log softmax over the sampled subset K
                log_probs = F.log_softmax(sampled_predictions, dim=0)
                
                # The positive sample is always at index 0 in our sampled set
                # L = -log(ŷ_positive) where y_positive = 1 and y_negative = 0
                sample_loss = -log_probs[0] * soft_labels[i]
                
                total_loss = total_loss + sample_loss
                num_valid += 1
        
        # Average over valid samples
        if num_valid > 0:
            return total_loss / num_valid
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)


def train_epoch_sampled_softmax(model, train_loader, optimizer, loss_fn, device, epoch,
                               memory_optimizer, amp_manager, gradient_accumulator, 
                               user_behavior_stats=None):
    """Training with adaptive sampled softmax loss"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    memory_optimizer.before_epoch(epoch)
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", 
                      mininterval=5.0, miniters=50)
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(progress_bar):
        memory_optimizer.before_batch(batch_idx)
        batch = memory_optimizer.optimize_batch(batch)
        
        with amp_manager:
            outputs = model(batch)
            predictions = outputs['predictions']
            
            # Pass full batch data to loss function for adaptive weighting
            loss = loss_fn(predictions, batch, user_behavior_stats)
        
        original_loss = gradient_accumulator.backward(loss)
        
        total_loss += original_loss.item()
        num_batches += 1
        
        if gradient_accumulator.should_step():
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        if gradient_accumulator.step(optimizer):
            pass
        
        memory_optimizer.after_batch(batch_idx)
        
        postfix = {
            'loss': total_loss / num_batches,
            'lr': optimizer.param_groups[0]['lr']
        }
        
        if memory_optimizer.track_memory and memory_optimizer.peak_memory > 0:
            postfix['mem_mb'] = f"{memory_optimizer.peak_memory / (1024*1024):.0f}"
            
        progress_bar.set_postfix(postfix)
    
    if not gradient_accumulator.should_step():
        gradient_accumulator.step(optimizer)
    
    memory_optimizer.after_epoch(epoch)
    
    return total_loss / num_batches


def main(performance_mode="balanced", dataset="13months", event_data_path=None, epochs=None):
    """Main training function with Sampled Softmax loss"""
    
    # Device and memory setup
    device, device_type = detect_device()
    
    # Initialize metrics history for convergence visualization
    metrics_history = []
    model_name = "natr_sampled_softmax"
    print(f"📊 Tracking metrics for: {model_name}")
    memory_config = create_memory_config(device_type, performance_mode)
    memory_optimizer = MemoryOptimizer(device, memory_config)
    
    logger.info(f"Using device: {device} ({device_type})")
    logger.info(f"Performance mode: {performance_mode}")
    
    # Clear caches
    package_features_cache = "data/cache/package_features.pkl"
    if os.path.exists(package_features_cache):
        logger.info("Removing package features cache")
        os.remove(package_features_cache)
        
    for cache_file in glob.glob("data/cache/dataset_*.pkl"):
        logger.info(f"Removing dataset cache: {cache_file}")
        os.remove(cache_file)
    
    # Data paths
    package_data_path = "data/feed.parquet"
    
    if event_data_path is None:
        # Map dataset selection to file path (matching working script)
        dataset_map = {
            '13months': 'data/13_months_new2_clean.parquet',
            '2months': 'data/bookit_events_2_months.parquet'
        }
        
        event_data_path = dataset_map.get(dataset)
        if not event_data_path:
            raise ValueError(f"Unknown dataset: {dataset}. Use '13months' or '2months'")
        
        # Check if file exists
        if not os.path.exists(event_data_path):
            raise FileNotFoundError(f"Event data file not found: {event_data_path}")
    
    logger.info(f"Using event data: {event_data_path}")
    
    # Training parameters
    base_batch_size = get_optimal_batch_size(
        device_type, memory_config, model_size_mb=100, tensor_size_mb=5
    )
    
    if performance_mode == "fastest":
        num_epochs = 30
        learning_rate = 0.0005  # Reduced from 0.002 for stability
    elif performance_mode == "accurate":
        num_epochs = 100
        learning_rate = 0.0003  # Reduced from 0.0005
    else:
        num_epochs = 50
        learning_rate = 0.0005  # Reduced from 0.001 for stability
    
    # Override epochs if specified
    if epochs is not None:
        num_epochs = epochs
    
    # Initialize processors
    print("\nInitializing data processors...")
    package_processor = PackageProcessor(
        feed_data_path=package_data_path,
        cache_dir='data/cache',
        load_coordinates=True,
        load_embeddings=True,
        api_key=os.environ.get("OPENAI_API_KEY"),
        embedding_model='text-embedding-3-small',
        use_reduced_embeddings=True if device_type != 'cpu' else False
    )
    
    session_processor = SessionProcessor2(
        event_data_path=event_data_path,
        cache_dir='data/cache',
        session_timeout_hours=30,
        min_interactions=8,
        max_sessions_per_user=20,
        max_samples_per_user=10
    )
    
    print(f"\nTraining parameters:")
    print(f"  Dataset: {dataset}")
    print(f"  Device: {device}")
    print(f"  Batch size: {base_batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Loss: Adaptive Sampled Softmax with user normalization")
    
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
    
    # Create intent-based samples like in the working script
    print("\n5b. Creating intent-based samples...")
    event_to_idx = session_processor.get_idx_mappings()['event_to_idx']
    samples = create_intent_samples_enhanced(samples, event_to_idx)
    
    print("\n6. Applying data filters...")
    quality_samples = filter_by_min_session_length(samples, min_session_length=3)
    
    # Filter out single_view samples to focus on higher-intent interactions
    print("Filtering out single_view samples for better training efficiency...")
    samples_before_intent_filter = len(quality_samples)
    quality_samples = [s for s in quality_samples if s.get('intent_level', 'multi_view') != 'single_view']
    samples_after_intent_filter = len(quality_samples)
    print(f"Removed {samples_before_intent_filter - samples_after_intent_filter:,} single_view samples ({(samples_before_intent_filter - samples_after_intent_filter)/samples_before_intent_filter*100:.1f}%)")
    
    filtered_samples, valid_packages = filter_items_by_frequency(quality_samples, min_frequency=5)
    
    # Time-based split
    train_samples, test_samples, split_date = time_based_split_year(filtered_samples, train_ratio=0.91)
    
    # Ensure test set has purchases
    purchase_samples = [s for s in test_samples if s.get('is_purchase', False)]
    if len(purchase_samples) < 50:
        print(f"Warning: Only {len(purchase_samples)} purchases in test set, adding more...")
        extra_purchases = [s for s in train_samples if s.get('is_purchase', False)][:50-len(purchase_samples)]
        test_samples.extend(extra_purchases)
        train_samples = [s for s in train_samples if s not in extra_purchases]
    
    # Update user mappings
    print("\n6b. Updating user mappings...")
    all_users = set()
    for sample in train_samples + test_samples:
        all_users.add(sample['user_id'])
    
    unmapped_users = all_users - set(session_processor.user_to_idx.keys())
    if unmapped_users:
        print(f"Found {len(unmapped_users)} unmapped users")
        max_idx = max(session_processor.user_to_idx.values()) if session_processor.user_to_idx else 0
        for user_id in unmapped_users:
            max_idx += 1
            session_processor.user_to_idx[user_id] = max_idx
    
    analyze_data_distribution(train_samples, "Train")
    analyze_data_distribution(test_samples, "Test")
    
    # Calculate user behavior statistics for adaptive weighting
    print("\n7. Calculating user behavior statistics...")
    user_behavior_stats = UserBehaviorStats()
    user_behavior_stats.update_from_samples(train_samples)
    
    # Print some statistics
    total_users = len(user_behavior_stats.user_stats)
    avg_purchase_rate = np.mean([s['purchases'] / max(s['total_events'], 1) 
                                 for s in user_behavior_stats.user_stats.values()])
    print(f"  Total users: {total_users}")
    print(f"  Average purchase rate: {avg_purchase_rate:.2%}")
    
    # Create datasets
    print("\n8. Creating datasets...")
    print("📍 Using adaptive sampled softmax with intent-based targets")
    
    purchase_train_samples = [s for s in train_samples if s.get('is_purchase', False)]
    purchase_test_samples = [s for s in test_samples if s.get('is_purchase', False)]
    
    print(f"Purchase sessions: Train={len(purchase_train_samples):,} ({len(purchase_train_samples)/len(train_samples)*100:.1f}%), Test={len(purchase_test_samples):,}")
    
    # Use ALL sessions with adaptive target selection
    
    # Adaptive approach: Use ALL sessions with intent-based targets
    print("✅ Using ALL sessions with adaptive intent-based targets")
    print("📍 Targets: purchase > checkout > cart > multi-view items")
    print("📍 User-normalized weights for equal contribution")
    train_loader, test_loader = create_dataloaders(
        train_samples, test_samples, package_processor, session_processor,
        batch_size=base_batch_size, num_workers=0 if device_type == 'mps' else 2,
        use_weighted_sampling=False
    )
    samples_for_frequency = train_samples
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset):,} samples ({len(train_loader):,} batches)")
    print(f"  Test: {len(test_loader.dataset):,} samples ({len(test_loader):,} batches)")
    
    # Create model
    print("\n9. Creating model...")
    num_users = max(session_processor.user_to_idx.values()) + 1
    num_packages = max(session_processor.package_to_idx.values()) + 1
    num_countries = max(package_processor.country_to_idx.values()) + 1
    num_categories = max(package_processor.category_to_idx.values()) + 1
    num_themes = max(package_processor.theme_to_idx.values()) + 1
    
    package_tensors = package_processor.prepare_package_tensors()
    actual_embedding_dim = package_tensors['title_embeddings'].shape[1]
    
    config = NATRConfig(
        num_users=num_users,
        num_packages=num_packages,
        num_countries=num_countries,
        num_categories=num_categories,
        num_themes=num_themes,
        title_embedding_dim=actual_embedding_dim,
        hidden_dim=256,
        embedding_dim=128,  # Match working script
        user_embedding_dim=128,  # Match working script
        dropout=0.5,  # Higher dropout for stronger regularization
        max_short_term=10,
        max_long_term=20
    )
    
    # Create Enhanced NATR model
    print("\n🌟 Creating Enhanced NATR model with 6 views and user price bias...")
    print("  Views: Title, Coordinates, Country, Category/Theme, Price (w/ user bias), Time, Events")
    model = NATREnhanced(config).to(device)
    model_size_mb = get_model_size(model)
    print(f"Model size: {model_size_mb} MB")
    
    # Calculate item frequencies for negative sampling (NATR paper approach)
    print("Calculating purchase frequencies for log-uniform negative sampling...")
    item_frequencies = torch.zeros(num_packages)
    print(f"Using {len(samples_for_frequency)} samples for frequency calculation")
    
    for sample in samples_for_frequency:
        pkg_id = sample.get('purchased_package')
        if pkg_id and pkg_id in session_processor.package_to_idx:
            idx = session_processor.package_to_idx[pkg_id]
            # NATR paper: count actual purchase frequencies for negative sampling
            item_frequencies[idx] += 1.0
    
    # Initialize popularity bias based on item frequencies
    if hasattr(model, 'initialize_popularity'):
        print("Calculating purchase frequencies for popularity initialization...")
        package_counts = Counter()
        for sample in train_samples:
            if sample.get('is_purchase', False):
                package_counts[sample['purchased_package']] += 1
        
        # Create frequency tensor
        item_frequencies_for_pop = torch.zeros(num_packages, device=device)
        for package_id, count in package_counts.items():
            if isinstance(package_id, str):
                try:
                    package_id = int(package_id)
                except ValueError:
                    continue
            if 0 <= package_id < num_packages:
                item_frequencies_for_pop[package_id] = count
        
        max_freq = item_frequencies_for_pop.max().item()
        model.initialize_popularity(item_frequencies_for_pop)
        print(f"Initializing popularity bias with purchase frequencies (max freq: {max_freq})")
    
    # Calculate recent popularity scores using most recent 20 days of training data
    if hasattr(model, 'set_recent_popularity'):
        print("\n🔥 Calculating recent popularity boost...")
        recent_popularity, stats = get_recent_popularity_from_samples(
            samples=train_samples,
            num_packages=num_packages,
            days=20  # Use most recent 20 days
        )
        model.set_recent_popularity(recent_popularity)
        print(f"🔥 Recent popularity applied with weight: {model.recent_popularity_weight}")
    
    # Set up base weights for adaptive loss (matching working script)
    base_weights = {
        'Purchase': 4.0,          # Strong purchase signal 
        'InitiateCheckout': 0.12, # Reduced to prevent checkout dominance  
        'AddToCart': 0.05,        # Moderate cart signal
        'ViewContent': 0.001,     # Very low view signal
        # Map to simplified names used in loss function
        'purchase': 4.0,
        'checkout': 0.12,
        'cart': 0.05,
        'multi_view': 0.001,
        'single_view': 0.001
    }
    
    print("\n📊 Adaptive Sampled Softmax Configuration:")
    print("   • Enhanced NATR with 6 views + user price bias")
    print("   • User-specific weight normalization")
    print("   • Intent-based target selection (purchase > checkout > cart > multi-view)")
    print(f"   • Base weights: Purchase={base_weights['purchase']}, Checkout={base_weights['checkout']}, Cart={base_weights['cart']}, View={base_weights['multi_view']}")
    print("   • Log-uniform negative sampling from all other packages")
    print("   • Recent popularity boost (20 days)")
    print("   • Single_view samples filtered out at data level")
    print("   • Min session length: 3 interactions")
    print("   • 200 negative samples per positive")
    print(f"   • Training: {len(train_samples):,} sessions with {len(purchase_train_samples):,} purchases")
    
    # Create optimizer with enhanced user learning (following working script pattern)
    print("Creating custom optimizer with enhanced user learning...")
    
    # Separate parameters for different learning rates (matching working script)
    user_params = []           # user_encoder.user_embedding
    user_transform_params = [] # user_transform 
    user_price_params = []     # user_price_bias (new in Enhanced NATR)
    other_params = []
    
    for name, param in model.named_parameters():
        if 'user_encoder.user_embedding' in name:
            user_params.append(param)
        elif 'user_transform' in name:
            user_transform_params.append(param)
        elif 'user_price_bias' in name:
            user_price_params.append(param)
        else:
            other_params.append(param)
    
    # Use same learning rate multipliers as working script
    user_lr = learning_rate * 2.0  # 2x learning rate for user embeddings
    user_transform_lr = learning_rate * 10.0  # 10x learning rate for user transform
    user_price_lr = learning_rate * 3.0  # 3x learning rate for user price bias
    
    optimizer = optim.AdamW([
        {'params': other_params, 'lr': learning_rate, 'weight_decay': 1e-5},
        {'params': user_params, 'lr': user_lr, 'weight_decay': 1e-6},  # Less regularization for users
        {'params': user_transform_params, 'lr': user_transform_lr, 'weight_decay': 1e-6},
        {'params': user_price_params, 'lr': user_price_lr, 'weight_decay': 1e-6}
    ], eps=1e-8)
    
    print(f"  Base learning rate: {learning_rate}")
    print(f"  User embedding LR: {user_lr} ({sum(p.numel() for p in user_params):,} params)")
    print(f"  User transform LR: {user_transform_lr} ({sum(p.numel() for p in user_transform_params):,} params)")
    print(f"  User price bias LR: {user_price_lr} ({sum(p.numel() for p in user_price_params):,} params)")
    print(f"  Other params LR: {learning_rate} ({sum(p.numel() for p in other_params):,} params)")
    
    # Create AMP manager and gradient accumulator
    amp_manager = AMPManager(device, memory_config)
    gradient_accumulator = GradientAccumulator(
        steps=memory_config.gradient_accumulation_steps,
        amp_manager=amp_manager
    )
    
    # Create adaptive sampled softmax loss
    loss_fn = SampledSoftmaxLoss(
        num_items=num_packages,
        num_negative_samples=200,  # Reduced to focus predictions and lower item coverage
        temperature=0.8,           # Reduced for more confident predictions
        item_frequencies=item_frequencies,
        base_weights=base_weights,
        skip_single_views=False  # No longer needed since filtered at data level
    )
    print(f"✓ Adaptive Sampled Softmax loss initialized with {loss_fn.num_negative_samples} negative samples")
    
    # Add learning rate scheduler for stability
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=learning_rate * 0.1
    )
    print(f"✓ Cosine annealing scheduler with warm restarts (T_0=5, T_mult=2)")
    
    # Training loop
    print("\n10. Starting training...")
    
    # Record training start time
    start_time = time.time()
    
    best_purchase_recall = 0.0
    best_purchase_mrr = 0.0
    patience = 10 if performance_mode != "fastest" else 5
    epochs_without_improvement = 0
    previous_checkpoint = None  # Track previous checkpoint for cleanup
    
    os.makedirs('checkpoints/natr', exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # Train
        train_loss = train_epoch_sampled_softmax(
            model, train_loader, optimizer, loss_fn, device, epoch,
            memory_optimizer, amp_manager, gradient_accumulator, user_behavior_stats
        )
        
        print(f"Training loss: {train_loss:.4f}")
        
        # Step the scheduler
        scheduler.step()
        
        # Evaluate
        metrics = evaluate_with_unified_metrics(
            model, test_loader, device, memory_optimizer, k_values=[5, 10, 20]
        )
        
        # Print metrics
        print("\nEvaluation Results:")
        print(f"  Purchase Recall@10: {metrics.get('purchase_recall@10', 0)*100:.2f}%")
        print(f"  Purchase Recall@20: {metrics.get('purchase_recall@20', 0)*100:.2f}%")
        print(f"  Checkout Recall@20: {metrics.get('checkout_recall@20', 0)*100:.2f}%")
        print(f"  Item Coverage@20: {metrics.get('item_coverage@20', 0)*100:.2f}%")
        print(f"  Purchase MRR: {metrics.get('purchase_mrr', 0):.4f}")
        
        # Save metrics for convergence visualization
        metrics_history.append({
            'epoch': epoch + 1,
            'purchase_recall@10': metrics.get('purchase_recall@10', 0),
            'purchase_recall@20': metrics.get('purchase_recall@20', 0),
            'purchase_mrr': metrics.get('purchase_mrr', 0),
            'item_coverage@20': metrics.get('item_coverage@20', 0),
            'checkout_recall@20': metrics.get('checkout_recall@20', 0),
            'train_loss': train_loss
        })
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Track best metrics
        current_purchase_recall_20 = metrics.get('purchase_recall@20', 0)
        current_purchase_mrr = metrics.get('purchase_mrr', 0)
        
        epsilon = 1e-6
        improvement = (current_purchase_recall_20 > best_purchase_recall + epsilon) or \
                     (abs(current_purchase_recall_20 - best_purchase_recall) <= epsilon and 
                      current_purchase_mrr > best_purchase_mrr + epsilon)
        
        if improvement:
            best_purchase_recall = current_purchase_recall_20
            best_purchase_mrr = current_purchase_mrr
            epochs_without_improvement = 0
            
            # Save best model with performance in filename
            # Format recall with 2 decimals (e.g., 74_11 for 74.11%)
            recall_whole = int(best_purchase_recall * 100)
            recall_decimal = int((best_purchase_recall * 10000) % 100)
            checkpoint_filename = f'checkpoints/natr/best_model_sampled_softmax_enhanced_recall{recall_whole:02d}_{recall_decimal:02d}.pth'
            
            # Delete previous checkpoint if it exists
            if previous_checkpoint and os.path.exists(previous_checkpoint):
                os.remove(previous_checkpoint)
                print(f"  Removed previous checkpoint: {os.path.basename(previous_checkpoint)}")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.__dict__,
                'metrics': metrics,
                'train_loss': train_loss
            }, checkpoint_filename)
            print(f"✓ New best model saved to {checkpoint_filename}")
            print(f"  Recall@20: {best_purchase_recall*100:.2f}%, MRR: {best_purchase_mrr:.4f}")
            
            # Update previous checkpoint tracker
            previous_checkpoint = checkpoint_filename
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping after {patience} epochs without improvement")
            break
    
    # Calculate training duration
    end_time = time.time()
    training_duration = end_time - start_time
    
    # Final summary
    print("\n11. Training complete!")
    print(f"Best Purchase Recall@20: {best_purchase_recall*100:.2f}%")
    print(f"Best Purchase MRR: {best_purchase_mrr:.4f}")
    print(f"Training Duration: {training_duration//3600:.0f}h {(training_duration%3600)//60:.0f}m {training_duration%60:.0f}s")
    
    
    # Save metrics history for convergence visualization
    import json
    output_dir = 'output/graphs_data'
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_file = os.path.join(output_dir, f'{model_name}_metrics_history.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics_history, f, indent=2)
    print(f"📊 Saved metrics history to: {metrics_file}")
    
    # Save model info
    model_info = {
        'model_type': 'natr_enhanced_7views_user_price_bias',
        'config': config.__dict__,
        'valid_packages': list(valid_packages),
        'best_purchase_recall_20': best_purchase_recall,
        'best_purchase_mrr': best_purchase_mrr,
        'training_duration_seconds': training_duration,
        'training_duration_minutes': training_duration / 60,
        'training_duration_formatted': f"{training_duration//3600:.0f}h {(training_duration%3600)//60:.0f}m {training_duration%60:.0f}s",
        'checkpoint_path': checkpoint_filename,
        'training_approach': 'sampled_softmax_negative_sampling_enhanced',
        'loss_function': 'SampledSoftmaxLoss',
        'negative_samples': 200,
        'event_weights': base_weights,
        'split_date': split_date,
        'train_ratio': 0.91,
        'model_features': [
            'events_as_7th_view',
            'user_price_bias', 
            'recent_popularity_boost_20days',
            'intent_based_sampling',
            'session_processor2_timestamp_adjustment',
            'adaptive_sampled_softmax'
        ]
    }
    
    output_dir = 'output/model_info' if dataset == '13months' else 'output/model_info_2months'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'model_info_sampled_softmax_enhanced.json'), 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("\nFiles saved:")
    print("  - checkpoints/natr/best_model_sampled_softmax_enhanced.pth")
    print(f"  - {output_dir}/model_info_sampled_softmax_enhanced.json")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train NATR with Sampled Softmax loss')
    parser.add_argument('--mode', type=str, default='balanced', 
                        choices=['fastest', 'balanced', 'accurate'],
                        help='Performance mode')
    parser.add_argument('--dataset', type=str, default='13months',
                        choices=['13months', '2months'],
                        help='Which dataset to use')
    parser.add_argument('--event-data', type=str, default=None,
                        help='Path to event data file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides mode default)')
    
    args = parser.parse_args()
    
    # Override epochs if specified
    if args.epochs is not None:
        print(f"Overriding epochs to {args.epochs}")
    
    main(performance_mode=args.mode, dataset=args.dataset, event_data_path=args.event_data, epochs=args.epochs)
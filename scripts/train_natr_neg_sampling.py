"""
NATR training script with Log-Uniform Negative Sampling
Based on train_natr.py but implements efficient negative sampling from the paper
This allows for faster training and potentially better performance
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
from models.natr import NATR, NATRConfig
from models.natr_with_event_types import NATRWithEventTypes, NATRWithAdditiveEventTypes, EventAwareDataProcessor
from utils.session_processor import SessionProcessor
from utils.package_processor import PackageProcessor, TravelPackageDataset
from utils.unified_metrics import UnifiedMetricsTracker, MetricsTracker, EnhancedEventMetrics
from utils.loss_functions import NaturalPurchaseLoss, LogUniformSampledSoftmaxLoss, SampledSoftmaxEvaluator
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
    limit_samples_for_testing
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('train_natr')


def evaluate_with_unified_metrics(model, test_loader, device, memory_optimizer, k_values=[5, 10, 20], use_event_types=False):
    """
    Enhanced evaluation function using UnifiedMetricsTracker with memory optimization
    Combines detailed event-specific metrics from train_natr.py with memory optimization from train_natr_optimized.py
    """
    model.eval()
    
    # Initialize unified metrics tracker for comprehensive metrics
    metrics_tracker = UnifiedMetricsTracker(k_values=k_values)
    
    # Track sample counts for reporting
    total_samples = 0
    
    with torch.no_grad():
        # Clear memory before evaluation
        memory_optimizer.before_epoch(0)
        
        progress_bar = tqdm(test_loader, desc="Evaluating",
                           mininterval=5.0,   # Update at most every 2 seconds
                           ncols=None,        # Auto-adjust width
                           leave=False)       # Don't leave progress bar
        
        for batch_idx, batch in enumerate(progress_bar):
            # Prepare batch and memory optimization
            memory_optimizer.before_batch(batch_idx)
            
            # Move batch to device with optimization
            batch = memory_optimizer.optimize_batch(batch)
            
            # Add event types to batch if using event-aware model
            if use_event_types:
                batch = EventAwareDataProcessor.add_event_types_to_batch(batch)
            
            # Forward pass
            outputs = model(batch)
            predictions = outputs['predictions']
            targets = batch['purchased']['package_ids']
            
            # Get event type indicators with fallback to zeros
            is_purchase = batch.get('is_purchase', torch.zeros_like(targets, dtype=torch.bool))
            has_checkout = batch.get('has_checkout', torch.zeros_like(targets, dtype=torch.bool))
            has_add_to_cart = batch.get('has_add_to_cart', torch.zeros_like(targets, dtype=torch.bool))
            
            # Get inclusive flags for individual event recalls
            has_checkout_inclusive = batch.get('has_checkout_inclusive', has_checkout)  # Default to hierarchical if not present
            has_add_to_cart_inclusive = batch.get('has_add_to_cart_inclusive', has_add_to_cart)  # Default to hierarchical if not present
            
            # Update unified metrics tracker with all event types
            # For individual recalls, we need inclusive flags (samples that had these events)
            # For weighted recall, we need hierarchical flags (only highest priority event)
            metrics_tracker.update(
                predictions=predictions,
                targets=targets,
                is_purchase=is_purchase,
                has_checkout=has_checkout,  # Hierarchical for weighted recall
                has_add_to_cart=has_add_to_cart,  # Hierarchical for weighted recall
                has_checkout_inclusive=has_checkout_inclusive,  # Inclusive for individual recalls
                has_add_to_cart_inclusive=has_add_to_cart_inclusive  # Inclusive for individual recalls
            )
            
            total_samples += targets.size(0)
            
            # Memory optimization after batch
            memory_optimizer.after_batch(batch_idx)
            
            # Update progress bar with memory info if available
            if memory_optimizer.track_memory and batch_idx % 50 == 0:
                memory_stats = f"Mem: {memory_optimizer.peak_memory / (1024*1024):.0f}MB" if memory_optimizer.peak_memory > 0 else ""
                if memory_stats:
                    progress_bar.set_postfix_str(memory_stats)
    
    # Clean up memory after evaluation
    memory_optimizer.after_epoch(0)
    
    # Compute comprehensive metrics
    metrics = metrics_tracker.compute()
    
    print(f"\nEvaluation completed on {total_samples:,} samples")
    
    return metrics


def train_epoch_with_memory_optimization(model, train_loader, optimizer, loss_fn, device, epoch,
                                       memory_optimizer, amp_manager, gradient_accumulator, max_epochs=50, use_event_types=False):
    """
    Enhanced training function combining memory optimization with detailed loss tracking
    Uses standardized memory management from train_natr_optimized.py
    
    Note: device parameter is kept for API compatibility but device operations are handled by memory_optimizer
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Prepare for epoch
    memory_optimizer.before_epoch(epoch)
    
    # Create progress bar with better formatting
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", 
                      mininterval=5.0,    # Update at most every 2 seconds
                      ncols=None,         # Auto-adjust width
                      leave=False)        # Don't leave progress bar after completion
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(progress_bar):
        # Memory optimization before batch
        memory_optimizer.before_batch(batch_idx)
        
        # Move batch to device with optimization
        batch = memory_optimizer.optimize_batch(batch)
        
        # Add event types to batch if using event-aware model
        if use_event_types:
            batch = EventAwareDataProcessor.add_event_types_to_batch(batch)
        
        # Forward pass with AMP if available
        with amp_manager:
            outputs = model(batch)
            predictions = outputs['predictions']
            
            # Get targets and event indicators
            targets = batch['purchased']['package_ids']
            is_purchase = batch.get('is_purchase', torch.zeros_like(targets, dtype=torch.bool))
            has_checkout = batch.get('has_checkout', torch.zeros_like(targets, dtype=torch.bool))
            has_add_to_cart = batch.get('has_add_to_cart', torch.zeros_like(targets, dtype=torch.bool))
            
            # Calculate loss with NaturalPurchaseLoss (hierarchical event weighting)
            if hasattr(loss_fn, 'forward'):
                # Check the forward method's parameters for other loss functions
                import inspect
                sig = inspect.signature(loss_fn.forward)
                params = list(sig.parameters.keys())
                
                if 'has_add_to_cart' in params:
                    loss = loss_fn(predictions, targets, is_purchase, has_checkout, has_add_to_cart)
                elif 'has_checkout' in params:
                    loss = loss_fn(predictions, targets, is_purchase, has_checkout)
                elif 'is_purchase' in params:
                    loss = loss_fn(predictions, targets, is_purchase)
                else:
                    loss = loss_fn(predictions, targets)
            else:
                # Fallback for simpler loss functions
                loss = loss_fn(predictions, targets)
        
        # Backward pass with gradient accumulation and AMP scaling
        original_loss = gradient_accumulator.backward(loss)
        
        # Update metrics
        total_loss += original_loss.item()
        num_batches += 1
        
        # Monitor gradients every 100 batches for user learning analysis
        if batch_idx % 100 == 0 and batch_idx > 0:
            user_grads = []
            other_grads = []
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if 'user_encoder.user_embedding' in name or 'user_transform' in name:
                        user_grads.append(grad_norm)
                    else:
                        other_grads.append(grad_norm)
            
            if user_grads and other_grads:
                avg_user_grad = sum(user_grads) / len(user_grads)
                avg_other_grad = sum(other_grads) / len(other_grads)
                
                # Print gradient monitoring occasionally
                if batch_idx % 300 == 0:
                    ratio = avg_user_grad / avg_other_grad if avg_other_grad > 0 else 0
                    print(f"🔍 Gradient Monitor - User: {avg_user_grad:.6f}, Other: {avg_other_grad:.6f}, Ratio: {ratio:.3f}")
        
        # Step optimizer if accumulation is complete
        if gradient_accumulator.should_step():
            # Clip gradients BEFORE optimizer step to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Step optimizer
            gradient_accumulator.step(optimizer)
            
            # Check and fix NaN in user embeddings after update
            with torch.no_grad():
                user_emb_weight = model.user_encoder.user_embedding.weight
                nan_mask = torch.isnan(user_emb_weight).any(dim=1)
                if nan_mask.any():
                    num_nan = nan_mask.sum().item()
                    if batch_idx % 100 == 0:  # Log every 100 batches
                        print(f"  Fixing {num_nan} NaN user embeddings after optimizer step")
                    # Replace NaN embeddings with small random values
                    user_emb_weight[nan_mask] = torch.randn(num_nan, user_emb_weight.shape[1], device=user_emb_weight.device) * 0.01
                    # Ensure padding stays zero
                    user_emb_weight[0] = 0
        
        # Memory optimization after batch
        memory_optimizer.after_batch(batch_idx)
        
        # Update progress bar with comprehensive info
        postfix = {
            'loss': total_loss / num_batches,
            'lr': optimizer.param_groups[0]['lr']
        }
        
        # Add memory info if tracking is enabled
        if memory_optimizer.track_memory and memory_optimizer.peak_memory > 0:
            postfix['mem_mb'] = f"{memory_optimizer.peak_memory / (1024*1024):.0f}"
            
        progress_bar.set_postfix(postfix)
    
    # Handle remaining gradients if any
    if not gradient_accumulator.should_step():
        gradient_accumulator.step(optimizer)
    
    # Clean up after epoch
    memory_optimizer.after_epoch(epoch)
    
    return total_loss / num_batches


def main(performance_mode="balanced", dataset="13months", event_data_path=None, num_negatives=500, purchase_boost=10.0, model_type="standard", args=None):
    """Main training function with enhanced memory management and metrics
    
    Args:
        performance_mode (str): One of 'fastest', 'balanced', 'accurate'
        dataset (str): Dataset to use - '13months' or '2months'
        event_data_path (str): Optional explicit path to event data file
        num_negatives (int): Number of negative samples per positive
        purchase_boost (float): Weight boost for purchase events
    """
    # Detect device and create memory configuration
    device, device_type = detect_device()
    memory_config = create_memory_config(device_type, performance_mode)
    
    # Create memory optimizer
    memory_optimizer = MemoryOptimizer(device, memory_config)
    
    # Log device and memory config
    logger.info(f"Using device: {device} ({device_type})")
    logger.info(f"Performance mode: {performance_mode}")
    
    # Clear old caches to ensure fresh embeddings
    package_features_cache = "data/cache/package_features.pkl"
    if os.path.exists(package_features_cache):
        logger.info(f"Removing package features cache to use latest embeddings")
        os.remove(package_features_cache)
        
    # Clear dataset caches
    for cache_file in glob.glob("data/cache/dataset_*.pkl"):
        logger.info(f"Removing dataset cache: {cache_file}")
        os.remove(cache_file)
    
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
    
    logger.info(f"Using event data: {event_data_path}")
    
    # Determine batch size based on memory configuration
    base_batch_size = get_optimal_batch_size(
        device_type, memory_config, model_size_mb=100, tensor_size_mb=5
    )
    
    # Adjust for performance mode
    if performance_mode == "fastest":
        num_epochs = 20
        learning_rate = 0.002
        base_batch_size = min(base_batch_size * 2, 512)  # Larger batches for speed
    elif performance_mode == "accurate":
        num_epochs = 100
        learning_rate = 0.0005
        base_batch_size = max(base_batch_size // 2, 32)  # Smaller batches for accuracy
    else:  # "balanced"
        num_epochs = 50
        learning_rate = 0.001  # Same as standard NATR for better convergence
    
    # Initialize processors
    print("\nInitializing data processors...")
    package_processor = PackageProcessor(
        feed_data_path=package_data_path,  # Updated parameter name
        cache_dir='data/cache',
        load_coordinates=True,
        load_embeddings=True,
        api_key=os.environ.get("OPENAI_API_KEY"),
        embedding_model='text-embedding-3-small',
        use_reduced_embeddings=True if device_type != 'cpu' else False
    )
    
    session_processor = SessionProcessor(
        event_data_path=event_data_path,  # Updated parameter name
        cache_dir='data/cache',
        min_interactions=5,
        max_sessions_per_user=20,
        max_samples_per_user=10
    )
    
    # Display training parameters
    print(f"\nTraining parameters:")
    print(f"  Dataset: {dataset} ({os.path.basename(event_data_path)})")
    print(f"  Device: {device} ({device_type})")
    print(f"  Performance mode: {performance_mode}")
    print(f"  Batch size: {base_batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Gradient accumulation steps: {memory_config.gradient_accumulation_steps}")
    print(f"  Number of epochs: {num_epochs}")
    print(f"  Memory optimization: {memory_config.enable_memory_tracking}")
    
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
    
    # Identify event types in samples (checkout, add-to-cart)
    from utils.training_utils import identify_event_types
    samples = identify_event_types(samples, session_processor.event_to_idx)
    
    print("\n6. Applying data filters...")
    
    # Apply quality filters
    quality_samples = filter_by_min_session_length(samples, min_session_length=2)  # Aligned with train_natr.py
    
    # Use consistent min_frequency across all datasets for quality
    min_frequency = 5  # Standard threshold for package quality
    print(f"Using min_frequency={min_frequency} for consistent package quality")
        
    filtered_samples, valid_packages = filter_items_by_frequency(quality_samples, min_frequency=min_frequency)
    
    # Validate purchase events
    purchase_count = sum(1 for s in filtered_samples if s.get('is_purchase', False))
    if purchase_count == 0:
        raise ValueError("No purchase events found in the filtered samples. Cannot train the model.")
    
    print(f"Purchase events: {purchase_count} ({purchase_count / len(filtered_samples) * 100:.2f}% of all events)")
    
    # Time-based split
    train_samples, test_samples, split_date = time_based_split_year(filtered_samples, train_ratio=0.91)
    
    # IMPORTANT: Update user mappings to include ALL users from both train and test sets
    print("\n6b. Updating user mappings to include test users...")
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
    
    # Debug: print purchase counts after split
    train_purchases = sum(1 for s in train_samples if s.get('is_purchase', False))
    test_purchases = sum(1 for s in test_samples if s.get('is_purchase', False))
    print(f"\nPurchase distribution after time-based split:")
    print(f"  Train: {train_purchases} purchases out of {len(train_samples)} samples")
    print(f"  Test: {test_purchases} purchases out of {len(test_samples)} samples")
    print(f"  Total: {train_purchases + test_purchases} purchases")
    
    # Ensure test set has enough purchase samples
    purchase_samples = [s for s in test_samples if s.get('is_purchase', False)]
    if len(purchase_samples) < 50:
        print(f"Warning: Only {len(purchase_samples)} purchases in test set, adding more...")
        extra_purchases = [s for s in train_samples if s.get('is_purchase', False)][:50-len(purchase_samples)]
        test_samples.extend(extra_purchases)
        train_samples = [s for s in train_samples if s not in extra_purchases]
        print(f"Added {len(extra_purchases)} more purchase samples to test set")
    
    # Analyze distributions
    analyze_data_distribution(train_samples, "Train")
    analyze_data_distribution(test_samples, "Test")
    
    # Create datasets and dataloaders
    print("\n7. Creating datasets...")
    train_loader, test_loader = create_dataloaders(
        train_samples, test_samples, package_processor, session_processor,
        batch_size=base_batch_size, num_workers=0 if device_type == 'mps' else 2,
        use_weighted_sampling=True
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset):,} samples ({len(train_loader):,} batches)")
    print(f"  Test: {len(test_loader.dataset):,} samples ({len(test_loader):,} batches)")
    
    # Create model
    print("\n8. Creating model...")
    
    # Get dimensions from mappings (add 1 because indices start at 1, not 0)
    # IMPORTANT: These dimensions must include ALL users/packages from both train and test sets
    num_users = max(session_processor.user_to_idx.values()) + 1
    num_packages = max(session_processor.package_to_idx.values()) + 1
    num_countries = max(package_processor.country_to_idx.values()) + 1
    num_categories = max(package_processor.category_to_idx.values()) + 1
    num_themes = max(package_processor.theme_to_idx.values()) + 1
    
    print(f"\nDimension check (after including test users):")
    print(f"  num_users: {num_users} (max idx: {max(session_processor.user_to_idx.values())})")
    print(f"  num_packages: {num_packages} (max idx: {max(session_processor.package_to_idx.values())})")
    
    # Debug: Check some user mappings
    sample_users = list(session_processor.user_to_idx.items())[:5]
    print(f"\nSample user mappings:")
    for user_str, user_idx in sample_users:
        print(f"  '{user_str}' -> {user_idx}")
    
    # Check if any valid user is mapped to 0 (padding index)
    reverse_mapping = {v: k for k, v in session_processor.user_to_idx.items()}
    if 0 in reverse_mapping:
        print(f"\nWARNING: User '{reverse_mapping[0]}' is mapped to padding index 0!")
    
    # Detect embedding dimension
    package_tensors = package_processor.prepare_package_tensors()
    actual_embedding_dim = package_tensors['title_embeddings'].shape[1]
    print(f"Detected title embedding dimension: {actual_embedding_dim}")
    
    # Simplified model dimensions for consistent training
    # Use moderate size to balance capacity and training stability
    hidden_dim = 512  # Increased from 256 for more capacity
    embedding_dim = 256
    dropout = 0.1  # Reduced dropout to help learning
    
    # Create model configuration with correct parameter names
    config = NATRConfig(
        num_users=num_users,
        num_packages=num_packages,
        num_countries=num_countries,
        num_categories=num_categories,
        num_themes=num_themes,
        title_embedding_dim=actual_embedding_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        user_embedding_dim=128,  # Explicitly set user embedding dimension
        dropout=dropout
    )
    
    # Create model based on selected type
    print(f"\n8. Creating {args.model_type} model...")
    use_event_types = False
    if args.model_type == 'event-aware':
        model = NATRWithEventTypes(config)
        use_event_types = True
        print("Using event-aware NATR with full event-type integration")
    elif args.model_type == 'event-additive':
        model = NATRWithAdditiveEventTypes(config)
        use_event_types = True
        print("Using additive event-type NATR (simple bias addition)")
    else:
        model = NATR(config)
        print("Using standard NATR model")
    
    model = model.to(device)
    
    # Get model size for memory calculations
    model_size_mb = get_model_size(model)
    print(f"Model size: {model_size_mb} MB")
    
    # Create custom optimizer with different learning rates for user components
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
    
    # Create optimizer with balanced learning rates (same as standard NATR)
    user_lr = learning_rate * 2.0  # 2x learning rate for user embeddings
    user_transform_lr = learning_rate * 10.0  # 10x learning rate for user transform
    
    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': learning_rate, 'weight_decay': 1e-4},  # Increased regularization
        {'params': user_params, 'lr': user_lr, 'weight_decay': 1e-5},  # Moderate regularization for users
        {'params': user_transform_params, 'lr': user_transform_lr, 'weight_decay': 1e-5}  # Moderate regularization
    ])
    
    print(f"  Base learning rate: {learning_rate}")
    print(f"  User embedding learning rate: {user_lr}")
    print(f"  User transform learning rate: {user_transform_lr}")
    print(f"  User embedding params: {sum(p.numel() for p in user_params)}")
    print(f"  User transform params: {sum(p.numel() for p in user_transform_params)}")
    print(f"  Other params: {sum(p.numel() for p in other_params)}")
    
    # Add learning rate scheduler to prevent model collapse
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )
    
    # Create AMP manager and gradient accumulator
    amp_manager = AMPManager(device, memory_config)
    # Use configured gradient accumulation steps (not forced to 1)
    gradient_accumulator = GradientAccumulator(
        steps=memory_config.gradient_accumulation_steps,
        amp_manager=amp_manager
    )
    
    # Calculate package frequencies from training data with event-based weighting
    print("\nCalculating package frequencies for proper log-uniform sampling...")
    package_frequencies = torch.zeros(config.num_packages)
    
    # Calculate separate frequencies for different purposes
    purchase_frequencies = torch.zeros(config.num_packages)  # For popularity modeling
    
    # Debug counters
    purchase_count_debug = 0
    out_of_range_count = 0
    
    # Event weights for popularity calculation (used for negative sampling)
    # More balanced weights to avoid over-biasing the sampling distribution
    event_weights = {
        'purchase': 5.0,     # Purchases are strongest signal
        'checkout': 2.0,     # Checkout shows strong intent  
        'add_to_cart': 1.0,  # Add to cart shows interest
        'view': 0.5          # Views are weakest signal
    }
    
    for sample in train_samples:
        # Determine event weight based on sample type
        if sample.get('is_purchase', False):
            weight = event_weights['purchase']
        elif sample.get('has_checkout_inclusive', False):
            weight = event_weights['checkout']
        elif sample.get('has_add_to_cart_inclusive', False):
            weight = event_weights['add_to_cart']
        else:
            weight = event_weights['view']
        
        # Count packages in short-term and long-term with appropriate weight (for negative sampling)
        for pkg_id in sample.get('short_term_packages', []):
            # Map the package ID to index (same as done in TravelPackageDataset)
            pkg_id_str = str(pkg_id)
            pkg_idx = session_processor.package_to_idx.get(pkg_id_str, -1)
            if pkg_idx >= 0 and pkg_idx < config.num_packages:
                package_frequencies[pkg_idx] += weight
        for pkg_id in sample.get('long_term_packages', []):
            # Map the package ID to index (same as done in TravelPackageDataset)
            pkg_id_str = str(pkg_id)
            pkg_idx = session_processor.package_to_idx.get(pkg_id_str, -1)
            if pkg_idx >= 0 and pkg_idx < config.num_packages:
                package_frequencies[pkg_idx] += weight * 0.5  # Long-term interactions have less weight
        
        # Count only actual purchases for popularity modeling
        if sample.get('is_purchase', False):
            purchase_count_debug += 1
            purchased_pkg = sample.get('purchased_package')
            if purchased_pkg is not None:
                # Map the package ID to index (same as done in TravelPackageDataset)
                purchased_pkg_str = str(purchased_pkg)
                purchased_pkg_idx = session_processor.package_to_idx.get(purchased_pkg_str, -1)
                
                if purchased_pkg_idx >= 0 and purchased_pkg_idx < config.num_packages:
                    purchase_frequencies[purchased_pkg_idx] += 1
                else:
                    out_of_range_count += 1
                    if out_of_range_count <= 5:  # Print first 5 out of range
                        print(f"WARNING: purchased_pkg {purchased_pkg} (mapped to {purchased_pkg_idx}) out of range [0, {config.num_packages})")
    
    # Use standard NaturalPurchaseLoss instead of broken sampled softmax
    # The LogUniformSampledSoftmaxLoss is causing model collapse
    loss_fn = NaturalPurchaseLoss(
        purchase_boost=purchase_boost  # Same purchase boost as before
    )
    
    print("\n=== Using Standard NaturalPurchaseLoss (Fixed) ===")
    print(f"Purchase weight: {loss_fn.purchase_weight}x")
    print(f"Computing loss over ALL {config.num_packages} items (no negative sampling)")
    print("This ensures stable training without model collapse")
    
    # Initialize popularity features in the model based solely on purchase frequencies
    print(f"\n=== Initializing Model with Purchase-Based Popularity ===")
    print(f"Total unique items purchased: {(purchase_frequencies > 0).sum().item()}")
    print(f"Total purchase events counted: {purchase_frequencies.sum().item()}")
    print(f"Debug: purchase_count_debug = {purchase_count_debug}")
    print(f"Debug: out_of_range_count = {out_of_range_count}")
    print(f"Expected purchase events: {len([s for s in train_samples if s.get('is_purchase', False)])}")
    print(f"Model type: {type(model).__name__}")
    print(f"Args model type: {args.model_type}")
    
    if hasattr(model, 'initialize_popularity'):
        model.initialize_popularity(purchase_frequencies.to(device))
        print("✓ Popularity bias and embeddings initialized based on actual purchase frequencies")
    else:
        print("✗ Model does not have initialize_popularity method")
        
    # Verify popularity bias was set
    if hasattr(model, 'popularity_bias'):
        pop_bias_stats = model.popularity_bias.data
        print(f"Popularity bias range: [{pop_bias_stats.min().item():.4f}, {pop_bias_stats.max().item():.4f}]")
        print(f"Non-zero popularity items: {(pop_bias_stats != 0).sum().item()}")
    
    # Set up checkpointing
    os.makedirs('checkpoints/natr', exist_ok=True)
    
    # Training loop
    print("\n9. Starting training...")
    best_purchase_recall = 0.0
    best_purchase_mrr = 0.0  # Track best purchase MRR for tiebreaking
    patience = 10 if performance_mode != "fastest" else 5
    epochs_without_improvement = 0
    
    # Print initial memory stats
    if memory_optimizer.track_memory:
        print_memory_stats(device, "Training Start")
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # Training
        train_loss = train_epoch_with_memory_optimization(
            model, train_loader, optimizer, loss_fn, device, epoch,
            memory_optimizer, amp_manager, gradient_accumulator, max_epochs=num_epochs, 
            use_event_types=use_event_types
        )
        
        print(f"Training loss: {train_loss:.4f}")
        
        # Evaluation with comprehensive metrics
        metrics = evaluate_with_unified_metrics(
            model, test_loader, device, memory_optimizer, k_values=[5, 10, 20], 
            use_event_types=use_event_types
        )
        
        # Print detailed metrics
        print("\nEvaluation Results:")
        print(f"  Purchase Recall@10: {metrics.get('purchase_recall@10', 0)*100:.2f}%")
        print(f"  Purchase Recall@20: {metrics.get('purchase_recall@20', 0)*100:.2f}%")
        print(f"  Checkout Recall@20: {metrics.get('checkout_recall@20', 0)*100:.2f}%")
        print(f"  Add-to-Cart Recall@20: {metrics.get('add_to_cart_recall@20', 0)*100:.2f}%")
        print(f"  Item Coverage@20: {metrics.get('item_coverage@20', 0)*100:.2f}%")
        print(f"  Purchase MRR: {metrics.get('purchase_mrr', 0):.4f}")
        
        # Debug: Check diversity of predictions
        if epoch == 0 or epoch == 4 or epoch == 9:
            print(f"\n[DEBUG Epoch {epoch}] Analyzing prediction diversity...")
            # Get a sample batch from test loader
            for batch in test_loader:
                batch = memory_optimizer.optimize_batch(batch)
                with torch.no_grad():
                    # Debug user ID issues
                    user_ids = batch.get('user_ids', None)
                    if user_ids is not None:
                        max_user_id = user_ids.max().item()
                        if max_user_id >= num_users:
                            print(f"ERROR: User ID {max_user_id} >= num_embeddings {num_users}")
                            print(f"User IDs sample: {user_ids[:10].tolist()}")
                            # Skip this debug batch if user IDs are invalid
                            break
                    
                    outputs = model(batch)
                    predictions = outputs['predictions']
                    # Get top-20 predictions for each user
                    top_k_items = torch.topk(predictions, k=20, dim=-1).indices
                    # Count unique items across all predictions
                    unique_items = torch.unique(top_k_items)
                    print(f"  Unique items in top-20 predictions across batch: {len(unique_items)}")
                    # Show most common predictions
                    flat_predictions = top_k_items.flatten()
                    item_counts = torch.bincount(flat_predictions, minlength=config.num_packages)
                    top_10_items = torch.topk(item_counts, k=10)
                    print(f"  Top 10 most predicted items (ID, count): {[(idx.item(), cnt.item()) for idx, cnt in zip(top_10_items.indices, top_10_items.values)]}")
                    
                    # Check user representation diversity
                    user_repr = outputs.get('user_representation')
                    if user_repr is not None:
                        # Check if user representations are different
                        user_std = user_repr.std(dim=0).mean().item()
                        print(f"  User representation std: {user_std:.4f}")
                        # Check cosine similarity between users
                        user_norm = user_repr / (user_repr.norm(dim=1, keepdim=True) + 1e-8)
                        user_sim = torch.mm(user_norm[:10], user_norm[:10].t())
                        avg_sim = (user_sim.sum() - user_sim.trace()) / (user_sim.shape[0] * (user_sim.shape[0] - 1))
                        print(f"  Average user similarity (first 10): {avg_sim:.4f}")
                    
                    # Check prediction variance
                    pred_std = predictions.std(dim=0).mean().item()
                    print(f"  Prediction std across users: {pred_std:.4f}")
                    break
        
        # Track best metrics - using Purchase Recall@20 as primary metric
        current_purchase_recall_20 = metrics.get('purchase_recall@20', 0)
        current_purchase_mrr = metrics.get('purchase_mrr', 0)
        
        # Update learning rate scheduler based on performance
        scheduler.step(current_purchase_recall_20)
        
        # Combined improvement metric: prioritize Purchase Recall@20, use Purchase MRR as tiebreaker
        # Use epsilon for floating-point comparison to avoid precision issues
        epsilon = 1e-6  # tolerance for considering recalls as similar
        improvement = (current_purchase_recall_20 > best_purchase_recall + epsilon) or \
                     (abs(current_purchase_recall_20 - best_purchase_recall) <= epsilon and 
                      current_purchase_mrr > best_purchase_mrr + epsilon)
        
        if improvement:
            best_purchase_recall = current_purchase_recall_20
            best_purchase_mrr = current_purchase_mrr
            epochs_without_improvement = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.__dict__,
                'metrics': metrics,
                'train_loss': train_loss
            }, 'checkpoints/natr/best_model_neg_sampling.pth')
            print(f"✓ New best model saved! Purchase Recall@20: {best_purchase_recall*100:.2f}%, Purchase MRR: {best_purchase_mrr:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.__dict__,
                'metrics': metrics,
                'train_loss': train_loss
            }, f'checkpoints/natr/model_epoch_{epoch+1}.pth')
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement")
            break
    
    # Final summary
    print("\n10. Training complete!")
    print(f"Best Purchase Recall@20: {best_purchase_recall*100:.2f}%")
    print(f"Best Purchase MRR: {best_purchase_mrr:.4f}")
    
    # Save model info
    model_info = {
        'config': config.__dict__,
        'valid_packages': list(valid_packages),
        'best_purchase_recall_20': best_purchase_recall,
        'best_purchase_mrr': best_purchase_mrr,
        'checkpoint_path': 'checkpoints/natr/best_model_neg_sampling.pth',
        'package_data_path': package_data_path,
        'event_data_path': event_data_path,
        'performance_mode': performance_mode,
        'device_type': device_type,
        'training_date': datetime.now().strftime("%Y-%m-%d"),
        'package_count': len(valid_packages),
        'user_count': num_users,
        'model_size_mb': model_size_mb,
        'split_date': split_date,  # Save the train/test split date
        'train_ratio': 0.91  # Save the train ratio used
    }
    
    # Save model info to appropriate directory based on dataset
    output_dir = 'output/model_info' if dataset == '13months' else 'output/model_info_2months'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'model_info_neg_sampling.json'), 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("\nFiles saved:")
    print("  - checkpoints/natr/best_model_neg_sampling.pth")
    print("  - checkpoints/natr/model_epoch_X.pth")
    print(f"  - {output_dir}/model_info_neg_sampling.json")
    
    # Final memory stats
    if memory_optimizer.track_memory:
        print_memory_stats(device, "Training Complete")


if __name__ == "__main__":
    import sys
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train NATR model with enhanced memory optimization')
    parser.add_argument('--mode', type=str, default='balanced', 
                        choices=['fastest', 'balanced', 'accurate'],
                        help='Performance mode (default: balanced)')
    parser.add_argument('--dataset', type=str, default='13months',
                        choices=['13months', '2months'],
                        help='Which dataset to use: 13months or 2months (default: 13months)')
    parser.add_argument('--event-data', type=str, default=None,
                        help='Path to event data file (overrides --dataset)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size (default: determined by mode and device)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Override learning rate (default: determined by mode)')
    parser.add_argument('--num-negatives', type=int, default=500,
                        help='Number of negative samples per positive (default: 500)')
    parser.add_argument('--purchase-boost', type=float, default=10.0,
                        help='Weight boost for purchase events (default: 10.0)')
    parser.add_argument('--model-type', type=str, default='standard',
                        choices=['standard', 'event-aware', 'event-additive'],
                        help='Model type: standard, event-aware (full integration), or event-additive (simple) (default: standard)')
    
    # Parse arguments while maintaining backward compatibility
    if len(sys.argv) > 1 and sys.argv[1] in ["fastest", "balanced", "accurate"]:
        # Old style: just the mode as first argument
        performance_mode = sys.argv[1]
        args = parser.parse_args([])
        args.mode = performance_mode
    else:
        # New style: proper argument parsing
        args = parser.parse_args()
    
    # Check for Apple Silicon and optimize automatically
    is_mps = torch.backends.mps.is_available()
    if is_mps:
        print("Apple Silicon (MPS) device detected! Using optimized settings.")
    
    # Override with command-line arguments if provided
    if args.batch_size is not None:
        print(f"Overriding batch size to {args.batch_size}")
    
    if args.learning_rate is not None:
        print(f"Overriding learning rate to {args.learning_rate}")
    
    # Run training
    main(performance_mode=args.mode, dataset=args.dataset, event_data_path=args.event_data, 
         num_negatives=args.num_negatives, purchase_boost=args.purchase_boost, model_type=args.model_type, args=args)
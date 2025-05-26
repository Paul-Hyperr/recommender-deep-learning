"""
Shared training utilities for NATR recommender models.
Contains common functions used across different model variants to reduce code duplication.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import gc
import time
import json
import os
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset, random_split


def filter_by_min_session_length(samples: List[Dict], min_session_length: int = 2) -> List[Dict]:
    """Filter out sessions that are too short for quality training
    
    Args:
        samples: List of training samples
        min_session_length: Minimum number of interactions required in short-term sequence
        
    Returns:
        List of filtered samples
    """
    print(f"\nFiltering sessions by length (min: {min_session_length})...")
    
    # Count samples before
    total_before = len(samples)
    
    # Group samples by user
    user_samples = {}
    for sample in samples:
        user_id = sample.get('user_id')
        if user_id not in user_samples:
            user_samples[user_id] = []
        user_samples[user_id].append(sample)
    
    # Filter by session length while preserving at least one sample per user
    quality_samples = []
    users_with_no_quality_sessions = 0
    
    for user_id, user_session_samples in user_samples.items():
        # First find quality samples for this user
        user_quality_samples = []
        
        for sample in user_session_samples:
            # Count interactions in short-term
            short_term_length = len(sample.get('short_term_packages', []))
            
            # Only keep samples with sufficient interactions
            if short_term_length >= min_session_length:
                user_quality_samples.append(sample)
        
        # If user has quality samples, add them all
        if user_quality_samples:
            quality_samples.extend(user_quality_samples)
        # If user has no quality samples but has purchase samples, keep those
        else:
            # Find purchase samples for this user
            purchase_samples = [s for s in user_session_samples if s.get('is_purchase', False)]
            
            if purchase_samples:
                # Keep all purchase samples even if they're short
                quality_samples.extend(purchase_samples)
            else:
                # Otherwise keep their longest session
                if user_session_samples:
                    best_sample = max(user_session_samples, 
                                     key=lambda s: len(s.get('short_term_packages', [])))
                    quality_samples.append(best_sample)
                    
            users_with_no_quality_sessions += 1
    
    # Count samples after
    total_after = len(quality_samples)
    
    print(f"Samples after session length filtering: {total_after} ({total_after/total_before*100:.1f}%)")
    print(f"Users without quality sessions (fallback to best): {users_with_no_quality_sessions}")
    
    return quality_samples


def filter_items_by_frequency(samples: List[Dict], min_frequency: int = 50) -> Tuple[List[Dict], set]:
    """Filter items that appear less than min_frequency times
    
    For non-purchase samples, a package must have at least min_frequency interactions to be included
    Purchase samples are always preserved regardless of package frequency
    
    Args:
        samples: List of training samples
        min_frequency: Minimum frequency for packages to be included
        
    Returns:
        Tuple of (filtered_samples, valid_packages)
    """
    # Check if we're in test mode and adjust min_frequency
    if os.environ.get("NATR_TEST_MODE") == "1":
        test_min_frequency = int(os.environ.get("NATR_MIN_FREQUENCY", 10))
        if test_min_frequency < min_frequency:
            print(f"\n*** TEST MODE: Using smaller min_frequency: {test_min_frequency} instead of {min_frequency} ***")
            min_frequency = test_min_frequency
    print(f"\nFiltering items by frequency (min: {min_frequency})...")
    
    # First separate purchase and non-purchase samples
    purchase_samples = [s for s in samples if s.get('is_purchase', False)]
    non_purchase_samples = [s for s in samples if not s.get('is_purchase', False)]
    
    # Get all packages that were purchased (we'll keep these regardless of frequency)
    purchased_packages = set()
    for sample in purchase_samples:
        purchased = str(sample.get('purchased_package', ''))
        if purchased:
            purchased_packages.add(purchased)
    
    print(f"Found {len(purchased_packages)} unique purchased packages (will be preserved)")
    
    # Count package occurrences efficiently
    package_counts = Counter()
    
    for sample in samples:
        # Count all packages
        all_packages = []
        all_packages.extend(sample.get('short_term_packages', []))
        all_packages.extend(sample.get('long_term_packages', []))
        all_packages.append(sample.get('purchased_package'))
        
        for pkg in all_packages:
            if pkg:
                package_counts[str(pkg)] += 1
    
    # Find valid packages - include all purchased packages plus frequent non-purchased ones
    valid_packages = {pkg for pkg, count in package_counts.items() if count >= min_frequency or pkg in purchased_packages}
    print(f"Valid packages: {len(valid_packages)} out of {len(package_counts)}")
    
    # All purchase samples are kept
    filtered_samples = list(purchase_samples)  # Create a copy
    
    # For non-purchase samples, filter packages
    for sample in non_purchase_samples:
        # Filter packages in sequences
        sample_copy = sample.copy()
        sample_copy['short_term_packages'] = [pkg for pkg in sample.get('short_term_packages', []) 
                                              if str(pkg) in valid_packages]
        sample_copy['long_term_packages'] = [pkg for pkg in sample.get('long_term_packages', []) 
                                             if str(pkg) in valid_packages]
        
        # Keep sample if it has valid packages
        if sample_copy['short_term_packages'] or sample_copy['long_term_packages']:
            filtered_samples.append(sample_copy)
    
    print(f"Samples after filtering: {len(filtered_samples)} ({len(filtered_samples)/len(samples)*100:.1f}%)")
    print(f"Purchase samples preserved: {len(purchase_samples)} (100%)")
    
    return filtered_samples, valid_packages


def time_based_split_year(samples: List[Dict], train_ratio: float = 0.93) -> Tuple[List[Dict], List[Dict]]:
    """Split samples based on time for a year of data
    
    Args:
        samples: List of training samples
        train_ratio: Ratio of samples to use for training (default: 0.93)
        
    Returns:
        Tuple of (train_samples, test_samples)
    """
    print(f"\nApplying time-based split ({train_ratio*100:.0f}% train, {(1-train_ratio)*100:.0f}% test)...")
    
    # Check if timestamps exist
    if not samples or 'timestamp' not in samples[0]:
        print("Warning: No timestamps found, using random split")
        indices = np.random.permutation(len(samples))
        split_idx = int(len(samples) * train_ratio)
        return [samples[i] for i in indices[:split_idx]], [samples[i] for i in indices[split_idx:]]
    
    # Determine if timestamps are epoch or datetime strings
    # Try to infer timestamp format from first sample
    first_timestamp = samples[0]['timestamp']
    timestamp_is_numeric = isinstance(first_timestamp, (int, float)) or \
                          (isinstance(first_timestamp, str) and first_timestamp.isdigit())
    
    # Define sorting key function based on format
    def get_timestamp(sample):
        ts = sample.get('timestamp', 0)
        if timestamp_is_numeric:
            # Convert string numbers to float if needed
            return float(ts) if isinstance(ts, str) else ts
        else:
            # Keep as is for datetime conversion later
            return ts
    
    # Sort by timestamp
    samples_sorted = sorted(samples, key=get_timestamp)
    
    # Get time range
    first_time = get_timestamp(samples_sorted[0])
    last_time = get_timestamp(samples_sorted[-1])
    
    # Convert to datetime in a format-agnostic way
    try:
        if timestamp_is_numeric:
            # Determine if milliseconds or seconds by magnitude
            # If timestamp is very large (>1e12), it's likely milliseconds
            # This is roughly year 2001 in seconds vs 1970 in milliseconds
            is_milliseconds = first_time > 1e12 if first_time > 0 else last_time > 1e12
            
            if is_milliseconds:
                print("Detected millisecond timestamps")
                first_date = pd.to_datetime(first_time, unit='ms')
                last_date = pd.to_datetime(last_time, unit='ms')
            else:
                print("Detected second timestamps")
                first_date = pd.to_datetime(first_time, unit='s')
                last_date = pd.to_datetime(last_time, unit='s')
        else:
            # Try parsing as datetime string
            first_date = pd.to_datetime(first_time)
            last_date = pd.to_datetime(last_time)
            
        print(f"Data spans from {first_date.date()} to {last_date.date()}")
        
        # Calculate split point
        total_duration = last_date - first_date
        train_duration = total_duration * train_ratio
        split_date = first_date + train_duration
        split_date = split_date.normalize() + pd.Timedelta(days=1)
        
        print(f"Split date: {split_date.date()}")
        
        # Split samples
        train_samples = []
        test_samples = []
        
        for sample in samples_sorted:
            ts = get_timestamp(sample)
            
            # Convert timestamp to datetime consistently
            if timestamp_is_numeric:
                if is_milliseconds:
                    sample_date = pd.to_datetime(ts, unit='ms')
                else:
                    sample_date = pd.to_datetime(ts, unit='s')
            else:
                sample_date = pd.to_datetime(ts)
                
            if sample_date < split_date:
                train_samples.append(sample)
            else:
                test_samples.append(sample)
    
    except Exception as e:
        print(f"Warning: Error processing timestamps ({str(e)}), falling back to ratio-based split")
        # Fallback to ratio-based split on sorted data (still time-ordered)
        split_idx = int(len(samples_sorted) * train_ratio)
        train_samples = samples_sorted[:split_idx]
        test_samples = samples_sorted[split_idx:]
    
    print(f"Train samples: {len(train_samples):,}")
    print(f"Test samples: {len(test_samples):,}")
    
    return train_samples, test_samples


def identify_event_types(samples: List[Dict], event_to_idx: Dict[str, int]) -> List[Dict]:
    """Identify checkout, add-to-cart, and other event types in samples
    
    Args:
        samples: List of training samples
        event_to_idx: Dictionary mapping event names to indices
        
    Returns:
        Updated samples with event type flags
    """
    # Find indices for checkout and add-to-cart events
    checkout_idx = event_to_idx.get('InitiateCheckout', 3)  # Default to 3 if not found
    add_to_cart_idx = event_to_idx.get('AddToCart', 2)      # Default to 2 if not found
    
    print(f"InitiateCheckout event index: {checkout_idx}")
    print(f"AddToCart event index: {add_to_cart_idx}")
    
    # Identify checkout and add-to-cart samples
    checkout_only_samples = []
    add_to_cart_only_samples = []
    
    for sample in samples:
        # Skip purchase samples for this analysis
        if sample.get('is_purchase', False):
            continue
            
        has_checkout = False
        has_add_to_cart = False
        
        # Check short-term events
        short_term_events = sample.get('short_term_events', [])
        for event in short_term_events:
            if event == checkout_idx:
                has_checkout = True
            elif event == add_to_cart_idx:
                has_add_to_cart = True
        
        # Add flags to sample
        sample['has_checkout'] = has_checkout
        sample['has_add_to_cart'] = has_add_to_cart
        
        # Categorize samples
        if has_checkout and not has_add_to_cart:
            checkout_only_samples.append(sample)
        elif has_add_to_cart and not has_checkout:
            add_to_cart_only_samples.append(sample)
    
    # Count purchase samples with checkout or add-to-cart
    purchase_checkout_both = 0
    purchase_add_to_cart_both = 0
    
    for sample in samples:
        if not sample.get('is_purchase', False):
            continue
            
        has_checkout = False
        has_add_to_cart = False
        
        # Check short-term events
        short_term_events = sample.get('short_term_events', [])
        for event in short_term_events:
            if event == checkout_idx:
                has_checkout = True
            elif event == add_to_cart_idx:
                has_add_to_cart = True
        
        # Add flags to sample
        sample['has_checkout'] = has_checkout
        sample['has_add_to_cart'] = has_add_to_cart
        
        # Count purchases with checkout or add-to-cart
        if has_checkout:
            purchase_checkout_both += 1
        if has_add_to_cart:
            purchase_add_to_cart_both += 1
    
    # Print statistics
    print(f"Identified {len(checkout_only_samples)} checkout-only samples")
    print(f"Identified {len(add_to_cart_only_samples)} add-to-cart-only samples")
    print(f"Found {purchase_checkout_both} samples with both purchase and checkout events")
    print(f"Found {purchase_add_to_cart_both} samples with both purchase and add-to-cart events")
    
    # Count samples by type
    purchase_count = sum(1 for s in samples if s.get('is_purchase', False))
    checkout_count = len(checkout_only_samples)
    add_to_cart_count = sum(1 for s in samples if not s.get('is_purchase', False) and s.get('has_add_to_cart', False) and not s.get('has_checkout', False))
    view_only_count = sum(1 for s in samples if not s.get('is_purchase', False) and not s.get('has_checkout', False) and not s.get('has_add_to_cart', False))
    
    print("Original distribution:")
    print(f"  Purchases: {purchase_count}")
    print(f"  Checkout: {checkout_count}")
    print(f"  Add-to-cart: {add_to_cart_count}")
    print(f"  View-only: {view_only_count}")
    
    return samples


def ensure_balanced_test_set(train_samples: List[Dict], test_samples: List[Dict], min_purchases: int = 50) -> Tuple[List[Dict], List[Dict]]:
    """Ensure test set has enough purchase samples for meaningful evaluation
    
    Args:
        train_samples: Training samples
        test_samples: Test samples
        min_purchases: Minimum number of purchase samples required in test set
        
    Returns:
        Updated (train_samples, test_samples)
    """
    purchase_samples = [s for s in test_samples if s.get('is_purchase', False)]
    
    if len(purchase_samples) < min_purchases:
        print(f"Warning: Only {len(purchase_samples)} purchases in test set, adding more...")
        # Find more purchases from train samples
        extra_purchases = [s for s in train_samples if s.get('is_purchase', False)][:min_purchases-len(purchase_samples)]
        test_samples.extend(extra_purchases)
        # Remove these from train samples
        train_samples = [s for s in train_samples if s not in extra_purchases]
        print(f"Added {len(extra_purchases)} more purchase samples to test set")
    
    # Also ensure checkout and add-to-cart samples
    checkout_samples = [s for s in test_samples if not s.get('is_purchase', False) and s.get('has_checkout', False)]
    if len(checkout_samples) < min_purchases:
        print(f"Warning: Only {len(checkout_samples)} checkout samples in test set, adding more...")
        extra_checkouts = [s for s in train_samples 
                           if not s.get('is_purchase', False) and s.get('has_checkout', False)][:min_purchases-len(checkout_samples)]
        test_samples.extend(extra_checkouts)
        train_samples = [s for s in train_samples if s not in extra_checkouts]
        print(f"Added {len(extra_checkouts)} more checkout samples to test set")
    
    add_to_cart_samples = [s for s in test_samples 
                          if not s.get('is_purchase', False) and s.get('has_add_to_cart', False) and not s.get('has_checkout', False)]
    if len(add_to_cart_samples) < min_purchases:
        print(f"Warning: Only {len(add_to_cart_samples)} add-to-cart samples in test set, adding more...")
        extra_carts = [s for s in train_samples 
                      if not s.get('is_purchase', False) and s.get('has_add_to_cart', False) and not s.get('has_checkout', False)][:min_purchases-len(add_to_cart_samples)]
        test_samples.extend(extra_carts)
        train_samples = [s for s in train_samples if s not in extra_carts]
        print(f"Added {len(extra_carts)} more add-to-cart samples to test set")
    
    return train_samples, test_samples


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move batch to device with optimized handling for both CUDA and MPS
    
    Args:
        batch: Batch data (can be tensor, dict, list, or other types)
        device: PyTorch device to move data to
        
    Returns:
        Batch data moved to device
    """
    if isinstance(batch, dict):
        # Process in single pass to reduce Python overhead
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, torch.Tensor):
        is_large_tensor = batch.numel() > 1000
        
        # Device-specific optimizations
        if device.type == 'cuda' and batch.dtype == torch.int64 and \
           is_large_tensor and not batch.requires_grad:
            # Convert CPU int64 to int32 for efficiency on CUDA
            return batch.type(torch.int32).to(device, non_blocking=True)
        elif device.type == 'mps':
            # MPS optimizations
            if batch.dtype == torch.int64 and is_large_tensor and not batch.requires_grad:
                # Convert large integer tensors to int32 for MPS as well
                return batch.type(torch.int32).to(device, non_blocking=True)
            elif batch.dtype == torch.float64:
                # Convert double to float, as MPS works better with float32
                return batch.type(torch.float32).to(device, non_blocking=True)
            else:
                # Default case
                return batch.to(device, non_blocking=True)
        else:
            # CPU or other default handling
            return batch.to(device, non_blocking=True)
    elif isinstance(batch, list) and batch and isinstance(batch[0], torch.Tensor):
        # Handle lists of tensors more efficiently
        return [t.to(device, non_blocking=True) for t in batch]
    else:
        return batch


# clear_memory moved to utils/memory_utils.py for consolidation
from utils.memory_utils import clear_memory


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, 
            loss_fn: Any, device: torch.device, epoch: int,
            accumulation_steps: int = 1, use_amp: bool = False) -> float:
    """Train for one epoch with optimizations and MPS enhancements
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        optimizer: Optimizer for updating weights
        loss_fn: Loss function
        device: Device to train on (cuda, mps, cpu)
        epoch: Current epoch number
        accumulation_steps: Number of batches to accumulate gradients for
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Use mixed precision if specified and available
    # Note: As of PyTorch 2.0+, MPS doesn't support AMP yet
    use_amp = use_amp and device.type == 'cuda' and hasattr(torch.cuda, 'amp')
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # MPS-specific memory management
    is_mps = device.type == 'mps'
    mps_memory_optimization = is_mps
    
    # Create a progress bar that updates less frequently
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", 
                      mininterval=10.0,  # Update at most every 10 seconds
                      miniters=1000)      # Update after at least 1000 iterations
    
    optimizer.zero_grad()
    
    # Track peak memory usage for MPS
    if is_mps and hasattr(torch.mps, 'current_allocated_memory'):
        peak_memory = 0
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        batch = move_batch_to_device(batch, device)
        
        # Mixed precision forward pass
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(batch)
                predictions = outputs['predictions']
                
                # Get targets
                targets = batch['purchased']['package_ids']
                
                # Handle different loss function signatures
                if 'is_purchase' in batch:
                    is_purchase = batch['is_purchase']
                    
                    # Check for checkout and add_to_cart flags
                    if 'has_checkout' in batch and 'has_add_to_cart' in batch:
                        has_checkout = batch['has_checkout']
                        has_add_to_cart = batch['has_add_to_cart']
                        loss = loss_fn(predictions, targets, is_purchase, has_checkout, has_add_to_cart)
                    elif 'has_checkout' in batch:
                        has_checkout = batch['has_checkout'] 
                        loss = loss_fn(predictions, targets, is_purchase, has_checkout)
                    else:
                        loss = loss_fn(predictions, targets, is_purchase)
                else:
                    # Fallback for simpler loss functions
                    loss = loss_fn(predictions, targets)
                
                loss = loss / accumulation_steps  # Scale for accumulation
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
        else:
            # Standard forward pass
            outputs = model(batch)
            predictions = outputs['predictions']
            
            # Get targets
            targets = batch['purchased']['package_ids']
            
            # Handle different loss function signatures
            if 'is_purchase' in batch:
                is_purchase = batch['is_purchase']
                
                # Check for checkout and add_to_cart flags
                if 'has_checkout' in batch and 'has_add_to_cart' in batch:
                    has_checkout = batch['has_checkout']
                    has_add_to_cart = batch['has_add_to_cart']
                    loss = loss_fn(predictions, targets, is_purchase, has_checkout, has_add_to_cart)
                elif 'has_checkout' in batch:
                    has_checkout = batch['has_checkout']
                    loss = loss_fn(predictions, targets, is_purchase, has_checkout)
                else:
                    loss = loss_fn(predictions, targets, is_purchase)
            else:
                # Fallback for simpler loss functions
                loss = loss_fn(predictions, targets)
            
            loss = loss / accumulation_steps  # Scale for accumulation
            
            # Backward pass
            loss.backward()
            
            # MPS memory tracking for debugging
            if is_mps and hasattr(torch.mps, 'current_allocated_memory'):
                current_memory = torch.mps.current_allocated_memory() / (1024 * 1024)  # MB
                peak_memory = max(peak_memory, current_memory)
        
        # Update metrics
        total_loss += loss.item() * accumulation_steps
        num_batches += 1
        
        # Step optimizer after accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            
            # MPS-specific memory management - clear cache periodically
            if mps_memory_optimization and (batch_idx + 1) % (accumulation_steps * 5) == 0:
                torch.mps.empty_cache()
        
        # Update progress bar with more detailed info
        postfix = {
            'loss': total_loss / num_batches,
            'lr': optimizer.param_groups[0]['lr']
        }
        
        # Add memory info for MPS
        if is_mps and hasattr(torch.mps, 'current_allocated_memory'):
            postfix['mem_mb'] = f"{torch.mps.current_allocated_memory() / (1024 * 1024):.0f}"
            
        progress_bar.set_postfix(postfix)
    
    # Handle remaining gradients
    if (batch_idx + 1) % accumulation_steps != 0:
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    
    # Final MPS memory cleanup
    if is_mps:
        torch.mps.empty_cache()
        if hasattr(torch.mps, 'current_allocated_memory'):
            print(f"MPS peak memory usage: {peak_memory:.2f} MB")
    
    return total_loss / num_batches


def calculate_batch_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                          is_purchase: Optional[torch.Tensor] = None,
                          has_checkout: Optional[torch.Tensor] = None,
                          has_add_to_cart: Optional[torch.Tensor] = None,
                          purchase_intent: Optional[torch.Tensor] = None,
                          k_values: List[int] = [5, 10, 20]) -> Dict[str, Any]:
    """Calculate metrics for a single batch efficiently with support for multiple event types
    
    Args:
        predictions: Predicted scores for each package
        targets: Target package indices
        is_purchase: Boolean tensor indicating purchase samples
        has_checkout: Boolean tensor indicating checkout samples
        has_add_to_cart: Boolean tensor indicating add-to-cart samples
        purchase_intent: Combined purchase + checkout intent (can be computed if not provided)
        k_values: List of k values for recall@k metrics
        
    Returns:
        Dictionary of metrics
    """
    batch_size = predictions.size(0)
    max_k = max(k_values)
    
    # Initialize metrics
    metrics = {}
    for k in k_values:
        metrics[f'recall@k_{k}'] = 0.0
        if is_purchase is not None:
            metrics[f'purchase_recall@k_{k}'] = 0.0
        if has_checkout is not None:
            metrics[f'checkout_recall@k_{k}'] = 0.0
        if has_add_to_cart is not None:
            metrics[f'add_to_cart_recall@k_{k}'] = 0.0
        if purchase_intent is not None or (is_purchase is not None and has_checkout is not None):
            metrics[f'intent_recall@k_{k}'] = 0.0
    
    metrics['overall_mrr'] = 0.0
    if is_purchase is not None:
        metrics['purchase_mrr'] = 0.0
    if has_checkout is not None:
        metrics['checkout_mrr'] = 0.0
    if has_add_to_cart is not None:
        metrics['add_to_cart_mrr'] = 0.0
    if purchase_intent is not None or (is_purchase is not None and has_checkout is not None):
        metrics['intent_mrr'] = 0.0
    
    # Compute purchase_intent if not provided but components are
    if purchase_intent is None and is_purchase is not None and has_checkout is not None:
        purchase_intent = is_purchase | has_checkout
    
    # Get top-k predictions efficiently
    _, top_indices = torch.topk(predictions, max_k, dim=1)
    
    # Create a mask for each position in top-k
    target_expanded = targets.unsqueeze(1).expand_as(top_indices)
    correct = (top_indices == target_expanded)
    
    # Calculate metrics
    for k in k_values:
        # Recall@k - did the target appear in top k?
        in_top_k = correct[:, :k].any(dim=1)
        metrics[f'recall@k_{k}'] = in_top_k.float().mean().item()
        
        # Purchase-specific recall
        if is_purchase is not None and is_purchase.sum() > 0:
            purchase_in_top_k = in_top_k & is_purchase
            metrics[f'purchase_recall@k_{k}'] = (
                purchase_in_top_k.float().sum() / is_purchase.sum().float()
            ).item() if is_purchase.sum() > 0 else 0.0
        
        # Checkout-specific recall
        if has_checkout is not None and has_checkout.sum() > 0:
            checkout_in_top_k = in_top_k & has_checkout
            metrics[f'checkout_recall@k_{k}'] = (
                checkout_in_top_k.float().sum() / has_checkout.sum().float()
            ).item() if has_checkout.sum() > 0 else 0.0
        
        # Add-to-cart-specific recall
        if has_add_to_cart is not None and has_add_to_cart.sum() > 0:
            add_to_cart_in_top_k = in_top_k & has_add_to_cart
            metrics[f'add_to_cart_recall@k_{k}'] = (
                add_to_cart_in_top_k.float().sum() / has_add_to_cart.sum().float()
            ).item() if has_add_to_cart.sum() > 0 else 0.0
            
        # Intent (purchase + checkout) recall
        if purchase_intent is not None and purchase_intent.sum() > 0:
            intent_in_top_k = in_top_k & purchase_intent
            metrics[f'intent_recall@k_{k}'] = (
                intent_in_top_k.float().sum() / purchase_intent.sum().float()
            ).item() if purchase_intent.sum() > 0 else 0.0
    
    # Calculate MRR (Mean Reciprocal Rank)
    ranks = torch.zeros_like(targets, dtype=torch.float)
    for i in range(batch_size):
        rank_positions = (top_indices[i] == targets[i]).nonzero(as_tuple=True)[0]
        if len(rank_positions) > 0:
            # +1 because ranks start from 1
            ranks[i] = 1.0 / (rank_positions[0].item() + 1)
    
    metrics['overall_mrr'] = ranks.mean().item()
    
    # Calculate MRR for purchases
    if is_purchase is not None:
        metrics['purchase_mrr'] = (
            ranks[is_purchase].mean().item() if is_purchase.sum() > 0 else 0.0
        )
    
    # Calculate MRR for checkouts
    if has_checkout is not None:
        metrics['checkout_mrr'] = (
            ranks[has_checkout].mean().item() if has_checkout.sum() > 0 else 0.0
        )
    
    # Calculate MRR for add-to-cart
    if has_add_to_cart is not None:
        metrics['add_to_cart_mrr'] = (
            ranks[has_add_to_cart].mean().item() if has_add_to_cart.sum() > 0 else 0.0
        )
    
    # Calculate MRR for combined intent (purchase + checkout)
    if purchase_intent is not None:
        metrics['intent_mrr'] = (
            ranks[purchase_intent].mean().item() if purchase_intent.sum() > 0 else 0.0
        )
    
    return metrics


def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device, 
           k_values: List[int] = [5, 10, 20]) -> Dict[str, Any]:
    """Evaluate model with memory optimization and support for multiple event types
    
    Args:
        model: The neural network model
        test_loader: DataLoader for test data
        device: Device to evaluate on (cuda, mps, cpu)
        k_values: List of k values for recall@k metrics
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    # Initialize metrics accumulators to avoid storing all predictions
    metrics_acc = defaultdict(float)
    total_samples = 0
    purchase_samples = 0
    checkout_samples = 0
    add_to_cart_samples = 0
    intent_samples = 0  # Combined purchase + checkout
    
    # MPS-specific optimization
    is_mps = device.type == 'mps'
    
    with torch.no_grad():
        # Handle MPS garbage collection
        if is_mps:
            torch.mps.empty_cache()
        
        # Use smaller batches for evaluation on MPS to avoid OOM
        progress_bar = tqdm(test_loader, desc="Evaluating",
                           mininterval=10.0,  # Update at most every 10 seconds
                           miniters=1000)     # Update after at least 1000 iterations
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = move_batch_to_device(batch, device)
            
            # Forward pass
            outputs = model(batch)
            predictions = outputs['predictions']
            targets = batch['purchased']['package_ids']
            
            # Get event type indicators if present
            is_purchase = batch.get('is_purchase', None)
            has_checkout = batch.get('has_checkout', None)
            has_add_to_cart = batch.get('has_add_to_cart', None)
            
            # Create combined purchase+checkout intent signal if we have both
            purchase_intent = None
            if is_purchase is not None and has_checkout is not None:
                purchase_intent = is_purchase | has_checkout
            
            # Calculate metrics per batch to save memory
            batch_metrics = calculate_batch_metrics(
                predictions, targets, is_purchase, has_checkout, 
                has_add_to_cart, purchase_intent, k_values
            )
            
            # Accumulate metrics
            batch_size = targets.size(0)
            batch_purchases = is_purchase.sum().item() if is_purchase is not None else 0
            batch_checkouts = has_checkout.sum().item() if has_checkout is not None else 0
            batch_add_to_carts = has_add_to_cart.sum().item() if has_add_to_cart is not None else 0
            batch_intents = purchase_intent.sum().item() if purchase_intent is not None else 0
            
            # Update counts
            total_samples += batch_size
            purchase_samples += batch_purchases
            checkout_samples += batch_checkouts
            add_to_cart_samples += batch_add_to_carts
            intent_samples += batch_intents
            
            # Accumulate all metrics by multiplying with count
            for k in k_values:
                # Overall recall
                metrics_acc[f'recall@k_{k}'] += batch_metrics[f'recall@k_{k}'] * batch_size
                
                # Purchase recall
                if batch_purchases > 0:
                    metrics_acc[f'purchase_recall@k_{k}'] += batch_metrics[f'purchase_recall@k_{k}'] * batch_purchases
                
                # Checkout recall
                if batch_checkouts > 0 and f'checkout_recall@k_{k}' in batch_metrics:
                    metrics_acc[f'checkout_recall@k_{k}'] += batch_metrics[f'checkout_recall@k_{k}'] * batch_checkouts
                
                # Add-to-cart recall
                if batch_add_to_carts > 0 and f'add_to_cart_recall@k_{k}' in batch_metrics:
                    metrics_acc[f'add_to_cart_recall@k_{k}'] += batch_metrics[f'add_to_cart_recall@k_{k}'] * batch_add_to_carts
                
                # Intent recall
                if batch_intents > 0 and f'intent_recall@k_{k}' in batch_metrics:
                    metrics_acc[f'intent_recall@k_{k}'] += batch_metrics[f'intent_recall@k_{k}'] * batch_intents
            
            # Accumulate MRR metrics
            metrics_acc['overall_mrr'] += batch_metrics['overall_mrr'] * batch_size
            
            if batch_purchases > 0:
                metrics_acc['purchase_mrr'] += batch_metrics['purchase_mrr'] * batch_purchases
            
            if batch_checkouts > 0 and 'checkout_mrr' in batch_metrics:
                metrics_acc['checkout_mrr'] += batch_metrics['checkout_mrr'] * batch_checkouts
            
            if batch_add_to_carts > 0 and 'add_to_cart_mrr' in batch_metrics:
                metrics_acc['add_to_cart_mrr'] += batch_metrics['add_to_cart_mrr'] * batch_add_to_carts
            
            if batch_intents > 0 and 'intent_mrr' in batch_metrics:
                metrics_acc['intent_mrr'] += batch_metrics['intent_mrr'] * batch_intents
            
            # Periodic memory cleanup for MPS
            if is_mps and (batch_idx + 1) % 10 == 0:
                torch.mps.empty_cache()
                
                # Show memory usage in progress bar if available
                if hasattr(torch.mps, 'current_allocated_memory'):
                    mem_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
                    progress_bar.set_postfix({'memory_mb': f"{mem_mb:.0f}"})
    
    # Final cleanup for MPS
    if is_mps:
        torch.mps.empty_cache()
    
    # Normalize accumulated metrics
    metrics = {
        'recall@k': {k: metrics_acc[f'recall@k_{k}'] / total_samples for k in k_values},
        'overall_mrr': metrics_acc['overall_mrr'] / total_samples,
        'total_count': total_samples,
    }
    
    # Add purchase metrics if we have purchase samples
    if purchase_samples > 0:
        metrics['purchase_recall@k'] = {k: metrics_acc[f'purchase_recall@k_{k}'] / purchase_samples for k in k_values}
        metrics['purchase_mrr'] = metrics_acc['purchase_mrr'] / purchase_samples
        metrics['purchase_count'] = purchase_samples
    
    # Add checkout metrics if we have checkout samples
    if checkout_samples > 0:
        metrics['checkout_recall@k'] = {k: metrics_acc[f'checkout_recall@k_{k}'] / checkout_samples 
                                      for k in k_values if f'checkout_recall@k_{k}' in metrics_acc}
        if 'checkout_mrr' in metrics_acc:
            metrics['checkout_mrr'] = metrics_acc['checkout_mrr'] / checkout_samples
        metrics['checkout_count'] = checkout_samples
    
    # Add add-to-cart metrics if we have add-to-cart samples
    if add_to_cart_samples > 0:
        metrics['add_to_cart_recall@k'] = {k: metrics_acc[f'add_to_cart_recall@k_{k}'] / add_to_cart_samples 
                                         for k in k_values if f'add_to_cart_recall@k_{k}' in metrics_acc}
        if 'add_to_cart_mrr' in metrics_acc:
            metrics['add_to_cart_mrr'] = metrics_acc['add_to_cart_mrr'] / add_to_cart_samples
        metrics['add_to_cart_count'] = add_to_cart_samples
    
    # Add intent metrics if we have intent samples
    if intent_samples > 0:
        metrics['intent_recall@k'] = {k: metrics_acc[f'intent_recall@k_{k}'] / intent_samples 
                                    for k in k_values if f'intent_recall@k_{k}' in metrics_acc}
        if 'intent_mrr' in metrics_acc:
            metrics['intent_mrr'] = metrics_acc['intent_mrr'] / intent_samples
        metrics['intent_count'] = intent_samples
    
    return metrics


def limit_samples_for_testing(samples: List[Dict]) -> List[Dict]:
    """Limit the number of samples for testing if NATR_TEST_MODE is set
    
    Args:
        samples: Original list of samples
        
    Returns:
        Limited list of samples for testing
    """
    # Check if we're in test mode
    if os.environ.get("NATR_TEST_MODE") == "1":
        # Get max samples from environment or use default small test size
        max_samples = int(os.environ.get("NATR_MAX_SAMPLES", 10000))
        
        if len(samples) > max_samples:
            # Keep all purchase samples to ensure we have enough for testing
            purchase_samples = [s for s in samples if s.get("is_purchase", False)]
            non_purchase_samples = [s for s in samples if not s.get("is_purchase", False)]
            
            # Ensure we have at least 100 purchase samples or all available
            min_purchases = min(100, len(purchase_samples))
            chosen_purchases = purchase_samples[:min_purchases]
            
            # Calculate how many non-purchases to keep
            remaining_slots = max_samples - len(chosen_purchases)
            if remaining_slots > 0 and len(non_purchase_samples) > 0:
                # Pick a random subset of non-purchases
                import random
                random.seed(42)  # For reproducibility
                chosen_non_purchases = random.sample(non_purchase_samples, 
                                                   min(remaining_slots, len(non_purchase_samples)))
            else:
                chosen_non_purchases = []
            
            # Combine and shuffle
            limited_samples = chosen_purchases + chosen_non_purchases
            random.shuffle(limited_samples)
            
            print(f"\n*** TEST MODE: Limiting dataset from {len(samples)} to {len(limited_samples)} samples ***")
            print(f"*** TEST MODE: {len(chosen_purchases)} purchase samples, {len(chosen_non_purchases)} non-purchase samples ***")
            
            return limited_samples
    
    # If not in test mode or samples is already small enough, return as is
    return samples


def create_dataloaders(train_samples: List[Dict], test_samples: List[Dict],
                      package_processor: Any, session_processor: Any,
                      batch_size: int = 32, num_workers: int = 0,
                      use_weighted_sampling: bool = True,
                      train_dataset: Optional[Any] = None,
                      test_dataset: Optional[Any] = None) -> Tuple[DataLoader, DataLoader]:
    """Create optimized dataloaders for training and testing with enhanced sampling
    
    Args:
        train_samples: List of training samples
        test_samples: List of test samples
        package_processor: Package processor instance
        session_processor: Session processor instance
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        use_weighted_sampling: Whether to use weighted sampling for imbalanced data
        train_dataset: Optional custom training dataset (to override default TravelPackageDataset)
        test_dataset: Optional custom test dataset (to override default TravelPackageDataset)
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Get mappings for when we need to create default datasets
    user_to_idx = session_processor.get_idx_mappings()['user_to_idx']
    package_to_idx = session_processor.get_idx_mappings()['package_to_idx']
    event_to_idx = session_processor.get_idx_mappings()['event_to_idx']
    
    # Create datasets if not provided
    if train_dataset is None:
        from utils.package_processor import TravelPackageDataset
        print("Creating default training dataset...")
        train_dataset = TravelPackageDataset(
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
    
    if test_dataset is None:
        from utils.package_processor import TravelPackageDataset
        print("Creating default test dataset...")
        test_dataset = TravelPackageDataset(
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
        # Calculate different event type counts
        purchase_count = sum(1 for s in train_samples if s.get('is_purchase', False))
        checkout_count = sum(1 for s in train_samples if not s.get('is_purchase', False) and s.get('has_checkout', False))
        add_to_cart_count = sum(1 for s in train_samples if not s.get('is_purchase', False) and 
                               not s.get('has_checkout', False) and s.get('has_add_to_cart', False))
        
        # Set weights for different sample types
        purchase_boost = 20.0    # Very high weight for actual purchases
        checkout_boost = 8.0     # Medium-high weight for checkouts
        add_to_cart_boost = 3.0  # Medium weight for add-to-cart events
        regular_weight = 1.0     # Base weight for regular browsing
        
        # Calculate weights for each sample
        sample_weights = []
        for sample in train_samples:
            if sample.get('is_purchase', False):
                # Purchase samples get highest weight
                weight = purchase_boost
                
                # Boost rare purchasers even more
                user_id = sample.get('user_id')
                if user_id is not None:
                    # Count purchases for this user
                    user_purchases = sum(1 for s in train_samples 
                                       if s.get('user_id') == user_id and s.get('is_purchase', False))
                    if user_purchases <= 3:  # Boost rare purchasers
                        weight *= 1.5
            elif sample.get('has_checkout', False):
                # Checkout samples get medium-high weight
                weight = checkout_boost
            elif sample.get('has_add_to_cart', False):
                # Add-to-cart samples get medium weight
                weight = add_to_cart_boost
            else:
                # Regular browsing samples get base weight
                weight = regular_weight
            
            sample_weights.append(weight)
        
        print(f"Using weighted sampling for training:")
        print(f"  Purchase samples boost: {purchase_boost}")
        print(f"  Checkout samples boost: {checkout_boost}")
        print(f"  Add-to-cart samples boost: {add_to_cart_boost}")
        print(f"  Regular samples weight: {regular_weight}")
        
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


def analyze_data_distribution(samples: List[Dict], name: str = "Dataset") -> Dict[str, Any]:
    """Analyze the distribution of data with support for multiple event types
    
    Args:
        samples: List of samples
        name: Name of the dataset for display
        
    Returns:
        Dictionary with distribution statistics
    """
    # Count events by type
    purchase_count = sum(1 for s in samples if s.get('is_purchase', False))
    checkout_count = sum(1 for s in samples if not s.get('is_purchase', False) 
                        and s.get('has_checkout', False))
    add_to_cart_count = sum(1 for s in samples if not s.get('is_purchase', False) 
                            and not s.get('has_checkout', False)
                            and s.get('has_add_to_cart', False))
    regular_count = len(samples) - purchase_count - checkout_count - add_to_cart_count
    
    # Calculate combined intent signals
    intent_count = purchase_count + checkout_count
    
    # Print distribution
    print(f"\n{name} Distribution:")
    print(f"  Total samples: {len(samples):,}")
    print(f"  Purchases: {purchase_count:,} ({purchase_count/len(samples)*100:.2f}%)")
    
    if checkout_count > 0:
        print(f"  Checkout events: {checkout_count:,} ({checkout_count/len(samples)*100:.2f}%)")
    
    if add_to_cart_count > 0:
        print(f"  Add-to-cart events: {add_to_cart_count:,} ({add_to_cart_count/len(samples)*100:.2f}%)")
    
    if checkout_count > 0 or add_to_cart_count > 0:
        print(f"  Purchase intent: {intent_count:,} ({intent_count/len(samples)*100:.2f}%)")
        print(f"  Regular browsing: {regular_count:,} ({regular_count/len(samples)*100:.2f}%)")
    else:
        print(f"  Non-purchases: {len(samples)-purchase_count:,}")
    
    # Return distribution stats
    return {
        'total': len(samples),
        'purchase': purchase_count,
        'checkout': checkout_count,
        'add_to_cart': add_to_cart_count,
        'intent': intent_count,
        'regular': regular_count
    }


def set_performance_mode(mode: str = "balanced", device_override: Optional[str] = None) -> Dict[str, Any]:
    """
    Configure training parameters based on performance mode with device-specific optimizations
    
    Args:
        mode: One of "fastest", "balanced", "accurate", "apple_silicon"
        device_override: Override detected device type
        
    Returns:
        dict: Configuration parameters
    """
    config = {}
    
    # Detect device
    if device_override:
        is_mps = device_override == 'mps'
        is_cuda = device_override == 'cuda'
    else:
        is_mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        is_cuda = torch.cuda.is_available()
    
    # Special mode for Apple Silicon (M1/M2/M3)
    if mode == "apple_silicon" or (mode == "balanced" and is_mps):
        # Specific optimizations for Apple Silicon
        config.update({
            # Batch size settings
            "batch_size": 256,                   # Increased explicit batch size
            "embedding_dim": 256,                # Increased from 192 for better representation
            "hidden_dim": 256,                   # Increased from 192 for better model capacity
            
            # Gradient accumulation is important for MPS
            "accumulation_steps": 2,             # Reduced from 4 for faster updates
            
            # MPS doesn't support AMP, but we'll keep this for CUDA compatibility
            "use_amp": False,                    # MPS doesn't support AMP yet
            
            # MPS doesn't yet support torch.compile() in PyTorch stable release
            "compile_model": False,              # Don't attempt to compile for MPS
            
            # Other settings
            "dropout": 0.25,                     # Increased from 0.2 for better regularization
            "patience": 10,                      # Increased to 10 to allow more epochs without improvement
            "prefetch_factor": 2,                # Less prefetching for MPS (tends to use more CPU memory)
            "max_package_count": 50000,          # Increased from 30000 for better coverage
            "use_reduced_embeddings": True,      # Definitely use reduced embeddings for memory efficiency
            "learning_rate": 0.001,              # Reduced to 0.001 for more stable training
            
            # MPS-specific performance flags
            "mps_clear_cache_frequency": 5,      # Clear MPS cache every N batches
            "use_half_precision_embeddings": True, # Use float16 for embeddings on MPS
            "reduce_memory_usage": True,         # Enable additional memory optimizations
            "num_workers": 0,                   # No workers for MPS (avoid multiprocessing errors)
        })
    elif mode == "fastest":
        # Prioritize speed over accuracy
        config.update({
            "batch_size": 320,                   # Explicit larger batch size
            "embedding_dim": 128,                # Smaller embeddings
            "hidden_dim": 128,                   # Smaller hidden dims
            "accumulation_steps": 4,             # More gradient accumulation
            "use_amp": not is_mps,               # Use AMP if not on MPS
            "compile_model": not is_mps,         # Compile if not on MPS
            "dropout": 0.1,                      # Less regularization
            "patience": 2,                       # Less patience for early stopping
            "prefetch_factor": 4,                # More prefetching
            "max_package_count": 10000,          # Limit package vocabulary
            "use_reduced_embeddings": True,      # Use reduced embeddings (768 dim)
            "learning_rate": 0.002,              # Higher learning rate
            "num_workers": 4 if is_cuda else 0   # Workers for CUDA, none for MPS
        })
    elif mode == "accurate":
        # Prioritize accuracy over speed
        config.update({
            "batch_size": 128 if not is_mps else 192,            # Explicit batch size
            "embedding_dim": 512 if not is_mps else 384,          # Larger embeddings
            "hidden_dim": 384 if not is_mps else 320,             # Larger hidden dims
            "accumulation_steps": 1 if not is_mps else 2,         # No/less gradient accumulation
            "use_amp": False,                                      # No mixed precision
            "compile_model": False,                                # No model compilation
            "dropout": 0.3,                                        # More regularization 
            "patience": 5,                                         # More patience for early stopping
            "prefetch_factor": 2,                                  # Less prefetching
            "max_package_count": None,                             # No package limit
            "use_reduced_embeddings": False if not is_mps else True, # Use full embeddings if not on MPS
            "learning_rate": 0.0005,                               # Lower learning rate
            "num_workers": 2 if is_cuda else 0                     # Fewer workers
        })
    else:  # "balanced" mode (non-MPS)
        # Balance between speed and accuracy for non-MPS devices
        config.update({
            "batch_size": 192,                   # Increased explicit batch size
            "embedding_dim": 256,                # Medium embeddings
            "hidden_dim": 256,                   # Medium hidden dims
            "accumulation_steps": 2,             # Some gradient accumulation
            "use_amp": not is_mps,               # Use mixed precision if not MPS
            "compile_model": not is_mps,         # Compile model if not MPS
            "dropout": 0.2,                      # Medium regularization
            "patience": 3,                       # Medium patience
            "prefetch_factor": 3,                # Medium prefetching
            "max_package_count": 50000,          # Some package limit
            "use_reduced_embeddings": True,      # Use reduced embeddings
            "learning_rate": 0.001,              # Standard learning rate
            "num_workers": 2 if is_cuda else 0   # Moderate worker count
        })
    
    # Print detailed configuration
    mode_display = mode
    if mode == "balanced" and is_mps:
        mode_display = "APPLE_SILICON (auto)"
    print(f"\nUsing {mode_display.upper()} performance mode")
    
    if is_mps:
        print("Apple Silicon optimizations active")
    
    return config
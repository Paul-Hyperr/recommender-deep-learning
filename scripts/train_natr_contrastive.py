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
from tqdm import tqdm
from collections import defaultdict, Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.backends.cudnn as cudnn

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enable optimization settings
cudnn.benchmark = True
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')

# Import all necessary components
from models.natr import NATR, NATRConfig
from utils.session_processor import SessionProcessor
from utils.package_processor import PackageProcessor, TravelPackageDataset
from utils.loss_functions import ContrastiveEventLoss, EnhancedEventMetrics


def filter_by_min_session_length(samples, min_session_length=3):
    """Filter out sessions that are too short for quality training"""
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
        # If user has no quality samples but has purchase or checkout samples, keep those
        else:
            # Find purchase, checkout or add_to_cart samples for this user
            important_samples = [
                s for s in user_session_samples if 
                s.get('is_purchase', False) or 
                s.get('has_checkout', False) or
                s.get('has_add_to_cart', False)
            ]
            
            if important_samples:
                # Keep all important samples even if they're short
                quality_samples.extend(important_samples)
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


def filter_items_by_frequency(samples, min_frequency=50):
    """Filter items that appear less than min_frequency times"""
    print(f"\nFiltering items by frequency (min: {min_frequency})...")
    
    # Separate samples by type
    purchase_samples = [s for s in samples if s.get('is_purchase', False)]
    checkout_samples = [s for s in samples if (not s.get('is_purchase', False) and s.get('has_checkout', False))]
    add_to_cart_samples = [s for s in samples if (not s.get('is_purchase', False) and not s.get('has_checkout', False) 
                                                and s.get('has_add_to_cart', False))]
    regular_samples = [s for s in samples if 
                     (not s.get('is_purchase', False) and not s.get('has_checkout', False) and not s.get('has_add_to_cart', False))]
    
    # Get all packages that were part of important events
    retained_packages = set()
    
    for sample in purchase_samples:
        purchased = str(sample.get('purchased_package', ''))
        if purchased:
            retained_packages.add(purchased)
            
    for sample in checkout_samples:
        checkout = str(sample.get('checkout_package', ''))
        if checkout:
            retained_packages.add(checkout)
    
    for sample in add_to_cart_samples:
        add_to_cart = str(sample.get('add_to_cart_package', ''))
        if add_to_cart:
            retained_packages.add(add_to_cart)
    
    print(f"Found {len(retained_packages)} unique packages from important events (will be preserved)")
    
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
    
    # Find valid packages - include all important packages plus frequent ones
    valid_packages = {pkg for pkg, count in package_counts.items() 
                     if count >= min_frequency or pkg in retained_packages}
    print(f"Valid packages: {len(valid_packages)} out of {len(package_counts)}")
    
    # All important samples are kept
    filtered_samples = list(purchase_samples + checkout_samples + add_to_cart_samples)
    
    # For regular samples, filter packages
    for sample in regular_samples:
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
    print(f"Checkout samples preserved: {len(checkout_samples)} (100%)")
    print(f"Add-to-cart samples preserved: {len(add_to_cart_samples)} (100%)")
    
    return filtered_samples, valid_packages


def time_based_split_year(samples, train_ratio=0.93):
    """Split samples based on time for a year of data"""
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


def analyze_data_distribution(samples, name="Dataset"):
    """Analyze the distribution of data with all event types"""
    purchase_count = sum(1 for s in samples if s.get('is_purchase', False))
    checkout_count = sum(1 for s in samples if not s.get('is_purchase', False) and s.get('has_checkout', False))
    add_to_cart_count = sum(1 for s in samples if not s.get('is_purchase', False) and 
                          not s.get('has_checkout', False) and s.get('has_add_to_cart', False))
    view_count = len(samples) - purchase_count - checkout_count - add_to_cart_count
    
    print(f"\n{name} Distribution:")
    print(f"  Total samples: {len(samples):,}")
    print(f"  Purchases: {purchase_count:,} ({purchase_count/len(samples)*100:.2f}%)")
    print(f"  Checkout events: {checkout_count:,} ({checkout_count/len(samples)*100:.2f}%)")
    print(f"  Add-to-cart events: {add_to_cart_count:,} ({add_to_cart_count/len(samples)*100:.2f}%)")
    print(f"  Purchase + Checkout + Cart: {purchase_count + checkout_count + add_to_cart_count:,} " + 
          f"({(purchase_count + checkout_count + add_to_cart_count)/len(samples)*100:.2f}%)")
    print(f"  View-only browsing: {view_count:,} ({view_count/len(samples)*100:.2f}%)")


def identify_event_types(samples, session_processor):
    """
    Identify and mark samples with different event types
    Implements a hierarchy of event importance:
    1. Purchase (strongest signal)
    2. InitiateCheckout (strong intent)
    3. AddToCart (medium intent)
    4. ViewContent (baseline)
    
    Samples can only have one type based on the hierarchy
    """
    print("\nIdentifying samples with different event types...")
    
    # Get event mappings
    event_to_idx = session_processor.get_idx_mappings()['event_to_idx']
    idx_to_event = {v: k for k, v in event_to_idx.items()}
    
    checkout_event_idx = event_to_idx.get('InitiateCheckout')
    add_to_cart_event_idx = event_to_idx.get('AddToCart')
    
    if not checkout_event_idx:
        print("Warning: 'InitiateCheckout' event not found in mapping")
    
    if not add_to_cart_event_idx:
        print("Warning: 'AddToCart' event not found in mapping")
    
    print(f"InitiateCheckout event index: {checkout_event_idx}")
    print(f"AddToCart event index: {add_to_cart_event_idx}")
    
    # Process samples to identify event types
    enhanced_samples = []
    checkout_count = 0
    add_to_cart_count = 0
    purchase_with_checkout_count = 0
    purchase_with_cart_count = 0
    
    for sample in samples:
        # Add event flags (all false by default)
        sample_copy = sample.copy()
        sample_copy['has_checkout'] = False
        sample_copy['has_add_to_cart'] = False
        
        # Check if sample has short-term events
        short_term_events = sample_copy.get('short_term_events', [])
        short_term_pkgs = sample_copy.get('short_term_packages', [])
        
        # Early skip for purchase samples - they already have the strongest signal
        if sample_copy.get('is_purchase', False):
            # Check for both checkout and purchase - just for statistics
            if checkout_event_idx and checkout_event_idx in short_term_events:
                purchase_with_checkout_count += 1
                
            # Check for both add-to-cart and purchase - just for statistics
            if add_to_cart_event_idx and add_to_cart_event_idx in short_term_events:
                purchase_with_cart_count += 1
                
            enhanced_samples.append(sample_copy)
            continue
            
        # CHECKOUT: Look for InitiateCheckout events in short-term sequence
        if checkout_event_idx and checkout_event_idx in short_term_events:
            # Get the most recent checkout package
            checkout_package = None
            for i, event_idx in enumerate(reversed(short_term_events)):
                if event_idx == checkout_event_idx:
                    # Get the package associated with this checkout
                    rev_index = len(short_term_events) - 1 - i
                    if rev_index < len(short_term_pkgs):
                        checkout_package = short_term_pkgs[rev_index]
                    break
            
            # Mark as checkout and set the checkout package
            if checkout_package:
                sample_copy['has_checkout'] = True
                sample_copy['checkout_package'] = checkout_package
                sample_copy['purchased_package'] = checkout_package  # Use as target
                checkout_count += 1
                enhanced_samples.append(sample_copy)
                continue
        
        # ADD TO CART: Only if not a purchase or checkout
        if add_to_cart_event_idx and add_to_cart_event_idx in short_term_events:
            # Get the most recent add-to-cart package
            add_to_cart_package = None
            for i, event_idx in enumerate(reversed(short_term_events)):
                if event_idx == add_to_cart_event_idx:
                    # Get the package associated with this add-to-cart
                    rev_index = len(short_term_events) - 1 - i
                    if rev_index < len(short_term_pkgs):
                        add_to_cart_package = short_term_pkgs[rev_index]
                    break
            
            # Mark as add-to-cart and set the add-to-cart package
            if add_to_cart_package:
                sample_copy['has_add_to_cart'] = True
                sample_copy['add_to_cart_package'] = add_to_cart_package
                sample_copy['purchased_package'] = add_to_cart_package  # Use as target
                add_to_cart_count += 1
                enhanced_samples.append(sample_copy)
                continue
        
        # VIEW CONTENT: Regular browsing sample - no special handling
        enhanced_samples.append(sample_copy)
    
    print(f"Identified {checkout_count} checkout-only samples")
    print(f"Identified {add_to_cart_count} add-to-cart-only samples")
    print(f"Found {purchase_with_checkout_count} samples with both purchase and checkout events")
    print(f"Found {purchase_with_cart_count} samples with both purchase and add-to-cart events")
    
    return enhanced_samples


def move_batch_to_device(batch, device):
    """Recursively move batch to device with optimized handling for both CUDA and MPS"""
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


def train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, 
            accumulation_steps=1, use_amp=False):
    """Train for one epoch with optimizations, with specific enhancements for MPS"""
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
                
                # Get event type flags
                is_purchase = batch.get('is_purchase', torch.zeros_like(targets, dtype=torch.bool))
                has_checkout = batch.get('has_checkout', torch.zeros_like(targets, dtype=torch.bool))
                has_add_to_cart = batch.get('has_add_to_cart', torch.zeros_like(targets, dtype=torch.bool))
                
                # Calculate loss with contrastive margins for different event types
                loss = loss_fn(predictions, targets, is_purchase, has_checkout, has_add_to_cart)
                loss = loss / accumulation_steps  # Scale for accumulation
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
        else:
            # Print batch structure for debugging (just for the first batch of first epoch)
            if batch_idx == 0 and epoch == 0:
                print("Batch structure:")
                for k in batch:
                    if isinstance(batch[k], dict):
                        print(f"  {k}:")
                        for k2 in batch[k]:
                            print(f"    {k2}: {type(batch[k][k2])} {batch[k][k2].shape if hasattr(batch[k][k2], 'shape') else ''}")
                    elif isinstance(batch[k], torch.Tensor):
                        print(f"  {k}: {batch[k].shape}")
                    else:
                        print(f"  {k}: {type(batch[k])}")
            
            # Standard forward pass
            outputs = model(batch)
            predictions = outputs['predictions']
            
            # Get targets
            targets = batch['purchased']['package_ids']
            
            # Get event type flags
            is_purchase = batch.get('is_purchase', torch.zeros_like(targets, dtype=torch.bool))
            has_checkout = batch.get('has_checkout', torch.zeros_like(targets, dtype=torch.bool))
            has_add_to_cart = batch.get('has_add_to_cart', torch.zeros_like(targets, dtype=torch.bool))
            
            # Calculate loss with contrastive margins for different event types
            loss = loss_fn(predictions, targets, is_purchase, has_checkout, has_add_to_cart)
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


def evaluate(model, test_loader, device, k_values=[5, 10, 20]):
    """Evaluate model with enhanced metrics for different event types"""
    model.eval()
    
    # Initialize metrics accumulator
    metrics_acc = defaultdict(float)
    event_type_counts = defaultdict(int)
    
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
            
            # Get event type flags
            is_purchase = batch.get('is_purchase', torch.zeros_like(targets, dtype=torch.bool))
            has_checkout = batch.get('has_checkout', torch.zeros_like(targets, dtype=torch.bool))
            has_add_to_cart = batch.get('has_add_to_cart', torch.zeros_like(targets, dtype=torch.bool))
            
            # Calculate comprehensive metrics
            batch_metrics = EnhancedEventMetrics.calculate_metrics(
                predictions, targets, is_purchase, has_checkout, has_add_to_cart, k_values
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
                torch.mps.empty_cache()
                
                # Show memory usage in progress bar if available
                if hasattr(torch.mps, 'current_allocated_memory'):
                    mem_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
                    progress_bar.set_postfix({'memory_mb': f"{mem_mb:.0f}"})
    
    # Final cleanup for MPS
    if is_mps:
        torch.mps.empty_cache()
    
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


class ContrastiveAwareDataset(TravelPackageDataset):
    """
    Extended dataset that handles different event types for contrastive learning
    This dataset adds event type flags to model input
    """
    
    def __getitem__(self, idx):
        """Enhanced getitem that passes event type flags to the model"""
        # Get base sample
        sample = super().__getitem__(idx)
        
        # Add event type flags
        sample['is_purchase'] = self.samples[idx].get('is_purchase', False)
        sample['has_checkout'] = self.samples[idx].get('has_checkout', False)
        sample['has_add_to_cart'] = self.samples[idx].get('has_add_to_cart', False)
        
        return sample


def create_dataloaders(train_samples, test_samples, package_processor, session_processor, 
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
        from torch.utils.data import WeightedRandomSampler
        
        # Calculate event type counts
        purchase_count = sum(1 for s in train_samples if s.get('is_purchase', False))
        checkout_count = sum(1 for s in train_samples if not s.get('is_purchase', False) 
                            and s.get('has_checkout', False))
        add_to_cart_count = sum(1 for s in train_samples if not s.get('is_purchase', False) 
                              and not s.get('has_checkout', False) and s.get('has_add_to_cart', False))
        view_count = len(train_samples) - purchase_count - checkout_count - add_to_cart_count
        
        # Set weights for different sample types
        purchase_boost = 15.0    # Very high weight for actual purchases
        checkout_boost = 7.5     # Medium-high weight for checkouts
        add_to_cart_boost = 3.0  # Medium weight for add-to-cart
        view_weight = 1.0        # Base weight for regular browsing
        
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


def set_performance_mode(mode="balanced"):
    """Configure training parameters based on performance mode with contrastive learning parameters"""
    config = {}
    
    # Special mode for Apple Silicon (M1/M2/M3)
    is_mps = torch.backends.mps.is_available()
    
    if mode == "apple_silicon" or (mode == "balanced" and is_mps):
        # Specific optimizations for Apple Silicon
        config.update({
            # Batch size settings
            "batch_size": 256,                   # Increased batch size for Apple Silicon (was using multiplier 1.5)
            "embedding_dim": 256,                # Increased from 192 for better representation
            "hidden_dim": 256,                   # Increased from 192 for better model capacity
            "accumulation_steps": 2,             # Reduced from 4 for faster updates
            "use_amp": False,                    # MPS doesn't support AMP yet
            "compile_model": False,              # Don't attempt to compile for MPS
            "dropout": 0.25,                     # Increased from 0.2 for better regularization
            "patience": 10,                      # Increased to 10 to allow more epochs without improvement
            "prefetch_factor": 2,                # Less prefetching for MPS (tends to use more CPU memory)
            "max_package_count": 50000,          # Increased from 30000 for better coverage
            "use_reduced_embeddings": True,      # Definitely use reduced embeddings for memory efficiency
            "learning_rate": 0.001,              # Reduced to 0.001 for more stable training
            "mps_clear_cache_frequency": 5,      # Clear MPS cache every N batches
            "use_half_precision_embeddings": True, # Use float16 for embeddings on MPS
            "reduce_memory_usage": True,         # Enable additional memory optimizations
            "num_workers": 0,                    # No workers for MPS (avoid multiprocessing errors)
            "epochs": 30,                        # Number of epochs
            
            # Contrastive margin parameters
            "temperature": 0.1,                  # Contrastive temperature - lower = sharper distinctions
            "purchase_margin": 0.0,              # No margin for purchases (strongest signal)
            "checkout_margin": 0.3,              # Small margin for checkouts (medium signal)
            "add_to_cart_margin": 0.6,           # Medium margin for add-to-cart events
            "view_margin": 1.0,                  # Full margin for view events (weakest signal)
            "hard_negative_ratio": 0.7           # Proportion of hard negatives to keep
        })
    elif mode == "fastest":
        # Prioritize speed over accuracy
        config.update({
            "batch_size": 320,                   # Larger batches (was using multiplier 2.0)
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
            "epochs": 15,                        # Fewer epochs
            
            # Contrastive margin parameters - less aggressive for faster convergence
            "temperature": 0.2,                  # Higher temperature = smoother distinctions
            "purchase_margin": 0.0,              # No margin for purchases
            "checkout_margin": 0.2,              # Smaller checkout margin for faster convergence
            "add_to_cart_margin": 0.4,           # Smaller add-to-cart margin
            "view_margin": 0.8,                  # Smaller view margin
            "hard_negative_ratio": 0.5           # Fewer hard negatives for faster training
        })
    elif mode == "accurate":
        # Prioritize accuracy over speed
        config.update({
            "batch_size": 128 if not is_mps else 192,              # Smaller batches (was using multiplier 0.5/0.75)
            "embedding_dim": 512 if not is_mps else 384,          # Larger embeddings (adjusted for MPS)
            "hidden_dim": 384 if not is_mps else 320,             # Larger hidden dims (adjusted for MPS)
            "accumulation_steps": 1 if not is_mps else 2,         # No/less gradient accumulation
            "use_amp": False,                                      # No mixed precision
            "compile_model": False,                                # No model compilation
            "dropout": 0.3,                                        # More regularization 
            "patience": 5,                                         # More patience for early stopping
            "prefetch_factor": 2,                                  # Less prefetching
            "max_package_count": None,                             # No package limit
            "use_reduced_embeddings": False if not is_mps else True, # Use full embeddings if not on MPS
            "learning_rate": 0.0005,                               # Lower learning rate
            "epochs": 40,                                          # More epochs
            
            # Contrastive margin parameters - more aggressive for better accuracy
            "temperature": 0.05,                 # Lower temperature = sharper distinctions
            "purchase_margin": 0.0,              # No margin for purchases
            "checkout_margin": 0.25,             # Precise checkout margin
            "add_to_cart_margin": 0.5,           # Precise add-to-cart margin
            "view_margin": 1.0,                  # Full margin for views
            "hard_negative_ratio": 0.8           # More hard negatives for better discrimination
        })
    else:  # "balanced" mode (non-MPS)
        # Balance between speed and accuracy for non-MPS devices
        config.update({
            "batch_size": 192,                   # Increased batch size (was using multiplier 1.0)
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
            "epochs": 30,                        # Standard epochs
            
            # Contrastive margin parameters - balanced configuration
            "temperature": 0.1,                  # Balanced temperature
            "purchase_margin": 0.0,              # No margin for purchases
            "checkout_margin": 0.3,              # Standard checkout margin
            "add_to_cart_margin": 0.6,           # Standard add-to-cart margin
            "view_margin": 1.0,                  # Standard view margin
            "hard_negative_ratio": 0.7           # Standard hard negative ratio
        })
    
    # Print detailed configuration
    mode_display = mode
    if mode == "balanced" and is_mps:
        mode_display = "APPLE_SILICON (auto)"
    print(f"\nUsing {mode_display.upper()} performance mode")
    
    if is_mps:
        print("Apple Silicon optimizations active")
    
    return config


def main(performance_config=None):
    """Main function implementing contrastive learning with event-based margins"""
    # Set device - for Apple Silicon (M1/M2/M3), we can use MPS
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        # MPS-specific optimizations
        torch.mps.empty_cache()
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
    
    # Data paths
    package_data_path = "data/feed.parquet"
    event_data_path = "data/bookit_events_data_13_months.parquet"
    
    # Initialize processors
    print("\nInitializing data processors...")
    package_processor = PackageProcessor(
        data_path=package_data_path,
        cache_dir='data/cache',
        load_coordinates=True,
        load_embeddings=True,
        api_key=os.environ.get("OPENAI_API_KEY"),
        embedding_model='text-embedding-3-small',  # Use small model for lower dimensionality
        use_reduced_embeddings=performance_config.get("use_reduced_embeddings", True)
    )
    
    session_processor = SessionProcessor(
        data_path=event_data_path,
        cache_dir='data/cache',
        min_interactions=10,  # Users must have at least 10 interactions
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
    
    # Enhance samples with event type identification
    print("\n6. Enhancing samples with event types...")
    enhanced_samples = identify_event_types(samples, session_processor)
    
    print("\n7. Applying data filters...")
    
    # Apply session length filtering - quality improves with longer sessions
    quality_samples = filter_by_min_session_length(enhanced_samples, min_session_length=3)
    
    # Filter items by frequency - focus on packages with sufficient data
    filtered_samples, valid_packages = filter_items_by_frequency(quality_samples, min_frequency=50)
    
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
    train_samples, test_samples = time_based_split_year(filtered_samples, train_ratio=0.93)
    
    # Ensure evaluation set has enough events of each type for meaningful metrics
    purchase_samples = [s for s in test_samples if s.get('is_purchase', False)]
    checkout_samples = [s for s in test_samples if not s.get('is_purchase', False) and s.get('has_checkout', False)]
    add_to_cart_samples = [s for s in test_samples if not s.get('is_purchase', False) 
                          and not s.get('has_checkout', False) and s.get('has_add_to_cart', False)]
    
    # Ensure enough purchases in test set
    if len(purchase_samples) < 50:
        print(f"Warning: Only {len(purchase_samples)} purchases in test set, adding more...")
        # Find more purchases from train samples
        extra_purchases = [s for s in train_samples if s.get('is_purchase', False)][:50-len(purchase_samples)]
        test_samples.extend(extra_purchases)
        # Remove these from train samples
        train_samples = [s for s in train_samples if s not in extra_purchases]
        print(f"Added {len(extra_purchases)} more purchase samples to test set")
    
    # Ensure enough checkouts in test set
    if len(checkout_samples) < 50:
        print(f"Warning: Only {len(checkout_samples)} checkout events in test set, adding more...")
        # Find more checkout samples from train samples
        extra_checkouts = [
            s for s in train_samples if 
            (not s.get('is_purchase', False) and s.get('has_checkout', False))
        ][:50-len(checkout_samples)]
        test_samples.extend(extra_checkouts)
        # Remove these from train samples
        train_samples = [s for s in train_samples if s not in extra_checkouts]
        print(f"Added {len(extra_checkouts)} more checkout samples to test set")
    
    # Ensure enough add-to-cart in test set
    if len(add_to_cart_samples) < 50:
        print(f"Warning: Only {len(add_to_cart_samples)} add-to-cart events in test set, adding more...")
        # Find more add-to-cart samples from train samples
        extra_carts = [
            s for s in train_samples if 
            (not s.get('is_purchase', False) and not s.get('has_checkout', False) and s.get('has_add_to_cart', False))
        ][:50-len(add_to_cart_samples)]
        test_samples.extend(extra_carts)
        # Remove these from train samples
        train_samples = [s for s in train_samples if s not in extra_carts]
        print(f"Added {len(extra_carts)} more add-to-cart samples to test set")
    
    # Analyze distributions
    analyze_data_distribution(train_samples, "Train")
    analyze_data_distribution(test_samples, "Test")
    
    # Create datasets and dataloaders
    print("\n8. Creating dataloaders...")
    train_loader, test_loader = create_dataloaders(
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
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.05,  # Increased weight decay for stronger regularization (from 0.03)
        eps=1e-8  # More stable epsilon value
    )
    
    # Use a simple step scheduler with no warmup
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',           # Optimize for maximizing recall
        factor=0.5,           # Halve the learning rate when plateauing
        patience=2,           # Wait 2 epochs before reducing
        threshold=0.0001,     # Minimum improvement to count as progress
        min_lr=1e-6           # Minimum learning rate
    )
    
    # Create contrastive loss function with margin-based event learning
    loss_fn = ContrastiveEventLoss(
        temperature=performance_config.get("temperature", 0.1),
        purchase_margin=performance_config.get("purchase_margin", 0.0),
        checkout_margin=performance_config.get("checkout_margin", 0.3),
        add_to_cart_margin=performance_config.get("add_to_cart_margin", 0.6),
        view_margin=performance_config.get("view_margin", 1.0),
        hard_negative_mining=True,
        hard_negative_ratio=performance_config.get("hard_negative_ratio", 0.7)
    )
    
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
    
    # Training loop
    # Get performance parameters
    use_amp = performance_config.get("use_amp", False) and device.type == 'cuda'
    early_stopping_patience = performance_config.get("patience", 5)
    
    print("\n10. Starting training...")
    print(f"Training for {num_epochs} epochs with contrastive margins")
    print(f"Early stopping patience: {early_stopping_patience} epochs")
    
    # Track the metric we care about most - combined purchase+checkout+cart performance
    best_combined_recall = 0
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
        
        # Evaluate
        start_time = time.time()
        test_metrics = evaluate(model, test_loader, device, k_values=[5, 10, 20])
        eval_time = time.time() - start_time
        
        # Calculate combined recall metric (purchase + checkout + add-to-cart), weighed by importance
        # Purchase gets full weight, checkout gets 0.7 weight, add-to-cart gets 0.3 weight
        purchase_recall = test_metrics.get('purchase_recall@k', {}).get(10, 0)
        checkout_recall = test_metrics.get('checkout_recall@k', {}).get(10, 0)
        cart_recall = test_metrics.get('add_to_cart_recall@k', {}).get(10, 0)
        
        # Get counts
        purchase_count = test_metrics.get('purchase', 0)
        checkout_count = test_metrics.get('checkout', 0)
        cart_count = test_metrics.get('add_to_cart', 0)
        total_important = purchase_count + checkout_count + cart_count
        
        # Weighted average based on relative importance and count
        if total_important > 0:
            purchase_weight = 1.0
            checkout_weight = 0.7
            cart_weight = 0.3
            
            purchase_contribution = purchase_recall * purchase_weight * purchase_count
            checkout_contribution = checkout_recall * checkout_weight * checkout_count
            cart_contribution = cart_recall * cart_weight * cart_count
            
            current_combined_recall = (purchase_contribution + checkout_contribution + cart_contribution) / \
                                     (purchase_weight * purchase_count + checkout_weight * checkout_count + cart_weight * cart_count)
        else:
            current_combined_recall = 0
        
        # Update scheduler with the combined metric
        scheduler.step(current_combined_recall)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['test_metrics'].append(test_metrics)
        
        # Print metrics
        print(f"\nTiming: Train={train_time:.1f}s, Eval={eval_time:.1f}s")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"\nTest Metrics:")
        print(f"  Overall Recall@10: {test_metrics['recall@k'][10]*100:.2f}%")
        print(f"  Purchase Recall@10: {purchase_recall*100:.2f}%")
        
        if 'checkout_recall@k' in test_metrics:
            print(f"  Checkout Recall@10: {checkout_recall*100:.2f}%")
        
        if 'add_to_cart_recall@k' in test_metrics:
            print(f"  Add-to-cart Recall@10: {cart_recall*100:.2f}%")
        
        if 'intent_recall@k' in test_metrics:
            print(f"  Intent (P+C) Recall@10: {test_metrics['intent_recall@k'][10]*100:.2f}%")
        
        print(f"  Combined weighted Recall@10: {current_combined_recall*100:.2f}%")
        print(f"  Purchase MRR: {test_metrics.get('purchase_mrr', 0):.4f}")
        
        # Save best model based on our combined metric
        if current_combined_recall > best_combined_recall:
            best_combined_recall = current_combined_recall
            epochs_without_improvement = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_combined_recall': best_combined_recall,
                'config': config.__dict__,
                'valid_packages': list(valid_packages),
                'history': history,
                'performance_config': performance_config,
                'margins': loss_fn.get_margins(),
                'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            }
            
            # Create checkpoints directory
            os.makedirs('checkpoints/natr', exist_ok=True)
            
            # Save to both a versioned file and the best model file
            torch.save(checkpoint, f"checkpoints/natr/contrastive_model_epoch_{epoch+1}.pth")
            torch.save(checkpoint, 'checkpoints/natr/contrastive_best_model.pth')
            print(f"  ✓ New best model saved! Combined Recall@10: {best_combined_recall*100:.2f}%")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epochs")
            
            # Save periodic checkpoint every 5 epochs for recovery purposes
            if (epoch + 1) % 5 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_combined_recall': best_combined_recall,
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
    print(f"Best Combined Recall@10: {best_combined_recall*100:.2f}%")
    
    # Save model info
    model_info = {
        'config': config.__dict__,
        'valid_packages': list(valid_packages),
        'best_combined_recall': best_combined_recall,
        'checkpoint_path': 'checkpoints/natr/contrastive_best_model.pth',
        'package_data_path': package_data_path,
        'event_data_path': event_data_path,
        'performance_config': performance_config,
        'contrastive_margins': loss_fn.get_margins(),
        'training_date': datetime.now().strftime("%Y-%m-%d"),
        'package_count': len(valid_packages),
        'user_count': num_users,
        'training_strategy': 'contrastive_learning'
    }
    
    with open('model_info_contrastive.json', 'w') as f:
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
    main(performance_config)
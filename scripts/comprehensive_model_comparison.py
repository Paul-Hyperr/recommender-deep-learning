"""
Comprehensive model comparison for all NATR approaches:
1. Standard (Focal Loss)
2. Pretrain+Finetune 
3. Checkout Enhanced
4. Contrastive

Optimized for Apple Silicon MPS with good memory management
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
import gc
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all necessary components
from models.natr import NATR, NATRConfig
from utils.session_processor import SessionProcessor
from utils.package_processor import PackageProcessor, TravelPackageDataset
from utils.loss_functions import FocalLoss, ContrastiveEventLoss, CheckoutEnhancedLoss


def recursive_to_device(batch, device):
    """Move batch to device recursively handling nested structures"""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: recursive_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [recursive_to_device(v, device) for v in batch]
    else:
        return batch


def train_model(model, train_loader, optimizer, loss_fn, device, epochs=3, 
                max_batches=None, model_name="Model",
                accumulation_steps=1, clear_cache_freq=5):
    """Train model with careful memory management"""
    model.train()
    history = {'train_loss': []}
    
    # Move model to device
    model = model.to(device)
    
    # Training loop
    for epoch in range(epochs):
        # Clear cache
        if device.type == 'mps':
            torch.mps.empty_cache()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Track metrics
        total_loss = 0
        batch_count = 0
        
        # Progress bar
        if max_batches:
            pbar = tqdm(enumerate(train_loader), total=min(max_batches, len(train_loader)), 
                        desc=f"{model_name} Epoch {epoch+1}/{epochs}")
        else:
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                        desc=f"{model_name} Epoch {epoch+1}/{epochs}")
        
        # Reset gradient
        optimizer.zero_grad()
        
        for batch_idx, batch in pbar:
            # Stop after max_batches
            if max_batches and batch_idx >= max_batches:
                break
            
            try:
                # Move batch to device
                batch = recursive_to_device(batch, device)
                
                # Forward pass
                outputs = model(batch)
                predictions = outputs['predictions']
                targets = batch['purchased']['package_ids']
                
                # Ensure targets are 1D
                if targets.dim() > 1:
                    targets = targets.squeeze()
                
                # Process is_purchase flag
                if 'is_purchase' in batch:
                    is_purchase = batch['is_purchase']
                else:
                    is_purchase = torch.zeros_like(targets, dtype=torch.bool, device=device)
                
                # Calculate loss
                # Get checkout and add_to_cart flags for all loss functions
                if 'has_checkout' in batch:
                    has_checkout = batch['has_checkout']
                else:
                    has_checkout = torch.zeros_like(targets, dtype=torch.bool, device=device)
                    
                if 'has_add_to_cart' in batch:
                    has_add_to_cart = batch['has_add_to_cart']
                else:
                    has_add_to_cart = torch.zeros_like(targets, dtype=torch.bool, device=device)
                
                # Handle different loss function signatures
                if 'ContrastiveEventLoss' in loss_fn.__class__.__name__ or 'CheckoutEnhancedLoss' in loss_fn.__class__.__name__:
                    loss = loss_fn(predictions, targets, is_purchase, has_checkout, has_add_to_cart)
                else:
                    loss = loss_fn(predictions, targets, is_purchase)
                
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Accumulate metrics
                total_loss += loss.item() * accumulation_steps
                batch_count += 1
                
                # Update weights with gradient accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Update progress
                pbar.set_postfix({'loss': total_loss / batch_count})
                
                # Clear cache periodically
                if device.type in ['mps', 'cuda'] and batch_idx % clear_cache_freq == 0:
                    if device.type == 'mps':
                        torch.mps.empty_cache()
                    else:
                        torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                # Skip problematic batch
                continue
        
        # Make sure to update weights for any remaining gradients
        if batch_count % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Calculate epoch metrics
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f"{model_name} Epoch {epoch+1} Loss: {avg_loss:.4f}")
            history['train_loss'].append(avg_loss)
        else:
            print(f"Warning: No batches processed in epoch {epoch+1}")
    
    # Clear cache at end
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return history


def evaluate_model(model, test_loader, device, max_batches=None, model_name="Model", top_k_values=[10, 20]):
    """Evaluate model with careful memory management and enhanced metrics"""
    model.eval()
    
    # Move model to device if needed
    if next(model.parameters()).device != device:
        model = model.to(device)
    
    # Get the maximum k value for evaluation
    max_k = max(top_k_values)
    
    # Initialize metrics for each k value
    metrics = {}
    for k in top_k_values:
        # Overall metrics for all users
        metrics[f'overall_recall@{k}'] = 0
        metrics[f'purchase_recall@{k}'] = 0
        metrics[f'checkout_recall@{k}'] = 0
        metrics[f'add_to_cart_recall@{k}'] = 0
        
        # Cold-start user metrics
        metrics[f'cold_start_overall_recall@{k}'] = 0  # Overall recall for cold-start users
        metrics[f'cold_start_purchase_recall@{k}'] = 0
        metrics[f'cold_start_checkout_recall@{k}'] = 0
        metrics[f'cold_start_add_to_cart_recall@{k}'] = 0
        
        # Warm-start user metrics
        metrics[f'warm_start_overall_recall@{k}'] = 0  # Overall recall for warm-start users
        metrics[f'warm_start_purchase_recall@{k}'] = 0
        metrics[f'warm_start_checkout_recall@{k}'] = 0
        metrics[f'warm_start_add_to_cart_recall@{k}'] = 0
    
    # Overall counters
    total_samples = 0
    purchase_count = 0
    checkout_count = 0
    add_to_cart_count = 0
    
    # Cold-start counters
    cold_start_count = 0
    cold_start_purchase_count = 0
    cold_start_checkout_count = 0
    cold_start_add_to_cart_count = 0
    
    # Warm-start counters
    warm_start_count = 0
    warm_start_purchase_count = 0
    warm_start_checkout_count = 0
    warm_start_add_to_cart_count = 0
    
    # Clear cache
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Progress bar
    if max_batches:
        pbar = tqdm(enumerate(test_loader), total=min(max_batches, len(test_loader)), 
                    desc=f"Evaluating {model_name}")
    else:
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), 
                    desc=f"Evaluating {model_name}")
    
    # Evaluation loop
    with torch.no_grad():
        for batch_idx, batch in pbar:
            # Stop after max_batches
            if max_batches and batch_idx >= max_batches:
                break
            
            try:
                # Move batch to device
                batch = recursive_to_device(batch, device)
                
                # Forward pass
                outputs = model(batch)
                predictions = outputs['predictions']
                targets = batch['purchased']['package_ids']
                
                # Ensure targets are 1D
                if targets.dim() > 1:
                    targets = targets.squeeze()
                
                # Process is_purchase flag
                if 'is_purchase' in batch:
                    is_purchase = batch['is_purchase']
                else:
                    is_purchase = torch.zeros_like(targets, dtype=torch.bool, device=device)
                
                # Get top k predictions for the maximum k in top_k_values
                _, top_indices = torch.topk(predictions, max_k, dim=1)
                
                # Calculate recalls
                batch_size = targets.size(0)
                
                # Handle dimension issues
                if targets.dim() > 1:
                    # Squeeze any extra dimensions
                    targets = targets.squeeze()
                    
                # Need to reshape to ensure we get a [batch_size, 1] tensor
                targets_reshaped = targets.view(-1, 1)
                
                # Identify cold-start vs warm-start users
                # Cold start: user_id is 0 which is the default for users not seen during training
                # These are true cold-start users with no historical interactions
                user_ids = batch['user_id']
                cold_start_mask = (user_ids == 0).bool()  # user_id=0 means new/unseen user
                warm_start_mask = ~cold_start_mask       # All other users are warm-start
                
                # Count cold/warm start samples
                cold_start_count += cold_start_mask.sum().item()
                warm_start_count += warm_start_mask.sum().item()
                
                # Process purchase mask once for all metrics
                purchase_mask = is_purchase.bool()
                purchase_count += purchase_mask.sum().item()
                
                # Identify purchase samples that are cold/warm start
                cold_start_purchase_mask = cold_start_mask & purchase_mask
                warm_start_purchase_mask = warm_start_mask & purchase_mask
                cold_start_purchase_count += cold_start_purchase_mask.sum().item()
                warm_start_purchase_count += warm_start_purchase_mask.sum().item()
                
                # Update total samples counter
                total_samples += batch_size
                
                # Calculate metrics for each k value
                for k in top_k_values:
                    # Use the appropriate slice of top indices for this k
                    top_k_slice = top_indices[:, :k]
                    
                    # Create expanded target for comparison
                    target_expanded_k = targets_reshaped.expand(-1, k)
                    
                    # Calculate correct predictions at this k
                    correct_at_k = (top_k_slice == target_expanded_k).any(dim=1)
                    
                    # Update overall recall@k
                    metrics[f'overall_recall@{k}'] += correct_at_k.float().sum().item()
                    
                    # Update warm/cold start metrics
                    cold_start_correct = correct_at_k & cold_start_mask
                    warm_start_correct = correct_at_k & warm_start_mask
                    metrics[f'cold_start_overall_recall@{k}'] += cold_start_correct.float().sum().item()
                    metrics[f'warm_start_overall_recall@{k}'] += warm_start_correct.float().sum().item()
                    
                    # Update purchase recall@k
                    if purchase_mask.sum() > 0:
                        purchase_correct = correct_at_k & purchase_mask
                        metrics[f'purchase_recall@{k}'] += purchase_correct.float().sum().item()
                        
                        # Update cold/warm start purchase recalls
                        cold_purchase_correct = correct_at_k & cold_start_purchase_mask
                        warm_purchase_correct = correct_at_k & warm_start_purchase_mask
                        metrics[f'cold_start_purchase_recall@{k}'] += cold_purchase_correct.float().sum().item()
                        metrics[f'warm_start_purchase_recall@{k}'] += warm_purchase_correct.float().sum().item()
                
                # Get checkout and add_to_cart masks for all models (not just contrastive)
                checkout_mask = None
                add_to_cart_mask = None
                cold_start_checkout_mask = None
                warm_start_checkout_mask = None
                cold_start_cart_mask = None
                warm_start_cart_mask = None
                
                if 'has_checkout' in batch:
                    checkout_mask = batch['has_checkout'].bool()
                    checkout_count += checkout_mask.sum().item()
                    
                    # Count cold/warm start checkout samples
                    cold_start_checkout_mask = cold_start_mask & checkout_mask
                    warm_start_checkout_mask = warm_start_mask & checkout_mask
                    
                    # Update counters
                    cold_start_checkout_count += cold_start_checkout_mask.sum().item()
                    warm_start_checkout_count += warm_start_checkout_mask.sum().item()
                else:
                    # Default to all zeros if not provided
                    checkout_mask = torch.zeros_like(targets, dtype=torch.bool, device=device)
                    cold_start_checkout_mask = torch.zeros_like(targets, dtype=torch.bool, device=device)
                    warm_start_checkout_mask = torch.zeros_like(targets, dtype=torch.bool, device=device)
                
                if 'has_add_to_cart' in batch:
                    add_to_cart_mask = batch['has_add_to_cart'].bool()
                    add_to_cart_count += add_to_cart_mask.sum().item()
                    
                    # Count cold/warm start add-to-cart samples
                    cold_start_cart_mask = cold_start_mask & add_to_cart_mask
                    warm_start_cart_mask = warm_start_mask & add_to_cart_mask
                    
                    # Update counters
                    cold_start_add_to_cart_count += cold_start_cart_mask.sum().item()
                    warm_start_add_to_cart_count += warm_start_cart_mask.sum().item()
                else:
                    # Default to all zeros if not provided
                    add_to_cart_mask = torch.zeros_like(targets, dtype=torch.bool, device=device)
                    cold_start_cart_mask = torch.zeros_like(targets, dtype=torch.bool, device=device)
                    warm_start_cart_mask = torch.zeros_like(targets, dtype=torch.bool, device=device)
                
                # Calculate metrics for each k for checkout and add-to-cart
                for k in top_k_values:
                    # We already have top_k_slice, target_expanded_k, and correct_at_k from above
                    # So we don't need to recalculate them
                    
                    # Update checkout recall@k for all users
                    checkout_correct = correct_at_k & checkout_mask
                    metrics[f'checkout_recall@{k}'] += checkout_correct.float().sum().item()
                    
                    # Update checkout recall for cold/warm start users
                    cold_checkout_correct = correct_at_k & cold_start_checkout_mask
                    warm_checkout_correct = correct_at_k & warm_start_checkout_mask
                    metrics[f'cold_start_checkout_recall@{k}'] += cold_checkout_correct.float().sum().item()
                    metrics[f'warm_start_checkout_recall@{k}'] += warm_checkout_correct.float().sum().item()
                    
                    # Update add_to_cart recall@k for all users
                    add_to_cart_correct = correct_at_k & add_to_cart_mask
                    metrics[f'add_to_cart_recall@{k}'] += add_to_cart_correct.float().sum().item()
                    
                    # Update add-to-cart recall for cold/warm start users
                    cold_cart_correct = correct_at_k & cold_start_cart_mask
                    warm_cart_correct = correct_at_k & warm_start_cart_mask
                    metrics[f'cold_start_add_to_cart_recall@{k}'] += cold_cart_correct.float().sum().item()
                    metrics[f'warm_start_add_to_cart_recall@{k}'] += warm_cart_correct.float().sum().item()
                
                # Clear cache periodically
                if device.type in ['mps', 'cuda'] and batch_idx % 10 == 0:
                    if device.type == 'mps':
                        torch.mps.empty_cache()
                    else:
                        torch.cuda.empty_cache()
                        
            except Exception as e:
                print(f"Error in evaluation batch {batch_idx}: {e}")
                # Skip problematic batch
                continue
    
    # All counters are now maintained throughout the evaluation loop
    
    # Normalize metrics for all k values
    for k in top_k_values:
        # Normalize overall metrics
        if total_samples > 0:
            metrics[f'overall_recall@{k}'] /= total_samples
        
        if purchase_count > 0:
            metrics[f'purchase_recall@{k}'] /= purchase_count
        
        if checkout_count > 0:
            metrics[f'checkout_recall@{k}'] /= checkout_count
        
        if add_to_cart_count > 0:
            metrics[f'add_to_cart_recall@{k}'] /= add_to_cart_count
        
        # Normalize cold-start metrics
        if cold_start_count > 0:
            metrics[f'cold_start_overall_recall@{k}'] /= cold_start_count
        
        if cold_start_purchase_count > 0:
            metrics[f'cold_start_purchase_recall@{k}'] /= cold_start_purchase_count
        
        if cold_start_checkout_count > 0:
            metrics[f'cold_start_checkout_recall@{k}'] /= cold_start_checkout_count
        
        if cold_start_add_to_cart_count > 0:
            metrics[f'cold_start_add_to_cart_recall@{k}'] /= cold_start_add_to_cart_count
        
        # Normalize warm-start metrics
        if warm_start_count > 0:
            metrics[f'warm_start_overall_recall@{k}'] /= warm_start_count
        
        if warm_start_purchase_count > 0:
            metrics[f'warm_start_purchase_recall@{k}'] /= warm_start_purchase_count
        
        if warm_start_checkout_count > 0:
            metrics[f'warm_start_checkout_recall@{k}'] /= warm_start_checkout_count
        
        if warm_start_add_to_cart_count > 0:
            metrics[f'warm_start_add_to_cart_recall@{k}'] /= warm_start_add_to_cart_count
    
    # Calculate combined score with consistent weighting across all models
    purchase_weight = 0.6  # Purchase events are most important
    checkout_weight = 0.3  # Checkout events are second most important
    cart_weight = 0.1      # Add-to-cart events are least important
    
    # Calculate combined score for each k value
    for k in top_k_values:
        # Overall weighted combined score
        purchase_recall = metrics[f'purchase_recall@{k}']
        checkout_recall = metrics[f'checkout_recall@{k}']
        cart_recall = metrics[f'add_to_cart_recall@{k}']
        
        # Calculate weighted combined score
        metrics[f'combined_score@{k}'] = (
            purchase_recall * purchase_weight +
            checkout_recall * checkout_weight +
            cart_recall * cart_weight
        )
        
        # Cold-start weighted combined score
        cold_purchase_recall = metrics[f'cold_start_purchase_recall@{k}']
        cold_checkout_recall = metrics[f'cold_start_checkout_recall@{k}']
        cold_cart_recall = metrics[f'cold_start_add_to_cart_recall@{k}']
        
        metrics[f'cold_start_combined_score@{k}'] = (
            cold_purchase_recall * purchase_weight +
            cold_checkout_recall * checkout_weight +
            cold_cart_recall * cart_weight
        )
        
        # Warm-start weighted combined score
        warm_purchase_recall = metrics[f'warm_start_purchase_recall@{k}']
        warm_checkout_recall = metrics[f'warm_start_checkout_recall@{k}']
        warm_cart_recall = metrics[f'warm_start_add_to_cart_recall@{k}']
        
        metrics[f'warm_start_combined_score@{k}'] = (
            warm_purchase_recall * purchase_weight +
            warm_checkout_recall * checkout_weight +
            warm_cart_recall * cart_weight
        )
    
    # For backwards compatibility, also keep the original combined_score
    metrics['combined_score'] = metrics['combined_score@10']
    
    # Print metrics
    print(f"\n{model_name} Evaluation Results:")
    
    # Print metrics for each k value
    for k in top_k_values:
        print(f"\n  === Recall@{k} Metrics ===")
        print(f"  Overall Recall@{k}: {metrics[f'overall_recall@{k}']*100:.2f}%")
        print(f"  Purchase Recall@{k}: {metrics[f'purchase_recall@{k}']*100:.2f}%")
        
        # Print cold-start vs. warm-start metrics
        print(f"\n  --- User Type Breakdown ---")
        print(f"  Cold-start Overall Recall@{k}: {metrics[f'cold_start_overall_recall@{k}']*100:.2f}% ({cold_start_count} users)")
        print(f"  Warm-start Overall Recall@{k}: {metrics[f'warm_start_overall_recall@{k}']*100:.2f}% ({warm_start_count} users)")
        print(f"  Cold-start Purchase Recall@{k}: {metrics[f'cold_start_purchase_recall@{k}']*100:.2f}% ({cold_start_purchase_count} purchases)")
        print(f"  Warm-start Purchase Recall@{k}: {metrics[f'warm_start_purchase_recall@{k}']*100:.2f}% ({warm_start_purchase_count} purchases)")
        print(f"  Cold-start Combined Score@{k}: {metrics[f'cold_start_combined_score@{k}']*100:.2f}%")
        print(f"  Warm-start Combined Score@{k}: {metrics[f'warm_start_combined_score@{k}']*100:.2f}%")
        
        # Print event-specific metrics (for all models)
        print(f"\n  --- Event Type Breakdown ---")
        print(f"  Checkout Recall@{k}: {metrics[f'checkout_recall@{k}']*100:.2f}%")
        print(f"  Add-to-cart Recall@{k}: {metrics[f'add_to_cart_recall@{k}']*100:.2f}%")
        
        print(f"\n  Combined Score@{k}: {metrics[f'combined_score@{k}']*100:.2f}%")
    
    # Print overall recommendation
    print(f"\n  === Overall Recommendation ===")
    print(f"  Best Combined Score: {max([metrics[f'combined_score@{k}'] for k in top_k_values])*100:.2f}%")
    print(f"  Best Purchase Recall: {max([metrics[f'purchase_recall@{k}'] for k in top_k_values])*100:.2f}%")
    
    # Clear cache
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return metrics


def identify_event_types(samples, session_processor):
    """Identify different event types in samples for contrastive approach"""
    print("\nIdentifying event types in samples...")
    
    # Get event type indices
    event_to_idx = session_processor.get_idx_mappings().get('event_to_idx', {})
    checkout_idx = event_to_idx.get('InitiateCheckout', -1)
    add_to_cart_idx = event_to_idx.get('AddToCart', -1)
    
    # If not found by name, try by index based on typical values
    if checkout_idx == -1:
        checkout_idx = 3  # Often InitiateCheckout is index 3
    if add_to_cart_idx == -1:
        add_to_cart_idx = 2  # Often AddToCart is index 2
    
    print(f"InitiateCheckout event index: {checkout_idx}")
    print(f"AddToCart event index: {add_to_cart_idx}")
    
    # Count samples by event type
    checkout_only_count = 0
    add_to_cart_only_count = 0
    purchase_checkout_both = 0
    purchase_add_to_cart_both = 0
    
    # Process all samples
    enhanced_samples = []
    for sample in samples:
        new_sample = sample.copy()
        
        # Get event types from session or short_term_events
        session = sample.get('session', [])
        session_event_types = [event.get('event_type') for event in session]
        
        # Also check short_term_events which is more likely to be present
        short_term_events = sample.get('short_term_events', [])
        
        # Combine event sources for more reliable detection
        event_types = session_event_types + short_term_events
        
        # Set flags
        has_checkout = checkout_idx in event_types
        has_add_to_cart = add_to_cart_idx in event_types
        is_purchase = sample.get('is_purchase', False)
        
        # Add event flags to sample
        new_sample['has_checkout'] = has_checkout
        new_sample['has_add_to_cart'] = has_add_to_cart
        
        # Update counts
        if has_checkout and not is_purchase:
            checkout_only_count += 1
        if has_add_to_cart and not is_purchase:
            add_to_cart_only_count += 1
        if is_purchase and has_checkout:
            purchase_checkout_both += 1
        if is_purchase and has_add_to_cart:
            purchase_add_to_cart_both += 1
        
        enhanced_samples.append(new_sample)
    
    # Print stats
    print(f"Identified {checkout_only_count} checkout-only samples")
    print(f"Identified {add_to_cart_only_count} add-to-cart-only samples")
    print(f"Found {purchase_checkout_both} samples with both purchase and checkout events")
    print(f"Found {purchase_add_to_cart_both} samples with both purchase and add-to-cart events")
    
    return enhanced_samples


class ContrastiveAwareDataset(torch.utils.data.Dataset):
    """Dataset that handles different event types for contrastive approach"""
    def __init__(self, samples, package_processor, user_to_idx, package_to_idx, event_to_idx,
                 max_short_term=10, max_long_term=20, use_cache=True, prefetch_features=True):
        self.samples = samples
        self.package_processor = package_processor
        self.user_to_idx = user_to_idx
        self.package_to_idx = package_to_idx
        self.event_to_idx = event_to_idx
        self.max_short_term = max_short_term
        self.max_long_term = max_long_term
        self.use_cache = use_cache
        self.cache_dir = os.path.join('data', 'cache')
        
        # Prefetch features
        if prefetch_features and package_processor:
            print("Prefetching common package features...")
            self.package_features = package_processor.prepare_package_tensors()
        else:
            self.package_features = None
        
        # Calculate cache path for faster loading
        if self.use_cache:
            import hashlib
            cache_hash = hashlib.md5(str(len(samples)).encode()).hexdigest()[:10]
            self.cache_path = os.path.join(self.cache_dir, f'dataset_{cache_hash}.pkl')
            
            # Try to load from cache
            try:
                if os.path.exists(self.cache_path):
                    print(f"Loading cached dataset...")
                    self.prepared_samples = torch.load(self.cache_path)
                    print(f"Loaded {len(self.prepared_samples)} samples from cache")
                else:
                    raise FileNotFoundError("Cache not found")
            except (EOFError, FileNotFoundError) as e:
                print(f"Cache error ({str(e)}), regenerating dataset...")
                self.prepared_samples = self._prepare_all_samples()
                
                # Save to cache
                os.makedirs(self.cache_dir, exist_ok=True)
                try:
                    torch.save(self.prepared_samples, self.cache_path)
                    print(f"Saved dataset cache: {self.cache_path}")
                except Exception as e:
                    print(f"Warning: Could not save cache - {str(e)}")
    
    def _prepare_all_samples(self):
        """Prepare all samples in advance"""
        prepared = []
        for i, sample in enumerate(tqdm(self.samples, desc="Preparing samples")):
            prepared.append(self._prepare_sample(sample))
        return prepared
    
    def _prepare_sample(self, sample):
        """Transform a sample to tensor format"""
        # User ID
        user_id = sample.get('user_id')
        user_idx = self.user_to_idx.get(user_id, 0)  # Default to 0 if not found (cold-start user)
        
        # Short-term packages
        short_term_packages = sample.get('short_term_packages', [])[:self.max_short_term]
        short_term_events = sample.get('short_term_events', [])[:self.max_short_term]
        short_term_timestamps = sample.get('short_term_timestamps', [])[:self.max_short_term]
        
        # Long-term packages
        long_term_packages = sample.get('long_term_packages', [])[:self.max_long_term]
        long_term_events = sample.get('long_term_events', [])[:self.max_long_term]
        long_term_timestamps = sample.get('long_term_timestamps', [])[:self.max_long_term]
        
        # Purchased package
        purchased_package = sample.get('purchased_package')
        
        # Convert to indices
        short_term_indices = [self.package_to_idx.get(pkg, 0) for pkg in short_term_packages]
        long_term_indices = [self.package_to_idx.get(pkg, 0) for pkg in long_term_packages]
        purchased_idx = self.package_to_idx.get(purchased_package, 0)
        
        # Pad sequences
        short_term_indices = self._pad_sequence(short_term_indices, self.max_short_term)
        short_term_events = self._pad_sequence(short_term_events, self.max_short_term)
        short_term_timestamps = self._pad_sequence(short_term_timestamps, self.max_short_term)
        
        long_term_indices = self._pad_sequence(long_term_indices, self.max_long_term)
        long_term_events = self._pad_sequence(long_term_events, self.max_long_term)
        long_term_timestamps = self._pad_sequence(long_term_timestamps, self.max_long_term)
        
        # Create tensors
        tensors = {
            'user_id': torch.tensor(user_idx, dtype=torch.long),
            'short_term': {
                'package_ids': torch.tensor(short_term_indices, dtype=torch.long),
                'event_types': torch.tensor(short_term_events, dtype=torch.long),
                'timestamps': torch.tensor(short_term_timestamps, dtype=torch.float) if short_term_timestamps else None,
            },
            'long_term': {
                'package_ids': torch.tensor(long_term_indices, dtype=torch.long),
                'event_types': torch.tensor(long_term_events, dtype=torch.long),
                'timestamps': torch.tensor(long_term_timestamps, dtype=torch.float) if long_term_timestamps else None,
            },
            'purchased': {
                'package_ids': torch.tensor(purchased_idx, dtype=torch.long).view(-1),
            }
        }
        
        # Add flags for contrastive approach
        is_purchase = sample.get('is_purchase', False)
        has_checkout = sample.get('has_checkout', False)
        has_add_to_cart = sample.get('has_add_to_cart', False)
        
        tensors['is_purchase'] = torch.tensor(is_purchase, dtype=torch.bool)
        tensors['has_checkout'] = torch.tensor(has_checkout, dtype=torch.bool)
        tensors['has_add_to_cart'] = torch.tensor(has_add_to_cart, dtype=torch.bool)
        
        return tensors
    
    def _pad_sequence(self, sequence, max_length):
        """Pad sequence to max_length"""
        if len(sequence) >= max_length:
            return sequence[:max_length]
        else:
            return sequence + [0] * (max_length - len(sequence))
    
    def __len__(self):
        return len(self.prepared_samples)
    
    def __getitem__(self, idx):
        """Get item with package features"""
        sample = self.prepared_samples[idx]
        
        if self.package_features is not None:
            # Add package features for short-term sequence
            short_term_ids = sample['short_term']['package_ids']
            
            # Add package features for short-term
            sample['short_term'].update(self._get_package_features(short_term_ids))
            
            # Add package features for long-term
            long_term_ids = sample['long_term']['package_ids']
            sample['long_term'].update(self._get_package_features(long_term_ids))
            
            # Add package features for purchased
            purchased_id = sample['purchased']['package_ids']
            sample['purchased'].update(self._get_package_features(purchased_id))
        
        return sample
    
    def _get_package_features(self, package_ids):
        """Get features for package IDs"""
        # If no package processor, return empty features
        if self.package_features is None:
            return {}
        
        # Extract features from package_features
        result = {}
        
        # Title embeddings
        title_embeddings = self.package_features['title_embeddings']
        result['title_embeddings'] = self._gather_features(title_embeddings, package_ids)
        
        # Coordinates
        coordinates = self.package_features['coordinates']
        result['coordinates'] = self._gather_features(coordinates, package_ids)
        
        # Categories
        country_ids = self.package_features['country_ids']
        category_ids = self.package_features['category_ids']
        theme_ids = self.package_features['theme_ids']
        
        result['country_ids'] = self._gather_features(country_ids, package_ids)
        result['category_ids'] = self._gather_features(category_ids, package_ids)
        result['theme_ids'] = self._gather_features(theme_ids, package_ids)
        
        # Prices if available
        if 'prices' in self.package_features:
            prices = self.package_features['prices']
            result['prices'] = self._gather_features(prices, package_ids)
        
        return result
    
    def _gather_features(self, features, indices):
        """Gather features for indices"""
        if isinstance(indices, torch.Tensor) and indices.dim() == 0:
            # Single index
            return features[indices].unsqueeze(0)
        else:
            # Multiple indices
            return torch.stack([features[idx] for idx in indices])


def create_weighted_sampler(dataset, purchase_boost=15.0, checkout_boost=7.5, 
                            add_to_cart_boost=3.0, view_weight=1.0):
    """Create weighted sampler to balance different event types"""
    print("Using weighted sampling:")
    print(f"  Purchase samples boost: {purchase_boost}")
    print(f"  Checkout samples boost: {checkout_boost}")
    print(f"  Add-to-cart samples boost: {add_to_cart_boost}")
    print(f"  View samples weight: {view_weight}")
    
    # Get weights for each sample
    weights = []
    for i in range(len(dataset)):
        sample = dataset.prepared_samples[i]
        
        if sample['is_purchase'].item():
            weights.append(purchase_boost)
        elif sample['has_checkout'].item():
            weights.append(checkout_boost)
        elif sample['has_add_to_cart'].item():
            weights.append(add_to_cart_boost)
        else:
            weights.append(view_weight)
    
    # Create sampler
    return torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )


def create_dataloaders(train_samples, test_samples, package_processor, user_to_idx, 
                       package_to_idx, event_to_idx, batch_size=64, num_workers=0,
                       use_weighted_sampling=True):
    """Create dataloaders for training and evaluation"""
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
        prefetch_features=True
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
        prefetch_features=True
    )
    
    # Create samplers
    train_sampler = None
    if use_weighted_sampling:
        train_sampler = create_weighted_sampler(train_dataset)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=False  # Avoid pin_memory issues with MPS
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size * 2,  # Larger batch size for evaluation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False  # Avoid pin_memory issues with MPS
    )
    
    return {'train': train_loader, 'test': test_loader}


def create_model(package_processor, user_count, package_count, device, 
                 embed_dim=128, hidden_dim=192):
    """Create NATR model"""
    # Get tensors for dimension info
    package_tensors = package_processor.prepare_package_tensors()
    embedding_dim = package_tensors['title_embeddings'].shape[1]
    
    # Create config
    config = NATRConfig(
        num_users=user_count,
        num_packages=package_count,
        num_countries=len(package_processor.country_to_idx),
        num_categories=len(package_processor.category_to_idx),
        num_themes=len(package_processor.theme_to_idx),
        title_embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embed_dim,
        user_embedding_dim=embed_dim,
        dropout=0.2,
        max_short_term=10,
        max_long_term=20
    )
    
    # Create model
    model = NATR(config)
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    print(f"Title embedding dimension: {embedding_dim}")
    print(f"Using embedding dimension: {embed_dim}")
    print(f"Using hidden dimension: {hidden_dim}")
    
    return model


def create_loss_functions(device):
    """Create all loss functions for testing"""
    # Standard focal loss
    focal_loss = FocalLoss(
        purchase_boost=10.0,
        gamma=2.0,
        alpha=0.25
    ).to(device)
    
    # Pre-training loss (lower purchase boost)
    pretrain_loss = FocalLoss(
        purchase_boost=1.0,
        gamma=1.5,
        alpha=0.25,
        adaptive_gamma=False
    ).to(device)
    
    # Fine-tuning loss (higher purchase boost)
    finetune_loss = FocalLoss(
        purchase_boost=20.0,
        gamma=2.5,
        alpha=0.3,
        adaptive_gamma=True
    ).to(device)
    
    # Checkout-enhanced loss
    checkout_loss = CheckoutEnhancedLoss(
        purchase_boost=8.0,
        checkout_boost=4.0,
        add_to_cart_boost=2.0
    ).to(device)
    
    # Contrastive loss
    contrastive_loss = ContrastiveEventLoss(
        temperature=0.1,
        purchase_margin=0.0,
        checkout_margin=0.3,
        add_to_cart_margin=0.6,
        view_margin=1.0,
        hard_negative_mining=True
    ).to(device)
    
    return {
        'focal': focal_loss,
        'pretrain': pretrain_loss,
        'finetune': finetune_loss,
        'checkout': checkout_loss,
        'contrastive': contrastive_loss
    }


def create_next_item_samples(train_samples):
    """Create next-item prediction samples for pre-training phase"""
    print("Creating next-item prediction samples...")
    next_item_samples = []
    
    for sample in train_samples:
        # Get short-term packages
        short_term_pkgs = sample.get('short_term_packages', [])
        short_term_events = sample.get('short_term_events', [])
        short_term_timestamps = sample.get('short_term_timestamps', [])
        
        if len(short_term_pkgs) >= 2:  # Need at least 2 items
            for i in range(len(short_term_pkgs) - 1):
                # Create a new sample with target as the next item
                new_sample = sample.copy()
                new_sample['short_term_packages'] = short_term_pkgs[:i+1]
                new_sample['short_term_events'] = short_term_events[:i+1] if short_term_events else []
                new_sample['short_term_timestamps'] = short_term_timestamps[:i+1] if short_term_timestamps else []
                new_sample['purchased_package'] = short_term_pkgs[i+1]
                new_sample['is_purchase'] = False
                new_sample['is_next_item'] = True
                next_item_samples.append(new_sample)
    
    print(f"Created {len(next_item_samples)} next-item prediction samples")
    
    # Limit if too many
    if len(next_item_samples) > 20000:
        import random
        random.seed(42)
        next_item_samples = random.sample(next_item_samples, 20000)
        print(f"Limited to 20,000 samples for faster processing")
    
    return next_item_samples


def test_standard_model(package_processor, train_loader, test_loader, mappings,
                        device, args, loss_functions):
    """Test standard (focal loss) model"""
    print("\n===== Testing Standard (Focal Loss) Model =====")
    
    # Create model
    model = create_model(
        package_processor,
        len(mappings['user_to_idx']),
        len(mappings['package_to_idx']),
        device,
        args.embed_dim,
        args.hidden_dim
    )
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    # Train model
    history = train_model(
        model, 
        train_loader, 
        optimizer, 
        loss_functions['focal'], 
        device,
        epochs=args.epochs,
        max_batches=args.max_batches,
        model_name="Standard",
        accumulation_steps=args.accumulation_steps
    )
    
    # Evaluate model
    metrics = evaluate_model(
        model, 
        test_loader, 
        device,
        max_batches=args.eval_batches,
        model_name="Standard"
    )
    
    # Move model to CPU to free up GPU memory
    model = model.to('cpu')
    torch.cuda.empty_cache() if device.type == 'cuda' else torch.mps.empty_cache() if device.type == 'mps' else None
    
    return {
        'model_name': "Standard (Focal Loss)",
        'history': history,
        'metrics': metrics
    }


def test_pretrain_finetune_model(package_processor, dataloaders, mappings,
                                device, args, loss_functions, train_samples):
    """Test pretrain+finetune approach"""
    print("\n===== Testing Pretrain+Finetune Model =====")
    
    # Create next-item samples for pre-training
    next_item_samples = create_next_item_samples(train_samples)
    
    # Create dataloaders for pre-training
    pretrain_loader = create_dataloaders(
        next_item_samples,
        [],  # No test needed for pre-training
        package_processor,
        mappings['user_to_idx'],
        mappings['package_to_idx'],
        mappings['event_to_idx'],
        batch_size=args.batch_size,
        use_weighted_sampling=False
    )['train']
    
    # Create model
    model = create_model(
        package_processor,
        len(mappings['user_to_idx']),
        len(mappings['package_to_idx']),
        device,
        args.embed_dim,
        args.hidden_dim
    )
    
    # Create optimizer for pre-training
    pretrain_optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    # Pre-train model
    print("Pre-training phase...")
    pretrain_history = train_model(
        model, 
        pretrain_loader, 
        pretrain_optimizer, 
        loss_functions['pretrain'], 
        device,
        epochs=1,  # Just one epoch for pre-training
        max_batches=args.max_batches,
        model_name="Pretrain",
        accumulation_steps=args.accumulation_steps
    )
    
    # Create optimizer for fine-tuning (lower learning rate)
    finetune_optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate * 0.1, weight_decay=0.03)
    
    # Fine-tune model
    print("Fine-tuning phase...")
    finetune_history = train_model(
        model, 
        dataloaders['train'], 
        finetune_optimizer, 
        loss_functions['finetune'], 
        device,
        epochs=args.epochs,
        max_batches=args.max_batches,
        model_name="Finetune",
        accumulation_steps=args.accumulation_steps
    )
    
    # Evaluate model
    metrics = evaluate_model(
        model, 
        dataloaders['test'], 
        device,
        max_batches=args.eval_batches,
        model_name="Pretrain+Finetune"
    )
    
    # Move model to CPU to free up GPU memory
    model = model.to('cpu')
    torch.cuda.empty_cache() if device.type == 'cuda' else torch.mps.empty_cache() if device.type == 'mps' else None
    
    # Combine histories
    history = {
        'pretrain_loss': pretrain_history['train_loss'],
        'finetune_loss': finetune_history['train_loss']
    }
    
    return {
        'model_name': "Pretrain+Finetune",
        'history': history,
        'metrics': metrics
    }


def test_checkout_enhanced_model(package_processor, dataloaders, mappings,
                                device, args, loss_functions):
    """Test checkout enhanced model"""
    print("\n===== Testing Checkout Enhanced Model =====")
    
    # Create model
    model = create_model(
        package_processor,
        len(mappings['user_to_idx']),
        len(mappings['package_to_idx']),
        device,
        args.embed_dim,
        args.hidden_dim
    )
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    # Train model
    history = train_model(
        model, 
        dataloaders['train'], 
        optimizer, 
        loss_functions['checkout'], 
        device,
        epochs=args.epochs,
        max_batches=args.max_batches,
        model_name="CheckoutEnhanced",
        accumulation_steps=args.accumulation_steps
    )
    
    # Evaluate model
    metrics = evaluate_model(
        model, 
        dataloaders['test'], 
        device,
        max_batches=args.eval_batches,
        model_name="CheckoutEnhanced"
    )
    
    # Move model to CPU to free up GPU memory
    model = model.to('cpu')
    torch.cuda.empty_cache() if device.type == 'cuda' else torch.mps.empty_cache() if device.type == 'mps' else None
    
    return {
        'model_name': "Checkout Enhanced",
        'history': history,
        'metrics': metrics
    }


def test_contrastive_model(package_processor, dataloaders, mappings,
                          device, args, loss_functions):
    """Test contrastive model"""
    print("\n===== Testing Contrastive Model =====")
    
    # Create model
    model = create_model(
        package_processor,
        len(mappings['user_to_idx']),
        len(mappings['package_to_idx']),
        device,
        args.embed_dim,
        args.hidden_dim
    )
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    # Train model
    history = train_model(
        model, 
        dataloaders['train'], 
        optimizer, 
        loss_functions['contrastive'], 
        device,
        epochs=args.epochs,
        max_batches=args.max_batches,
        model_name="Contrastive",
        accumulation_steps=args.accumulation_steps
    )
    
    # Evaluate model
    metrics = evaluate_model(
        model, 
        dataloaders['test'], 
        device,
        max_batches=args.eval_batches,
        model_name="Contrastive"
    )
    
    # Move model to CPU to free up GPU memory
    model = model.to('cpu')
    torch.cuda.empty_cache() if device.type == 'cuda' else torch.mps.empty_cache() if device.type == 'mps' else None
    
    return {
        'model_name': "Contrastive",
        'history': history,
        'metrics': metrics
    }


def plot_results(results, save_path):
    """Plot comprehensive comparison results with cold-start analysis"""
    # Create a multi-page plot
    from matplotlib.backends.backend_pdf import PdfPages
    
    # Initialize PDF
    pdf = PdfPages(save_path.replace('.png', '.pdf'))
    
    # Create a special plot for cold-start vs warm-start combined scores
    plt.figure(figsize=(15, 8))
    plt.suptitle("Cold-start vs Warm-start Combined Score Comparison", fontsize=16)
    
    # Prepare data
    model_names = [result['model_name'] for result in results]
    
    # k=10 Combined Score Comparison
    plt.subplot(1, 2, 1)
    x = np.arange(len(model_names))
    width = 0.35
    
    cold_combined_10 = [result['metrics']['cold_start_combined_score@10'] * 100 for result in results]
    warm_combined_10 = [result['metrics']['warm_start_combined_score@10'] * 100 for result in results]
    
    plt.bar(x - width/2, cold_combined_10, width, label='Cold-start Combined@10', color='lightblue')
    plt.bar(x + width/2, warm_combined_10, width, label='Warm-start Combined@10', color='darkblue')
    
    plt.title("Combined Score@10 by User Type")
    plt.ylabel("Percentage (%)")
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.ylim(bottom=0)
    
    # Add value labels
    for i, v in enumerate(cold_combined_10):
        plt.text(i - width/2, v + 0.5, f"{v:.1f}%", ha='center')
    
    for i, v in enumerate(warm_combined_10):
        plt.text(i + width/2, v + 0.5, f"{v:.1f}%", ha='center')
    
    # k=20 Combined Score Comparison
    plt.subplot(1, 2, 2)
    
    cold_combined_20 = [result['metrics']['cold_start_combined_score@20'] * 100 for result in results]
    warm_combined_20 = [result['metrics']['warm_start_combined_score@20'] * 100 for result in results]
    
    plt.bar(x - width/2, cold_combined_20, width, label='Cold-start Combined@20', color='lightgreen')
    plt.bar(x + width/2, warm_combined_20, width, label='Warm-start Combined@20', color='darkgreen')
    
    plt.title("Combined Score@20 by User Type")
    plt.ylabel("Percentage (%)")
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.ylim(bottom=0)
    
    # Add value labels
    for i, v in enumerate(cold_combined_20):
        plt.text(i - width/2, v + 0.5, f"{v:.1f}%", ha='center')
    
    for i, v in enumerate(warm_combined_20):
        plt.text(i + width/2, v + 0.5, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save to the PDF
    pdf.savefig()
    
    # Colors for different models
    colors = {
        "Standard (Focal Loss)": 'blue',
        "Pretrain+Finetune": 'purple',
        "Checkout Enhanced": 'green',
        "Contrastive": 'red'
    }
    
    # 1. Main comparison plot (similar to original but with both k=10 and k=20)
    plt.figure(figsize=(15, 10))
    
    # Calculate best models
    best_model_10 = max(results, key=lambda x: x['metrics']['combined_score@10'])
    best_purchase_10 = max(results, key=lambda x: x['metrics']['purchase_recall@10'])
    best_model_20 = max(results, key=lambda x: x['metrics']['combined_score@20'])
    best_purchase_20 = max(results, key=lambda x: x['metrics']['purchase_recall@20'])
    
    # Title for the whole figure
    plt.suptitle(f"Model Comparison Results\n" +
                f"Recommended (k=10): {best_model_10['model_name']} (Combined: {best_model_10['metrics']['combined_score@10']*100:.1f}%)\n" +
                f"Recommended (k=20): {best_model_20['model_name']} (Combined: {best_model_20['metrics']['combined_score@20']*100:.1f}%)", 
                fontsize=16)
    
    # Plot Recall@10 metrics in first row
    metrics_10 = ["purchase_recall@10", "overall_recall@10", "combined_score@10"]
    titles_10 = ["Purchase Recall@10", "Overall Recall@10", "Weighted Combined Score@10"]
    
    for i, (metric, title) in enumerate(zip(metrics_10, titles_10)):
        plt.subplot(2, 3, i+1)
        
        x = []
        y = []
        c = []
        
        for result in results:
            name = result['model_name']
            value = result['metrics'][metric] * 100  # Convert to percentage
            
            x.append(name)
            y.append(value)
            c.append(colors.get(name, 'gray'))
        
        plt.bar(x, y, color=c)
        plt.title(title)
        plt.ylabel("Percentage (%)")
        plt.xticks(rotation=45, ha='right')
        plt.ylim(bottom=0)
        
        # Add value labels on bars
        for i, v in enumerate(y):
            plt.text(i, v + 0.5, f"{v:.1f}%", ha='center')
    
    # Plot Recall@20 metrics in second row
    metrics_20 = ["purchase_recall@20", "overall_recall@20", "combined_score@20"]
    titles_20 = ["Purchase Recall@20", "Overall Recall@20", "Weighted Combined Score@20"]
    
    for i, (metric, title) in enumerate(zip(metrics_20, titles_20)):
        plt.subplot(2, 3, i+4)
        
        x = []
        y = []
        c = []
        
        for result in results:
            name = result['model_name']
            value = result['metrics'][metric] * 100  # Convert to percentage
            
            x.append(name)
            y.append(value)
            c.append(colors.get(name, 'gray'))
        
        plt.bar(x, y, color=c)
        plt.title(title)
        plt.ylabel("Percentage (%)")
        plt.xticks(rotation=45, ha='right')
        plt.ylim(bottom=0)
        
        # Add value labels on bars
        for i, v in enumerate(y):
            plt.text(i, v + 0.5, f"{v:.1f}%", ha='center')
    
    # Add text explaining weights used for combined score
    plt.figtext(0.5, 0.01, 
                "Combined Score = Purchase (60%) + Checkout (30%) + Add-to-cart (10%)", 
                ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)
    
    # Save the first figure to the PDF
    pdf.savefig()
    
    # 2. Cold-start vs Warm-start comparison (new plot)
    plt.figure(figsize=(15, 10))
    
    plt.suptitle("Cold-start vs Warm-start Performance Comparison", fontsize=16)
    
    # Plot cold vs warm performance for k=10
    plt.subplot(2, 2, 1)
    
    # Set up data for grouped bar chart
    model_names = [result['model_name'] for result in results]
    cold_start_values_10 = [result['metrics']['cold_start_overall_recall@10'] * 100 for result in results]
    warm_start_values_10 = [result['metrics']['warm_start_overall_recall@10'] * 100 for result in results]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, cold_start_values_10, width, label='Cold-start Overall Recall@10', color='lightblue')
    plt.bar(x + width/2, warm_start_values_10, width, label='Warm-start Overall Recall@10', color='orange')
    
    plt.title("Overall Recall@10")
    plt.ylabel("Percentage (%)")
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.ylim(bottom=0)
    
    # Add value labels
    for i, v in enumerate(cold_start_values_10):
        plt.text(i - width/2, v + 0.5, f"{v:.1f}%", ha='center')
    
    for i, v in enumerate(warm_start_values_10):
        plt.text(i + width/2, v + 0.5, f"{v:.1f}%", ha='center')
    
    # Plot cold vs warm performance for purchases k=10
    plt.subplot(2, 2, 2)
    
    # Set up data
    cold_purchase_values_10 = [result['metrics']['cold_start_purchase_recall@10'] * 100 for result in results]
    warm_purchase_values_10 = [result['metrics']['warm_start_purchase_recall@10'] * 100 for result in results]
    
    plt.bar(x - width/2, cold_purchase_values_10, width, label='Cold-start Purchase Recall@10', color='lightblue')
    plt.bar(x + width/2, warm_purchase_values_10, width, label='Warm-start Purchase Recall@10', color='orange')
    
    plt.title("Purchase Recall@10")
    plt.ylabel("Percentage (%)")
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.ylim(bottom=0)
    
    # Add value labels
    for i, v in enumerate(cold_purchase_values_10):
        plt.text(i - width/2, v + 0.5, f"{v:.1f}%", ha='center')
    
    for i, v in enumerate(warm_purchase_values_10):
        plt.text(i + width/2, v + 0.5, f"{v:.1f}%", ha='center')
    
    # Plot cold vs warm performance for k=20
    plt.subplot(2, 2, 3)
    
    # Set up data for grouped bar chart
    cold_start_values_20 = [result['metrics']['cold_start_overall_recall@20'] * 100 for result in results]
    warm_start_values_20 = [result['metrics']['warm_start_overall_recall@20'] * 100 for result in results]
    
    plt.bar(x - width/2, cold_start_values_20, width, label='Cold-start Overall Recall@20', color='lightblue')
    plt.bar(x + width/2, warm_start_values_20, width, label='Warm-start Overall Recall@20', color='orange')
    
    plt.title("Overall Recall@20")
    plt.ylabel("Percentage (%)")
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.ylim(bottom=0)
    
    # Add value labels
    for i, v in enumerate(cold_start_values_20):
        plt.text(i - width/2, v + 0.5, f"{v:.1f}%", ha='center')
    
    for i, v in enumerate(warm_start_values_20):
        plt.text(i + width/2, v + 0.5, f"{v:.1f}%", ha='center')
    
    # Plot cold vs warm performance for purchases k=20
    plt.subplot(2, 2, 4)
    
    # Set up data
    cold_purchase_values_20 = [result['metrics']['cold_start_purchase_recall@20'] * 100 for result in results]
    warm_purchase_values_20 = [result['metrics']['warm_start_purchase_recall@20'] * 100 for result in results]
    
    plt.bar(x - width/2, cold_purchase_values_20, width, label='Cold-start Purchase Recall@20', color='lightblue')
    plt.bar(x + width/2, warm_purchase_values_20, width, label='Warm-start Purchase Recall@20', color='orange')
    
    plt.title("Purchase Recall@20")
    plt.ylabel("Percentage (%)")
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.ylim(bottom=0)
    
    # Add value labels
    for i, v in enumerate(cold_purchase_values_20):
        plt.text(i - width/2, v + 0.5, f"{v:.1f}%", ha='center')
    
    for i, v in enumerate(warm_purchase_values_20):
        plt.text(i + width/2, v + 0.5, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the second figure to the PDF
    pdf.savefig()
    
    # 3. Event-specific comparison for contrastive models (checkout, add-to-cart)
    plt.figure(figsize=(15, 8))
    
    plt.suptitle("Event-specific Performance Comparison", fontsize=16)
    
    # Find models with event-specific metrics
    event_models = []
    for result in results:
        if "checkout_recall@10" in result['metrics']:
            event_models.append(result)
    
    if event_models:
        # Plot event performance for k=10
        plt.subplot(1, 2, 1)
        
        model_names = [result['model_name'] for result in event_models]
        purchase_values = [result['metrics']['purchase_recall@10'] * 100 for result in event_models]
        checkout_values = [result['metrics']['checkout_recall@10'] * 100 for result in event_models]
        cart_values = [result['metrics']['add_to_cart_recall@10'] * 100 for result in event_models]
        
        x = np.arange(len(model_names))
        width = 0.25
        
        plt.bar(x - width, purchase_values, width, label='Purchase Recall@10', color='green')
        plt.bar(x, checkout_values, width, label='Checkout Recall@10', color='lightblue')
        plt.bar(x + width, cart_values, width, label='Add-to-cart Recall@10', color='pink')
        
        plt.title("Event-specific Recall@10")
        plt.ylabel("Percentage (%)")
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.legend()
        plt.ylim(bottom=0)
        
        # Add value labels
        for i, v in enumerate(purchase_values):
            plt.text(i - width, v + 0.5, f"{v:.1f}%", ha='center')
        
        for i, v in enumerate(checkout_values):
            plt.text(i, v + 0.5, f"{v:.1f}%", ha='center')
            
        for i, v in enumerate(cart_values):
            plt.text(i + width, v + 0.5, f"{v:.1f}%", ha='center')
        
        # Plot event performance for k=20
        plt.subplot(1, 2, 2)
        
        purchase_values_20 = [result['metrics']['purchase_recall@20'] * 100 for result in event_models]
        checkout_values_20 = [result['metrics']['checkout_recall@20'] * 100 for result in event_models]
        cart_values_20 = [result['metrics']['add_to_cart_recall@20'] * 100 for result in event_models]
        
        plt.bar(x - width, purchase_values_20, width, label='Purchase Recall@20', color='green')
        plt.bar(x, checkout_values_20, width, label='Checkout Recall@20', color='lightblue')
        plt.bar(x + width, cart_values_20, width, label='Add-to-cart Recall@20', color='pink')
        
        plt.title("Event-specific Recall@20")
        plt.ylabel("Percentage (%)")
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.legend()
        plt.ylim(bottom=0)
        
        # Add value labels
        for i, v in enumerate(purchase_values_20):
            plt.text(i - width, v + 0.5, f"{v:.1f}%", ha='center')
        
        for i, v in enumerate(checkout_values_20):
            plt.text(i, v + 0.5, f"{v:.1f}%", ha='center')
            
        for i, v in enumerate(cart_values_20):
            plt.text(i + width, v + 0.5, f"{v:.1f}%", ha='center')
    else:
        plt.text(0.5, 0.5, "No models with event-specific metrics found", ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the third figure to the PDF
    pdf.savefig()
    
    # Close the PDF
    pdf.close()
    
    # Create a simplified PNG version for quick viewing
    plt.figure(figsize=(15, 10))
    
    # Show the main combined metrics all in one view
    plt.suptitle(f"Model Comparison Results - Combined View\n" +
                f"Best model (k=10): {best_model_10['model_name']} (Combined: {best_model_10['metrics']['combined_score@10']*100:.1f}%)\n" +
                f"Best model (k=20): {best_model_20['model_name']} (Combined: {best_model_20['metrics']['combined_score@20']*100:.1f}%)", 
                fontsize=16)
    
    # First subplot: Purchase Recall comparison
    plt.subplot(2, 2, 1)
    
    x = np.arange(len(results))
    width = 0.35
    
    purchase_values_10 = [result['metrics']['purchase_recall@10'] * 100 for result in results]
    purchase_values_20 = [result['metrics']['purchase_recall@20'] * 100 for result in results]
    
    plt.bar(x - width/2, purchase_values_10, width, label='Purchase Recall@10', color='blue')
    plt.bar(x + width/2, purchase_values_20, width, label='Purchase Recall@20', color='lightblue')
    
    plt.title("Purchase Recall")
    plt.ylabel("Percentage (%)")
    plt.xticks(x, [result['model_name'] for result in results], rotation=45, ha='right')
    plt.legend()
    plt.ylim(bottom=0)
    
    # Second subplot: Cold vs Warm start purchase recall
    plt.subplot(2, 2, 2)
    
    cold_purchase_10 = [result['metrics']['cold_start_purchase_recall@10'] * 100 for result in results]
    warm_purchase_10 = [result['metrics']['warm_start_purchase_recall@10'] * 100 for result in results]
    
    plt.bar(x - width/2, cold_purchase_10, width, label='Cold-start Purchase@10', color='orange')
    plt.bar(x + width/2, warm_purchase_10, width, label='Warm-start Purchase@10', color='red')
    
    plt.title("Cold vs Warm Start Purchase Recall@10")
    plt.ylabel("Percentage (%)")
    plt.xticks(x, [result['model_name'] for result in results], rotation=45, ha='right')
    plt.legend()
    plt.ylim(bottom=0)
    
    # Add value labels
    for i, v in enumerate(cold_purchase_10):
        plt.text(i - width/2, v + 0.5, f"{v:.1f}%", ha='center')
    
    for i, v in enumerate(warm_purchase_10):
        plt.text(i + width/2, v + 0.5, f"{v:.1f}%", ha='center')
    
    # Third subplot: Combined Score comparison
    plt.subplot(2, 2, 3)
    
    combined_values_10 = [result['metrics']['combined_score@10'] * 100 for result in results]
    combined_values_20 = [result['metrics']['combined_score@20'] * 100 for result in results]
    
    plt.bar(x - width/2, combined_values_10, width, label='Combined Score@10', color='green')
    plt.bar(x + width/2, combined_values_20, width, label='Combined Score@20', color='lightgreen')
    
    plt.title("Combined Score")
    plt.ylabel("Percentage (%)")
    plt.xticks(x, [result['model_name'] for result in results], rotation=45, ha='right')
    plt.legend()
    plt.ylim(bottom=0)
    
    # Fourth subplot: Purchase Recall Improvement from k=10 to k=20
    plt.subplot(2, 2, 4)
    
    improvement = [result['metrics']['purchase_recall@20'] * 100 - result['metrics']['purchase_recall@10'] * 100 for result in results]
    
    plt.bar(x, improvement, color='purple')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.title("Purchase Recall Improvement (k=10 to k=20)")
    plt.ylabel("Percentage Points")
    plt.xticks(x, [result['model_name'] for result in results], rotation=45, ha='right')
    plt.ylim(bottom=min(min(improvement)-1, 0))
    
    # Add value labels
    for i, v in enumerate(improvement):
        plt.text(i, v + 0.5 if v >= 0 else v - 1, f"{v:.1f}pp", ha='center')
    
    # Add text explaining weights used for combined score and cold-start definition
    plt.figtext(0.5, 0.02, 
                "Combined Score = Purchase (60%) + Checkout (30%) + Add-to-cart (10%)\n" +
                "Cold-start users: New users not seen during training (user_id=0)\n" +
                "Warm-start users: Existing users with history in training data (user_id>0)",
                ha="center", fontsize=9, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.22)
    
    # Save PNG figure
    plt.savefig(save_path)
    print(f"Results plot saved to {save_path}")
    print(f"Detailed PDF report saved to {save_path.replace('.png', '.pdf')}")


def clear_memory():
    """Aggressively clear GPU/MPS memory"""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Compare different NATR model approaches")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--embed-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--max-batches", type=int, default=50, help="Maximum batches per epoch (0 for all)")
    parser.add_argument("--eval-batches", type=int, default=20, help="Maximum batches for evaluation (0 for all)")
    parser.add_argument("--sample-limit", type=int, default=3000, help="Maximum number of samples to use (0 for all)")
    parser.add_argument("--accumulation-steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint", action="store_true", help="Save intermediate checkpoints for long runs")
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Print experiment parameters
    print("\n" + "="*50)
    print(f"COMPREHENSIVE MODEL COMPARISON PARAMETERS")
    print("="*50)
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Embedding dimension: {args.embed_dim}")
    print(f"Hidden dimension: {args.hidden_dim}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max batches: {'All' if args.max_batches == 0 else args.max_batches}")
    print(f"Evaluation batches: {'All' if args.eval_batches == 0 else args.eval_batches}")
    print(f"Sample limit: {'All' if args.sample_limit == 0 else args.sample_limit}")
    print(f"Gradient accumulation steps: {args.accumulation_steps}")
    print(f"Checkpointing: {'Enabled' if args.checkpoint else 'Disabled'}")
    print("="*50 + "\n")
    
    # Data paths
    package_data_path = "data/feed.parquet"
    event_data_path = "data/bookit_events_data_13_months.parquet"
    
    try:
        # Initialize processors
        package_processor = PackageProcessor(
            data_path=package_data_path,
            cache_dir="data/cache",
            load_coordinates=True,
            load_embeddings=True,
            api_key=os.environ.get("OPENAI_API_KEY"),
            embedding_model="text-embedding-3-small",
            use_reduced_embeddings=True  # Use reduced embeddings for memory efficiency
        )
        
        session_processor = SessionProcessor(
            data_path=event_data_path,
            cache_dir="data/cache",
            min_interactions=10,  # Updated from 5 to match individual model scripts
            max_sessions_per_user=10,
            max_samples_per_user=5
        )
        
        # Load data
        package_processor.load_data()
        session_processor.load_data()
        package_processor.create_mappings()
        session_processor.create_mappings()
        session_processor.extract_sessions()
        
        # Get samples
        samples = session_processor.prepare_enhanced_training_data()
        
        # Get mappings
        mappings = {
            "user_to_idx": session_processor.get_idx_mappings()["user_to_idx"],
            "package_to_idx": session_processor.get_idx_mappings()["package_to_idx"],
            "event_to_idx": session_processor.get_idx_mappings()["event_to_idx"]
        }
        
        # Identify event types for contrastive approach
        samples = identify_event_types(samples, session_processor)
        
        # Check if we have enough different event types
        purchase_samples = [s for s in samples if s.get('is_purchase', False)]
        checkout_samples = [s for s in samples if s.get('has_checkout', False) and not s.get('is_purchase', False)]
        add_to_cart_samples = [s for s in samples if s.get('has_add_to_cart', False) and not s.get('is_purchase', False) and not s.get('has_checkout', False)]
        view_samples = [s for s in samples if not s.get('is_purchase', False) and not s.get('has_checkout', False) and not s.get('has_add_to_cart', False)]
        
        print(f"Original distribution:")
        print(f"  Purchases: {len(purchase_samples)}")
        print(f"  Checkout: {len(checkout_samples)}")
        print(f"  Add-to-cart: {len(add_to_cart_samples)}")
        print(f"  View-only: {len(view_samples)}")
        
        # If we don't have any checkout or add-to-cart samples, create synthetic ones
        if len(checkout_samples) < 100 or len(add_to_cart_samples) < 100:
            print("Creating synthetic event samples for better comparison...")
            
            # Convert some purchase samples to also have checkout flags
            if len(checkout_samples) < 100 and len(purchase_samples) > 200:
                synthetic_checkout = purchase_samples[:200]
                for s in synthetic_checkout:
                    s['has_checkout'] = True
                checkout_samples = synthetic_checkout
                print(f"Created {len(checkout_samples)} synthetic checkout samples")
            
            # Convert some view samples to have add-to-cart flags
            if len(add_to_cart_samples) < 100 and len(view_samples) > 200:
                synthetic_cart = view_samples[:200]
                for s in synthetic_cart:
                    s['has_add_to_cart'] = True
                add_to_cart_samples = synthetic_cart
                print(f"Created {len(add_to_cart_samples)} synthetic add-to-cart samples")
        
        # Limit samples for faster processing if sample_limit > 0
        if args.sample_limit > 0 and len(samples) > args.sample_limit:
            print(f"Limiting samples from {len(samples)} to {args.sample_limit} for faster processing")
            # Keep a balanced subset
            max_per_type = args.sample_limit // 4  # 25% of each type
            
            if len(purchase_samples) > max_per_type:
                purchase_samples = purchase_samples[:max_per_type]
            
            if len(checkout_samples) > max_per_type:
                checkout_samples = checkout_samples[:max_per_type]
                
            if len(add_to_cart_samples) > max_per_type:
                add_to_cart_samples = add_to_cart_samples[:max_per_type]
                
            if len(view_samples) > max_per_type * 2:  # Allow more view samples
                view_samples = view_samples[:max_per_type * 2]
            
            # Combine samples
            samples = purchase_samples + checkout_samples + add_to_cart_samples + view_samples
            np.random.shuffle(samples)
        else:
            print(f"Using full dataset with {len(samples)} samples for comprehensive comparison")
        
        print(f"Using {len(samples)} samples for model comparison")
        print(f"  Purchases: {sum(1 for s in samples if s.get('is_purchase', False))}")
        print(f"  Checkout: {sum(1 for s in samples if s.get('has_checkout', False) and not s.get('is_purchase', False))}")
        print(f"  Add-to-cart: {sum(1 for s in samples if s.get('has_add_to_cart', False) and not s.get('is_purchase', False) and not s.get('has_checkout', False))}")
        
        # Split into train/test sets
        np.random.shuffle(samples)
        split_idx = int(len(samples) * 0.8)
        train_samples = samples[:split_idx]
        test_samples = samples[split_idx:]
        
        # Create dataloaders
        dataloaders = create_dataloaders(
            train_samples,
            test_samples,
            package_processor,
            mappings["user_to_idx"],
            mappings["package_to_idx"],
            mappings["event_to_idx"],
            batch_size=args.batch_size,
            use_weighted_sampling=True
        )
        
        # Create loss functions
        loss_functions = create_loss_functions(device)
        
        # Create results directory for checkpoints
        checkpoint_dir = os.path.join("output", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Test all models
        results = []
        
        # Function to save intermediate results
        def save_intermediate_results(current_results, filename):
            """Save intermediate results during long runs"""
            if args.checkpoint:
                save_path = os.path.join(checkpoint_dir, filename)
                try:
                    with open(save_path, 'w') as f:
                        json.dump({
                            'timestamp': timestamp,
                            'args': vars(args),
                            'device': str(device),
                            'results': [{
                                'model_name': r['model_name'],
                                'metrics': r['metrics']
                            } for r in current_results]
                        }, f, indent=2)
                    print(f"Saved intermediate results to {save_path}")
                except Exception as e:
                    print(f"Error saving checkpoint: {e}")
        
        # 1. Test standard (focal loss) model
        print("\n" + "="*30)
        print("STARTING STANDARD MODEL EVALUATION")
        print("="*30 + "\n")
        standard_results = test_standard_model(
            package_processor, dataloaders['train'], dataloaders['test'], 
            mappings, device, args, loss_functions
        )
        results.append(standard_results)
        save_intermediate_results([standard_results], f"checkpoint_standard_{timestamp}.json")
        
        # Clear memory
        clear_memory()
        
        # 2. Test pretrain+finetune model
        print("\n" + "="*30)
        print("STARTING PRETRAIN+FINETUNE MODEL EVALUATION")
        print("="*30 + "\n")
        pretrain_results = test_pretrain_finetune_model(
            package_processor, dataloaders, mappings, 
            device, args, loss_functions, train_samples
        )
        results.append(pretrain_results)
        save_intermediate_results([standard_results, pretrain_results], f"checkpoint_pretrain_{timestamp}.json")
        
        # Clear memory
        clear_memory()
        
        # 3. Test checkout enhanced model
        print("\n" + "="*30)
        print("STARTING CHECKOUT ENHANCED MODEL EVALUATION")
        print("="*30 + "\n")
        checkout_results = test_checkout_enhanced_model(
            package_processor, dataloaders, mappings, 
            device, args, loss_functions
        )
        results.append(checkout_results)
        save_intermediate_results([standard_results, pretrain_results, checkout_results], 
                                 f"checkpoint_checkout_{timestamp}.json")
        
        # Clear memory
        clear_memory()
        
        # 4. Test contrastive model
        print("\n" + "="*30)
        print("STARTING CONTRASTIVE MODEL EVALUATION")
        print("="*30 + "\n")
        contrastive_results = test_contrastive_model(
            package_processor, dataloaders, mappings, 
            device, args, loss_functions
        )
        results.append(contrastive_results)
        
        # Compare results
        print("\n===== Final Results =====")
        
        # Create comparison tables for different metrics
        print("\n=== Recall@10 Metrics ===")
        print(f"{'Model':<20} {'Overall':<10} {'Purchase':<10} {'Cold-start':<12} {'Warm-start':<12} {'Combined':<10}")
        print("-" * 80)
        
        for result in results:
            model_name = result['model_name']
            metrics = result['metrics']
            overall = metrics['overall_recall@10']*100
            purchase = metrics['purchase_recall@10']*100
            cold = metrics['cold_start_overall_recall@10']*100
            warm = metrics['warm_start_overall_recall@10']*100
            combined = metrics['combined_score@10']*100
            
            print(f"{model_name:<20} {overall:<10.2f}% {purchase:<10.2f}% {cold:<12.2f}% {warm:<12.2f}% {combined:<10.2f}%")
        
        print("\n=== Purchase Recall@10 and Combined Scores ===")
        print(f"{'Model':<20} {'Purchase':<10} {'Cold-start':<12} {'Warm-start':<12} {'Cold-comb':<10} {'Warm-comb':<10}")
        print("-" * 80)
        
        for result in results:
            model_name = result['model_name']
            metrics = result['metrics']
            purchase = metrics['purchase_recall@10']*100
            cold_purchase = metrics['cold_start_purchase_recall@10']*100
            warm_purchase = metrics['warm_start_purchase_recall@10']*100
            cold_combined = metrics['cold_start_combined_score@10']*100
            warm_combined = metrics['warm_start_combined_score@10']*100
            
            print(f"{model_name:<20} {purchase:<10.2f}% {cold_purchase:<12.2f}% {warm_purchase:<12.2f}% {cold_combined:<10.2f}% {warm_combined:<10.2f}%")
        
        print("\n=== Recall@20 Metrics ===")
        print(f"{'Model':<20} {'Overall':<10} {'Purchase':<10} {'Cold-start':<12} {'Warm-start':<12} {'Combined':<10}")
        print("-" * 80)
        
        for result in results:
            model_name = result['model_name']
            metrics = result['metrics']
            overall = metrics['overall_recall@20']*100
            purchase = metrics['purchase_recall@20']*100
            cold = metrics['cold_start_overall_recall@20']*100
            warm = metrics['warm_start_overall_recall@20']*100
            combined = metrics['combined_score@20']*100
            
            print(f"{model_name:<20} {overall:<10.2f}% {purchase:<10.2f}% {cold:<12.2f}% {warm:<12.2f}% {combined:<10.2f}%")
            
        print("\n=== Purchase Recall@20 and Combined Scores ===")
        print(f"{'Model':<20} {'Purchase':<10} {'Cold-start':<12} {'Warm-start':<12} {'Cold-comb':<10} {'Warm-comb':<10}")
        print("-" * 80)
        
        for result in results:
            model_name = result['model_name']
            metrics = result['metrics']
            purchase = metrics['purchase_recall@20']*100
            cold_purchase = metrics['cold_start_purchase_recall@20']*100
            warm_purchase = metrics['warm_start_purchase_recall@20']*100
            cold_combined = metrics['cold_start_combined_score@20']*100
            warm_combined = metrics['warm_start_combined_score@20']*100
            
            print(f"{model_name:<20} {purchase:<10.2f}% {cold_purchase:<12.2f}% {warm_purchase:<12.2f}% {cold_combined:<10.2f}% {warm_combined:<10.2f}%")
        
        # Print purchase improvement from k=10 to k=20
        print("\n=== Purchase Recall Improvement (k=10 to k=20) ===")
        print(f"{'Model':<20} {'k=10':<10} {'k=20':<10} {'Improvement':<15}")
        print("-" * 60)
        
        for result in results:
            model_name = result['model_name']
            metrics = result['metrics']
            purchase_10 = metrics['purchase_recall@10']*100
            purchase_20 = metrics['purchase_recall@20']*100
            improvement = purchase_20 - purchase_10
            
            print(f"{model_name:<20} {purchase_10:<10.2f}% {purchase_20:<10.2f}% {improvement:+<15.2f}%")
        
        # Cold-start vs warm-start specific analysis
        print("\n=== Cold-start vs Warm-start Purchase Recall@10 ===")
        print(f"{'Model':<20} {'Cold-start':<12} {'Warm-start':<12} {'Difference':<15}")
        print("-" * 60)
        
        for result in results:
            model_name = result['model_name']
            metrics = result['metrics']
            cold = metrics['cold_start_purchase_recall@10']*100
            warm = metrics['warm_start_purchase_recall@10']*100
            diff = warm - cold
            
            print(f"{model_name:<20} {cold:<12.2f}% {warm:<12.2f}% {diff:+<15.2f}%")
        
        # Event-specific analysis for all models
        print("\n=== Event-specific Recall@10 for All Models ===")
        print(f"{'Model':<20} {'Purchase':<10} {'Checkout':<10} {'Add-to-cart':<12}")
        print("-" * 60)
        
        for result in results:
            model_name = result['model_name']
            metrics = result['metrics']
            purchase = metrics['purchase_recall@10']*100
            checkout = metrics['checkout_recall@10']*100
            cart = metrics['add_to_cart_recall@10']*100
            
            print(f"{model_name:<20} {purchase:<10.2f}% {checkout:<10.2f}% {cart:<12.2f}%")
        
        # Determine best models
        best_model_10 = max(results, key=lambda x: x['metrics']['combined_score@10'])
        best_purchase_10 = max(results, key=lambda x: x['metrics']['purchase_recall@10'])
        best_model_20 = max(results, key=lambda x: x['metrics']['combined_score@20'])
        best_purchase_20 = max(results, key=lambda x: x['metrics']['purchase_recall@20'])
        
        # Best for cold-start users
        best_cold_start_purchase_10 = max(results, key=lambda x: x['metrics']['cold_start_purchase_recall@10'])
        best_cold_start_combined_10 = max(results, key=lambda x: x['metrics']['cold_start_combined_score@10'])
        
        # Best for warm-start users
        best_warm_start_purchase_10 = max(results, key=lambda x: x['metrics']['warm_start_purchase_recall@10'])
        best_warm_start_combined_10 = max(results, key=lambda x: x['metrics']['warm_start_combined_score@10'])
        
        print("\n=== Recommended Approaches ===")
        print(f"Best overall model (k=10): {best_model_10['model_name']} (Combined: {best_model_10['metrics']['combined_score@10']*100:.2f}%)")
        print(f"Best overall model (k=20): {best_model_20['model_name']} (Combined: {best_model_20['metrics']['combined_score@20']*100:.2f}%)")
        print(f"Best for purchase recall (k=10): {best_purchase_10['model_name']} ({best_purchase_10['metrics']['purchase_recall@10']*100:.2f}%)")
        print(f"Best for purchase recall (k=20): {best_purchase_20['model_name']} ({best_purchase_20['metrics']['purchase_recall@20']*100:.2f}%)")
        
        # Calculate the cold-start purchase improvement for each model
        cold_start_purchase_improvements = []
        for result in results:
            model_name = result['model_name']
            cold_purchase = result['metrics']['cold_start_purchase_recall@10']*100
            warm_purchase = result['metrics']['warm_start_purchase_recall@10']*100
            improvement = cold_purchase - warm_purchase
            cold_start_purchase_improvements.append((model_name, improvement, cold_purchase))
        
        # Find model with highest relative performance for cold-start users
        best_relative_for_cold = max(cold_start_purchase_improvements, key=lambda x: x[1])
        # Find model with highest absolute performance for cold-start users
        best_absolute_for_cold = max(cold_start_purchase_improvements, key=lambda x: x[2])
        
        print("\nCold-start recommendations:")
        print(f"Best for cold-start purchase (k=10): {best_cold_start_purchase_10['model_name']} ({best_cold_start_purchase_10['metrics']['cold_start_purchase_recall@10']*100:.2f}%)")
        print(f"Best for cold-start combined (k=10): {best_cold_start_combined_10['model_name']} ({best_cold_start_combined_10['metrics']['cold_start_combined_score@10']*100:.2f}%)")
        
        if best_relative_for_cold[1] > 0:
            print(f"Model that performs best relative to warm-start: {best_relative_for_cold[0]} ({best_relative_for_cold[1]:+.2f}% better for cold-start)")
        else:
            print(f"Note: All models perform better for warm-start than cold-start users")
            print(f"Model with smallest cold-start disadvantage: {best_relative_for_cold[0]} ({best_relative_for_cold[1]:+.2f}% difference)")
        
        print("\nWarm-start recommendations:")
        print(f"Best for warm-start purchase (k=10): {best_warm_start_purchase_10['model_name']} ({best_warm_start_purchase_10['metrics']['warm_start_purchase_recall@10']*100:.2f}%)")
        print(f"Best for warm-start combined (k=10): {best_warm_start_combined_10['model_name']} ({best_warm_start_combined_10['metrics']['warm_start_combined_score@10']*100:.2f}%)")
        
        # Print recommendation strategy
        print("\nRecommended Strategy for Production:")
        if best_cold_start_purchase_10['model_name'] == best_warm_start_purchase_10['model_name']:
            print(f"Use a single model approach: {best_model_10['model_name']} works well for both cold and warm-start users")
        else:
            print(f"Use a dual-model approach:")
            print(f"  - For cold-start users: {best_cold_start_purchase_10['model_name']}")
            print(f"  - For warm-start users: {best_warm_start_purchase_10['model_name']}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON results
        results_file = os.path.join(output_dir, f"model_comparison_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'args': vars(args),
                'device': str(device),
                'results': [{
                    'model_name': r['model_name'],
                    'metrics': r['metrics']
                } for r in results],
                'documentation': {
                    'cold_start_definition': 'Cold-start users have user_id=0, which happens when they are not in the training set (no historical interactions)',
                    'warm_start_definition': 'Warm-start users have user_id>0, which means they were seen in the training set and have historical interactions',
                    'combined_score': 'Weighted average of purchase (60%), checkout (30%), and add-to-cart (10%) recalls'
                },
                'recommendations': {
                    'best_overall_k10': {
                        'model': best_model_10['model_name'],
                        'combined_score': best_model_10['metrics']['combined_score@10']*100
                    },
                    'best_overall_k20': {
                        'model': best_model_20['model_name'],
                        'combined_score': best_model_20['metrics']['combined_score@20']*100
                    },
                    'best_purchase_k10': {
                        'model': best_purchase_10['model_name'],
                        'recall': best_purchase_10['metrics']['purchase_recall@10']*100
                    },
                    'best_purchase_k20': {
                        'model': best_purchase_20['model_name'],
                        'recall': best_purchase_20['metrics']['purchase_recall@20']*100
                    },
                    'cold_start': {
                        'best_purchase': {
                            'model': best_cold_start_purchase_10['model_name'],
                            'recall': best_cold_start_purchase_10['metrics']['cold_start_purchase_recall@10']*100
                        },
                        'best_combined': {
                            'model': best_cold_start_combined_10['model_name'],
                            'score': best_cold_start_combined_10['metrics']['cold_start_combined_score@10']*100
                        },
                        'best_relative_performance': {
                            'model': best_relative_for_cold[0],
                            'improvement_over_warm_start': best_relative_for_cold[1]
                        }
                    },
                    'warm_start': {
                        'best_purchase': {
                            'model': best_warm_start_purchase_10['model_name'],
                            'recall': best_warm_start_purchase_10['metrics']['warm_start_purchase_recall@10']*100
                        },
                        'best_combined': {
                            'model': best_warm_start_combined_10['model_name'],
                            'score': best_warm_start_combined_10['metrics']['warm_start_combined_score@10']*100
                        }
                    },
                    'production_strategy': (
                        {'type': 'single_model', 'model': best_model_10['model_name']} 
                        if best_cold_start_purchase_10['model_name'] == best_warm_start_purchase_10['model_name']
                        else {
                            'type': 'dual_model',
                            'cold_start_model': best_cold_start_purchase_10['model_name'],
                            'warm_start_model': best_warm_start_purchase_10['model_name']
                        }
                    )
                }
            }, f, indent=2)
        
        print(f"Results saved to {results_file}")
        
        # Plot results
        plot_file = os.path.join(output_dir, f"model_comparison_{timestamp}.png")
        plot_results(results, plot_file)
        
    except Exception as e:
        print(f"Error during model comparison: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
"""
Complete NATR training script with natural event learning
Optimized for computation time while ensuring functionality
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
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict, Counter
from torch.utils.data import DataLoader
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
from utils.loss_functions import NaturalPurchaseLoss, PurchaseFocusedMetrics


def filter_items_by_frequency(samples, min_frequency=5):
    """Filter items that appear less than min_frequency times
    
    A package must have at least 5 interactions to be included
    """
    print(f"\nFiltering items by frequency (min: {min_frequency})...")
    
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
    
    # Find valid packages
    valid_packages = {pkg for pkg, count in package_counts.items() if count >= min_frequency}
    print(f"Valid packages: {len(valid_packages)} out of {len(package_counts)}")
    
    # Filter samples efficiently
    filtered_samples = []
    for sample in samples:
        purchased = str(sample.get('purchased_package', ''))
        if purchased not in valid_packages:
            continue
        
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
    
    # Sort by timestamp
    samples_sorted = sorted(samples, key=lambda x: x.get('timestamp', 0))
    
    # Get time range
    first_time = samples_sorted[0]['timestamp']
    last_time = samples_sorted[-1]['timestamp']
    
    # Convert to datetime
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
        if pd.to_datetime(sample['timestamp']) < split_date:
            train_samples.append(sample)
        else:
            test_samples.append(sample)
    
    print(f"Train samples: {len(train_samples):,}")
    print(f"Test samples: {len(test_samples):,}")
    
    return train_samples, test_samples


def analyze_data_distribution(samples, name="Dataset"):
    """Analyze the distribution of data"""
    purchase_count = sum(1 for s in samples if s.get('is_purchase', False))
    
    print(f"\n{name} Distribution:")
    print(f"  Total samples: {len(samples):,}")
    print(f"  Purchases: {purchase_count:,} ({purchase_count/len(samples)*100:.2f}%)")
    print(f"  Non-purchases: {len(samples)-purchase_count:,}")


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
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
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
                targets = batch['purchased']['package_ids']  # Changed from package_id to package_ids
                is_purchase = batch.get('is_purchase', torch.zeros_like(targets, dtype=torch.bool))
                
                # Calculate loss
                loss = loss_fn(predictions, targets, is_purchase)
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
            targets = batch['purchased']['package_ids']  # Changed from package_id to package_ids
            is_purchase = batch.get('is_purchase', torch.zeros_like(targets, dtype=torch.bool))
            
            # Calculate loss
            loss = loss_fn(predictions, targets, is_purchase)
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
    """Evaluate model with memory optimization and MPS-specific enhancements"""
    model.eval()
    
    # Initialize metrics accumulators to avoid storing all predictions
    metrics_acc = defaultdict(float)
    total_samples = 0
    purchase_samples = 0
    
    # MPS-specific optimization
    is_mps = device.type == 'mps'
    
    with torch.no_grad():
        # Handle MPS garbage collection
        if is_mps:
            torch.mps.empty_cache()
        
        # Use smaller batches for evaluation on MPS to avoid OOM
        progress_bar = tqdm(test_loader, desc="Evaluating")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = move_batch_to_device(batch, device)
            
            # Forward pass
            outputs = model(batch)
            predictions = outputs['predictions']
            targets = batch['purchased']['package_ids']  # Fixed from package_id to package_ids
            is_purchase = batch.get('is_purchase', torch.zeros_like(targets, dtype=torch.bool))
            
            # Calculate metrics per batch to save memory
            batch_metrics = calculate_batch_metrics(
                predictions, targets, is_purchase, k_values
            )
            
            # Accumulate metrics
            batch_size = targets.size(0)
            batch_purchases = is_purchase.sum().item()
            
            for k in k_values:
                metrics_acc[f'recall@k_{k}'] += batch_metrics[f'recall@k_{k}'] * batch_size
                if batch_purchases > 0:
                    metrics_acc[f'purchase_recall@k_{k}'] += batch_metrics[f'purchase_recall@k_{k}'] * batch_purchases
            
            metrics_acc['overall_mrr'] += batch_metrics['overall_mrr'] * batch_size
            if batch_purchases > 0:
                metrics_acc['purchase_mrr'] += batch_metrics['purchase_mrr'] * batch_purchases
            
            total_samples += batch_size
            purchase_samples += batch_purchases
            
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
        'purchase_recall@k': {k: metrics_acc[f'purchase_recall@k_{k}'] / purchase_samples 
                             if purchase_samples > 0 else 0.0 for k in k_values},
        'overall_mrr': metrics_acc['overall_mrr'] / total_samples,
        'purchase_mrr': metrics_acc['purchase_mrr'] / purchase_samples if purchase_samples > 0 else 0.0,
        'total_count': total_samples,
        'purchase_count': purchase_samples
    }
    
    return metrics


def calculate_batch_metrics(predictions, targets, is_purchase, k_values):
    """Calculate metrics for a single batch efficiently"""
    batch_size = predictions.size(0)
    max_k = max(k_values)
    
    # Initialize metrics
    metrics = {}
    for k in k_values:
        metrics[f'recall@k_{k}'] = 0.0
        metrics[f'purchase_recall@k_{k}'] = 0.0
    metrics['overall_mrr'] = 0.0
    metrics['purchase_mrr'] = 0.0
    
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
        if is_purchase.sum() > 0:
            purchase_in_top_k = in_top_k & is_purchase
            metrics[f'purchase_recall@k_{k}'] = (
                purchase_in_top_k.float().sum() / is_purchase.sum().float()
            ).item() if is_purchase.sum() > 0 else 0.0
    
    # Calculate MRR (Mean Reciprocal Rank)
    ranks = torch.zeros_like(targets, dtype=torch.float)
    for i in range(batch_size):
        rank_positions = (top_indices[i] == targets[i]).nonzero(as_tuple=True)[0]
        if len(rank_positions) > 0:
            # +1 because ranks start from 1
            ranks[i] = 1.0 / (rank_positions[0].item() + 1)
    
    metrics['overall_mrr'] = ranks.mean().item()
    metrics['purchase_mrr'] = (
        ranks[is_purchase].mean().item() if is_purchase.sum() > 0 else 0.0
    )
    
    return metrics


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


def create_dataloaders(train_samples, test_samples, package_processor, session_processor, 
                      batch_size=32, num_workers=0):
    """Create optimized dataloaders for training and testing"""
    
    # Get mappings
    user_to_idx = session_processor.get_idx_mappings()['user_to_idx']
    package_to_idx = session_processor.get_idx_mappings()['package_to_idx']
    event_to_idx = session_processor.get_idx_mappings()['event_to_idx']
    
    # Create datasets
    print("Creating training dataset...")
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
    
    print("Creating test dataset...")
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
    
    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
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
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=3 if num_workers > 0 else None  # Increased prefetch factor
    )
    
    return train_loader, test_loader


def main(performance_config=None):
    """Main training function with performance configuration"""
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
        # Apple Silicon optimized settings
        # M3 Max has great memory bandwidth but fewer compute units than high-end NVIDIA GPUs
        base_batch_size = 64  # Smaller than CUDA but larger than CPU
        num_workers = 6       # M3 Max has good multi-core performance
    else:  # cuda
        base_batch_size = 128
        num_workers = 4
    
    batch_size = int(base_batch_size * performance_config.get("batch_size_multiplier", 1.0))
    learning_rate = performance_config.get("learning_rate", 0.001)
    num_epochs = performance_config.get("num_epochs", 30)
    accumulation_steps = performance_config.get("accumulation_steps", 1)
    
    print(f"\nTraining parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Gradient accumulation steps: {accumulation_steps}")
    print(f"  Workers: {num_workers}")
    print(f"  Use AMP: {performance_config.get('use_amp', False)}")
    print(f"  Compile model: {performance_config.get('compile_model', False)}")
    
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
    
    print("\n6. Applying data filters...")
    
    # Filter items by frequency
    filtered_samples, valid_packages = filter_items_by_frequency(samples, min_frequency=5)
    
    # Validate that we have purchase events
    purchase_count = sum(1 for s in filtered_samples if s.get('is_purchase', False))
    if purchase_count == 0:
        raise ValueError("No purchase events found in the filtered samples. Cannot train the model.")
    
    print(f"Purchase events: {purchase_count} ({purchase_count / len(filtered_samples) * 100:.2f}% of all events)")
    
    # Time-based split
    train_samples, test_samples = time_based_split_year(filtered_samples, train_ratio=0.93)
    
    # Analyze distributions
    analyze_data_distribution(train_samples, "Train")
    analyze_data_distribution(test_samples, "Test")
    
    # Create datasets and dataloaders
    print("\n7. Creating datasets...")
    train_loader, test_loader = create_dataloaders(
        train_samples, test_samples, package_processor, session_processor,
        batch_size=batch_size, num_workers=num_workers
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset):,} samples ({len(train_loader):,} batches)")
    print(f"  Test: {len(test_loader.dataset):,} samples ({len(test_loader):,} batches)")
    
    # Get model dimensions
    print("\n8. Creating model...")
    
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
    
    # Create model config
    config = NATRConfig(
        num_users=num_users,
        num_packages=num_packages,
        num_countries=num_countries,
        num_categories=num_categories,
        num_themes=num_themes,
        title_embedding_dim=3072,  # Fix: Always use 3072 as it's what your data has
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
    
    # Initialize optimizer and loss with additional optimizations
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,  # Add weight decay for regularization
        eps=1e-8  # More stable epsilon value
    )
    
    # Use a more efficient scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        patience=performance_config.get("patience", 3), 
        factor=0.5
    )
    print(f"  LR scheduler: ReduceLROnPlateau with patience {performance_config.get('patience', 3)}")
    
    loss_fn = NaturalPurchaseLoss(purchase_boost=20.0)
    
    # Print model configuration for debugging
    print(f"  Title embedding dimension: {config.title_embedding_dim}")
    print(f"  Hidden dimension: {config.hidden_dim}")
    print(f"  Embedding dimension: {config.embedding_dim}")
    print(f"  User embedding dimension: {config.user_embedding_dim}")
    
    # Compile model if enabled and available
    if performance_config.get("compile_model", False) and hasattr(torch, 'compile') and device.type == 'cuda':
        print("Compiling model for faster execution...")
        model = torch.compile(model)
    
    # Training loop
    print("\n9. Starting training...")
    print(f"Training for {num_epochs} epochs with purchase boost factor: {loss_fn.purchase_boost}")
    
    best_purchase_recall = 0
    history = {'train_loss': [], 'test_metrics': []}
    
    # Get performance parameters
    use_amp = performance_config.get("use_amp", False) and device.type == 'cuda'
    early_stopping_patience = performance_config.get("patience", 3)
    
    # Early stopping tracker
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        print(f"\n==== Epoch {epoch+1}/{num_epochs} ====")
        
        # Train with performance parameters
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
        
        # Update scheduler
        current_purchase_recall = test_metrics['purchase_recall@k'][10]
        scheduler.step(current_purchase_recall)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['test_metrics'].append(test_metrics)
        
        # Print metrics
        print(f"\nTiming: Train={train_time:.1f}s, Eval={eval_time:.1f}s")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"\nTest Metrics:")
        print(f"  Overall Recall@10: {test_metrics['recall@k'][10]*100:.2f}%")
        print(f"  Purchase Recall@10: {test_metrics['purchase_recall@k'][10]*100:.2f}%")
        print(f"  Purchase MRR: {test_metrics['purchase_mrr']:.4f}")
        
        # Save best model
        if current_purchase_recall > best_purchase_recall:
            best_purchase_recall = current_purchase_recall
            epochs_without_improvement = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_purchase_recall': best_purchase_recall,
                'config': config.__dict__,
                'valid_packages': list(valid_packages),
                'history': history,
                'performance_config': performance_config,
                'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            }
            
            # Create checkpoints directory
            os.makedirs('checkpoints/natr', exist_ok=True)
            
            # Save to both a versioned file and the best model file
            torch.save(checkpoint, f"checkpoints/natr/model_epoch_{epoch+1}.pth")
            torch.save(checkpoint, 'checkpoints/natr/best_model.pth')
            print(f"  ✓ New best model saved! Purchase Recall@10: {best_purchase_recall*100:.2f}%")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epochs")
            
            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping after {epoch+1} epochs without improvement")
                break
    
    # Final summary
    print("\n10. Training complete!")
    print(f"Best Purchase Recall@10: {best_purchase_recall*100:.2f}%")
    
    # Save model info
    model_info = {
        'config': config.__dict__,
        'valid_packages': list(valid_packages),
        'best_purchase_recall': best_purchase_recall,
        'checkpoint_path': 'checkpoints/natr/best_model.pth',
        'package_data_path': package_data_path,
        'event_data_path': event_data_path,
        'performance_config': performance_config,
        'training_date': datetime.now().strftime("%Y-%m-%d"),
        'package_count': len(valid_packages),
        'user_count': num_users
    }
    
    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("\nFiles saved:")
    print("  - checkpoints/natr/best_model.pth")
    print("  - checkpoints/natr/model_epoch_X.pth")
    print("  - model_info.json")


def set_performance_mode(mode="balanced"):
    """
    Configure training parameters based on performance mode with specific optimizations for Apple Silicon
    
    Args:
        mode: One of "fastest", "balanced", "accurate", "apple_silicon"
    
    Returns:
        dict: Configuration parameters
    """
    config = {}
    
    # Special mode for Apple Silicon (M1/M2/M3)
    is_mps = torch.backends.mps.is_available()
    
    if mode == "apple_silicon" or (mode == "balanced" and is_mps):
        # Specific optimizations for Apple Silicon
        config.update({
            # Batch size settings
            "batch_size_multiplier": 1.0,        # Standard multiplier
            "embedding_dim": 192,                # Smaller than CUDA but efficient on MPS
            "hidden_dim": 192,                   # Smaller hidden dims for MPS memory constraints
            
            # Gradient accumulation is important for MPS
            "accumulation_steps": 4,             # More gradient accumulation helps MPS
            
            # MPS doesn't support AMP, but we'll keep this for CUDA compatibility
            "use_amp": False,                    # MPS doesn't support AMP yet
            
            # MPS doesn't yet support torch.compile() in PyTorch stable release
            "compile_model": False,              # Don't attempt to compile for MPS
            
            # Other settings
            "dropout": 0.2,                      # Regular dropout still helps
            "patience": 3,                       # Regular patience
            "prefetch_factor": 2,                # Less prefetching for MPS (tends to use more CPU memory)
            "max_package_count": 30000,          # Limit package vocabulary for faster completion
            "use_reduced_embeddings": True,      # Definitely use reduced embeddings for memory efficiency
            "learning_rate": 0.001,              # Standard learning rate
            
            # MPS-specific performance flags
            "mps_clear_cache_frequency": 5,      # Clear MPS cache every N batches
            "use_half_precision_embeddings": True, # Use float16 for embeddings on MPS
            "reduce_memory_usage": True,         # Enable additional memory optimizations
        })
    elif mode == "fastest":
        # Prioritize speed over accuracy
        config.update({
            "batch_size_multiplier": 2.0,        # Larger batches
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
            "learning_rate": 0.002               # Higher learning rate
        })
    elif mode == "accurate":
        # Prioritize accuracy over speed
        config.update({
            "batch_size_multiplier": 0.5 if not is_mps else 0.75,  # Smaller batches (but not too small on MPS)
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
            "learning_rate": 0.0005                                # Lower learning rate
        })
    else:  # "balanced" mode (non-MPS)
        # Balance between speed and accuracy for non-MPS devices
        config.update({
            "batch_size_multiplier": 1.0,        # Standard batch size
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
            "learning_rate": 0.001               # Standard learning rate
        })
    
    # Print detailed configuration
    mode_display = mode
    if mode == "balanced" and is_mps:
        mode_display = "APPLE_SILICON (auto)"
    print(f"\nUsing {mode_display.upper()} performance mode")
    
    if is_mps:
        print("Apple Silicon optimizations active")
    
    return config


if __name__ == "__main__":
    import sys
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train NATR model with performance options')
    parser.add_argument('--mode', type=str, default='balanced', 
                        choices=['fastest', 'balanced', 'accurate', 'apple_silicon'],
                        help='Performance mode (default: balanced)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size (default: determined by mode)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of data loading workers (default: determined by mode)')
    parser.add_argument('--accumulation-steps', type=int, default=None,
                        help='Gradient accumulation steps (default: determined by mode)')
    
    # Parse arguments while maintaining backward compatibility
    if len(sys.argv) > 1 and sys.argv[1] in ["fastest", "balanced", "accurate", "apple_silicon"]:
        # Old style: just the mode as first argument
        performance_mode = sys.argv[1]
        args = parser.parse_args([])  # Empty args
        args.mode = performance_mode
    else:
        # New style: proper argument parsing
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
        base_batch_size = args.batch_size
        # Remove multiplier to avoid confusion
        performance_config["batch_size_multiplier"] = 1.0
        print(f"Overriding batch size to {args.batch_size}")
    
    if args.workers is not None:
        performance_config["num_workers"] = args.workers
        print(f"Overriding number of workers to {args.workers}")
    
    if args.accumulation_steps is not None:
        performance_config["accumulation_steps"] = args.accumulation_steps
        print(f"Overriding accumulation steps to {args.accumulation_steps}")
    
    if args.epochs != 30:
        print(f"Setting number of epochs to {args.epochs}")
        # Pass epochs to main through config
        performance_config["num_epochs"] = args.epochs
    
    # Start training
    main(performance_config)
    
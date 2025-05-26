"""
Complete NATR training script with natural event learning
Optimized for computation time while ensuring functionality
Combines the best features from train_natr.py and train_natr_optimized.py
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
from utils.session_processor import SessionProcessor
from utils.package_processor import PackageProcessor, TravelPackageDataset
from utils.unified_metrics import UnifiedMetricsTracker, MetricsTracker, EnhancedEventMetrics
from utils.loss_functions import NaturalPurchaseLoss
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


def evaluate_with_unified_metrics(model, test_loader, device, memory_optimizer, k_values=[5, 10, 20]):
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
                           mininterval=10.0,  # Update at most every 10 seconds
                           miniters=100)      # Update after at least 100 iterations
        
        for batch_idx, batch in enumerate(progress_bar):
            # Prepare batch and memory optimization
            memory_optimizer.before_batch(batch_idx)
            
            # Move batch to device with optimization
            batch = memory_optimizer.optimize_batch(batch)
            
            # Forward pass
            outputs = model(batch)
            predictions = outputs['predictions']
            targets = batch['purchased']['package_ids']
            
            # Get event type indicators with fallback to zeros
            is_purchase = batch.get('is_purchase', torch.zeros_like(targets, dtype=torch.bool))
            has_checkout = batch.get('has_checkout', torch.zeros_like(targets, dtype=torch.bool))
            has_add_to_cart = batch.get('has_add_to_cart', torch.zeros_like(targets, dtype=torch.bool))
            
            # Update unified metrics tracker with all event types
            metrics_tracker.update(
                predictions=predictions,
                targets=targets,
                is_purchase=is_purchase,
                has_checkout=has_checkout,
                has_add_to_cart=has_add_to_cart
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
    print(f"Event distribution:")
    print(f"  Purchase: {metrics.get('purchase', 0):,} ({metrics.get('purchase', 0)/total_samples*100:.1f}%)")
    print(f"  Checkout: {metrics.get('checkout', 0):,} ({metrics.get('checkout', 0)/total_samples*100:.1f}%)")
    print(f"  Add-to-cart: {metrics.get('add_to_cart', 0):,} ({metrics.get('add_to_cart', 0)/total_samples*100:.1f}%)")
    print(f"  Intent (purchase+checkout): {metrics.get('intent', 0):,} ({metrics.get('intent', 0)/total_samples*100:.1f}%)")
    
    return metrics


def train_epoch_with_memory_optimization(model, train_loader, optimizer, loss_fn, device, epoch,
                                       memory_optimizer, amp_manager, gradient_accumulator):
    """
    Enhanced training function combining memory optimization with detailed loss tracking
    Uses standardized memory management from train_natr_optimized.py
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Prepare for epoch
    memory_optimizer.before_epoch(epoch)
    
    # Create progress bar with better formatting
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", 
                      mininterval=5.0,   # Update at most every 5 seconds
                      miniters=50)       # Update after at least 50 iterations
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(progress_bar):
        # Memory optimization before batch
        memory_optimizer.before_batch(batch_idx)
        
        # Move batch to device with optimization
        batch = memory_optimizer.optimize_batch(batch)
        
        # Forward pass with AMP if available
        with amp_manager:
            outputs = model(batch)
            predictions = outputs['predictions']
            
            # Get targets and event indicators
            targets = batch['purchased']['package_ids']
            is_purchase = batch.get('is_purchase', torch.zeros_like(targets, dtype=torch.bool))
            has_checkout = batch.get('has_checkout', torch.zeros_like(targets, dtype=torch.bool))
            has_add_to_cart = batch.get('has_add_to_cart', torch.zeros_like(targets, dtype=torch.bool))
            
            # Calculate loss with event-specific handling
            if hasattr(loss_fn, '__call__') and len(loss_fn.__code__.co_varnames) > 3:
                # Enhanced loss function with event awareness
                if 'has_add_to_cart' in loss_fn.__code__.co_varnames:
                    loss = loss_fn(predictions, targets, is_purchase, has_checkout, has_add_to_cart)
                elif 'has_checkout' in loss_fn.__code__.co_varnames:
                    loss = loss_fn(predictions, targets, is_purchase, has_checkout)
                else:
                    loss = loss_fn(predictions, targets, is_purchase)
            else:
                # Fallback for simpler loss functions
                loss = loss_fn(predictions, targets)
        
        # Backward pass with gradient accumulation and AMP scaling
        original_loss = gradient_accumulator.backward(loss)
        
        # Update metrics
        total_loss += original_loss.item()
        num_batches += 1
        
        # Step optimizer if accumulation is complete
        if gradient_accumulator.step(optimizer):
            # Optimizer step was performed, reset gradients already handled in step()
            pass
        
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


def main(performance_mode="balanced"):
    """Main training function with enhanced memory management and metrics"""
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
    event_data_path = "data/bookit_events_data_13_months.parquet"
    
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
        learning_rate = 0.001
    
    # Initialize processors
    print("\nInitializing data processors...")
    package_processor = PackageProcessor(
        data_path=package_data_path,
        cache_dir='data/cache',
        load_coordinates=True,
        load_embeddings=True,
        api_key=os.environ.get("OPENAI_API_KEY"),
        embedding_model='text-embedding-3-small',
        use_reduced_embeddings=True if device_type != 'cpu' else False
    )
    
    session_processor = SessionProcessor(
        data_path=event_data_path,
        cache_dir='data/cache',
        min_interactions=10,
        max_sessions_per_user=20,
        max_samples_per_user=10
    )
    
    # Display training parameters
    print(f"\nTraining parameters:")
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
    
    print("\n6. Applying data filters...")
    
    # Apply quality filters
    quality_samples = filter_by_min_session_length(samples, min_session_length=3)
    filtered_samples, valid_packages = filter_items_by_frequency(quality_samples, min_frequency=50)
    
    # Validate purchase events
    purchase_count = sum(1 for s in filtered_samples if s.get('is_purchase', False))
    if purchase_count == 0:
        raise ValueError("No purchase events found in the filtered samples. Cannot train the model.")
    
    print(f"Purchase events: {purchase_count} ({purchase_count / len(filtered_samples) * 100:.2f}% of all events)")
    
    # Time-based split
    train_samples, test_samples = time_based_split_year(filtered_samples, train_ratio=0.93)
    
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
    
    # Get dimensions from mappings
    num_users = max(session_processor.user_to_idx.values()) + 1
    num_packages = max(session_processor.package_to_idx.values()) + 1
    num_countries = max(package_processor.country_to_idx.values()) + 1
    num_categories = max(package_processor.category_to_idx.values()) + 1
    num_themes = max(package_processor.theme_to_idx.values()) + 1
    
    # Detect embedding dimension
    package_tensors = package_processor.prepare_package_tensors()
    actual_embedding_dim = package_tensors['title_embeddings'].shape[1]
    print(f"Detected title embedding dimension: {actual_embedding_dim}")
    
    # Simplified model dimensions for consistent training
    # Use moderate size to balance capacity and training stability
    hidden_dim = 256
    embedding_dim = 256
    dropout = 0.2  # Fixed dropout for stability
    
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
        dropout=dropout
    )
    
    # Create model
    model = NATR(config)
    model = model.to(device)
    
    # Get model size for memory calculations
    model_size_mb = get_model_size(model)
    print(f"Model size: {model_size_mb} MB")
    
    # Create optimized optimizer
    optimizer = create_optimizer_with_memory_optimizations(
        model, learning_rate, device_type, memory_config
    )
    
    # Create AMP manager and gradient accumulator
    amp_manager = AMPManager(device, memory_config)
    gradient_accumulator = GradientAccumulator(
        steps=memory_config.gradient_accumulation_steps,
        amp_manager=amp_manager
    )
    
    # Create loss function
    loss_fn = NaturalPurchaseLoss(
        purchase_boost=3.0  # Reduced from 15.0 to prevent gradient explosion
    )
    
    # Set up checkpointing
    os.makedirs('checkpoints/natr', exist_ok=True)
    
    # Training loop
    print("\n9. Starting training...")
    best_purchase_recall = 0.0
    best_weighted_recall = 0.0
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
            memory_optimizer, amp_manager, gradient_accumulator
        )
        
        print(f"Training loss: {train_loss:.4f}")
        
        # Evaluation with comprehensive metrics
        metrics = evaluate_with_unified_metrics(
            model, test_loader, device, memory_optimizer, k_values=[5, 10, 20]
        )
        
        # Print detailed metrics
        print("\nEvaluation Results:")
        print(f"  Overall Recall@10: {metrics.get('recall@10', 0)*100:.2f}%")
        print(f"  Purchase Recall@10: {metrics.get('purchase_recall@10', 0)*100:.2f}%")
        print(f"  Checkout Recall@10: {metrics.get('checkout_recall@10', 0)*100:.2f}%")
        print(f"  Intent Recall@10: {metrics.get('intent_recall@10', 0)*100:.2f}%")
        print(f"  Weighted Recall@10: {metrics.get('weighted_recall@10', 0)*100:.2f}%")
        print(f"  Overall MRR: {metrics.get('mrr', 0):.4f}")
        print(f"  Purchase MRR: {metrics.get('purchase_mrr', 0):.4f}")
        
        # Track best metrics
        current_purchase_recall = metrics.get('purchase_recall@10', 0)
        current_weighted_recall = metrics.get('weighted_recall@10', 0)
        
        # Combined improvement metric
        improvement = (current_purchase_recall > best_purchase_recall) or \
                     (current_purchase_recall == best_purchase_recall and current_weighted_recall > best_weighted_recall)
        
        if improvement:
            best_purchase_recall = current_purchase_recall
            best_weighted_recall = current_weighted_recall
            epochs_without_improvement = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.__dict__,
                'metrics': metrics,
                'train_loss': train_loss
            }, 'checkpoints/natr/best_model.pth')
            print(f"✓ New best model saved! Purchase Recall@10: {best_purchase_recall*100:.2f}%")
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
    print(f"Best Purchase Recall@10: {best_purchase_recall*100:.2f}%")
    print(f"Best Weighted Recall@10: {best_weighted_recall*100:.2f}%")
    
    # Save model info
    model_info = {
        'config': config.__dict__,
        'valid_packages': list(valid_packages),
        'best_purchase_recall': best_purchase_recall,
        'best_weighted_recall': best_weighted_recall,
        'checkpoint_path': 'checkpoints/natr/best_model.pth',
        'package_data_path': package_data_path,
        'event_data_path': event_data_path,
        'performance_mode': performance_mode,
        'device_type': device_type,
        'training_date': datetime.now().strftime("%Y-%m-%d"),
        'package_count': len(valid_packages),
        'user_count': num_users,
        'model_size_mb': model_size_mb
    }
    
    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("\nFiles saved:")
    print("  - checkpoints/natr/best_model.pth")
    print("  - checkpoints/natr/model_epoch_X.pth")
    print("  - model_info.json")
    
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
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size (default: determined by mode and device)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Override learning rate (default: determined by mode)')
    
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
    main(performance_mode=args.mode)
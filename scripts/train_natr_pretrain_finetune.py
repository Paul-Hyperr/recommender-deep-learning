"""
Enhanced NATR training script with pre-training and fine-tuning strategy
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
from utils.unified_metrics import UnifiedMetricsTracker, MetricsTracker, EnhancedEventMetrics
from utils.loss_functions import NaturalPurchaseLoss, FocalLoss
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
    analyze_data_distribution,
    move_batch_to_device,
    train_epoch,
    evaluate,
    calculate_batch_metrics,
    create_dataloaders,
    set_performance_mode,
    clear_memory,
    limit_samples_for_testing,
    identify_event_types
)

# Import the enhanced evaluation function from train_natr.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_natr import evaluate_with_unified_metrics


def create_next_item_prediction_targets(samples):
    """
    Convert ALL samples (purchase + non-purchase) to next-item prediction format for pre-training
    Uses soft labels based on event hierarchy: Purchase=1.0, Checkout=0.8, AddToCart=0.5, View=0.1
    """
    print("\nConverting ALL samples to next-item prediction format for pre-training...")
    next_item_samples = []
    
    # Event hierarchy for soft labeling - consistent across all files
    event_weights = {
        'Purchase': 1.0,
        'InitiateCheckout': 0.6, 
        'AddToCart': 0.2,
        'View': 0.05
    }
    
    for sample in samples:
        # Get short-term session packages
        short_term_pkgs = sample.get('short_term_packages', [])
        
        if len(short_term_pkgs) >= 2:  # Need at least 2 items for next-item prediction
            # Use a more reasonable sampling strategy - only sample a few positions per session
            # This prevents explosion of training samples while keeping diversity
            max_samples_per_session = min(3, len(short_term_pkgs) - 1)  # Max 3 samples per session
            
            if len(short_term_pkgs) <= 4:
                # For short sessions, use all positions
                positions = list(range(len(short_term_pkgs) - 1))
            else:
                # For longer sessions, sample strategic positions: early, middle, late
                positions = [0]  # Always include first position
                if len(short_term_pkgs) > 3:
                    middle = len(short_term_pkgs) // 2
                    positions.append(middle)
                positions.append(len(short_term_pkgs) - 2)  # Always include second-to-last
                positions = list(set(positions))  # Remove duplicates
            
            for i in positions:
                # Create a new sample for each position
                new_sample = sample.copy()
                
                # Use context up to position i as short-term context
                new_sample['short_term_packages'] = short_term_pkgs[:i+1]
                # If there are timestamps, also trim those
                if 'short_term_timestamps' in sample:
                    new_sample['short_term_timestamps'] = sample['short_term_timestamps'][:i+1]
                if 'short_term_events' in sample:
                    new_sample['short_term_events'] = sample['short_term_events'][:i+1]
                
                # Target is the next item in the sequence
                new_sample['purchased_package'] = short_term_pkgs[i+1]
                new_sample['is_next_item'] = True  # Mark that this is a next-item prediction sample
                
                # For pre-training, is_purchase should only be True if:
                # 1. This is the final item in the session AND
                # 2. The original session actually ended in a purchase
                is_final_item = (i+1 == len(short_term_pkgs) - 1)
                session_was_purchase = sample.get('is_purchase', False)
                new_sample['is_purchase'] = is_final_item and session_was_purchase
                
                # Assign soft labels based on event type hierarchy (for weighting loss)
                target_event_idx = i+1
                short_term_events = sample.get('short_term_events', [])
                
                if target_event_idx < len(short_term_events):
                    event_idx = short_term_events[target_event_idx]
                    # Map event index to event name for soft labeling
                    if event_idx == 1:  # Purchase event
                        new_sample['soft_label'] = event_weights['Purchase']
                    elif event_idx == 3:  # InitiateCheckout event  
                        new_sample['soft_label'] = event_weights['InitiateCheckout']
                    elif event_idx == 2:  # AddToCart event
                        new_sample['soft_label'] = event_weights['AddToCart'] 
                    else:  # View or other events
                        new_sample['soft_label'] = event_weights['View']
                else:
                    # Fallback for sessions without event data
                    if new_sample['is_purchase']:
                        new_sample['soft_label'] = event_weights['Purchase']
                    else:
                        new_sample['soft_label'] = event_weights['View']
                
                next_item_samples.append(new_sample)
    
    print(f"Created {len(next_item_samples)} next-item prediction samples from {len(samples)} original samples")
    return next_item_samples


def create_balanced_fine_tuning_set(samples):
    """Create a balanced dataset for fine-tuning"""
    print("\nCreating balanced dataset for fine-tuning...")
    
    # Separate purchase and non-purchase samples
    purchase_samples = [s for s in samples if s.get('is_purchase', False)]
    non_purchase_samples = [s for s in samples if not s.get('is_purchase', False)]
    
    print(f"Original distribution: {len(purchase_samples)} purchases, {len(non_purchase_samples)} non-purchases")
    
    # Subsample non-purchases to match purchase count (1:1 ratio)
    if len(purchase_samples) < len(non_purchase_samples):
        # Randomly sample non-purchases equal to purchase count
        np.random.seed(42)  # For reproducibility
        sampled_indices = np.random.choice(
            len(non_purchase_samples), 
            size=len(purchase_samples), 
            replace=False
        )
        balanced_non_purchases = [non_purchase_samples[i] for i in sampled_indices]
        
        # Combine to create balanced dataset
        balanced_samples = purchase_samples + balanced_non_purchases
        
        print(f"Balanced dataset: {len(purchase_samples)} purchases, {len(balanced_non_purchases)} non-purchases")
        print(f"Total balanced samples: {len(balanced_samples)}")
    else:
        print("Already balanced or more purchases than non-purchases")
        balanced_samples = purchase_samples + non_purchase_samples
    
    return balanced_samples


def train_epoch_with_soft_labels(model, train_loader, optimizer, loss_fn, device, epoch):
    """Training epoch for pre-training with soft labels"""
    model.train()
    total_loss = 0.0
    batch_count = 0
    
    progress_bar = tqdm(train_loader, desc=f"Pre-training Epoch {epoch+1}")
    
    for batch_idx, batch in enumerate(progress_bar):
        batch = move_batch_to_device(batch, device)
        
        optimizer.zero_grad()
        
        outputs = model(batch)
        predictions = outputs['predictions']
        targets = batch['purchased']['package_ids']
        
        # Get soft labels for weighted loss
        soft_labels = batch.get('soft_labels', torch.ones_like(targets, dtype=torch.float))
        
        # Compute loss with soft labels
        loss = loss_fn(predictions, targets, soft_labels)
        
        loss.backward()
        
        # Apply gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Memory cleanup for MPS
        if device.type == 'mps' and batch_idx % 10 == 0:
            torch.mps.empty_cache()
    
    return total_loss / max(batch_count, 1)


def pretrain_phase(model, train_loader, test_loader, device, performance_config):
    """Pre-training phase focusing on next-item prediction"""
    print("\n-------- Starting Pre-training Phase --------")
    print("Pre-training model on next-item prediction task...")
    
    # Extract parameters
    learning_rate = performance_config.get("learning_rate", 0.001)
    num_epochs = performance_config.get("pretrain_epochs", 1)
    accumulation_steps = performance_config.get("accumulation_steps", 2)
    use_amp = performance_config.get("use_amp", False) and device.type == 'cuda'
    
    # Configure optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.02,  # Less regularization for pre-training (increased from 0.01)
        eps=1e-8
    )
    
    # Use a weighted cross-entropy loss for pre-training with soft labels
    # This allows the model to learn from the event hierarchy
    class SoftLabelLoss(nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, predictions, targets, soft_labels):
            """
            Weighted loss based on soft labels from event hierarchy
            Higher weight for more valuable events (Purchase > Checkout > AddToCart > View)
            """
            ce_loss = F.cross_entropy(predictions, targets, reduction='none')
            # Weight the loss by soft labels - more valuable events get higher weight
            weighted_loss = ce_loss * soft_labels
            return weighted_loss.mean()
    
    loss_fn = SoftLabelLoss()
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',           # Optimize for maximizing recall
        factor=0.5,           # Halve the learning rate when plateauing
        patience=1,           # Wait 1 epoch before reducing
        threshold=0.0001,     # Minimum improvement to count as progress
        min_lr=1e-6           # Minimum learning rate
    )
    
    # Training loop for pre-training
    best_purchase_recall = 0
    history = {'train_loss': [], 'test_metrics': []}
    
    for epoch in range(num_epochs):
        print(f"\n==== Pre-train Epoch {epoch+1}/{num_epochs} ====")
        
        # Train for one epoch with soft labels (pre-training)
        start_time = time.time()
        train_loss = train_epoch_with_soft_labels(
            model, train_loader, optimizer, loss_fn, device, epoch
        )
        train_time = time.time() - start_time
        
        # Evaluate with unified metrics (includes item coverage)
        start_time = time.time()
        # Create simple memory optimizer for evaluation
        memory_config = create_memory_config(device.type, "balanced")
        memory_optimizer = MemoryOptimizer(device, memory_config)
        test_metrics = evaluate_with_unified_metrics(model, test_loader, device, memory_optimizer, k_values=[10, 20, 50])
        eval_time = time.time() - start_time
        
        # Update scheduler based on purchase recall@20 metric
        # Purchase recall@20 aligns directly with the final fine-tuning objective
        current_purchase_recall = test_metrics.get('purchase_recall@20', 0.0)
        scheduler.step(current_purchase_recall)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['test_metrics'].append(test_metrics)
        
        # Print metrics
        print(f"\nTiming: Train={train_time:.1f}s, Eval={eval_time:.1f}s")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"\nTest Metrics (Next Item Prediction):")
        print(f"  Overall MRR: {test_metrics.get('overall_mrr', 0.0):.4f}")
        print(f"  Purchase Recall@20: {test_metrics.get('purchase_recall@20', 0.0)*100:.2f}%")
        if 'checkout_recall@20' in test_metrics:
            print(f"  Checkout Recall@20: {test_metrics['checkout_recall@20']*100:.2f}%")
        if 'item_coverage@20' in test_metrics:
            print(f"  Item Coverage@20: {test_metrics['item_coverage@20']*100:.2f}%")
        
        # Save best model based on purchase recall@20
        if current_purchase_recall > best_purchase_recall:
            best_purchase_recall = current_purchase_recall
            
            # Save pre-trained checkpoint
            os.makedirs('checkpoints/natr', exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_purchase_recall': best_purchase_recall,
                'history': history,
                'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            }
            torch.save(checkpoint, 'checkpoints/natr/pretrained_model.pth')
            print(f"  ✓ New best pre-trained model saved! Purchase Recall@20: {best_purchase_recall*100:.2f}%")
    
    print("\nPre-training complete!")
    print(f"Best Purchase Recall@20: {best_purchase_recall*100:.2f}%")
    
    return model, best_purchase_recall, history


def finetune_phase(model, train_loader, test_loader, device, performance_config):
    """Fine-tuning phase focusing on purchase prediction"""
    print("\n-------- Starting Fine-tuning Phase --------")
    print("Fine-tuning model on purchase prediction task...")
    
    # Extract parameters
    learning_rate = performance_config.get("learning_rate", 0.001) * 0.1  # Lower LR for fine-tuning
    num_epochs = performance_config.get("finetune_epochs", 10)
    accumulation_steps = performance_config.get("accumulation_steps", 2)
    use_amp = performance_config.get("use_amp", False) and device.type == 'cuda'
    early_stopping_patience = performance_config.get("patience", 8)
    
    # Configure optimizer with enhanced user learning (multi-rate optimization)
    print("Creating custom optimizer with enhanced user learning for fine-tuning...")
    
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
        {'params': other_params, 'lr': learning_rate, 'weight_decay': 0.05},
        {'params': user_params, 'lr': user_lr, 'weight_decay': 1e-6},  # Less regularization for users
        {'params': user_transform_params, 'lr': user_transform_lr, 'weight_decay': 1e-6}
    ], eps=1e-8)
    
    print(f"  Base learning rate: {learning_rate}")
    print(f"  User embedding learning rate: {user_lr}")
    print(f"  User transform learning rate: {user_transform_lr}")
    print(f"  User embedding params: {sum(p.numel() for p in user_params)}")
    print(f"  User transform params: {sum(p.numel() for p in user_transform_params)}")
    print(f"  Other params: {sum(p.numel() for p in other_params)}")
    
    # Option 2 (Alternative): Fine-tune specific layers with different learning rates
    # This creates layer-specific learning rates, higher for layers that need more adaptation
    """
    # Parameter groups with different learning rates
    decoder_params = list(model.prediction_head.parameters())
    attention_params = []
    for name, param in model.named_parameters():
        if 'attention' in name or 'fusion' in name:
            attention_params.append(param)
    other_params = [p for p in model.parameters() 
                   if p not in decoder_params and p not in attention_params]
    
    optimizer = optim.AdamW([
        {'params': decoder_params, 'lr': learning_rate},
        {'params': attention_params, 'lr': learning_rate * 0.5},
        {'params': other_params, 'lr': learning_rate * 0.1}
    ], weight_decay=0.05, eps=1e-8)
    """
    
    # Use NaturalPurchaseLoss with strong boost for fine-tuning
    loss_fn = NaturalPurchaseLoss(
        purchase_boost=20.0  # Increased from 4.0 for much stronger purchase signal
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',           # Optimize for maximizing recall
        factor=0.5,           # Halve the learning rate when plateauing
        patience=2,           # Wait 2 epochs before reducing
        threshold=0.0001,     # Minimum improvement to count as progress
        min_lr=1e-6           # Minimum learning rate
    )
    
    # Training loop for fine-tuning
    best_purchase_recall = 0
    best_purchase_mrr = 0
    epochs_without_improvement = 0
    history = {'train_loss': [], 'test_metrics': []}
    
    for epoch in range(num_epochs):
        print(f"\n==== Fine-tune Epoch {epoch+1}/{num_epochs} ====")
        
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
        test_metrics = evaluate_with_unified_metrics(model, test_loader, device, memory_optimizer, k_values=[10, 20, 50])
        eval_time = time.time() - start_time
        
        # Update scheduler based on purchase recall metric (using @20 as primary)
        current_purchase_recall = test_metrics.get('purchase_recall@20', 0.0)
        scheduler.step(current_purchase_recall)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['test_metrics'].append(test_metrics)
        
        # Print metrics
        print(f"\nTiming: Train={train_time:.1f}s, Eval={eval_time:.1f}s")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"\nTest Metrics:")
        print(f"  Purchase Recall@10: {test_metrics.get('purchase_recall@10', 0.0)*100:.2f}%")
        print(f"  Purchase Recall@20: {test_metrics.get('purchase_recall@20', 0.0)*100:.2f}%")
        print(f"  Purchase MRR: {test_metrics.get('purchase_mrr', 0.0):.4f}")
        if 'item_coverage@20' in test_metrics:
            print(f"  Item Coverage@20: {test_metrics['item_coverage@20']*100:.2f}%")
        
        # Save best model based on Purchase Recall@20 with MRR as tiebreaker
        epsilon = 1e-6  # tolerance for considering recalls as similar
        current_purchase_mrr = test_metrics.get('purchase_mrr', 0.0)
        
        is_better = False
        if current_purchase_recall > best_purchase_recall + epsilon:
            # Clear improvement in recall@20
            is_better = True
        elif abs(current_purchase_recall - best_purchase_recall) <= epsilon:
            # Recall@20 is effectively the same, use MRR as tiebreaker
            if current_purchase_mrr > best_purchase_mrr:
                is_better = True
        
        if is_better:
            best_purchase_recall = current_purchase_recall
            best_purchase_mrr = current_purchase_mrr
            epochs_without_improvement = 0
            
            # Save fine-tuned checkpoint
            os.makedirs('checkpoints/natr', exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_purchase_recall': best_purchase_recall,
                'history': history,
                'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            }
            torch.save(checkpoint, 'checkpoints/natr/finetuned_model.pth')
            print(f"  ✓ New best fine-tuned model saved! Purchase Recall@20: {best_purchase_recall*100:.2f}%, MRR: {current_purchase_mrr:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epochs")
            
            # Save periodic checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_purchase_recall': best_purchase_recall,
                    'history': history,
                    'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                }
                torch.save(checkpoint, f"checkpoints/natr/finetuned_checkpoint_epoch_{epoch+1}.pth")
                print(f"  ✓ Periodic checkpoint saved at epoch {epoch+1}")
            
            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping after {epoch+1} epochs without improvement")
                break
    
    print("\nFine-tuning complete!")
    print(f"Best Purchase Recall@20: {best_purchase_recall*100:.2f}%")
    print(f"Best Purchase MRR: {best_purchase_mrr:.4f}")
    
    return model, best_purchase_recall, best_purchase_mrr


def get_pretrain_finetune_config(mode="balanced"):
    """Get specific parameters for pretrain-finetune approach based on performance mode"""
    config = {}
    
    # Special mode for Apple Silicon (M1/M2/M3)
    is_mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    
    if mode == "apple_silicon" or (mode == "balanced" and is_mps):
        config.update({
            "pretrain_epochs": 3,                # Changed from 7 to 3 - testing with fewer pretrain epochs
            "finetune_epochs": 20                # Number of epochs for fine-tuning phase
        })
    elif mode == "fastest":
        config.update({
            "pretrain_epochs": 3,                # Changed from 5 to 3 - testing with fewer pretrain epochs
            "finetune_epochs": 20                # Finetune epochs for fastest mode
        })
    elif mode == "accurate":
        config.update({
            "pretrain_epochs": 3,                # Changed from 5 to 3 - testing with fewer pretrain epochs
            "finetune_epochs": 20                # Finetune epochs for accurate mode
        })
    else:  # "balanced" mode (non-MPS)
        config.update({
            "pretrain_epochs": 3,                # Changed from 7 to 3 - testing with fewer pretrain epochs
            "finetune_epochs": 20                # Finetune epochs for balanced mode
        })
    
    return config


def main(performance_config=None, dataset="13months", event_data_path=None):
    """Main function implementing pre-training + fine-tuning strategy
    
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
        
    # Add pretrain-finetune specific parameters
    pretrain_finetune_config = get_pretrain_finetune_config(
        "apple_silicon" if device.type == "mps" else performance_config.get("mode", "balanced")
    )
    performance_config.update(pretrain_finetune_config)
    
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
    
    print(f"\nTraining parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Pre-training epochs: {performance_config.get('pretrain_epochs', 1)}")
    print(f"  Fine-tuning epochs: {performance_config.get('finetune_epochs', 10)}")
    print(f"  Learning rate: {performance_config.get('learning_rate', 0.001)}")
    print(f"  Gradient accumulation steps: {performance_config.get('accumulation_steps', 2)}")
    print(f"  Workers: {num_workers}")
    
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
    samples = identify_event_types(samples, session_processor.event_to_idx)
    
    # Limit samples for testing if needed
    samples = limit_samples_for_testing(samples)
    
    print("\n6. Applying data filters...")
    
    # Apply session length filtering - quality improves with longer sessions
    quality_samples = filter_by_min_session_length(samples, min_session_length=2)
    
    # Filter items by frequency - focus on packages with sufficient data
    filtered_samples, valid_packages = filter_items_by_frequency(quality_samples, min_frequency=5)  # Reduced from 10
    
    # Validate that we have purchase events
    purchase_count = sum(1 for s in filtered_samples if s.get('is_purchase', False))
    if purchase_count == 0:
        raise ValueError("No purchase events found in the filtered samples. Cannot train the model.")
    
    print(f"Purchase events: {purchase_count} ({purchase_count / len(filtered_samples) * 100:.2f}% of all events)")
    
    # Time-based split for main dataset
    all_train_samples, test_samples, split_date = time_based_split_year(filtered_samples, train_ratio=0.91)
    
    # Ensure evaluation set has enough purchases for meaningful metrics
    purchase_samples = [s for s in test_samples if s.get('is_purchase', False)]
    if len(purchase_samples) < 50:
        print(f"Warning: Only {len(purchase_samples)} purchases in test set, adding more...")
        # Find more purchases from train samples
        extra_purchases = [s for s in all_train_samples if s.get('is_purchase', False)][:50-len(purchase_samples)]
        test_samples.extend(extra_purchases)
        # Remove these from train samples
        all_train_samples = [s for s in all_train_samples if s not in extra_purchases]
        print(f"Added {len(extra_purchases)} more purchase samples to test set")
    
    # IMPORTANT: Update user mappings to include ALL users from both train and test sets
    print("\n6b. Updating user mappings to include test users...")
    all_users = set()
    for sample in all_train_samples + test_samples:
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
    analyze_data_distribution(all_train_samples, "Train (Original)")
    analyze_data_distribution(test_samples, "Test")
    
    # ---- Create next-item prediction samples for pre-training ----
    next_item_samples = create_next_item_prediction_targets(all_train_samples)
    analyze_data_distribution(next_item_samples, "Pre-training (Next-Item)")
    
    # ---- Create balanced dataset for fine-tuning ----
    balanced_samples = create_balanced_fine_tuning_set(all_train_samples)
    analyze_data_distribution(balanced_samples, "Fine-tuning (Balanced)")
    
    # Get model dimensions
    print("\n7. Creating model...")
    
    # Create config with performance parameters
    hidden_dim = performance_config.get("hidden_dim", 256)
    embedding_dim = performance_config.get("embedding_dim", 256)
    dropout = performance_config.get("dropout", 0.2)
    
    # Get actual dimensions from mappings (add 1 because indices start at 1, not 0)
    # IMPORTANT: These dimensions must include ALL users/packages from both train and test sets
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
        title_embedding_dim=actual_embedding_dim,  # Use detected dimension from data
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
    
    # Print model configuration for debugging
    print(f"  Title embedding dimension: {config.title_embedding_dim}")
    print(f"  Hidden dimension: {config.hidden_dim}")
    print(f"  Embedding dimension: {config.embedding_dim}")
    print(f"  User embedding dimension: {config.user_embedding_dim}")
    
    # Initialize popularity bias based on item frequencies in training data
    print("Calculating item frequencies for popularity initialization...")
    item_frequencies = torch.zeros(num_packages)
    
    # Use balanced_samples for popularity calculation as it includes purchase events
    for sample in balanced_samples:
        # Give higher weight to purchased items
        if sample.get('is_purchase', False) and 'purchased_package' in sample:
            pkg_id = sample['purchased_package']
            if str(pkg_id) in session_processor.package_to_idx:
                idx = session_processor.package_to_idx[str(pkg_id)]
                if idx < num_packages:  # Ensure index is within bounds
                    item_frequencies[idx] += 1.0  # Full weight for purchases
        
        # Give lower weight to other interactions
        for pkg_id in sample.get('short_term_packages', []) + sample.get('long_term_packages', []):
            if str(pkg_id) in session_processor.package_to_idx:
                idx = session_processor.package_to_idx[str(pkg_id)]
                if idx < num_packages:  # Ensure index is within bounds
                    item_frequencies[idx] += 0.1  # Lower weight for non-purchased interactions
    
    # Initialize popularity in the model
    # COMMENTED OUT: Testing without popularity features
    # print(f"Initializing popularity bias with item frequencies (max freq: {item_frequencies.max().item():.0f})")
    # model.initialize_popularity(item_frequencies.to(device))
    
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
    
    # ---- PRE-TRAINING PHASE ----
    print("\n8. Creating pre-training dataloaders...")
    pretrain_train_loader, pretrain_test_loader = create_dataloaders(
        next_item_samples, test_samples, package_processor, session_processor,
        batch_size=batch_size, num_workers=num_workers, use_weighted_sampling=False
    )
    
    print(f"\nPre-training dataset sizes:")
    print(f"  Train: {len(pretrain_train_loader.dataset):,} samples ({len(pretrain_train_loader):,} batches)")
    print(f"  Test: {len(pretrain_test_loader.dataset):,} samples ({len(pretrain_test_loader):,} batches)")
    
    # Pre-train model
    print("\n9. Starting pre-training phase...")
    model, best_pretrain_recall, _ = pretrain_phase(
        model, pretrain_train_loader, pretrain_test_loader, device, performance_config
    )
    
    # ---- FINE-TUNING PHASE ----
    print("\n10. Creating fine-tuning dataloaders...")
    finetune_train_loader, finetune_test_loader = create_dataloaders(
        balanced_samples, test_samples, package_processor, session_processor,
        batch_size=batch_size, num_workers=num_workers, use_weighted_sampling=False  # No need for weighted sampling with balanced data
    )
    
    print(f"\nFine-tuning dataset sizes:")
    print(f"  Train: {len(finetune_train_loader.dataset):,} samples ({len(finetune_train_loader):,} batches)")
    print(f"  Test: {len(finetune_test_loader.dataset):,} samples ({len(finetune_test_loader):,} batches)")
    
    # Fine-tune model
    print("\n11. Starting fine-tuning phase...")
    model, best_finetune_recall, best_finetune_mrr = finetune_phase(
        model, finetune_train_loader, finetune_test_loader, device, performance_config
    )
    
    # Final summary
    print("\n12. Training complete!")
    print(f"Pre-training best Purchase Recall@20: {best_pretrain_recall*100:.2f}%")
    print(f"Fine-tuning best Purchase Recall@20: {best_finetune_recall*100:.2f}%")
    
    # Save model info with @20 metrics
    model_info = {
        'config': config.__dict__,
        'valid_packages': list(valid_packages),
        'best_pretrain_purchase_recall@20': best_pretrain_recall,
        'best_finetune_recall@20': best_finetune_recall,
        'best_finetune_mrr': best_finetune_mrr,
        'checkpoint_path': 'checkpoints/natr/finetuned_model.pth',
        'package_data_path': package_data_path,
        'event_data_path': event_data_path,
        'performance_config': performance_config,
        'training_date': datetime.now().strftime("%Y-%m-%d"),
        'package_count': len(valid_packages),
        'user_count': num_users,
        'next_item_sample_count': len(next_item_samples),
        'balanced_sample_count': len(balanced_samples),
        'training_strategy': 'pretrain_finetune',
        'primary_metric': 'purchase_recall@20',
        'tiebreaker_metric': 'purchase_mrr',
        'split_date': split_date,  # Save the train/test split date
        'train_ratio': 0.91  # Save the train ratio used
    }
    
    # Save model info to appropriate directory based on dataset
    output_dir = 'output/model_info' if args.dataset == '13months' else 'output/model_info_2months'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'model_info_pretrain_finetune.json'), 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("\nFiles saved:")
    print("  - checkpoints/natr/pretrained_model.pth")
    print("  - checkpoints/natr/finetuned_model.pth")
    print("  - model_info_pretrain_finetune.json")


if __name__ == "__main__":
    import sys
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train NATR model with pre-training and fine-tuning strategy')
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
    parser.add_argument('--pretrain-epochs', type=int, default=None,
                        help='Number of pre-training epochs (default: determined by mode)')
    parser.add_argument('--finetune-epochs', type=int, default=None,
                        help='Number of fine-tuning epochs (default: determined by mode)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of data loading workers (default: determined by mode)')
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
    performance_config["mode"] = args.mode  # Store the mode for pretrain-finetune config
    
    # Override with specific command-line arguments if provided
    if args.batch_size is not None:
        performance_config["batch_size"] = args.batch_size
        performance_config["batch_size_multiplier"] = 1.0
        print(f"Overriding batch size to {args.batch_size}")
    
    if args.workers is not None:
        performance_config["num_workers"] = args.workers
        print(f"Overriding number of workers to {args.workers}")
    
    if args.pretrain_epochs is not None:
        performance_config["pretrain_epochs"] = args.pretrain_epochs
        print(f"Overriding pre-training epochs to {args.pretrain_epochs}")
    
    if args.finetune_epochs is not None:
        performance_config["finetune_epochs"] = args.finetune_epochs
        print(f"Overriding fine-tuning epochs to {args.finetune_epochs}")
    
    if args.learning_rate is not None:
        performance_config["learning_rate"] = args.learning_rate
        print(f"Overriding learning rate to {args.learning_rate}")
    
    # Start training with pre-train + fine-tune strategy
    main(performance_config, dataset=args.dataset, event_data_path=args.event_data)
"""
Enhanced NATR training script with pre-training and fine-tuning strategy
Uses enhanced architecture with events as 7th view, user price bias, and recent popularity boost (20 days)
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
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
from utils.loss_functions import NaturalPurchaseLoss, FocalLoss
from utils.recent_popularity import calculate_recent_popularity_scores, get_recent_popularity_from_samples
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

# Import the enhanced evaluation function directly
from utils.unified_metrics import UnifiedMetricsTracker


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


def create_balanced_finetune_dataset(all_samples, epoch=0):
    """
    Create balanced dataset for fine-tuning with resampling each epoch
    
    Args:
        all_samples: All intent-based samples
        epoch: Current epoch (for different random sampling each time)
    
    Returns:
        Balanced samples with equal purchases and non-purchases
    """
    # Separate purchase and non-purchase samples
    purchase_samples = [s for s in all_samples if s.get('is_purchase', False)]
    non_purchase_samples = [s for s in all_samples if not s.get('is_purchase', False)]
    
    if epoch == 0:  # Only print once
        print(f"\nCreating balanced fine-tuning dataset...")
        print(f"  Available: {len(purchase_samples)} purchases, {len(non_purchase_samples)} non-purchases")
    
    if len(purchase_samples) == 0:
        print("  Warning: No purchase samples found!")
        return all_samples
    
    # Resample non-purchases to match purchase count (different each epoch)
    if len(non_purchase_samples) >= len(purchase_samples):
        # Use epoch as seed for different sampling each time
        np.random.seed(42 + epoch)
        sampled_indices = np.random.choice(
            len(non_purchase_samples), 
            size=len(purchase_samples), 
            replace=False
        )
        balanced_non_purchases = [non_purchase_samples[i] for i in sampled_indices]
    else:
        # If we have fewer non-purchases, use all of them
        balanced_non_purchases = non_purchase_samples
    
    # Combine to create balanced dataset
    balanced_samples = purchase_samples + balanced_non_purchases
    
    # Shuffle the balanced dataset
    np.random.seed(42 + epoch)
    np.random.shuffle(balanced_samples)
    
    if epoch == 0:  # Only print once
        print(f"  Balanced dataset: {len(purchase_samples)} purchases, {len(balanced_non_purchases)} non-purchases")
        print(f"  Total: {len(balanced_samples)} samples")
    
    return balanced_samples


def create_intent_samples_enhanced(samples, event_to_idx):
    """
    Enhanced intent-based sample creation that preserves ALL purchases
    
    For each session, creates multiple samples - one for each purchase/high-intent event.
    Example: [View A, Purchase A, View B, Cart B, Purchase C] creates:
    - Sample 1: [padding] → Purchase A  
    - Sample 2: [View A, Purchase A, View B, Cart B] → Purchase C
    This ensures NO purchases are lost for training.
    """
    print("\nCreating intent-based samples with enhanced temporal modeling...")
    print("Strategy: Create samples for EVERY purchase + highest non-purchase intent per session")
    
    intent_samples = []
    intent_hierarchy = {'Purchase': 4, 'InitiateCheckout': 3, 'AddToCart': 2, 'ViewContent': 1}
    
    # Event type statistics  
    intent_stats = defaultdict(int)
    data_leakage_check = 0
    total_samples_created = 0
    original_purchase_samples = sum(1 for s in samples if s.get('is_purchase', False))
    purchase_samples_processed = 0
    
    for sample in tqdm(samples, desc="Creating intent-based samples"):
        # Get sequences
        short_term_packages = sample.get('short_term_packages', [])
        short_term_events = sample.get('short_term_events', [])
        short_term_timestamps = sample.get('short_term_timestamps', [])
        
        if len(short_term_packages) < 1 or len(short_term_events) != len(short_term_packages):
            # For purchase samples with empty sequences, create a minimal context
            if sample.get('is_purchase', False):
                # Create purchase sample with padding as context
                new_sample = sample.copy()
                new_sample['short_term_packages'] = [0]  # Padding token
                new_sample['short_term_events'] = [0]    # Padding event
                new_sample['short_term_timestamps'] = [sample.get('timestamp', 0)]
                new_sample['target_intent'] = 'Purchase'
                new_sample['has_checkout'] = False
                new_sample['has_add_to_cart'] = False
                new_sample['context_length'] = 1
                
                intent_samples.append(new_sample)
                intent_stats['Purchase'] += 1
                total_samples_created += 1
                purchase_samples_processed += 1
            continue
        
        # Find all positions with Purchase, Checkout, or Cart events
        intent_positions = []
        purchase_positions = []
        
        for i, event_idx in enumerate(short_term_events):
            # Map event index back to event name
            event_name = None
            for name, idx in event_to_idx.items():
                if idx == event_idx:
                    event_name = name
                    break
            
            if event_name in intent_hierarchy and intent_hierarchy[event_name] >= 2:  # Cart or higher
                intent_positions.append((i, event_name, intent_hierarchy[event_name]))
                
                # Track purchases separately
                if event_name == 'Purchase':
                    purchase_positions.append((i, event_name, intent_hierarchy[event_name]))
        
        # Don't skip samples without intent_positions - they might be purchase samples
        # if not intent_positions:
        #     continue
        
        # STRATEGY 1: If this sample is already a purchase (from session processor), keep it as-is
        if sample.get('is_purchase', False):
            purchase_samples_processed += 1
            # This is already a purchase sample from session processor
            # Just add intent information and keep the original context
            new_sample = sample.copy()
            new_sample['target_intent'] = 'Purchase'
            new_sample['has_checkout'] = False
            new_sample['has_add_to_cart'] = False
            new_sample['context_length'] = len(new_sample.get('short_term_packages', []))
            
            # Data leakage check - this is normal behavior (view → cart → checkout → purchase)
            target_package = new_sample.get('purchased_package')
            context_packages = new_sample.get('short_term_packages', [])
            if target_package and target_package in context_packages:
                data_leakage_check += 1
            
            intent_samples.append(new_sample)
            intent_stats['Purchase'] += 1
            total_samples_created += 1
        
        # STRATEGY 2: For non-purchase samples, look for purchases in the sequence  
        elif purchase_positions:
            # Create samples for purchases found within the sequence
            for purchase_pos, purchase_intent, _ in purchase_positions:
                target_package = short_term_packages[purchase_pos]
                
                # Create sequence up to (but NOT including) the purchase position
                context_packages = short_term_packages[:purchase_pos]
                context_events = short_term_events[:purchase_pos]
                context_timestamps = short_term_timestamps[:purchase_pos] if short_term_timestamps else []
                
                # For purchases with no context, add a dummy context item (padding)
                if len(context_packages) == 0:
                    context_packages = [0]  # Padding token
                    context_events = [0]    # Padding event
                    context_timestamps = [sample.get('short_term_timestamps', [0])[0] if sample.get('short_term_timestamps') else 0]
                
                # Create enhanced sample
                new_sample = sample.copy()
                new_sample['short_term_packages'] = context_packages
                new_sample['short_term_events'] = context_events
                if context_timestamps:
                    new_sample['short_term_timestamps'] = context_timestamps
                
                # Set target
                new_sample['purchased_package'] = target_package
                new_sample['is_purchase'] = True
                new_sample['target_intent'] = 'Purchase'
                new_sample['target_position'] = purchase_pos
                new_sample['context_length'] = len(context_packages)
                
                # Enhanced flags for metrics
                new_sample['has_checkout'] = False
                new_sample['has_add_to_cart'] = False
                
                # Data leakage check - should not happen since we create context up to target position
                if target_package in context_packages:
                    data_leakage_check += 1
                    print(f"Warning: Data leakage in Strategy 2 - target {target_package} found in context {context_packages[:5]}...")
                
                intent_samples.append(new_sample)
                intent_stats['Purchase'] += 1
                total_samples_created += 1
        
        # STRATEGY 3: Create a sample for the highest non-purchase intent (if any and if not already a purchase sample)
        if not sample.get('is_purchase', False):
            non_purchase_positions = [(pos, intent, score) for pos, intent, score in intent_positions if intent != 'Purchase']
            
            if non_purchase_positions:
                # Find the LAST occurrence of the HIGHEST non-purchase intent
                highest_non_purchase_score = max(pos[2] for pos in non_purchase_positions)
                highest_non_purchase_positions = [(pos[0], pos[1]) for pos in non_purchase_positions if pos[2] == highest_non_purchase_score]
                
                # Use the LAST occurrence of the highest non-purchase intent
                target_position, target_intent = highest_non_purchase_positions[-1]
                target_package = short_term_packages[target_position]
                
                # Create sequence up to (but NOT including) the target position
                context_packages = short_term_packages[:target_position]
                context_events = short_term_events[:target_position]
                context_timestamps = short_term_timestamps[:target_position] if short_term_timestamps else []
                
                # Skip if context is empty for non-purchases
                if len(context_packages) > 0:
                    # Create enhanced sample
                    new_sample = sample.copy()
                    new_sample['short_term_packages'] = context_packages
                    new_sample['short_term_events'] = context_events
                    if context_timestamps:
                        new_sample['short_term_timestamps'] = context_timestamps
                    
                    # Set target
                    new_sample['purchased_package'] = target_package
                    new_sample['is_purchase'] = False
                    new_sample['target_intent'] = target_intent
                    new_sample['target_position'] = target_position
                    new_sample['context_length'] = len(context_packages)
                    
                    # Enhanced flags for metrics
                    new_sample['has_checkout'] = (target_intent == 'InitiateCheckout')
                    new_sample['has_add_to_cart'] = (target_intent == 'AddToCart')
                    
                    # Data leakage check
                    if target_package in context_packages:
                        data_leakage_check += 1
                    
                    intent_samples.append(new_sample)
                    intent_stats[target_intent] += 1
                    total_samples_created += 1
    
    print(f"Created {len(intent_samples)} intent-based samples from {len(samples)} original samples")
    print("Intent distribution:")
    for intent, count in sorted(intent_stats.items(), key=lambda x: intent_hierarchy.get(x[0], 0), reverse=True):
        percentage = (count / len(intent_samples)) * 100 if intent_samples else 0
        print(f"  {intent}: {count} ({percentage:.1f}%)")
    
    print(f"Normal user behavior: {data_leakage_check}/{len(intent_samples)} samples have target in context ({data_leakage_check/len(intent_samples)*100:.1f}%) - this is expected")
    
    # Detailed purchase preservation check
    intent_purchase_samples = intent_stats.get('Purchase', 0)
    print(f"Purchase processing stats:")
    print(f"  Original purchase samples in input: {original_purchase_samples}")
    print(f"  Purchase samples processed by Strategy 1: {purchase_samples_processed}")
    print(f"  Final purchase samples in output: {intent_purchase_samples}")
    
    if intent_purchase_samples < original_purchase_samples:
        print(f"⚠️  Lost {original_purchase_samples - intent_purchase_samples} purchase samples!")
        print(f"    Strategy 1 processed: {purchase_samples_processed}")
        print(f"    Strategy 2 created: {intent_purchase_samples - purchase_samples_processed}")
    else:
        print(f"✅ All purchases preserved (or gained from multi-purchase sessions)")
    
    return intent_samples


def create_convergence_plot(pretrain_history, finetune_history, output_dir):
    """
    Create a visualization of purchase recall convergence during pre-training and fine-tuning
    
    Args:
        pretrain_history: List of metric dicts from pre-training phase
        finetune_history: List of metric dicts from fine-tuning phase
        output_dir: Directory to save the plot
    """
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Combine histories for continuous x-axis
    all_history = pretrain_history + finetune_history
    
    if not all_history:
        print("No metrics history to plot!")
        return
    
    # Extract epochs and metrics
    epochs = [m['epoch'] for m in all_history]
    recall_10 = [m['purchase_recall@10'] * 100 for m in all_history]
    recall_20 = [m['purchase_recall@20'] * 100 for m in all_history]
    recall_50 = [m['purchase_recall@50'] * 100 for m in all_history]
    item_coverage = [m['item_coverage@20'] * 100 for m in all_history]
    
    # Find transition point between pretrain and finetune
    pretrain_epochs = len(pretrain_history)
    
    # Plot 1: Purchase Recall@K
    ax1.plot(epochs, recall_10, 'o-', label='Recall@10', linewidth=2, markersize=6)
    ax1.plot(epochs, recall_20, 's-', label='Recall@20', linewidth=2, markersize=6)
    ax1.plot(epochs, recall_50, '^-', label='Recall@50', linewidth=2, markersize=6)
    
    # Add vertical line at phase transition
    if pretrain_epochs > 0:
        ax1.axvline(x=pretrain_epochs + 0.5, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax1.text(pretrain_epochs/2, ax1.get_ylim()[1]*0.95, 'Pre-training', 
                ha='center', va='top', fontsize=12, fontweight='bold')
        ax1.text(pretrain_epochs + (len(epochs)-pretrain_epochs)/2, ax1.get_ylim()[1]*0.95, 'Fine-tuning', 
                ha='center', va='top', fontsize=12, fontweight='bold')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Purchase Recall (%)', fontsize=12)
    ax1.set_title('Purchase Recall Convergence During Training', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, len(epochs) + 0.5)
    
    # Format y-axis as percentage
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    
    # Plot 2: Item Coverage@20
    ax2.plot(epochs, item_coverage, 'g-', label='Item Coverage@20', linewidth=2, marker='o', markersize=6)
    
    # Add vertical line at phase transition
    if pretrain_epochs > 0:
        ax2.axvline(x=pretrain_epochs + 0.5, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Item Coverage (%)', fontsize=12)
    ax2.set_title('Item Coverage@20 During Training', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.5, len(epochs) + 0.5)
    
    # Format y-axis as percentage
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    
    # Add annotations for best values
    best_recall20_idx = np.argmax(recall_20)
    best_recall20 = recall_20[best_recall20_idx]
    best_recall20_epoch = epochs[best_recall20_idx]
    
    ax1.annotate(f'Best: {best_recall20:.2f}%', 
                xy=(best_recall20_epoch, best_recall20),
                xytext=(best_recall20_epoch + 2, best_recall20 + 2),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10, fontweight='bold')
    
    best_coverage_idx = np.argmax(item_coverage)
    best_coverage_val = item_coverage[best_coverage_idx]
    best_coverage_epoch = epochs[best_coverage_idx]
    
    ax2.annotate(f'Best: {best_coverage_val:.2f}%', 
                xy=(best_coverage_epoch, best_coverage_val),
                xytext=(best_coverage_epoch + 2, best_coverage_val + 2),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'purchase_recall_convergence.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  - Saved convergence plot to: {plot_path}")
    
    # Also save as PDF for publication quality
    pdf_path = os.path.join(output_dir, 'purchase_recall_convergence.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"  - Saved PDF version to: {pdf_path}")
    
    plt.close()
    
    # Save metrics history as CSV for further analysis
    if all_history:
        df = pd.DataFrame(all_history)
        csv_path = os.path.join(output_dir, 'training_metrics_history.csv')
        df.to_csv(csv_path, index=False)
        print(f"  - Saved metrics history to: {csv_path}")


def get_pretrain_finetune_config(mode="balanced"):
    """Get configuration optimized for pre-training + fine-tuning strategy"""
    
    configs = {
        "fastest": {
            "pretrain_epochs": 5,
            "finetune_epochs": 10,
            "batch_size_multiplier": 2.0,
            "gradient_accumulation_steps": 1,
            "embedding_dim": 64,
            "hidden_dim": 128,
            "pretrain_lr": 2e-4,
            "finetune_lr": 1e-4,
            "eval_every": 3,
            "early_stopping_patience": 5,
            "dropout": 0.2,
            "use_amp": True,
            "max_package_count": 5000  # Limit for speed
        },
        "balanced": {
            "pretrain_epochs": 6,
            "finetune_epochs": 25,
            "batch_size_multiplier": 1.5,
            "gradient_accumulation_steps": 2,
            "embedding_dim": 128,
            "hidden_dim": 256,
            "pretrain_lr": 2e-4,
            "finetune_lr": 5e-5,
            "user_lr_multiplier": 2.0,
            "eval_every": 1,
            "early_stopping_patience": 8,
            "dropout": 0.5,
            "use_amp": True,
            "max_package_count": None
        },
        "accurate": {
            "pretrain_epochs": 25,
            "finetune_epochs": 40,
            "batch_size_multiplier": 1.0,
            "gradient_accumulation_steps": 4,
            "embedding_dim": 256,
            "hidden_dim": 512,
            "pretrain_lr": 8e-5,
            "finetune_lr": 3e-5,
            "eval_every": 5,
            "early_stopping_patience": 12,
            "dropout": 0.3,
            "use_amp": True,
            "max_package_count": None
        },
        "apple_silicon": {
            "pretrain_epochs": 50,
            "finetune_epochs": 10,
            "batch_size_multiplier": 1.0,
            "gradient_accumulation_steps": 2,
            "embedding_dim": 128,
            "hidden_dim": 256,
            "pretrain_lr": 2e-4,
            "finetune_lr": 5e-5,
            "user_lr_multiplier": 2.0,
            "eval_every": 1,
            "early_stopping_patience": 6,
            "dropout": 0.5,
            "use_amp": False,  # AMP can be problematic on MPS
            "max_package_count": None,
            "num_workers": 0  # Avoid multiprocessing issues on Apple Silicon
        }
    }
    
    return configs.get(mode, configs["balanced"])


def main(performance_config=None, dataset="13months", event_data_path=None):
    """Main function implementing enhanced pre-training + fine-tuning strategy
    
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
    
    # Create memory optimizer for evaluation
    memory_optimizer = MemoryOptimizer(device)
    
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
            '13months': 'data/13_months_new2_clean.parquet',
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
        feed_data_path=package_data_path,
        cache_dir='data/cache',
        load_coordinates=True,
        load_embeddings=True,
        api_key=os.environ.get("OPENAI_API_KEY"),
        embedding_model='text-embedding-3-small',
        use_reduced_embeddings=performance_config.get("use_reduced_embeddings", True)
    )
    
    session_processor = SessionProcessor2(
        event_data_path=event_data_path,
        cache_dir='data/cache',
        session_timeout_hours=30,  # Reasonable timeout for multi-day research
        min_interactions=8,  # Match working script
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
        base_batch_size = 128
        num_workers = performance_config.get("num_workers", 0)
    else:  # cuda
        base_batch_size = 128
        num_workers = 4
    
    # If direct batch_size is specified, use it, otherwise calculate from multiplier
    if "batch_size" in performance_config:
        batch_size = performance_config["batch_size"]
    else:
        batch_size = int(base_batch_size * performance_config.get("batch_size_multiplier", 1.0))
    
    # Extract configuration parameters for model training
    pretrain_epochs = performance_config.get("pretrain_epochs", 15)
    finetune_epochs = performance_config.get("finetune_epochs", 25)
    embedding_dim = performance_config.get("embedding_dim", 128)
    hidden_dim = performance_config.get("hidden_dim", 256)
    pretrain_lr = performance_config.get("pretrain_lr", 1e-4)
    finetune_lr = performance_config.get("finetune_lr", 5e-5)
    dropout = performance_config.get("dropout", 0.35)
    eval_every = performance_config.get("eval_every", 3)
    early_stopping_patience = performance_config.get("early_stopping_patience", 8)
    gradient_accumulation_steps = performance_config.get("gradient_accumulation_steps", 2)
    use_amp = performance_config.get("use_amp", True)
    
    print(f"\nPerformance configuration:")
    print(f"  Mode: {performance_config.get('mode', 'balanced')}")
    print(f"  Batch size: {batch_size}")
    print(f"  Pre-train epochs: {pretrain_epochs}")
    print(f"  Fine-tune epochs: {finetune_epochs}")
    print(f"  Pre-train LR: {pretrain_lr}")
    print(f"  Fine-tune LR: {finetune_lr}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Dropout: {dropout}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Use AMP: {use_amp}")
    
    # Load and process data
    print("\nLoading and processing data...")
    package_processor.load_data()
    package_processor.create_mappings()
    
    session_processor.load_data()
    session_processor.create_mappings()
    
    # Prepare training samples
    print("Preparing enhanced training samples...")
    training_samples = session_processor.prepare_enhanced_training_data()
    
    # Apply session length filtering - quality improves with longer sessions
    quality_samples = filter_by_min_session_length(training_samples, min_session_length=2)
    
    # Filter items by frequency - focus on packages with sufficient data
    filtered_samples, valid_packages = filter_items_by_frequency(quality_samples, min_frequency=5)
    
    # Validate that we have purchase events
    purchase_count = sum(1 for s in filtered_samples if s.get('is_purchase', False))
    if purchase_count == 0:
        raise ValueError("No purchase events found in the filtered samples. Cannot train the model.")
    
    print(f"✅ After filtering: {len(filtered_samples):,} samples with {purchase_count:,} purchases")
    
    # Create intent-based samples for pre-training
    event_to_idx = session_processor.get_idx_mappings()['event_to_idx']
    intent_samples = create_intent_samples_enhanced(filtered_samples, event_to_idx)
    
    # IMPORTANT: Update user mappings to include ALL users from samples
    print("📊 Updating user mappings to include all sample users...")
    all_users = set()
    for sample in intent_samples:
        all_users.add(sample['user_id'])
    
    # Check if we have unmapped users
    user_to_idx = session_processor.user_to_idx
    unmapped_users = all_users - set(user_to_idx.keys())
    if unmapped_users:
        print(f"Found {len(unmapped_users)} unmapped users, adding to mapping...")
        # Add them to the mapping
        max_idx = max(user_to_idx.values()) if user_to_idx else 0
        for user_id in unmapped_users:
            max_idx += 1
            user_to_idx[user_id] = max_idx
        
        # Update session_processor's mapping too
        session_processor.user_to_idx = user_to_idx
        print(f"Updated user mappings. Total users: {len(user_to_idx)}")
    else:
        print("All users already mapped ✓")
    
    # Split data temporally
    print("Splitting data temporally...")
    train_samples, test_samples, split_date = time_based_split_year(
        intent_samples, train_ratio=0.91
    )
    
    print(f"Train samples: {len(train_samples):,}")
    print(f"Test samples: {len(test_samples):,}")
    print(f"Split date: {split_date}")
    
    # Get actual dimensions from mappings (add 1 because indices start at 1, not 0)
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
    package_tensors = package_processor.prepare_package_tensors()
    actual_embedding_dim = package_tensors['title_embeddings'].shape[1]
    print(f"Detected title embedding dimension: {actual_embedding_dim}")
    
    # Create Enhanced NATR model config
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
    
    # Create Enhanced NATR model
    print("\n🌟 Creating Enhanced NATR model with 6 views and user price bias...")
    print("  Views: Title, Coordinates, Country, Category/Theme, Price (w/ user bias), Events")
    model = NATREnhanced(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {get_model_size(model):.0f} MB")
    
    # Initialize popularity bias based on item frequencies in training data
    if hasattr(model, 'initialize_popularity'):
        print("Calculating purchase frequencies for popularity initialization...")
        package_counts = Counter()
        for sample in train_samples:
            if sample.get('is_purchase', False):
                package_counts[sample['purchased_package']] += 1
        
        # Create frequency tensor
        item_frequencies = torch.zeros(num_packages, device=device)
        for package_id, count in package_counts.items():
            if isinstance(package_id, str):
                try:
                    package_id = int(package_id)
                except ValueError:
                    continue
            if 0 <= package_id < num_packages:
                item_frequencies[package_id] = count
        
        max_freq = item_frequencies.max().item()
        model.initialize_popularity(item_frequencies)
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
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, test_loader = create_dataloaders(
        train_samples, test_samples, package_processor, session_processor,
        batch_size=batch_size, 
        num_workers=num_workers,
        use_weighted_sampling=False
    )
    
    print(f"Train loader: {len(train_loader.dataset):,} samples ({len(train_loader):,} batches)")
    print(f"Test loader: {len(test_loader.dataset):,} samples ({len(test_loader):,} batches)")
    
    # Create loss functions
    # Pre-training: Use SoftLabelLoss with intent hierarchy weights
    class IntentSoftLabelLoss(nn.Module):
        def __init__(self, intent_weights=None):
            super().__init__()
            # Use your intent weights from the script
            self.intent_weights = intent_weights or {
                'Purchase': 4.0,
                'InitiateCheckout': 0.12, 
                'AddToCart': 0.05,
                'ViewContent': 0.001
            }
            print(f"IntentSoftLabelLoss weights: {self.intent_weights}")
            
        def forward(self, predictions, targets, is_purchase=None, has_checkout=None, has_add_to_cart=None, target_intent=None):
            """
            Weighted loss based on intent hierarchy from your intent_weights
            Higher weight for more valuable events (Purchase > Checkout > AddToCart > View)
            """
            ce_loss = F.cross_entropy(predictions, targets, reduction='none')
            
            # Create weights based on intent flags
            weights = torch.ones_like(ce_loss, dtype=torch.float)
            
            if is_purchase is not None:
                # Use the intent hierarchy weights
                for i in range(len(weights)):
                    if is_purchase[i]:
                        weights[i] = self.intent_weights['Purchase']
                    elif has_checkout is not None and has_checkout[i]:
                        weights[i] = self.intent_weights['InitiateCheckout']
                    elif has_add_to_cart is not None and has_add_to_cart[i]:
                        weights[i] = self.intent_weights['AddToCart']
                    else:
                        weights[i] = self.intent_weights['ViewContent']
            
            weighted_loss = ce_loss * weights
            return weighted_loss.mean()
    
    pretrain_criterion = IntentSoftLabelLoss().to(device)
    
    # Fine-tuning: Use NaturalPurchaseLoss for purchase prediction
    finetune_criterion = NaturalPurchaseLoss(
        purchase_boost=5.0
    ).to(device)
    
    print(f"\nLoss functions configured:")
    print(f"  Pre-training: IntentSoftLabelLoss with intent hierarchy weights")
    print(f"  Fine-tuning: NaturalPurchaseLoss for purchase prediction")
    
    # Training tracking
    best_purchase_recall = 0.0
    best_mrr = 0.0
    best_item_coverage = 0.0
    training_results = []
    
    # Metrics tracking for visualization
    pretrain_metrics_history = []
    finetune_metrics_history = []
    
    # Create checkpoints directory
    os.makedirs('checkpoints/natr_enhanced', exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🚀 PHASE 1: PRE-TRAINING Enhanced NATR")
    print(f"{'='*60}")
    
    # Create optimizer for pre-training with enhanced user learning (following working script pattern)
    print("Creating custom optimizer with enhanced user learning for pre-training...")
    
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
    
    # Create optimizer with different learning rates for enhanced user learning
    user_lr = pretrain_lr * 2.0  # 2x learning rate for user embeddings
    user_transform_lr = pretrain_lr * 10.0  # 10x learning rate for user transform
    user_price_lr = pretrain_lr * 3.0  # 3x learning rate for user price bias
    
    pretrain_optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': pretrain_lr, 'weight_decay': 1e-5},
        {'params': user_params, 'lr': user_lr, 'weight_decay': 1e-6},  # Less regularization for users
        {'params': user_transform_params, 'lr': user_transform_lr, 'weight_decay': 1e-6},
        {'params': user_price_params, 'lr': user_price_lr, 'weight_decay': 1e-6}
    ], eps=1e-8)
    
    print(f"  Base learning rate: {pretrain_lr:.2e}")
    print(f"  User embedding LR: {user_lr:.2e} ({sum(p.numel() for p in user_params):,} params)")
    print(f"  User transform LR: {user_transform_lr:.2e} ({sum(p.numel() for p in user_transform_params):,} params)")
    print(f"  User price bias LR: {user_price_lr:.2e} ({sum(p.numel() for p in user_price_params):,} params)")
    print(f"  Other params LR: {pretrain_lr:.2e} ({sum(p.numel() for p in other_params):,} params)")
    
    # Learning rate scheduler
    pretrain_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        pretrain_optimizer, mode='max', factor=0.5, patience=3
    )
    
    pretrain_start_time = time.time()
    
    for epoch in range(pretrain_epochs):
        print(f"\n--- Pre-training Epoch {epoch+1}/{pretrain_epochs} ---")
        
        # Training
        train_loss = train_epoch(
            model, train_loader, pretrain_optimizer, pretrain_criterion, device, epoch,
            accumulation_steps=gradient_accumulation_steps,
            use_amp=use_amp
        )
        
        print(f"Pre-training loss: {train_loss:.4f}")
        
        # Evaluation every few epochs
        if (epoch + 1) % eval_every == 0:
            print(f"\n📊 Evaluating pre-training progress...")
            
            metrics = evaluate_with_unified_metrics(model, test_loader, device, memory_optimizer, k_values=[10, 20, 50])
            
            current_purchase_recall = metrics.get('purchase_recall@20', 0)
            current_mrr = metrics.get('purchase_mrr', 0)
            current_item_coverage = metrics.get('item_coverage@20', 0)
            
            print(f"  Purchase Recall@10: {metrics.get('purchase_recall@10', 0)*100:.2f}%")
            print(f"  Purchase Recall@20: {current_purchase_recall*100:.2f}%")
            print(f"  Checkout Recall@20: {metrics.get('checkout_recall@20', 0)*100:.2f}%")
            print(f"  Item Coverage@20: {current_item_coverage*100:.2f}%")
            print(f"  Purchase MRR: {current_mrr:.4f}")
            
            # Save metrics for visualization
            pretrain_metrics_history.append({
                'epoch': epoch + 1,
                'phase': 'pretrain',
                'purchase_recall@10': metrics.get('purchase_recall@10', 0),
                'purchase_recall@20': current_purchase_recall,
                'purchase_recall@50': metrics.get('purchase_recall@50', 0),
                'purchase_mrr': current_mrr,
                'item_coverage@20': current_item_coverage,
                'checkout_recall@20': metrics.get('checkout_recall@20', 0),
                'train_loss': train_loss
            })
            
            # Scheduler step
            pretrain_scheduler.step(current_mrr)
            
            # Save best pre-training model
            if current_purchase_recall > best_purchase_recall:
                best_purchase_recall = current_purchase_recall
                best_mrr = current_mrr
                best_item_coverage = current_item_coverage
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': pretrain_optimizer.state_dict(),
                    'epoch': epoch,
                    'metrics': metrics,
                    'config': config.__dict__
                }, 'checkpoints/natr_enhanced/pretrained_model.pth')
                
                print(f"💾 Saved best pre-training model (Purchase Recall@20: {current_purchase_recall*100:.2f}%)")
    
    pretrain_time = time.time() - pretrain_start_time
    print(f"\nPre-training completed in {pretrain_time/60:.1f} minutes")
    print(f"Best pre-training Purchase Recall@20: {best_purchase_recall*100:.2f}%")
    
    print(f"\n{'='*60}")
    print(f"🚀 PHASE 2: FINE-TUNING Enhanced NATR")
    print(f"{'='*60}")
    
    # Create optimizer for fine-tuning with enhanced user learning (same pattern as pre-training)
    print("Creating custom optimizer with enhanced user learning for fine-tuning...")
    
    # Use same learning rate multipliers for fine-tuning
    finetune_user_lr = finetune_lr * 2.0  # 2x learning rate for user embeddings
    finetune_user_transform_lr = finetune_lr * 10.0  # 10x learning rate for user transform
    finetune_user_price_lr = finetune_lr * 3.0  # 3x learning rate for user price bias
    
    finetune_optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': finetune_lr, 'weight_decay': 1e-5},
        {'params': user_params, 'lr': finetune_user_lr, 'weight_decay': 1e-6},  # Less regularization for users
        {'params': user_transform_params, 'lr': finetune_user_transform_lr, 'weight_decay': 1e-6},
        {'params': user_price_params, 'lr': finetune_user_price_lr, 'weight_decay': 1e-6}
    ], eps=1e-8)
    
    print(f"  Base fine-tune LR: {finetune_lr:.2e}")
    print(f"  User embedding LR: {finetune_user_lr:.2e}")
    print(f"  User transform LR: {finetune_user_transform_lr:.2e}")
    print(f"  User price bias LR: {finetune_user_price_lr:.2e}")
    
    # Reset scheduler for fine-tuning
    finetune_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        finetune_optimizer, mode='max', factor=0.7, patience=5
    )
    
    finetune_start_time = time.time()
    patience_counter = 0
    
    for epoch in range(finetune_epochs):
        print(f"\n--- Fine-tuning Epoch {epoch+1}/{finetune_epochs} ---")
        
        # Create purchase-only dataset for fine-tuning
        purchase_train_samples = [s for s in train_samples if s.get('is_purchase', False)]
        
        if epoch == 0:  # Only print once
            print(f"  Fine-tuning with purchase-only data: {len(purchase_train_samples):,} samples")
        
        # Create new dataloader with purchase samples only
        purchase_train_loader, _ = create_dataloaders(
            purchase_train_samples, test_samples, package_processor, session_processor,
            batch_size=batch_size, 
            num_workers=num_workers,
            use_weighted_sampling=False
        )
        
        if epoch == 0:  # Only print once
            print(f"  Purchase train loader: {len(purchase_train_loader.dataset):,} samples ({len(purchase_train_loader):,} batches)")
        
        # Training
        train_loss = train_epoch(
            model, purchase_train_loader, finetune_optimizer, finetune_criterion, device, 
            epoch + pretrain_epochs,
            accumulation_steps=gradient_accumulation_steps,
            use_amp=use_amp
        )
        
        print(f"Fine-tuning loss: {train_loss:.4f}")
        
        # Evaluation
        if (epoch + 1) % eval_every == 0:
            print(f"\n📊 Evaluating fine-tuning progress...")
            
            metrics = evaluate_with_unified_metrics(model, test_loader, device, memory_optimizer, k_values=[10, 20, 50])
            
            current_purchase_recall = metrics.get('purchase_recall@20', 0)
            current_mrr = metrics.get('purchase_mrr', 0)
            current_item_coverage = metrics.get('item_coverage@20', 0)
            
            print(f"  Purchase Recall@10: {metrics.get('purchase_recall@10', 0)*100:.2f}%")
            print(f"  Purchase Recall@20: {current_purchase_recall*100:.2f}%")
            print(f"  Checkout Recall@20: {metrics.get('checkout_recall@20', 0)*100:.2f}%")
            print(f"  Item Coverage@20: {current_item_coverage*100:.2f}%")
            print(f"  Purchase MRR: {current_mrr:.4f}")
            
            # Save metrics for visualization
            finetune_metrics_history.append({
                'epoch': pretrain_epochs + epoch + 1,
                'phase': 'finetune',
                'purchase_recall@10': metrics.get('purchase_recall@10', 0),
                'purchase_recall@20': current_purchase_recall,
                'purchase_recall@50': metrics.get('purchase_recall@50', 0),
                'purchase_mrr': current_mrr,
                'item_coverage@20': current_item_coverage,
                'checkout_recall@20': metrics.get('checkout_recall@20', 0),
                'train_loss': train_loss
            })
            
            # Scheduler step
            finetune_scheduler.step(current_purchase_recall)
            
            # Track best results
            if current_purchase_recall > best_purchase_recall:
                best_purchase_recall = current_purchase_recall
                best_mrr = current_mrr
                best_item_coverage = current_item_coverage
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': finetune_optimizer.state_dict(),
                    'epoch': epoch + pretrain_epochs,
                    'metrics': metrics,
                    'config': config.__dict__
                }, 'checkpoints/natr_enhanced/finetuned_model.pth')
                
                print(f"💾 Saved best fine-tuned model (Purchase Recall@20: {current_purchase_recall*100:.2f}%)")
            else:
                patience_counter += 1
                print(f"⏳ No improvement for {patience_counter} evaluation(s)")
                
                if patience_counter >= early_stopping_patience:
                    print(f"🛑 Early stopping triggered")
                    break
            
            # Record results
            training_results.append({
                'phase': 'finetune',
                'epoch': epoch + pretrain_epochs,
                'purchase_recall@20': current_purchase_recall,
                'mrr': current_mrr,
                'is_best': current_purchase_recall == best_purchase_recall
            })
    
    finetune_time = time.time() - finetune_start_time
    total_training_time = pretrain_time + finetune_time
    
    print(f"\n{'='*60}")
    print(f"🎉 Enhanced NATR Training Complete!")
    print(f"{'='*60}")
    print(f"⏱️  Pre-training time: {pretrain_time/60:.1f} minutes")
    print(f"⏱️  Fine-tuning time: {finetune_time/60:.1f} minutes")
    print(f"⏱️  Total training time: {total_training_time/60:.1f} minutes")
    print(f"🏆 Best Purchase Recall@20: {best_purchase_recall*100:.2f}%")
    print(f"🏆 Best Item Coverage@20: {best_item_coverage*100:.2f}%")
    print(f"🏆 Best MRR: {best_mrr:.4f}")
    
    # Save model info
    model_info = {
        'model_type': 'natr_enhanced_7views_user_price_bias',
        'dataset': dataset,
        'event_data_path': event_data_path,
        'config': config.__dict__,
        'performance_config': performance_config,
        'best_purchase_recall@20': best_purchase_recall,
        'best_item_coverage@20': best_item_coverage,
        'best_mrr': best_mrr,
        'total_training_time_minutes': total_training_time/60,
        'pretrain_time_minutes': pretrain_time/60,
        'finetune_time_minutes': finetune_time/60,
        'num_users': num_users,
        'num_packages': num_packages,
        'training_samples': len(train_samples),
        'test_samples': len(test_samples),
        'training_strategy': 'pretrain_finetune_enhanced',
        'primary_metric': 'purchase_recall@20',
        'tiebreaker_metric': 'mrr',
        'split_date': split_date,
        'train_ratio': 0.91,
        'model_features': [
            'events_as_7th_view',
            'user_price_bias', 
            'recent_popularity_boost_20days',
            'intent_based_sampling',
            'pretrain_finetune'
        ]
    }
    
    # Save model info to appropriate directory
    output_dir = 'output/model_info' if dataset == '13months' else 'output/model_info_2months'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'model_info_enhanced_pretrain_finetune.json'), 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("\nFiles saved:")
    print("  - checkpoints/natr_enhanced/pretrained_model.pth")
    print("  - checkpoints/natr_enhanced/finetuned_model.pth")  
    print("  - output/model_info/model_info_enhanced_pretrain_finetune.json")
    
    # Create visualization of purchase recall convergence
    print("\n📊 Creating purchase recall convergence visualization...")
    create_convergence_plot(pretrain_metrics_history, finetune_metrics_history, output_dir)
    
    # Also save metrics to graphs_data for comparison visualization
    graphs_output_dir = 'output/graphs_data'
    os.makedirs(graphs_output_dir, exist_ok=True)
    
    # Combine pretrain and finetune histories
    all_metrics = pretrain_metrics_history + finetune_metrics_history
    
    # Save as JSON for comparison visualization
    model_name = "natr_enhanced_pretrain_finetune"
    metrics_file = os.path.join(graphs_output_dir, f'{model_name}_metrics_history.json')
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"📊 Saved enhanced metrics to: {metrics_file}")


if __name__ == "__main__":
    import sys
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train Enhanced NATR model with pre-training and fine-tuning strategy')
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
    if is_mps and args.mode != 'apple_silicon':
        print("🍎 Apple Silicon detected! Consider using --mode apple_silicon for optimized settings")
    
    # Build performance config from arguments
    performance_config = set_performance_mode(args.mode)
    
    # Override specific parameters if provided
    if args.batch_size is not None:
        performance_config['batch_size'] = args.batch_size
    if args.pretrain_epochs is not None:
        performance_config['pretrain_epochs'] = args.pretrain_epochs
    if args.finetune_epochs is not None:
        performance_config['finetune_epochs'] = args.finetune_epochs
    if args.workers is not None:
        performance_config['num_workers'] = args.workers
    if args.learning_rate is not None:
        performance_config['pretrain_lr'] = args.learning_rate
        performance_config['finetune_lr'] = args.learning_rate * 0.5  # Half for fine-tuning
    
    # Run training
    main(performance_config, dataset=args.dataset, event_data_path=args.event_data)
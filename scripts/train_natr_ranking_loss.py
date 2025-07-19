"""
NATR training with Ranking Loss approach
Based on the successful contrastive learning that achieved 27.44%
Focuses purely on purchase prediction with ranking margins
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
from models.natr import NATR, NATRConfig
from utils.session_processor import SessionProcessor
from utils.package_processor import PackageProcessor, TravelPackageDataset
from utils.unified_metrics import UnifiedMetricsTracker, MetricsTracker, EnhancedEventMetrics
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

# Import evaluation function
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_natr import evaluate_with_unified_metrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('train_natr_ranking_loss')


class EnhancedRankingLoss(nn.Module):
    """
    Enhanced ranking loss based on successful contrastive approach
    Ensures purchased items rank in top-K with margin-based learning
    """
    def __init__(self, k=20, margin=1.0, coverage_weight=0.01, item_frequencies=None):
        super().__init__()
        self.k = k
        self.margin = margin
        self.coverage_weight = coverage_weight
        
        # Store item frequencies for popularity bias prevention
        if item_frequencies is not None:
            freq_sum = item_frequencies.sum()
            if freq_sum > 0:
                self.register_buffer('item_popularity', item_frequencies / freq_sum)
            else:
                self.item_popularity = None
        else:
            self.item_popularity = None
            
        # Track predicted items for coverage
        self.predicted_items = set()
        
    def forward(self, predictions, targets, is_purchase, has_checkout=None, has_add_to_cart=None):
        """
        Ranking loss that focuses on getting purchases into top-K
        
        Args:
            predictions: Model output logits [batch_size, num_items]
            targets: Target item indices [batch_size]
            is_purchase: Boolean mask for purchase samples [batch_size]
            has_checkout: Boolean mask for checkout samples [batch_size]
            has_add_to_cart: Boolean mask for cart samples [batch_size]
        """
        device = predictions.device
        batch_size = predictions.size(0)
        
        if targets.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Convert targets to long if needed
        if targets.dtype != torch.long:
            targets = targets.long()
        
        # Initialize total loss
        total_loss = torch.tensor(0.0, device=device)
        num_samples = 0
        
        # 1. PURCHASE RANKING LOSS (Primary objective)
        if is_purchase.any():
            purchase_indices = is_purchase.nonzero(as_tuple=True)[0]
            purchase_predictions = predictions[purchase_indices]
            purchase_targets = targets[purchase_indices]
            
            # Get scores for target items
            target_scores = torch.gather(purchase_predictions, 1, purchase_targets.unsqueeze(1)).squeeze(1)
            
            # Get K-th highest scores (threshold for top-K)
            top_k_values, _ = torch.topk(purchase_predictions, min(self.k, purchase_predictions.size(1)), dim=1)
            kth_scores = top_k_values[:, -1]
            
            # Ranking loss: target should score higher than K-th item by margin
            purchase_ranking_loss = F.relu(kth_scores - target_scores + self.margin)
            total_loss += purchase_ranking_loss.mean() * 2.0  # Higher weight for purchases
            num_samples += len(purchase_indices)
        
        # 2. CHECKOUT RANKING LOSS (Secondary objective)
        if has_checkout is not None and has_checkout.any():
            checkout_indices = has_checkout.nonzero(as_tuple=True)[0]
            checkout_predictions = predictions[checkout_indices]
            checkout_targets = targets[checkout_indices]
            
            target_scores = torch.gather(checkout_predictions, 1, checkout_targets.unsqueeze(1)).squeeze(1)
            top_k_values, _ = torch.topk(checkout_predictions, min(self.k * 2, checkout_predictions.size(1)), dim=1)
            kth_scores = top_k_values[:, -1]
            
            # Smaller margin for checkouts
            checkout_ranking_loss = F.relu(kth_scores - target_scores + self.margin * 0.5)
            total_loss += checkout_ranking_loss.mean() * 1.0
            num_samples += len(checkout_indices)
        
        # 3. ADD-TO-CART RANKING LOSS (Tertiary objective)
        if has_add_to_cart is not None and has_add_to_cart.any():
            cart_indices = has_add_to_cart.nonzero(as_tuple=True)[0]
            cart_predictions = predictions[cart_indices]
            cart_targets = targets[cart_indices]
            
            target_scores = torch.gather(cart_predictions, 1, cart_targets.unsqueeze(1)).squeeze(1)
            top_k_values, _ = torch.topk(cart_predictions, min(self.k * 3, cart_predictions.size(1)), dim=1)
            kth_scores = top_k_values[:, -1]
            
            # Even smaller margin for cart events
            cart_ranking_loss = F.relu(kth_scores - target_scores + self.margin * 0.2)
            total_loss += cart_ranking_loss.mean() * 0.5
            num_samples += len(cart_indices)
        
        # 4. REGULARIZATION TERMS
        if self.coverage_weight > 0:
            # Prediction diversity: encourage varied predictions
            pred_std = torch.std(predictions, dim=1).mean()
            diversity_loss = F.relu(0.5 - pred_std)
            total_loss += self.coverage_weight * diversity_loss
            
            # Item coverage tracking
            with torch.no_grad():
                _, top_items = torch.topk(predictions, min(self.k, predictions.size(1)), dim=1)
                for batch_items in top_items:
                    for item in batch_items:
                        self.predicted_items.add(item.item())
                
                # Reset periodically
                if len(self.predicted_items) > 10000:
                    self.predicted_items.clear()
            
            # Anti-popularity bias
            if self.item_popularity is not None:
                _, top_predicted = torch.topk(predictions, min(self.k, predictions.size(1)), dim=1)
                
                batch_avg_popularity = torch.tensor(0.0, device=device)
                for i in range(top_predicted.size(0)):
                    item_pops = self.item_popularity[top_predicted[i]]
                    batch_avg_popularity += item_pops.mean()
                
                avg_popularity = batch_avg_popularity / top_predicted.size(0)
                popularity_penalty = F.relu(avg_popularity - 0.1) * 2.0
                total_loss += self.coverage_weight * popularity_penalty
        
        # Handle empty batches
        if num_samples == 0:
            return torch.tensor(self.coverage_weight, device=device, requires_grad=True)
        
        return total_loss


def train_epoch_ranking(model, train_loader, optimizer, loss_fn, device, epoch,
                       memory_optimizer, amp_manager, gradient_accumulator):
    """Training with ranking loss"""
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
            targets = batch['purchased']['package_ids']
            
            # Get event type flags
            is_purchase = batch.get('is_purchase', torch.zeros_like(targets, dtype=torch.bool))
            has_checkout = batch.get('has_checkout', torch.zeros_like(targets, dtype=torch.bool))
            has_add_to_cart = batch.get('has_add_to_cart', torch.zeros_like(targets, dtype=torch.bool))
            
            # Calculate ranking loss
            loss = loss_fn(predictions, targets, is_purchase, has_checkout, has_add_to_cart)
        
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
    """Main training function with Ranking Loss"""
    
    # Device and memory setup
    device, device_type = detect_device()
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
        dataset_map = {
            '13months': 'data/bookit_events_data_13_months.parquet',
            '2months': 'data/bookit_events_2_months.parquet'
        }
        event_data_path = dataset_map.get(dataset)
        if not event_data_path or not os.path.exists(event_data_path):
            raise FileNotFoundError(f"Event data file not found: {event_data_path}")
    
    logger.info(f"Using event data: {event_data_path}")
    
    # Training parameters
    base_batch_size = get_optimal_batch_size(
        device_type, memory_config, model_size_mb=100, tensor_size_mb=5
    )
    
    if performance_mode == "fastest":
        num_epochs = 30
        learning_rate = 0.002
    elif performance_mode == "accurate":
        num_epochs = 100
        learning_rate = 0.0005
    else:
        num_epochs = 50
        learning_rate = 0.001
    
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
    
    session_processor = SessionProcessor(
        event_data_path=event_data_path,
        cache_dir='data/cache',
        min_interactions=5,
        max_sessions_per_user=20,
        max_samples_per_user=10
    )
    
    print(f"\nTraining parameters:")
    print(f"  Dataset: {dataset}")
    print(f"  Device: {device}")
    print(f"  Batch size: {base_batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Loss: Enhanced Ranking Loss (based on 27.44% contrastive)")
    
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
    samples = identify_event_types(samples, session_processor.event_to_idx)
    
    print("\n6. Applying data filters...")
    quality_samples = filter_by_min_session_length(samples, min_session_length=2)
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
    
    # Create datasets WITHOUT next-item transformation (use original samples)
    print("\n7. Creating datasets...")
    print("📍 Using ORIGINAL samples (not next-item) for ranking loss")
    train_loader, test_loader = create_dataloaders(
        train_samples, test_samples, package_processor, session_processor,
        batch_size=base_batch_size, num_workers=0 if device_type == 'mps' else 2,
        use_weighted_sampling=True  # Weight by event importance
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset):,} samples ({len(train_loader):,} batches)")
    print(f"  Test: {len(test_loader.dataset):,} samples ({len(test_loader):,} batches)")
    
    # Create model
    print("\n8. Creating model...")
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
        embedding_dim=256,
        dropout=0.2
    )
    
    model = NATR(config).to(device)
    model_size_mb = get_model_size(model)
    print(f"Model size: {model_size_mb} MB")
    
    # Calculate item frequencies
    print("Calculating item frequencies...")
    item_frequencies = torch.zeros(num_packages)
    for sample in train_samples:
        pkg_id = sample.get('purchased_package')
        if pkg_id and pkg_id in session_processor.package_to_idx:
            idx = session_processor.package_to_idx[pkg_id]
            # Weight by event type for ranking importance
            if sample.get('is_purchase', False):
                item_frequencies[idx] += 6.0  # Boosted weight for purchases
            elif sample.get('has_checkout', False):
                item_frequencies[idx] += 1.5  # Reduced weight for checkouts
            elif sample.get('has_add_to_cart', False):
                item_frequencies[idx] += 1.0  # Base weight for cart
            else:
                item_frequencies[idx] += 0.2  # Low weight for views
    
    print("🎯 Using proven contrastive configuration (27.44%):")
    print("   • NO popularity initialization")
    print("   • Ranking loss with margin=1.0")
    print("   • Focus on purchases, secondary on checkout/cart")
    print("   • Multi-rate optimizer: 2x user, 10x transform")
    print("   • Anti-popularity regularization")
    
    # Create optimizer with multi-rate learning
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
    
    user_lr = learning_rate * 2.0
    user_transform_lr = learning_rate * 10.0
    
    optimizer = optim.AdamW([
        {'params': other_params, 'lr': learning_rate, 'weight_decay': 1e-4},
        {'params': user_params, 'lr': user_lr, 'weight_decay': 1e-5},
        {'params': user_transform_params, 'lr': user_transform_lr, 'weight_decay': 1e-5}
    ], eps=1e-8)
    
    print(f"  Base LR: {learning_rate}, User LR: {user_lr}, Transform LR: {user_transform_lr}")
    
    # Create AMP manager and gradient accumulator
    amp_manager = AMPManager(device, memory_config)
    gradient_accumulator = GradientAccumulator(
        steps=memory_config.gradient_accumulation_steps,
        amp_manager=amp_manager
    )
    
    # Create ranking loss
    loss_fn = EnhancedRankingLoss(
        k=20,
        margin=1.0,
        coverage_weight=0.01,
        item_frequencies=item_frequencies
    )
    print(f"✓ Ranking loss initialized: top-{loss_fn.k} with margin={loss_fn.margin}")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, threshold=0.0001, min_lr=1e-6
    )
    
    # Training loop
    print("\n9. Starting training...")
    best_purchase_recall = 0.0
    best_purchase_mrr = 0.0
    patience = 10 if performance_mode != "fastest" else 5
    epochs_without_improvement = 0
    
    os.makedirs('checkpoints/natr', exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # Train
        train_loss = train_epoch_ranking(
            model, train_loader, optimizer, loss_fn, device, epoch,
            memory_optimizer, amp_manager, gradient_accumulator
        )
        
        print(f"Training loss: {train_loss:.4f}")
        
        # Evaluate
        metrics = evaluate_with_unified_metrics(
            model, test_loader, device, memory_optimizer, k_values=[5, 10, 20]
        )
        
        # Update scheduler
        current_purchase_recall_20 = metrics.get('purchase_recall@20', 0)
        scheduler.step(current_purchase_recall_20)
        
        # Print metrics
        print("\nEvaluation Results:")
        print(f"  Purchase Recall@10: {metrics.get('purchase_recall@10', 0)*100:.2f}%")
        print(f"  Purchase Recall@20: {metrics.get('purchase_recall@20', 0)*100:.2f}%")
        print(f"  Checkout Recall@20: {metrics.get('checkout_recall@20', 0)*100:.2f}%")
        print(f"  Item Coverage@20: {metrics.get('item_coverage@20', 0)*100:.2f}%")
        print(f"  Purchase MRR: {metrics.get('purchase_mrr', 0):.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Track best metrics
        current_purchase_mrr = metrics.get('purchase_mrr', 0)
        
        epsilon = 1e-6
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
            }, 'checkpoints/natr/best_model_ranking_loss.pth')
            print(f"✓ New best model saved! Recall@20: {best_purchase_recall*100:.2f}%, MRR: {best_purchase_mrr:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")
        
        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.__dict__,
                'metrics': metrics,
                'train_loss': train_loss
            }, f'checkpoints/natr/ranking_checkpoint_epoch_{epoch+1}.pth')
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping after {patience} epochs without improvement")
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
        'checkpoint_path': 'checkpoints/natr/best_model_ranking_loss.pth',
        'training_approach': 'ranking_loss_margin_based',
        'loss_function': 'EnhancedRankingLoss',
        'loss_params': {
            'k': 20,
            'margin': 1.0,
            'coverage_weight': 0.01
        },
        'no_popularity_init': True,
        'multi_rate_learning': {
            'base': learning_rate,
            'user_embeddings': user_lr,
            'user_transform': user_transform_lr
        },
        'split_date': split_date,
        'train_ratio': 0.91
    }
    
    output_dir = 'output/model_info' if dataset == '13months' else 'output/model_info_2months'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'model_info_ranking_loss.json'), 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("\nFiles saved:")
    print("  - checkpoints/natr/best_model_ranking_loss.pth")
    print("  - model_info_ranking_loss.json")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train NATR with Ranking Loss')
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
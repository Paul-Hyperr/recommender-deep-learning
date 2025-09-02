"""
Baseline NATR training script using SessionProcessor2 and NATREnhanced
Clean implementation for comparing against advanced training strategies
"""

import torch
import os
import sys

from tqdm import tqdm
import torch.backends.cudnn as cudnn
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enable optimization settings
cudnn.benchmark = True
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')

# Import all necessary components
import time
import json
from models.natr_enhanced import NATREnhanced, NATRConfig
from utils.session_processor2 import SessionProcessor2
from utils.package_processor import PackageProcessor
from utils.unified_metrics import UnifiedMetricsTracker
from utils.loss_functions import NaturalPurchaseLoss
from utils.memory_utils import (
    detect_device, create_memory_config, MemoryOptimizer, AMPManager, 
    GradientAccumulator, get_optimal_batch_size, get_model_size
)
from utils.training_utils import (
    filter_by_min_session_length, 
    filter_items_by_frequency, 
    time_based_split_year,
    analyze_data_distribution,
    create_dataloaders
)

# Import identify_event_types from training_utils to avoid circular import
from utils.training_utils import identify_event_types

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('train_natr')


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


def train_epoch(model, train_loader, optimizer, loss_fn, device, epoch,
                memory_optimizer, amp_manager, gradient_accumulator):
    """
    Simple training function for one epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Prepare for epoch
    memory_optimizer.before_epoch(epoch)
    
    # Create progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
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
            
            # Get targets
            targets = batch['purchased']['package_ids']
            
            # Ensure targets are long tensors
            if targets.dtype != torch.long:
                targets = targets.long()
                
            # Get event flags
            is_purchase = batch.get('is_purchase', torch.zeros_like(targets, dtype=torch.bool))
            has_checkout = batch.get('has_checkout', torch.zeros_like(targets, dtype=torch.bool))
            has_add_to_cart = batch.get('has_add_to_cart', torch.zeros_like(targets, dtype=torch.bool))
            
            # Calculate loss
            loss = loss_fn(predictions, targets, is_purchase, has_checkout, has_add_to_cart)
        
        # Backward pass with gradient accumulation and AMP scaling
        original_loss = gradient_accumulator.backward(loss)
        
        # Update metrics
        total_loss += original_loss.item()
        num_batches += 1
        
        # Apply gradient clipping before optimizer step
        if gradient_accumulator.should_step():
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Step optimizer if accumulation is complete
        gradient_accumulator.step(optimizer)
        
        # Memory optimization after batch
        memory_optimizer.after_batch(batch_idx)
        
        # Update progress bar
        postfix = {
            'loss': total_loss / num_batches,
            'lr': optimizer.param_groups[0]['lr']
        }
        
        if memory_optimizer.track_memory and memory_optimizer.peak_memory > 0:
            postfix['mem_mb'] = f"{memory_optimizer.peak_memory / (1024*1024):.0f}"
            
        progress_bar.set_postfix(postfix)
    
    # Handle remaining gradients if any
    if not gradient_accumulator.should_step():
        gradient_accumulator.step(optimizer)
    
    # Clean up after epoch
    memory_optimizer.after_epoch(epoch)
    
    return total_loss / num_batches


def main(performance_mode="balanced", dataset="13months", event_data_path=None, oversample_purchases=1):
    """Simple baseline NATR training with SessionProcessor2 and NATREnhanced"""
    # Detect device and create memory configuration
    device, device_type = detect_device()
    
    # Initialize metrics history for convergence visualization
    metrics_history = []
    use_oversampling = oversample_purchases > 1
    model_name = "natr_oversampled" if use_oversampling else "natr_baseline"
    print(f"📊 Tracking metrics for: {model_name}")
    memory_config = create_memory_config(device_type, performance_mode)
    
    # Create memory optimizer
    memory_optimizer = MemoryOptimizer(device, memory_config)
    
    # Log device and memory config
    logger.info(f"Using device: {device} ({device_type})")
    logger.info(f"Performance mode: {performance_mode}")
    
    # Clear old caches if needed
    if os.path.exists("data/cache/package_features.pkl"):
        os.remove("data/cache/package_features.pkl")
        
    # Clear dataset caches
    import glob
    for cache_file in glob.glob("data/cache/dataset_*.pkl"):
        os.remove(cache_file)
    
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
        learning_rate = 5e-5  # Match enhanced script learning rates
        base_batch_size = max(base_batch_size // 2, 32)  # Smaller batches for accuracy
    else:  # "balanced"
        num_epochs = 50
        learning_rate = 1e-4  # Match enhanced script pre-training learning rate
    
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
    
    session_processor = SessionProcessor2(
        event_data_path=event_data_path,
        cache_dir='data/cache',
        session_timeout_hours=30,
        min_interactions=8,
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
    if oversample_purchases > 1:
        print(f"  Purchase oversampling: {oversample_purchases}x")
    
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
    
    # Identify event types in samples
    samples = identify_event_types(samples, session_processor.event_to_idx)
    
    print("\n5b. Creating intent-based samples for consistency with other scripts...")
    # Use the same intent-based sampling as the working scripts for fair comparison
    from train_natr_enhanced_pre_finetune_fixed import create_intent_samples_enhanced
    samples = create_intent_samples_enhanced(samples, session_processor.event_to_idx)
    
    print("\n6. Applying data filters...")
    
    # Apply quality filters
    quality_samples = filter_by_min_session_length(samples, min_session_length=3)
    # Filter out single_view samples for better performance
    quality_samples = [s for s in quality_samples if s.get('intent_level', 'multi_view') != 'single_view']
    filtered_samples, valid_packages = filter_items_by_frequency(quality_samples, min_frequency=5)
    
    # Validate purchase events
    purchase_count = sum(1 for s in filtered_samples if s.get('is_purchase', False))
    if purchase_count == 0:
        raise ValueError("No purchase events found in the filtered samples. Cannot train the model.")
    
    print(f"Purchase events: {purchase_count} ({purchase_count / len(filtered_samples) * 100:.2f}% of all events)")
    
    # Time-based split
    train_samples, test_samples, split_date = time_based_split_year(filtered_samples, train_ratio=0.91)
    
    # Apply purchase oversampling if requested
    if oversample_purchases > 1:
        print(f"\n📈 Applying {oversample_purchases}x oversampling to purchase samples...")
        original_train_size = len(train_samples)
        
        # Separate purchase and non-purchase samples
        train_purchase_samples = [s for s in train_samples if s.get('is_purchase', False)]
        train_non_purchase_samples = [s for s in train_samples if not s.get('is_purchase', False)]
        
        print(f"  Original: {len(train_purchase_samples)} purchases, {len(train_non_purchase_samples)} non-purchases")
        
        # Oversample purchases
        oversampled_purchases = []
        for _ in range(oversample_purchases):
            oversampled_purchases.extend(train_purchase_samples)
        
        # Combine oversampled purchases with non-purchases
        train_samples = oversampled_purchases + train_non_purchase_samples
        
        # Shuffle to mix purchases throughout the dataset
        import random
        random.seed(42)
        random.shuffle(train_samples)
        
        print(f"  After oversampling: {len(oversampled_purchases)} purchases, {len(train_non_purchase_samples)} non-purchases")
        print(f"  Total train samples: {original_train_size} → {len(train_samples)}")
        print(f"  Purchase ratio: {len(train_purchase_samples)/original_train_size*100:.1f}% → {len(oversampled_purchases)/len(train_samples)*100:.1f}%")
    
    # Ensure test set has enough purchase samples
    purchase_samples = [s for s in test_samples if s.get('is_purchase', False)]
    if len(purchase_samples) < 50:
        print(f"Warning: Only {len(purchase_samples)} purchases in test set, adding more...")
        extra_purchases = [s for s in train_samples if s.get('is_purchase', False)][:50-len(purchase_samples)]
        test_samples.extend(extra_purchases)
        train_samples = [s for s in train_samples if s not in extra_purchases]
        print(f"Added {len(extra_purchases)} more purchase samples to test set")
    
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
    
    # Analyze distributions
    analyze_data_distribution(train_samples, "Train")
    analyze_data_distribution(test_samples, "Test")
    
    # Create datasets and dataloaders
    print("\n7. Creating datasets...")
    train_loader, test_loader = create_dataloaders(
        train_samples, test_samples, package_processor, session_processor,
        batch_size=base_batch_size, num_workers=0 if device_type == 'mps' else 2,
        use_weighted_sampling=False  # Disable weighted sampling for speed
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
    
    # Detect embedding dimension
    package_tensors = package_processor.prepare_package_tensors()
    actual_embedding_dim = package_tensors['title_embeddings'].shape[1]
    print(f"Detected title embedding dimension: {actual_embedding_dim}")
    
    # Align with enhanced pre-training configuration for fair comparison
    hidden_dim = 256
    embedding_dim = 128  # Match enhanced script embedding dimension
    dropout = 0.35  # Match enhanced script dropout for consistency
    
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
    
    # Create Enhanced NATR model with 6 views
    print("Creating Enhanced NATR model with 6 views (including events as separate view)...")
    model = NATREnhanced(config)
    model = model.to(device)
    
    # Get model size for memory calculations
    model_size_mb = get_model_size(model)
    print(f"Model size: {model_size_mb} MB")
    
    # Initialize popularity bias based on item frequencies in training data
    print("Calculating item frequencies for popularity initialization...")
    item_frequencies = torch.zeros(num_packages)
    for sample in train_samples:
        # Count purchased item
        pkg_id = sample.get('purchased_package')
        if pkg_id and pkg_id in session_processor.package_to_idx:
            idx = session_processor.package_to_idx[pkg_id]
            item_frequencies[idx] += 1
        
        # Also count items in sessions (with lower weight)
        for pkg_id in sample.get('short_term_packages', []):
            if pkg_id in session_processor.package_to_idx:
                idx = session_processor.package_to_idx[pkg_id]
                item_frequencies[idx] += 0.1  # Lower weight for non-purchased interactions
    
    # Initialize popularity in the model
    print(f"Initializing popularity bias with item frequencies (max freq: {item_frequencies.max().item():.0f})")
    model.initialize_popularity(item_frequencies.to(device))
    
    # Calculate recent popularity scores using most recent 20 days of training data
    if hasattr(model, 'set_recent_popularity'):
        print("\n🔥 Calculating recent popularity boost...")
        from utils.recent_popularity import get_recent_popularity_from_samples
        recent_popularity, stats = get_recent_popularity_from_samples(
            samples=train_samples,
            num_packages=num_packages,
            days=20  # Use most recent 20 days
        )
        model.set_recent_popularity(recent_popularity)
        print(f"🔥 Recent popularity applied with weight: {model.recent_popularity_weight}")
    else:
        print("⚠️  Model does not support recent popularity boost")
    
    # Create custom optimizer with different learning rates for user components
    print("Creating custom optimizer with enhanced user learning...")
    
    # Separate parameters for different learning rates
    user_params = []
    user_transform_params = []
    user_price_bias_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'user_encoder.user_embedding' in name:
            user_params.append(param)
        elif 'user_transform' in name:
            user_transform_params.append(param)
        elif 'user_price_bias' in name:
            user_price_bias_params.append(param)
        else:
            other_params.append(param)
    
    # Create optimizer with different learning rates
    user_lr = learning_rate * 2.0  # 2x learning rate for user embeddings
    user_transform_lr = learning_rate * 10.0  # 10x learning rate for user transform
    price_bias_lr = learning_rate * 3.0  # 3x learning rate for price bias
    
    param_groups = [
        {'params': other_params, 'lr': learning_rate, 'weight_decay': 1e-4},
        {'params': user_params, 'lr': user_lr, 'weight_decay': 1e-5},
        {'params': user_transform_params, 'lr': user_transform_lr, 'weight_decay': 1e-5}
    ]
    
    # Add price bias parameters if they exist
    if user_price_bias_params:
        param_groups.append({
            'params': user_price_bias_params, 
            'lr': price_bias_lr, 
            'weight_decay': 1e-4
        })
    
    optimizer = torch.optim.Adam(param_groups)
    
    print(f"  Base learning rate: {learning_rate}")
    print(f"  User embedding learning rate: {user_lr}")
    print(f"  User transform learning rate: {user_transform_lr}")
    if user_price_bias_params:
        print(f"  User price bias learning rate: {price_bias_lr}")
    print(f"  User embedding params: {sum(p.numel() for p in user_params)}")
    print(f"  User transform params: {sum(p.numel() for p in user_transform_params)}")
    if user_price_bias_params:
        print(f"  User price bias params: {sum(p.numel() for p in user_price_bias_params)}")
    print(f"  Other params: {sum(p.numel() for p in other_params)}")
    
    # Create AMP manager and gradient accumulator
    amp_manager = AMPManager(device, memory_config)
    gradient_accumulator = GradientAccumulator(
        steps=memory_config.gradient_accumulation_steps,
        amp_manager=amp_manager
    )
    
    # Create loss function with same weights as other scripts
    loss_fn = NaturalPurchaseLoss(
        purchase_boost=5.0  # Use default weight from loss_functions.py for consistency
    )
    
    # Set up checkpointing
    os.makedirs('checkpoints/natr', exist_ok=True)
    
    # Record training start time
    start_time = time.time()
    
    # Training loop
    print("\n9. Starting training...")
    best_purchase_recall = 0.0
    best_purchase_mrr = 0.0
    patience = 10 if performance_mode != "fastest" else 5
    epochs_without_improvement = 0
    previous_checkpoint = None  # Track previous checkpoint for cleanup
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # Disable verbose monitoring after first 3 epochs
        if epoch >= 3:
            model.monitor_user_learning = False
        
        # Training
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch,
            memory_optimizer, amp_manager, gradient_accumulator
        )
        
        print(f"Training loss: {train_loss:.4f}")
        
        # Evaluation with comprehensive metrics
        metrics = evaluate_with_unified_metrics(
            model, test_loader, device, memory_optimizer, k_values=[5, 10, 20]
        )
        
        # Print key metrics only
        print("\nEvaluation Results:")
        print(f"  Purchase Recall@10: {metrics.get('purchase_recall@10', 0)*100:.2f}%")
        print(f"  Purchase Recall@20: {metrics.get('purchase_recall@20', 0)*100:.2f}%")
        print(f"  Checkout Recall@20: {metrics.get('checkout_recall@20', 0)*100:.2f}%")
        print(f"  Item Coverage@20: {metrics.get('item_coverage@20', 0)*100:.2f}%")
        print(f"  Purchase MRR: {metrics.get('purchase_mrr', 0):.4f}")
        
        # Track best metrics - using Purchase Recall@20 as primary metric
        current_purchase_recall_20 = metrics.get('purchase_recall@20', 0)
        current_purchase_mrr = metrics.get('purchase_mrr', 0)
        
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
            
            # Save best model with performance in filename
            # Format recall with 2 decimals (e.g., 74_11 for 74.11%)
            recall_whole = int(best_purchase_recall * 100)
            recall_decimal = int((best_purchase_recall * 10000) % 100)
            checkpoint_filename = f'checkpoints/natr/best_model_{model_name}_recall{recall_whole:02d}_{recall_decimal:02d}.pth'
            
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
            print(f"  Purchase Recall@20: {best_purchase_recall*100:.2f}%, Purchase MRR: {best_purchase_mrr:.4f}")
            
            # Update previous checkpoint tracker
            previous_checkpoint = checkpoint_filename
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
    
    # Calculate training duration
    end_time = time.time()
    training_duration = end_time - start_time
    
    # Save model info with timing
    model_info = {
        'model_type': model_name,
        'training_type': 'oversampled' if use_oversampling else 'baseline',
        'oversample_factor': oversample_purchases if use_oversampling else 1,
        'config': config.__dict__,
        'best_purchase_recall@20': best_purchase_recall,
        'best_purchase_mrr': best_purchase_mrr,
        'training_duration_seconds': training_duration,
        'training_duration_minutes': training_duration / 60,
        'training_duration_formatted': f"{training_duration//3600:.0f}h {(training_duration%3600)//60:.0f}m {training_duration%60:.0f}s",
        'checkpoint_path': f'checkpoints/natr/best_model_{model_name}.pth'
    }
    
    output_dir = 'output/model_info' if dataset == '13months' else 'output/model_info_2months'
    os.makedirs(output_dir, exist_ok=True)
    
    model_info_path = os.path.join(output_dir, f'model_info_{model_name}.json')
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Save metrics history for convergence visualization
    graphs_output_dir = 'output/graphs_data'
    os.makedirs(graphs_output_dir, exist_ok=True)
    
    metrics_file = os.path.join(graphs_output_dir, f'{model_name}_metrics_history.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics_history, f, indent=2)
    print(f"📊 Saved metrics history to: {metrics_file}")
    
    # Final summary
    print("\n10. Training complete!")
    print(f"Best Purchase Recall@20: {best_purchase_recall*100:.2f}%")
    print(f"Best Purchase MRR: {best_purchase_mrr:.4f}")
    print(f"Training Duration: {model_info['training_duration_formatted']}")
    
    print("\nFiles saved:")
    print(f"  - checkpoints/natr/best_model_{model_name}.pth")
    print(f"  - {model_info_path}")


if __name__ == "__main__":
    import sys
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train baseline NATR model')
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
    parser.add_argument('--oversample-purchases', type=int, default=1,
                        help='Oversample purchase samples by this factor (default: 1 = no oversampling)')
    parser.add_argument('--oversample', action='store_true',
                        help='Enable 2x purchase oversampling (shortcut for --oversample-purchases 2)')
    
    args = parser.parse_args()
    
    # Handle --oversample flag
    if args.oversample:
        args.oversample_purchases = 2
    
    # Run training
    main(performance_mode=args.mode, dataset=args.dataset, event_data_path=args.event_data,
         oversample_purchases=args.oversample_purchases)
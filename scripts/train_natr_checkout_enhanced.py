"""
Enhanced NATR training script leveraging InitiateCheckout events as strong purchase signals
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
from utils.loss_functions import FocalLoss

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
    ensure_balanced_test_set
)


def identify_checkout_events(samples, session_processor):
    """
    Identify and mark samples that contain InitiateCheckout events
    This adds a strong purchase signal to the data beyond actual purchases
    
    Note: Addresses the edge case where a session has both InitiateCheckout and Purchase
    by ensuring we don't double-count or overweight these sessions
    """
    print("\nIdentifying samples with InitiateCheckout events...")
    
    # Get event mappings
    event_to_idx = session_processor.get_idx_mappings()['event_to_idx']
    idx_to_event = {v: k for k, v in event_to_idx.items()}
    
    checkout_event_idx = event_to_idx.get('InitiateCheckout')
    if not checkout_event_idx:
        print("Warning: 'InitiateCheckout' event not found in mapping")
        return samples
    
    print(f"InitiateCheckout event index: {checkout_event_idx}")
    
    # Process samples to identify checkout events
    enhanced_samples = []
    checkout_count = 0
    purchase_with_checkout_count = 0
    
    for sample in samples:
        # Add checkout flag
        sample_copy = sample.copy()
        
        # Check if sample has short-term events
        short_term_events = sample_copy.get('short_term_events', [])
        short_term_pkgs = sample_copy.get('short_term_packages', [])
        
        # Look for InitiateCheckout events in short-term sequence
        has_checkout = False
        checkout_package = None
        
        if short_term_events:
            # Find the most recent checkout event
            for i, event_idx in enumerate(reversed(short_term_events)):
                if event_idx == checkout_event_idx:
                    has_checkout = True
                    # Get the package associated with this checkout
                    rev_index = len(short_term_events) - 1 - i
                    if rev_index < len(short_term_pkgs):
                        checkout_package = short_term_pkgs[rev_index]
                    break
        
        # Store checkout information in the sample
        sample_copy['has_checkout'] = has_checkout
        if has_checkout and checkout_package:
            sample_copy['checkout_package'] = checkout_package
            
            # IMPORTANT: Special handling for sessions with both checkout and purchase
            if sample_copy.get('is_purchase', False):
                # This is a purchase that also had a checkout - mark it but don't modify
                # Don't override the purchased_package, as it's already set correctly
                purchase_with_checkout_count += 1
            else:
                # This is a checkout-only session - use checkout package as target
                sample_copy['purchased_package'] = checkout_package
                checkout_count += 1
        
        enhanced_samples.append(sample_copy)
    
    print(f"Identified {checkout_count} additional checkout-only samples")
    print(f"Found {purchase_with_checkout_count} samples with both purchase and checkout events")
    return enhanced_samples


class CheckoutAwareDataset(TravelPackageDataset):
    """
    Extended dataset that handles checkout signals as a special case
    This dataset adds checkout flags to model input
    """
    
    def __getitem__(self, idx):
        """Enhanced getitem that passes checkout flags to the model"""
        # Get base sample
        sample = super().__getitem__(idx)
        
        # Add checkout flag
        sample['has_checkout'] = self.samples[idx].get('has_checkout', False)
        
        return sample


def create_checkout_dataloaders(train_samples, test_samples, package_processor, session_processor, 
                        batch_size=32, num_workers=0, use_weighted_sampling=True):
    """Create optimized dataloaders for training and testing with checkout awareness
    
    This function customizes the consolidated dataloader creation for checkout-specific datasets
    """
    # Get mappings
    user_to_idx = session_processor.get_idx_mappings()['user_to_idx']
    package_to_idx = session_processor.get_idx_mappings()['package_to_idx']
    event_to_idx = session_processor.get_idx_mappings()['event_to_idx']
    
    # Create custom checkout-aware datasets
    print("Creating training dataset...")
    train_dataset = CheckoutAwareDataset(
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
    test_dataset = CheckoutAwareDataset(
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
    
    # Use consolidated utilities from training_utils.py to create the dataloaders
    # But use custom checkout-specific datasets
    from utils.training_utils import create_dataloaders
    train_loader, test_loader = create_dataloaders(
        train_samples, test_samples, 
        package_processor, session_processor,
        batch_size=batch_size, 
        num_workers=num_workers,
        use_weighted_sampling=use_weighted_sampling,
        # Override the datasets to use our checkout-aware versions
        train_dataset=train_dataset,
        test_dataset=test_dataset
    )
    
    return train_loader, test_loader


class ContrastiveEventLoss(nn.Module):
    """
    Contrastive loss with temperature scaling and event-based margins
    
    This loss function lets the model organically learn the importance of different events
    by applying contrastive margins rather than explicit weights:
    - Purchase events: No margin (strongest positive signal)
    - Checkout events: Small margin (partial positive signal)
    - Regular browsing: Full margin (negative samples)
    
    The temperature parameter controls how much to amplify small differences in similarity.
    """
    def __init__(self, temperature=0.1, purchase_margin=0.0, checkout_margin=0.3, 
                 view_margin=1.0, hard_negative_mining=True, hard_negative_ratio=0.7,
                 add_to_cart_margin=0.6):
        super().__init__()
        self.temperature = temperature  # Controls sensitivity to similarity differences
        self.purchase_margin = purchase_margin  # No margin for purchases
        self.checkout_margin = checkout_margin  # Small margin for checkout events
        self.view_margin = view_margin  # Full margin for view events
        self.add_to_cart_margin = add_to_cart_margin  # Medium margin for add to cart events
        self.hard_negative_mining = hard_negative_mining  # Whether to use hard negative mining
        self.hard_negative_ratio = hard_negative_ratio  # Proportion of hard negatives to keep
    
    def forward(self, predictions, targets, is_purchase, has_checkout=None, has_add_to_cart=None):
        """
        Forward pass with contrastive margins for different event types
        
        Args:
            predictions: Predicted scores for each package
            targets: Target package indices
            is_purchase: Boolean tensor indicating purchase samples
            has_checkout: Boolean tensor indicating checkout samples
            has_add_to_cart: Boolean tensor indicating add_to_cart samples (optional)
        
        Returns:
            Loss value
        """
        # For cross entropy calculation, we need to mask invalid indices (-1)
        valid_mask = (targets >= 0).float()
        
        # Scale logits by temperature
        # Lower temperature = more emphasis on top predictions
        logits = predictions / self.temperature
        
        # Create masks for different event types
        purchase_mask = is_purchase.float()
        
        # Initialize checkout and add_to_cart masks
        checkout_mask = torch.zeros_like(purchase_mask)
        add_to_cart_mask = torch.zeros_like(purchase_mask)
        
        # Only use checkout info if provided
        if has_checkout is not None:
            # Only count as checkout if NOT also a purchase
            checkout_mask = has_checkout.float() * (~is_purchase).float()
        
        # Only use add_to_cart info if provided
        if has_add_to_cart is not None:
            # Only count as add_to_cart if NOT also a purchase or checkout
            add_to_cart_mask = has_add_to_cart.float() * (~is_purchase).float() * (~has_checkout).float()
        
        # Regular browsing is everything else
        view_mask = 1.0 - purchase_mask - checkout_mask - add_to_cart_mask
        
        # Apply different contrastive margins based on event type
        # The margin is subtracted from the logits, making it harder to predict
        # correctly for that class. Higher margin = more penalty.
        margins = torch.zeros_like(logits)
        
        batch_size = predictions.size(0)
        for i in range(batch_size):
            target_idx = targets[i].item()
            if target_idx >= 0:  # Valid target
                # Set margin for the target item based on event type
                if purchase_mask[i] > 0:
                    # Purchase event - no/minimal margin (easiest to learn)
                    margins[i, target_idx] = self.purchase_margin
                elif checkout_mask[i] > 0:
                    # Checkout event - small margin
                    margins[i, target_idx] = self.checkout_margin
                elif add_to_cart_mask[i] > 0:
                    # Add to cart event - medium margin
                    margins[i, target_idx] = self.add_to_cart_margin
                else:
                    # View event - largest margin (hardest to learn)
                    margins[i, target_idx] = self.view_margin
        
        # Apply margins to logits
        margin_logits = logits - margins
        
        # Calculate cross entropy loss with temperature scaling
        ce_loss = nn.functional.cross_entropy(margin_logits, targets, reduction='none')
        
        # Apply hard negative mining if enabled
        if self.hard_negative_mining:
            # Sort losses in descending order, but only for view samples
            view_losses = ce_loss * view_mask
            sorted_losses, _ = torch.sort(view_losses, descending=True)
            
            # Take top N% hardest samples
            if len(sorted_losses) > 10:  # Only if we have enough samples
                cutoff_idx = int(len(sorted_losses) * self.hard_negative_ratio)
                hard_mining_threshold = sorted_losses[cutoff_idx]
                
                # Create hard mining mask:
                # - Keep all purchases
                # - Keep all checkouts
                # - Keep all add_to_carts
                # - Only keep hard negatives from view samples
                mining_mask = (
                    purchase_mask +  # Keep all purchases
                    checkout_mask +  # Keep all checkouts
                    add_to_cart_mask +  # Keep all add to carts
                    (view_mask * (view_losses >= hard_mining_threshold).float())  # Hard negatives
                )
            else:
                mining_mask = torch.ones_like(purchase_mask)
        else:
            mining_mask = torch.ones_like(purchase_mask)
        
        # Apply mining mask and valid mask
        masked_loss = ce_loss * mining_mask * valid_mask
        
        # Return mean loss
        return masked_loss.sum() / (valid_mask.sum() + 1e-6)
    
    def get_margins(self):
        """Return current margin values for reporting"""
        return {
            "purchase_margin": self.purchase_margin,
            "checkout_margin": self.checkout_margin,
            "add_to_cart_margin": self.add_to_cart_margin,
            "view_margin": self.view_margin,
            "temperature": self.temperature
        }


def evaluate_with_checkout_metrics(model, test_loader, device, k_values=[5, 10, 20]):
    """Evaluate model with enhanced checkout-specific metrics using consolidated utilities"""
    # Use the consolidated evaluate function from training_utils.py
    # This function now handles event-specific metrics including checkout events
    return evaluate(model, test_loader, device, k_values)


def get_checkout_config(mode="balanced"):
    """Get specific parameters for checkout-enhanced approach based on performance mode"""
    config = {}
    
    # Special mode for Apple Silicon (M1/M2/M3)
    is_mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    
    if mode == "apple_silicon" or (mode == "balanced" and is_mps):
        config.update({
            # Contrastive margin parameters
            "temperature": 0.1,                  # Contrastive temperature - lower = sharper distinctions
            "purchase_margin": 0.0,              # No margin for purchases (strongest signal)
            "checkout_margin": 0.3,              # Small margin for checkouts (medium signal)
            "add_to_cart_margin": 0.6,           # Medium margin for add-to-cart events
            "view_margin": 1.0,                  # Full margin for view events (weakest signal)
            "hard_negative_ratio": 0.7           # Proportion of hard negatives to keep
        })
    elif mode == "fastest":
        config.update({
            # Contrastive margin parameters - less aggressive for faster convergence
            "temperature": 0.2,                  # Higher temperature = smoother distinctions
            "purchase_margin": 0.0,              # No margin for purchases
            "checkout_margin": 0.2,              # Smaller checkout margin for faster convergence
            "add_to_cart_margin": 0.4,           # Smaller add-to-cart margin
            "view_margin": 0.8,                  # Smaller view margin
            "hard_negative_ratio": 0.5           # Fewer hard negatives for faster training
        })
    elif mode == "accurate":
        config.update({
            # Contrastive margin parameters - more aggressive for better accuracy
            "temperature": 0.05,                 # Lower temperature = sharper distinctions
            "purchase_margin": 0.0,              # No margin for purchases
            "checkout_margin": 0.25,             # Precise checkout margin
            "add_to_cart_margin": 0.5,           # Precise add-to-cart margin
            "view_margin": 1.0,                  # Full margin for views
            "hard_negative_ratio": 0.8           # More hard negatives for better discrimination
        })
    else:  # "balanced" mode (non-MPS)
        config.update({
            # Contrastive margin parameters - balanced configuration
            "temperature": 0.1,                  # Balanced temperature
            "purchase_margin": 0.0,              # No margin for purchases
            "checkout_margin": 0.3,              # Standard checkout margin
            "add_to_cart_margin": 0.6,           # Standard add-to-cart margin
            "view_margin": 1.0,                  # Standard view margin
            "hard_negative_ratio": 0.7           # Standard hard negative ratio
        })
    
    return config


def main(performance_config=None):
    """Main function implementing InitiateCheckout-enhanced training strategy
    using consolidated utilities from training_utils.py
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
    
    # Add checkout-specific parameters
    checkout_config = get_checkout_config(
        "apple_silicon" if device.type == "mps" else performance_config.get("mode", "balanced")
    )
    performance_config.update(checkout_config)
    
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
    print(f"  Purchase boost: {performance_config.get('purchase_boost', 20.0)}")
    print(f"  Checkout boost: {performance_config.get('checkout_boost', 10.0)}")
    
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
    
    # Limit samples for testing if needed
    samples = limit_samples_for_testing(samples)
    
    # Add checkout flag to samples - extract InitiateCheckout events
    print("\n6. Enhancing samples with checkout events...")
    enhanced_samples = identify_checkout_events(samples, session_processor)
    
    print("\n7. Applying data filters...")
    
    # Apply session length filtering - quality improves with longer sessions
    # Use consolidated utilities from training_utils.py
    quality_samples = filter_by_min_session_length(enhanced_samples, min_session_length=3)
    
    # Filter items by frequency - focus on packages with sufficient data
    # Use consolidated utilities from training_utils.py
    filtered_samples, valid_packages = filter_items_by_frequency(quality_samples, min_frequency=50)
    
    # Validate that we have purchase and checkout events
    purchase_count = sum(1 for s in filtered_samples if s.get('is_purchase', False))
    checkout_count = sum(1 for s in filtered_samples if not s.get('is_purchase', False) and s.get('has_checkout', False))
    
    if purchase_count == 0:
        raise ValueError("No purchase events found in the filtered samples. Cannot train the model.")
    
    print(f"Purchase events: {purchase_count} ({purchase_count / len(filtered_samples) * 100:.2f}% of all events)")
    print(f"Checkout events: {checkout_count} ({checkout_count / len(filtered_samples) * 100:.2f}% of all events)")
    print(f"Combined purchase intent: {purchase_count + checkout_count} " + 
          f"({(purchase_count + checkout_count) / len(filtered_samples) * 100:.2f}% of all events)")
    
    # Time-based split - use consolidated utilities from training_utils.py
    train_samples, test_samples = time_based_split_year(filtered_samples, train_ratio=0.93)
    
    # Use consolidated ensure_balanced_test_set utility
    # Ensure test set has enough purchases, checkouts, and add-to-cart events
    train_samples, test_samples = ensure_balanced_test_set(train_samples, test_samples, min_purchases=50)
    
    # Analyze distributions - use consolidated utility from training_utils.py
    analyze_data_distribution(train_samples, "Train")
    analyze_data_distribution(test_samples, "Test")
    
    # Create datasets and dataloaders with checkout-specific dataset
    print("\n8. Creating dataloaders...")
    train_loader, test_loader = create_checkout_dataloaders(
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
    
    # Initialize optimizer and loss with additional optimizations
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.05,  # Increased weight decay for stronger regularization (from 0.03)
        eps=1e-8  # More stable epsilon value
    )
    
    # Use a simple step scheduler with no warmup - more reliable for training
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',           # Optimize for maximizing recall
        factor=0.5,           # Halve the learning rate when plateauing
        patience=2,           # Wait 2 epochs before reducing
        threshold=0.0001,     # Minimum improvement to count as progress
        min_lr=1e-6           # Minimum learning rate
    )
    
    # Use contrastive event loss with margins for different event types
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
    early_stopping_patience = performance_config.get("patience", 8)
    
    print("\n10. Starting training...")
    # Get margin values from the loss function
    margins = loss_fn.get_margins()
    print(f"Training for {num_epochs} epochs with contrastive margins:")
    print(f"  Temperature: {margins['temperature']}")
    print(f"  Purchase margin: {margins['purchase_margin']}")
    print(f"  Checkout margin: {margins['checkout_margin']}")
    print(f"  Add to cart margin: {margins['add_to_cart_margin']}")
    print(f"  View margin: {margins['view_margin']}")
    print(f"Early stopping patience: {early_stopping_patience} epochs")
    
    best_intent_recall = 0  # Track combined purchase + checkout recall
    epochs_without_improvement = 0
    history = {'train_loss': [], 'test_metrics': []}
    
    for epoch in range(num_epochs):
        print(f"\n==== Epoch {epoch+1}/{num_epochs} ====")
        
        # Train for one epoch - use consolidated train_epoch utility
        start_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch,
            accumulation_steps=accumulation_steps,
            use_amp=use_amp
        )
        train_time = time.time() - start_time
        
        # Evaluate using our simplified evaluation function that leverages consolidated evaluate
        start_time = time.time()
        test_metrics = evaluate_with_checkout_metrics(model, test_loader, device, k_values=[5, 10, 20])
        eval_time = time.time() - start_time
        
        # Update scheduler based on intent recall metric (combined purchase + checkout)
        current_intent_recall = test_metrics['intent_recall@k'][10]
        scheduler.step(current_intent_recall)
        
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
        print(f"  Checkout Recall@10: {test_metrics['checkout_recall@k'][10]*100:.2f}%")
        print(f"  Intent (P+C) Recall@10: {test_metrics['intent_recall@k'][10]*100:.2f}%")
        print(f"  Purchase MRR: {test_metrics['purchase_mrr']:.4f}")
        print(f"  Intent MRR: {test_metrics['intent_mrr']:.4f}")
        
        # Save best model based on intent recall (combined purchase + checkout)
        if current_intent_recall > best_intent_recall:
            best_intent_recall = current_intent_recall
            epochs_without_improvement = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_intent_recall': best_intent_recall,
                'config': config.__dict__,
                'valid_packages': list(valid_packages),
                'history': history,
                'performance_config': performance_config,
                'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            }
            
            # Create checkpoints directory
            os.makedirs('checkpoints/natr', exist_ok=True)
            
            # Save to both a versioned file and the best model file
            torch.save(checkpoint, f"checkpoints/natr/checkout_model_epoch_{epoch+1}.pth")
            torch.save(checkpoint, 'checkpoints/natr/checkout_best_model.pth')
            print(f"  ✓ New best model saved! Intent Recall@10: {best_intent_recall*100:.2f}%")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epochs")
            
            # Save periodic checkpoint every 5 epochs for recovery purposes
            if (epoch + 1) % 5 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_intent_recall': best_intent_recall,
                    'config': config.__dict__,
                    'valid_packages': list(valid_packages),
                    'history': history,
                    'performance_config': performance_config,
                    'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                }
                torch.save(checkpoint, f"checkpoints/natr/checkout_checkpoint_epoch_{epoch+1}.pth")
                print(f"  ✓ Periodic checkpoint saved at epoch {epoch+1}")
            
            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping after {epoch+1} epochs without improvement")
                break
    
    # Final summary
    print("\n11. Training complete!")
    print(f"Best Intent (Purchase+Checkout) Recall@10: {best_intent_recall*100:.2f}%")
    
    # Save model info
    model_info = {
        'config': config.__dict__,
        'valid_packages': list(valid_packages),
        'best_intent_recall': best_intent_recall,
        'checkpoint_path': 'checkpoints/natr/checkout_best_model.pth',
        'package_data_path': package_data_path,
        'event_data_path': event_data_path,
        'performance_config': performance_config,
        'training_date': datetime.now().strftime("%Y-%m-%d"),
        'package_count': len(valid_packages),
        'user_count': num_users,
        'training_strategy': 'checkout_enhanced',
        'purchase_count': purchase_count,
        'checkout_count': checkout_count,
        'intent_count': purchase_count + checkout_count
    }
    
    with open('model_info_checkout_enhanced.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("\nFiles saved:")
    print("  - checkpoints/natr/checkout_best_model.pth")
    print("  - checkpoints/natr/checkout_model_epoch_X.pth")
    print("  - model_info_checkout_enhanced.json")


if __name__ == "__main__":
    import sys
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train NATR model with InitiateCheckout enhancement')
    parser.add_argument('--mode', type=str, default='balanced', 
                        choices=['fastest', 'balanced', 'accurate', 'apple_silicon'],
                        help='Performance mode (default: balanced)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size (default: determined by mode)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (default: determined by mode)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of data loading workers (default: determined by mode)')
    parser.add_argument('--purchase-boost', type=float, default=None,
                        help='Boost factor for purchase samples (default: determined by mode)')
    parser.add_argument('--checkout-boost', type=float, default=None,
                        help='Boost factor for checkout samples (default: determined by mode)')
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
    performance_config["mode"] = args.mode  # Store the mode for checkout config
    
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
    
    if args.purchase_boost is not None:
        performance_config["purchase_boost"] = args.purchase_boost
        print(f"Overriding purchase boost to {args.purchase_boost}")
    
    if args.checkout_boost is not None:
        performance_config["checkout_boost"] = args.checkout_boost
        print(f"Overriding checkout boost to {args.checkout_boost}")
    
    if args.learning_rate is not None:
        performance_config["learning_rate"] = args.learning_rate
        print(f"Overriding learning rate to {args.learning_rate}")
    
    # Start training with checkout enhancement
    main(performance_config)
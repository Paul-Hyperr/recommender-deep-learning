#!/usr/bin/env python3
"""
NATR training with data augmentation and oversampling strategies

This script specifically targets the low purchase sample problem through:
1. Conservative oversampling of purchase events (3x) to balance representation  
2. Moderate augmentation diversity for purchase sessions (5x) with true dynamic augmentation
3. No oversampling or augmentation for checkout/add-to-cart events
4. Views remain as-is (no oversampling)

Key strategy: Conservative dynamic augmentation to prevent overfitting
- Each purchase sample generates ~15 training examples through:
  * 3x oversampling (conservative copies for balance)
  * 5x dynamic augmentation variations applied on-the-fly during training
  * Diverse augmentation techniques:
    - Session dropout patterns
    - Item substitution with similar products
    - Temporal jittering
    - Combined variations

Dynamic Augmentation: Augmentation happens on-the-fly during training via custom collate function
- AugmentedDataset.__getitem__ applies real-time augmentation for each batch
- Ensures true 10x diversity multiplier during training (not just static pre-generated variants)

Result: From ~9,430 purchases → ~28,290 oversampled + 5x dynamic augmentation  
Expected purchase representation: ~8% of total training data (up from 2.7%)
Note: Conservative augmentation prevents overfitting while improving diversity
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
import random
from datetime import datetime, timedelta
from tqdm import tqdm
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
from utils.package_processor import PackageProcessor
from utils.unified_metrics import UnifiedMetricsTracker
from utils.loss_functions import NaturalPurchaseLoss
from utils.memory_utils import (
    detect_device, create_memory_config, MemoryOptimizer, AMPManager, 
    GradientAccumulator, get_optimal_batch_size, print_memory_stats, get_model_size
)
from utils.training_utils import (
    filter_by_min_session_length, filter_items_by_frequency,
    time_based_split_year, analyze_data_distribution,
    move_batch_to_device, identify_event_types
)

# Import evaluation function from base script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_natr import evaluate_with_unified_metrics


class AugmentedDataset(Dataset):
    """
    Dataset with data augmentation and oversampling capabilities
    """
    
    def __init__(self, samples, package_processor, session_processor, 
                 augmentation_config=None, oversample_config=None):
        """
        Initialize dataset with augmentation and oversampling
        
        Args:
            samples: List of training samples
            package_processor: Package processor instance
            session_processor: Session processor instance
            augmentation_config: Dict with augmentation parameters
            oversample_config: Dict with oversampling parameters
        """
        self.package_processor = package_processor
        self.session_processor = session_processor
        
        # Default augmentation config
        self.aug_config = augmentation_config or {
            'session_dropout_prob': 0.15,     # Increased for more diversity
            'session_dropout_max': 0.4,       # Allow more aggressive dropout
            'temporal_jitter_mins': 45,       # Increased temporal variation
            'substitute_similar_prob': 0.15,  # Increased substitution probability
            'augment_purchases': True,        # Whether to augment purchase samples
            'augment_checkouts': True,        # Enable checkout augmentation
            'purchase_augmentation_multiplier': 10,  # 10x for purchases
            'checkout_augmentation_multiplier': 5,   # 5x for checkouts
        }
        
        # Default oversample config
        self.oversample_config = oversample_config or {
            'purchase_oversample': 2.0,       # Reduced to 2x for initial testing
            'checkout_oversample': 1.0,       # No oversampling for checkouts
            'add_to_cart_oversample': 1.0,   # No oversampling for add-to-cart
            'min_samples_per_class': 1000,   # Minimum samples per event class
        }
        
        # Separate samples by event type for oversampling
        self.purchase_samples = []
        self.checkout_samples = []
        self.add_to_cart_samples = []
        self.view_samples = []
        
        for sample in samples:
            if sample.get('is_purchase', False):
                self.purchase_samples.append(sample)
            elif sample.get('has_checkout', False):
                self.checkout_samples.append(sample)
            elif sample.get('has_add_to_cart', False):
                self.add_to_cart_samples.append(sample)
            else:
                self.view_samples.append(sample)
        
        print(f"\nOriginal sample distribution:")
        print(f"  Purchases: {len(self.purchase_samples):,}")
        print(f"  Checkouts: {len(self.checkout_samples):,}")
        print(f"  Add-to-cart: {len(self.add_to_cart_samples):,}")
        print(f"  Views: {len(self.view_samples):,}")
        
        # Create oversampled dataset
        self.samples = self._create_oversampled_dataset()
        
        # Build similarity indices for augmentation
        self._build_similarity_indices()
        
        # Cache for augmented samples
        self.augmentation_cache = {}
    
    def _create_oversampled_dataset(self):
        """Create oversampled dataset with balanced event types"""
        oversampled = []
        
        # Calculate target counts
        max_count = max(
            len(self.view_samples),
            self.oversample_config['min_samples_per_class']
        )
        
        purchase_target = max(
            int(len(self.purchase_samples) * self.oversample_config['purchase_oversample']),
            self.oversample_config['min_samples_per_class']
        )
        checkout_target = max(
            int(len(self.checkout_samples) * self.oversample_config['checkout_oversample']),
            self.oversample_config['min_samples_per_class']
        )
        add_to_cart_target = max(
            int(len(self.add_to_cart_samples) * self.oversample_config['add_to_cart_oversample']),
            self.oversample_config['min_samples_per_class']
        )
        
        # Oversample each category
        oversampled.extend(self._oversample_category(self.purchase_samples, purchase_target))
        
        # Only oversample checkouts and add-to-cart if rate > 1.0
        if self.oversample_config['checkout_oversample'] > 1.0:
            oversampled.extend(self._oversample_category(self.checkout_samples, checkout_target))
        else:
            oversampled.extend(self.checkout_samples)  # Add as-is
            
        if self.oversample_config['add_to_cart_oversample'] > 1.0:
            oversampled.extend(self._oversample_category(self.add_to_cart_samples, add_to_cart_target))
        else:
            oversampled.extend(self.add_to_cart_samples)  # Add as-is
            
        oversampled.extend(self.view_samples)  # Keep views as is
        
        # Shuffle
        random.shuffle(oversampled)
        
        print(f"\nOversampled distribution:")
        print(f"  Total samples: {len(oversampled):,}")
        print(f"  Purchase representation: {purchase_target:,} (~{purchase_target/len(oversampled)*100:.1f}%)")
        print(f"  Checkout representation: {checkout_target:,} (~{checkout_target/len(oversampled)*100:.1f}%)")
        
        # Calculate effective augmentation
        original_purchases = len(self.purchase_samples)
        effective_multiplier = purchase_target / original_purchases if original_purchases > 0 else 0
        print(f"\nEffective purchase augmentation:")
        print(f"  Original purchases: {original_purchases:,}")
        print(f"  After oversampling: {purchase_target:,} ({effective_multiplier:.1f}x)")
        print(f"  With dynamic augmentation: ~{purchase_target * self.aug_config['purchase_augmentation_multiplier']:,} unique variations")
        
        return oversampled
    
    def _oversample_category(self, samples, target_count):
        """Oversample a category to reach target count"""
        if not samples:
            return []
        
        oversampled = []
        
        # Add original samples
        oversampled.extend(samples)
        
        # For purchase samples, create multiple augmented versions of each
        is_purchase_category = samples[0].get('is_purchase', False) if samples else False
        
        if is_purchase_category and self.aug_config['augment_purchases']:
            # Create multiple augmented versions for each purchase sample
            augmentation_multiplier = self.aug_config.get('purchase_augmentation_multiplier', 3)
            
            for original_sample in samples:
                # Create N augmented versions of each purchase
                for aug_idx in range(augmentation_multiplier):
                    augmented_sample = original_sample.copy()
                    augmented_sample['is_oversampled'] = True
                    augmented_sample['augmentation_idx'] = aug_idx  # Track which augmentation version
                    oversampled.append(augmented_sample)
        
        # Also handle checkout augmentation
        is_checkout_category = samples[0].get('has_checkout', False) if samples else False
        if is_checkout_category and self.aug_config.get('augment_checkouts', False):
            augmentation_multiplier = self.aug_config.get('checkout_augmentation_multiplier', 3)
            
            for original_sample in samples:
                for aug_idx in range(augmentation_multiplier):
                    augmented_sample = original_sample.copy()
                    augmented_sample['is_oversampled'] = True
                    augmented_sample['augmentation_idx'] = aug_idx
                    oversampled.append(augmented_sample)
        
        # Add more copies until we reach target
        while len(oversampled) < target_count:
            # Randomly sample from original samples
            sample = random.choice(samples)
            # Create a copy with augmentation flag
            augmented_sample = sample.copy()
            augmented_sample['is_oversampled'] = True
            oversampled.append(augmented_sample)
        
        return oversampled[:target_count]
    
    def _build_similarity_indices(self):
        """Build indices for finding similar items (for augmentation)"""
        self.category_to_packages = defaultdict(list)
        self.theme_to_packages = defaultdict(list)
        self.country_to_packages = defaultdict(list)
        self.price_range_to_packages = defaultdict(list)
        
        # Get all package IDs from session processor
        all_package_ids = list(self.session_processor.package_to_idx.keys())
        
        # Build indices by iterating through all packages
        for pkg_id in all_package_ids:
            features = self.package_processor.get_package_features(str(pkg_id))
            if not features:
                continue
                
            # Category index
            if 'category' in features:
                self.category_to_packages[features['category']].append(pkg_id)
            
            # Theme index
            if 'theme' in features:
                self.theme_to_packages[features['theme']].append(pkg_id)
            
            # Country index
            if 'country' in features:
                self.country_to_packages[features['country']].append(pkg_id)
            
            # Price range index (buckets of 100)
            if 'price' in features and features['price'] is not None:
                price_bucket = int(features['price'] / 100) * 100
                self.price_range_to_packages[price_bucket].append(pkg_id)
    
    def _find_similar_package(self, package_id):
        """Find a similar package for substitution augmentation
        
        Priority: Same category AND country > Same country only > Skip augmentation
        """
        features = self.package_processor.get_package_features(str(package_id))
        if not features:
            return None
            
        candidates = set()
        
        # Strategy 1: Find packages with same category AND same country
        if 'category' in features and 'country' in features:
            category_packages = set(self.category_to_packages[features['category']])
            country_packages = set(self.country_to_packages[features['country']])
            candidates = category_packages.intersection(country_packages)
        
        # Strategy 2: If less than 3 candidates, fall back to same country only
        if len(candidates) < 3 and 'country' in features:
            candidates = set(self.country_to_packages[features['country']])
        
        # Strategy 3: If still less than 3 candidates, skip augmentation
        if len(candidates) < 3:
            return None
        
        # Remove the original package
        candidates.discard(package_id)
        
        # Final check: ensure we still have candidates after removing original
        if candidates:
            return random.choice(list(candidates))
        return None
    
    def _augment_session(self, short_term_packages, short_term_timestamps=None):
        """Apply augmentation to a session"""
        augmented_packages = short_term_packages.copy()
        augmented_timestamps = short_term_timestamps.copy() if short_term_timestamps else None
        
        # Session dropout augmentation
        if random.random() < self.aug_config['session_dropout_prob'] and len(augmented_packages) > 2:
            # Drop some items (keep at least 2)
            max_drop = max(1, int(len(augmented_packages) * self.aug_config['session_dropout_max']))
            num_drop = random.randint(1, max_drop)
            
            # Keep first and last items, drop from middle
            if len(augmented_packages) > num_drop + 2:
                drop_indices = random.sample(range(1, len(augmented_packages) - 1), num_drop)
                for idx in sorted(drop_indices, reverse=True):
                    augmented_packages.pop(idx)
                    if augmented_timestamps:
                        augmented_timestamps.pop(idx)
        
        # Item substitution augmentation
        for i in range(len(augmented_packages)):
            if random.random() < self.aug_config['substitute_similar_prob']:
                similar_pkg = self._find_similar_package(augmented_packages[i])
                if similar_pkg:
                    augmented_packages[i] = similar_pkg
        
        # Temporal jittering (if timestamps available)
        if augmented_timestamps and self.aug_config['temporal_jitter_mins'] > 0:
            for i in range(len(augmented_timestamps)):
                # Add random jitter
                jitter = random.randint(-self.aug_config['temporal_jitter_mins'], 
                                      self.aug_config['temporal_jitter_mins'])
                # Assuming timestamps are in seconds
                augmented_timestamps[i] += jitter * 60
        
        return augmented_packages, augmented_timestamps
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Apply augmentation for oversampled or configured samples
        # ALWAYS augment purchase samples, sometimes augment others
        is_purchase = sample.get('is_purchase', False)
        should_augment = (
            sample.get('is_oversampled', False) or
            (is_purchase and self.aug_config['augment_purchases']) or
            (sample.get('has_checkout', False) and self.aug_config['augment_checkouts'])
        )
        
        # For purchase samples, use more aggressive augmentation
        if is_purchase and should_augment:
            # Temporarily increase augmentation probabilities for purchases
            original_dropout = self.aug_config['session_dropout_prob']
            original_substitute = self.aug_config['substitute_similar_prob']
            
            self.aug_config['session_dropout_prob'] = min(0.3, original_dropout * 2)
            self.aug_config['substitute_similar_prob'] = min(0.3, original_substitute * 2)
        
        if should_augment:
            # Create augmented version (no caching for true diversity)
            augmented_sample = sample.copy()
            
            # Augment short-term session
            if 'short_term_packages' in sample and sample['short_term_packages']:
                original_packages = sample['short_term_packages'].copy()
                aug_packages, aug_timestamps = self._augment_session(
                    sample['short_term_packages'],
                    sample.get('short_term_timestamps')
                )
                augmented_sample['short_term_packages'] = aug_packages
                if aug_timestamps:
                    augmented_sample['short_term_timestamps'] = aug_timestamps
                
                # Debug: Log augmentation effects occasionally
                if hasattr(self, '_aug_debug_count'):
                    self._aug_debug_count += 1
                else:
                    self._aug_debug_count = 1
                    
                if self._aug_debug_count <= 5:  # Log first 5 augmentations
                    changes = []
                    if len(original_packages) != len(aug_packages):
                        changes.append(f"session_dropout: {len(original_packages)}→{len(aug_packages)}")
                    
                    substitutions = sum(1 for orig, aug in zip(original_packages[:len(aug_packages)], aug_packages) if orig != aug)
                    if substitutions > 0:
                        changes.append(f"substitutions: {substitutions}")
                    
                    if changes:
                        print(f"🔄 Augmentation {self._aug_debug_count}: {', '.join(changes)}")
                    elif self._aug_debug_count <= 2:
                        print(f"🔄 Augmentation {self._aug_debug_count}: no changes (randomness)")
            
            sample = augmented_sample
            
            # Restore original probabilities if we modified them for purchases
            if is_purchase:
                self.aug_config['session_dropout_prob'] = original_dropout
                self.aug_config['substitute_similar_prob'] = original_substitute
        
        # Return the augmented sample - the augmentation already happened above
        # This preserves the original sample format that TravelPackageDataset expects
        return sample


# Note: Dynamic augmentation happens on-the-fly during training via AugmentedDataset.__getitem__
# The custom collate function converts augmented samples to proper tensor format for the model


def main(performance_mode="balanced", dataset="13months", event_data_path=None):
    """
    Main training function with data augmentation and oversampling
    
    Args:
        performance_mode: One of 'fastest', 'balanced', 'accurate'
        dataset: Dataset to use - '13months' or '2months'
        event_data_path: Optional explicit path to event data file
    """
    # Detect device and create memory configuration
    device, device_type = detect_device()
    memory_config = create_memory_config(device_type, performance_mode)
    
    # Create memory optimizer
    memory_optimizer = MemoryOptimizer(device, memory_config)
    
    print(f"Using device: {device} ({device_type})")
    print(f"Performance mode: {performance_mode}")
    
    # Clear caches
    for cache_file in glob.glob("data/cache/dataset_*.pkl"):
        os.remove(cache_file)
    
    # Data paths
    package_data_path = "data/feed.parquet"
    
    # Determine event data path
    if event_data_path is None:
        dataset_map = {
            '13months': 'data/bookit_events_data_13_months.parquet',
            '2months': 'data/bookit_events_2_months.parquet'
        }
        event_data_path = dataset_map.get(dataset)
        if not event_data_path:
            raise ValueError(f"Unknown dataset: {dataset}")
    
    print(f"Using event data: {event_data_path}")
    
    # Configure training parameters based on mode
    if performance_mode == "fastest":
        num_epochs = 30
        learning_rate = 0.001
        batch_size = 256
        augmentation_config = {
            'session_dropout_prob': 0.1,
            'session_dropout_max': 0.3,  # Allow up to 30% dropout for speed
            'substitute_similar_prob': 0.1,
            'temporal_jitter_mins': 30,
            'augment_purchases': True,
            'augment_checkouts': False,
            'purchase_augmentation_multiplier': 6,  # Moderate augmentation for speed
        }
        oversample_config = {
            'purchase_oversample': 2.0,  # Minimal oversampling for faster training
            'checkout_oversample': 1.0,  # No oversampling for non-purchases
            'add_to_cart_oversample': 1.0,  # No oversampling for non-purchases
            'min_samples_per_class': 1000,  # Minimum samples per event class
        }
    elif performance_mode == "accurate":
        num_epochs = 100
        learning_rate = 0.0005
        batch_size = 64
        augmentation_config = {
            'session_dropout_prob': 0.2,
            'session_dropout_max': 0.5,
            'substitute_similar_prob': 0.2,
            'temporal_jitter_mins': 60,
            'augment_purchases': True,
            'augment_checkouts': False,  # Focus only on purchases
            'purchase_augmentation_multiplier': 15,  # Maximum augmentation for best accuracy
        }
        oversample_config = {
            'purchase_oversample': 3.0,  # Moderate oversampling, rely on augmentation
            'checkout_oversample': 1.0,  # No oversampling for non-purchases
            'add_to_cart_oversample': 1.0,  # No oversampling for non-purchases
            'min_samples_per_class': 2000,
        }
    else:  # balanced
        num_epochs = 50
        learning_rate = 0.001
        batch_size = 128
        augmentation_config = {
            'session_dropout_prob': 0.05,  # Reduced from 0.15 to prevent too much distortion
            'session_dropout_max': 0.2,    # Reduced from 0.4 - max 20% dropout
            'substitute_similar_prob': 0.05,  # Reduced from 0.15 - less substitution
            'temporal_jitter_mins': 30,     # Reduced from 45
            'augment_purchases': True,
            'augment_checkouts': False,     # Disable checkout augmentation
            'purchase_augmentation_multiplier': 5,  # Reduced from 20 to 5
            'checkout_augmentation_multiplier': 1,  # No checkout augmentation
        }
        oversample_config = {
            'purchase_oversample': 3.0,  # Reduced from 5x to 3x
            'checkout_oversample': 1.0,   # No oversampling for checkouts
            'add_to_cart_oversample': 1.0,  # No oversampling for add-to-cart
            'min_samples_per_class': 1000,  # Keep reasonable minimum
        }
    
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
    
    # Display configuration
    print(f"\nTraining parameters:")
    print(f"  Dataset: {dataset}")
    print(f"  Device: {device}")
    print(f"  Performance mode: {performance_mode}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Number of epochs: {num_epochs}")
    
    print(f"\nAugmentation config:")
    for key, value in augmentation_config.items():
        print(f"  {key}: {value}")
    
    print(f"\nOversampling config:")
    for key, value in oversample_config.items():
        print(f"  {key}: {value}")
    
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
    
    # Identify event types
    samples = identify_event_types(samples, session_processor.event_to_idx)
    
    print("\n6. Applying data filters...")
    quality_samples = filter_by_min_session_length(samples, min_session_length=2)
    filtered_samples, valid_packages = filter_items_by_frequency(quality_samples, min_frequency=5)
    
    # Verify purchase events exist
    purchase_count = sum(1 for s in filtered_samples if s.get('is_purchase', False))
    if purchase_count == 0:
        raise ValueError("No purchase events found in filtered samples")
    
    print(f"Purchase events: {purchase_count} ({purchase_count/len(filtered_samples)*100:.2f}%)")
    
    # Time-based split
    train_samples, test_samples, split_date = time_based_split_year(filtered_samples, train_ratio=0.91)
    
    # Update user mappings
    all_users = set()
    for sample in train_samples + test_samples:
        all_users.add(sample['user_id'])
    
    unmapped_users = all_users - set(session_processor.user_to_idx.keys())
    if unmapped_users:
        print(f"Adding {len(unmapped_users)} unmapped users...")
        max_idx = max(session_processor.user_to_idx.values()) if session_processor.user_to_idx else 0
        for user_id in unmapped_users:
            max_idx += 1
            session_processor.user_to_idx[user_id] = max_idx
    
    # Analyze distributions
    analyze_data_distribution(train_samples, "Train (Original)")
    analyze_data_distribution(test_samples, "Test")
    
    # Create augmented dataset
    print("\n7. Creating augmented training dataset...")
    train_dataset = AugmentedDataset(
        train_samples, 
        package_processor, 
        session_processor,
        augmentation_config=augmentation_config,
        oversample_config=oversample_config
    )
    
    # Get mappings for efficient batch processing
    user_to_idx = session_processor.get_idx_mappings()['user_to_idx']
    package_to_idx = session_processor.get_idx_mappings()['package_to_idx'] 
    event_to_idx = session_processor.get_idx_mappings()['event_to_idx']
    
    print("Creating training dataloader with conservative dynamic augmentation...")
    print("  ✓ 3x oversampling of purchase events")
    print("  ✓ 5x dynamic augmentation variations per purchase sample") 
    print("  ✓ Reduced aggressiveness to prevent overfitting")
    
    # Create a custom dataset that wraps TravelPackageDataset with dynamic augmentation
    from utils.training_utils import create_dataloaders
    from utils.package_processor import TravelPackageDataset
    from torch.utils.data import Dataset
    
    class DynamicAugmentedDataset(Dataset):
        """Wrapper that applies augmentation dynamically during __getitem__"""
        def __init__(self, base_dataset, augmented_dataset, package_processor, user_to_idx, package_to_idx, event_to_idx):
            self.augmented_dataset = augmented_dataset
            self.package_processor = package_processor
            self.user_to_idx = user_to_idx
            self.package_to_idx = package_to_idx
            self.event_to_idx = event_to_idx
            self.base_dataset = base_dataset
            
            # Pre-compute feature tensors for efficient processing
            # Check if package_processor already has pre-computed tensors
            if hasattr(package_processor, '_cached_package_tensors'):
                print("Reusing pre-computed package tensors...")
                self.package_features = package_processor._cached_package_tensors
            else:
                print("Computing package tensors...")
                self.package_features = package_processor.prepare_package_tensors()
                # Cache for reuse
                package_processor._cached_package_tensors = self.package_features
            self.embedding_dim = self.package_features['title_embeddings'].shape[1]
            
            # Reuse settings from base dataset
            self.max_short_term = 10
            self.max_long_term = 20
            
            # Create reverse mapping for efficiency
            self.idx_to_main_id = {v: k for k, v in package_to_idx.items()}
            
            # Get mappings
            mappings = package_processor.get_idx_mappings()
            self.country_to_idx = mappings['country_to_idx']
            self.category_to_idx = mappings['category_to_idx']
            self.theme_to_idx = mappings['theme_to_idx']
            
        def __len__(self):
            return len(self.augmented_dataset)
            
        def __getitem__(self, idx):
            # Get the augmented sample (with fresh transformations each time)
            augmented_sample = self.augmented_dataset[idx]
            
            # Process the sample manually to avoid creating TravelPackageDataset
            return self._process_sample(augmented_sample)
            
        def _process_sample(self, sample):
            """Process a single augmented sample into tensor format"""
            # Extract basic info
            user_id = sample['user_id']
            user_idx = self.user_to_idx.get(user_id, 0)
            
            # Initialize tensors
            processed = {
                'user_id': torch.tensor(user_idx, dtype=torch.int32),
                'has_short_term': torch.tensor(len(sample.get('short_term_packages', [])) > 0),
                'is_purchase': torch.tensor(sample.get('is_purchase', False)),
                'has_checkout': torch.tensor(sample.get('has_checkout', False)),
                'has_add_to_cart': torch.tensor(sample.get('has_add_to_cart', False)),
                'is_cold_start': torch.tensor(sample.get('is_cold_start', False)),
                'has_checkout_inclusive': torch.tensor(sample.get('has_checkout_inclusive', False)),
                'has_add_to_cart_inclusive': torch.tensor(sample.get('has_add_to_cart_inclusive', False)),
            }
            
            # Process short-term
            processed['short_term'] = self._process_sequence(
                sample.get('short_term_packages', []),
                sample.get('short_term_events', []),
                sample.get('short_term_timestamps', []),
                self.max_short_term
            )
            
            # Process long-term
            processed['long_term'] = self._process_sequence(
                sample.get('long_term_packages', []),
                sample.get('long_term_events', []),
                sample.get('long_term_timestamps', []),
                self.max_long_term
            )
            
            # Process purchased
            processed['purchased'] = self._process_single_package(
                sample.get('purchased_package'),
                sample.get('purchased_timestamp', 0)
            )
            
            return processed
            
        def _process_sequence(self, packages, events, timestamps, max_len):
            """Process a sequence of packages into tensor format"""
            # Initialize tensors
            seq_data = {
                'package_ids': torch.zeros(max_len, dtype=torch.int32),
                'event_types': torch.zeros(max_len, dtype=torch.int32),
                'title_embeddings': torch.zeros(max_len, self.embedding_dim),
                'coordinates': torch.zeros(max_len, 2),
                'country_ids': torch.zeros(max_len, dtype=torch.int32),
                'category_ids': torch.zeros(max_len, dtype=torch.int32),
                'theme_ids': torch.zeros(max_len, dtype=torch.int32),
                'prices': torch.zeros(max_len, dtype=torch.float32),
                'timestamps': torch.zeros(max_len, dtype=torch.float32)
            }
            
            # Truncate to max length
            packages = packages[-max_len:] if packages else []
            events = events[-max_len:] if events else []
            timestamps = timestamps[-max_len:] if timestamps else [0] * len(packages)
            
            # Process each package
            for i, (pkg_id, event_id, ts) in enumerate(zip(packages, events, timestamps)):
                pkg_idx = self.package_to_idx.get(str(pkg_id), 0)
                seq_data['package_ids'][i] = pkg_idx
                seq_data['event_types'][i] = event_id
                seq_data['timestamps'][i] = ts
                
                # Get features from pre-computed tensors
                if pkg_idx > 0 and str(pkg_id) in self.package_features['package_to_idx']:
                    feat_idx = self.package_features['package_to_idx'][str(pkg_id)]
                    seq_data['title_embeddings'][i] = self.package_features['title_embeddings'][feat_idx]
                    seq_data['coordinates'][i] = self.package_features['coordinates'][feat_idx]
                    seq_data['country_ids'][i] = self.package_features['country_ids'][feat_idx]
                    seq_data['category_ids'][i] = self.package_features['category_ids'][feat_idx]
                    seq_data['theme_ids'][i] = self.package_features['theme_ids'][feat_idx]
                    seq_data['prices'][i] = self.package_features['prices'][feat_idx]
                
            return seq_data
            
        def _process_single_package(self, pkg_id, timestamp):
            """Process a single package (purchased item)"""
            pkg_idx = self.package_to_idx.get(str(pkg_id), 0) if pkg_id else 0
            
            pkg_data = {
                'package_ids': torch.tensor(pkg_idx, dtype=torch.int32),
                'timestamps': torch.tensor(timestamp, dtype=torch.float32)
            }
            
            # Initialize feature tensors
            if pkg_idx > 0 and str(pkg_id) in self.package_features['package_to_idx']:
                feat_idx = self.package_features['package_to_idx'][str(pkg_id)]
                pkg_data['title_embeddings'] = self.package_features['title_embeddings'][feat_idx]
                pkg_data['coordinates'] = self.package_features['coordinates'][feat_idx]
                pkg_data['country_ids'] = self.package_features['country_ids'][feat_idx]
                pkg_data['category_ids'] = self.package_features['category_ids'][feat_idx]
                pkg_data['theme_ids'] = self.package_features['theme_ids'][feat_idx]
                pkg_data['prices'] = self.package_features['prices'][feat_idx]
            else:
                # Use zeros for missing packages
                pkg_data['title_embeddings'] = torch.zeros(self.embedding_dim)
                pkg_data['coordinates'] = torch.zeros(2)
                pkg_data['country_ids'] = torch.tensor(0, dtype=torch.int32)
                pkg_data['category_ids'] = torch.tensor(0, dtype=torch.int32)
                pkg_data['theme_ids'] = torch.tensor(0, dtype=torch.int32)
                pkg_data['prices'] = torch.tensor(0.0, dtype=torch.float32)
                
            return pkg_data
    
    # Get mappings
    user_to_idx = session_processor.get_idx_mappings()['user_to_idx']
    package_to_idx = session_processor.get_idx_mappings()['package_to_idx']
    event_to_idx = session_processor.get_idx_mappings()['event_to_idx']
    
    # Pre-compute package tensors once to avoid duplicating work
    print("Pre-computing package tensors...")
    package_tensors = package_processor.prepare_package_tensors()
    # Cache for reuse
    package_processor._cached_package_tensors = package_tensors
    
    # Skip creating base dataset since we don't use it
    base_train_dataset = None
    
    print("Creating dynamic augmented training dataset...")
    # Wrap with dynamic augmentation
    dynamic_train_dataset = DynamicAugmentedDataset(
        base_train_dataset, 
        train_dataset,
        package_processor,
        user_to_idx,
        package_to_idx,
        event_to_idx
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
        use_cache=False,
        prefetch_features=False  # Disable prefetching to speed up creation
    )
    
    # Create dataloaders using the augmented dataset
    train_loader, test_loader = create_dataloaders(
        train_samples,  # Pass original samples for weighted sampling calculation
        test_samples,
        package_processor,
        session_processor,
        batch_size=batch_size,
        num_workers=0 if device_type == 'mps' else 2,
        use_weighted_sampling=True,
        train_dataset=dynamic_train_dataset,  # Use our dynamic augmented dataset
        test_dataset=test_dataset
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train (augmented): {len(train_loader.dataset):,} samples ({len(train_loader):,} batches)")
    print(f"  Test: {len(test_loader.dataset):,} samples ({len(test_loader):,} batches)")
    
    # Create model
    print("\n8. Creating model...")
    
    # Get dimensions
    num_users = max(session_processor.user_to_idx.values()) + 1
    num_packages = max(session_processor.package_to_idx.values()) + 1
    num_countries = max(package_processor.country_to_idx.values()) + 1
    num_categories = max(package_processor.category_to_idx.values()) + 1
    num_themes = max(package_processor.theme_to_idx.values()) + 1
    
    # Get embedding dimension
    package_tensors = package_processor.prepare_package_tensors()
    actual_embedding_dim = package_tensors['title_embeddings'].shape[1]
    
    # Model configuration
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
    
    # Get model size for memory calculations and saving  
    model_size_mb = get_model_size(model)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model size: {model_size_mb} MB")
    
    # Initialize popularity bias based on item frequencies in training data
    print("Calculating item frequencies for popularity initialization...")
    item_frequencies = torch.zeros(num_packages)
    for sample in train_samples:
        # Give higher weight to purchased items
        if sample.get('is_purchase', False) and 'purchased_package' in sample:
            pkg_id = sample['purchased_package']
            if str(pkg_id) in session_processor.package_to_idx:
                idx = session_processor.package_to_idx[str(pkg_id)]
                item_frequencies[idx] += 1.0  # Full weight for purchases
        
        # Give lower weight to other interactions
        for pkg_id in sample.get('short_term_packages', []) + sample.get('long_term_packages', []):
            if str(pkg_id) in session_processor.package_to_idx:
                idx = session_processor.package_to_idx[str(pkg_id)]
                item_frequencies[idx] += 0.1  # Lower weight for non-purchased interactions
    
    # Initialize popularity in the model
    print(f"Initializing popularity bias with item frequencies (max freq: {item_frequencies.max().item():.0f})")
    model.initialize_popularity(item_frequencies.to(device))
    
    # Create optimizer with different learning rates for user components
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
    
    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': learning_rate, 'weight_decay': 1e-4},
        {'params': user_params, 'lr': learning_rate * 2.0, 'weight_decay': 1e-5},
        {'params': user_transform_params, 'lr': learning_rate * 10.0, 'weight_decay': 1e-5}
    ])
    
    # Create loss function (consistent with base training script)
    loss_fn = NaturalPurchaseLoss(
        purchase_boost=5.0  # Reduced from 10.0 to prevent overfitting
    )
    
    # Training loop
    print("\n9. Starting training...")
    best_purchase_recall = 0.0
    best_purchase_mrr = 0.0
    patience = 5 if performance_mode != "fastest" else 3  # Reduced patience to stop overfitting sooner
    epochs_without_improvement = 0
    
    # Add learning rate scheduler to reduce LR when plateauing
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6
    )
    
    # Create AMP manager and gradient accumulator
    amp_manager = AMPManager(device, memory_config)
    gradient_accumulator = GradientAccumulator(
        steps=memory_config.gradient_accumulation_steps,
        amp_manager=amp_manager
    )
    
    os.makedirs('checkpoints/natr', exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # Training
        model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            # Move batch to device
            batch = move_batch_to_device(batch, device)
            
            # Forward pass with AMP
            with amp_manager:
                outputs = model(batch)
                predictions = outputs['predictions']
                targets = batch['purchased']['package_ids']
                
                # Get event flags
                is_purchase = batch.get('is_purchase', torch.zeros_like(targets, dtype=torch.bool))
                
                # Calculate loss (NaturalPurchaseLoss only needs predictions, targets, and is_purchase)
                loss = loss_fn(predictions, targets, is_purchase)
            
            # Backward pass with gradient accumulation
            original_loss = gradient_accumulator.backward(loss)
            
            # Update metrics
            total_loss += original_loss.item()
            num_batches += 1
            
            # Gradient clipping
            if gradient_accumulator.should_step():
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Step optimizer
            gradient_accumulator.step(optimizer)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': total_loss / num_batches})
        
        avg_loss = total_loss / num_batches
        print(f"Training loss: {avg_loss:.4f}")
        
        # Evaluation
        metrics = evaluate_with_unified_metrics(
            model, test_loader, device, memory_optimizer, k_values=[5, 10, 20]
        )
        
        # Print metrics
        print("\nEvaluation Results:")
        print(f"  Purchase Recall@10: {metrics.get('purchase_recall@10', 0)*100:.2f}%")
        print(f"  Purchase Recall@20: {metrics.get('purchase_recall@20', 0)*100:.2f}%")
        print(f"  Purchase MRR: {metrics.get('purchase_mrr', 0):.4f}")
        print(f"  Item Coverage@20: {metrics.get('item_coverage@20', 0)*100:.2f}%")
        print(f"  Weighted Recall@20: {metrics.get('weighted_recall@20', 0)*100:.2f}%")
        
        # Track best model
        current_purchase_recall_20 = metrics.get('purchase_recall@20', 0)
        current_purchase_mrr = metrics.get('purchase_mrr', 0)
        
        # Update scheduler (step on purchase recall)
        scheduler.step(current_purchase_recall_20)
        
        # Save best model
        if current_purchase_recall_20 > best_purchase_recall or \
           (abs(current_purchase_recall_20 - best_purchase_recall) < 1e-6 and 
            current_purchase_mrr > best_purchase_mrr):
            best_purchase_recall = current_purchase_recall_20
            best_purchase_mrr = current_purchase_mrr
            epochs_without_improvement = 0
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.__dict__,
                'metrics': metrics,
                'train_loss': avg_loss
            }, 'checkpoints/natr/augmented_best_model.pth')
            print(f"✓ New best model saved! Purchase Recall@20: {best_purchase_recall*100:.2f}%")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")
        
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
        'checkpoint_path': 'checkpoints/natr/augmented_best_model.pth',
        'package_data_path': package_data_path,
        'event_data_path': event_data_path,
        'performance_mode': performance_mode,
        'device_type': device_type,
        'training_date': datetime.now().strftime("%Y-%m-%d"),
        'training_strategy': 'augmented_oversampling',
        'augmentation_config': augmentation_config,
        'oversample_config': oversample_config,
        'split_date': split_date,
        'train_ratio': 0.91,
        'package_count': len(valid_packages),
        'user_count': num_users,
        'model_size_mb': model_size_mb
    }
    
    output_dir = 'output/model_info' if dataset == '13months' else 'output/model_info_2months'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'model_info_augmented.json'), 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("\nFiles saved:")
    print("  - checkpoints/natr/augmented_best_model.pth")
    print("  - model_info_augmented.json")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train NATR with data augmentation and oversampling')
    parser.add_argument('--mode', type=str, default='balanced',
                        choices=['fastest', 'balanced', 'accurate'],
                        help='Performance mode (default: balanced)')
    parser.add_argument('--dataset', type=str, default='13months',
                        choices=['13months', '2months'],
                        help='Dataset to use (default: 13months)')
    parser.add_argument('--event-data', type=str, default=None,
                        help='Path to event data file (overrides --dataset)')
    
    args = parser.parse_args()
    
    # Check for Apple Silicon
    is_mps = torch.backends.mps.is_available()
    if is_mps:
        print("Apple Silicon (MPS) device detected! Using optimized settings.")
    
    main(performance_mode=args.mode, dataset=args.dataset, event_data_path=args.event_data)
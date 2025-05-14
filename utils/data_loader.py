import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Tuple
import sys
import os

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from your data directory
from utils.package_processor import PackageProcessor, TravelPackageDataset
from utils.session_processor import SessionProcessor


def prepare_natr_dataloaders(
    samples: List[Dict[str, Any]],
    package_processor: PackageProcessor,
    session_processor: SessionProcessor,
    batch_size: int = 32,
    test_size: float = 0.2,
    random_seed: int = 42,
    num_workers: int = 4,
    max_short_term: int = 10,
    max_long_term: int = 20,
    balance_purchases: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare train and test dataloaders specifically for NATR model
    
    Args:
        samples: List of training samples from session processor
        package_processor: PackageProcessor instance with all features
        session_processor: SessionProcessor instance
        batch_size: Batch size for training
        test_size: Proportion of data for testing
        random_seed: Random seed for reproducibility
        num_workers: Number of workers for data loading
        max_short_term: Maximum short-term sequence length (paper suggests 10)
        max_long_term: Maximum long-term sequence length (paper suggests 20)
        balance_purchases: Whether to balance purchase/non-purchase samples
    
    Returns:
        train_loader, test_loader: DataLoaders for training and testing
    """
    # Set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Get mappings
    user_to_idx = session_processor.get_idx_mappings()['user_to_idx']
    package_to_idx = session_processor.get_idx_mappings()['package_to_idx']
    event_to_idx = session_processor.get_idx_mappings()['event_to_idx']
    
    # Ensure package processor has mappings created
    package_processor.create_mappings()
    
    # Prepare package features
    print("Preparing package features for NATR model...")
    package_processor.prepare_package_tensors()
    
    # Fix samples to ensure package IDs are strings
    fixed_samples = []
    for sample in samples:
        fixed_sample = sample.copy()
        fixed_sample['purchased_package'] = str(sample['purchased_package'])
        fixed_sample['short_term_packages'] = [str(pkg) for pkg in sample['short_term_packages']]
        if 'long_term_packages' in sample:
            fixed_sample['long_term_packages'] = [str(pkg) for pkg in sample['long_term_packages']]
        fixed_samples.append(fixed_sample)
    
    # Balance samples if requested
    if balance_purchases and 'is_purchase' in fixed_samples[0]:
        purchase_samples = [s for s in fixed_samples if s['is_purchase']]
        non_purchase_samples = [s for s in fixed_samples if not s['is_purchase']]
        
        # Balance the dataset
        target_size = min(len(purchase_samples), len(non_purchase_samples))
        
        if len(purchase_samples) > target_size:
            np.random.shuffle(purchase_samples)
            purchase_samples = purchase_samples[:target_size]
        
        if len(non_purchase_samples) > target_size:
            np.random.shuffle(non_purchase_samples)
            non_purchase_samples = non_purchase_samples[:target_size]
        
        # Combine and shuffle
        balanced_samples = purchase_samples + non_purchase_samples
        np.random.shuffle(balanced_samples)
        
        print(f"Balanced NATR dataset: {len(balanced_samples)} samples")
        print(f"  - Purchase: {len(purchase_samples)}")
        print(f"  - Non-purchase: {len(non_purchase_samples)}")
        
        fixed_samples = balanced_samples
    
    # Split into train and test
    num_samples = len(fixed_samples)
    num_test = int(num_samples * test_size)
    
    indices = np.random.permutation(num_samples)
    test_indices = indices[:num_test]
    train_indices = indices[num_test:]
    
    train_samples = [fixed_samples[i] for i in train_indices]
    test_samples = [fixed_samples[i] for i in test_indices]
    
    print(f"Split into {len(train_samples)} train and {len(test_samples)} test samples")
    
    # Create NATR datasets
    train_dataset = TravelPackageDataset(
        train_samples, 
        package_processor,
        user_to_idx,
        package_to_idx,
        event_to_idx,
        max_short_term=max_short_term,
        max_long_term=max_long_term,
        empty_token=-1,
        unknown_token=-1
    )
    
    test_dataset = TravelPackageDataset(
        test_samples, 
        package_processor,
        user_to_idx,
        package_to_idx,
        event_to_idx,
        max_short_term=max_short_term,
        max_long_term=max_long_term,
        empty_token=-1,
        unknown_token=-1
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True  # For stable training
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, test_loader


def prepare_data_for_natr(
    package_data_path: str, 
    event_data_path: str, 
    api_key: Optional[str] = None,
    cache_dir: str = 'data/cache',
    batch_size: int = 32,
    test_size: float = 0.2,
    min_interactions: int = 5,
    max_short_term: int = 10,
    max_long_term: int = 20,
    balance_purchases: bool = True
) -> Tuple[DataLoader, DataLoader, PackageProcessor, SessionProcessor]:
    """
    Complete data preparation pipeline for NATR model
    
    Args:
        package_data_path: Path to package data file (CSV/Parquet)
        event_data_path: Path to event data file (CSV/Parquet)
        api_key: OpenAI API key for embeddings (optional)
        cache_dir: Directory for caching
        batch_size: Batch size for training
        test_size: Proportion of data for testing
        min_interactions: Minimum interactions for user filtering
        max_short_term: Maximum short-term sequence length
        max_long_term: Maximum long-term sequence length
        balance_purchases: Whether to balance purchase/non-purchase samples
    
    Returns:
        Tuple of (train_loader, test_loader, package_processor, session_processor)
    """
    print("=== Starting NATR Data Preparation ===")
    
    # Initialize processors
    package_processor = PackageProcessor(
        data_path=package_data_path,
        cache_dir=cache_dir,
        load_coordinates=True,
        load_embeddings=True,
        api_key=api_key
    )
    
    session_processor = SessionProcessor(
        data_path=event_data_path,
        cache_dir=cache_dir,
        min_interactions=min_interactions,
        max_sessions_per_user=20,
        max_samples_per_user=10
    )
    
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
    
    print("\n6. Creating NATR dataloaders...")
    train_loader, test_loader = prepare_natr_dataloaders(
        samples=samples,
        package_processor=package_processor,
        session_processor=session_processor,
        batch_size=batch_size,
        test_size=test_size,
        max_short_term=max_short_term,
        max_long_term=max_long_term,
        balance_purchases=balance_purchases
    )
    
    print("\n=== NATR Data Preparation Complete ===")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Print sample batch structure
    sample_batch = next(iter(train_loader))
    print("\nSample batch structure:")
    print(f"  User IDs: {sample_batch['user_id'].shape}")
    print(f"  Short-term packages: {sample_batch['short_term']['package_ids'].shape}")
    print(f"  Short-term embeddings: {sample_batch['short_term']['title_embeddings'].shape}")
    print(f"  Short-term coordinates: {sample_batch['short_term']['coordinates'].shape}")
    print(f"  Long-term packages: {sample_batch['long_term']['package_ids'].shape}")
    print(f"  Purchased package: {sample_batch['purchased']['package_id'].shape}")
    
    return train_loader, test_loader, package_processor, session_processor


def validate_data_quality(dataloader: DataLoader, name: str = "DataLoader"):
    """
    Validate the quality of the prepared data
    
    Args:
        dataloader: DataLoader to validate
        name: Name for logging
    """
    print(f"\n=== Validating {name} ===")
    
    total_batches = len(dataloader)
    total_samples = 0
    empty_sequences = 0
    missing_embeddings = 0
    missing_coordinates = 0
    event_type_counts = {}
    
    for batch_idx, batch in enumerate(dataloader):
        batch_size = batch['user_id'].shape[0]
        total_samples += batch_size
        
        # Check for empty sequences
        empty_short = (batch['short_term']['package_ids'] == -1).sum(dim=1).eq(batch['short_term']['package_ids'].shape[1]).sum()
        empty_long = (batch['long_term']['package_ids'] == -1).sum(dim=1).eq(batch['long_term']['package_ids'].shape[1]).sum()
        empty_sequences += empty_short.item() + empty_long.item()
        
        # Check for missing embeddings
        zero_embeddings_short = (batch['short_term']['title_embeddings'].sum(dim=2) == 0).sum()
        zero_embeddings_long = (batch['long_term']['title_embeddings'].sum(dim=2) == 0).sum()
        missing_embeddings += zero_embeddings_short.item() + zero_embeddings_long.item()
        
        # Check for missing coordinates
        zero_coords_short = ((batch['short_term']['coordinates'] == 0).all(dim=2)).sum()
        zero_coords_long = ((batch['long_term']['coordinates'] == 0).all(dim=2)).sum()
        missing_coordinates += zero_coords_short.item() + zero_coords_long.item()
        
        # Count event types
        for event_type in batch['short_term']['event_types'].flatten():
            event_id = event_type.item()
            event_type_counts[event_id] = event_type_counts.get(event_id, 0) + 1
        
        if batch_idx == 0:  # Print first batch details
            print(f"\nFirst batch details:")
            print(f"  Batch size: {batch_size}")
            print(f"  User IDs range: {batch['user_id'].min()} - {batch['user_id'].max()}")
            print(f"  Has purchase info: {'is_purchase' in batch}")
            if 'is_purchase' in batch:
                purchases = batch['is_purchase'].sum().item()
                print(f"  Purchases in batch: {purchases}/{batch_size}")
    
    print(f"\nData quality summary:")
    print(f"  Total samples: {total_samples}")
    print(f"  Total batches: {total_batches}")
    print(f"  Empty sequences: {empty_sequences}")
    print(f"  Missing embeddings: {missing_embeddings}")
    print(f"  Missing coordinates: {missing_coordinates}")
    
    print(f"\nEvent type distribution:")
    event_names = {0: 'None', 1: 'ViewContent', 2: 'AddToCart', 3: 'InitiateCheckout', 4: 'Purchase'}
    for event_id, count in sorted(event_type_counts.items()):
        event_name = event_names.get(event_id, f'Unknown_{event_id}')
        print(f"  {event_name}: {count}")
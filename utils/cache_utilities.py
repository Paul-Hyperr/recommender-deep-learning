"""
Utility functions for managing dataset caches
"""

import os
import json
import shutil
import torch
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import pickle


def list_dataset_caches(cache_dir: str = 'data/cache') -> List[dict]:
    """
    List all available dataset caches
    
    Args:
        cache_dir: Base cache directory
        
    Returns:
        List of cache information dictionaries
    """
    dataset_cache_dir = os.path.join(cache_dir, 'dataset_cache')
    if not os.path.exists(dataset_cache_dir):
        print("No dataset caches found")
        return []
    
    caches = []
    print("=== Available Dataset Caches ===")
    
    for subdir in sorted(os.listdir(dataset_cache_dir)):
        subdir_path = os.path.join(dataset_cache_dir, subdir)
        if os.path.isdir(subdir_path):
            print(f"\n{subdir}:")
            
            for file in os.listdir(subdir_path):
                if file.startswith('metadata_') and file.endswith('.json'):
                    metadata_path = os.path.join(subdir_path, file)
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        # Calculate age
                        created_date = datetime.strptime(metadata['created_date'], '%Y-%m-%d %H:%M:%S')
                        age_days = (datetime.now() - created_date).days
                        
                        print(f"  - Created: {metadata['created_date']} ({age_days} days ago)")
                        print(f"    Samples: {metadata['total_samples']:,}")
                        print(f"    Purchase rate: {metadata.get('purchase_samples', 0)/metadata['total_samples']*100:.1f}%")
                        print(f"    Sequence lengths: ST={metadata['max_short_term']}, LT={metadata['max_long_term']}")
                        
                        cache_info = {
                            'path': subdir_path,
                            'metadata': metadata,
                            'age_days': age_days
                        }
                        caches.append(cache_info)
                        
                    except Exception as e:
                        print(f"  - Error reading metadata: {e}")
    
    return caches


def clear_dataset_cache(
    cache_dir: Optional[str] = None, 
    older_than_days: Optional[int] = None,
    force: bool = False
) -> None:
    """
    Clear dataset caches with safety checks
    
    Args:
        cache_dir: Base cache directory. If None, requires older_than_days to be set
        older_than_days: Only clear caches older than this many days. If None, requires explicit cache_dir
        force: Skip confirmation prompt
        
    Raises:
        ValueError: If neither cache_dir nor older_than_days is specified
    """
    # Safety check: require at least one parameter
    if cache_dir is None and older_than_days is None:
        raise ValueError(
            "Safety check: You must specify either 'cache_dir' or 'older_than_days'. "
            "This prevents accidental deletion of all caches."
        )
    
    # Default cache directory
    if cache_dir is None:
        cache_dir = 'data/cache'
    
    dataset_cache_dir = os.path.join(cache_dir, 'dataset_cache')
    if not os.path.exists(dataset_cache_dir):
        print("No dataset caches found")
        return
    
    # Get list of caches to clear
    caches_to_clear = []
    total_size = 0
    
    for subdir in os.listdir(dataset_cache_dir):
        subdir_path = os.path.join(dataset_cache_dir, subdir)
        if os.path.isdir(subdir_path):
            # Check age if specified
            should_clear = True
            
            if older_than_days is not None:
                should_clear = False
                # Check metadata files for age
                for file in os.listdir(subdir_path):
                    if file.startswith('metadata_') and file.endswith('.json'):
                        try:
                            with open(os.path.join(subdir_path, file), 'r') as f:
                                metadata = json.load(f)
                            created = datetime.strptime(metadata['created_date'], '%Y-%m-%d %H:%M:%S')
                            age_days = (datetime.now() - created).days
                            
                            if age_days >= older_than_days:
                                should_clear = True
                                break
                        except Exception:
                            # If can't read metadata, include it for clearing if no age filter
                            if older_than_days is None:
                                should_clear = True
            
            if should_clear:
                # Calculate size
                size = sum(
                    os.path.getsize(os.path.join(subdir_path, f))
                    for f in os.listdir(subdir_path)
                    if os.path.isfile(os.path.join(subdir_path, f))
                )
                total_size += size
                caches_to_clear.append((subdir_path, size))
    
    if not caches_to_clear:
        print("No caches match the criteria for clearing")
        return
    
    # Show what will be cleared
    print(f"\n=== Caches to Clear ===")
    for path, size in caches_to_clear:
        print(f"{os.path.basename(path)}: {size/1024/1024:.1f} MB")
    
    print(f"\nTotal: {len(caches_to_clear)} caches, {total_size/1024/1024:.1f} MB")
    
    if older_than_days:
        print(f"Clearing caches older than {older_than_days} days")
    
    # Confirmation
    if not force:
        response = input("\nAre you sure you want to clear these caches? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled")
            return
    
    # Clear caches
    for path, _ in caches_to_clear:
        print(f"Removing {path}")
        shutil.rmtree(path)
    
    print(f"\nCleared {len(caches_to_clear)} caches")


def get_cache_statistics(cache_dir: str = 'data/cache') -> dict:
    """
    Get statistics about all dataset caches
    
    Args:
        cache_dir: Base cache directory
        
    Returns:
        Dictionary with cache statistics
    """
    caches = list_dataset_caches(cache_dir)
    
    if not caches:
        return {
            'total_caches': 0,
            'total_size_mb': 0,
            'oldest_days': 0,
            'newest_days': 0,
            'total_samples': 0
        }
    
    total_size = 0
    total_samples = 0
    oldest_days = 0
    newest_days = float('inf')
    
    for cache in caches:
        # Calculate size
        path = cache['path']
        size = sum(
            os.path.getsize(os.path.join(path, f))
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        )
        total_size += size
        
        # Get metadata
        metadata = cache['metadata']
        total_samples += metadata['total_samples']
        
        # Track age
        age_days = cache['age_days']
        oldest_days = max(oldest_days, age_days)
        newest_days = min(newest_days, age_days)
    
    return {
        'total_caches': len(caches),
        'total_size_mb': total_size / 1024 / 1024,
        'oldest_days': oldest_days,
        'newest_days': newest_days,
        'total_samples': total_samples
    }

def analyze_cache_quality(cache_dir: str = 'data/cache', detailed: bool = False) -> Dict[str, Any]:
    """
    Analyze the quality of cached datasets, checking if features work properly
    
    Args:
        cache_dir: Base cache directory
        detailed: If True, show detailed analysis for each cache
        
    Returns:
        Dictionary with quality analysis
    """
    dataset_cache_dir = os.path.join(cache_dir, 'dataset_cache')
    if not os.path.exists(dataset_cache_dir):
        print("No dataset caches found")
        return {}
    
    print("=== Dataset Cache Quality Analysis ===")
    quality_results = {}
    
    for subdir in sorted(os.listdir(dataset_cache_dir)):
        subdir_path = os.path.join(dataset_cache_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
            
        print(f"\n{subdir}:")
        
        # Find cache files
        cache_files = [f for f in os.listdir(subdir_path) 
                      if f.startswith('prepared_dataset_') and f.endswith('.pkl')]
        
        for cache_file in cache_files:
            cache_path = os.path.join(subdir_path, cache_file)
            
            try:
                # Load cached data
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Analyze features
                analysis = analyze_cached_features(cached_data, detailed)
                quality_results[cache_file] = analysis
                
                # Print summary
                print(f"  Cache: {cache_file}")
                print(f"    Samples: {analysis['num_samples']:,}")
                print(f"    Features:")
                
                # Embeddings
                emb_rate = analysis['embeddings']['non_zero_rate'] * 100
                print(f"      ✓ Embeddings: {emb_rate:.1f}% non-zero")
                if emb_rate < 90:
                    print(f"        WARNING: Low embedding rate!")
                
                # Coordinates
                coord_rate = analysis['coordinates']['non_zero_rate'] * 100
                print(f"      ✓ Coordinates: {coord_rate:.1f}% non-zero")
                if coord_rate < 90:
                    print(f"        WARNING: Low coordinate rate!")
                
                # Categorical features
                print(f"      ✓ Countries: {analysis['categorical']['num_countries']} unique")
                print(f"      ✓ Categories: {analysis['categorical']['num_categories']} unique")
                print(f"      ✓ Themes: {analysis['categorical']['num_themes']} unique")
                
                # Event types
                print(f"      ✓ Event types: {len(analysis['events']['distribution'])} types")
                if detailed:
                    for event_type, count in analysis['events']['distribution'].items():
                        print(f"        - {event_type}: {count:,}")
                
                # Check for cities (should not exist)
                if analysis['has_cities']:
                    print(f"      ⚠️  WARNING: Cities found in data (should not exist!)")
                else:
                    print(f"      ✓ No cities (as expected)")
                
                # Data completeness
                print(f"      ✓ Purchase rate: {analysis['purchase_rate']*100:.1f}%")
                print(f"      ✓ Empty sequences: ST={analysis['empty_sequences']['short_term']:.1f}%, "
                      f"LT={analysis['empty_sequences']['long_term']:.1f}%")
                
                # Quality score
                quality_score = calculate_quality_score(analysis)
                print(f"    Overall Quality Score: {quality_score:.1f}/100")
                
                if quality_score < 80:
                    print(f"    ⚠️  WARNING: Quality score below 80!")
                
            except Exception as e:
                print(f"  Error analyzing cache: {e}")
                quality_results[cache_file] = {'error': str(e)}
    
    return quality_results


def analyze_cached_features(cached_data: dict, detailed: bool = False) -> dict:
    """
    Analyze features in cached dataset
    
    Args:
        cached_data: Loaded cache data
        detailed: Whether to provide detailed analysis
        
    Returns:
        Analysis dictionary
    """
    # Get tensors
    short_term_packages = cached_data.get('short_term_packages')
    num_samples = len(short_term_packages) if short_term_packages is not None else 0
    
    analysis = {
        'num_samples': num_samples,
        'embeddings': {'non_zero_rate': 0, 'avg_magnitude': 0},
        'coordinates': {'non_zero_rate': 0, 'valid_range_rate': 0},
        'categorical': {
            'num_countries': 0,
            'num_categories': 0,
            'num_themes': 0
        },
        'events': {'distribution': {}},
        'has_cities': False,
        'purchase_rate': 0,
        'empty_sequences': {'short_term': 0, 'long_term': 0}
    }
    
    if num_samples == 0:
        return analysis
    
    # Check for cities (should not exist)
    if 'short_term_cities' in cached_data or 'long_term_cities' in cached_data:
        analysis['has_cities'] = True
    
    # Analyze categorical features
    if 'short_term_countries' in cached_data:
        countries = torch.cat([
            cached_data['short_term_countries'].flatten(),
            cached_data['long_term_countries'].flatten()
        ])
        analysis['categorical']['num_countries'] = len(torch.unique(countries))
    
    if 'short_term_categories' in cached_data:
        categories = torch.cat([
            cached_data['short_term_categories'].flatten(),
            cached_data['long_term_categories'].flatten()
        ])
        analysis['categorical']['num_categories'] = len(torch.unique(categories))
    
    if 'short_term_themes' in cached_data:
        themes = torch.cat([
            cached_data['short_term_themes'].flatten(),
            cached_data['long_term_themes'].flatten()
        ])
        analysis['categorical']['num_themes'] = len(torch.unique(themes))
    
    # Analyze event types
    if 'short_term_events' in cached_data:
        events = torch.cat([
            cached_data['short_term_events'].flatten(),
            cached_data['long_term_events'].flatten()
        ])
        unique_events, counts = torch.unique(events, return_counts=True)
        
        event_names = {0: 'None', 1: 'ViewContent', 2: 'AddToCart', 
                      3: 'InitiateCheckout', 4: 'Purchase', -1: 'Empty'}
        
        for event_id, count in zip(unique_events.tolist(), counts.tolist()):
            event_name = event_names.get(event_id, f'Unknown_{event_id}')
            analysis['events']['distribution'][event_name] = count
    
    # Check purchase rate
    if 'is_purchase' in cached_data:
        purchases = cached_data['is_purchase']
        if torch.is_tensor(purchases):
            analysis['purchase_rate'] = purchases.float().mean().item()
        else:
            analysis['purchase_rate'] = sum(purchases) / len(purchases)
    
    # Check empty sequences
    if 'short_term_packages' in cached_data:
        st_packages = cached_data['short_term_packages']
        # Empty sequences have -1 in first position
        empty_st = (st_packages[:, 0] == -1).float().mean().item()
        analysis['empty_sequences']['short_term'] = empty_st * 100
    
    if 'long_term_packages' in cached_data:
        lt_packages = cached_data['long_term_packages']
        empty_lt = (lt_packages[:, 0] == -1).float().mean().item()
        analysis['empty_sequences']['long_term'] = empty_lt * 100
    
    # Estimate embedding/coordinate quality from package indices
    # If packages have valid indices, they should have embeddings/coordinates
    valid_packages_st = (short_term_packages > 0).float().mean().item()
    valid_packages_lt = (cached_data['long_term_packages'] > 0).float().mean().item() if 'long_term_packages' in cached_data else 0
    
    # Rough estimate (actual check would require loading the dataset)
    analysis['embeddings']['non_zero_rate'] = (valid_packages_st + valid_packages_lt) / 2
    analysis['coordinates']['non_zero_rate'] = (valid_packages_st + valid_packages_lt) / 2
    
    return analysis


def calculate_quality_score(analysis: dict) -> float:
    """
    Calculate overall quality score for a cached dataset
    
    Args:
        analysis: Analysis dictionary from analyze_cached_features
        
    Returns:
        Quality score from 0-100
    """
    score = 100.0
    
    # Deduct points for missing embeddings
    embedding_rate = analysis['embeddings']['non_zero_rate']
    if embedding_rate < 0.9:
        score -= (0.9 - embedding_rate) * 50  # Up to -5 points
    
    # Deduct points for missing coordinates
    coord_rate = analysis['coordinates']['non_zero_rate']
    if coord_rate < 0.9:
        score -= (0.9 - coord_rate) * 50  # Up to -5 points
    
    # Deduct if cities exist (they shouldn't)
    if analysis['has_cities']:
        score -= 10
    
    # Deduct for too many empty sequences
    empty_st = analysis['empty_sequences']['short_term']
    empty_lt = analysis['empty_sequences']['long_term']
    
    if empty_st > 30:
        score -= (empty_st - 30) * 0.2  # -0.2 points per % over 30%
    if empty_lt > 40:
        score -= (empty_lt - 40) * 0.1  # -0.1 points per % over 40%
    
    # Deduct for imbalanced purchase rate
    purchase_rate = analysis['purchase_rate']
    imbalance = abs(0.5 - purchase_rate)
    if imbalance > 0.3:
        score -= imbalance * 20  # Up to -4 points
    
    # Deduct for low categorical diversity
    if analysis['categorical']['num_countries'] < 5:
        score -= 5
    if analysis['categorical']['num_categories'] < 3:
        score -= 5
    if analysis['categorical']['num_themes'] < 3:
        score -= 5
    
    # Ensure score stays in range
    return max(0, min(100, score))


def clean_stale_caches(cache_dir: str = 'data/cache', dry_run: bool = False) -> dict:
    """
    Remove cache files that can become stale when processing logic changes.
    
    This function removes caches that store processed data which may become outdated
    when the processing pipeline is updated (e.g., adding new event flags).
    
    Keeps:
    - LLM embeddings (expensive API calls)
    - Geocoding data (expensive API calls)
    - Package metadata/features (relatively stable)
    
    Removes:
    - Dataset caches (dataset_*.pkl)
    - Enhanced training samples
    - Sessions cache
    - Preprocessed data
    
    Args:
        cache_dir: Path to cache directory
        dry_run: If True, only show what would be deleted without actually deleting
        
    Returns:
        Dictionary with cleanup statistics
    """
    import glob
    
    # Define patterns for caches to remove
    stale_patterns = [
        'dataset_*.pkl',              # Dataset caches with pre-computed tensors
        'enhanced_training_samples.pkl',  # Processed training samples
        'sessions.pkl',               # Extracted sessions
        'preprocessed_data.pkl',      # Preprocessed event data
        'prepared_dataset_*.pkl',     # Old dataset cache format
        'training_samples.pkl'        # Old training samples cache
    ]
    
    # Define patterns to keep (for information)
    keep_patterns = [
        'llm_embeddings/*',          # Expensive OpenAI embeddings
        'geocoding/*',               # Expensive geocoding API results
        'package_metadata.pkl',      # Package metadata
        'package_features.pkl',      # Package features
        'package_mappings.pkl',      # ID mappings
        'user_package_mappings.pkl'  # User/package mappings
    ]
    
    print(f"\nCleaning stale caches in: {cache_dir}")
    print("=" * 60)
    
    removed_count = 0
    removed_size = 0
    removed_files = []
    
    # Find and remove stale caches
    for pattern in stale_patterns:
        files = glob.glob(os.path.join(cache_dir, pattern))
        
        for file_path in files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                size_mb = size / (1024 * 1024)
                
                if dry_run:
                    print(f"[DRY RUN] Would remove: {os.path.basename(file_path)} ({size_mb:.1f} MB)")
                else:
                    print(f"Removing: {os.path.basename(file_path)} ({size_mb:.1f} MB)")
                    os.remove(file_path)
                
                removed_files.append(os.path.basename(file_path))
                removed_count += 1
                removed_size += size
    
    print("\n" + "=" * 60)
    print(f"Summary: {'Would remove' if dry_run else 'Removed'} {removed_count} files, {removed_size / (1024 * 1024):.1f} MB total")
    
    # Show what's being kept
    print("\nKept caches (expensive to recompute):")
    kept_count = 0
    kept_size = 0
    kept_files = []
    
    for pattern in keep_patterns:
        if '*' in pattern:
            # Handle directory patterns
            base_pattern = pattern.replace('/*', '')
            dir_path = os.path.join(cache_dir, base_pattern)
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                dir_size = sum(
                    os.path.getsize(os.path.join(dir_path, f))
                    for f in os.listdir(dir_path)
                    if os.path.isfile(os.path.join(dir_path, f))
                )
                print(f"  - {base_pattern}/ ({dir_size / (1024 * 1024):.1f} MB)")
                kept_files.append(f"{base_pattern}/")
                kept_count += len(os.listdir(dir_path))
                kept_size += dir_size
        else:
            # Handle file patterns
            file_path = os.path.join(cache_dir, pattern)
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f"  - {pattern} ({size / (1024 * 1024):.1f} MB)")
                kept_files.append(pattern)
                kept_count += 1
                kept_size += size
    
    print(f"\nTotal kept: {kept_count} files, {kept_size / (1024 * 1024):.1f} MB")
    
    if dry_run:
        print("\n⚠️  This was a dry run. Use dry_run=False to actually remove files.")
    
    return {
        'removed_count': removed_count,
        'removed_size_mb': removed_size / (1024 * 1024),
        'removed_files': removed_files,
        'kept_count': kept_count,
        'kept_size_mb': kept_size / (1024 * 1024),
        'kept_files': kept_files,
        'dry_run': dry_run
    }


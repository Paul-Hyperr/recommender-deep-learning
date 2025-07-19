"""
Recent popularity calculation for boosting model performance on new dataset.
Uses only the last month of training data to avoid stale popularity signals.
"""

import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple


def calculate_recent_popularity_scores(
    events_df: pd.DataFrame,
    num_packages: int,
    last_month_only: bool = True,
    temporal_decay: bool = True,
    event_weights: Optional[Dict[str, float]] = None
) -> torch.Tensor:
    """
    Calculate recent popularity scores using only the last month of training data.
    
    Args:
        events_df: DataFrame with events (must have 'main_id', 'event', 'timestamp' columns)
        num_packages: Total number of packages for tensor size
        last_month_only: If True, only use events from last 30 days of data
        temporal_decay: If True, apply exponential decay to older events
        event_weights: Weights for different event types (Purchase > Checkout > Cart > View)
        
    Returns:
        Tensor of shape [num_packages] with popularity scores
    """
    
    if event_weights is None:
        # Use the same intent weights as training
        event_weights = {
            'Purchase': 4,          # Strong purchase signal (updated)
            'InitiateCheckout': 0.12, # Reduced to prevent checkout dominance  
            'AddToCart': 0.05,      # Moderate cart signal
            'ViewContent': 0.001    # Very low view signal (single view equivalent)
        }
    
    print(f"📊 Calculating recent popularity scores...")
    print(f"  Total events in dataset: {len(events_df):,}")
    print(f"  Date range: {events_df['timestamp'].min()} to {events_df['timestamp'].max()}")
    
    # Convert timestamps to datetime if they're not already
    if not pd.api.types.is_datetime64_any_dtype(events_df['timestamp']):
        events_df = events_df.copy()
        events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
    
    # Filter to last month if requested
    if last_month_only:
        max_date = events_df['timestamp'].max()
        min_date = max_date - timedelta(days=30)
        recent_events = events_df[events_df['timestamp'] >= min_date]
        print(f"  Using last 30 days: {min_date} to {max_date}")
        print(f"  Recent events: {len(recent_events):,} ({len(recent_events)/len(events_df)*100:.1f}%)")
    else:
        recent_events = events_df
    
    # Apply temporal decay if requested
    if temporal_decay and len(recent_events) > 0:
        max_timestamp = recent_events['timestamp'].max()
        recent_events = recent_events.copy()
        
        # Calculate days from most recent event
        recent_events['days_ago'] = (max_timestamp - recent_events['timestamp']).dt.total_seconds() / (24 * 3600)
        
        # Apply exponential decay with 7-day half-life
        recent_events['temporal_weight'] = np.exp(-recent_events['days_ago'] / 7)
        print(f"  Applied temporal decay (7-day half-life)")
    else:
        recent_events = recent_events.copy()
        recent_events['temporal_weight'] = 1.0
    
    # Calculate weighted popularity for each package
    popularity_scores = {}
    
    for event_type, weight in event_weights.items():
        event_data = recent_events[recent_events['event'] == event_type]
        if len(event_data) > 0:
            # Group by package and sum temporal weights
            package_scores = event_data.groupby('main_id')['temporal_weight'].sum()
            
            # Apply event type weight
            package_scores = package_scores * weight
            
            # Add to total popularity
            for package_id, score in package_scores.items():
                if package_id in popularity_scores:
                    popularity_scores[package_id] += score
                else:
                    popularity_scores[package_id] = score
            
            print(f"    {event_type}: {len(event_data):,} events, {len(package_scores)} packages")
    
    print(f"  Total packages with popularity: {len(popularity_scores):,}")
    
    # Convert to tensor
    scores_tensor = torch.zeros(num_packages)
    
    for package_id, score in popularity_scores.items():
        try:
            # Convert package_id to int if it's a string
            if isinstance(package_id, str):
                package_idx = int(package_id)
            else:
                package_idx = int(package_id)
            
            # Ensure it's within bounds
            if 0 <= package_idx < num_packages:
                scores_tensor[package_idx] = score
        except (ValueError, TypeError):
            # Skip invalid package IDs
            continue
    
    # Normalize scores
    if scores_tensor.sum() > 0:
        # Apply log scaling to compress extreme values
        scores_tensor = torch.log(scores_tensor + 1.0)
        
        # Normalize to range (0.05, 0.95) to avoid exact 0/1 values
        max_score = scores_tensor.max()
        if max_score > 0:
            # Min-max normalization to (0, 1) first
            min_score = scores_tensor.min()
            scores_tensor = (scores_tensor - min_score) / (max_score - min_score)
            
            # Scale to (0.05, 0.95) to avoid exact boundaries
            scores_tensor = scores_tensor * 0.9 + 0.05
        
        print(f"  Score statistics: min={scores_tensor.min():.4f}, max={scores_tensor.max():.4f}, mean={scores_tensor.mean():.4f}")
        print(f"  Non-zero scores: {(scores_tensor > 0).sum()}/{len(scores_tensor)} ({(scores_tensor > 0).sum()/len(scores_tensor)*100:.1f}%)")
    else:
        print("⚠️ All popularity scores are zero!")
    
    return scores_tensor


def get_recent_popularity_from_samples(
    samples: list,
    num_packages: int,
    days: int = 20,
    event_weights: Optional[Dict[str, float]] = None
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Calculate recent popularity from training samples using the most recent N days.
    This is useful when you have already processed samples and don't want to reload raw events.
    
    Args:
        samples: List of training samples with 'timestamp' and event flags
        num_packages: Total number of packages
        days: Number of recent days to consider (default: 20)
        event_weights: Weights for different event types
        
    Returns:
        Tuple of (popularity_tensor, stats_dict)
    """
    
    if event_weights is None:
        # Intent-based weights that match training
        event_weights = {
            'purchase': 4.0,
            'checkout': 0.12,
            'cart': 0.05,
            'view': 0.001
        }
    
    print(f"\n🔥 Calculating recent popularity boost...")
    
    # Find date range in samples
    timestamps = [s['timestamp'] for s in samples if 'timestamp' in s]
    if not timestamps:
        print("⚠️ No timestamps found in samples!")
        return torch.zeros(num_packages), {}
    
    max_timestamp = max(timestamps)
    min_timestamp = max_timestamp - timedelta(days=days)
    
    # Filter recent samples
    recent_samples = [s for s in samples if s.get('timestamp', max_timestamp) >= min_timestamp]
    
    print(f"Using {len(recent_samples):,} samples from most recent {days} days out of {len(samples):,} total")
    print(f"Date range: {min_timestamp} to {max_timestamp}")
    
    # Calculate popularity scores
    popularity_scores = {}
    event_counts = {'purchase': 0, 'checkout': 0, 'cart': 0, 'view': 0}
    
    for sample in recent_samples:
        package_id = sample.get('purchased_package', None)
        if package_id is None:
            continue
            
        # Calculate days ago for temporal decay
        sample_timestamp = sample.get('timestamp', max_timestamp)
        days_ago = (max_timestamp - sample_timestamp).total_seconds() / (24 * 3600)
        temporal_weight = np.exp(-days_ago / 7)  # 7-day half-life
        
        # Determine event type and apply weight
        if sample.get('is_purchase', False):
            weight = event_weights['purchase'] * temporal_weight
            event_counts['purchase'] += 1
        elif sample.get('intent_level') == 'checkout' or sample.get('has_checkout_inclusive', False):
            weight = event_weights['checkout'] * temporal_weight
            event_counts['checkout'] += 1
        elif sample.get('intent_level') == 'cart' or sample.get('has_add_to_cart_inclusive', False):
            weight = event_weights['cart'] * temporal_weight
            event_counts['cart'] += 1
        else:
            weight = event_weights['view'] * temporal_weight
            event_counts['view'] += 1
        
        # Add to popularity
        if package_id in popularity_scores:
            popularity_scores[package_id] += weight
        else:
            popularity_scores[package_id] = weight
    
    # Convert to tensor
    scores_tensor = torch.zeros(num_packages)
    
    for package_id, score in popularity_scores.items():
        try:
            # Convert package_id to int if it's a string
            if isinstance(package_id, str):
                package_idx = int(package_id)
            else:
                package_idx = int(package_id)
            
            # Ensure it's within bounds
            if 0 <= package_idx < num_packages:
                scores_tensor[package_idx] = score
        except (ValueError, TypeError):
            # Skip invalid package IDs
            continue
    
    # Normalize scores
    if scores_tensor.sum() > 0:
        # Apply log scaling
        scores_tensor = torch.log(scores_tensor + 1.0)
        
        # Normalize to range (0.05, 0.95) to avoid exact 0/1 values
        max_score = scores_tensor.max()
        if max_score > 0:
            min_score = scores_tensor.min()
            scores_tensor = (scores_tensor - min_score) / (max_score - min_score)
            scores_tensor = scores_tensor * 0.9 + 0.05
    
    print(f"Recent popularity stats: min={scores_tensor.min():.4f}, max={scores_tensor.max():.4f}")
    print(f"Non-zero packages: {(scores_tensor > 0.05).sum()}/{num_packages}")
    
    stats = {
        'recent_samples': len(recent_samples),
        'total_samples': len(samples),
        'days': days,
        'event_counts': event_counts,
        'unique_packages': len(popularity_scores)
    }
    
    return scores_tensor, stats
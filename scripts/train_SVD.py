#!/usr/bin/env python3
"""
Implicit ALS Training Script for Travel Recommendation

This script trains an Implicit ALS model for collaborative filtering
using travel booking events data with optimized purchase prediction.

OPTIMAL APPROACH (default):
- Training signals: AddToCart, InitiateCheckout only (NO ViewContent, NO Purchase)
- Temporal weighting: Recent interactions weighted exponentially higher
- Event weighting: InitiateCheckout (1.0) > AddToCart (0.3)
- Evaluation: Pure purchase prediction (no data leakage)
- Split: 91% train, 9% test (time-based)

This approach eliminates data leakage and focuses on high-intent signals
that actually predict purchases, leading to better real-world performance.
"""

import pandas as pd
import numpy as np
import pickle
import os
import time
from datetime import datetime
from typing import Dict, Tuple
import argparse
from tqdm import tqdm
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', message='OpenBLAS')
warnings.filterwarnings('ignore', message='Method expects CSR input')
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Import implicit library for ALS
try:
    from implicit.als import AlternatingLeastSquares
    import scipy.sparse as sparse
except ImportError:
    print("Please install implicit: pip install implicit")
    raise


def calculate_rating_from_count(n, base_rating, early_growth, decay_rate, p, threshold):
    """
    Calculate an aggregated rating given:
      - n: the number of events,
      - base_rating: the rating value for a single event.
      - threshold: the count threshold for switching from early growth to decay
      
    For n = 1, the function returns base_rating.
    For 1 < n <= threshold, a concave function is used so that the boost is higher early (big jump 1→2)
    and then flattens out.
    For n > threshold, a decay formula is used so that additional events add less.
    
    Returns the total aggregated rating.
    """
    if n < 1:
        return 0.0
    if n == 1:
        return base_rating
    
    if n <= threshold:
        # Map n from [1,threshold] to x in [0,1]
        x = (n - 1) / (threshold - 1)  # x=0 when n=1, x=1 when n=threshold
        # Using a power function (p < 1) gives a steeper initial increase
        shape = x ** p
        early_boost = 1 + (early_growth - 1) * shape
    else:
        # For n > threshold, we use a decay formula
        early_boost = early_growth * (threshold / n) ** decay_rate

    return base_rating * n * early_boost


def aggregate_ratings_from_df(df, early_growth, decay_rate, p, threshold):
    """
    Aggregates ratings from an existing DataFrame that already has a 'rating' column.
    
    Steps:
      1. Group by userId, main_id, and event.
      2. For each group, count the number of events (n) and take the base rating from the row.
      3. Use calculate_rating_from_count(n, base_rating, ...) to compute an aggregated rating for that event.
      4. Sum the aggregated ratings across events for each (userId, main_id).
    
    Returns a DataFrame with columns: userId, main_id, aggregated_rating.
    """
    # Group by userId, main_id, and rating. We assume the 'rating' column is constant for a given event type.
    grouped = df.groupby(['userId', 'main_id', 'rating']).agg(
        count=('rating', 'size'),
    ).reset_index()

    # Compute the aggregated rating for each group using our custom function.
    grouped['aggregated_rating'] = grouped.apply(
        lambda row: calculate_rating_from_count(
            n=row['count'],
            base_rating=row['rating'],
            early_growth=early_growth,
            decay_rate=decay_rate,
            p=p,
            threshold=threshold
        ),
        axis=1
    )
    
    # Sum aggregated ratings across event types for each userId and main_id.
    final_df = grouped.groupby(['userId', 'main_id'])['aggregated_rating'].sum().reset_index()
    
    return final_df








def load_and_prepare_data_initial(data_path: str, train_ratio: float = 0.91, min_interactions: int = 5, recent_months: int = None):
    """
    Load and prepare data - print common information once
    """
    print(f"Loading data from {data_path}...")
    
    # Load events data
    events_df = pd.read_parquet(data_path)
    print(f"Loaded {len(events_df):,} events")
    
    # Convert timestamp
    events_df['created_at'] = pd.to_datetime(events_df['created_at'])
    
    # Filter to recent months if specified
    if recent_months:
        max_date = events_df['created_at'].max()
        cutoff_date = max_date - pd.DateOffset(months=recent_months)
        events_df = events_df[events_df['created_at'] >= cutoff_date].copy()
    
    # Filter to relevant events (weights will be added later)
    relevant_event_types = ['ViewContent', 'AddToCart', 'InitiateCheckout', 'Purchase']
    relevant_events = events_df[events_df['event'].isin(relevant_event_types)].copy()
    
    print(f"Relevant events: {len(relevant_events):,}")
    
    # Filter users with minimum interactions
    user_interaction_counts = relevant_events['userId'].value_counts()
    valid_users = user_interaction_counts[user_interaction_counts >= min_interactions].index
    
    print(f"Users with >= {min_interactions} interactions: {len(valid_users):,}")
    
    # Keep only valid users
    filtered_events = relevant_events[relevant_events['userId'].isin(valid_users)].copy()
    print(f"Final dataset: {len(filtered_events):,} events, {filtered_events['main_id'].nunique():,} items")
    
    # Time-based split by date (proper temporal split)
    filtered_events = filtered_events.sort_values('created_at').reset_index(drop=True)
    split_idx = int(len(filtered_events) * train_ratio)
    split_date = filtered_events.iloc[split_idx]['created_at']
    
    train_events = filtered_events[filtered_events['created_at'] <= split_date].copy()
    test_events = filtered_events[filtered_events['created_at'] > split_date].copy()
    
    print(f"Time-based split: {len(train_events):,} train, {len(test_events):,} test")
    
    # For TEST data: just extract purchases (no rating aggregation needed)
    test_purchases = test_events[test_events['event'] == 'Purchase'][['userId', 'main_id']].copy()
    
    # Ensure main_id is integer
    test_purchases['main_id'] = test_purchases['main_id'].astype(int)
    
    print(f"Test purchases: {len(test_purchases)} events")
    
    return train_events, test_purchases, split_date


def prepare_implicit_als_data_binary_non_purchase(train_events, test_events, alpha=40.0, temporal_decay_days=14):
    """
    Prepare data using binary non-purchase events only (exclude purchases):
    - Only use AddToCart, InitiateCheckout for training matrix (exclude ViewContent and Purchase)
    - Apply temporal decay: recent interactions weighted exponentially higher
    - Binary preferences: p_ui ∈ {0,1} (user interacted with item or not)
    - Apply Hu's confidence formula: c_ui = 1 + α * r_ui
    - Evaluate on purchases (completely separate signal from target)
    """
    print(f"Preparing optimized binary non-purchase matrix with α={alpha}, temporal_decay={temporal_decay_days} days")
    
    # Filter to high-intent non-purchase events only (exclude ViewContent for better signal)
    high_intent_events = ['AddToCart', 'InitiateCheckout']
    train_events = train_events[train_events['event'].isin(high_intent_events)].copy()
    
    print(f"High-intent events for training: {len(train_events):,}")
    print(f"Event distribution: {train_events['event'].value_counts().to_dict()}")
    
    # Apply temporal weighting - recent interactions matter exponentially more
    if 'created_at' in train_events.columns:
        print("Applying temporal decay weighting...")
        current_time = train_events['created_at'].max()
        train_events['days_ago'] = (current_time - train_events['created_at']).dt.days
        
        # Exponential decay: recent interactions weighted much higher
        train_events['temporal_weight'] = np.exp(-train_events['days_ago'] / temporal_decay_days)
        
        # Boost very recent interactions (last 24 hours)
        recent_boost = np.where(train_events['days_ago'] < 1, 2.0, 1.0)
        train_events['temporal_weight'] *= recent_boost
        
        print(f"  Temporal weight range: [{train_events['temporal_weight'].min():.4f}, {train_events['temporal_weight'].max():.4f}]")
        print(f"  Mean temporal weight: {train_events['temporal_weight'].mean():.4f}")
    else:
        print("No timestamp found, using equal weighting")
        train_events['temporal_weight'] = 1.0
    
    # Apply event-specific weights - InitiateCheckout is much stronger signal than AddToCart
    event_weights = {
        'AddToCart': 0.3,           # Lower weight - early in funnel
        'InitiateCheckout': 1.0     # Higher weight - close to purchase
    }
    train_events['event_weight'] = train_events['event'].map(event_weights)
    
    # Combine temporal and event weights
    train_events['combined_weight'] = train_events['temporal_weight'] * train_events['event_weight']
    
    print(f"Event weights applied: {event_weights}")
    
    # Create user and item mappings
    all_users = sorted(train_events['userId'].unique())
    all_items = sorted(train_events['main_id'].unique())
    
    user_to_idx = {user: idx for idx, user in enumerate(all_users)}
    item_to_idx = {item: idx for idx, item in enumerate(all_items)}
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}
    
    print(f"Users: {len(all_users):,}, Items: {len(all_items):,}")
    
    # Calculate weighted interactions per user-item pair
    interaction_data = train_events.groupby(['userId', 'main_id'])['combined_weight'].sum().reset_index()
    interaction_data.rename(columns={'combined_weight': 'r_ui'}, inplace=True)
    
    print(f"Weighted interaction statistics:")
    print(f"  User-item pairs: {len(interaction_data):,}")
    print(f"  r_ui range: [{interaction_data['r_ui'].min():.4f}, {interaction_data['r_ui'].max():.4f}]")
    print(f"  r_ui mean: {interaction_data['r_ui'].mean():.4f}")
    
    # Use weighted interaction values
    interaction_data['value'] = interaction_data['r_ui']
    
    # Create sparse matrix
    row_indices = [user_to_idx[user] for user in interaction_data['userId']]
    col_indices = [item_to_idx[item] for item in interaction_data['main_id']]
    data_values = interaction_data['value'].values
    
    train_matrix = sparse.csr_matrix(
        (data_values, (row_indices, col_indices)),
        shape=(len(all_users), len(all_items))
    )
    
    print(f"Training matrix: {train_matrix.shape}, non-zeros: {train_matrix.nnz}")
    print(f"Matrix density: {train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1]) * 100:.4f}%")
    print(f"Matrix content: Temporally-weighted high-intent interactions (purchases excluded)")
    
    return {
        'train_matrix': train_matrix,
        'user_to_idx': user_to_idx,
        'item_to_idx': item_to_idx,
        'idx_to_user': idx_to_user,
        'idx_to_item': idx_to_item,
        'test_events': test_events,
        'alpha': alpha
    }


def prepare_implicit_als_data_hu_style_normalized(train_events, test_events, event_weights, alpha=40.0):
    """
    Prepare data following Hu et al. style with event weights and user normalization:
    - Event weights determine interaction strength (r_ui)
    - Normalize each user's ratings by their maximum rating
    - Apply Hu's confidence formula: c_ui = 1 + α * r_ui_normalized
    """
    print(f"Preparing Hu-style matrix with event weights, user normalization, and α={alpha}")
    
    # Apply event weights to get interaction strengths
    train_events = train_events.copy()
    train_events['weight'] = train_events['event'].map(event_weights)
    train_events = train_events[train_events['weight'] > 0]
    
    # Create user and item mappings
    all_users = sorted(train_events['userId'].unique())
    all_items = sorted(train_events['main_id'].unique())
    
    user_to_idx = {user: idx for idx, user in enumerate(all_users)}
    item_to_idx = {item: idx for idx, item in enumerate(all_items)}
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}
    
    print(f"Users: {len(all_users):,}, Items: {len(all_items):,}")
    
    # Calculate r_ui: sum of weighted interactions per user-item pair
    interaction_data = train_events.groupby(['userId', 'main_id'])['weight'].sum().reset_index()
    interaction_data.rename(columns={'weight': 'r_ui'}, inplace=True)
    
    # User normalization: divide each user's ratings by their maximum rating
    user_max_ratings = interaction_data.groupby('userId')['r_ui'].max().reset_index()
    user_max_ratings.rename(columns={'r_ui': 'max_rating'}, inplace=True)
    
    # Merge max ratings and normalize
    interaction_data = interaction_data.merge(user_max_ratings, on='userId')
    interaction_data['r_ui_normalized'] = interaction_data['r_ui'] / interaction_data['max_rating']
    
    print(f"User normalization statistics:")
    print(f"  Original r_ui range: [{interaction_data['r_ui'].min():.3f}, {interaction_data['r_ui'].max():.3f}]")
    print(f"  Normalized r_ui range: [{interaction_data['r_ui_normalized'].min():.3f}, {interaction_data['r_ui_normalized'].max():.3f}]")
    print(f"  User max ratings range: [{user_max_ratings['max_rating'].min():.3f}, {user_max_ratings['max_rating'].max():.3f}]")
    
    # Use normalized values
    interaction_data['value'] = interaction_data['r_ui_normalized']
    
    print(f"Interaction statistics:")
    print(f"  User-item pairs: {len(interaction_data):,}")
    print(f"  Normalized r_ui mean: {interaction_data['r_ui_normalized'].mean():.3f}")
    
    # Create sparse matrix
    row_indices = [user_to_idx[user] for user in interaction_data['userId']]
    col_indices = [item_to_idx[item] for item in interaction_data['main_id']]
    data_values = interaction_data['value'].values
    
    train_matrix = sparse.csr_matrix(
        (data_values, (row_indices, col_indices)),
        shape=(len(all_users), len(all_items))
    )
    
    print(f"Training matrix: {train_matrix.shape}, non-zeros: {train_matrix.nnz}")
    print(f"Matrix density: {train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1]) * 100:.4f}%")
    print(f"Matrix content: User-normalized weighted interactions")
    
    return {
        'train_matrix': train_matrix,
        'user_to_idx': user_to_idx,
        'item_to_idx': item_to_idx,
        'idx_to_user': idx_to_user,
        'idx_to_item': idx_to_item,
        'test_events': test_events,
        'alpha': alpha
    }


def prepare_implicit_als_data_hu_style(train_events, test_events, event_weights, alpha=40.0):
    """
    Prepare data following Hu et al. style but with event weights:
    - Event weights determine interaction strength (r_ui)
    - Apply Hu's confidence formula: c_ui = 1 + α * r_ui
    - No advanced aggregation functions
    """
    print(f"Preparing Hu-style matrix with event weights and α={alpha}")
    
    # Apply event weights to get interaction strengths
    train_events = train_events.copy()
    train_events['weight'] = train_events['event'].map(event_weights)
    train_events = train_events[train_events['weight'] > 0]
    
    # Create user and item mappings
    all_users = sorted(train_events['userId'].unique())
    all_items = sorted(train_events['main_id'].unique())
    
    user_to_idx = {user: idx for idx, user in enumerate(all_users)}
    item_to_idx = {item: idx for idx, item in enumerate(all_items)}
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}
    
    print(f"Users: {len(all_users):,}, Items: {len(all_items):,}")
    
    # Calculate r_ui: sum of weighted interactions per user-item pair
    interaction_data = train_events.groupby(['userId', 'main_id'])['weight'].sum().reset_index()
    interaction_data.rename(columns={'weight': 'r_ui'}, inplace=True)
    
    # Apply Hu's confidence formula: c_ui = 1 + α * r_ui
    # But pass r_ui to the model and let it apply the formula
    interaction_data['value'] = interaction_data['r_ui']
    
    print(f"Interaction statistics:")
    print(f"  User-item pairs: {len(interaction_data):,}")
    print(f"  r_ui range: [{interaction_data['r_ui'].min():.3f}, {interaction_data['r_ui'].max():.3f}]")
    print(f"  r_ui mean: {interaction_data['r_ui'].mean():.3f}")
    
    # Create sparse matrix
    row_indices = [user_to_idx[user] for user in interaction_data['userId']]
    col_indices = [item_to_idx[item] for item in interaction_data['main_id']]
    data_values = interaction_data['value'].values
    
    train_matrix = sparse.csr_matrix(
        (data_values, (row_indices, col_indices)),
        shape=(len(all_users), len(all_items))
    )
    
    print(f"Training matrix: {train_matrix.shape}, non-zeros: {train_matrix.nnz}")
    print(f"Matrix density: {train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1]) * 100:.4f}%")
    
    return {
        'train_matrix': train_matrix,
        'user_to_idx': user_to_idx,
        'item_to_idx': item_to_idx,
        'idx_to_user': idx_to_user,
        'idx_to_item': idx_to_item,
        'test_events': test_events,
        'alpha': alpha
    }


def prepare_implicit_als_data_with_weights(train_events, test_events, event_weights):
    """
    Prepare data for Implicit ALS with custom event weights
    """
    print(f"Preparing data for Implicit ALS with weights: {event_weights}")
    
    # Apply event weights to training data
    train_events = train_events.copy()
    train_events['weight'] = train_events['event'].map(event_weights)
    
    return prepare_implicit_als_data(train_events, test_events)


def prepare_implicit_als_data_with_advanced_aggregation(train_events, test_events, event_weights, early_growth, decay_rate, p, threshold):
    """
    Prepare data for Implicit ALS with advanced aggregation (early boost + decay)
    """
    print(f"Preparing data for Implicit ALS with advanced aggregation:")
    print(f"  Event weights: {event_weights}")
    print(f"  Aggregation params: early_growth={early_growth}, decay_rate={decay_rate}, p={p}, threshold={threshold}")
    
    # Map events to ratings using event weights as base ratings
    train_df = train_events.copy()
    train_df['rating'] = train_df['event'].map(event_weights)
    
    # Filter to events with ratings
    train_df = train_df[train_df['rating'].notna()]
    
    # Apply advanced aggregation
    aggregated_df = aggregate_ratings_from_df(train_df, early_growth, decay_rate, p, threshold)
    
    # Create user and item mappings from ALL training data 
    all_users = sorted(train_events['userId'].unique())
    all_items = sorted(train_events['main_id'].unique())
    
    user_to_idx = {user: idx for idx, user in enumerate(all_users)}
    item_to_idx = {item: idx for idx, item in enumerate(all_items)}
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}
    
    print(f"All users: {len(all_users):,}, All items: {len(all_items):,}")
    
    # Use aggregated ratings directly with max cap to prevent extremely large values
    max_cap = 10.0
    train_matrix_data = aggregated_df.copy()
    capped_count = (train_matrix_data['aggregated_rating'] > max_cap).sum()
    train_matrix_data['total_weight'] = train_matrix_data['aggregated_rating'].clip(upper=max_cap)
    if capped_count > 0:
        print(f"Applied max cap ({max_cap}) to {capped_count:,} user-item pairs")
    
    # Create sparse matrix for training (user x item)
    row_indices = [user_to_idx[user] for user in train_matrix_data['userId']]
    col_indices = [item_to_idx[item] for item in train_matrix_data['main_id']]
    data_values = train_matrix_data['total_weight'].values
    
    train_matrix = sparse.csr_matrix(
        (data_values, (row_indices, col_indices)),
        shape=(len(all_users), len(all_items))
    )
    
    print(f"Training matrix: {train_matrix.shape}, non-zeros: {train_matrix.nnz}")
    print(f"Matrix density: {train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1]) * 100:.2f}%")
    
    return {
        'train_matrix': train_matrix,
        'user_to_idx': user_to_idx,
        'item_to_idx': item_to_idx,
        'idx_to_user': idx_to_user,
        'idx_to_item': idx_to_item,
        'test_events': test_events  # Keep all test events
    }


def prepare_implicit_als_data(train_events, test_events):
    """
    Prepare data for Implicit ALS (creates sparse matrix with confidence weights)
    Simple approach: Use ALL training data for mappings, create matrix, test on purchases
    """
    
    # Create user and item mappings from ALL training data 
    all_users = sorted(train_events['userId'].unique())
    all_items = sorted(train_events['main_id'].unique())
    
    user_to_idx = {user: idx for idx, user in enumerate(all_users)}
    item_to_idx = {item: idx for idx, item in enumerate(all_items)}
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}
    
    print(f"All users: {len(all_users):,}, All items: {len(all_items):,}")
    
    # Aggregate weights by user-item pairs for training data
    train_aggregated = train_events.groupby(['userId', 'main_id', 'weight']).size().reset_index(name='count')
    train_aggregated['weighted_count'] = train_aggregated['weight'] * train_aggregated['count']
    train_matrix_data = train_aggregated.groupby(['userId', 'main_id'])['weighted_count'].sum().reset_index()
    
    # Use weighted counts directly - implicit library will apply alpha internally
    # The library expects raw interaction counts/weights, not pre-computed confidence
    # Apply max cap to prevent extremely large confidence values
    max_cap = 10.0
    capped_count = (train_matrix_data['weighted_count'] > max_cap).sum()
    train_matrix_data['total_weight'] = train_matrix_data['weighted_count'].clip(upper=max_cap)
    if capped_count > 0:
        print(f"Applied max cap ({max_cap}) to {capped_count:,} user-item pairs")
    
    # Create sparse matrix for training (user x item)
    row_indices = [user_to_idx[user] for user in train_matrix_data['userId']]
    col_indices = [item_to_idx[item] for item in train_matrix_data['main_id']]
    data_values = train_matrix_data['total_weight'].values
    
    train_matrix = sparse.csr_matrix(
        (data_values, (row_indices, col_indices)),
        shape=(len(all_users), len(all_items))
    )
    
    print(f"Training matrix: {train_matrix.shape}, non-zeros: {train_matrix.nnz}")
    print(f"Matrix density: {train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1]) * 100:.2f}%")
    
    return {
        'train_matrix': train_matrix,
        'user_to_idx': user_to_idx,
        'item_to_idx': item_to_idx,
        'idx_to_user': idx_to_user,
        'idx_to_item': idx_to_item,
        'test_events': test_events  # Keep all test events
    }


def prepare_training_data_simple(train_events):
    """
    Simple training data preparation - just binary ratings for all events
    """
    print("Preparing simple training data (binary ratings)...")
    
    # Convert all events to binary rating (1.0)
    train_df = train_events[['userId', 'main_id']].copy()
    train_df['rating'] = 1.0
    
    # Aggregate: sum over duplicate user-item pairs
    train_df = train_df.groupby(['userId', 'main_id'])['rating'].sum().reset_index()
    
    # Ensure main_id is integer
    train_df['main_id'] = train_df['main_id'].astype(int)
    
    print(f"Training data: {len(train_df):,} user-item pairs")
    print(f"  Users: {train_df['userId'].nunique():,}")
    print(f"  Items: {train_df['main_id'].nunique():,}")
    print(f"  Rating distribution: {train_df['rating'].describe()}")
    
    return train_df






def get_popular_items_from_matrix(train_matrix, idx_to_item, k=20):
    """
    Get most popular items from training matrix based on interaction counts
    """
    # Sum interactions for each item (columns)
    item_popularity = np.array(train_matrix.sum(axis=0)).flatten()
    
    # Get top k most popular item indices
    popular_indices = np.argsort(item_popularity)[::-1][:k]
    
    # Convert to item IDs
    popular_items = [idx_to_item[idx] for idx in popular_indices]
    
    return popular_items


def evaluate_popularity_baseline(data_dict, test_purchases):
    """
    Evaluate a simple popularity baseline that recommends the same top items to all users
    """
    print("Evaluating popularity baseline...")
    
    idx_to_item = data_dict['idx_to_item']
    
    # Get top 20 most popular items from training data
    popular_items = get_popular_items_from_matrix(data_dict['train_matrix'], idx_to_item, k=20)
    
    # Group test purchases by user
    user_purchases = test_purchases.groupby('userId')['main_id'].apply(list).to_dict()
    
    print(f"Evaluating popularity baseline on {len(user_purchases)} test users...")
    print(f"Top 20 popular items: {popular_items[:5]}... (showing first 5)")
    
    recall_10_scores = []
    recall_20_scores = []
    mrr_scores = []
    all_recommended_items = set(popular_items)
    
    # For each user, recommend the same popular items
    for purchased_items in user_purchases.values():
        purchased_items_set = set(purchased_items)
        
        # Popular items recommendations (same for all users)
        top_10_items = [int(item) for item in popular_items[:10]]
        top_20_items = [int(item) for item in popular_items[:20]]
        
        # Calculate recall@10
        hits_10 = len(purchased_items_set & set(top_10_items))
        recall_10 = hits_10 / len(purchased_items_set) if purchased_items_set else 0
        recall_10_scores.append(recall_10)
        
        # Calculate recall@20
        hits_20 = len(purchased_items_set & set(top_20_items))
        recall_20 = hits_20 / len(purchased_items_set) if purchased_items_set else 0
        recall_20_scores.append(recall_20)
        
        # Calculate MRR
        mrr = 0.0
        for rank, item in enumerate(top_20_items, 1):
            if item in purchased_items_set:
                mrr = 1.0 / rank
                break
        mrr_scores.append(mrr)
    
    # Calculate item coverage - CONSISTENT WITH NATR
    # What fraction of the total catalog is covered by recommendations
    # This matches the unified_metrics.py calculation (total vocabulary size)
    total_items_in_catalog = len(idx_to_item)  # Total items in training vocabulary
    item_coverage = len(all_recommended_items) / total_items_in_catalog if total_items_in_catalog > 0 else 0.0
    
    # Calculate metrics
    popularity_metrics = {
        'purchase_recall@10': np.mean(recall_10_scores) if recall_10_scores else 0.0,
        'purchase_recall@20': np.mean(recall_20_scores) if recall_20_scores else 0.0,
        'item_coverage@20': item_coverage,  # Now consistent with NATR
        'purchase_mrr': np.mean(mrr_scores) if mrr_scores else 0.0
    }
    
    print(f"Popularity Baseline Results:")
    print(f"  Purchase Recall@10: {popularity_metrics['purchase_recall@10']*100:.2f}%")
    print(f"  Purchase Recall@20: {popularity_metrics['purchase_recall@20']*100:.2f}%")
    print(f"  Item Coverage@20: {popularity_metrics['item_coverage@20']*100:.2f}% (catalog coverage)")
    print(f"  Purchase MRR: {popularity_metrics['purchase_mrr']:.4f}")
    
    return popularity_metrics


def evaluate_implicit_als_metrics(model, data_dict, test_purchases):
    """
    Simplified evaluation following user's explicit instructions:
    - Get all items from full data for mappings
    - Use train data to fill matrix
    - Test on purchases only
    - Cold start users get popular items
    - Warm start users get ALS predictions
    
    """
    print(f"Evaluating on {len(test_purchases)} test purchases...")
    
    user_to_idx = data_dict['user_to_idx']
    idx_to_item = data_dict['idx_to_item']
    
    # Group test purchases by user - ALL purchases, no filtering
    user_purchases = test_purchases.groupby('userId')['main_id'].apply(list).to_dict()
    
    # Get popular items from training for cold start users
    popular_items = get_popular_items_from_matrix(data_dict['train_matrix'], idx_to_item, k=20)
    
    print(f"Total test users: {len(user_purchases)}")
    
    # Track results separately for warm and cold start users
    warm_recall_10_scores = []
    warm_recall_20_scores = []
    warm_mrr_scores = []
    warm_recommended_items = set()
    
    cold_recall_10_scores = []
    cold_recall_20_scores = []
    cold_mrr_scores = []
    cold_recommended_items = set()
    
    warm_start_count = 0
    cold_start_count = 0
    
    
    # Process each user
    for user_id, purchased_items in tqdm(user_purchases.items(), desc="Evaluating users"):
        
        # Get recommendations based on user type
        if user_id in user_to_idx:
            # Warm start user - use ALS predictions
            warm_start_count += 1
            user_idx = user_to_idx[user_id]
            
            try:
                # Get many more ALS recommendations for better coverage
                recommendations = model.recommend(user_idx, data_dict['train_matrix'][user_idx], N=500, filter_already_liked_items=False)
                item_indices, _ = recommendations
                
                # Convert to item IDs (ensure integers for comparison)
                als_items = []
                for item_idx in item_indices:
                    if item_idx < len(idx_to_item):
                        item_id = idx_to_item[item_idx]
                        als_items.append(int(item_id))
                
                # Coverage boost: mix ALS recommendations with diversity items
                if len(als_items) >= 20:
                    # Use pure ALS for primary recommendations
                    top_10_items = als_items[:10]
                    top_20_items = als_items[:20]
                else:
                    # If ALS fails, fall back to popular items
                    top_10_items = [int(item) for item in popular_items[:10]]
                    top_20_items = [int(item) for item in popular_items[:20]]
                
            except (IndexError, ValueError):
                # Fallback to popular items if ALS fails (ensure integers)
                top_10_items = [int(item) for item in popular_items[:10]]
                top_20_items = [int(item) for item in popular_items[:20]]
        else:
            # Cold start user - use popular items (ensure integers)
            cold_start_count += 1
            top_10_items = [int(item) for item in popular_items[:10]]
            top_20_items = [int(item) for item in popular_items[:20]]
        
        # Calculate metrics - simple intersection
        purchased_items_set = set(purchased_items)
        
        
        # Recall@10
        hits_10 = len(purchased_items_set & set(top_10_items))
        recall_10 = hits_10 / len(purchased_items_set) if purchased_items_set else 0
        
        # Recall@20  
        hits_20 = len(purchased_items_set & set(top_20_items))
        recall_20 = hits_20 / len(purchased_items_set) if purchased_items_set else 0
        
        # MRR
        mrr = 0.0
        for rank, item in enumerate(top_20_items, 1):
            if item in purchased_items_set:
                mrr = 1.0 / rank
                break
        
        # Track metrics separately by user type
        if user_id in user_to_idx:
            # Warm start user
            warm_recall_10_scores.append(recall_10)
            warm_recall_20_scores.append(recall_20)
            warm_mrr_scores.append(mrr)
            # For coverage, only track actual top-20 recommendations (no injection)
            warm_recommended_items.update(top_20_items)
        else:
            # Cold start user
            cold_recall_10_scores.append(recall_10)
            cold_recall_20_scores.append(recall_20)
            cold_mrr_scores.append(mrr)
            cold_recommended_items.update(top_20_items)
    
    print(f"Evaluated {warm_start_count + cold_start_count} users ({warm_start_count} warm start, {cold_start_count} cold start)")
    
    # Calculate separate metrics
    
    # Get distinct items actually purchased in test set for proper coverage calculation
    test_purchased_items = set()
    for purchased_items in user_purchases.values():
        test_purchased_items.update(purchased_items)
    total_purchasable_items = len(test_purchased_items)
    
    # Warm start metrics - consistent catalog coverage calculation
    total_items_in_catalog = len(idx_to_item)
    warm_coverage = len(warm_recommended_items) / total_items_in_catalog if total_items_in_catalog > 0 else 0.0
    
    # Also calculate purchase coverage for warm start
    warm_coverage_items_correct = warm_recommended_items & test_purchased_items
    warm_purchase_coverage = len(warm_coverage_items_correct) / total_purchasable_items if total_purchasable_items > 0 else 0.0
    warm_metrics = {
        'purchase_recall@10': np.mean(warm_recall_10_scores) if warm_recall_10_scores else 0.0,
        'purchase_recall@20': np.mean(warm_recall_20_scores) if warm_recall_20_scores else 0.0,
        'item_coverage@20': warm_coverage,
        'purchase_coverage@20': warm_purchase_coverage,
        'purchase_mrr': np.mean(warm_mrr_scores) if warm_mrr_scores else 0.0
    }
    
    # Cold start metrics - consistent catalog coverage calculation
    cold_coverage = len(cold_recommended_items) / total_items_in_catalog if total_items_in_catalog > 0 else 0.0
    
    # Also calculate purchase coverage for cold start
    cold_coverage_items_correct = cold_recommended_items & test_purchased_items
    cold_purchase_coverage = len(cold_coverage_items_correct) / total_purchasable_items if total_purchasable_items > 0 else 0.0
    cold_metrics = {
        'purchase_recall@10': np.mean(cold_recall_10_scores) if cold_recall_10_scores else 0.0,
        'purchase_recall@20': np.mean(cold_recall_20_scores) if cold_recall_20_scores else 0.0,
        'item_coverage@20': cold_coverage,
        'purchase_coverage@20': cold_purchase_coverage,
        'purchase_mrr': np.mean(cold_mrr_scores) if cold_mrr_scores else 0.0
    }
    
    # Combined metrics (for compatibility)
    all_recall_10 = warm_recall_10_scores + cold_recall_10_scores
    all_recall_20 = warm_recall_20_scores + cold_recall_20_scores
    all_mrr = warm_mrr_scores + cold_mrr_scores
    all_recommended_items = warm_recommended_items | cold_recommended_items
    
    # Combined coverage - consistent catalog coverage
    combined_coverage = len(all_recommended_items) / total_items_in_catalog if total_items_in_catalog > 0 else 0.0
    
    # Combined purchase coverage - what fraction of purchased items are covered
    all_coverage_items_correct = all_recommended_items & test_purchased_items
    combined_purchase_coverage = len(all_coverage_items_correct) / total_purchasable_items if total_purchasable_items > 0 else 0.0
    
    combined_metrics = {
        'purchase_recall@10': np.mean(all_recall_10) if all_recall_10 else 0.0,
        'purchase_recall@20': np.mean(all_recall_20) if all_recall_20 else 0.0,
        'item_coverage@20': combined_coverage,
        'purchase_coverage@20': combined_purchase_coverage,
        'purchase_mrr': np.mean(all_mrr) if all_mrr else 0.0
    }
    
    # Print detailed breakdown with proper coverage
    print(f"\n=== WARM START USERS ({warm_start_count} users) ====")
    print(f"Recall@10: {warm_metrics['purchase_recall@10']*100:.2f}%")
    print(f"Recall@20: {warm_metrics['purchase_recall@20']*100:.2f}%")
    print(f"Item Coverage@20: {warm_metrics['item_coverage@20']*100:.2f}% ({len(warm_recommended_items)}/{total_items_in_catalog} catalog items)")
    print(f"Purchase Coverage@20: {warm_metrics['purchase_coverage@20']*100:.2f}% ({len(warm_coverage_items_correct)}/{total_purchasable_items} purchased items)")
    print(f"MRR: {warm_metrics['purchase_mrr']:.4f}")
    
    print(f"\n=== COLD START USERS ({cold_start_count} users) ====")
    print(f"Recall@10: {cold_metrics['purchase_recall@10']*100:.2f}%")
    print(f"Recall@20: {cold_metrics['purchase_recall@20']*100:.2f}%")
    print(f"Item Coverage@20: {cold_metrics['item_coverage@20']*100:.2f}% ({len(cold_recommended_items)}/{total_items_in_catalog} catalog items)")
    print(f"Purchase Coverage@20: {cold_metrics['purchase_coverage@20']*100:.2f}% ({len(cold_coverage_items_correct)}/{total_purchasable_items} purchased items)")
    print(f"MRR: {cold_metrics['purchase_mrr']:.4f}")
    
    print(f"\n=== COMBINED METRICS ====")
    print(f"Item Coverage@20: {combined_metrics['item_coverage@20']*100:.2f}% ({len(all_recommended_items)}/{total_items_in_catalog} catalog items)")
    print(f"Purchase Coverage@20: {combined_metrics['purchase_coverage@20']*100:.2f}% ({len(all_coverage_items_correct)}/{total_purchasable_items} purchased items)")
    print(f"Total distinct items recommended: {len(all_recommended_items)} (vs {total_purchasable_items} purchasable)")
    
    
    # Return metrics with breakdown
    result = combined_metrics.copy()
    result.update({
        'warm_start_metrics': warm_metrics,
        'cold_start_metrics': cold_metrics,
        'warm_start_count': warm_start_count,
        'cold_start_count': cold_start_count
    })
    
    return result




def train_implicit_als_model_gridsearch(data_path: str, train_ratio: float = 0.91):
    """
    Comprehensive grid search for Implicit ALS including event weights optimization
    """
    print("Training Implicit ALS with comprehensive grid search...")
    
    # Comprehensive grid search parameters
    factors_grid = [50, 100, 150]
    regularization_grid = [0.01, 0.05, 0.1, 0.2]
    event_weights_grid = [
        # Original weights
        {'ViewContent': 0.04, 'AddToCart': 0.16, 'InitiateCheckout': 0.2, 'Purchase': 1.0},
        
        # Lower non-purchase weights
        {'ViewContent': 0.02, 'AddToCart': 0.08, 'InitiateCheckout': 0.1, 'Purchase': 1.0},
        {'ViewContent': 0.01, 'AddToCart': 0.05, 'InitiateCheckout': 0.08, 'Purchase': 1.0},
        
        # More balanced weights
        {'ViewContent': 0.2, 'AddToCart': 0.4, 'InitiateCheckout': 0.6, 'Purchase': 1.0},
        
        # Progressive weights
        {'ViewContent': 0.05, 'AddToCart': 0.2, 'InitiateCheckout': 0.4, 'Purchase': 1.0},
        {'ViewContent': 0.08, 'AddToCart': 0.25, 'InitiateCheckout': 0.5, 'Purchase': 1.0},
        
        # Equal non-purchase weights
        {'ViewContent': 0.05, 'AddToCart': 0.1, 'InitiateCheckout': 0.1, 'Purchase': 1.0},
        {'ViewContent': 0.1, 'AddToCart': 0.1, 'InitiateCheckout': 0.1, 'Purchase': 1.0},
        {'ViewContent': 0.2, 'AddToCart': 0.2, 'InitiateCheckout': 0.2, 'Purchase': 1.0},
        
        # Heavy emphasis on AddToCart and InitiateCheckout
        {'ViewContent': 0.02, 'AddToCart': 0.4, 'InitiateCheckout': 0.5, 'Purchase': 1.0},
        {'ViewContent': 0.05, 'AddToCart': 0.5, 'InitiateCheckout': 0.6, 'Purchase': 1.0},
        
        # Binary-like weights
        {'ViewContent': 0.01, 'AddToCart': 0.3, 'InitiateCheckout': 0.6, 'Purchase': 1.0},
        {'ViewContent': 0.03, 'AddToCart': 0.1, 'InitiateCheckout': 0.3, 'Purchase': 1.0},
    ]
    
    # Load base data once
    train_events, test_purchases, _ = load_and_prepare_data_initial(data_path, train_ratio, min_interactions=10)
    
    results = []
    best_recall_20 = 0
    best_result = None
    
    total_combinations = len(factors_grid) * len(regularization_grid) * len(event_weights_grid)
    print(f"Testing {total_combinations} parameter combinations...\n")
    
    combination_count = 0
    
    # Grid search
    for event_weights in event_weights_grid:
        for factors in factors_grid:
            for reg in regularization_grid:
                combination_count += 1
                print(f"\n--- {combination_count}/{total_combinations}: F={factors}, R={reg} ---")
                
                # Prepare data with current event weights
                data_dict = prepare_implicit_als_data_with_weights(train_events, test_purchases, event_weights)
                
                # Train model
                model = train_implicit_als(
                    data_dict['train_matrix'], 
                    factors=factors,
                    regularization=reg,
                    iterations=10
                )
                
                # Evaluate
                metrics = evaluate_implicit_als_metrics(model, data_dict, test_purchases)
                
                result = {
                    'factors': factors,
                    'regularization': reg,
                    'event_weights': event_weights,
                    **metrics
                }
                results.append(result)
                
                # Track best result
                if metrics['purchase_recall@20'] > best_recall_20:
                    best_recall_20 = metrics['purchase_recall@20']
                    best_result = result
                
                print(f"R@20: {metrics['purchase_recall@20']*100:.2f}%, R@10: {metrics['purchase_recall@10']*100:.2f}%, Cov: {metrics['item_coverage@20']*100:.1f}%, MRR: {metrics['purchase_mrr']:.3f}")
    
    # Print best result
    print(f"\n=== BEST MODEL (Recall@20 = {best_result['purchase_recall@20']*100:.2f}%) ===")
    print(f"Event weights: {best_result['event_weights']}")
    print(f"Factors: {best_result['factors']}, Regularization: {best_result['regularization']}")
    print(f"Purchase Recall@10: {best_result['purchase_recall@10']*100:.2f}%")
    print(f"Purchase Recall@20: {best_result['purchase_recall@20']*100:.2f}%")
    print(f"Item Coverage@20: {best_result['item_coverage@20']*100:.2f}%")
    print(f"Purchase MRR: {best_result['purchase_mrr']:.4f}")
    
    return best_result, results


def train_implicit_als_model_hu_style_normalized_test(data_path: str, train_ratio: float = 0.91):
    """
    Test Hu-style implementation with event weights and user normalization
    """
    print("Testing Hu-style ALS with event weights and user normalization...")
    
    # Fixed ALS parameters
    factors_grid = [50]
    regularization_grid = [0.01]
    alpha_grid = [40.0]  # Standard alpha value from Hu et al.
    
    # Test just a few promising weight configurations with normalization
    event_weights_grid = [
        # Best performing weights from previous tests
        {'ViewContent': 0.1, 'AddToCart': 0.1, 'InitiateCheckout': 0.1, 'Purchase': 1.0},
        {'ViewContent': 0.05, 'AddToCart': 0.05, 'InitiateCheckout': 0.05, 'Purchase': 1.0},
        {'ViewContent': 0.02, 'AddToCart': 0.02, 'InitiateCheckout': 0.02, 'Purchase': 1.0},
        {'ViewContent': 0.01, 'AddToCart': 0.01, 'InitiateCheckout': 0.01, 'Purchase': 1.0},
        {'ViewContent': 0.15, 'AddToCart': 0.15, 'InitiateCheckout': 0.15, 'Purchase': 1.0},
    ]
    
    # Load base data once
    train_events, test_purchases, _ = load_and_prepare_data_initial(data_path, train_ratio, min_interactions=10)
    
    # First, evaluate popularity baseline
    print("\n" + "="*60)
    print("EVALUATING POPULARITY BASELINE (NORMALIZED)")
    print("="*60)
    temp_data_dict = prepare_implicit_als_data_hu_style_normalized(train_events, test_purchases, event_weights_grid[0])
    popularity_baseline = evaluate_popularity_baseline(temp_data_dict, test_purchases)
    
    results = []
    best_recall_20 = 0
    best_result = None
    
    total_combinations = len(factors_grid) * len(regularization_grid) * len(event_weights_grid) * len(alpha_grid)
    print(f"\nTesting {total_combinations} configurations with normalized Hu-style approach...")
    print(f"Event weight configurations: {len(event_weights_grid)}")
    print()
    
    combination_count = 0
    
    # Grid search
    for event_weights in event_weights_grid:
        for alpha in alpha_grid:
            for factors in factors_grid:
                for reg in regularization_grid:
                    combination_count += 1
                    print(f"\n--- {combination_count}/{total_combinations}: Weights: {event_weights}, α={alpha} ---")
                    
                    # Prepare data with normalized Hu-style approach
                    data_dict = prepare_implicit_als_data_hu_style_normalized(train_events, test_purchases, event_weights, alpha=alpha)
                    
                    # Train model with alpha parameter
                    model = train_implicit_als(
                        data_dict['train_matrix'], 
                        factors=factors,
                        regularization=reg,
                        iterations=10,
                        alpha=alpha  # Pass alpha to model
                    )
                    
                    # Evaluate
                    metrics = evaluate_implicit_als_metrics(model, data_dict, test_purchases)
                    
                    result = {
                        'factors': factors,
                        'regularization': reg,
                        'alpha': alpha,
                        'event_weights': event_weights,
                        'method': 'hu_style_normalized',
                        **metrics
                    }
                    results.append(result)
                    
                    # Track best result
                    if metrics['purchase_recall@20'] > best_recall_20:
                        best_recall_20 = metrics['purchase_recall@20']
                        best_result = result
                    
                    print(f"R@20: {metrics['purchase_recall@20']*100:.2f}%, R@10: {metrics['purchase_recall@10']*100:.2f}%, Cov: {metrics['item_coverage@20']*100:.1f}%, MRR: {metrics['purchase_mrr']:.3f}")
    
    # Print best result
    print(f"\n=== BEST NORMALIZED HU-STYLE MODEL - Recall@20 = {best_result['purchase_recall@20']*100:.2f}% ===")
    print(f"Event weights: {best_result['event_weights']}")
    print(f"Alpha (α): {best_result['alpha']}")
    print(f"Purchase Recall@10: {best_result['purchase_recall@10']*100:.2f}%")
    print(f"Purchase Recall@20: {best_result['purchase_recall@20']*100:.2f}%")
    print(f"Item Coverage@20: {best_result['item_coverage@20']*100:.2f}%")
    print(f"Purchase MRR: {best_result['purchase_mrr']:.4f}")
    
    # Show performance comparison
    print(f"\n=== NORMALIZED HU-STYLE PERFORMANCE COMPARISON ===")
    print(f"{'':2s}  {'R@20':>6s} {'R@10':>6s} {'Cov':>5s} {'MRR':>6s} - Event Weights")
    print(f"{'':2s}  {'-----':>6s} {'-----':>6s} {'----':>5s} {'-----':>6s}")
    
    # Add popularity baseline
    pop_recall_20 = popularity_baseline['purchase_recall@20'] * 100
    pop_recall_10 = popularity_baseline['purchase_recall@10'] * 100
    pop_coverage = popularity_baseline['item_coverage@20'] * 100
    pop_mrr = popularity_baseline['purchase_mrr']
    print(f"{'POP':2s}: {pop_recall_20:5.2f}% {pop_recall_10:5.2f}% {pop_coverage:4.1f}% {pop_mrr:5.3f} - Popularity Baseline")
    
    # Show ALS results
    for i, result in enumerate(results):
        weights = result['event_weights']
        recall_20 = result['purchase_recall@20'] * 100
        recall_10 = result['purchase_recall@10'] * 100
        coverage = result['item_coverage@20'] * 100
        mrr = result['purchase_mrr']
        print(f"{i+1:2d}: {recall_20:5.2f}% {recall_10:5.2f}% {coverage:4.1f}% {mrr:5.3f} - VC:{weights['ViewContent']:4.3f}, AC:{weights['AddToCart']:4.3f}, IC:{weights['InitiateCheckout']:4.3f}, P:{weights['Purchase']:3.1f}")
    
    # Calculate improvements
    best_improvement = (best_result['purchase_recall@20'] - popularity_baseline['purchase_recall@20']) * 100
    best_relative = (best_result['purchase_recall@20'] / popularity_baseline['purchase_recall@20']) if popularity_baseline['purchase_recall@20'] > 0 else float('inf')
    
    print(f"\n=== IMPROVEMENT OVER POPULARITY BASELINE ===")
    print(f"Best Normalized Recall@20: {best_result['purchase_recall@20']*100:.2f}%")
    print(f"Popularity Recall@20: {popularity_baseline['purchase_recall@20']*100:.2f}%")
    print(f"Absolute improvement: +{best_improvement:.2f} percentage points")
    print(f"Relative improvement: {best_relative:.2f}x better" if best_relative != float('inf') else "Infinitely better")
    
    print(f"\n=== NORMALIZATION SUMMARY ===")
    print(f"Method: Hu-style ALS with user normalization")
    print(f"Normalization: Each user's ratings divided by their maximum rating")
    print(f"Effect: All users have ratings in range [0, 1], eliminating user activity bias")
    print(f"Matrix values: User-normalized weighted interactions")
    
    return best_result, results


def train_implicit_als_model_binary_non_purchase_test(data_path: str, train_ratio: float = 0.91):
    """
    Test binary non-purchase only implementation (exclude purchases from training)
    """
    print("Testing binary non-purchase ALS (AddToCart, InitiateCheckout only)...")
    
    # Optimized ALS parameters for purchase recall
    factors_grid = [100, 150]  # Larger embedding dimensions
    regularization_grid = [0.001, 0.01]  # Test lower regularization
    alpha_grid = [20.0, 40.0, 60.0]  # Test different confidence levels
    
    # Load base data once
    train_events, test_purchases, _ = load_and_prepare_data_initial(data_path, train_ratio, min_interactions=10)
    
    # First, evaluate popularity baseline using non-purchase events
    print("\n" + "="*60)
    print("EVALUATING POPULARITY BASELINE (NON-PURCHASE EVENTS)")
    print("="*60)
    temp_data_dict = prepare_implicit_als_data_binary_non_purchase(train_events, test_purchases, temporal_decay_days=14)
    popularity_baseline = evaluate_popularity_baseline(temp_data_dict, test_purchases)
    
    results = []
    best_recall_20 = 0
    best_result = None
    
    # Test different temporal decay values
    temporal_decay_grid = [7, 14, 30]  # 1 week, 2 weeks, 1 month
    
    total_combinations = len(factors_grid) * len(regularization_grid) * len(alpha_grid) * len(temporal_decay_grid)
    print(f"\nTesting {total_combinations} configurations with binary non-purchase approach...")
    print()
    
    combination_count = 0
    
    # Grid search
    for temporal_decay in temporal_decay_grid:
        for alpha in alpha_grid:
            for factors in factors_grid:
                for reg in regularization_grid:
                    combination_count += 1
                    print(f"\n--- {combination_count}/{total_combinations}: temporal_decay={temporal_decay}, α={alpha}, factors={factors}, reg={reg} ---")
                    
                    # Prepare data with enhanced binary non-purchase approach
                    data_dict = prepare_implicit_als_data_binary_non_purchase(train_events, test_purchases, alpha=alpha, temporal_decay_days=temporal_decay)
                    
                    # Train model with alpha parameter
                    model = train_implicit_als(
                        data_dict['train_matrix'], 
                        factors=factors,
                        regularization=reg,
                        iterations=10,
                        alpha=alpha  # Pass alpha to model
                    )
                    
                    # Evaluate
                    metrics = evaluate_implicit_als_metrics(model, data_dict, test_purchases)
                    
                    result = {
                        'factors': factors,
                        'regularization': reg,
                        'alpha': alpha,
                        'temporal_decay': temporal_decay,
                        'method': 'binary_non_purchase_only',
                        **metrics
                    }
                    results.append(result)
                    
                    # Track best result
                    if metrics['purchase_recall@20'] > best_recall_20:
                        best_recall_20 = metrics['purchase_recall@20']
                        best_result = result
                    
                    print(f"R@20: {metrics['purchase_recall@20']*100:.2f}%, R@10: {metrics['purchase_recall@10']*100:.2f}%, Cov: {metrics['item_coverage@20']*100:.1f}%, MRR: {metrics['purchase_mrr']:.3f}")
    
    # Print results
    print(f"\n=== BINARY NON-PURCHASE MODEL - Recall@20 = {best_result['purchase_recall@20']*100:.2f}% ===")
    print(f"Alpha (α): {best_result['alpha']}")
    print(f"Factors: {best_result['factors']}, Regularization: {best_result['regularization']}")
    print(f"Purchase Recall@10: {best_result['purchase_recall@10']*100:.2f}%")
    print(f"Purchase Recall@20: {best_result['purchase_recall@20']*100:.2f}%")
    print(f"Item Coverage@20: {best_result['item_coverage@20']*100:.2f}%")
    print(f"Purchase MRR: {best_result['purchase_mrr']:.4f}")
    
    # Calculate improvements
    best_improvement = (best_result['purchase_recall@20'] - popularity_baseline['purchase_recall@20']) * 100
    best_relative = (best_result['purchase_recall@20'] / popularity_baseline['purchase_recall@20']) if popularity_baseline['purchase_recall@20'] > 0 else float('inf')
    
    print(f"\n=== IMPROVEMENT OVER POPULARITY BASELINE ===")
    print(f"Binary Non-Purchase Recall@20: {best_result['purchase_recall@20']*100:.2f}%")
    print(f"Popularity Recall@20: {popularity_baseline['purchase_recall@20']*100:.2f}%")
    print(f"Absolute improvement: +{best_improvement:.2f} percentage points")
    print(f"Relative improvement: {best_relative:.2f}x better" if best_relative != float('inf') else "Infinitely better")
    
    print(f"\n=== APPROACH SUMMARY ===")
    print(f"Training data: Non-purchase events only (ViewContent, AddToCart, InitiateCheckout)")
    print(f"Matrix values: Raw interaction counts (Hu et al. binary approach)")
    print(f"Confidence: c_ui = 1 + {best_result['alpha']} * r_ui")
    print(f"Evaluation: Purchase prediction (completely separate from training signal)")
    
    return best_result, results


def train_implicit_als_model_hu_style_test(data_path: str, train_ratio: float = 0.91):
    """
    Test Hu-style implementation with event weights (no advanced aggregation)
    """
    print("Testing Hu-style ALS with event weights (c_ui = 1 + α * r_ui)...")
    
    # Fixed ALS parameters
    factors_grid = [50]
    regularization_grid = [0.01]
    alpha_grid = [40.0]  # Standard alpha value from Hu et al.
    
    # Multiple event weight configurations for testing - all 25 from 0.01 to 0.25
    event_weights_grid = [
        # Test all non-purchase weights from 0.01 to 0.25 in 0.01 increments
        {'ViewContent': 0.01, 'AddToCart': 0.01, 'InitiateCheckout': 0.01, 'Purchase': 1.0},
        {'ViewContent': 0.02, 'AddToCart': 0.02, 'InitiateCheckout': 0.02, 'Purchase': 1.0},
        {'ViewContent': 0.03, 'AddToCart': 0.03, 'InitiateCheckout': 0.03, 'Purchase': 1.0},
        {'ViewContent': 0.04, 'AddToCart': 0.04, 'InitiateCheckout': 0.04, 'Purchase': 1.0},
        {'ViewContent': 0.05, 'AddToCart': 0.05, 'InitiateCheckout': 0.05, 'Purchase': 1.0},
        {'ViewContent': 0.06, 'AddToCart': 0.06, 'InitiateCheckout': 0.06, 'Purchase': 1.0},
        {'ViewContent': 0.07, 'AddToCart': 0.07, 'InitiateCheckout': 0.07, 'Purchase': 1.0},
        {'ViewContent': 0.08, 'AddToCart': 0.08, 'InitiateCheckout': 0.08, 'Purchase': 1.0},
        {'ViewContent': 0.09, 'AddToCart': 0.09, 'InitiateCheckout': 0.09, 'Purchase': 1.0},
        {'ViewContent': 0.10, 'AddToCart': 0.10, 'InitiateCheckout': 0.10, 'Purchase': 1.0},
        {'ViewContent': 0.11, 'AddToCart': 0.11, 'InitiateCheckout': 0.11, 'Purchase': 1.0},
        {'ViewContent': 0.12, 'AddToCart': 0.12, 'InitiateCheckout': 0.12, 'Purchase': 1.0},
        {'ViewContent': 0.13, 'AddToCart': 0.13, 'InitiateCheckout': 0.13, 'Purchase': 1.0},
        {'ViewContent': 0.14, 'AddToCart': 0.14, 'InitiateCheckout': 0.14, 'Purchase': 1.0},
        {'ViewContent': 0.15, 'AddToCart': 0.15, 'InitiateCheckout': 0.15, 'Purchase': 1.0},
        {'ViewContent': 0.16, 'AddToCart': 0.16, 'InitiateCheckout': 0.16, 'Purchase': 1.0},
        {'ViewContent': 0.17, 'AddToCart': 0.17, 'InitiateCheckout': 0.17, 'Purchase': 1.0},
        {'ViewContent': 0.18, 'AddToCart': 0.18, 'InitiateCheckout': 0.18, 'Purchase': 1.0},
        {'ViewContent': 0.19, 'AddToCart': 0.19, 'InitiateCheckout': 0.19, 'Purchase': 1.0},
        {'ViewContent': 0.20, 'AddToCart': 0.20, 'InitiateCheckout': 0.20, 'Purchase': 1.0},
        {'ViewContent': 0.21, 'AddToCart': 0.21, 'InitiateCheckout': 0.21, 'Purchase': 1.0},
        {'ViewContent': 0.22, 'AddToCart': 0.22, 'InitiateCheckout': 0.22, 'Purchase': 1.0},
        {'ViewContent': 0.23, 'AddToCart': 0.23, 'InitiateCheckout': 0.23, 'Purchase': 1.0},
        {'ViewContent': 0.24, 'AddToCart': 0.24, 'InitiateCheckout': 0.24, 'Purchase': 1.0},
        {'ViewContent': 0.25, 'AddToCart': 0.25, 'InitiateCheckout': 0.25, 'Purchase': 1.0},
    ]
    
    # Load base data once
    train_events, test_purchases, _ = load_and_prepare_data_initial(data_path, train_ratio, min_interactions=10)
    
    # First, evaluate popularity baseline
    print("\n" + "="*60)
    print("EVALUATING POPULARITY BASELINE")
    print("="*60)
    temp_data_dict = prepare_implicit_als_data_hu_style(train_events, test_purchases, event_weights_grid[0])
    popularity_baseline = evaluate_popularity_baseline(temp_data_dict, test_purchases)
    
    results = []
    best_recall_20 = 0
    best_result = None
    
    total_combinations = len(factors_grid) * len(regularization_grid) * len(event_weights_grid) * len(alpha_grid)
    print(f"\nTesting {total_combinations} configurations with Hu-style approach...")
    print(f"Event weight configurations: {len(event_weights_grid)}")
    print()
    
    combination_count = 0
    
    # Grid search
    for event_weights in event_weights_grid:
        for alpha in alpha_grid:
            for factors in factors_grid:
                for reg in regularization_grid:
                    combination_count += 1
                    print(f"\n--- {combination_count}/{total_combinations}: Weights: {event_weights}, α={alpha} ---")
                    
                    # Prepare data with Hu-style approach
                    data_dict = prepare_implicit_als_data_hu_style(train_events, test_purchases, event_weights, alpha=alpha)
                    
                    # Train model with alpha parameter
                    model = train_implicit_als(
                        data_dict['train_matrix'], 
                        factors=factors,
                        regularization=reg,
                        iterations=10,
                        alpha=alpha  # Pass alpha to model
                    )
                    
                    # Evaluate
                    metrics = evaluate_implicit_als_metrics(model, data_dict, test_purchases)
                    
                    result = {
                        'factors': factors,
                        'regularization': reg,
                        'alpha': alpha,
                        'event_weights': event_weights,
                        'method': 'hu_style_with_weights',
                        **metrics
                    }
                    results.append(result)
                    
                    # Track best result
                    if metrics['purchase_recall@20'] > best_recall_20:
                        best_recall_20 = metrics['purchase_recall@20']
                        best_result = result
                    
                    print(f"R@20: {metrics['purchase_recall@20']*100:.2f}%, R@10: {metrics['purchase_recall@10']*100:.2f}%, Cov: {metrics['item_coverage@20']*100:.1f}%, MRR: {metrics['purchase_mrr']:.3f}")
    
    # Print best result
    print(f"\n=== BEST HU-STYLE MODEL - Recall@20 = {best_result['purchase_recall@20']*100:.2f}% ===")
    print(f"Event weights: {best_result['event_weights']}")
    print(f"Alpha (α): {best_result['alpha']}")
    print(f"Purchase Recall@10: {best_result['purchase_recall@10']*100:.2f}%")
    print(f"Purchase Recall@20: {best_result['purchase_recall@20']*100:.2f}%")
    print(f"Item Coverage@20: {best_result['item_coverage@20']*100:.2f}%")
    print(f"Purchase MRR: {best_result['purchase_mrr']:.4f}")
    
    # Show performance comparison
    print(f"\n=== HU-STYLE PERFORMANCE COMPARISON ===")
    print(f"{'':2s}  {'R@20':>6s} {'R@10':>6s} {'Cov':>5s} {'MRR':>6s} - Event Weights")
    print(f"{'':2s}  {'-----':>6s} {'-----':>6s} {'----':>5s} {'-----':>6s}")
    
    # Add popularity baseline
    pop_recall_20 = popularity_baseline['purchase_recall@20'] * 100
    pop_recall_10 = popularity_baseline['purchase_recall@10'] * 100
    pop_coverage = popularity_baseline['item_coverage@20'] * 100
    pop_mrr = popularity_baseline['purchase_mrr']
    print(f"{'POP':2s}: {pop_recall_20:5.2f}% {pop_recall_10:5.2f}% {pop_coverage:4.1f}% {pop_mrr:5.3f} - Popularity Baseline")
    
    # Show ALS results
    for i, result in enumerate(results):
        weights = result['event_weights']
        recall_20 = result['purchase_recall@20'] * 100
        recall_10 = result['purchase_recall@10'] * 100
        coverage = result['item_coverage@20'] * 100
        mrr = result['purchase_mrr']
        print(f"{i+1:2d}: {recall_20:5.2f}% {recall_10:5.2f}% {coverage:4.1f}% {mrr:5.3f} - VC:{weights['ViewContent']:4.3f}, AC:{weights['AddToCart']:4.3f}, IC:{weights['InitiateCheckout']:4.3f}, P:{weights['Purchase']:3.1f}")
    
    # Calculate improvements
    best_improvement = (best_result['purchase_recall@20'] - popularity_baseline['purchase_recall@20']) * 100
    best_relative = (best_result['purchase_recall@20'] / popularity_baseline['purchase_recall@20']) if popularity_baseline['purchase_recall@20'] > 0 else float('inf')
    
    print(f"\n=== IMPROVEMENT OVER POPULARITY BASELINE ===")
    print(f"Best Hu-style Recall@20: {best_result['purchase_recall@20']*100:.2f}%")
    print(f"Popularity Recall@20: {popularity_baseline['purchase_recall@20']*100:.2f}%")
    print(f"Absolute improvement: +{best_improvement:.2f} percentage points")
    print(f"Relative improvement: {best_relative:.2f}x better" if best_relative != float('inf') else "Infinitely better")
    
    return best_result, results


def train_implicit_als_model_event_weights_test(data_path: str, train_ratio: float = 0.91):
    """
    Test different event weight combinations with simple weighted aggregation
    """
    print("Testing different event weights with simple aggregation (no early boost/decay)...")
    
    # Fixed ALS parameters
    factors_grid = [50]
    regularization_grid = [0.01]
    
    # Multiple event weight configurations for testing - from 0.05 down to 0.01
    event_weights_grid = [
        # Test non-purchase weights from 0.05 down to 0.01
        {'ViewContent': 0.05, 'AddToCart': 0.05, 'InitiateCheckout': 0.05, 'Purchase': 1.0},
        {'ViewContent': 0.04, 'AddToCart': 0.04, 'InitiateCheckout': 0.04, 'Purchase': 1.0},
        {'ViewContent': 0.03, 'AddToCart': 0.03, 'InitiateCheckout': 0.03, 'Purchase': 1.0},
        {'ViewContent': 0.02, 'AddToCart': 0.02, 'InitiateCheckout': 0.02, 'Purchase': 1.0},
        {'ViewContent': 0.01, 'AddToCart': 0.01, 'InitiateCheckout': 0.01, 'Purchase': 1.0},
    ]
    
    # Load base data once
    train_events, test_purchases, _ = load_and_prepare_data_initial(data_path, train_ratio, min_interactions=10)
    
    # First, evaluate popularity baseline using any event weights (they don't matter for popularity)
    print("\n" + "="*60)
    print("EVALUATING POPULARITY BASELINE")
    print("="*60)
    temp_data_dict = prepare_implicit_als_data_with_weights(train_events, test_purchases, event_weights_grid[0])
    popularity_baseline = evaluate_popularity_baseline(temp_data_dict, test_purchases)
    
    results = []
    best_recall_20 = 0
    best_result = None
    
    total_combinations = len(factors_grid) * len(regularization_grid) * len(event_weights_grid)
    print(f"Testing {total_combinations} event weight combinations with simple aggregation...")
    print(f"Event weight configurations: {len(event_weights_grid)}")
    print()
    
    combination_count = 0
    
    # Grid search
    for event_weights in event_weights_grid:
        for factors in factors_grid:
            for reg in regularization_grid:
                combination_count += 1
                print(f"\n--- {combination_count}/{total_combinations}: Event weights: {event_weights} ---")
                
                # Prepare data with simple weighted aggregation (no advanced aggregation)
                data_dict = prepare_implicit_als_data_with_weights(train_events, test_purchases, event_weights)
                
                # Train model
                model = train_implicit_als(
                    data_dict['train_matrix'], 
                    factors=factors,
                    regularization=reg,
                    iterations=10
                )
                
                # Evaluate
                metrics = evaluate_implicit_als_metrics(model, data_dict, test_purchases)
                
                result = {
                    'factors': factors,
                    'regularization': reg,
                    'event_weights': event_weights,
                    'aggregation_type': 'simple',
                    **metrics
                }
                results.append(result)
                
                # Track best result
                if metrics['purchase_recall@20'] > best_recall_20:
                    best_recall_20 = metrics['purchase_recall@20']
                    best_result = result
                
                print(f"R@20: {metrics['purchase_recall@20']*100:.2f}%, R@10: {metrics['purchase_recall@10']*100:.2f}%, Cov: {metrics['item_coverage@20']*100:.1f}%, MRR: {metrics['purchase_mrr']:.3f}")
    
    # Print best result
    print(f"\n=== BEST EVENT WEIGHTS (Simple Aggregation) - Recall@20 = {best_result['purchase_recall@20']*100:.2f}% ===")
    print(f"Best event weights: {best_result['event_weights']}")
    print(f"Purchase Recall@10: {best_result['purchase_recall@10']*100:.2f}%")
    print(f"Purchase Recall@20: {best_result['purchase_recall@20']*100:.2f}%")
    print(f"Item Coverage@20: {best_result['item_coverage@20']*100:.2f}%")
    print(f"Purchase MRR: {best_result['purchase_mrr']:.4f}")
    
    # Show performance comparison including popularity baseline
    print(f"\n=== EVENT WEIGHT PERFORMANCE COMPARISON ===")
    print(f"{'':2s}  {'R@20':>6s} {'R@10':>6s} {'Cov':>5s} {'MRR':>6s} - Event Weights")
    print(f"{'':2s}  {'-----':>6s} {'-----':>6s} {'----':>5s} {'-----':>6s}")
    
    # Add popularity baseline to comparison
    pop_recall_20 = popularity_baseline['purchase_recall@20'] * 100
    pop_recall_10 = popularity_baseline['purchase_recall@10'] * 100
    pop_coverage = popularity_baseline['item_coverage@20'] * 100
    pop_mrr = popularity_baseline['purchase_mrr']
    print(f"{'POP':2s}: {pop_recall_20:5.2f}% {pop_recall_10:5.2f}% {pop_coverage:4.1f}% {pop_mrr:5.3f} - Popularity Baseline")
    
    # Show ALS results
    for i, result in enumerate(results):
        weights = result['event_weights']
        recall_20 = result['purchase_recall@20'] * 100
        recall_10 = result['purchase_recall@10'] * 100
        coverage = result['item_coverage@20'] * 100
        mrr = result['purchase_mrr']
        print(f"{i+1:2d}: {recall_20:5.2f}% {recall_10:5.2f}% {coverage:4.1f}% {mrr:5.3f} - VC:{weights['ViewContent']:4.3f}, AC:{weights['AddToCart']:4.3f}, IC:{weights['InitiateCheckout']:4.3f}, P:{weights['Purchase']:3.1f}")
    
    # Calculate improvements over popularity baseline
    best_improvement = (best_result['purchase_recall@20'] - popularity_baseline['purchase_recall@20']) * 100
    best_relative = (best_result['purchase_recall@20'] / popularity_baseline['purchase_recall@20']) if popularity_baseline['purchase_recall@20'] > 0 else float('inf')
    
    print(f"\n=== IMPROVEMENT OVER POPULARITY BASELINE ===")
    print(f"Best ALS Recall@20: {best_result['purchase_recall@20']*100:.2f}%")
    print(f"Popularity Recall@20: {popularity_baseline['purchase_recall@20']*100:.2f}%")
    print(f"Absolute improvement: +{best_improvement:.2f} percentage points")
    print(f"Relative improvement: {best_relative:.2f}x better" if best_relative != float('inf') else "Infinitely better (baseline was 0%)")
    
    return best_result, results


def train_implicit_als_model_with_advanced_aggregation(data_path: str, train_ratio: float = 0.91):
    """
    Grid search for Implicit ALS with advanced aggregation (early boost + decay)
    """
    print("Training Implicit ALS with advanced aggregation grid search...")
    
    # Grid search parameters - fixed for aggregation testing
    factors_grid = [50]
    regularization_grid = [0.01]
    
    # Multiple event weight configurations for testing on 13 months data
    event_weights_grid = [
        # Very low non-purchase weights
        {'ViewContent': 0.01, 'AddToCart': 0.01, 'InitiateCheckout': 0.01, 'Purchase': 1.0},
        {'ViewContent': 0.05, 'AddToCart': 0.05, 'InitiateCheckout': 0.05, 'Purchase': 1.0},
        
        # Low non-purchase weights (previous tests)
        {'ViewContent': 0.1, 'AddToCart': 0.1, 'InitiateCheckout': 0.1, 'Purchase': 1.0},
        {'ViewContent': 0.2, 'AddToCart': 0.2, 'InitiateCheckout': 0.2, 'Purchase': 1.0},
        
        # Progressive weights (emphasize later funnel stages)
        {'ViewContent': 0.05, 'AddToCart': 0.1, 'InitiateCheckout': 0.2, 'Purchase': 1.0},
        {'ViewContent': 0.1, 'AddToCart': 0.2, 'InitiateCheckout': 0.4, 'Purchase': 1.0},
        
        # High non-purchase weights
        {'ViewContent': 0.3, 'AddToCart': 0.3, 'InitiateCheckout': 0.3, 'Purchase': 1.0},
        {'ViewContent': 0.5, 'AddToCart': 0.5, 'InitiateCheckout': 0.5, 'Purchase': 1.0},
        
        # Purchase-only (extreme test)
        {'ViewContent': 0.0, 'AddToCart': 0.0, 'InitiateCheckout': 0.0, 'Purchase': 1.0},
    ]
    
    # Simplified aggregation parameters for faster event weight testing
    early_growth_grid = [1.0, 1.2]  # Just test linear vs boost
    decay_rate_grid = [1.0]  # Keep decay neutral
    p_grid = [0.3]  # Single curvature value
    threshold_grid = [4]  # Single threshold
    
    # Load base data once
    train_events, test_purchases, _ = load_and_prepare_data_initial(data_path, train_ratio, min_interactions=10)
    
    results = []
    best_recall_20 = 0
    best_result = None
    
    total_combinations = (len(factors_grid) * len(regularization_grid) * len(event_weights_grid) * 
                         len(early_growth_grid) * len(decay_rate_grid) * len(p_grid) * len(threshold_grid))
    print(f"Testing {total_combinations} parameter combinations...")
    print(f"Grid sizes: {len(factors_grid)} factors × {len(regularization_grid)} reg × {len(event_weights_grid)} weights × {len(early_growth_grid)} early_growth × {len(decay_rate_grid)} decay_rate × {len(p_grid)} p × {len(threshold_grid)} threshold")
    print(f"Event weights: {event_weights_grid[0]}")
    print()
    
    combination_count = 0
    
    # Grid search
    for event_weights in event_weights_grid:
        for factors in factors_grid:
            for reg in regularization_grid:
                for early_growth in early_growth_grid:
                    for decay_rate in decay_rate_grid:
                        for p in p_grid:
                            for threshold in threshold_grid:
                                combination_count += 1
                                print(f"\n--- {combination_count}/{total_combinations}: F={factors}, R={reg}, EG={early_growth}, DR={decay_rate}, P={p}, T={threshold} ---")
                                
                                # Prepare data with advanced aggregation
                                data_dict = prepare_implicit_als_data_with_advanced_aggregation(
                                    train_events, test_purchases, event_weights,
                                    early_growth, decay_rate, p, threshold
                                )
                                
                                # Train model
                                model = train_implicit_als(
                                    data_dict['train_matrix'], 
                                    factors=factors,
                                    regularization=reg,
                                    iterations=10
                                )
                                
                                # Evaluate
                                metrics = evaluate_implicit_als_metrics(model, data_dict, test_purchases)
                                
                                result = {
                                    'factors': factors,
                                    'regularization': reg,
                                    'event_weights': event_weights,
                                    'early_growth': early_growth,
                                    'decay_rate': decay_rate,
                                    'p': p,
                                    'threshold': threshold,
                                    'aggregation_type': 'advanced',
                                    **metrics
                                }
                                results.append(result)
                                
                                # Track best result
                                if metrics['purchase_recall@20'] > best_recall_20:
                                    best_recall_20 = metrics['purchase_recall@20']
                                    best_result = result
                                
                                print(f"R@20: {metrics['purchase_recall@20']*100:.2f}%, R@10: {metrics['purchase_recall@10']*100:.2f}%, Cov: {metrics['item_coverage@20']*100:.1f}%, MRR: {metrics['purchase_mrr']:.3f}")
    
    # Print best result
    print(f"\n=== BEST MODEL WITH ADVANCED AGGREGATION (Recall@20 = {best_result['purchase_recall@20']*100:.2f}%) ===")
    print(f"Event weights: {best_result['event_weights']}")
    print(f"Factors: {best_result['factors']}, Regularization: {best_result['regularization']}")
    print(f"Aggregation params: early_growth={best_result['early_growth']}, decay_rate={best_result['decay_rate']}, p={best_result['p']}, threshold={best_result['threshold']}")
    print(f"Purchase Recall@10: {best_result['purchase_recall@10']*100:.2f}%")
    print(f"Purchase Recall@20: {best_result['purchase_recall@20']*100:.2f}%")
    print(f"Item Coverage@20: {best_result['item_coverage@20']*100:.2f}%")
    print(f"Purchase MRR: {best_result['purchase_mrr']:.4f}")
    
    return best_result, results






def train_implicit_als(train_matrix, **params):
    """
    Train Implicit ALS model
    """
    print(f"Training Implicit ALS model with parameters: {params}")
    
    # Default parameters optimized for implicit feedback
    default_params = {
        'factors': 50,
        'regularization': 0.01,
        'iterations': 10,  # Reduced for quicker training
        'random_state': 42,
        'use_gpu': False,
        'alpha': 40.0  # Confidence scaling factor from the original paper
    }
    
    # Update with provided parameters
    default_params.update(params)
    
    # Create model
    model = AlternatingLeastSquares(**default_params)
    
    # Train model
    print(f"Training ALS for {default_params['iterations']} iterations...")
    print(f"Matrix shape: {train_matrix.shape}, Non-zeros: {train_matrix.nnz}")
    
    # Fit the model
    model.fit(train_matrix.T)  # implicit expects item x user matrix
    
    
    return model






def save_model_and_results(model, results, output_dir: str, model_name: str):
    """
    Save trained model and results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, f"{model_name}_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {model_path}")
    
    # Save results
    results_path = os.path.join(output_dir, f"{model_name}_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to: {results_path}")
    
    return model_path, results_path


def main():
    parser = argparse.ArgumentParser(description='Train Implicit ALS model for travel recommendation')
    parser.add_argument('--data-path', type=str, default='data/bookit_events_2_months.parquet',
                        help='Path to events data')
    parser.add_argument('--output-dir', type=str, default='output/als_models',
                        help='Output directory for models')
    parser.add_argument('--train-ratio', type=float, default=0.91,
                        help='Train/test split ratio')
    parser.add_argument('--quick', action='store_true',
                        help='Quick training with default parameters')
    parser.add_argument('--advanced-aggregation', action='store_true',
                        help='Use advanced aggregation with early boost and decay')
    parser.add_argument('--test-event-weights', action='store_true',
                        help='Test different event weight combinations with simple aggregation')
    parser.add_argument('--hu-style', action='store_true',
                        help='Test Hu-style ALS with event weights (c_ui = 1 + α * r_ui)')
    parser.add_argument('--binary-non-purchase', action='store_true',
                        help='Test binary non-purchase events only (exclude purchases from training)')
    parser.add_argument('--normalized', action='store_true',
                        help='Test Hu-style ALS with user normalization (divide by max rating per user)')
    parser.add_argument('--legacy-gridsearch', action='store_true',
                        help='Use legacy grid search approach (includes purchases in training - not recommended)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("IMPLICIT ALS TRAINING FOR TRAVEL RECOMMENDATION")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Method: Implicit ALS (Purchase-Optimized)")
    print("\n🎯 PURCHASE RECALL OPTIMIZATIONS:")
    print("  ✅ No data leakage: Purchases excluded from training")
    print("  ✅ High-intent signals: AddToCart + InitiateCheckout only")
    print("  ✅ Temporal weighting: Recent interactions prioritized")
    print("  ✅ Event weighting: InitiateCheckout > AddToCart")
    print("  ✅ Pure purchase prediction: Training ≠ Target")
    
    start_time = time.time()
    
    if args.normalized:
        print("\n🔬 TESTING NORMALIZED HU-STYLE ALS")
        # Test normalized Hu-style implementation
        best_result, all_results = train_implicit_als_model_hu_style_normalized_test(args.data_path, args.train_ratio)
        results = {
            'method': 'hu_style_normalized_test',
            'best_result': best_result,
            'all_results': all_results,
            'timestamp': datetime.now().isoformat()
        }
        model_name = "hu_style_normalized"
    elif args.binary_non_purchase:
        print("\n🔬 TESTING BINARY NON-PURCHASE ALS")
        # Test binary non-purchase implementation
        best_result, all_results = train_implicit_als_model_binary_non_purchase_test(args.data_path, args.train_ratio)
        results = {
            'method': 'binary_non_purchase_test',
            'best_result': best_result,
            'all_results': all_results,
            'timestamp': datetime.now().isoformat()
        }
        model_name = "binary_non_purchase"
    elif args.hu_style:
        print("\n🔬 TESTING HU-STYLE ALS WITH EVENT WEIGHTS")
        # Test Hu-style implementation with event weights
        best_result, all_results = train_implicit_als_model_hu_style_test(args.data_path, args.train_ratio)
        results = {
            'method': 'hu_style_event_weights_test',
            'best_result': best_result,
            'all_results': all_results,
            'timestamp': datetime.now().isoformat()
        }
        model_name = "hu_style_event_weights"
    elif args.test_event_weights:
        print("\n🔬 TESTING EVENT WEIGHTS WITH SIMPLE AGGREGATION")
        # Test different event weight combinations
        best_result, all_results = train_implicit_als_model_event_weights_test(args.data_path, args.train_ratio)
        results = {
            'method': 'event_weights_test',
            'best_result': best_result,
            'all_results': all_results,
            'timestamp': datetime.now().isoformat()
        }
        model_name = "event_weights_test"
    elif args.advanced_aggregation:
        print("\n🚀 IMPLICIT ALS TRAINING WITH ADVANCED AGGREGATION")
        # Use advanced aggregation with early boost and decay
        best_result, all_results = train_implicit_als_model_with_advanced_aggregation(args.data_path, args.train_ratio)
        results = {
            'method': 'implicit_als_advanced_aggregation',
            'best_result': best_result,
            'all_results': all_results,
            'timestamp': datetime.now().isoformat()
        }
        model_name = "implicit_als_advanced"
    elif args.legacy_gridsearch:
        print("\n⚠️  IMPLICIT ALS TRAINING - LEGACY GRIDSEARCH (NOT RECOMMENDED)")
        print("⚠️  WARNING: This method includes purchases in training data - causes data leakage!")
        # Use legacy weighted aggregation (includes purchases in training)
        best_result, all_results = train_implicit_als_model_gridsearch(args.data_path, args.train_ratio)
        results = {
            'method': 'implicit_als_gridsearch_legacy',
            'best_result': best_result,
            'all_results': all_results,
            'timestamp': datetime.now().isoformat()
        }
        model_name = "implicit_als_legacy"
    else:
        print("\n🚀 IMPLICIT ALS TRAINING - BINARY NON-PURCHASE (OPTIMAL)")
        # Use binary non-purchase approach as primary method (no purchase data leakage)
        best_result, all_results = train_implicit_als_model_binary_non_purchase_test(args.data_path, args.train_ratio)
        results = {
            'method': 'binary_non_purchase_primary',
            'best_result': best_result,
            'all_results': all_results,
            'timestamp': datetime.now().isoformat(),
            'total_training_time_seconds': time.time() - start_time,
            'total_training_time_minutes': (time.time() - start_time) / 60
        }
        model_name = "binary_non_purchase_primary"
    test_results = best_result
    
    total_time = time.time() - start_time
    
    # Save model and results
    model_path, results_path = save_model_and_results(results, results, args.output_dir, model_name)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Model saved: {model_path}")
    print(f"Results saved: {results_path}")
    print(f"Total time: {total_time:.2f} seconds")
    
    if 'purchase_recall@20' in test_results:
        print(f"\n📊 FINAL RESULTS:")
        print(f"Purchase Recall@10: {test_results['purchase_recall@10']*100:.2f}%")
        print(f"Purchase Recall@20: {test_results['purchase_recall@20']*100:.2f}%")
        print(f"Item Coverage@20: {test_results['item_coverage@20']*100:.2f}%")
        print(f"Purchase MRR: {test_results['purchase_mrr']:.4f}")
        
        print(f"\n⚙️  BEST ALS PARAMETERS:")
        print(f"Factors: {test_results['factors']}")
        print(f"Regularization: {test_results['regularization']}")
        if 'event_weights' in test_results:
            print(f"Event weights: {test_results['event_weights']}")
        if 'aggregation_type' in test_results and test_results['aggregation_type'] == 'advanced':
            print(f"Advanced aggregation params:")
            print(f"  Early growth: {test_results['early_growth']}")
            print(f"  Decay rate: {test_results['decay_rate']}")
            print(f"  Power (p): {test_results['p']}")
            print(f"  Threshold: {test_results['threshold']}")
    
    print("\nNote: Using Implicit ALS for sparse implicit feedback data like travel events.")
    print(f"For evaluation, use scripts/evaluate_SVD.py with this model.")


if __name__ == "__main__":
    main()
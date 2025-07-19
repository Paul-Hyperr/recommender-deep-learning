import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import torch
import os
import pickle
import time
import multiprocessing
from typing import List, Dict, Optional, Any
from tqdm import tqdm

class SessionProcessor2:
    """
    Enhanced SessionProcessor that handles artificial purchase timestamps
    
    This processor addresses the issue where all purchases have 00:00:00 timestamps
    by adjusting them to 23:59:59 on the same day.
    
    Key improvements:
    - Detects artificial purchase timestamps (00:00:00)
    - Adjusts purchase times to end of day (23:59:59)
    - Creates more realistic sessions with proper purchase context
    - Maintains all original functionality
    """
    def __init__(self, event_data_path=None, session_timeout_hours=30, min_interactions=10, 
                 cache_dir='data/cache', max_sessions_per_user=20, max_samples_per_user=10):
        """
        Initialize SessionProcessor2
        
        Args:
            event_data_path (str): Path to the event data file
            session_timeout_hours (int): Timeout for defining user sessions
            min_interactions (int): Minimum number of interactions to keep a user
            cache_dir (str): Directory for caching processed data
            max_sessions_per_user (int): Maximum number of sessions to consider per user
            max_samples_per_user (int): Maximum number of training samples to create per user
        """
        self.event_data_path = event_data_path
        self.data_path = event_data_path  # Keep for backward compatibility
        self.session_timeout_hours = session_timeout_hours
        self.min_interactions = min_interactions
        self.cache_dir = cache_dir
        self.max_sessions_per_user = max_sessions_per_user
        self.max_samples_per_user = max_samples_per_user
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize dictionaries for mapping IDs
        self.user_to_idx = {}
        self.package_to_idx = {}
        self.event_to_idx = {
            'ViewContent': 1,
            'AddToCart': 2,
            'InitiateCheckout': 3,
            'Purchase': 4
        }
        
        # Processed data
        self.df = None
        self.sessions = None
        self.filtered_users = None
        self.purchase_adjustments_made = 0
    
    def _get_session_cache_filename(self):
        """
        Generate a unique cache filename based on the event data file and purchase adjustment setting
        """
        if self.event_data_path:
            base_name = os.path.splitext(os.path.basename(self.event_data_path))[0]
            return f'sessions2_{base_name}_2359.pkl'
        return 'sessions2.pkl'
    
    def clear_cache(self):
        """
        Clear sessions cache file
        """
        try:
            cache_file = os.path.join(self.cache_dir, self._get_session_cache_filename())
            
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"Removed sessions cache file: {cache_file}")
            else:
                print(f"No sessions cache to clear: {cache_file}")
        except Exception as e:
            print(f"Error clearing cache: {e}")
    
    def adjust_purchase_timestamps(self):
        """
        Adjust artificial purchase timestamps to 23:59:59 on the same day
        
        This simple approach sets all 00:00:00 purchases to end of day,
        ensuring they come after all same-day browsing activity.
        """
        if self.df is None:
            print("Error: No data loaded. Call load_data() first.")
            return
        
        print("Adjusting artificial purchase timestamps to 23:59:59...")
        
        # Identify purchases with artificial timestamps (00:00:00)
        purchases_mask = self.df['event'] == 'Purchase'
        artificial_mask = self.df['created_at'].dt.time.astype(str) == '00:00:00'
        artificial_purchases_mask = purchases_mask & artificial_mask
        
        artificial_count = artificial_purchases_mask.sum()
        print(f"Found {artificial_count} purchases with artificial 00:00:00 timestamps")
        
        if artificial_count == 0:
            print("No artificial purchase timestamps found. No adjustments needed.")
            return
        
        # Simple adjustment: Set all artificial purchases to 23:59:59 same day
        self.df.loc[artificial_purchases_mask, 'created_at'] = (
            self.df.loc[artificial_purchases_mask, 'created_at'].apply(
                lambda x: x.replace(hour=23, minute=59, second=59)
            )
        )
        
        self.purchase_adjustments_made = artificial_count
        print(f"Adjusted {artificial_count} purchase timestamps to 23:59:59")
        
        # Verify adjustments
        remaining_artificial = self.df[
            (self.df['event'] == 'Purchase') & 
            (self.df['created_at'].dt.time.astype(str) == '00:00:00')
        ]
        print(f"Remaining artificial purchase timestamps: {len(remaining_artificial)}")
    
    def load_data(self, event_data_path=None):
        """
        Load event data and adjust purchase timestamps
        
        Args:
            event_data_path (str): Optional path to event data file
        """
        # Allow overriding the event data path
        if event_data_path:
            self.event_data_path = event_data_path
            self.data_path = event_data_path
            
        # Load data from source
        if self.event_data_path:
            print(f"Loading event data from {self.event_data_path}")
            
            # Check file extension
            if self.event_data_path.endswith('.parquet'):
                try:
                    self.df = pd.read_parquet(self.event_data_path)
                    
                    # Convert timestamp to datetime if needed
                    if 'created_at' in self.df.columns:
                        self.df['created_at'] = pd.to_datetime(self.df['created_at'])
                    
                    print(f"Loaded {len(self.df)} events from parquet file")
                except Exception as e:
                    print(f"Error loading parquet file: {e}")
                    return
            else:
                # Use chunking for large CSV files
                chunks = pd.read_csv(self.event_data_path, chunksize=1000000)
                
                dfs = []
                for chunk in chunks:
                    if 'created_at' in chunk.columns:
                        chunk['created_at'] = pd.to_datetime(chunk['created_at'])
                    dfs.append(chunk)
                
                self.df = pd.concat(dfs)
            
            # Ensure required columns exist
            required_columns = ['userId', 'main_id', 'event', 'created_at']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            if missing_columns:
                print(f"Error: Missing required columns: {missing_columns}")
                return
            
            # IMPORTANT: Adjust purchase timestamps BEFORE filtering users
            self.adjust_purchase_timestamps()
            
            # Filter users based on interaction count
            self._filter_users()
            
            print(f"Data loading complete. Loaded {len(self.df)} events after filtering.")
        else:
            print("Error: No event data path provided")
    
    def _filter_users(self):
        """Filter users based on minimum interaction count and preserve all purchase users"""
        # Count interactions per user
        user_interaction_counts = self.df['userId'].value_counts()
        
        # Get users with at least min_interactions
        users_with_enough_interactions = user_interaction_counts[user_interaction_counts >= self.min_interactions].index.tolist()
        
        # Keep ALL users who have made purchases, regardless of interaction count
        users_with_purchases = self.df[self.df['event'] == 'Purchase']['userId'].unique().tolist()
        
        # Combine both sets
        self.filtered_users = list(set(users_with_enough_interactions + users_with_purchases))
        
        # Filter dataframe
        self.df = self.df[self.df['userId'].isin(self.filtered_users)]
        
        print(f"\nFiltered to {len(self.filtered_users)} users:")
        print(f"  - {len(users_with_enough_interactions)} users with at least {self.min_interactions} interactions")
        print(f"  - {len(users_with_purchases)} users with purchases total")
        print(f"  - Purchase timestamp adjustments made: {self.purchase_adjustments_made}")
    
    def create_mappings(self):
        """Create mappings from user and package IDs to indices"""
        print("Creating user and package mappings...")
        
        if self.df is None:
            self.load_data()
        
        # Create user mapping (start from 1 to reserve 0 for padding)
        users = self.df['userId'].unique()
        self.user_to_idx = {user: i+1 for i, user in enumerate(users)}
        
        # Create package mapping
        packages = self.df['main_id'].unique()
        self.package_to_idx = {package: i+1 for i, package in enumerate(packages)}
        
        print(f"Created mappings for {len(self.user_to_idx)} users and {len(self.package_to_idx)} packages")
    
    def _process_user_group(self, user_group_data):
        """
        Process sessions for a single user group
        """
        user_group, session_timeout_hours = user_group_data
        
        sessions = []
        current_session = None
        last_timestamp = None
        
        for _, event in user_group.iterrows():
            timestamp = event['created_at']
            event_type = event['event']
            product_id = event['main_id']
            
            # Session timeout logic
            if (current_session is None or 
                (last_timestamp and 
                (timestamp - last_timestamp).total_seconds() / 3600 > session_timeout_hours)):
                
                # Save previous session
                if current_session and current_session['package_ids']:
                    sessions.append(current_session)
                
                # Start new session
                current_session = {
                    'user_id': event['userId'],
                    'start_time': timestamp,
                    'end_time': timestamp,
                    'package_ids': [],
                    'event_types': [],
                    'is_purchase': []
                }
            
            # Update session details
            current_session['end_time'] = timestamp
            last_timestamp = timestamp
            
            # Add interaction
            current_session['package_ids'].append(product_id)
            current_session['event_types'].append(event_type)
            current_session['is_purchase'].append(1 if event_type == 'Purchase' else 0)
        
        # Add final session
        if current_session and current_session['package_ids']:
            sessions.append(current_session)
        
        return sessions

    def extract_sessions(self, use_cache=True):
        """
        Extract user sessions from event data with adjusted purchase timestamps
        """
        cache_file = os.path.join(self.cache_dir, self._get_session_cache_filename())

        # Check cache
        if use_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    loaded_sessions = pickle.load(f)
                
                if loaded_sessions and isinstance(loaded_sessions, list):
                    print(f"Loaded {len(loaded_sessions)} sessions from cache: {os.path.basename(cache_file)}")
                    self.sessions = loaded_sessions
                    return loaded_sessions
            except Exception as e:
                print(f"Cache loading error: {e}")
        
        print("Starting session extraction with adjusted purchase timestamps...")
        start_time = time.time()

        # Sort dataframe by user and timestamp
        df_sorted = self.df.sort_values(['userId', 'created_at']).reset_index(drop=True)

        print("Preparing for parallel processing...")
        num_cores = multiprocessing.cpu_count()
        print(f"Using {num_cores} CPU cores")

        # Prepare user groups with session timeout
        user_groups_with_timeout = [
            (group, self.session_timeout_hours) 
            for _, group in df_sorted.groupby('userId')
        ]
        
        # Use multiprocessing
        with multiprocessing.Pool(processes=num_cores) as pool:
            all_sessions = pool.map(self._process_user_group, user_groups_with_timeout)
        
        # Flatten sessions
        sessions = [session for user_sessions in all_sessions for session in user_sessions]

        total_duration = time.time() - start_time
        print(f"Session extraction time: {total_duration:.2f} seconds")
        print(f"Total sessions extracted: {len(sessions)}")
        
        # Session statistics
        session_lengths = [len(session['package_ids']) for session in sessions]
        print(f"Session length statistics:")
        print(f"Average: {np.mean(session_lengths):.2f}, Median: {np.median(session_lengths):.2f}, Max: {np.max(session_lengths)}")
        
        # Remove outlier sessions but preserve all sessions with purchases
        length_threshold = np.percentile(session_lengths, 99.99)
        purchases_before = sum(sum(session['is_purchase']) for session in sessions)
        
        filtered_sessions = []
        for session in sessions:
            session_length = len(session['package_ids'])
            has_purchase = any(session['is_purchase'])
            
            if session_length <= length_threshold or has_purchase:
                filtered_sessions.append(session)
        
        purchases_after = sum(sum(session['is_purchase']) for session in filtered_sessions)
        
        print(f"Filtered sessions: {len(sessions)} -> {len(filtered_sessions)}")
        print(f"Purchase events preserved: {purchases_before} -> {purchases_after}")
        
        # Cache sessions
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(filtered_sessions, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Sessions saved to cache: {os.path.basename(cache_file)}")
        except Exception as e:
            print(f"Error saving sessions: {e}")
        
        self.sessions = filtered_sessions
        return filtered_sessions
    
    def prepare_enhanced_training_data(self):
        """
        Prepare enhanced training data from sessions with adjusted purchase timestamps
        
        This should now create samples where purchases have proper browsing context
        since purchase timestamps have been adjusted to follow browsing activity.
        """
        
        if not self.sessions:
            self.extract_sessions(use_cache=True)
        
        print("Preparing enhanced training data with adjusted purchase timestamps...")
        
        samples = []
        
        # Group sessions by user
        user_sessions = defaultdict(list)
        for session in self.sessions:
            user_sessions[session['user_id']].append(session)
        
        total_users = len(user_sessions)
        print(f"Processing {total_users} users...")
        
        for user_id, sessions in tqdm(user_sessions.items(), desc="Preparing enhanced training data"):
            # Sort sessions by time
            sessions.sort(key=lambda x: x['start_time'])
            
            # Limit sessions but prioritize those with purchases
            if len(sessions) > self.max_sessions_per_user:
                sessions_with_purchases = [s for s in sessions if any(s['is_purchase'])]
                sessions_without_purchases = [s for s in sessions if not any(s['is_purchase'])]
                
                recent_sessions = sessions_with_purchases[-self.max_sessions_per_user:]
                
                remaining_slots = self.max_sessions_per_user - len(recent_sessions)
                if remaining_slots > 0:
                    recent_sessions = sessions_without_purchases[-remaining_slots:] + recent_sessions
                    recent_sessions.sort(key=lambda x: x['start_time'])
            else:
                recent_sessions = sessions
            
            user_samples = []
            
            # Create samples from sessions
            for i, session in enumerate(recent_sessions):
                purchase_indices = [j for j, is_purchase in enumerate(session['is_purchase']) if is_purchase]
                
                # Create purchase samples
                for purchase_idx in purchase_indices:
                    # Short-term: events before purchase in current session
                    short_term_packages = []
                    short_term_events = []
                    
                    for j in range(purchase_idx):
                        short_term_packages.append(session['package_ids'][j])
                        event_type = session['event_types'][j]
                        short_term_events.append(self.event_to_idx.get(event_type, 0))
                    
                    # Long-term: events from previous sessions
                    long_term_packages = []
                    long_term_events = []
                    
                    for prev_session in recent_sessions[:i]:
                        for j, package_id in enumerate(prev_session['package_ids']):
                            long_term_packages.append(package_id)
                            event_type = prev_session['event_types'][j]
                            long_term_events.append(self.event_to_idx.get(event_type, 0))
                    
                    purchased_package = session['package_ids'][purchase_idx]
                    
                    # Check for checkout and cart events in this sample's context (using event indices)
                    checkout_idx = self.event_to_idx.get('InitiateCheckout', 0)
                    cart_idx = self.event_to_idx.get('AddToCart', 0)
                    has_checkout = checkout_idx in (short_term_events + long_term_events)
                    has_add_to_cart = cart_idx in (short_term_events + long_term_events)
                    
                    sample = {
                        'user_id': user_id,
                        'short_term_packages': short_term_packages,
                        'short_term_events': short_term_events,
                        'long_term_packages': long_term_packages,
                        'long_term_events': long_term_events,
                        'purchased_package': purchased_package,
                        'is_purchase': True,
                        'has_checkout': has_checkout,
                        'has_add_to_cart': has_add_to_cart,
                        'has_checkout_inclusive': False,  # Purchase samples aren't checkout-only
                        'has_add_to_cart_inclusive': False,  # Purchase samples aren't cart-only
                        'session_id': i,
                        'timestamp': session['end_time']
                    }
                    user_samples.append(sample)
                
                # Create non-purchase samples
                if len(purchase_indices) == 0 and len(session['package_ids']) > 0:
                    short_term_packages = []
                    short_term_events = []
                    
                    for j in range(len(session['package_ids'])):
                        short_term_packages.append(session['package_ids'][j])
                        event_type = session['event_types'][j]
                        short_term_events.append(self.event_to_idx.get(event_type, 0))
                    
                    long_term_packages = []
                    long_term_events = []
                    
                    for prev_session in recent_sessions[:i]:
                        for j, package_id in enumerate(prev_session['package_ids']):
                            long_term_packages.append(package_id)
                            event_type = prev_session['event_types'][j]
                            long_term_events.append(self.event_to_idx.get(event_type, 0))
                    
                    # Find highest intent item
                    checkout_indices = [j for j, event in enumerate(session['event_types']) if event == 'InitiateCheckout']
                    cart_indices = [j for j, event in enumerate(session['event_types']) if event == 'AddToCart']
                    
                    if checkout_indices:
                        target_package = session['package_ids'][checkout_indices[-1]]
                        intent_level = 'checkout'
                    elif cart_indices:
                        target_package = session['package_ids'][cart_indices[-1]]
                        intent_level = 'cart'
                    else:
                        from collections import Counter
                        package_counts = Counter(session['package_ids'])
                        most_viewed_package, view_count = package_counts.most_common(1)[0]
                        
                        if view_count > 1:
                            target_package = most_viewed_package
                            intent_level = 'multi_view'
                        else:
                            target_package = session['package_ids'][-1]
                            intent_level = 'single_view'
                    
                    # Check for checkout and cart events in this sample's context (using event indices)
                    checkout_idx = self.event_to_idx.get('InitiateCheckout', 0)
                    cart_idx = self.event_to_idx.get('AddToCart', 0)
                    has_checkout = checkout_idx in (short_term_events + long_term_events)
                    has_add_to_cart = cart_idx in (short_term_events + long_term_events)
                    
                    sample = {
                        'user_id': user_id,
                        'short_term_packages': short_term_packages,
                        'short_term_events': short_term_events,
                        'long_term_packages': long_term_packages,
                        'long_term_events': long_term_events,
                        'purchased_package': target_package,
                        'is_purchase': False,
                        'has_checkout': has_checkout,
                        'has_add_to_cart': has_add_to_cart,
                        'has_checkout_inclusive': (intent_level == 'checkout'),  # True if this is a checkout-only sample
                        'has_add_to_cart_inclusive': (intent_level == 'cart'),  # True if this is a cart-only sample
                        'intent_level': intent_level,
                        'session_id': i,
                        'timestamp': session['end_time']
                    }
                    user_samples.append(sample)
            
            # Limit samples per user
            if user_samples:
                user_samples.sort(key=lambda x: x['timestamp'], reverse=True)
                
                purchase_samples_user = [s for s in user_samples if s['is_purchase']]
                no_purchase_samples_user = [s for s in user_samples if not s['is_purchase']]
                
                limited_samples = purchase_samples_user  # Keep ALL purchases
                
                remaining_slots = max(0, self.max_samples_per_user - len(purchase_samples_user))
                limited_samples.extend(no_purchase_samples_user[:remaining_slots])
                
                samples.extend(limited_samples)
        
        # Statistics
        purchase_count = sum(1 for s in samples if s['is_purchase'])
        no_purchase_count = len(samples) - purchase_count
        empty_short_term = sum(1 for s in samples if not s['short_term_packages'])
        empty_long_term = sum(1 for s in samples if not s['long_term_packages'])
        
        print(f"\nCreated {len(samples)} enhanced training samples")
        print(f"  - Purchase samples: {purchase_count} ({purchase_count/len(samples)*100:.2f}%)")
        print(f"  - No-purchase samples: {no_purchase_count} ({no_purchase_count/len(samples)*100:.2f}%)")
        print(f"  - Samples with empty short-term: {empty_short_term} ({empty_short_term/len(samples)*100:.2f}%)")
        print(f"  - Samples with empty long-term: {empty_long_term} ({empty_long_term/len(samples)*100:.2f}%)")
        
        # Count event types
        event_counts = defaultdict(int)
        for sample in samples:
            for event_idx in sample.get('short_term_events', []):
                if event_idx > 0:
                    event_name = [k for k, v in self.event_to_idx.items() if v == event_idx][0]
                    event_counts[event_name] += 1
        
        print("\nEvent type distribution in short-term sequences:")
        for event_type, count in sorted(event_counts.items()):
            print(f"  - {event_type}: {count}")
        
        return samples
    
    def get_idx_mappings(self):
        """
        Get the index mappings
        """
        if not self.user_to_idx or not self.package_to_idx:
            self.create_mappings()
        
        return {
            'user_to_idx': self.user_to_idx,
            'package_to_idx': self.package_to_idx,
            'event_to_idx': self.event_to_idx
        }
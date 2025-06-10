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

class SessionProcessor:
    """
    Enhanced SessionProcessor that processes event data to create user sessions and behavioral data
    
    This processor focuses on:
    - Loading and filtering events
    - Identifying sessions based on time gaps
    - Extracting short-term and long-term behaviors with event types
    - Creating training samples with both purchase and non-purchase data
    - Incorporating more sessions for richer user profiles
    """
    def __init__(self, event_data_path=None, session_timeout_hours=12, min_interactions=10, 
                 cache_dir='data/cache', max_sessions_per_user=20, max_samples_per_user=10):
        """
        Initialize SessionProcessor
        
        Args:
            event_data_path (str): Path to the event data file (e.g., 'bookit_events_13_months.parquet' or 'bookit_events_2_months.parquet')
            session_timeout_hours (int): Timeout for defining user sessions
            min_interactions (int): Minimum number of interactions to keep a user
            cache_dir (str): Directory for caching processed data (only sessions.pkl will be cached)
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
    
    def _get_session_cache_filename(self):
        """
        Generate a unique cache filename based on the event data file
        """
        if self.event_data_path:
            # Extract just the filename without path and extension
            base_name = os.path.splitext(os.path.basename(self.event_data_path))[0]
            return f'sessions_{base_name}.pkl'
        return 'sessions.pkl'  # Default fallback
    
    def clear_cache(self):
        """
        Clear only sessions cache file
        """
        try:
            # Get the specific cache file for this dataset
            cache_file = os.path.join(self.cache_dir, self._get_session_cache_filename())
            
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"Removed sessions cache file: {cache_file}")
            else:
                print(f"No sessions cache to clear: {cache_file}")
        except Exception as e:
            print(f"Error clearing cache: {e}")
    
    def load_data(self, event_data_path=None):
        """
        Load event data directly without caching
        
        Args:
            event_data_path (str): Optional path to event data file. If not provided, uses instance event_data_path
        """
        # Allow overriding the event data path
        if event_data_path:
            self.event_data_path = event_data_path
            self.data_path = event_data_path  # Backward compatibility
            
        # Load data from source
        if self.event_data_path:
            print(f"Loading event data from {self.event_data_path}")
            
            # Check file extension
            if self.event_data_path.endswith('.parquet'):
                # Load parquet file
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
                # Use chunking to process large CSV datasets
                chunks = pd.read_csv(self.event_data_path, chunksize=1000000)
                
                # Process chunks
                dfs = []
                for chunk in chunks:
                    # Convert timestamp to datetime
                    if 'created_at' in chunk.columns:
                        chunk['created_at'] = pd.to_datetime(chunk['created_at'])
                    dfs.append(chunk)
                
                # Combine chunks
                self.df = pd.concat(dfs)
            
            # Ensure required columns exist
            required_columns = ['userId', 'main_id', 'event', 'created_at']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            if missing_columns:
                print(f"Error: Missing required columns in event data: {missing_columns}")
                return
            
            # Filter users based on interaction count
            self._filter_users()
            
            print(f"Data loading complete. Loaded {len(self.df)} events after filtering.")
        else:
            print("Error: No event data path provided")
    
    def _filter_users(self):
        """Filter users based on minimum interaction count and remove outlier sessions"""
        # Count interactions per user
        user_interaction_counts = self.df['userId'].value_counts()
        
        # Get users with at least min_interactions
        users_with_enough_interactions = user_interaction_counts[user_interaction_counts >= self.min_interactions].index.tolist()
        
        # IMPORTANT: Also keep ALL users who have made purchases, regardless of interaction count
        users_with_purchases = self.df[self.df['event'] == 'Purchase']['userId'].unique().tolist()
        
        # Combine both sets of users
        self.filtered_users = list(set(users_with_enough_interactions + users_with_purchases))
        
        # Filter dataframe
        self.df = self.df[self.df['userId'].isin(self.filtered_users)]
        
        print(f"\nFiltered to {len(self.filtered_users)} users:")
        print(f"  - {len(users_with_enough_interactions)} users with at least {self.min_interactions} interactions")
        print(f"  - {len(users_with_purchases)} users with purchases total")
    
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
        self.package_to_idx = {package: i+1 for i, package in enumerate(packages)}  # Reserve 0 for padding
        
        print(f"Created mappings for {len(self.user_to_idx)} users and {len(self.package_to_idx)} packages")
    
    def _process_user_group(self, user_group_data):
        """
        Process sessions for a single user group
        
        Args:
            user_group_data (tuple): Contains user group DataFrame and session timeout
        
        Returns:
            list: Extracted sessions for the user group
        """
        # Unpack arguments
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
                    'event_types': [],  # Store the event type for each interaction
                    'is_purchase': []
                }
            
            # Update session details
            current_session['end_time'] = timestamp
            last_timestamp = timestamp
            
            # Add product interaction
            current_session['package_ids'].append(product_id)
            current_session['event_types'].append(event_type)
            # Only mark Purchase events as purchases
            current_session['is_purchase'].append(1 if event_type == 'Purchase' else 0)
        
        # Add final session
        if current_session and current_session['package_ids']:
            sessions.append(current_session)
        
        return sessions

    def extract_sessions(self, use_cache=True):
        """
        Extract user sessions from event data
        
        Args:
            use_cache (bool): Whether to use cached sessions if available
        
        Returns:
            list: Extracted sessions
        """
        # Generate dataset-specific cache file path
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
        
        # Start extraction
        print("Starting session extraction...")
        start_time = time.time()

        # Sort dataframe
        df_sorted = self.df.sort_values(['userId', 'created_at']).reset_index(drop=True)

        # Parallel processing preparation
        print("Preparing for parallel processing...")
        
        # Use all available CPU cores
        num_cores = multiprocessing.cpu_count()
        print(f"Using {num_cores} CPU cores")

        # Prepare user groups with session timeout
        user_groups_with_timeout = [
            (group, self.session_timeout_hours) 
            for _, group in df_sorted.groupby('userId')
        ]
        
        # Use multiprocessing Pool
        with multiprocessing.Pool(processes=num_cores) as pool:
            # Map user groups to processing function
            all_sessions = pool.map(self._process_user_group, user_groups_with_timeout)
        
        # Flatten sessions
        sessions = [session for user_sessions in all_sessions for session in user_sessions]

        # Performance logging
        total_duration = time.time() - start_time
        print(f"Total session extraction time: {total_duration:.2f} seconds")
        print(f"Total sessions extracted: {len(sessions)}")
        
        # Display session length statistics
        session_lengths = [len(session['package_ids']) for session in sessions]
        print(f"Session length statistics before filtering:")
        print(f"Average session length: {np.mean(session_lengths):.2f}")
        print(f"Median session length: {np.median(session_lengths):.2f}")
        print(f"Max session length: {np.max(session_lengths)}")
        
        # Remove the upper 0.1% sessions by length, BUT preserve all sessions with purchases
        length_threshold = np.percentile(session_lengths, 99.99)
        
        # Count purchases before filtering
        purchases_before = sum(sum(session['is_purchase']) for session in sessions)
        
        # Filter sessions but ALWAYS keep sessions with purchases
        filtered_sessions = []
        removed_count = 0
        removed_with_purchases = 0
        
        for session in sessions:
            session_length = len(session['package_ids'])
            has_purchase = any(session['is_purchase'])
            
            # Keep session if it's below threshold OR has a purchase
            if session_length <= length_threshold or has_purchase:
                filtered_sessions.append(session)
            else:
                removed_count += 1
                if has_purchase:
                    removed_with_purchases += 1
        
        # Count purchases after filtering
        purchases_after = sum(sum(session['is_purchase']) for session in filtered_sessions)
        
        print(f"Removed {removed_count} sessions ({removed_count/len(sessions)*100:.2f}%) with length > {length_threshold:.0f}")
        print(f"  - {removed_with_purchases} of removed sessions had purchases (these were preserved)")
        print(f"Remaining sessions: {len(filtered_sessions)}")
        print(f"Purchase events: {purchases_before} -> {purchases_after} (preserved {purchases_after/purchases_before*100:.1f}%)")
        
        # Save to cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(filtered_sessions, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Sessions saved to cache: {os.path.basename(cache_file)}")
        except Exception as e:
            print(f"Error saving sessions to cache: {e}")
        
        # Store sessions in instance variable
        self.sessions = filtered_sessions
        
        return filtered_sessions
    
    def prepare_enhanced_training_data(self):
        """
        Prepare enhanced training data from sessions including event types
        
        This improved version:
        1. Creates samples from all sessions, not just those with purchases
        2. Tracks sessions without purchases as negative examples
        3. Incorporates more historical context
        4. Includes event types for all interactions
        
        Returns:
            list: Enhanced training samples
        """
        
        if not self.sessions:
            self.extract_sessions(use_cache=True)  # Sessions still use cache
        
        print("Preparing enhanced training data with event types...")
        
        # Create samples with short-term and long-term behaviors
        samples = []
        
        # Group sessions by user
        user_sessions = defaultdict(list)
        for session in self.sessions:
            user_sessions[session['user_id']].append(session)
        
        # Process users
        total_users = len(user_sessions)
        print(f"Processing {total_users} users...")
        
        for user_id, sessions in tqdm(user_sessions.items(), desc="Preparing enhanced training data"):
            # Sort sessions by time
            sessions.sort(key=lambda x: x['start_time'])
            
            # Limit sessions but prioritize those with purchases
            if len(sessions) > self.max_sessions_per_user:
                # Separate sessions with and without purchases
                sessions_with_purchases = [s for s in sessions if any(s['is_purchase'])]
                sessions_without_purchases = [s for s in sessions if not any(s['is_purchase'])]
                
                # Take all sessions with purchases (up to limit)
                recent_sessions = sessions_with_purchases[-self.max_sessions_per_user:]
                
                # Fill remaining slots with non-purchase sessions
                remaining_slots = self.max_sessions_per_user - len(recent_sessions)
                if remaining_slots > 0:
                    recent_sessions = sessions_without_purchases[-remaining_slots:] + recent_sessions
                    # Re-sort by time to maintain chronological order
                    recent_sessions.sort(key=lambda x: x['start_time'])
            else:
                recent_sessions = sessions
            
            # Track all samples created for this user
            user_samples = []
            
            # Iterate through sessions to create samples
            for i, session in enumerate(recent_sessions):
                # Find purchase events in the session
                purchase_indices = [j for j, is_purchase in enumerate(session['is_purchase']) if is_purchase]
                
                # Create positive samples (with purchases)
                for purchase_idx in purchase_indices:
                    # Short-term: packages and events in current session before purchase
                    short_term_packages = []
                    short_term_events = []
                    
                    for j in range(purchase_idx):
                        # Include all event types, not just views
                        short_term_packages.append(session['package_ids'][j])
                        event_type = session['event_types'][j]
                        short_term_events.append(self.event_to_idx.get(event_type, 0))
                    
                    # Long-term: packages and events from previous sessions
                    long_term_packages = []
                    long_term_events = []
                    
                    for prev_session in recent_sessions[:i]:
                        for j, package_id in enumerate(prev_session['package_ids']):
                            long_term_packages.append(package_id)
                            event_type = prev_session['event_types'][j]
                            long_term_events.append(self.event_to_idx.get(event_type, 0))
                    
                    # Purchased package
                    purchased_package = session['package_ids'][purchase_idx]
                    
                    # Create sample
                    sample = {
                        'user_id': user_id,
                        'short_term_packages': short_term_packages,
                        'short_term_events': short_term_events,  # Add event types
                        'long_term_packages': long_term_packages,
                        'long_term_events': long_term_events,  # Add event types
                        'purchased_package': purchased_package,
                        'is_purchase': True,  # Flag to indicate real purchase
                        'session_id': i,  # Track session index
                        'timestamp': session['end_time']  # Track timestamp
                    }
                    user_samples.append(sample)
                
                # If no purchases in this session, create a "no purchase" sample
                if len(purchase_indices) == 0 and len(session['package_ids']) > 0:
                    # Short-term: packages and events in current session
                    short_term_packages = []
                    short_term_events = []
                    
                    for j in range(len(session['package_ids'])):
                        short_term_packages.append(session['package_ids'][j])
                        event_type = session['event_types'][j]
                        short_term_events.append(self.event_to_idx.get(event_type, 0))
                    
                    # Long-term: packages and events from previous sessions
                    long_term_packages = []
                    long_term_events = []
                    
                    for prev_session in recent_sessions[:i]:
                        for j, package_id in enumerate(prev_session['package_ids']):
                            long_term_packages.append(package_id)
                            event_type = prev_session['event_types'][j]
                            long_term_events.append(self.event_to_idx.get(event_type, 0))
                    
                    # Last viewed package (not purchased)
                    last_viewed = session['package_ids'][-1]
                    
                    # Create sample
                    sample = {
                        'user_id': user_id,
                        'short_term_packages': short_term_packages,
                        'short_term_events': short_term_events,  # Add event types
                        'long_term_packages': long_term_packages,
                        'long_term_events': long_term_events,  # Add event types
                        'purchased_package': last_viewed,  # Use last viewed as reference
                        'is_purchase': False,  # Flag to indicate no purchase
                        'session_id': i,  # Track session index
                        'timestamp': session['end_time']  # Track timestamp
                    }
                    user_samples.append(sample)
            
            # Limit samples per user and add to overall samples
            if user_samples:
                # Sort by timestamp to get the most recent ones
                user_samples.sort(key=lambda x: x['timestamp'], reverse=True)
                
                # Separate purchase and non-purchase samples
                purchase_samples_user = [s for s in user_samples if s['is_purchase']]
                no_purchase_samples_user = [s for s in user_samples if not s['is_purchase']]
                
                # Always keep all purchase samples, limit non-purchase samples
                limited_samples = purchase_samples_user  # Keep ALL purchases
                
                # Add non-purchase samples up to the limit
                remaining_slots = max(0, self.max_samples_per_user - len(purchase_samples_user))
                limited_samples.extend(no_purchase_samples_user[:remaining_slots])
                
                samples.extend(limited_samples)
        
        # Track statistics
        purchase_count = sum(1 for s in samples if s['is_purchase'])
        no_purchase_count = len(samples) - purchase_count
        empty_short_term = sum(1 for s in samples if not s['short_term_packages'])
        empty_long_term = sum(1 for s in samples if not s['long_term_packages'])
        
        print(f"\nCreated {len(samples)} enhanced training samples")
        print(f"  - Purchase samples: {purchase_count} ({purchase_count/len(samples)*100:.2f}%)")
        print(f"  - No-purchase samples: {no_purchase_count} ({no_purchase_count/len(samples)*100:.2f}%)")
        print(f"  - Samples with empty short-term: {empty_short_term} ({empty_short_term/len(samples)*100:.2f}%)")
        print(f"  - Samples with empty long-term: {empty_long_term} ({empty_long_term/len(samples)*100:.2f}%)")
        
        # Additional debugging for purchase tracking
        if purchase_count < 1000:  # Alert if suspiciously low
            print(f"\nWARNING: Only {purchase_count} purchase samples created from sessions!")
            # Count total purchase events in original sessions
            total_purchase_events = 0
            for session in self.sessions:
                total_purchase_events += sum(session['is_purchase'])
            print(f"Total purchase events in sessions: {total_purchase_events}")
            if total_purchase_events > purchase_count:
                print(f"Lost {total_purchase_events - purchase_count} purchase events during sample creation")
        
        # Count event types in samples
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
        
        Returns:
            dict: Dictionary with user_to_idx, package_to_idx, and event_to_idx mappings
        """
        if not self.user_to_idx or not self.package_to_idx:
            self.create_mappings()
        
        return {
            'user_to_idx': self.user_to_idx,
            'package_to_idx': self.package_to_idx,
            'event_to_idx': self.event_to_idx
        }
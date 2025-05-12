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
    def __init__(self, data_path=None, session_timeout_hours=24, min_interactions=5, 
                 cache_dir='data/cache', max_sessions_per_user=20, max_samples_per_user=10):
        """
        Initialize SessionProcessor
        
        Args:
            data_path (str): Path to the event data file
            session_timeout_hours (int): Timeout for defining user sessions
            min_interactions (int): Minimum number of interactions to keep a user
            cache_dir (str): Directory for caching processed data
            max_sessions_per_user (int): Maximum number of sessions to consider per user
            max_samples_per_user (int): Maximum number of training samples to create per user
        """
        self.data_path = data_path
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
    
    def clear_cache(self):
        """
        Clear all cached files in the cache directory
        
        Helps prevent using stale or incorrect cached data
        """
        try:
            # Remove cache files
            cache_files = [
                os.path.join(self.cache_dir, 'preprocessed_data.pkl'),
                os.path.join(self.cache_dir, 'sessions.pkl'),
                os.path.join(self.cache_dir, 'user_package_mappings.pkl'),
                os.path.join(self.cache_dir, 'training_samples.pkl'),
                os.path.join(self.cache_dir, 'enhanced_training_samples.pkl')
            ]
            
            for file in cache_files:
                if os.path.exists(file):
                    os.remove(file)
                    print(f"Removed cache file: {file}")
            
            print(f"Cleared session processing cache")
        except Exception as e:
            print(f"Error clearing cache: {e}")
    
    def load_data(self, use_cache=True):
        """
        Load event data with caching for efficiency
        
        Args:
            use_cache (bool): Whether to use cached data if available
        """
        cache_file = os.path.join(self.cache_dir, 'preprocessed_data.pkl')
        
        # Check existing cache and clear if data path has changed
        if use_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    cached_data_path = cache_data.get('data_path')
                    
                    # If data path has changed, clear the cache
                    if cached_data_path != self.data_path:
                        print(f"Dataset path changed. Clearing previous cache.")
                        self.clear_cache()
                        # Fall through to load new data
                    else:
                        # Load cached data if path matches
                        self.df = cache_data['df']
                        self.filtered_users = cache_data.get('filtered_users')
                        
                        print(f"Loaded {len(self.df)} events from cache")
                        return
            except (pickle.UnpicklingError, KeyError, EOFError) as e:
                print(f"Cache loading error: {e}. Clearing cache and reloading.")
                self.clear_cache()
        
        # Load data from source
        if self.data_path:
            print(f"Loading event data from {self.data_path}")
            
            # Check file extension
            if self.data_path.endswith('.parquet'):
                # Load parquet file
                try:
                    self.df = pd.read_parquet(self.data_path)
                    
                    # Convert timestamp to datetime if needed
                    if 'created_at' in self.df.columns:
                        self.df['created_at'] = pd.to_datetime(self.df['created_at'])
                    
                    print(f"Loaded {len(self.df)} events from parquet file")
                except Exception as e:
                    print(f"Error loading parquet file: {e}")
                    return
            else:
                # Use chunking to process large CSV datasets
                chunks = pd.read_csv(self.data_path, chunksize=1000000)
                
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
            
            # Save to cache
            cache_data = {
                'df': self.df,
                'filtered_users': self.filtered_users,
                'data_path': self.data_path  # Store the current data path
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"Saved processed data to cache: {cache_file}")
    
    def _filter_users(self):
        """Filter users based on minimum interaction count and remove outlier sessions"""
        # Count interactions per user
        user_interaction_counts = self.df['userId'].value_counts()
        
        # Get users with at least min_interactions
        self.filtered_users = user_interaction_counts[user_interaction_counts >= self.min_interactions].index.tolist()
        
        # Filter dataframe
        self.df = self.df[self.df['userId'].isin(self.filtered_users)]
        
        print(f"Filtered to {len(self.filtered_users)} users with at least {self.min_interactions} interactions")
        print(f"Remaining events: {len(self.df)}")
    
    def create_mappings(self, use_cache=True):
        """Create mappings from user and package IDs to indices"""
        cache_file = os.path.join(self.cache_dir, 'user_package_mappings.pkl')
        
        if use_cache and os.path.exists(cache_file):
            print(f"Loading user and package mappings from cache")
            with open(cache_file, 'rb') as f:
                mappings = pickle.load(f)
                self.user_to_idx = mappings['user_to_idx']
                self.package_to_idx = mappings['package_to_idx']
                # Also load event mappings if available
                if 'event_to_idx' in mappings:
                    self.event_to_idx = mappings['event_to_idx']
            
            # Print mapping statistics
            print(f"Loaded mappings for {len(self.user_to_idx)} users and {len(self.package_to_idx)} packages")
            return
        
        print("Creating user and package mappings...")
        
        # Create user mapping
        users = self.df['userId'].unique()
        self.user_to_idx = {user: i for i, user in enumerate(users)}
        
        # Create package mapping
        packages = self.df['main_id'].unique()
        self.package_to_idx = {package: i+1 for i, package in enumerate(packages)}  # Reserve 0 for padding
        
        print(f"Created mappings for {len(self.user_to_idx)} users and {len(self.package_to_idx)} packages")
        
        # Save mappings to cache
        mappings = {
            'user_to_idx': self.user_to_idx,
            'package_to_idx': self.package_to_idx,
            'event_to_idx': self.event_to_idx
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(mappings, f)
        
        print(f"Saved mappings to cache: {cache_file}")
    
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
        # Cache file path
        cache_file = os.path.join(self.cache_dir, 'sessions.pkl')

        # Check cache
        if use_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    loaded_sessions = pickle.load(f)
                
                    if loaded_sessions and isinstance(loaded_sessions, list):
                        print(f"Loaded {len(loaded_sessions)} sessions from cache")
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
        
        # Remove the upper 0.1% sessions by length
        length_threshold = np.percentile(session_lengths, 99.9)
        filtered_sessions = [session for session in sessions if len(session['package_ids']) <= length_threshold]
        
        removed_count = len(sessions) - len(filtered_sessions)
        print(f"Removed {removed_count} sessions ({removed_count/len(sessions)*100:.2f}%) with length > {length_threshold:.0f}")
        print(f"Remaining sessions: {len(filtered_sessions)}")
        
        # Save to cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(filtered_sessions, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Sessions saved to cache: {cache_file}")
        except Exception as e:
            print(f"Error saving sessions to cache: {e}")
        
        # Store sessions in instance variable
        self.sessions = filtered_sessions
        
        return filtered_sessions
    
    def prepare_enhanced_training_data(self, use_cache=True):
        """
        Prepare enhanced training data from sessions including event types
        
        This improved version:
        1. Creates samples from all sessions, not just those with purchases
        2. Tracks sessions without purchases as negative examples
        3. Incorporates more historical context
        4. Includes event types for all interactions
        
        Args:
            use_cache (bool): Whether to use cached training data
            
        Returns:
            list: Enhanced training samples
        """
        cache_file = os.path.join(self.cache_dir, 'enhanced_training_samples.pkl')
        
        if use_cache and os.path.exists(cache_file):
            print(f"Loading enhanced training samples from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                samples = pickle.load(f)
            
            # Check if cached samples have event types
            if samples and 'short_term_events' not in samples[0]:
                print("Cached samples don't have event types. Regenerating...")
                use_cache = False
            else:
                print(f"Loaded {len(samples)} enhanced training samples from cache")
                return samples
        
        if not self.sessions:
            self.extract_sessions(use_cache=use_cache)
        
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
            
            # Limit to the most recent max_sessions_per_user sessions
            recent_sessions = sessions[-self.max_sessions_per_user:] if len(sessions) > self.max_sessions_per_user else sessions
            
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
                # Limit samples per user
                limited_samples = user_samples[:self.max_samples_per_user]
                samples.extend(limited_samples)
        
        # Track statistics
        purchase_count = sum(1 for s in samples if s['is_purchase'])
        no_purchase_count = len(samples) - purchase_count
        empty_short_term = sum(1 for s in samples if not s['short_term_packages'])
        empty_long_term = sum(1 for s in samples if not s['long_term_packages'])
        
        print(f"Created {len(samples)} enhanced training samples")
        print(f"  - Purchase samples: {purchase_count} ({purchase_count/len(samples)*100:.2f}%)")
        print(f"  - No-purchase samples: {no_purchase_count} ({no_purchase_count/len(samples)*100:.2f}%)")
        print(f"  - Samples with empty short-term: {empty_short_term} ({empty_short_term/len(samples)*100:.2f}%)")
        print(f"  - Samples with empty long-term: {empty_long_term} ({empty_long_term/len(samples)*100:.2f}%)")
        
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
        
        # Save samples to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(samples, f)
        
        print(f"Saved enhanced training samples to cache: {cache_file}")
        
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
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import pickle
import time
from tqdm import tqdm

class PackageProcessor:
    """
    Processes package metadata and provides efficient data loading facilities
    """
    def __init__(self, data_path=None, cache_dir='data/cache'):
        """
        Initialize PackageProcessor
        
        Args:
            data_path (str): Path to the package data file
            cache_dir (str): Directory for caching processed data
        """
        self.data_path = data_path
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize dictionaries for mapping IDs
        self.country_to_idx = {}
        self.category_to_idx = {}
        self.theme_to_idx = {}
        
        # Package metadata
        self.package_metadata = {}
        self.df = None
    
    def clear_cache(self):
        """
        Clear all cached files in the cache directory related to package data
        """
        try:
            # Remove cache files
            cache_files = [
                os.path.join(self.cache_dir, 'package_metadata.pkl'),
                os.path.join(self.cache_dir, 'package_mappings.pkl')
            ]
            
            for file in cache_files:
                if os.path.exists(file):
                    os.remove(file)
                    print(f"Removed cache file: {file}")
            
            print(f"Cleared package data cache")
        except Exception as e:
            print(f"Error clearing cache: {e}")
    
    def load_data(self, use_cache=True):
        """
        Load package data with caching for efficiency
        
        Args:
            use_cache (bool): Whether to use cached data if available
        """
        cache_file = os.path.join(self.cache_dir, 'package_metadata.pkl')
        
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
                        self.package_metadata = cache_data['package_metadata']
                        self.df = cache_data.get('df')
                        
                        print(f"Loaded metadata for {len(self.package_metadata)} packages from cache")
                        return
            except (pickle.UnpicklingError, KeyError, EOFError) as e:
                print(f"Cache loading error: {e}. Clearing cache and reloading.")
                self.clear_cache()
        
        # Load data from source
        if self.data_path:
            print(f"Loading package data from {self.data_path}")
            
            # Check file extension
            if self.data_path.endswith('.parquet'):
                # Load parquet file
                try:
                    self.df = pd.read_parquet(self.data_path)
                    print(f"Loaded {len(self.df)} packages from parquet file")
                except Exception as e:
                    print(f"Error loading parquet file: {e}")
                    return
            else:
                # Load CSV
                try:
                    self.df = pd.read_csv(self.data_path)
                    print(f"Loaded {len(self.df)} packages from CSV file")
                except Exception as e:
                    print(f"Error loading CSV file: {e}")
                    return
            
            # Ensure required columns exist
            required_columns = ['main_id']
            optional_columns = ['title', 'country', 'category', 'theme', 'min_price']
            
            missing_required = [col for col in required_columns if col not in self.df.columns]
            if missing_required:
                print(f"Error: Missing required columns in package data: {missing_required}")
                return
            
            missing_optional = [col for col in optional_columns if col not in self.df.columns]
            if missing_optional:
                print(f"Warning: Missing optional columns in package data: {missing_optional}")
            
            # Process package metadata
            self._process_package_metadata()
            
            # Save to cache
            cache_data = {
                'package_metadata': self.package_metadata,
                'df': self.df,
                'data_path': self.data_path  # Store the current data path
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"Saved processed package data to cache: {cache_file}")
    
    def _process_package_metadata(self):
        """Process package metadata from the dataframe"""
        print("Processing package metadata...")
        
        # Reset package metadata
        self.package_metadata = {}
        
        # Process each row in the dataframe
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing package metadata"):
            # Get main_id as string
            main_id = str(row['main_id'])
            
            # Store metadata
            self.package_metadata[main_id] = {
                'title': row.get('title', ''),
                'country': row.get('country', ''),
                'category': row.get('category', ''),
                'theme': row.get('theme', ''),
                'min_price': float(row.get('min_price', 0))
            }
        
        print(f"Processed metadata for {len(self.package_metadata)} packages")
    
    def create_mappings(self, use_cache=True):
        """Create mappings from package attributes to indices"""
        cache_file = os.path.join(self.cache_dir, 'package_mappings.pkl')
        
        if use_cache and os.path.exists(cache_file):
            print(f"Loading package attribute mappings from cache")
            with open(cache_file, 'rb') as f:
                mappings = pickle.load(f)
                self.country_to_idx = mappings['country_to_idx']
                self.category_to_idx = mappings['category_to_idx']
                self.theme_to_idx = mappings['theme_to_idx']
            
            # Print mapping statistics
            print(f"Loaded mappings for {len(self.country_to_idx)} countries, "
                  f"{len(self.category_to_idx)} categories, and "
                  f"{len(self.theme_to_idx)} themes")
            return
        
        print("Creating package attribute mappings...")
        
        if not self.package_metadata:
            self.load_data(use_cache=use_cache)
        
        # Extract unique values for each attribute
        countries = set()
        categories = set()
        themes = set()
        
        for metadata in self.package_metadata.values():
            if metadata.get('country'):
                countries.add(metadata['country'])
            if metadata.get('category'):
                categories.add(metadata['category'])
            if metadata.get('theme'):
                themes.add(metadata['theme'])
        
        # Create mappings (reserve index 0 for padding/unknown)
        self.country_to_idx = {country: i+1 for i, country in enumerate(sorted(countries))}
        self.category_to_idx = {category: i+1 for i, category in enumerate(sorted(categories))}
        self.theme_to_idx = {theme: i+1 for i, theme in enumerate(sorted(themes))}
        
        # Add 'Unknown' entries if they don't exist
        if 'Unknown' not in self.country_to_idx:
            self.country_to_idx['Unknown'] = len(self.country_to_idx) + 1
        if 'Unknown' not in self.category_to_idx:
            self.category_to_idx['Unknown'] = len(self.category_to_idx) + 1
        if 'Unknown' not in self.theme_to_idx:
            self.theme_to_idx['Unknown'] = len(self.theme_to_idx) + 1
        
        # Print mapping statistics
        print(f"Created mappings for {len(self.country_to_idx)} countries, "
              f"{len(self.category_to_idx)} categories, and "
              f"{len(self.theme_to_idx)} themes")
        
        # Save mappings to cache
        mappings = {
            'country_to_idx': self.country_to_idx,
            'category_to_idx': self.category_to_idx,
            'theme_to_idx': self.theme_to_idx
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(mappings, f)
        
        print(f"Saved package attribute mappings to cache: {cache_file}")
    
    def get_package_metadata(self):
        """
        Get package metadata
        
        Returns:
            dict: Dictionary mapping package ID to metadata
        """
        if not self.package_metadata:
            self.load_data()
        
        return self.package_metadata
    
    def get_idx_mappings(self):
        """
        Get the index mappings
        
        Returns:
            dict: Dictionary with country_to_idx, category_to_idx, and theme_to_idx mappings
        """
        if not self.country_to_idx or not self.category_to_idx or not self.theme_to_idx:
            self.create_mappings()
        
        return {
            'country_to_idx': self.country_to_idx,
            'category_to_idx': self.category_to_idx,
            'theme_to_idx': self.theme_to_idx
        }


import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Any
from tqdm import tqdm

class TravelPackageDataset(Dataset):
    """
    Enhanced Dataset for travel package recommendation
    
    Features:
    - Handles empty short-term packages with special tokens
    - Uses -1 for empty/unknown values across all fields
    - Includes event type information
    - Processes is_purchase flags for non-purchase samples
    - Preserves original user IDs for debugging
    - Includes session_id and timestamp information
    """
    def __init__(self, 
                samples: List[Dict[str, Any]], 
                package_metadata: Dict[str, Dict[str, Any]],
                user_to_idx: Dict[Any, int],
                package_to_idx: Dict[str, int],
                country_to_idx: Dict[str, int],
                category_to_idx: Dict[str, int],
                theme_to_idx: Dict[str, int],
                event_to_idx: Dict[str, int] = None,
                max_short_term: int = 5,
                max_long_term: int = 10,
                empty_token: int = -1,  # Special token for empty values
                unknown_token: int = -1):  # Token for unknown categories
        """
        Initialize the enhanced dataset
        
        Args:
            samples: List of training/testing samples
            package_metadata: Dictionary mapping package ID to its metadata
            user_to_idx: Mapping from user ID to index
            package_to_idx: Mapping from package ID to index
            country_to_idx: Mapping from country to index
            category_to_idx: Mapping from category to index
            theme_to_idx: Mapping from theme to index
            event_to_idx: Mapping from event type to index
            max_short_term: Maximum length of short-term sequence
            max_long_term: Maximum length of long-term sequence
            empty_token: Special token for empty values
            unknown_token: Token for unknown category values
        """
        self.samples = samples
        self.package_metadata = package_metadata
        self.user_to_idx = user_to_idx
        self.package_to_idx = package_to_idx
        self.country_to_idx = country_to_idx
        self.category_to_idx = category_to_idx
        self.theme_to_idx = theme_to_idx
        self.event_to_idx = event_to_idx or {'ViewContent': 1, 'AddToCart': 2, 'InitiateCheckout': 3, 'Purchase': 4}
        self.max_short_term = max_short_term
        self.max_long_term = max_long_term
        self.empty_token = empty_token
        self.unknown_token = unknown_token
        
        # Store original sample information for reference
        self.original_user_ids = []
        self.has_short_term = []
        self.is_purchase = []
        self.session_ids = []
        
        # Pre-process samples for faster retrieval
        self._prepare_samples()
    
    def __len__(self):
        """Return the size of the dataset"""
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        sample = {
            'user_id': self.user_ids[idx],
            'original_user_id': self.original_user_ids[idx] if self.original_user_ids else None,
            'has_short_term': self.has_short_term[idx],
            
            'short_term': {
                'package_ids': self.short_term_packages[idx],
                'country_ids': self.short_term_countries[idx],
                'category_ids': self.short_term_categories[idx],
                'theme_ids': self.short_term_themes[idx],
                'price_values': self.short_term_prices[idx],
                'event_types': self.short_term_events[idx]  # Add event types
            },
            
            'long_term': {
                'package_ids': self.long_term_packages[idx],
                'country_ids': self.long_term_countries[idx],
                'category_ids': self.long_term_categories[idx],
                'theme_ids': self.long_term_themes[idx],
                'price_values': self.long_term_prices[idx],
                'event_types': self.long_term_events[idx]  # Add event types
            },
            
            'purchased': {
                'package_ids': self.purchased_packages[idx],
                'country_ids': self.purchased_countries[idx],
                'category_ids': self.purchased_categories[idx],
                'theme_ids': self.purchased_themes[idx],
                'price_values': self.purchased_prices[idx]
            }
        }
        
        # Add optional fields only if they exist as tensors
        if isinstance(self.is_purchase, torch.Tensor):
            sample['is_purchase'] = self.is_purchase[idx]
        
        if isinstance(self.session_ids, torch.Tensor):
            sample['session_id'] = self.session_ids[idx]
        
        return sample
    
    def _prepare_samples(self):
        """Pre-process samples for faster retrieval"""
        print("Pre-processing samples for dataset...")
        
        # Initialize tensors for samples
        self.user_ids = []
        self.original_user_ids = []
        self.has_short_term = []
        self.is_purchase = []
        self.session_ids = []
        
        # Initialize tensors for package data
        self.short_term_packages = []
        self.short_term_countries = []
        self.short_term_categories = []
        self.short_term_themes = []
        self.short_term_prices = []
        self.short_term_events = []  # Add event types
        
        self.long_term_packages = []
        self.long_term_countries = []
        self.long_term_categories = []
        self.long_term_themes = []
        self.long_term_prices = []
        self.long_term_events = []  # Add event types
        
        self.purchased_packages = []
        self.purchased_countries = []
        self.purchased_categories = []
        self.purchased_themes = []
        self.purchased_prices = []
        
        # Track statistics
        empty_short_term_count = 0
        empty_long_term_count = 0
        
        # Process each sample
        for sample in tqdm(self.samples, desc="Preparing dataset samples"):
            # Extract and store metadata
            user_id = sample['user_id']
            self.original_user_ids.append(user_id)
            
            # Store additional fields if available
            if 'is_purchase' in sample:
                self.is_purchase.append(sample['is_purchase'])
            
            if 'session_id' in sample:
                self.session_ids.append(sample['session_id'])
            
            # Map user ID to index
            user_idx = 1  # Default to 1 (not 0) for padding
            if user_id in self.user_to_idx:
                user_idx = self.user_to_idx[user_id]
            else:
                # Try matching by prefix
                user_id_str = str(user_id)
                for full_id in self.user_to_idx.keys():
                    if isinstance(full_id, str) and full_id.startswith(user_id_str + " "):
                        user_idx = self.user_to_idx[full_id]
                        break
            
            self.user_ids.append(user_idx)
            
            # =========== Process Short-Term Packages ===========
            # Check if short-term packages exist and not empty
            has_short_term = len(sample['short_term_packages']) > 0
            self.has_short_term.append(has_short_term)
            
            if not has_short_term:
                # For empty short-term packages, use special token
                empty_short_term_count += 1
                
                # Use special token as the FIRST token, with rest being padding
                st_pkgs = [self.empty_token] + [0] * (self.max_short_term - 1)
                st_countries = [self.unknown_token] + [0] * (self.max_short_term - 1)
                st_categories = [self.unknown_token] + [0] * (self.max_short_term - 1)
                st_themes = [self.unknown_token] + [0] * (self.max_short_term - 1)
                st_prices = [0.0] * self.max_short_term
                st_events = [0] * self.max_short_term  # No events for empty sequences
            else:
                # Handle non-empty short-term packages
                short_term_pkgs = sample['short_term_packages'][-self.max_short_term:] if len(sample['short_term_packages']) > self.max_short_term else sample['short_term_packages']
                short_term_events = sample.get('short_term_events', [])[-self.max_short_term:] if len(sample.get('short_term_events', [])) > self.max_short_term else sample.get('short_term_events', [])
                short_term_len = len(short_term_pkgs)
                
                # Process package attributes and events
                st_pkgs = []
                st_countries = []
                st_categories = []
                st_themes = []
                st_prices = []
                st_events = []
                
                for i, pkg_id in enumerate(short_term_pkgs):
                    # Convert to string for consistent lookup
                    pkg_str = str(pkg_id)
                    pkg_idx = self.package_to_idx.get(pkg_str, 1)
                    st_pkgs.append(pkg_idx)
                    
                    # Get metadata
                    meta = self.package_metadata.get(pkg_str, {})
                    
                    # Get attributes with unknown token for missing values
                    country = meta.get('country', 'Unknown')
                    country_idx = self.country_to_idx.get(country, self.unknown_token)
                    st_countries.append(country_idx)
                    
                    category = meta.get('category', 'Unknown')
                    category_idx = self.category_to_idx.get(category, self.unknown_token)
                    st_categories.append(category_idx)
                    
                    theme = meta.get('theme', 'Unknown')
                    theme_idx = self.theme_to_idx.get(theme, self.unknown_token)
                    st_themes.append(theme_idx)
                    
                    price = float(meta.get('min_price', 0))
                    st_prices.append(price)
                    
                    # Get event type
                    event_idx = short_term_events[i] if i < len(short_term_events) else 0
                    st_events.append(event_idx)
                
                # Pad short-term sequences
                st_pkgs = st_pkgs + [0] * (self.max_short_term - short_term_len)
                st_countries = st_countries + [0] * (self.max_short_term - short_term_len)
                st_categories = st_categories + [0] * (self.max_short_term - short_term_len)
                st_themes = st_themes + [0] * (self.max_short_term - short_term_len)
                st_prices = st_prices + [0.0] * (self.max_short_term - short_term_len)
                st_events = st_events + [0] * (self.max_short_term - short_term_len)
            
            # Store short-term data
            self.short_term_packages.append(st_pkgs)
            self.short_term_countries.append(st_countries)
            self.short_term_categories.append(st_categories)
            self.short_term_themes.append(st_themes)
            self.short_term_prices.append(st_prices)
            self.short_term_events.append(st_events)
            
            # =========== Process Long-Term Packages ===========
            # Check if long-term packages exist and not empty
            has_long_term = 'long_term_packages' in sample and len(sample['long_term_packages']) > 0
            
            if not has_long_term:
                # For empty long-term packages, use special token
                empty_long_term_count += 1
                
                # Use special token as the FIRST token, with rest being padding
                lt_pkgs = [self.empty_token] + [0] * (self.max_long_term - 1)
                lt_countries = [self.unknown_token] + [0] * (self.max_long_term - 1)
                lt_categories = [self.unknown_token] + [0] * (self.max_long_term - 1)
                lt_themes = [self.unknown_token] + [0] * (self.max_long_term - 1)
                lt_prices = [0.0] * self.max_long_term
                lt_events = [0] * self.max_long_term  # No events for empty sequences
            else:
                # Get long-term packages with truncation if needed
                long_term_pkgs = sample['long_term_packages'][-self.max_long_term:] if len(sample['long_term_packages']) > self.max_long_term else sample['long_term_packages']
                long_term_events = sample.get('long_term_events', [])[-self.max_long_term:] if len(sample.get('long_term_events', [])) > self.max_long_term else sample.get('long_term_events', [])
                long_term_len = len(long_term_pkgs)
                
                # Process package attributes and events
                lt_pkgs = []
                lt_countries = []
                lt_categories = []
                lt_themes = []
                lt_prices = []
                lt_events = []
                
                for i, pkg_id in enumerate(long_term_pkgs):
                    # Convert to string for consistent lookup
                    pkg_str = str(pkg_id)
                    pkg_idx = self.package_to_idx.get(pkg_str, 1)
                    lt_pkgs.append(pkg_idx)
                    
                    # Get metadata
                    meta = self.package_metadata.get(pkg_str, {})
                    
                    # Get attributes with unknown token for missing values
                    country = meta.get('country', 'Unknown')
                    country_idx = self.country_to_idx.get(country, self.unknown_token)
                    lt_countries.append(country_idx)
                    
                    category = meta.get('category', 'Unknown')
                    category_idx = self.category_to_idx.get(category, self.unknown_token)
                    lt_categories.append(category_idx)
                    
                    theme = meta.get('theme', 'Unknown')
                    theme_idx = self.theme_to_idx.get(theme, self.unknown_token)
                    lt_themes.append(theme_idx)
                    
                    price = float(meta.get('min_price', 0))
                    lt_prices.append(price)
                    
                    # Get event type
                    event_idx = long_term_events[i] if i < len(long_term_events) else 0
                    lt_events.append(event_idx)
                
                # Pad long-term sequences
                lt_pkgs = lt_pkgs + [0] * (self.max_long_term - long_term_len)
                lt_countries = lt_countries + [0] * (self.max_long_term - long_term_len)
                lt_categories = lt_categories + [0] * (self.max_long_term - long_term_len)
                lt_themes = lt_themes + [0] * (self.max_long_term - long_term_len)
                lt_prices = lt_prices + [0.0] * (self.max_long_term - long_term_len)
                lt_events = lt_events + [0] * (self.max_long_term - long_term_len)
            
            # Store long-term data
            self.long_term_packages.append(lt_pkgs)
            self.long_term_countries.append(lt_countries)
            self.long_term_categories.append(lt_categories)
            self.long_term_themes.append(lt_themes)
            self.long_term_prices.append(lt_prices)
            self.long_term_events.append(lt_events)
            
            # =========== Process Purchased Package ===========
            purchased_pkg_id = str(sample['purchased_package'])
            
            # Get package index
            purchased_pkg_idx = self.package_to_idx.get(purchased_pkg_id, 1)
            
            # Get metadata for purchased package
            meta = self.package_metadata.get(purchased_pkg_id, {})
            
            # Get attributes for purchased package with unknown token
            country = meta.get('country', 'Unknown')
            country_idx = self.country_to_idx.get(country, self.unknown_token)
            
            category = meta.get('category', 'Unknown')
            category_idx = self.category_to_idx.get(category, self.unknown_token)
            
            theme = meta.get('theme', 'Unknown')
            theme_idx = self.theme_to_idx.get(theme, self.unknown_token)
            
            price = float(meta.get('min_price', 0))
            
            # Store purchased package data
            self.purchased_packages.append(purchased_pkg_idx)
            self.purchased_countries.append(country_idx)
            self.purchased_categories.append(category_idx)
            self.purchased_themes.append(theme_idx)
            self.purchased_prices.append(price)
        
        # Convert lists to tensors
        self.user_ids = torch.tensor(self.user_ids, dtype=torch.long)
        self.has_short_term = torch.tensor(self.has_short_term, dtype=torch.bool)
        
        if self.is_purchase:
            self.is_purchase = torch.tensor(self.is_purchase, dtype=torch.bool)
        
        if self.session_ids:
            self.session_ids = torch.tensor(self.session_ids, dtype=torch.long)
        
        # Convert package data to tensors
        self.short_term_packages = torch.tensor(self.short_term_packages, dtype=torch.long)
        self.short_term_countries = torch.tensor(self.short_term_countries, dtype=torch.long)
        self.short_term_categories = torch.tensor(self.short_term_categories, dtype=torch.long)
        self.short_term_themes = torch.tensor(self.short_term_themes, dtype=torch.long)
        self.short_term_prices = torch.tensor(self.short_term_prices, dtype=torch.float)
        self.short_term_events = torch.tensor(self.short_term_events, dtype=torch.long)  # Add event tensor
        
        self.long_term_packages = torch.tensor(self.long_term_packages, dtype=torch.long)
        self.long_term_countries = torch.tensor(self.long_term_countries, dtype=torch.long)
        self.long_term_categories = torch.tensor(self.long_term_categories, dtype=torch.long)
        self.long_term_themes = torch.tensor(self.long_term_themes, dtype=torch.long)
        self.long_term_prices = torch.tensor(self.long_term_prices, dtype=torch.float)
        self.long_term_events = torch.tensor(self.long_term_events, dtype=torch.long)  # Add event tensor
        
        self.purchased_packages = torch.tensor(self.purchased_packages, dtype=torch.long)
        self.purchased_countries = torch.tensor(self.purchased_countries, dtype=torch.long)
        self.purchased_categories = torch.tensor(self.purchased_categories, dtype=torch.long)
        self.purchased_themes = torch.tensor(self.purchased_themes, dtype=torch.long)
        self.purchased_prices = torch.tensor(self.purchased_prices, dtype=torch.float)
        
        # Print statistics
        print(f"Prepared dataset with {len(self.user_ids)} samples")
        print(f"Samples with empty short-term packages: {empty_short_term_count} ({empty_short_term_count/len(self.user_ids)*100:.2f}%)")
        print(f"Samples with empty long-term packages: {empty_long_term_count} ({empty_long_term_count/len(self.user_ids)*100:.2f}%)")
        
        # Count event types
        event_type_counts = {}
        event_name_mapping = {v: k for k, v in self.event_to_idx.items()}
        
        for event_idx in [1, 2, 3, 4]:  # ViewContent, AddToCart, InitiateCheckout, Purchase
            count_st = (self.short_term_events == event_idx).sum().item()
            count_lt = (self.long_term_events == event_idx).sum().item()
            event_name = event_name_mapping.get(event_idx, f"Unknown_{event_idx}")
            event_type_counts[event_name] = {'short_term': count_st, 'long_term': count_lt}
        
        print("\nEvent type distribution:")
        for event_name, counts in event_type_counts.items():
            print(f"  {event_name}: ST={counts['short_term']}, LT={counts['long_term']}")
        
        if self.is_purchase is not None and len(self.is_purchase) > 0:
            purchase_count = self.is_purchase.sum().item()
            non_purchase_count = len(self.is_purchase) - purchase_count
            print(f"\nPurchase samples: {purchase_count} ({purchase_count/len(self.user_ids)*100:.2f}%)")
            print(f"Non-purchase samples: {non_purchase_count} ({non_purchase_count/len(self.user_ids)*100:.2f}%)")

def prepare_dataloaders(
    samples: List[Dict[str, Any]],
    package_processor,
    session_processor,
    batch_size: int = 32,
    test_size: float = 0.2,
    random_seed: int = 42,
    num_workers: int = 4,
    empty_token: int = -1,  # Special token for empty values
    unknown_token: int = -1,  # Token for unknown categories
    balance_purchases: bool = True  # Balance purchase and non-purchase samples
) -> tuple:
    """
    Prepare train and test dataloaders with improved handling including event types
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Get mappings and ensure all keys are strings
    user_to_idx = session_processor.get_idx_mappings()['user_to_idx']
    package_to_idx = session_processor.get_idx_mappings()['package_to_idx']
    event_to_idx = session_processor.get_idx_mappings()['event_to_idx']
    country_to_idx = package_processor.get_idx_mappings()['country_to_idx']
    category_to_idx = package_processor.get_idx_mappings()['category_to_idx']
    theme_to_idx = package_processor.get_idx_mappings()['theme_to_idx']
    package_metadata = package_processor.get_package_metadata()
    
    # Fix package ID mapping - ensure all keys are strings
    string_package_to_idx = {str(k): v for k, v in package_to_idx.items()}
    
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
        # Separate purchase and non-purchase samples
        purchase_samples = [s for s in fixed_samples if s['is_purchase']]
        non_purchase_samples = [s for s in fixed_samples if not s['is_purchase']]
        
        # Determine target size for balanced dataset
        target_size = min(len(purchase_samples), len(non_purchase_samples))
        
        # Sample from larger group to match target size
        if len(purchase_samples) > target_size:
            np.random.shuffle(purchase_samples)
            purchase_samples = purchase_samples[:target_size]
        
        if len(non_purchase_samples) > target_size:
            np.random.shuffle(non_purchase_samples)
            non_purchase_samples = non_purchase_samples[:target_size]
        
        # Combine balanced samples
        balanced_samples = purchase_samples + non_purchase_samples
        np.random.shuffle(balanced_samples)
        
        print(f"Balanced dataset: {len(balanced_samples)} samples")
        print(f"  - {len(purchase_samples)} purchase samples")
        print(f"  - {len(non_purchase_samples)} non-purchase samples")
        
        fixed_samples = balanced_samples
    
    # Randomly split samples into train and test sets
    num_samples = len(fixed_samples)
    num_test = int(num_samples * test_size)
    
    # Shuffle indices
    indices = np.random.permutation(num_samples)
    test_indices = indices[:num_test]
    train_indices = indices[num_test:]
    
    # Create train and test samples
    train_samples = [fixed_samples[i] for i in train_indices]
    test_samples = [fixed_samples[i] for i in test_indices]
    
    print(f"Split data into {len(train_samples)} training and {len(test_samples)} testing samples")
    
    # Create datasets with empty and unknown tokens
    train_dataset = TravelPackageDataset(
        train_samples, 
        package_metadata,
        user_to_idx,
        string_package_to_idx,
        country_to_idx,
        category_to_idx,
        theme_to_idx,
        event_to_idx,
        empty_token=empty_token,
        unknown_token=unknown_token
    )
    
    test_dataset = TravelPackageDataset(
        test_samples, 
        package_metadata,
        user_to_idx,
        string_package_to_idx,
        country_to_idx,
        category_to_idx,
        theme_to_idx,
        event_to_idx,
        empty_token=empty_token,
        unknown_token=unknown_token
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, test_loader
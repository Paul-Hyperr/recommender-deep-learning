import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import pickle
import time
import json
from tqdm import tqdm
import sys
import hashlib
from functools import lru_cache
from collections import defaultdict, Counter


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the LLMPackageEncoder and GoogleDutchGeocoder
from models.llm_package_encoder import LLMPackageEncoder
from models.geocoder_dutch import GoogleDutchGeocoder

class PackageProcessor:
    """
    Enhanced PackageProcessor for NATR model
    Processes package metadata including:
    - Title embeddings
    - Coordinates (latitude/longitude)
    - Categorical features (country, city, category, theme)
    - Numerical features (price)
    """
    def __init__(self, feed_data_path='data/feed.parquet', cache_dir='data/cache', 
                 load_coordinates=True, 
                 load_embeddings=True,
                 api_key=None,
                 embedding_model='text-embedding-3-large',
                 use_reduced_embeddings=True):
        """
        Initialize PackageProcessor
        
        Args:
            feed_data_path (str): Path to the package data file (defaults to 'data/feed.parquet')
            cache_dir (str): Directory for caching processed data (only llm_embeddings and geocoding are cached)
            load_coordinates (bool): Whether to load coordinates from geocoder cache
            load_embeddings (bool): Whether to generate title embeddings
            api_key (str): OpenAI API key for embeddings
            embedding_model (str): OpenAI embedding model to use
        """
        self.feed_data_path = feed_data_path
        self.data_path = feed_data_path  # Keep for backward compatibility
        self.cache_dir = cache_dir
        self.load_coordinates = load_coordinates
        self.load_embeddings = load_embeddings
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.embedding_model = embedding_model
        self.use_reduced_embeddings = use_reduced_embeddings
        
        # Set embedding dimension based on model 
        # For text-embedding-3-small, we're using full dimensions to match existing cache
        if embedding_model == 'text-embedding-3-small':
            self.embedding_dim = 1536  # Always use full dimensions for small model
        else:  # text-embedding-3-large
            self.embedding_dim = 768 if use_reduced_embeddings else 3072
        
        print(f"Using embedding model: {embedding_model} with dimension: {self.embedding_dim}")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize dictionaries for mapping IDs
        self.country_to_idx = {}
        self.category_to_idx = {}
        self.theme_to_idx = {}
        self.city_to_idx = {}  
        
        # Package metadata, coordinates, and embeddings
        self.package_metadata = {}
        self.package_coordinates = {}
        self.package_embeddings = {}
        self.df = None
        
        # Initialize LLM encoder if embeddings are requested
        self.llm_encoder = None
        if self.load_embeddings and self.api_key:
            try:
                self.llm_encoder = LLMPackageEncoder(
                    package_vocab_size=100,  # Dummy values since we only need embeddings
                    country_vocab_size=100,
                    category_vocab_size=100,
                    theme_vocab_size=100,
                    api_key=self.api_key,
                    embedding_model=self.embedding_model,
                    cache_dir=os.path.join(self.cache_dir, 'llm_embeddings'),
                    embedding_dim=self.embedding_dim
                )
                print(f"Initialized LLM encoder with {embedding_model}")
            except Exception as e:
                print(f"Warning: Failed to initialize LLM encoder: {e}")
                self.load_embeddings = False
        
        # Load coordinates immediately if requested
        if self.load_coordinates:
            self._load_coordinates_from_geocoder()
    
    def clear_cache(self):
        """Note: LLM embeddings and geocoding caches are preserved"""
        print("Package processor no longer uses general caching.")
        print("LLM embeddings and geocoding caches are preserved in:")
        print(f"  - {os.path.join(self.cache_dir, 'llm_embeddings')}")
        print(f"  - {os.path.join(self.cache_dir, 'geocoding')}")
    
    def _load_coordinates_from_geocoder(self):
        """Load coordinates from geocoder_dutch.py cache and generate missing ones"""
        geocoder_cache_file = os.path.join(self.cache_dir, 'geocoding', 'main_id_coordinates.pkl')
        
        if os.path.exists(geocoder_cache_file):
            try:
                with open(geocoder_cache_file, 'rb') as f:
                    self.package_coordinates = pickle.load(f)
                # Ensure all keys are strings for consistency
                self.package_coordinates = {str(k): v for k, v in self.package_coordinates.items()}
                print(f"Loaded coordinates for {len(self.package_coordinates)} packages from geocoder cache")
            except Exception as e:
                print(f"Error loading geocoder coordinates: {e}")
                self.package_coordinates = {}
        else:
            print("No geocoder cache found. Initializing empty coordinates.")
            self.package_coordinates = {}
            
        # After loading/processing packages, check for missing coordinates
        # This will be called again in _process_package_metadata to geocode missing ones

    def _geocode_missing_packages(self, missing_packages):
        """Geocode packages that don't have coordinates"""
        try:
            # Use the already imported GoogleDutchGeocoder
            google_api_key = os.environ.get('GOOGLE_MAPS_API_KEY')
            if not google_api_key:
                print("Warning: GOOGLE_MAPS_API_KEY environment variable not set. Skipping geocoding.")
                return
                
            geocoder = GoogleDutchGeocoder(
                google_api_key=google_api_key,
                cache_dir=os.path.join(self.cache_dir, 'geocoding')
            )
            
            # Prepare data for geocoding
            geocode_data = []
            for main_id_str, metadata in missing_packages:
                geocode_data.append({
                    'main_id': main_id_str,
                    'city': metadata.get('city', ''),
                    'country': metadata.get('country', '')
                })
            
            # Process in batches
            batch_size = 100
            geocoded_count = 0
            
            for i in range(0, len(geocode_data), batch_size):
                batch = geocode_data[i:i+batch_size]
                
                for item in batch:
                    main_id_str = item['main_id']
                    city = item['city']
                    country = item['country']
                    
                    # Try to geocode
                    coords = geocoder.geocode_location(city, country)
                    
                    if coords is not None:
                        # coords is a tuple (lat, lng)
                        lat, lng = coords
                        
                        # Create coordinate dict
                        coord_dict = {
                            'latitude': lat,
                            'longitude': lng
                        }
                        
                        # Update package coordinates
                        self.package_coordinates[main_id_str] = coord_dict
                        
                        # Update metadata
                        self.package_metadata[main_id_str]['latitude'] = lat
                        self.package_metadata[main_id_str]['longitude'] = lng
                        
                        geocoded_count += 1
                
                print(f"Geocoded batch {i//batch_size + 1}/{(len(geocode_data) + batch_size - 1)//batch_size}")
            
            print(f"Successfully geocoded {geocoded_count} out of {len(missing_packages)} packages")
            
            # Save updated coordinates to cache
            if geocoded_count > 0:
                geocoder_cache_file = os.path.join(self.cache_dir, 'geocoding', 'main_id_coordinates.pkl')
                os.makedirs(os.path.dirname(geocoder_cache_file), exist_ok=True)
                
                with open(geocoder_cache_file, 'wb') as f:
                    pickle.dump(self.package_coordinates, f)
                print(f"Updated geocoder cache with new coordinates")
                
        except ImportError:
            print("Warning: Could not import GeocoderDutch. Skipping coordinate generation.")
        except Exception as e:
            print(f"Error geocoding missing packages: {e}")
    
    def _generate_embeddings(self):
        """Generate embeddings for all package titles using LLMPackageEncoder"""
        if not self.load_embeddings or not self.llm_encoder:
            return
        
        print("Generating title embeddings using LLMPackageEncoder...")
        
        # Collect packages that need embeddings
        packages_to_encode = []
        
        for main_id, metadata in self.package_metadata.items():
            main_id_str = str(main_id)
            
            if main_id_str not in self.package_embeddings:
                # Check LLM encoder cache first
                cached_embedding = self.llm_encoder.embedding_cache.get(main_id_str)
                
                if cached_embedding is not None:
                    if isinstance(cached_embedding, list):
                        cached_embedding = np.array(cached_embedding)
                    self.package_embeddings[main_id_str] = cached_embedding
                else:
                    # Add to list for batch processing
                    packages_to_encode.append({
                        'main_id': main_id_str,
                        'title': metadata.get('title', ''),
                        'theme': metadata.get('theme', ''),
                        'category': metadata.get('category', ''),
                        'city': metadata.get('city', ''),
                        'country': metadata.get('country', '')
                    })
        
        if not packages_to_encode:
            print("All packages already have embeddings")
            return
        
        print(f"Generating embeddings for {len(packages_to_encode)} packages...")
        
        # Process packages in batches
        batch_size = 200
        for i in range(0, len(packages_to_encode), batch_size):
            batch_packages = packages_to_encode[i:i+batch_size]
            
            # Extract data for processing
            batch_titles = [pkg['title'] for pkg in batch_packages]
            batch_main_ids = [pkg['main_id'] for pkg in batch_packages]
            
            # Get embeddings using enhanced LLM encoder (with fallback support)
            embeddings_tensor = self.llm_encoder.process_batch_titles(
                titles=batch_titles,
                main_ids=batch_main_ids,
                package_metadata=batch_packages  # Pass full metadata for fallback generation
            )
            
            # Convert to numpy and store
            for j, main_id_str in enumerate(batch_main_ids):
                embedding_np = embeddings_tensor[j].detach().numpy()
                self.package_embeddings[main_id_str] = embedding_np
            
            print(f"Processed batch {i//batch_size + 1}/{(len(packages_to_encode) + batch_size - 1)//batch_size}")
        
        # Save LLM encoder cache
        self.llm_encoder._save_embedding_cache()
        
        # Ensure all keys are strings
        self.package_embeddings = {str(k): v for k, v in self.package_embeddings.items()}
        
        print(f"Generated embeddings for {len(packages_to_encode)} packages")
        
        # Report any packages still without embeddings
        missing_embeddings = []
        for main_id, metadata in self.package_metadata.items():
            main_id_str = str(main_id)
            if main_id_str not in self.package_embeddings:
                missing_embeddings.append(main_id_str)
        
        if missing_embeddings:
            print(f"Warning: {len(missing_embeddings)} packages still without embeddings: {missing_embeddings[:5]}...")
        else:
            print("All packages now have embeddings!")
    
    def load_data(self, feed_data_path=None):
        """Load package data directly without general caching (llm_embeddings and geocoding are still cached)"""
        # Allow overriding the feed data path
        if feed_data_path:
            self.feed_data_path = feed_data_path
            self.data_path = feed_data_path  # Backward compatibility
            
        # Load data from source
        if self.feed_data_path:
            print(f"Loading package data from {self.feed_data_path}")
            
            # Load file based on extension
            try:
                if self.feed_data_path.endswith('.parquet'):
                    self.df = pd.read_parquet(self.feed_data_path)
                else:
                    self.df = pd.read_csv(self.feed_data_path)
                print(f"Loaded {len(self.df)} packages")
            except Exception as e:
                print(f"Error loading file: {e}")
                return
            
            # Process package metadata (this will handle coordinates and embeddings)
            self._process_package_metadata()
            
            print(f"Package data loading complete. No general caching applied.")
        else:
            print("Error: No feed data path provided")


    def _process_package_metadata(self):
        """Process package metadata from the dataframe with enhanced price handling"""
        print("Processing package metadata...")
        
        # First ensure coordinates are loaded
        if self.load_coordinates and not self.package_coordinates:
            self._load_coordinates_from_geocoder()
        
        self.package_metadata = {}
        
        # First pass - extract all data and collect price information
        category_prices = defaultdict(list)
        theme_prices = defaultdict(list)
        
        # Process each row
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing packages"):
            main_id_str = str(row['main_id'])  # ALWAYS use string for consistency
            
            # Get price - may be zero/null
            price = float(row.get('min_price', 0))
            category = row.get('category', '')
            theme = row.get('theme', '')
            
            # Store all metadata for NATR model
            metadata = {
                'title': row.get('title', ''),
                'city': row.get('city', ''),
                'country': row.get('country', ''),
                'category': category,
                'theme': theme,
                'min_price': price,
                'has_price': price > 0  # Add indicator for presence of price
            }
            
            # Add coordinates if available - check with string key
            if main_id_str in self.package_coordinates:
                coord_data = self.package_coordinates[main_id_str]
                metadata['latitude'] = coord_data.get('latitude')
                metadata['longitude'] = coord_data.get('longitude')
            
            self.package_metadata[main_id_str] = metadata
            
            # Collect non-zero prices for calculating medians later
            if price > 0:
                category_prices[category].append(price)
                theme_prices[theme].append(price)
        
        # Per client requirement:
        # - Every price above 5000 is wrong UNLESS the country is Saudi-Arabië
        # - Any 0.0 price needs to be filled
        print("Filtering out incorrect prices per client requirements...")
        price_min = 0.01   # Minimum valid price
        normal_price_max = 5000.0 # Maximum valid price for most countries
        saudi_price_max = 7000.0  # Maximum valid price for Saudi-Arabië (has package of 6472.0)
        
        # Filter out incorrect prices from category collections based on the specific rules
        filtered_category_prices = {}
        for cat, prices in category_prices.items():
            # A price is invalid if it's above 5000 (unless Saudi Arabia) or <= 0
            filtered_prices = []
            for p in prices:
                # Only include prices that are valid by the requirements
                if price_min <= p <= normal_price_max:
                    filtered_prices.append(p)
            
            if filtered_prices:
                filtered_category_prices[cat] = filtered_prices
                if len(prices) != len(filtered_prices):
                    print(f"  Category '{cat}': {len(prices) - len(filtered_prices)} out of {len(prices)} prices filtered out")
        
        # Collect prices by country-category combination for more accurate estimates
        # This is the preferred method per client requirements
        country_category_prices = defaultdict(list)
        for main_id_str, metadata in self.package_metadata.items():
            price = metadata['min_price']
            country = metadata['country']
            category = metadata['category']
            
            # Check if price is valid based on country
            # Saudi-Arabië can have higher prices (up to 7000)
            if country == "Saudi-Arabië":
                is_valid_price = price_min <= price <= saudi_price_max
            else:
                is_valid_price = price_min <= price <= normal_price_max
                
            # Only include valid prices
            if is_valid_price:
                # Create combined key for country-category pair
                cc_key = f"{country}_{category}"
                country_category_prices[cc_key].append(price)
        
        # Calculate median prices with min sample threshold
        min_samples = 3  # Require at least this many samples for stable medians
        
        # Price statistics by country-category combination (primary method)
        country_category_median_prices = {cc_key: np.median(prices) for cc_key, prices in country_category_prices.items() 
                                       if len(prices) >= min_samples}
        
        # Price statistics by category (fallback method)
        category_median_prices = {cat: np.median(prices) for cat, prices in filtered_category_prices.items() 
                                if len(prices) >= min_samples}
        
        # Calculate overall median for final fallback
        all_valid_prices = []
        for prices in country_category_prices.values():
            all_valid_prices.extend(prices)
        overall_median = np.median(all_valid_prices) if all_valid_prices else 100.0
        
        print(f"Created price statistics:")
        print(f"  Category medians: {len(category_median_prices)} categories")
        print(f"  Country-category medians: {len(country_category_median_prices)} combinations")
        
        # Overall median as fallback
        all_prices = [p for prices in category_prices.values() for p in prices]
        overall_median = np.median(all_prices) if all_prices else 100.0  # Default fallback
        
        # Note: Price extraction from theme names removed as it wasn't being used
        
        # Second pass - fix zero prices
        for main_id_str, metadata in self.package_metadata.items():
            price = metadata['min_price']
            
            # Check if this is an incorrect price based on client requirements
            # - Zero or negative price is always invalid
            # - Price > 5000 is invalid UNLESS country is Saudi-Arabië
            # - For Saudi-Arabië, price > 7000 is invalid
            country = metadata.get('country', '')
            
            if country == "Saudi-Arabië":
                price_invalid = price <= 0 or price > saudi_price_max
            else:
                price_invalid = price <= 0 or price > normal_price_max
            
            if price_invalid:
                # Store original price for reporting
                if 'original_price' not in metadata:
                    metadata['original_price'] = price
                
                # 1. Try country-category combination first (most specific)
                # This is the preferred method per client requirements
                if metadata['country'] and metadata['category']:
                    cc_key = f"{metadata['country']}_{metadata['category']}"
                    if cc_key in country_category_median_prices:
                        metadata['min_price'] = country_category_median_prices[cc_key]
                        metadata['price_source'] = 'country_category_median'
                
                # 2. If that fails, use category median
                elif metadata['category'] in category_median_prices:
                    metadata['min_price'] = category_median_prices[metadata['category']]
                    metadata['price_source'] = 'category_median'
                
                # 3. Last resort - use overall median
                else:
                    metadata['min_price'] = overall_median
                    metadata['price_source'] = 'overall_median'
                
                # Update has_price indicator and ensure price is always positive
                metadata['min_price'] = max(0.01, metadata['min_price'])  # Ensure no zero prices remain
                metadata['has_price'] = True
                
                # Add original price to track what was changed
                if 'original_price' not in metadata:
                    metadata['original_price'] = price
        
        # Print statistics about price fixes
        valid_original = sum(1 for m in self.package_metadata.values() if m.get('price_source') is None)
        from_country_category = sum(1 for m in self.package_metadata.values() if m.get('price_source') == 'country_category_median')
        from_category_median = sum(1 for m in self.package_metadata.values() if m.get('price_source') == 'category_median')
        from_overall = sum(1 for m in self.package_metadata.values() if m.get('price_source') == 'overall_median')
        total_fixed = from_country_category + from_category_median + from_overall
        
        # Calculate average prices from different sources for reporting
        avg_original = np.mean([m['min_price'] for m in self.package_metadata.values() if m.get('price_source') is None])
        avg_country_category = np.mean([m['min_price'] for m in self.package_metadata.values() if m.get('price_source') == 'country_category_median']) if from_country_category > 0 else 0
        avg_category_median = np.mean([m['min_price'] for m in self.package_metadata.values() if m.get('price_source') == 'category_median']) if from_category_median > 0 else 0
        avg_overall = np.mean([m['min_price'] for m in self.package_metadata.values() if m.get('price_source') == 'overall_median']) if from_overall > 0 else 0
        
        # Calculate statistics for high-priced packages
        high_price_non_saudi = sum(1 for m in self.package_metadata.values() 
                              if 'original_price' in m and 
                                 m.get('original_price', 0) > normal_price_max and
                                 m.get('country', '') != "Saudi-Arabië")
                                 
        high_price_saudi = sum(1 for m in self.package_metadata.values() 
                           if 'original_price' in m and 
                              m.get('original_price', 0) > saudi_price_max and
                              m.get('country', '') == "Saudi-Arabië")
                              
        saudi_packages = sum(1 for m in self.package_metadata.values() 
                          if m.get('country', '') == "Saudi-Arabië")
        
        print(f"\nPrice statistics:")
        print(f"  Original valid prices: {valid_original} ({valid_original/len(self.package_metadata)*100:.1f}%) - avg €{avg_original:.2f}")
        print(f"  Total fixed prices: {total_fixed} ({total_fixed/len(self.package_metadata)*100:.1f}%)")
        print(f"  Prices from country-category median: {from_country_category} - avg €{avg_country_category:.2f}")
        print(f"  Prices from category median: {from_category_median} - avg €{avg_category_median:.2f}")
        print(f"  Prices from overall median: {from_overall} - avg €{avg_overall:.2f}")
        print(f"  Non-Saudi packages with price > €{normal_price_max}: {high_price_non_saudi}")
        print(f"  Saudi packages: {saudi_packages} total, {high_price_saudi} with price > €{saudi_price_max}")
        print(f"  All packages now have valid prices")
        
        # Report on price distribution
        all_prices = [m['min_price'] for m in self.package_metadata.values()]
        price_percentiles = [np.percentile(all_prices, p) for p in [5, 25, 50, 75, 95]]
        print(f"\nPrice distribution (€): ")
        print(f"  Min: {min(all_prices):.2f}, Max: {max(all_prices):.2f}")
        print(f"  Percentiles [5, 25, 50, 75, 95]: {[f'{p:.2f}' for p in price_percentiles]}")
        
        # Generate embeddings after all metadata is loaded
        if self.load_embeddings:
            # First try loading from LLM embedding cache directly
            llm_cache_path = os.path.join(self.cache_dir, 'llm_embeddings', f'{self.embedding_model}_cache.json')
            if os.path.exists(llm_cache_path):
                try:
                    print(f"Loading embeddings directly from {llm_cache_path}")
                    with open(llm_cache_path, 'r') as f:
                        embedding_cache = json.load(f)
                        for main_id_str, embedding in embedding_cache.items():
                            if isinstance(embedding, list):
                                self.package_embeddings[main_id_str] = np.array(embedding)
                            else:
                                self.package_embeddings[main_id_str] = embedding
                    print(f"Loaded {len(self.package_embeddings)} embeddings directly from cache")
                    
                    # IMPORTANT: Check for missing embeddings and generate them
                    missing_packages = []
                    for main_id_str in self.package_metadata:
                        if main_id_str not in self.package_embeddings:
                            missing_packages.append(main_id_str)
                    
                    if missing_packages:
                        print(f"Found {len(missing_packages)} packages without embeddings")
                        print(f"Generating embeddings for missing packages...")
                        self._generate_embeddings()  # This will only generate for missing ones
                    
                except Exception as e:
                    print(f"Error loading embeddings directly: {e}, falling back to normal method")
                    self._generate_embeddings()
            else:
                self._generate_embeddings()
        
        # Now add embeddings to metadata - use string keys
        for main_id_str in self.package_metadata:
            if main_id_str in self.package_embeddings:
                self.package_metadata[main_id_str]['title_embedding'] = self.package_embeddings[main_id_str]
        
        # Report statistics
        print(f"\nProcessed {len(self.package_metadata)} packages")
        
        # Count packages with coordinates
        with_coords = sum(1 for m in self.package_metadata.values() 
                        if 'latitude' in m and m['latitude'] is not None)
        print(f"Packages with coordinates: {with_coords} ({with_coords/len(self.package_metadata)*100:.1f}%)")
        
        # Count packages with embeddings
        with_embeddings = sum(1 for m in self.package_metadata.values() 
                            if 'title_embedding' in m and m['title_embedding'] is not None)
        print(f"Packages with embeddings: {with_embeddings} ({with_embeddings/len(self.package_metadata)*100:.1f}%)")
        
        # IMPORTANT: Geocode missing packages if any
        if self.load_coordinates:
            missing_coords = []
            for main_id_str, metadata in self.package_metadata.items():
                if 'latitude' not in metadata or metadata.get('latitude') is None:
                    missing_coords.append((main_id_str, metadata))
            
            if missing_coords:
                print(f"\nFound {len(missing_coords)} packages without coordinates")
                print("Attempting to geocode missing packages...")
                self._geocode_missing_packages(missing_coords)
        
    
    
    def get_package_features(self, main_id: str) -> Dict[str, Any]:
        """
        Get all features for a package required by NATR model
        
        Args:
            main_id: Package ID
            
        Returns:
            Dictionary containing all package features
        """
        main_id = str(main_id)
        
        if main_id not in self.package_metadata:
            return None
        
        metadata = self.package_metadata[main_id]
        
        # Prepare feature dictionary for NATR
        features = {
            'main_id': main_id,
            'title': metadata.get('title', ''),
            'title_embedding': metadata.get('title_embedding'),
            'city': metadata.get('city', ''),
            'country': metadata.get('country', ''),
            'category': metadata.get('category', ''),
            'theme': metadata.get('theme', ''),
            'price': metadata.get('min_price', 0),
            'latitude': metadata.get('latitude'),
            'longitude': metadata.get('longitude')
        }
        
        # Ensure indices are available
        if not self.country_to_idx:
            self.create_mappings()
        
        # Add index mappings for categorical features
        features['city_idx'] = self.city_to_idx.get(features['city'], self.city_to_idx.get('Unknown', 0))
        features['country_idx'] = self.country_to_idx.get(features['country'], self.country_to_idx.get('Unknown', 0))
        features['category_idx'] = self.category_to_idx.get(features['category'], self.category_to_idx.get('Unknown', 0))
        features['theme_idx'] = self.theme_to_idx.get(features['theme'], self.theme_to_idx.get('Unknown', 0))
        
        return features
    
    def get_batch_features(self, main_ids: List[str]) -> Dict[str, torch.Tensor]:
        """
        Get features for a batch of packages as tensors ready for the model
        
        Args:
            main_ids: List of package IDs
            
        Returns:
            Dictionary of tensors for each feature type
        """
        batch_features = {
            'title_embeddings': [],
            'coordinates': [],
            'country_ids': [],
            'city_ids': [],
            'category_ids': [],
            'theme_ids': [],
            'prices': [],
            'price_sources': []  # Track price source for reporting
        }
        
        # Detect embedding dimension from existing features
        embedding_dim = self.embedding_dim if hasattr(self, 'embedding_dim') else 768
        
        for main_id in main_ids:
            features = self.get_package_features(main_id)
            
            if features:
                # Title embedding
                if features['title_embedding'] is not None:
                    batch_features['title_embeddings'].append(features['title_embedding'])
                else:
                    # Use zero embedding if not available
                    batch_features['title_embeddings'].append(np.zeros(embedding_dim))
                
                # Coordinates
                if features['latitude'] is not None and features['longitude'] is not None:
                    batch_features['coordinates'].append([features['latitude'], features['longitude']])
                else:
                    batch_features['coordinates'].append([0.0, 0.0])
                
                # Categorical features
                batch_features['country_ids'].append(features['country_idx'])
                batch_features['city_ids'].append(features['city_idx'])
                batch_features['category_ids'].append(features['category_idx'])
                batch_features['theme_ids'].append(features['theme_idx'])
                
                # Price and price source
                batch_features['prices'].append(features['price'])
                
                # Get price source if available
                if main_id in self.package_metadata:
                    source = self.package_metadata[main_id].get('price_source', 'original')
                    batch_features['price_sources'].append(source)
                else:
                    batch_features['price_sources'].append('unknown')
            else:
                # Handle missing packages
                batch_features['title_embeddings'].append(np.zeros(embedding_dim))
                batch_features['coordinates'].append([0.0, 0.0])
                batch_features['country_ids'].append(0)
                batch_features['city_ids'].append(0)
                batch_features['category_ids'].append(0)
                batch_features['theme_ids'].append(0)
                batch_features['prices'].append(0.0)
                batch_features['price_sources'].append('missing')
        
        # Convert to tensors
        raw_prices = torch.tensor(batch_features['prices'], dtype=torch.float)
        
        # Create tensor dictionary with both raw and normalized prices
        tensor_dict = {
            'title_embeddings': torch.tensor(np.array(batch_features['title_embeddings']), dtype=torch.float),
            'coordinates': torch.tensor(batch_features['coordinates'], dtype=torch.float),
            'country_ids': torch.tensor(batch_features['country_ids'], dtype=torch.long),
            'city_ids': torch.tensor(batch_features['city_ids'], dtype=torch.long),
            'category_ids': torch.tensor(batch_features['category_ids'], dtype=torch.long),
            'theme_ids': torch.tensor(batch_features['theme_ids'], dtype=torch.long),
            'prices': raw_prices,
            'normalized_prices': self.normalize_prices(raw_prices)
        }
        
        # Report price statistics on first batch or if explicitly requested
        if not hasattr(self, '_reported_price_stats') or len(main_ids) > 100:
            self._report_batch_price_stats(raw_prices, batch_features['price_sources'])
            self._reported_price_stats = True
            
        return tensor_dict
        
    def _report_batch_price_stats(self, prices: torch.Tensor, sources: List[str]) -> None:
        """Report price statistics for a batch"""
        if len(prices) == 0:
            return
            
        # Basic statistics
        min_price = float(torch.min(prices))
        max_price = float(torch.max(prices))
        mean_price = float(torch.mean(prices))
        median_price = float(torch.median(prices))
        
        # Count by source
        source_counts = {}
        for source in sources:
            if source in source_counts:
                source_counts[source] += 1
            else:
                source_counts[source] = 1
                
        # Print summary
        print(f"\nBatch price statistics (n={len(prices)}):")
        print(f"  Range: €{min_price:.2f} - €{max_price:.2f}")
        print(f"  Mean: €{mean_price:.2f}, Median: €{median_price:.2f}")
        
        if source_counts:
            print("  Price sources:")
            for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"    {source}: {count} ({count/len(prices)*100:.1f}%)")
    
    def prepare_package_tensors(self) -> Dict[str, torch.Tensor]:
        """
        Prepare all package features as tensors for efficient model usage
        
        Returns:
            Dictionary of tensors for all packages
        """
        
        print("Preparing package feature tensors...")
        
        if not self.package_metadata:
            self.load_data()
        
        # Ensure mappings are created
        if not self.country_to_idx:
            self.create_mappings()
        
        # Get all package IDs sorted by index - ensure strings
        package_to_idx = {str(main_id): idx for idx, main_id in enumerate(sorted(self.package_metadata.keys()))}
        num_packages = len(package_to_idx)
        
        # Initialize tensors based on the embedding model being used
        embedding_dim = self.embedding_dim  # Use the dimension specified at initialization
        
        # Find the correct cache file for the requested embedding model
        llm_cache_path = os.path.join(self.cache_dir, 'llm_embeddings', f'{self.embedding_model}_cache.json')
        
        if not os.path.exists(llm_cache_path):
            print(f"Warning: Cache for {self.embedding_model} not found")
            # Check for other embedding cache files
            for model_name in ['text-embedding-3-small', 'text-embedding-3-large']:
                potential_path = os.path.join(self.cache_dir, 'llm_embeddings', f'{model_name}_cache.json')
                if os.path.exists(potential_path):
                    print(f"Found alternative embedding cache: {potential_path}")
                    if model_name != self.embedding_model:
                        print(f"Warning: Using {model_name} cache, but requested {self.embedding_model}")
                        print(f"This will lead to dimension mismatch and poor performance")
                    llm_cache_path = potential_path
                    break
        
        # Check if we can detect the dimension from cache
        if os.path.exists(llm_cache_path):
            try:
                print(f"Verifying embedding dimension from cache file: {llm_cache_path}")
                with open(llm_cache_path, 'r') as f:
                    first_key = next(iter(json.load(f).keys()))
                    with open(llm_cache_path, 'r') as f2:
                        embedding = json.load(f2)[first_key]
                        if isinstance(embedding, list):
                            cache_dim = len(embedding)
                            if cache_dim != self.embedding_dim:
                                print(f"INFO: Cache dimension {cache_dim}, configured dimension {self.embedding_dim}")
                                # For text-embedding-3-small, we now expect 1536 dimensions
                                if self.embedding_model == 'text-embedding-3-small' and cache_dim == 1536:
                                    print(f"Using full dimensions (1536) for text-embedding-3-small as configured")
                                    embedding_dim = 1536
                                # Otherwise, adapt to the cache if reasonable
                                elif abs(cache_dim - self.embedding_dim) <= 500:
                                    print(f"Adapting to cache dimension: {cache_dim}")
                                    embedding_dim = cache_dim
                                else:
                                    print(f"Using configured dimension: {self.embedding_dim}")
                            else:
                                print(f"Cache dimension matches configured dimension: {embedding_dim}")
            except Exception as e:
                print(f"Error detecting embedding dimension: {e}, using default: {embedding_dim}")
        
        print(f"Using embedding dimension: {embedding_dim}")
        
        title_embeddings = np.zeros((num_packages, embedding_dim))
        coordinates = np.zeros((num_packages, 2))  # lat, lon
        country_ids = np.zeros(num_packages, dtype=np.int64)
        city_ids = np.zeros(num_packages, dtype=np.int64)
        category_ids = np.zeros(num_packages, dtype=np.int64)
        theme_ids = np.zeros(num_packages, dtype=np.int64)
        prices = np.zeros(num_packages, dtype=np.float32)
        
        # Load embeddings directly from cache for improved reliability
        print("Loading embeddings directly from cache...")
        embeddings_loaded = False
        
        if os.path.exists(llm_cache_path):
            try:
                with open(llm_cache_path, 'r') as f:
                    embeddings_cache = json.load(f)
                    print(f"Loaded {len(embeddings_cache)} embeddings from cache")
                    
                    # Create direct mapping from package index to embedding
                    print("Creating direct embedding mapping...")
                    direct_embeddings = {}
                    for main_id, idx in package_to_idx.items():
                        if main_id in embeddings_cache:
                            direct_embeddings[idx] = np.array(embeddings_cache[main_id])
                    
                    print(f"Created direct mappings for {len(direct_embeddings)} packages")
                    
                    # Apply direct embeddings
                    for idx, embedding in direct_embeddings.items():
                        title_embeddings[idx] = embedding
                    
                    embeddings_loaded = True
                    
            except Exception as e:
                print(f"Error loading embeddings directly: {e}")
                embeddings_loaded = False
        
        # If direct loading failed, fall back to regular loading
        if not embeddings_loaded:
            print("Falling back to regular feature loading method...")
            for main_id_str, idx in tqdm(package_to_idx.items(), desc="Creating feature tensors"):
                features = self.get_package_features(main_id_str)
                
                if features:
                    # Title embedding with dimension handling
                    if features['title_embedding'] is not None:
                        embedding = features['title_embedding']
                        
                        # Check for dimension mismatch (should be rare now with correct config)
                        if embedding.shape[0] != title_embeddings.shape[1]:
                            # Print warning only once per run
                            if not hasattr(self, '_dimension_warning_shown'):
                                print(f"WARNING: Found embedding with dimension {embedding.shape[0]}, expected {title_embeddings.shape[1]}")
                                print(f"This should be rare now that we're using full dimensions. Handling the mismatch...")
                                self._dimension_warning_shown = True
                            
                            # Handle the rare case of dimension mismatch
                            if embedding.shape[0] > title_embeddings.shape[1]:
                                embedding = embedding[:title_embeddings.shape[1]]
                            else:
                                padded = np.zeros(title_embeddings.shape[1])
                                padded[:embedding.shape[0]] = embedding
                                embedding = padded
                                
                        title_embeddings[idx] = embedding
        
        # Load other features (coordinates, categories, etc.)
        print("Loading other features...")
        for main_id_str, idx in tqdm(package_to_idx.items(), desc="Creating other features"):
            features = self.get_package_features(main_id_str)
            
            if features:
                # Coordinates
                if features['latitude'] is not None and features['longitude'] is not None:
                    coordinates[idx] = [features['latitude'], features['longitude']]
                
                # Categorical features
                country_ids[idx] = features['country_idx']
                city_ids[idx] = features['city_idx']
                category_ids[idx] = features['category_idx']
                theme_ids[idx] = features['theme_idx']
                
                # Price
                prices[idx] = features['price']
        
        # Report statistics
        non_zero_embeddings = np.sum(np.sum(title_embeddings, axis=1) != 0)
        non_zero_coordinates = np.sum(np.sum(coordinates, axis=1) != 0)
        print(f"  Non-zero embeddings: {non_zero_embeddings}/{num_packages}")
        print(f"  Non-zero coordinates: {non_zero_coordinates}/{num_packages}")
        
        # Convert to tensors
        feature_tensors = {
            'package_to_idx': package_to_idx,
            'title_embeddings': torch.tensor(title_embeddings, dtype=torch.float),
            'coordinates': torch.tensor(coordinates, dtype=torch.float),
            'country_ids': torch.tensor(country_ids, dtype=torch.long),
            'city_ids': torch.tensor(city_ids, dtype=torch.long),
            'category_ids': torch.tensor(category_ids, dtype=torch.long),
            'theme_ids': torch.tensor(theme_ids, dtype=torch.long),
            'prices': torch.tensor(prices, dtype=torch.float)
        }
        
        print(f"Prepared feature tensors for {num_packages} packages")
        return feature_tensors
    
    def create_mappings(self):
        """Create mappings from package attributes to indices"""
        
        print("Creating package attribute mappings...")
        
        if not self.package_metadata:
            self.load_data()
        
        # Extract unique values
        cities = set()
        countries = set()
        categories = set()
        themes = set()
        
        for metadata in self.package_metadata.values():
            if metadata.get('city'):
                cities.add(metadata['city'])
            if metadata.get('country'):
                countries.add(metadata['country'])
            if metadata.get('category'):
                categories.add(metadata['category'])
            if metadata.get('theme'):
                themes.add(metadata['theme'])
        
        # Create mappings (reserve index 0 for padding/unknown)
        self.city_to_idx = {city: i+1 for i, city in enumerate(sorted(cities))}
        self.country_to_idx = {country: i+1 for i, country in enumerate(sorted(countries))}
        self.category_to_idx = {category: i+1 for i, category in enumerate(sorted(categories))}
        self.theme_to_idx = {theme: i+1 for i, theme in enumerate(sorted(themes))}
        
        # Add 'Unknown' entries
        for mapping in [self.city_to_idx, self.country_to_idx, self.category_to_idx, self.theme_to_idx]:
            if 'Unknown' not in mapping:
                mapping['Unknown'] = len(mapping) + 1
        
        # Print statistics
        print(f"Created mappings for {len(self.country_to_idx)} countries, "
              f"{len(self.city_to_idx)} cities, "
              f"{len(self.category_to_idx)} categories, and "
              f"{len(self.theme_to_idx)} themes")

    def has_embeddings(self) -> bool:
        """Check if embeddings are loaded"""
        # Check if we have any embeddings in package_metadata
        if self.package_embeddings:
            return True
        
        # Check if any metadata has embeddings
        for metadata in self.package_metadata.values():
            if 'title_embedding' in metadata and metadata['title_embedding'] is not None:
                return True
        
        # Check if LLM encoder is available
        return self.llm_encoder is not None

    def has_coordinates(self) -> bool:
        """Check if coordinates are loaded"""
        # Check if we have coordinates dict
        if self.package_coordinates:
            return True
        
        # Check if any metadata has coordinates
        for metadata in self.package_metadata.values():
            if 'latitude' in metadata and metadata['latitude'] is not None:
                return True
        
        return False
        
    def normalize_prices(self, prices: torch.Tensor) -> torch.Tensor:
        """
        Normalize price values to a more useful range for the model
        
        Args:
            prices: Tensor of raw price values
            
        Returns:
            Normalized price tensor in [0, 1] range
        """
        # First clamp to reasonable maximum for stability (€10,000)
        max_price_value = 10000.0
        clamped_prices = torch.clamp(prices, min=0.01, max=max_price_value)
        
        # Apply log transformation to handle wide range of prices
        # Add small epsilon to avoid log(0)
        eps = 1e-6
        log_prices = torch.log(clamped_prices + eps)
        
        # Get min/max for normalization
        min_val = torch.min(log_prices)
        max_val = torch.max(log_prices)
        
        # Handle edge case where all prices are the same
        if max_val == min_val:
            return torch.ones_like(prices) * 0.5
            
        # Min-max normalization to [0, 1]
        normalized = (log_prices - min_val) / (max_val - min_val)
        
        # Final safety clamp to ensure values are in [0, 1]
        normalized = torch.clamp(normalized, min=0.0, max=1.0)
        
        return normalized

    def get_package_metadata(self):
        """Get all package metadata"""
        if not self.package_metadata:
            self.load_data()
        return self.package_metadata
    
    def get_idx_mappings(self):
        """Get the index mappings"""
        if not self.city_to_idx:
            self.create_mappings()
        
        return {
            'city_to_idx': self.city_to_idx,
            'country_to_idx': self.country_to_idx,
            'category_to_idx': self.category_to_idx,
            'theme_to_idx': self.theme_to_idx
        }

class TravelPackageDataset(Dataset):
    """
    Optimized Dataset for NATR travel package recommendation
    
    Key optimizations:
    - Lazy loading of embeddings/coordinates
    - Batch preprocessing with caching
    - Memory-efficient tensor operations
    - Reduced redundant computations
    - Temporal recency awareness
    """
    def __init__(self, 
                samples: List[Dict[str, Any]], 
                package_processor,
                user_to_idx: Dict[Any, int],
                package_to_idx: Dict[str, int],
                event_to_idx: Dict[str, int] = None,
                max_short_term: int = 10,
                max_long_term: int = 20,
                empty_token: int = -1,  
                unknown_token: int = -1,
                use_cache: bool = False,  # Disabled by default to avoid stale cache issues
                prefetch_features: bool = True):
        """Initialize with optimizations"""
        
        self.samples = samples
        self.package_processor = package_processor
        self.user_to_idx = user_to_idx
        self.package_to_idx = package_to_idx
        self.event_to_idx = event_to_idx or {'ViewContent': 1, 'AddToCart': 2, 'InitiateCheckout': 3, 'Purchase': 4}
        self.max_short_term = max_short_term
        self.max_long_term = max_long_term
        self.empty_token = empty_token
        self.unknown_token = unknown_token
        self.use_cache = use_cache
        self.prefetch_features = prefetch_features
        
        # Get mappings
        mappings = self.package_processor.get_idx_mappings()
        self.country_to_idx = mappings['country_to_idx']
        self.category_to_idx = mappings['category_to_idx']
        self.theme_to_idx = mappings['theme_to_idx']
        
        # Load package features tensors
        self.package_features = self.package_processor.prepare_package_tensors()
        
        # Store dimensions
        self.embedding_dim = self.package_features['title_embeddings'].shape[1]
        
        # Optimization: Create reverse mapping once
        self.idx_to_main_id = {v: k for k, v in self.package_to_idx.items()}
        
        # Optimization: Prepare samples with batch processing
        self._prepare_samples_optimized()
        
        # Optimization: Prefetch commonly used features
        if prefetch_features:
            self._prefetch_common_features()
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.samples)
    
    def _generate_cache_key(self):
        """Generate a unique cache key based on dataset parameters"""
        # Create a string with all parameters that affect the dataset
        key_str = (
            f"samples_{len(self.samples)}_"
            f"max_short_{self.max_short_term}_"
            f"max_long_{self.max_long_term}"
        )
        
        # Hash it for a shorter filename
        return hashlib.md5(key_str.encode()).hexdigest()[:10]
    
    def _prepare_samples_optimized(self):
        """Optimized sample preparation with batch processing"""
        
        # Check cache first
        cache_key = self._generate_cache_key()
        cache_file = os.path.join(self.package_processor.cache_dir, f'dataset_{cache_key}.pkl')
        
        if self.use_cache and os.path.exists(cache_file):
            print("Loading cached dataset...")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                for key, value in cache_data.items():
                    setattr(self, key, value)
                print(f"Loaded {len(self.user_ids)} samples from cache")
                return
        
        print("Preparing samples (optimized)...")
        
        # Process in batches for memory efficiency
        batch_size = 10000
        all_tensors = defaultdict(list)
        
        for i in range(0, len(self.samples), batch_size):
            batch_samples = self.samples[i:i+batch_size]
            batch_tensors = self._process_batch(batch_samples)
            
            for key, tensor in batch_tensors.items():
                all_tensors[key].append(tensor)
            
            # Clear memory periodically
            if i % 50000 == 0:
                torch.cuda.empty_cache()
        
        # Concatenate all batches
        for key, tensor_list in all_tensors.items():
            setattr(self, key, torch.cat(tensor_list, dim=0))
        
        # Save cache
        if self.use_cache:
            cache_data = {key: getattr(self, key) for key in all_tensors.keys()}
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved dataset cache: {cache_file}")
    
    def _process_batch(self, batch_samples):
        """Process a batch of samples efficiently with temporal information"""
        
        batch_size = len(batch_samples)
        
        # Pre-allocate arrays
        user_ids = np.zeros(batch_size, dtype=np.int32)  # Changed from int64 to int32 to match tensor type
        has_short_term = np.zeros(batch_size, dtype=bool)
        is_purchase = np.zeros(batch_size, dtype=bool)
        
        # Pre-computed event flags for consistent evaluation
        has_checkout = np.zeros(batch_size, dtype=bool)
        has_add_to_cart = np.zeros(batch_size, dtype=bool)
        is_cold_start = np.zeros(batch_size, dtype=bool)
        
        # Inclusive event flags (for individual event recalls)
        has_checkout_inclusive = np.zeros(batch_size, dtype=bool)
        has_add_to_cart_inclusive = np.zeros(batch_size, dtype=bool)
        
        # Short-term arrays - use int32 for indices
        st_packages = np.zeros((batch_size, self.max_short_term), dtype=np.int32)
        st_events = np.zeros((batch_size, self.max_short_term), dtype=np.int32)
        st_countries = np.zeros((batch_size, self.max_short_term), dtype=np.int32)
        st_categories = np.zeros((batch_size, self.max_short_term), dtype=np.int32)
        st_themes = np.zeros((batch_size, self.max_short_term), dtype=np.int32)
        st_prices = np.zeros((batch_size, self.max_short_term), dtype=np.float32)
        st_timestamps = np.zeros((batch_size, self.max_short_term), dtype=np.float32)
        
        # Long-term arrays - use int32 for indices
        lt_packages = np.zeros((batch_size, self.max_long_term), dtype=np.int32)
        lt_events = np.zeros((batch_size, self.max_long_term), dtype=np.int32)
        lt_countries = np.zeros((batch_size, self.max_long_term), dtype=np.int32)
        lt_categories = np.zeros((batch_size, self.max_long_term), dtype=np.int32)
        lt_themes = np.zeros((batch_size, self.max_long_term), dtype=np.int32)
        lt_prices = np.zeros((batch_size, self.max_long_term), dtype=np.float32)
        lt_timestamps = np.zeros((batch_size, self.max_long_term), dtype=np.float32)
        
        # Purchased arrays - use int32 to avoid overflow issues
        purchased_packages = np.zeros(batch_size, dtype=np.int32)
        purchased_countries = np.zeros(batch_size, dtype=np.int32)
        purchased_categories = np.zeros(batch_size, dtype=np.int32)
        purchased_themes = np.zeros(batch_size, dtype=np.int32)
        purchased_prices = np.zeros(batch_size, dtype=np.float32)
        purchased_timestamps = np.zeros(batch_size, dtype=np.float32)
        
        # Maximum price value for numerical stability
        max_price_value = 10000.0  # €10,000 cap for numerical stability
        
        # Process samples
        for i, sample in enumerate(batch_samples):
            # User info
            user_id = sample['user_id']
            mapped_id = self.user_to_idx.get(user_id, -1)
            if mapped_id == -1:
                if not hasattr(self, '_unmapped_users_logged'):
                    self._unmapped_users_logged = set()
                if user_id not in self._unmapped_users_logged:
                    print(f"WARNING: Unmapped user_id: {user_id}")
                    self._unmapped_users_logged.add(user_id)
                mapped_id = 0  # Use padding index
            user_ids[i] = mapped_id
            has_short_term[i] = len(sample.get('short_term_packages', [])) > 0
            is_purchase[i] = sample.get('is_purchase', False)
            
            # Extract pre-computed event flags for consistent evaluation
            has_checkout[i] = sample.get('has_checkout', False)
            has_add_to_cart[i] = sample.get('has_add_to_cart', False)
            is_cold_start[i] = sample.get('is_cold_start', False)
            
            # Extract inclusive flags for individual event recalls
            has_checkout_inclusive[i] = sample.get('has_checkout_inclusive', False)
            has_add_to_cart_inclusive[i] = sample.get('has_add_to_cart_inclusive', False)
            
            # Process short-term
            if has_short_term[i]:
                st_pkg_ids = sample['short_term_packages'][-self.max_short_term:]
                st_event_ids = sample.get('short_term_events', [])[-self.max_short_term:]
                st_times = sample.get('short_term_timestamps', [0] * len(st_pkg_ids))[-self.max_short_term:]
                
                for j, (pkg_id, event_id, timestamp) in enumerate(zip(st_pkg_ids, st_event_ids, st_times)):
                    pkg_idx = self.package_to_idx.get(str(pkg_id), 0)
                    st_packages[i, j] = pkg_idx
                    st_events[i, j] = event_id
                    st_timestamps[i, j] = timestamp
                    
                    # Get features efficiently
                    features = self._get_package_features_fast(str(pkg_id))
                    st_countries[i, j] = features[0]
                    st_categories[i, j] = features[1]
                    st_themes[i, j] = features[2]
                    # Clamp price for numerical stability
                    st_prices[i, j] = min(features[3], max_price_value)
            
            # Process long-term (similar logic)
            lt_pkg_ids = sample.get('long_term_packages', [])[-self.max_long_term:]
            if lt_pkg_ids:
                lt_event_ids = sample.get('long_term_events', [])[-self.max_long_term:]
                lt_times = sample.get('long_term_timestamps', [0] * len(lt_pkg_ids))[-self.max_long_term:]
                
                for j, (pkg_id, event_id, timestamp) in enumerate(zip(lt_pkg_ids, lt_event_ids, lt_times)):
                    pkg_idx = self.package_to_idx.get(str(pkg_id), 0)
                    lt_packages[i, j] = pkg_idx
                    lt_events[i, j] = event_id
                    lt_timestamps[i, j] = timestamp
                    
                    features = self._get_package_features_fast(str(pkg_id))
                    lt_countries[i, j] = features[0]
                    lt_categories[i, j] = features[1]
                    lt_themes[i, j] = features[2]
                    # Clamp price for numerical stability
                    lt_prices[i, j] = min(features[3], max_price_value)
            
            # Process purchased
            purchased_id = str(sample['purchased_package'])
            purchased_packages[i] = self.package_to_idx.get(purchased_id, 0)
            
            # Get purchase timestamp if available
            purchased_timestamps[i] = sample.get('purchased_timestamp', 0)
            
            features = self._get_package_features_fast(purchased_id)
            purchased_countries[i] = features[0]
            purchased_categories[i] = features[1]
            purchased_themes[i] = features[2]
            # Clamp price for numerical stability
            purchased_prices[i] = min(features[3], max_price_value)
        
        # Convert to tensors
        return {
            'user_ids': torch.from_numpy(user_ids.astype(np.int32)),  # Ensure int32 for compatibility
            'has_short_term': torch.from_numpy(has_short_term),
            'is_purchase': torch.from_numpy(is_purchase),
            'has_checkout': torch.from_numpy(has_checkout),
            'has_add_to_cart': torch.from_numpy(has_add_to_cart),
            'is_cold_start': torch.from_numpy(is_cold_start),
            'has_checkout_inclusive': torch.from_numpy(has_checkout_inclusive),
            'has_add_to_cart_inclusive': torch.from_numpy(has_add_to_cart_inclusive),
            'short_term_packages': torch.from_numpy(st_packages),
            'short_term_events': torch.from_numpy(st_events),
            'short_term_countries': torch.from_numpy(st_countries),
            'short_term_categories': torch.from_numpy(st_categories),
            'short_term_themes': torch.from_numpy(st_themes),
            'short_term_prices': torch.from_numpy(st_prices),
            'short_term_timestamps': torch.from_numpy(st_timestamps),
            'long_term_packages': torch.from_numpy(lt_packages),
            'long_term_events': torch.from_numpy(lt_events),
            'long_term_countries': torch.from_numpy(lt_countries),
            'long_term_categories': torch.from_numpy(lt_categories),
            'long_term_themes': torch.from_numpy(lt_themes),
            'long_term_prices': torch.from_numpy(lt_prices),
            'long_term_timestamps': torch.from_numpy(lt_timestamps),
            'purchased_packages': torch.from_numpy(purchased_packages),
            'purchased_countries': torch.from_numpy(purchased_countries),
            'purchased_categories': torch.from_numpy(purchased_categories),
            'purchased_themes': torch.from_numpy(purchased_themes),
            'purchased_prices': torch.from_numpy(purchased_prices),
            'purchased_timestamps': torch.from_numpy(purchased_timestamps)
        }
    
    @lru_cache(maxsize=50000)
    def _get_package_features_fast(self, pkg_id: str) -> tuple:
        """Fast cached package feature lookup"""
        features = self.package_processor.get_package_features(pkg_id)
        if features:
            return (
                features['country_idx'],
                features['category_idx'],
                features['theme_idx'],
                features['price']
            )
        return (0, 0, 0, 0.0)
    
    def _prefetch_common_features(self):
        """Prefetch features for frequently accessed packages"""
        print("Prefetching common package features...")
        
        # Safety check: ensure short_term_packages exists
        if not hasattr(self, 'short_term_packages'):
            print("Warning: short_term_packages not found, skipping prefetch")
            return
        
        # Get most common packages
        package_counts = Counter()
        for i in range(len(self.short_term_packages)):
            for pkg_idx in self.short_term_packages[i]:
                if pkg_idx > 0:
                    package_counts[pkg_idx.item()] += 1
        
        # Prefetch top packages
        top_packages = package_counts.most_common(1000)
        for pkg_idx, _ in top_packages:
            main_id = self.idx_to_main_id.get(pkg_idx)
            if main_id:
                self._get_package_features_fast(str(main_id))
    
    def __getitem__(self, idx):
        """Optimized getitem with minimal computation, including temporal information"""
        # Use pre-computed indices
        user_id = self.user_ids[idx]
        
        # Debug check for invalid user IDs
        if hasattr(self, '_debug_user_ids') and idx < 10:  # Only debug first 10
            if user_id < 0 or user_id > 1000000:  # Suspicious values
                print(f"[DEBUG] Suspicious user_id at idx {idx}: {user_id} (type: {type(user_id)}, item: {user_id.item() if hasattr(user_id, 'item') else user_id})")
        
        # Build sample efficiently
        sample = {
            'user_id': user_id,
            'has_short_term': self.has_short_term[idx],
            'is_purchase': self.is_purchase[idx],
            'has_checkout': self.has_checkout[idx] if hasattr(self, 'has_checkout') else False,
            'has_add_to_cart': self.has_add_to_cart[idx] if hasattr(self, 'has_add_to_cart') else False,
            'is_cold_start': self.is_cold_start[idx] if hasattr(self, 'is_cold_start') else False,
            'has_checkout_inclusive': self.has_checkout_inclusive[idx] if hasattr(self, 'has_checkout_inclusive') else False,
            'has_add_to_cart_inclusive': self.has_add_to_cart_inclusive[idx] if hasattr(self, 'has_add_to_cart_inclusive') else False,
            
            'short_term': {
                'package_ids': self.short_term_packages[idx],
                'event_types': self.short_term_events[idx],
                'title_embeddings': self._get_embeddings_fast(self.short_term_packages[idx], 'short'),
                'coordinates': self._get_coordinates_fast(self.short_term_packages[idx], 'short'),
                'country_ids': self.short_term_countries[idx],
                'category_ids': self.short_term_categories[idx],
                'theme_ids': self.short_term_themes[idx],
                'prices': self.short_term_prices[idx],
                'timestamps': self.short_term_timestamps[idx] if hasattr(self, 'short_term_timestamps') else None
            },
            
            'long_term': {
                'package_ids': self.long_term_packages[idx],
                'event_types': self.long_term_events[idx],
                'title_embeddings': self._get_embeddings_fast(self.long_term_packages[idx], 'long'),
                'coordinates': self._get_coordinates_fast(self.long_term_packages[idx], 'long'),
                'country_ids': self.long_term_countries[idx],
                'category_ids': self.long_term_categories[idx],
                'theme_ids': self.long_term_themes[idx],
                'prices': self.long_term_prices[idx],
                'timestamps': self.long_term_timestamps[idx] if hasattr(self, 'long_term_timestamps') else None
            },
            
            'purchased': {
                'package_ids': self.purchased_packages[idx],
                'title_embeddings': self._get_single_embedding_fast(self.purchased_packages[idx]),
                'coordinates': self._get_single_coordinates_fast(self.purchased_packages[idx]),
                'country_ids': self.purchased_countries[idx],
                'category_ids': self.purchased_categories[idx],
                'theme_ids': self.purchased_themes[idx],
                'prices': self.purchased_prices[idx],
                'timestamps': self.purchased_timestamps[idx] if hasattr(self, 'purchased_timestamps') else None
            }
        }
        
        return sample
    
    def _get_embeddings_fast(self, package_ids, seq_type):
        """Vectorized embedding retrieval"""
        max_len = self.max_short_term if seq_type == 'short' else self.max_long_term
        embeddings = torch.zeros(max_len, self.embedding_dim)
        
        # Get feature indices for all packages at once
        feature_indices = []
        valid_positions = []
        
        for i, pkg_idx in enumerate(package_ids):
            if pkg_idx > 0:
                main_id = self.idx_to_main_id.get(pkg_idx.item())
                if main_id:
                    feature_idx = self.package_features['package_to_idx'].get(str(main_id))
                    if feature_idx is not None:
                        feature_indices.append(feature_idx)
                        valid_positions.append(i)
        
        # Vectorized assignment
        if feature_indices:
            embeddings[valid_positions] = self.package_features['title_embeddings'][feature_indices]
        
        return embeddings
    
    def _get_coordinates_fast(self, package_ids, seq_type):
        """Vectorized coordinate retrieval"""
        max_len = self.max_short_term if seq_type == 'short' else self.max_long_term
        coordinates = torch.zeros(max_len, 2)
        
        # Similar vectorized logic as embeddings
        feature_indices = []
        valid_positions = []
        
        for i, pkg_idx in enumerate(package_ids):
            if pkg_idx > 0:
                main_id = self.idx_to_main_id.get(pkg_idx.item())
                if main_id:
                    feature_idx = self.package_features['package_to_idx'].get(str(main_id))
                    if feature_idx is not None:
                        feature_indices.append(feature_idx)
                        valid_positions.append(i)
        
        if feature_indices:
            coordinates[valid_positions] = self.package_features['coordinates'][feature_indices]
        
        return coordinates
    
    @lru_cache(maxsize=10000)
    def _get_single_embedding_fast(self, pkg_idx):
        """Cached single embedding retrieval"""
        if pkg_idx <= 0:
            return torch.zeros(self.embedding_dim)
        
        main_id = self.idx_to_main_id.get(pkg_idx.item() if torch.is_tensor(pkg_idx) else pkg_idx)
        if main_id:
            feature_idx = self.package_features['package_to_idx'].get(str(main_id))
            if feature_idx is not None:
                return self.package_features['title_embeddings'][feature_idx]
        
        return torch.zeros(self.embedding_dim)
    
    @lru_cache(maxsize=10000)
    def _get_single_coordinates_fast(self, pkg_idx):
        """Cached single coordinate retrieval"""
        if pkg_idx <= 0:
            return torch.zeros(2)
        
        main_id = self.idx_to_main_id.get(pkg_idx.item() if torch.is_tensor(pkg_idx) else pkg_idx)
        if main_id:
            feature_idx = self.package_features['package_to_idx'].get(str(main_id))
            if feature_idx is not None:
                return self.package_features['coordinates'][feature_idx]
        
        return torch.zeros(2)
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import pickle
import time
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
    def __init__(self, data_path=None, cache_dir='data/cache', 
                 load_coordinates=True, 
                 load_embeddings=True,
                 api_key=None,
                 embedding_model='text-embedding-3-large',
                 use_reduced_embeddings=True):
        """
        Initialize PackageProcessor
        
        Args:
            data_path (str): Path to the package data file
            cache_dir (str): Directory for caching processed data
            load_coordinates (bool): Whether to load coordinates from geocoder cache
            load_embeddings (bool): Whether to generate title embeddings
            api_key (str): OpenAI API key for embeddings
            embedding_model (str): OpenAI embedding model to use
        """
        self.data_path = data_path
        self.cache_dir = cache_dir
        self.load_coordinates = load_coordinates
        self.load_embeddings = load_embeddings
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.embedding_model = embedding_model
        self.use_reduced_embeddings = use_reduced_embeddings
        
        # Set embedding dimension based on use_reduced_embeddings
        self.embedding_dim = 768 if use_reduced_embeddings else 3072
        
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
        """Clear all cached files in the cache directory related to package data"""
        try:
            cache_files = [
                os.path.join(self.cache_dir, 'package_metadata.pkl'),
                os.path.join(self.cache_dir, 'package_mappings.pkl'),
                os.path.join(self.cache_dir, 'package_features.pkl')  # New cache for features
            ]
            
            for file in cache_files:
                if os.path.exists(file):
                    os.remove(file)
                    print(f"Removed cache file: {file}")
            
            print(f"Cleared package data cache")
        except Exception as e:
            print(f"Error clearing cache: {e}")
    
    def _load_coordinates_from_geocoder(self):
        """Load coordinates from geocoder_dutch.py cache - FIXED VERSION"""
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
            print("No geocoder cache found. Run geocoder_dutch.py first to geocode packages.")
            self.package_coordinates = {}

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
    
    def load_data(self, use_cache=True):
        """Load package data with caching for efficiency - FIXED VERSION"""
        cache_file = os.path.join(self.cache_dir, 'package_metadata.pkl')
        
        # Check and load cache
        if use_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    cached_data_path = cache_data.get('data_path')
                    
                    if cached_data_path != self.data_path:
                        print(f"Dataset path changed. Clearing previous cache.")
                        self.clear_cache()
                    else:
                        # Load cached data
                        self.package_metadata = cache_data['package_metadata']
                        self.df = cache_data.get('df')
                        
                        # IMPORTANT: Load coordinates if requested and add to metadata
                        if self.load_coordinates:
                            self._load_coordinates_from_geocoder()
                            # Add coordinates to metadata if not already there
                            for main_id_str in self.package_metadata:
                                if main_id_str in self.package_coordinates:
                                    coord_data = self.package_coordinates[main_id_str]
                                    if 'latitude' not in self.package_metadata[main_id_str]:
                                        self.package_metadata[main_id_str]['latitude'] = coord_data.get('latitude')
                                        self.package_metadata[main_id_str]['longitude'] = coord_data.get('longitude')
                        
                        # IMPORTANT: Load/generate embeddings if requested and add to metadata
                        if self.load_embeddings:
                            self._generate_embeddings()
                            # Add embeddings to metadata if not already there
                            for main_id_str in self.package_metadata:
                                if main_id_str in self.package_embeddings:
                                    if 'title_embedding' not in self.package_metadata[main_id_str]:
                                        self.package_metadata[main_id_str]['title_embedding'] = self.package_embeddings[main_id_str]
                        
                        print(f"Loaded metadata for {len(self.package_metadata)} packages from cache")
                        
                        # Report statistics
                        with_coords = sum(1 for m in self.package_metadata.values() 
                                        if 'latitude' in m and m['latitude'] is not None)
                        with_embeddings = sum(1 for m in self.package_metadata.values() 
                                            if 'title_embedding' in m and m['title_embedding'] is not None)
                        print(f"Packages with coordinates: {with_coords} ({with_coords/len(self.package_metadata)*100:.1f}%)")
                        print(f"Packages with embeddings: {with_embeddings} ({with_embeddings/len(self.package_metadata)*100:.1f}%)")
                        
                        return
            except Exception as e:
                print(f"Cache loading error: {e}. Clearing cache and reloading.")
                self.clear_cache()
        
        # Load data from source
        if self.data_path:
            print(f"Loading package data from {self.data_path}")
            
            # Load file based on extension
            try:
                if self.data_path.endswith('.parquet'):
                    self.df = pd.read_parquet(self.data_path)
                else:
                    self.df = pd.read_csv(self.data_path)
                print(f"Loaded {len(self.df)} packages")
            except Exception as e:
                print(f"Error loading file: {e}")
                return
            
            # Process package metadata
            self._process_package_metadata()
            
            # Save to cache
            cache_data = {
                'package_metadata': self.package_metadata,
                'df': self.df,
                'data_path': self.data_path
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"Saved processed data to cache")


    def _process_package_metadata(self):
        """Process package metadata from the dataframe - FIXED VERSION"""
        print("Processing package metadata...")
        
        # First ensure coordinates are loaded
        if self.load_coordinates and not self.package_coordinates:
            self._load_coordinates_from_geocoder()
        
        self.package_metadata = {}
        
        # Process each row
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing packages"):
            main_id_str = str(row['main_id'])  # ALWAYS use string for consistency
            
            # Store all metadata for NATR model
            metadata = {
                'title': row.get('title', ''),
                'city': row.get('city', ''),
                'country': row.get('country', ''),
                'category': row.get('category', ''),
                'theme': row.get('theme', ''),
                'min_price': float(row.get('min_price', 0))
            }
            
            # Add coordinates if available - check with string key
            if main_id_str in self.package_coordinates:
                coord_data = self.package_coordinates[main_id_str]
                metadata['latitude'] = coord_data.get('latitude')
                metadata['longitude'] = coord_data.get('longitude')
            
            self.package_metadata[main_id_str] = metadata
        
        # Generate embeddings after all metadata is loaded
        if self.load_embeddings:
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
            'prices': []
        }
        
        for main_id in main_ids:
            features = self.get_package_features(main_id)
            
            if features:
                # Title embedding
                if features['title_embedding'] is not None:
                    batch_features['title_embeddings'].append(features['title_embedding'])
                else:
                    # Use zero embedding if not available
                    batch_features['title_embeddings'].append(np.zeros(768))  # Adjust size as needed
                
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
                
                # Price
                batch_features['prices'].append(features['price'])
            else:
                # Handle missing packages
                batch_features['title_embeddings'].append(np.zeros(768))
                batch_features['coordinates'].append([0.0, 0.0])
                batch_features['country_ids'].append(0)
                batch_features['city_ids'].append(0)
                batch_features['category_ids'].append(0)
                batch_features['theme_ids'].append(0)
                batch_features['prices'].append(0.0)
        
        # Convert to tensors
        return {
            'title_embeddings': torch.tensor(np.array(batch_features['title_embeddings']), dtype=torch.float),
            'coordinates': torch.tensor(batch_features['coordinates'], dtype=torch.float),
            'country_ids': torch.tensor(batch_features['country_ids'], dtype=torch.long),
            'city_ids': torch.tensor(batch_features['city_ids'], dtype=torch.long),
            'category_ids': torch.tensor(batch_features['category_ids'], dtype=torch.long),
            'theme_ids': torch.tensor(batch_features['theme_ids'], dtype=torch.long),
            'prices': torch.tensor(batch_features['prices'], dtype=torch.float)
        }
    
    def prepare_package_tensors(self, use_cache=True) -> Dict[str, torch.Tensor]:
        """
        Prepare all package features as tensors for efficient model usage
        
        Returns:
            Dictionary of tensors for all packages
        """
        cache_file = os.path.join(self.cache_dir, 'package_features.pkl')
        
        if use_cache and os.path.exists(cache_file):
            print("Loading cached package feature tensors")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        print("Preparing package feature tensors...")
        
        if not self.package_metadata:
            self.load_data()
        
        # Ensure mappings are created
        if not self.country_to_idx:
            self.create_mappings()
        
        # Get all package IDs sorted by index - ensure strings
        package_to_idx = {str(main_id): idx for idx, main_id in enumerate(sorted(self.package_metadata.keys()))}
        num_packages = len(package_to_idx)
        
        # Initialize tensors - FIXED: Use correct embedding dimension
        # First check what dimension embeddings actually have
        sample_metadata = next(iter(self.package_metadata.values()))
        if 'title_embedding' in sample_metadata and sample_metadata['title_embedding'] is not None:
            embedding_dim = sample_metadata['title_embedding'].shape[0]
            print(f"Detected embedding dimension: {embedding_dim}")
        else:
            embedding_dim = 3072  # Default for text-embedding-3-large
            print(f"Using default embedding dimension: {embedding_dim}")
        
        title_embeddings = np.zeros((num_packages, embedding_dim))
        coordinates = np.zeros((num_packages, 2))  # lat, lon
        country_ids = np.zeros(num_packages, dtype=np.int64)
        city_ids = np.zeros(num_packages, dtype=np.int64)
        category_ids = np.zeros(num_packages, dtype=np.int64)
        theme_ids = np.zeros(num_packages, dtype=np.int64)
        prices = np.zeros(num_packages, dtype=np.float32)
        
        # Fill tensors
        for main_id_str, idx in tqdm(package_to_idx.items(), desc="Creating feature tensors"):
            features = self.get_package_features(main_id_str)
            
            if features:
                # Title embedding
                if features['title_embedding'] is not None:
                    title_embeddings[idx] = features['title_embedding']
                
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
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(feature_tensors, f)
        
        print(f"Prepared feature tensors for {num_packages} packages")
        return feature_tensors
    
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
                self.city_to_idx = mappings.get('city_to_idx', {})
            return
        
        print("Creating package attribute mappings...")
        
        if not self.package_metadata:
            self.load_data(use_cache=use_cache)
        
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
        
        # Save mappings
        mappings = {
            'country_to_idx': self.country_to_idx,
            'city_to_idx': self.city_to_idx,
            'category_to_idx': self.category_to_idx,
            'theme_to_idx': self.theme_to_idx
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(mappings, f)

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
                use_cache: bool = True,
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
        """Process a batch of samples efficiently"""
        
        batch_size = len(batch_samples)
        
        # Pre-allocate arrays
        user_ids = np.zeros(batch_size, dtype=np.int64)
        has_short_term = np.zeros(batch_size, dtype=bool)
        is_purchase = np.zeros(batch_size, dtype=bool)
        
        # Short-term arrays
        st_packages = np.zeros((batch_size, self.max_short_term), dtype=np.int64)
        st_events = np.zeros((batch_size, self.max_short_term), dtype=np.int64)
        st_countries = np.zeros((batch_size, self.max_short_term), dtype=np.int64)
        st_categories = np.zeros((batch_size, self.max_short_term), dtype=np.int64)
        st_themes = np.zeros((batch_size, self.max_short_term), dtype=np.int64)
        st_prices = np.zeros((batch_size, self.max_short_term), dtype=np.float32)
        
        # Long-term arrays (similar)
        lt_packages = np.zeros((batch_size, self.max_long_term), dtype=np.int64)
        lt_events = np.zeros((batch_size, self.max_long_term), dtype=np.int64)
        lt_countries = np.zeros((batch_size, self.max_long_term), dtype=np.int64)
        lt_categories = np.zeros((batch_size, self.max_long_term), dtype=np.int64)
        lt_themes = np.zeros((batch_size, self.max_long_term), dtype=np.int64)
        lt_prices = np.zeros((batch_size, self.max_long_term), dtype=np.float32)
        
        # Purchased arrays
        purchased_packages = np.zeros(batch_size, dtype=np.int64)
        purchased_countries = np.zeros(batch_size, dtype=np.int64)
        purchased_categories = np.zeros(batch_size, dtype=np.int64)
        purchased_themes = np.zeros(batch_size, dtype=np.int64)
        purchased_prices = np.zeros(batch_size, dtype=np.float32)
        
        # Process samples
        for i, sample in enumerate(batch_samples):
            # User info
            user_ids[i] = self.user_to_idx.get(sample['user_id'], 0)
            has_short_term[i] = len(sample.get('short_term_packages', [])) > 0
            is_purchase[i] = sample.get('is_purchase', False)
            
            # Process short-term
            if has_short_term[i]:
                st_pkg_ids = sample['short_term_packages'][-self.max_short_term:]
                st_event_ids = sample.get('short_term_events', [])[-self.max_short_term:]
                
                for j, (pkg_id, event_id) in enumerate(zip(st_pkg_ids, st_event_ids)):
                    pkg_idx = self.package_to_idx.get(str(pkg_id), 0)
                    st_packages[i, j] = pkg_idx
                    st_events[i, j] = event_id
                    
                    # Get features efficiently
                    features = self._get_package_features_fast(str(pkg_id))
                    st_countries[i, j] = features[0]
                    st_categories[i, j] = features[1]
                    st_themes[i, j] = features[2]
                    st_prices[i, j] = features[3]
            
            # Process long-term (similar logic)
            lt_pkg_ids = sample.get('long_term_packages', [])[-self.max_long_term:]
            if lt_pkg_ids:
                lt_event_ids = sample.get('long_term_events', [])[-self.max_long_term:]
                
                for j, (pkg_id, event_id) in enumerate(zip(lt_pkg_ids, lt_event_ids)):
                    pkg_idx = self.package_to_idx.get(str(pkg_id), 0)
                    lt_packages[i, j] = pkg_idx
                    lt_events[i, j] = event_id
                    
                    features = self._get_package_features_fast(str(pkg_id))
                    lt_countries[i, j] = features[0]
                    lt_categories[i, j] = features[1]
                    lt_themes[i, j] = features[2]
                    lt_prices[i, j] = features[3]
            
            # Process purchased
            purchased_id = str(sample['purchased_package'])
            purchased_packages[i] = self.package_to_idx.get(purchased_id, 0)
            
            features = self._get_package_features_fast(purchased_id)
            purchased_countries[i] = features[0]
            purchased_categories[i] = features[1]
            purchased_themes[i] = features[2]
            purchased_prices[i] = features[3]
        
        # Convert to tensors
        return {
            'user_ids': torch.from_numpy(user_ids),
            'has_short_term': torch.from_numpy(has_short_term),
            'is_purchase': torch.from_numpy(is_purchase),
            'short_term_packages': torch.from_numpy(st_packages),
            'short_term_events': torch.from_numpy(st_events),
            'short_term_countries': torch.from_numpy(st_countries),
            'short_term_categories': torch.from_numpy(st_categories),
            'short_term_themes': torch.from_numpy(st_themes),
            'short_term_prices': torch.from_numpy(st_prices),
            'long_term_packages': torch.from_numpy(lt_packages),
            'long_term_events': torch.from_numpy(lt_events),
            'long_term_countries': torch.from_numpy(lt_countries),
            'long_term_categories': torch.from_numpy(lt_categories),
            'long_term_themes': torch.from_numpy(lt_themes),
            'long_term_prices': torch.from_numpy(lt_prices),
            'purchased_packages': torch.from_numpy(purchased_packages),
            'purchased_countries': torch.from_numpy(purchased_countries),
            'purchased_categories': torch.from_numpy(purchased_categories),
            'purchased_themes': torch.from_numpy(purchased_themes),
            'purchased_prices': torch.from_numpy(purchased_prices)
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
        """Optimized getitem with minimal computation"""
        # Use pre-computed indices
        user_id = self.user_ids[idx]
        
        # Build sample efficiently
        sample = {
            'user_id': user_id,
            'has_short_term': self.has_short_term[idx],
            'is_purchase': self.is_purchase[idx],
            
            'short_term': {
                'package_ids': self.short_term_packages[idx],
                'event_types': self.short_term_events[idx],
                'title_embeddings': self._get_embeddings_fast(self.short_term_packages[idx], 'short'),
                'coordinates': self._get_coordinates_fast(self.short_term_packages[idx], 'short'),
                'country_ids': self.short_term_countries[idx],
                'category_ids': self.short_term_categories[idx],
                'theme_ids': self.short_term_themes[idx],
                'prices': self.short_term_prices[idx]
            },
            
            'long_term': {
                'package_ids': self.long_term_packages[idx],
                'event_types': self.long_term_events[idx],
                'title_embeddings': self._get_embeddings_fast(self.long_term_packages[idx], 'long'),
                'coordinates': self._get_coordinates_fast(self.long_term_packages[idx], 'long'),
                'country_ids': self.long_term_countries[idx],
                'category_ids': self.long_term_categories[idx],
                'theme_ids': self.long_term_themes[idx],
                'prices': self.long_term_prices[idx]
            },
            
            'purchased': {
                'package_ids': self.purchased_packages[idx],
                'title_embeddings': self._get_single_embedding_fast(self.purchased_packages[idx]),
                'coordinates': self._get_single_coordinates_fast(self.purchased_packages[idx]),
                'country_ids': self.purchased_countries[idx],
                'category_ids': self.purchased_categories[idx],
                'theme_ids': self.purchased_themes[idx],
                'prices': self.purchased_prices[idx]
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
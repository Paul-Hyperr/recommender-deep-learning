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
                 embedding_model='text-embedding-3-large'):
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
                    cache_dir=os.path.join(self.cache_dir, 'llm_embeddings')
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
    Enhanced Dataset for NATR travel package recommendation
    
    Features:
    - Handles package features: title embeddings, coordinates, categories (no cities!), prices
    - Incorporates event types from user sessions
    - Supports both purchase and non-purchase samples
    - Caches prepared samples for faster loading
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
                use_cache: bool = True):
        """
        Initialize the dataset for NATR model with caching support
        
        Args:
            samples: List of training/testing samples
            package_processor: PackageProcessor instance with features
            user_to_idx: Mapping from user ID to index
            package_to_idx: Mapping from package ID to index
            event_to_idx: Mapping from event type to index
            max_short_term: Maximum length of short-term sequence
            max_long_term: Maximum length of long-term sequence
            empty_token: Special token for empty values
            unknown_token: Token for unknown category values
            use_cache: Whether to use cached prepared samples
        """
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

        # Get package mappings from processor (excluding city_to_idx)
        mappings = self.package_processor.get_idx_mappings()
        self.country_to_idx = mappings['country_to_idx']
        self.category_to_idx = mappings['category_to_idx']
        self.theme_to_idx = mappings['theme_to_idx']
        # NOTE: We don't use city_to_idx since we have coordinates
        
        # Load all package features as tensors for efficiency
        self.package_features = self.package_processor.prepare_package_tensors()
        
        # Store embedding dimension
        if 'title_embeddings' in self.package_features:
            self.embedding_dim = self.package_features['title_embeddings'].shape[1]
        else:
            self.embedding_dim = 3072  # Default for text-embedding-3-large
        
        # Store original sample information
        self.original_user_ids = []
        self.has_short_term = []
        self.is_purchase = []
        self.session_ids = []
        
        # Pre-process samples (with caching)
        if self.use_cache:
            self._prepare_samples_with_cache()
        else:
            self._prepare_samples()
    
    def __len__(self):
        """Return the size of the dataset"""
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        sample = {
            'user_id': self.user_ids[idx],
            'original_user_id': self.original_user_ids[idx],
            'has_short_term': self.has_short_term[idx],
            
            # Short-term features (NO CITIES)
            'short_term': {
                'package_ids': self.short_term_packages[idx],
                'event_types': self.short_term_events[idx],
                'title_embeddings': self.get_sequence_embeddings(self.short_term_packages[idx], 'short'),
                'coordinates': self.get_sequence_coordinates(self.short_term_packages[idx], 'short'),
                'country_ids': self.short_term_countries[idx],
                'category_ids': self.short_term_categories[idx],
                'theme_ids': self.short_term_themes[idx],
                'prices': self.short_term_prices[idx]
            },
            
            # Long-term features (NO CITIES)
            'long_term': {
                'package_ids': self.long_term_packages[idx],
                'event_types': self.long_term_events[idx],
                'title_embeddings': self.get_sequence_embeddings(self.long_term_packages[idx], 'long'),
                'coordinates': self.get_sequence_coordinates(self.long_term_packages[idx], 'long'),
                'country_ids': self.long_term_countries[idx],
                'category_ids': self.long_term_categories[idx],
                'theme_ids': self.long_term_themes[idx],
                'prices': self.long_term_prices[idx]
            },
            
            # Purchased/target package features (NO CITIES)
            'purchased': {
                'package_id': self.purchased_packages[idx],
                'title_embedding': self.get_package_embedding(self.purchased_packages[idx]),
                'coordinates': self.get_package_coordinates(self.purchased_packages[idx]),
                'country_id': self.purchased_countries[idx],
                'category_id': self.purchased_categories[idx],
                'theme_id': self.purchased_themes[idx],
                'price': self.purchased_prices[idx]
            }
        }
        
        # Add optional fields
        if isinstance(self.is_purchase, torch.Tensor):
            sample['is_purchase'] = self.is_purchase[idx]
        
        if isinstance(self.session_ids, torch.Tensor):
            sample['session_id'] = self.session_ids[idx]
        
        return sample

    # Fixed TravelPackageDataset caching method

    def _prepare_samples_with_cache(self):
        """Pre-process samples with caching for faster subsequent runs - FIXED VERSION"""
        # Create a hash of the samples to ensure cache validity
        sample_info = f"{len(self.samples)}_{self.max_short_term}_{self.max_long_term}"
        sample_hash = hashlib.md5(sample_info.encode()).hexdigest()[:8]
        cache_file = os.path.join(
            self.package_processor.cache_dir, 
            f'prepared_dataset_samples_{sample_hash}.pkl'
        )
        
        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                print(f"Loading pre-processed samples from cache...")
                for key, value in cached_data.items():
                    setattr(self, key, value)
                print(f"Loaded {len(self.user_ids)} cached samples")
                return
            except Exception as e:
                print(f"Cache loading failed: {e}")
        
        # Process normally
        self._prepare_samples()
        
        # Save to cache - FIXED to use pickle only, no JSON
        cache_data = {
            'user_ids': self.user_ids,
            'original_user_ids': self.original_user_ids,
            'has_short_term': self.has_short_term,
            'is_purchase': self.is_purchase if hasattr(self, 'is_purchase') else [],
            'session_ids': self.session_ids if hasattr(self, 'session_ids') else [],
            'short_term_packages': self.short_term_packages,
            'short_term_countries': self.short_term_countries,
            'short_term_categories': self.short_term_categories,
            'short_term_themes': self.short_term_themes,
            'short_term_prices': self.short_term_prices,
            'short_term_events': self.short_term_events,
            'long_term_packages': self.long_term_packages,
            'long_term_countries': self.long_term_countries,
            'long_term_categories': self.long_term_categories,
            'long_term_themes': self.long_term_themes,
            'long_term_prices': self.long_term_prices,
            'long_term_events': self.long_term_events,
            'purchased_packages': self.purchased_packages,
            'purchased_countries': self.purchased_countries,
            'purchased_categories': self.purchased_categories,
            'purchased_themes': self.purchased_themes,
            'purchased_prices': self.purchased_prices,
        }
        
        try:
            # Just use pickle - no JSON metadata that could cause serialization issues
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Saved prepared samples to cache: {cache_file}")
        except Exception as e:
            print(f"Failed to save cache: {e}")
            # Continue anyway - caching is optional
    
    def _prepare_samples(self):
        """Pre-process samples for faster retrieval with all features"""
        print("Pre-processing samples for NATR dataset...")
        
        # Initialize lists
        self.user_ids = []
        self.original_user_ids = []
        self.has_short_term = []
        self.is_purchase = []
        self.session_ids = []
        
        # Short-term sequences (NO CITIES)
        self.short_term_packages = []
        self.short_term_countries = []
        self.short_term_categories = []
        self.short_term_themes = []
        self.short_term_prices = []
        self.short_term_events = []
        
        # Long-term sequences (NO CITIES)
        self.long_term_packages = []
        self.long_term_countries = []
        self.long_term_categories = []
        self.long_term_themes = []
        self.long_term_prices = []
        self.long_term_events = []
        
        # Purchased/target package (NO CITIES)
        self.purchased_packages = []
        self.purchased_countries = []
        self.purchased_categories = []
        self.purchased_themes = []
        self.purchased_prices = []
        
        # Statistics
        empty_short_term_count = 0
        empty_long_term_count = 0
        
        # Process each sample
        for sample in tqdm(self.samples, desc="Preparing NATR dataset"):
            # User information
            user_id = sample['user_id']
            self.original_user_ids.append(user_id)
            
            # Optional fields
            if 'is_purchase' in sample:
                self.is_purchase.append(sample['is_purchase'])
            
            if 'session_id' in sample:
                self.session_ids.append(sample['session_id'])
            
            # Map user ID
            user_idx = self.user_to_idx.get(user_id, 1)
            self.user_ids.append(user_idx)
            
            # Process short-term sequence
            has_short_term = len(sample['short_term_packages']) > 0
            self.has_short_term.append(has_short_term)
            
            if not has_short_term:
                empty_short_term_count += 1
                # Use empty token for first position
                st_pkgs = [self.empty_token] + [0] * (self.max_short_term - 1)
                st_countries = [self.unknown_token] + [0] * (self.max_short_term - 1)
                st_categories = [self.unknown_token] + [0] * (self.max_short_term - 1)
                st_themes = [self.unknown_token] + [0] * (self.max_short_term - 1)
                st_prices = [0.0] * self.max_short_term
                st_events = [self.empty_token] + [0] * (self.max_short_term - 1)
            else:
                # Truncate if necessary
                short_term_pkgs = sample['short_term_packages'][-self.max_short_term:]
                short_term_events = sample.get('short_term_events', [])[-self.max_short_term:]
                short_term_len = len(short_term_pkgs)
                
                # Process each package
                st_pkgs = []
                st_countries = []
                st_categories = []
                st_themes = []
                st_prices = []
                st_events = []
                
                for i, pkg_id in enumerate(short_term_pkgs):
                    pkg_str = str(pkg_id)
                    pkg_idx = self.package_to_idx.get(pkg_str, 1)
                    st_pkgs.append(pkg_idx)
                    
                    # Get package features (NO CITIES)
                    features = self.package_processor.get_package_features(pkg_str)
                    
                    if features:
                        st_countries.append(features['country_idx'])
                        st_categories.append(features['category_idx'])
                        st_themes.append(features['theme_idx'])
                        st_prices.append(features['price'])
                    else:
                        st_countries.append(self.unknown_token)
                        st_categories.append(self.unknown_token)
                        st_themes.append(self.unknown_token)
                        st_prices.append(0.0)
                    
                    # Event type
                    event_idx = short_term_events[i] if i < len(short_term_events) else 0
                    st_events.append(event_idx)
                
                # Pad sequences
                st_pkgs += [0] * (self.max_short_term - short_term_len)
                st_countries += [0] * (self.max_short_term - short_term_len)
                st_categories += [0] * (self.max_short_term - short_term_len)
                st_themes += [0] * (self.max_short_term - short_term_len)
                st_prices += [0.0] * (self.max_short_term - short_term_len)
                st_events += [0] * (self.max_short_term - short_term_len)
            
            # Store short-term data
            self.short_term_packages.append(st_pkgs)
            self.short_term_countries.append(st_countries)
            self.short_term_categories.append(st_categories)
            self.short_term_themes.append(st_themes)
            self.short_term_prices.append(st_prices)
            self.short_term_events.append(st_events)
            
            # Process long-term sequence (similar logic, NO CITIES)
            has_long_term = 'long_term_packages' in sample and len(sample['long_term_packages']) > 0
            
            if not has_long_term:
                empty_long_term_count += 1
                lt_pkgs = [self.empty_token] + [0] * (self.max_long_term - 1)
                lt_countries = [self.unknown_token] + [0] * (self.max_long_term - 1)
                lt_categories = [self.unknown_token] + [0] * (self.max_long_term - 1)
                lt_themes = [self.unknown_token] + [0] * (self.max_long_term - 1)
                lt_prices = [0.0] * self.max_long_term
                lt_events = [self.empty_token] + [0] * (self.max_long_term - 1)
            else:
                long_term_pkgs = sample['long_term_packages'][-self.max_long_term:]
                long_term_events = sample.get('long_term_events', [])[-self.max_long_term:]
                long_term_len = len(long_term_pkgs)
                
                lt_pkgs = []
                lt_countries = []
                lt_categories = []
                lt_themes = []
                lt_prices = []
                lt_events = []
                
                for i, pkg_id in enumerate(long_term_pkgs):
                    pkg_str = str(pkg_id)
                    pkg_idx = self.package_to_idx.get(pkg_str, 1)
                    lt_pkgs.append(pkg_idx)
                    
                    features = self.package_processor.get_package_features(pkg_str)
                    
                    if features:
                        lt_countries.append(features['country_idx'])
                        lt_categories.append(features['category_idx'])
                        lt_themes.append(features['theme_idx'])
                        lt_prices.append(features['price'])
                    else:
                        lt_countries.append(self.unknown_token)
                        lt_categories.append(self.unknown_token)
                        lt_themes.append(self.unknown_token)
                        lt_prices.append(0.0)
                    
                    event_idx = long_term_events[i] if i < len(long_term_events) else 0
                    lt_events.append(event_idx)
                
                # Pad sequences
                lt_pkgs += [0] * (self.max_long_term - long_term_len)
                lt_countries += [0] * (self.max_long_term - long_term_len)
                lt_categories += [0] * (self.max_long_term - long_term_len)
                lt_themes += [0] * (self.max_long_term - long_term_len)
                lt_prices += [0.0] * (self.max_long_term - long_term_len)
                lt_events += [0] * (self.max_long_term - long_term_len)
            
            # Store long-term data
            self.long_term_packages.append(lt_pkgs)
            self.long_term_countries.append(lt_countries)
            self.long_term_categories.append(lt_categories)
            self.long_term_themes.append(lt_themes)
            self.long_term_prices.append(lt_prices)
            self.long_term_events.append(lt_events)
            
            # Process purchased/target package (NO CITIES)
            purchased_pkg_id = str(sample['purchased_package'])
            purchased_pkg_idx = self.package_to_idx.get(purchased_pkg_id, 1)
            
            features = self.package_processor.get_package_features(purchased_pkg_id)
            
            if features:
                self.purchased_packages.append(purchased_pkg_idx)
                self.purchased_countries.append(features['country_idx'])
                self.purchased_categories.append(features['category_idx'])
                self.purchased_themes.append(features['theme_idx'])
                self.purchased_prices.append(features['price'])
            else:
                self.purchased_packages.append(purchased_pkg_idx)
                self.purchased_countries.append(self.unknown_token)
                self.purchased_categories.append(self.unknown_token)
                self.purchased_themes.append(self.unknown_token)
                self.purchased_prices.append(0.0)
        
        # Convert to tensors
        self.user_ids = torch.tensor(self.user_ids, dtype=torch.long)
        self.has_short_term = torch.tensor(self.has_short_term, dtype=torch.bool)
        
        if self.is_purchase:
            self.is_purchase = torch.tensor(self.is_purchase, dtype=torch.bool)
        
        if self.session_ids:
            self.session_ids = torch.tensor(self.session_ids, dtype=torch.long)
        
        # Convert package data to tensors (NO CITIES)
        self.short_term_packages = torch.tensor(self.short_term_packages, dtype=torch.long)
        self.short_term_countries = torch.tensor(self.short_term_countries, dtype=torch.long)
        self.short_term_categories = torch.tensor(self.short_term_categories, dtype=torch.long)
        self.short_term_themes = torch.tensor(self.short_term_themes, dtype=torch.long)
        self.short_term_prices = torch.tensor(self.short_term_prices, dtype=torch.float)
        self.short_term_events = torch.tensor(self.short_term_events, dtype=torch.long)
        
        self.long_term_packages = torch.tensor(self.long_term_packages, dtype=torch.long)
        self.long_term_countries = torch.tensor(self.long_term_countries, dtype=torch.long)
        self.long_term_categories = torch.tensor(self.long_term_categories, dtype=torch.long)
        self.long_term_themes = torch.tensor(self.long_term_themes, dtype=torch.long)
        self.long_term_prices = torch.tensor(self.long_term_prices, dtype=torch.float)
        self.long_term_events = torch.tensor(self.long_term_events, dtype=torch.long)
        
        self.purchased_packages = torch.tensor(self.purchased_packages, dtype=torch.long)
        self.purchased_countries = torch.tensor(self.purchased_countries, dtype=torch.long)
        self.purchased_categories = torch.tensor(self.purchased_categories, dtype=torch.long)
        self.purchased_themes = torch.tensor(self.purchased_themes, dtype=torch.long)
        self.purchased_prices = torch.tensor(self.purchased_prices, dtype=torch.float)
        
        # Print statistics
        print(f"NATR dataset prepared with {len(self.user_ids)} samples")
        print(f"Empty short-term: {empty_short_term_count} ({empty_short_term_count/len(self.user_ids)*100:.2f}%)")
        print(f"Empty long-term: {empty_long_term_count} ({empty_long_term_count/len(self.user_ids)*100:.2f}%)")
        
        # Print event type distribution
        if len(self.short_term_events) > 0:
            print("\nEvent type distribution in sequences:")
            for event_name, event_idx in self.event_to_idx.items():
                st_count = (self.short_term_events == event_idx).sum().item()
                lt_count = (self.long_term_events == event_idx).sum().item()
                print(f"  {event_name}: ST={st_count}, LT={lt_count}")
        
        # Print purchase statistics
        if hasattr(self, 'is_purchase') and len(self.is_purchase) > 0:
            purchase_count = self.is_purchase.sum().item() if torch.is_tensor(self.is_purchase) else sum(self.is_purchase)
            non_purchase_count = len(self.is_purchase) - purchase_count
            print(f"\nPurchase samples: {purchase_count} ({purchase_count/len(self.user_ids)*100:.2f}%)")
            print(f"Non-purchase samples: {non_purchase_count} ({non_purchase_count/len(self.user_ids)*100:.2f}%)")

    # Keep all the existing methods for embeddings and coordinates
    def get_sequence_embeddings(self, package_ids, sequence_type):
        """Get title embeddings for a sequence of packages"""
        max_len = self.max_short_term if sequence_type == 'short' else self.max_long_term
        embeddings = torch.zeros(max_len, self.embedding_dim)
        
        # Get reverse mapping from package_to_idx
        idx_to_main_id = {v: k for k, v in self.package_to_idx.items()}
        
        # Get feature package to idx mapping
        feature_package_to_idx = self.package_features.get('package_to_idx', {})
        
        for i, pkg_idx in enumerate(package_ids):
            if pkg_idx <= 0 or pkg_idx == self.empty_token:
                continue
            
            # Find the main_id for this package index
            main_id = idx_to_main_id.get(pkg_idx.item() if torch.is_tensor(pkg_idx) else pkg_idx)
            if main_id is None:
                continue
            
            # Convert to string (consistent with how it's stored)
            main_id_str = str(main_id)
            
            # Check in package features
            feature_idx = feature_package_to_idx.get(main_id_str)
            
            if feature_idx is not None and feature_idx < len(self.package_features['title_embeddings']):
                embeddings[i] = self.package_features['title_embeddings'][feature_idx]
        
        return embeddings
    
    def get_sequence_coordinates(self, package_ids, sequence_type):
        """Get coordinates for a sequence of packages"""
        max_len = self.max_short_term if sequence_type == 'short' else self.max_long_term
        coordinates = torch.zeros(max_len, 2)
        
        # Get reverse mapping from package_to_idx
        idx_to_main_id = {v: k for k, v in self.package_to_idx.items()}
        
        # Get feature package to idx mapping
        feature_package_to_idx = self.package_features.get('package_to_idx', {})
        
        for i, pkg_idx in enumerate(package_ids):
            if pkg_idx <= 0 or pkg_idx == self.empty_token:
                continue
            
            # Find the main_id for this package index
            main_id = idx_to_main_id.get(pkg_idx.item() if torch.is_tensor(pkg_idx) else pkg_idx)
            if main_id is None:
                continue
            
            # Convert to string (consistent with how it's stored)
            main_id_str = str(main_id)
            
            # Check in package features
            feature_idx = feature_package_to_idx.get(main_id_str)
            
            if feature_idx is not None and feature_idx < len(self.package_features['coordinates']):
                coordinates[i] = self.package_features['coordinates'][feature_idx]
        
        return coordinates
    
    def get_package_embedding(self, package_id):
        """Get embedding for a single package"""
        # Handle tensor input
        if torch.is_tensor(package_id):
            package_id = package_id.item()
        
        # Get reverse mapping
        idx_to_main_id = {v: k for k, v in self.package_to_idx.items()}
        
        # Find the main_id
        main_id = idx_to_main_id.get(package_id)
        if main_id is None:
            return torch.zeros(self.embedding_dim)
        
        # Convert to string
        main_id_str = str(main_id)
        
        # Get feature index
        feature_idx = self.package_features.get('package_to_idx', {}).get(main_id_str)
        
        if feature_idx is not None and feature_idx < len(self.package_features['title_embeddings']):
            return self.package_features['title_embeddings'][feature_idx]
        else:
            return torch.zeros(self.embedding_dim)

    def get_package_coordinates(self, package_id):
        """Get coordinates for a single package"""
        # Handle tensor input
        if torch.is_tensor(package_id):
            package_id = package_id.item()
        
        # Get reverse mapping
        idx_to_main_id = {v: k for k, v in self.package_to_idx.items()}
        
        # Find the main_id
        main_id = idx_to_main_id.get(package_id)
        if main_id is None:
            return torch.zeros(2)
        
        # Convert to string
        main_id_str = str(main_id)
        
        # Get feature index
        feature_idx = self.package_features.get('package_to_idx', {}).get(main_id_str)
        
        if feature_idx is not None and feature_idx < len(self.package_features['coordinates']):
            return self.package_features['coordinates'][feature_idx]
        else:
            return torch.zeros(2)
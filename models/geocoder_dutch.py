import pandas as pd
import numpy as np
import googlemaps
from typing import Dict, Tuple, Optional
import os
import json
from tqdm import tqdm
import pickle

class GoogleDutchGeocoder:
    """
    Standalone geocoder for Dutch city names using Google Maps API
    Saves coordinates mapped by main_id for consistency
    """
    def __init__(self, 
                 google_api_key: str,
                 cache_dir: str = "data/cache/geocoding"):
        """
        Initialize geocoder with Google Maps API
        
        Args:
            google_api_key: Google Maps API key (required)
            cache_dir: Directory for caching results
        """
        self.gmaps = googlemaps.Client(key=google_api_key)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Separate caches
        self.city_cache_file = os.path.join(cache_dir, "city_coordinates.json")
        self.main_id_cache_file = os.path.join(cache_dir, "main_id_coordinates.pkl")
        
        # Load existing caches
        self.city_coordinates = self._load_city_cache()
        self.main_id_coordinates = self._load_main_id_cache()
        
        # Track API usage
        self.api_calls = 0
        
        # Simple hardcoded cities for the rare cases Google Maps fails
        self.HARDCODED_CITIES = {
            'Santo Stefano al Mare, Italië': (43.837614, 7.897047), 
            # Add more cities here as needed:
            # 'City, Country': (latitude, longitude),
        }
    
    def _load_city_cache(self) -> Dict:
        """Load cached city coordinates"""
        if os.path.exists(self.city_cache_file):
            try:
                with open(self.city_cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                    print(f"Loaded {len(cache)} cached city coordinates")
                    return cache
            except Exception as e:
                print(f"Error loading city cache: {e}")
        return {}
    
    def _load_main_id_cache(self) -> Dict:
        """Load cached main_id coordinates"""
        if os.path.exists(self.main_id_cache_file):
            try:
                with open(self.main_id_cache_file, 'rb') as f:
                    cache = pickle.load(f)
                    print(f"Loaded {len(cache)} cached main_id coordinates")
                    return cache
            except Exception as e:
                print(f"Error loading main_id cache: {e}")
        return {}
    
    def _save_city_cache(self):
        """Save city coordinates cache"""
        try:
            with open(self.city_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.city_coordinates, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving city cache: {e}")
    
    def _save_main_id_cache(self):
        """Save main_id coordinates cache"""
        try:
            with open(self.main_id_cache_file, 'wb') as f:
                pickle.dump(self.main_id_coordinates, f)
        except Exception as e:
            print(f"Error saving main_id cache: {e}")
    
    def geocode_location(self, city: str, country: str) -> Optional[Tuple[float, float]]:
        """
        Geocode a city-country pair using Google Maps
        
        Args:
            city: City name in Dutch
            country: Country name in Dutch
            
        Returns:
            Tuple of (latitude, longitude) or None
        """
        # Create cache key
        cache_key = f"{city}|{country}"
        
        # Check cache
        if cache_key in self.city_coordinates:
            coords = self.city_coordinates[cache_key]
            return (coords['lat'], coords['lng'])
        
        try:
            # Google Maps API with Dutch language
            query = f"{city}, {country}"
            self.api_calls += 1
            
            result = self.gmaps.geocode(query, language='nl')
            
            if result:
                location = result[0]['geometry']['location']
                lat, lng = location['lat'], location['lng']
                
                # Cache the result
                self.city_coordinates[cache_key] = {
                    'lat': lat,
                    'lng': lng,
                    'formatted_address': result[0].get('formatted_address', ''),
                    'query': query
                }
                
                # Save cache periodically
                if self.api_calls % 50 == 0:
                    self._save_city_cache()
                
                return (lat, lng)
            else:
                # Check hardcoded cities if Google Maps fails
                hardcoded_key = f"{city}, {country}"
                if hardcoded_key in self.HARDCODED_CITIES:
                    lat, lng = self.HARDCODED_CITIES[hardcoded_key]
                    print(f"Using hardcoded coordinates for: {hardcoded_key}")
                    
                    # Cache it
                    self.city_coordinates[cache_key] = {
                        'lat': lat,
                        'lng': lng,
                        'formatted_address': hardcoded_key,
                        'query': query
                    }
                    return (lat, lng)
                else:
                    print(f"No results for: {city}, {country}")
                    return None
                
        except Exception as e:
            print(f"Google Maps error for '{city}, {country}': {e}")
            return None
    
    def process_feed_data(self, feed_path: str = "data/feed.parquet", 
                         output_coordinates: bool = True) -> pd.DataFrame:
        """
        Process feed.parquet and geocode all cities
        Stores coordinates by main_id for consistency
        
        Args:
            feed_path: Path to feed.parquet
            output_coordinates: Whether to add coordinate columns to dataframe
            
        Returns:
            DataFrame (optionally with coordinate columns added)
        """
        print(f"Loading {feed_path}...")
        df = pd.read_parquet(feed_path)
        
        print(f"Loaded {len(df)} packages")
        
        # Get unique city-country combinations
        unique_locations = df[['city', 'country']].drop_duplicates()
        unique_locations = unique_locations[unique_locations['city'].notna()]
        
        print(f"Found {len(unique_locations)} unique city-country combinations")
        
        # Geocode each unique location
        city_coord_dict = {}
        
        for _, row in tqdm(unique_locations.iterrows(), total=len(unique_locations), desc="Geocoding"):
            city = row['city']
            country = row['country']
            
            if pd.notna(city):
                coords = self.geocode_location(city, country)
                if coords:
                    city_coord_dict[(city, country)] = coords
        
        # Map coordinates to main_id
        print("\nMapping coordinates to main_id...")
        new_mappings = 0
        
        for idx, row in df.iterrows():
            main_id = str(row['main_id'])  # Ensure string for consistency
            city = row['city']
            country = row['country']
            
            # Only update if not already cached
            if main_id not in self.main_id_coordinates:
                if pd.notna(city) and (city, country) in city_coord_dict:
                    coords = city_coord_dict[(city, country)]
                    self.main_id_coordinates[main_id] = {
                        'latitude': coords[0],
                        'longitude': coords[1],
                        'city': city,
                        'country': country
                    }
                    new_mappings += 1
        
        # Save all caches
        self._save_city_cache()
        self._save_main_id_cache()
        
        # Print statistics
        print(f"\nGeocoding complete:")
        print(f"  Total main_ids with coordinates: {len(self.main_id_coordinates)}")
        print(f"  New main_id mappings added: {new_mappings}")
        print(f"  Total API calls this session: {self.api_calls}")
        
        # Optionally add coordinates to dataframe
        if output_coordinates:
            df['latitude'] = None
            df['longitude'] = None
            
            for idx, row in df.iterrows():
                main_id = str(row['main_id'])
                if main_id in self.main_id_coordinates:
                    coords = self.main_id_coordinates[main_id]
                    df.at[idx, 'latitude'] = coords['latitude']
                    df.at[idx, 'longitude'] = coords['longitude']
            
            geocoded_count = df['latitude'].notna().sum()
            print(f"  Packages with coordinates: {geocoded_count}/{len(df)} ({geocoded_count/len(df)*100:.1f}%)")
        
        return df
    
    def get_main_id_coordinates(self) -> Dict[str, Dict]:
        """
        Get all coordinates mapped by main_id
        
        Returns:
            Dictionary mapping main_id to coordinate data
        """
        return self.main_id_coordinates
    
    def get_coordinates_for_main_id(self, main_id: str) -> Optional[Dict]:
        """
        Get coordinates for a specific main_id
        
        Args:
            main_id: The main_id to look up (as string)
            
        Returns:
            Dictionary with latitude, longitude, city, country or None
        """
        return self.main_id_coordinates.get(str(main_id))
    
    def clear_cache(self):
        """Clear all geocoding caches"""
        self.city_coordinates = {}
        self.main_id_coordinates = {}
        self._save_city_cache()
        self._save_main_id_cache()
        print("Cleared all geocoding caches")
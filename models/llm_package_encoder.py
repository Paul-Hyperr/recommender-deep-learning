# Enhanced LLMPackageEncoder with fallback title generation

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict, Any
import json
import os
import time
from openai import OpenAI
import pickle
from tqdm import tqdm


class LLMPackageEncoder(nn.Module):
    """
    Enhanced Package encoder using LLM embeddings with fallback title generation
    """
    def __init__(
        self,
        package_vocab_size: int,
        country_vocab_size: int,
        category_vocab_size: int,
        theme_vocab_size: int,
        projection_dim: int = 256,
        api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-large",
        cache_dir: str = "data/cache/llm_embeddings",
        embedding_dim: int = 3072,  # Default for text-embedding-3-large
    ):
        super().__init__()
        
        self.projection_dim = projection_dim
        self.embedding_model = embedding_model
        self.cache_dir = cache_dir
        self.embedding_dim = embedding_dim
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize OpenAI client
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            print("Warning: No OpenAI API key provided. LLM embeddings will not be available.")
            self.client = None
        
        # Projection layers for LLM embeddings
        self.llm_projection = nn.Sequential(
            nn.Linear(self.embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, projection_dim)
        )
        
        # Load or initialize embedding cache
        self.cache_file = os.path.join(self.cache_dir, f"{embedding_model}_cache.json")
        self.embedding_cache = self._load_embedding_cache()
        
        print(f"Loaded {len(self.embedding_cache)} cached embeddings from {self.cache_file}")
    
    def _load_embedding_cache(self) -> Dict[str, List[float]]:
        """Load cached embeddings from file"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                    # Ensure all values are lists (not numpy arrays)
                    return {k: v if isinstance(v, list) else v.tolist() for k, v in cache.items()}
            except Exception as e:
                print(f"Error loading cache: {e}")
                return {}
        return {}
    
    def _save_embedding_cache(self):
        """Save embedding cache to file"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_cache = {}
            for k, v in self.embedding_cache.items():
                if isinstance(v, np.ndarray):
                    serializable_cache[k] = v.tolist()
                else:
                    serializable_cache[k] = v
            
            with open(self.cache_file, 'w') as f:
                json.dump(serializable_cache, f)
            
            print(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def _create_fallback_title(self, package_metadata: Dict[str, Any]) -> str:
        """
        Create a fallback title from package metadata
        
        Args:
            package_metadata: Dictionary containing package information
            
        Returns:
            Fallback title string
        """
        parts = []
        
        # Add theme
        theme = package_metadata.get('theme', '').strip()
        if theme and theme != 'Unknown':
            parts.append(theme)
        
        # Add category
        category = package_metadata.get('category', '').strip()
        if category and category != 'Unknown':
            parts.append(category)
        
        # Add location
        location_parts = []
        city = package_metadata.get('city', '').strip()
        if city and city != 'Unknown':
            location_parts.append(city)
        
        country = package_metadata.get('country', '').strip()
        if country and country != 'Unknown':
            location_parts.append(country)
        
        if location_parts:
            parts.append(f"in {', '.join(location_parts)}")
        
        if parts:
            fallback_title = " - ".join(parts)
        else:
            # Last resort: use main_id
            main_id = package_metadata.get('main_id', 'Unknown')
            fallback_title = f"Travel Package {main_id}"
        
        return fallback_title
    
    def get_embedding(self, text: str, cache_key: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Get embedding for a single text with caching
        
        Args:
            text: Text to embed
            cache_key: Optional cache key (defaults to text)
            
        Returns:
            Embedding array or None if failed
        """
        if not self.client:
            return None
        
        # Use text as cache key if not provided
        if cache_key is None:
            cache_key = text
        
        # Check cache
        if cache_key in self.embedding_cache:
            embedding = self.embedding_cache[cache_key]
            if isinstance(embedding, list):
                return np.array(embedding)
            return embedding
        
        try:
            # Get embedding from OpenAI
            response = self.client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            
            embedding = np.array(response.data[0].embedding)
            
            # Cache the embedding
            self.embedding_cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            print(f"Error getting embedding for '{text[:50]}...': {e}")
            return None
    
    def process_batch_packages(self, package_data: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Process a batch of packages, handling empty titles with fallbacks
        
        Args:
            package_data: List of package metadata dictionaries
            
        Returns:
            Tensor of embeddings
        """
        embeddings = []
        
        for package in package_data:
            main_id = str(package.get('main_id', ''))
            title = package.get('title' or '').strip()
            
            # Check if title is empty
            if not title:
                # Create fallback title
                fallback_title = self._create_fallback_title(package)
                print(f"Empty title for package {main_id}, using fallback: '{fallback_title}'")
                title = fallback_title
            
            # Try to get embedding (check cache first by main_id, then by title)
            embedding = None
            
            # Check cache by main_id first
            if main_id in self.embedding_cache:
                embedding = self.embedding_cache[main_id]
            
            # If not found, try by title
            if embedding is None and title in self.embedding_cache:
                embedding = self.embedding_cache[title]
            
            # If still not found, generate new embedding
            if embedding is None:
                embedding = self.get_embedding(title, cache_key=main_id)
            
            if embedding is not None:
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                embeddings.append(embedding)
            else:
                # Use zero embedding as last resort
                print(f"Failed to get embedding for {main_id}, using zero embedding")
                embeddings.append(np.zeros(self.embedding_dim))
        
        # Convert to tensor
        embeddings_tensor = torch.tensor(np.array(embeddings), dtype=torch.float)
        
        # Apply projection layer
        projected = self.llm_projection(embeddings_tensor)
        
        return projected
    
    def process_batch_titles(self, titles: List[str], main_ids: List[str], 
                           package_metadata: Optional[List[Dict[str, Any]]] = None) -> torch.Tensor:
        """
        Process a batch of titles with fallback support
        
        Args:
            titles: List of titles
            main_ids: List of main IDs
            package_metadata: Optional list of full package metadata for fallback generation
            
        Returns:
            Tensor of embeddings (not projected)
        """
        embeddings = []
        batch_size = 20
        
        for idx, (title, main_id) in enumerate(zip(titles, main_ids)):
            main_id_str = str(main_id)
            
            # Handle empty title
            if not title.strip():
                if package_metadata and idx < len(package_metadata):
                    # Use package metadata to create fallback
                    fallback_title = self._create_fallback_title(package_metadata[idx])
                    print(f"Empty title for {main_id_str}, using fallback: '{fallback_title}'")
                    title = fallback_title
                else:
                    # Basic fallback
                    title = f"Travel Package {main_id_str}"
                    print(f"Empty title for {main_id_str}, using basic fallback: '{title}'")
            
            # Get embedding
            embedding = self.get_embedding(title, cache_key=main_id_str)
            
            if embedding is not None:
                embeddings.append(embedding)
            else:
                print(f"Failed to get embedding for {main_id_str}, using zero embedding")
                embeddings.append(np.zeros(self.embedding_dim))
        
        # Save cache periodically
        if len(embeddings) % 100 == 0:
            self._save_embedding_cache()
        
        return torch.tensor(np.array(embeddings), dtype=torch.float)
    
    def forward(self, package_data: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            package_data: List of package metadata dictionaries
            
        Returns:
            Projected embeddings tensor
        """
        return self.process_batch_packages(package_data)



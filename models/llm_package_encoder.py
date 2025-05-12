import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import requests
import json
import time
from typing import Dict, List, Optional
from tqdm import tqdm

class LLMPackageEncoder(nn.Module):
    """
    Package Encoder that uses LLM (ChatGPT) for title embeddings
    
    This encoder uses the OpenAI API to generate embeddings for package titles.
    """
    def __init__(self, 
                package_vocab_size: int, 
                country_vocab_size: int, 
                category_vocab_size: int, 
                theme_vocab_size: int, 
                embedding_dim: int = 256,
                hidden_dim: int = 256,
                api_key: Optional[str] = None,
                embedding_model: str = "text-embedding-3-large",
                dropout: float = 0.2,
                cache_dir: str = "data/cache/llm_embeddings"):
        super(LLMPackageEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.embedding_model = embedding_model
        
        # Create cache directory
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Check if API key is available
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it as OPENAI_API_KEY environment variable or pass it to the constructor.")
        
        # Determine LLM embedding dimension based on model
        if embedding_model == "text-embedding-3-small":
            self.llm_embedding_dim = 1536
        elif embedding_model == "text-embedding-3-large":
            self.llm_embedding_dim = 3072
        elif embedding_model == "text-embedding-ada-002":
            self.llm_embedding_dim = 1536
        else:
            self.llm_embedding_dim = 1536  # Default
        
        # Linear layer to transform LLM embeddings to our dimension
        self.text_transform = nn.Linear(self.llm_embedding_dim, embedding_dim)
        
        # Embeddings for different package attributes
        self.package_embeddings = nn.Embedding(package_vocab_size, embedding_dim, padding_idx=0)
        self.country_embeddings = nn.Embedding(country_vocab_size, embedding_dim, padding_idx=0)
        self.category_embeddings = nn.Embedding(category_vocab_size, embedding_dim, padding_idx=0)
        self.theme_embeddings = nn.Embedding(theme_vocab_size, embedding_dim, padding_idx=0)
        
        # Bi-LSTM for title encoding
        self.title_lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim // 2,
            batch_first=True, 
            bidirectional=True
        )
        
        # MLPs for different views
        self.destination_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.category_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.theme_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # View-level attention weights
        self.view_attention_query = nn.Parameter(torch.zeros(hidden_dim))
        self.view_attention = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Load cached embeddings if available
        self.embedding_cache = {}
        self._load_embedding_cache()
        
        # Add API call tracking for cost management
        self.api_call_count = 0
        self.api_call_tokens = 0
    
    def _load_embedding_cache(self):
        """Load cached embeddings from disk"""
        cache_file = os.path.join(self.cache_dir, f"{self.embedding_model}_cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    # JSON doesn't support numpy arrays directly, so we store as lists
                    cache_data = json.load(f)
                    for key, value in cache_data.items():
                        self.embedding_cache[key] = np.array(value)
                print(f"Loaded {len(self.embedding_cache)} cached embeddings from {cache_file}")
            except Exception as e:
                print(f"Error loading embedding cache: {e}")
    
    def _save_embedding_cache(self):
        """Save embedding cache to disk"""
        cache_file = os.path.join(self.cache_dir, f"{self.embedding_model}_cache.json")
        try:
            # Convert numpy arrays to lists for JSON serialization
            cache_data = {k: v.tolist() for k, v in self.embedding_cache.items()}
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f)
            print(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            print(f"Error saving embedding cache: {e}")
    
    def get_package_embedding(self, title: str, main_id = None):
        """
        Get embedding for a package, prioritizing cache by main_id
        
        Args:
            title: Package title
            main_id: Unique package identifier
        
        Returns:
            Embedding vector
        """
        # Use main_id for cache key if provided, otherwise use title
        cache_key = str(main_id) if main_id is not None else title
        
        # Check cache first
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        elif title in self.embedding_cache:
            return self.embedding_cache[title]
        
        # Make API request with retry logic
        max_retries = 5
        retry_delay = 1  # starting delay in seconds
        
        for attempt in range(max_retries):
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                data = {
                    "input": title,
                    "model": self.embedding_model
                }
                
                self.api_call_count += 1
                self.api_call_tokens += len(title.split()) + 5  # Rough estimate
                
                response = requests.post(
                    "https://api.openai.com/v1/embeddings",
                    headers=headers,
                    data=json.dumps(data)
                )
                
                if response.status_code == 429:  # Rate limit exceeded
                    if attempt < max_retries - 1:
                        sleep_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        print(f"Rate limit exceeded. Retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
                        continue
                elif response.status_code != 200:
                    print(f"API error: {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        sleep_time = retry_delay * (2 ** attempt)
                        print(f"Retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
                        continue
                    else:
                        # Return zeros as fallback after max retries
                        print(f"Max retries reached, returning zero embedding")
                        embedding = np.zeros(self.llm_embedding_dim)
                else:
                    embedding = np.array(response.json()["data"][0]["embedding"])
                    
                    # Update cache with both main_id and title as keys
                    if main_id is not None:
                        self.embedding_cache[str(main_id)] = embedding
                    self.embedding_cache[title] = embedding
                    
                    # Save cache periodically (every 10 new embeddings)
                    if len(self.embedding_cache) % 10 == 0:
                        self._save_embedding_cache()
                
                return embedding
                
            except Exception as e:
                print(f"Error getting embedding: {e}")
                if attempt < max_retries - 1:
                    sleep_time = retry_delay * (2 ** attempt)
                    print(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    print(f"Max retries reached, returning zero embedding")
                    # Return zeros as fallback
                    return np.zeros(self.llm_embedding_dim)
    
    def process_batch_titles_optimized(self, titles, main_ids=None):
        """
        Process a batch of titles using LLM embeddings with batch API calls
        
        Args:
            titles: List of title strings
            main_ids: Optional list of main_ids for caching
        
        Returns:
            torch.Tensor: Transformed embeddings
        """
        # Check cache first
        uncached_titles = []
        uncached_indices = []
        cached_embeddings = []
        
        for i, title in enumerate(titles):
            main_id = main_ids[i] if main_ids is not None else None
            
            # Check if we have this embedding cached (by main_id or title)
            cache_key = str(main_id) if main_id is not None else title
            if cache_key in self.embedding_cache:
                cached_embeddings.append((i, self.embedding_cache[cache_key]))
            elif title in self.embedding_cache:
                cached_embeddings.append((i, self.embedding_cache[title]))
            else:
                uncached_titles.append(title)
                uncached_indices.append(i)
        
        # If there are uncached titles, get their embeddings
        if uncached_titles:
            try:
                # Use the batch API for efficiency
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                # For large batches, process in chunks to avoid rate limits
                max_batch_size = 20  # Adjust based on API limits
                
                for i in range(0, len(uncached_titles), max_batch_size):
                    batch_titles = uncached_titles[i:i+max_batch_size]
                    batch_indices = uncached_indices[i:i+max_batch_size]
                    
                    data = {
                        "input": batch_titles,
                        "model": self.embedding_model
                    }
                    
                    self.api_call_count += 1
                    # Rough estimate of total tokens
                    self.api_call_tokens += sum(len(t.split()) for t in batch_titles) + 5 * len(batch_titles)
                    
                    response = requests.post(
                        "https://api.openai.com/v1/embeddings",
                        headers=headers,
                        data=json.dumps(data)
                    )
                    
                    if response.status_code == 200:
                        for j, embedding_data in enumerate(response.json()["data"]):
                            title = batch_titles[j]
                            embedding = np.array(embedding_data["embedding"])
                            
                            # Cache by main_id (if provided) and title
                            if main_ids is not None:
                                main_id = main_ids[batch_indices[j]]
                                cache_key = str(main_id)
                                self.embedding_cache[cache_key] = embedding
                            
                            # Also cache by title
                            self.embedding_cache[title] = embedding
                    else:
                        print(f"API error: {response.status_code} - {response.text}")
                        # Fall back to zeros for all titles in this batch
                        for title in batch_titles:
                            self.embedding_cache[title] = np.zeros(self.llm_embedding_dim)
                
                # Save cache after processing batch
                self._save_embedding_cache()
                    
            except Exception as e:
                print(f"Error getting batch embeddings: {e}")
                # Fall back to zeros for all uncached titles
                for title in uncached_titles:
                    self.embedding_cache[title] = np.zeros(self.llm_embedding_dim)
        
        # Combine all embeddings in the original order
        all_embeddings = [None] * len(titles)
        
        # Add cached embeddings
        for idx, emb in cached_embeddings:
            all_embeddings[idx] = torch.tensor(emb, dtype=torch.float)
        
        # Add newly fetched embeddings
        for i, orig_idx in enumerate(uncached_indices):
            title = titles[orig_idx]
            main_id = main_ids[orig_idx] if main_ids is not None else None
            
            # Get from cache (either by main_id or title)
            cache_key = str(main_id) if main_id is not None else title
            
            if cache_key in self.embedding_cache:
                embedding = self.embedding_cache[cache_key]
            elif title in self.embedding_cache:
                embedding = self.embedding_cache[title]
            else:
                # This shouldn't happen, but just in case
                print(f"Warning: No embedding found for {title}")
                embedding = np.zeros(self.llm_embedding_dim)
            
            all_embeddings[orig_idx] = torch.tensor(embedding, dtype=torch.float)
        
        # Stack embeddings
        embeddings_tensor = torch.stack(all_embeddings)
        
        # Transform to our embedding dimension
        transformed_embeddings = self.text_transform(embeddings_tensor)
        
        print(f"Processed {len(titles)} titles: {len(cached_embeddings)} from cache, {len(uncached_titles)} from API")
        
        return transformed_embeddings
    
    def process_batch_titles(self, titles, main_ids=None):
        """
        Process a batch of titles using LLM embeddings
        
        Args:
            titles: List of title strings
            main_ids: Optional list of main_ids for caching
            
        Returns:
            Embeddings tensor
        """
        # Use the optimized version
        return self.process_batch_titles_optimized(titles, main_ids)
    
    def forward(self, titles, country_ids, category_ids, theme_ids, user_query=None):
        """
        Forward pass of the LLM-based package encoder
        
        Args:
            titles: List of title strings
            country_ids: Tensor of country indices
            category_ids: Tensor of category indices
            theme_ids: Tensor of theme indices
            user_query: Optional user representation for personalized attention
            
        Returns:
            Package representation vector
        """
        batch_size = len(titles)
        device = country_ids.device
        
        # Title encoding with LLM
        title_embeddings = self.process_batch_titles(titles).to(device)
        
        # Get contextual title representation using Bi-LSTM
        title_outputs, _ = self.title_lstm(title_embeddings.unsqueeze(1))
        title_representation = title_outputs.squeeze(1)
        
        # Destination (country) encoding
        country_embedded = self.country_embeddings(country_ids)
        country_representation = self.destination_mlp(country_embedded)
        
        # Category encoding
        category_embedded = self.category_embeddings(category_ids)
        category_representation = self.category_mlp(category_embedded)
        
        # Theme encoding
        theme_embedded = self.theme_embeddings(theme_ids)
        theme_representation = self.theme_mlp(theme_embedded)
        
        # Stack all representations
        view_representations = torch.stack([
            title_representation, 
            country_representation, 
            category_representation, 
            theme_representation
        ], dim=1)
        
        # View-level attention
        if user_query is not None:
            attention_query = user_query.unsqueeze(1)
        else:
            attention_query = self.view_attention_query.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
            
        view_attn_weights = torch.bmm(attention_query, view_representations.transpose(1, 2)).squeeze(1)
        view_attn_weights = F.softmax(view_attn_weights, dim=1).unsqueeze(1)
        
        # Apply view-level attention weights
        package_representation = torch.bmm(view_attn_weights, view_representations).squeeze(1)
        
        return package_representation
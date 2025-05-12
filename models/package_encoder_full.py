import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union, Tuple

class EnhancedPackageEncoder(nn.Module):
    """
    Enhanced Package Encoder module for the NATR model
    
    This module learns a unified travel package representation from multiple attributes:
    - Title
    - Country (destination)
    - Category
    - Theme
    - Price information
    
    It uses an attentive multi-view learning approach to combine these attributes.
    """
    def __init__(self, 
                package_vocab_size: int,
                country_vocab_size: int,
                category_vocab_size: int,
                theme_vocab_size: int,
                embedding_dim: int = 256,
                hidden_dim: int = 256,
                price_buckets: int = 10,
                dropout: float = 0.2):
        """
        Initialize the Enhanced Package Encoder
        
        Args:
            package_vocab_size: Size of package vocabulary
            country_vocab_size: Size of country vocabulary
            category_vocab_size: Size of category vocabulary
            theme_vocab_size: Size of theme vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of hidden layers
            price_buckets: Number of price range buckets
            dropout: Dropout probability
        """
        super(EnhancedPackageEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Package embeddings (for direct lookup)
        self.package_embeddings = nn.Embedding(package_vocab_size, embedding_dim, padding_idx=0)
        
        # Title encoder components
        # (Note: In a real implementation, this would use a pretrained word embedding model
        # or LLM embeddings as shown in your existing package_encoder.py and llm_package_encoder.py)
        self.title_embedding = nn.Embedding(package_vocab_size, embedding_dim, padding_idx=0)
        
        # Country (destination) embeddings
        self.country_embedding = nn.Embedding(country_vocab_size, embedding_dim, padding_idx=0)
        
        # Category embeddings 
        self.category_embedding = nn.Embedding(category_vocab_size, embedding_dim, padding_idx=0)
        
        # Theme embeddings
        self.theme_embedding = nn.Embedding(theme_vocab_size, embedding_dim, padding_idx=0)
        
        # Price embeddings (discretized into buckets)
        self.price_embedding = nn.Embedding(price_buckets + 1, embedding_dim, padding_idx=0)
        
        # MLPs for different views
        self.title_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.country_mlp = nn.Sequential(
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
        
        self.price_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # View-level attention
        self.view_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
        # Final projection layer
        self.final_projection = nn.Linear(hidden_dim, embedding_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def discretize_price(self, price: torch.Tensor, min_price: float = 0, max_price: float = 10000, num_buckets: int = 10) -> torch.Tensor:
        """
        Discretize continuous price values into buckets
        
        Args:
            price: Continuous price values [batch_size]
            min_price: Minimum price value
            max_price: Maximum price value
            num_buckets: Number of price buckets
            
        Returns:
            Discretized price buckets [batch_size]
        """
        # Clip prices to the range [min_price, max_price]
        price_clipped = torch.clamp(price, min_price, max_price)
        
        # Normalize to [0, 1]
        price_norm = (price_clipped - min_price) / (max_price - min_price)
        
        # Discretize into buckets (0 is reserved for padding)
        price_buckets = torch.floor(price_norm * num_buckets).long() + 1
        
        return price_buckets
    
    def encode_attributes(self, 
                         package_ids: Optional[torch.Tensor] = None,
                         country_ids: Optional[torch.Tensor] = None,
                         category_ids: Optional[torch.Tensor] = None,
                         theme_ids: Optional[torch.Tensor] = None,
                         price_values: Optional[torch.Tensor] = None,
                         user_query: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Encode package attributes into a unified representation
        
        Args:
            package_ids: IDs of packages [batch_size]
            country_ids: IDs of countries [batch_size]
            category_ids: IDs of categories [batch_size]
            theme_ids: IDs of themes [batch_size]
            price_values: Price values [batch_size]
            user_query: Optional user query for personalized attention [batch_size, embedding_dim]
            
        Returns:
            Unified package representation [batch_size, embedding_dim] and attention info dictionary
        """
        batch_size = package_ids.size(0) if package_ids is not None else country_ids.size(0)
        device = package_ids.device if package_ids is not None else country_ids.device
        
        view_representations = []
        view_names = []
        
        # Title/Package representation (using package_ids as proxy for title)
        if package_ids is not None:
            title_emb = self.title_embedding(package_ids)
            title_repr = self.title_mlp(title_emb)
            view_representations.append(title_repr)
            view_names.append('Title')
        
        # Country representation
        if country_ids is not None:
            country_emb = self.country_embedding(country_ids)
            country_repr = self.country_mlp(country_emb)
            view_representations.append(country_repr)
            view_names.append('Country')
        
        # Category representation
        if category_ids is not None:
            category_emb = self.category_embedding(category_ids)
            category_repr = self.category_mlp(category_emb)
            view_representations.append(category_repr)
            view_names.append('Category')
        
        # Theme representation
        if theme_ids is not None:
            theme_emb = self.theme_embedding(theme_ids)
            theme_repr = self.theme_mlp(theme_emb)
            view_representations.append(theme_repr)
            view_names.append('Theme')
        
        # Price representation
        if price_values is not None:
            price_buckets = self.discretize_price(price_values)
            price_emb = self.price_embedding(price_buckets)
            price_repr = self.price_mlp(price_emb)
            view_representations.append(price_repr)
            view_names.append('Price')
        
        # Stack view representations
        if view_representations:
            view_tensor = torch.stack(view_representations, dim=1)  # [batch_size, num_views, hidden_dim]
        else:
            # If no attributes provided, return zero embeddings
            return torch.zeros(batch_size, self.embedding_dim, device=device), {}
        
        # Apply view-level attention
        view_attention_scores = self.view_attention(view_tensor).squeeze(-1)  # [batch_size, num_views]
        view_attention_weights = F.softmax(view_attention_scores, dim=1).unsqueeze(1)  # [batch_size, 1, num_views]
        
        # Apply attention weights to get unified representation
        unified_repr = torch.bmm(view_attention_weights, view_tensor).squeeze(1)  # [batch_size, hidden_dim]
        
        # Apply dropout and final projection
        unified_repr = self.dropout(unified_repr)
        package_repr = self.final_projection(unified_repr)  # [batch_size, embedding_dim]
        
        # Store attention weights for visualization and analysis
        attention_info = {
            'view_names': view_names,
            'view_attention': view_attention_weights.squeeze(1).detach().cpu().numpy()
        }
        
        return package_repr, attention_info
    
    def forward(self, 
               package_ids: Optional[torch.Tensor] = None,
               country_ids: Optional[torch.Tensor] = None,
               category_ids: Optional[torch.Tensor] = None,
               theme_ids: Optional[torch.Tensor] = None,
               price_values: Optional[torch.Tensor] = None,
               user_query: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the Enhanced Package Encoder
        
        Args:
            package_ids: IDs of packages [batch_size]
            country_ids: IDs of countries [batch_size]
            category_ids: IDs of categories [batch_size]
            theme_ids: IDs of themes [batch_size]
            price_values: Price values [batch_size]
            user_query: Optional user query for personalized attention [batch_size, embedding_dim]
            
        Returns:
            Package representation [batch_size, embedding_dim]
        """
        # If only package_ids are provided, use the direct embedding lookup for efficiency
        if (package_ids is not None and 
            country_ids is None and 
            category_ids is None and 
            theme_ids is None and 
            price_values is None):
            return self.package_embeddings(package_ids), {}
        
        # Otherwise, encode all available attributes
        return self.encode_attributes(
            package_ids, country_ids, category_ids, theme_ids, price_values, user_query
        )
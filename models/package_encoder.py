import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, List, Set, Tuple, Optional

# Try importing fasttext, with helpful error message if not installed
try:
    import fasttext
except ImportError:
    print("fasttext package not installed. Please install with: pip install fasttext")
    print("Note: on Mac, you might need to install gcc first: brew install gcc")
    raise

# Try importing spacy, with helpful error message if not installed
try:
    import spacy
except ImportError:
    print("spacy package not installed. Please install with: pip install spacy")
    print("Then download Dutch model: python -m spacy download nl_core_news_sm")
    raise

class PackageEncoder(nn.Module):
    """
    Travel Package Encoder module with attentive multi-view learning
    
    This module learns a unified package representation by incorporating
    different views of the package with attention mechanisms.
    
    Uses FastText pre-trained embeddings for Dutch text.
    """
    def __init__(self, 
                package_vocab_size: int, 
                country_vocab_size: int, 
                category_vocab_size: int, 
                theme_vocab_size: int, 
                embedding_dim: int = 256, 
                hidden_dim: int = 256,
                fasttext_model_path: str = "data/cc.nl.300.bin",
                fasttext_embedding_dim: int = 300,
                dropout: float = 0.2):
        super(PackageEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.fasttext_embedding_dim = fasttext_embedding_dim
        
        # Find the fasttext model file - look in current dir and data dir
        if not os.path.exists(fasttext_model_path):
            # Try data directory
            alt_path = os.path.join("data", os.path.basename(fasttext_model_path))
            if os.path.exists(alt_path):
                fasttext_model_path = alt_path
            else:
                # Try parent's data directory
                alt_path = os.path.join("..", "data", os.path.basename(fasttext_model_path))
                if os.path.exists(alt_path):
                    fasttext_model_path = alt_path
                else:
                    raise FileNotFoundError(f"FastText model not found at {fasttext_model_path}")
        
        # Load FastText model for Dutch
        print(f"Loading FastText model from {fasttext_model_path}...")
        self.fasttext_model = fasttext.load_model(fasttext_model_path)
        
        # Load SpaCy's Dutch language model for stopwords
        try:
            self.nlp = spacy.load("nl_core_news_sm")
            self.stopwords = self.nlp.Defaults.stop_words
        except:
            print("Dutch language model not found, using blank model with custom stopwords.")
            self.nlp = spacy.blank("nl")
            self.stopwords = set()
         
        # Add custom stopwords specifically for Dutch travel packages
        custom_stopwords = {
            "incl.", "o.b.v.", "bijvoorbeeld", "minstens", "het", "een", "Incl.", 
            "&", "direct", "regio", "o.a.", "én", "easy", "going", "'s", "van", "extra",
            "de", "met", "naar", "voor", "door", "in", "op", "bij", "aan", "uit", "vakantie",
            "reis", "verblijf", "dagen", "nachten", "inclusief", "exclusief", "vanaf"
        }
        self.stopwords = self.stopwords.union(custom_stopwords)
        
        # Initialize a linear layer to transform FastText embeddings to our dimension
        self.fasttext_transform = nn.Linear(fasttext_embedding_dim, embedding_dim)
        
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
        
        # Word-level attention weights
        self.word_attention = nn.Linear(hidden_dim, 1)
        
        # View-level attention weights
        self.view_attention_query = nn.Parameter(torch.zeros(hidden_dim))
        self.view_attention = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def remove_stopwords(self, text: str) -> str:
        """Remove Dutch stopwords from text"""
        if not isinstance(text, str):
            return ""
        
        tokens = text.lower().split()  # Split into words
        filtered_tokens = [word for word in tokens if word not in self.stopwords]  # Remove stopwords
        return " ".join(filtered_tokens)
    
    def get_fasttext_embedding(self, title: str) -> torch.Tensor:
        """Get FastText embedding for a title"""
        # Remove stopwords
        filtered_title = self.remove_stopwords(title)
        if not filtered_title:
            # Return zeros if the title is empty after filtering
            return torch.zeros(self.fasttext_embedding_dim)
        
        # Get FastText vector
        vector = self.fasttext_model.get_sentence_vector(filtered_title)
        return torch.tensor(vector, dtype=torch.float)
    
    def process_batch_titles(self, titles: List[str], main_ids: Optional[List[int]] = None) -> torch.Tensor:
        """
        Process a batch of titles to get embeddings
    
        Args:
            titles (List[str]): List of title strings
            main_ids (Optional[List[int]]): List of corresponding main_ids
    
        Returns:
            torch.Tensor: Transformed embeddings
        """
        # Get FastText embeddings for each title
        embeddings = [self.get_fasttext_embedding(title) for title in titles]
        embeddings_tensor = torch.stack(embeddings)
    
        # Transform to our embedding dimension
        transformed_embeddings = self.fasttext_transform(embeddings_tensor)
    
        # If main_ids are provided, optionally cache or process differently
        if main_ids is not None:
            # You can add caching logic here if needed
            # For example, storing embeddings by main_id in a class-level dictionary
            if not hasattr(self, '_main_id_embeddings'):
                self._main_id_embeddings = {}
        
            for main_id, embedding in zip(main_ids, transformed_embeddings):
                self._main_id_embeddings[main_id] = embedding.detach().numpy()
    
        return transformed_embeddings
        
    def forward(self, titles: List[str], country_ids: torch.Tensor, 
                category_ids: torch.Tensor, theme_ids: torch.Tensor, 
                user_query: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the package encoder
        
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
        
        # Title encoding with FastText and Bi-LSTM
        title_embeddings = self.process_batch_titles(titles).to(device)
        
        # Get contextual title representation using Bi-LSTM
        title_outputs, _ = self.title_lstm(title_embeddings.unsqueeze(1))
        
        # Since we're using the single FastText embedding for each title,
        # we can just squeeze the sequence dimension
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
        ], dim=1)  # [batch_size, 4, hidden_dim]
        
        # View-level attention
        if user_query is not None:
            # Personalized attention with user query
            attention_query = user_query.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        else:
            # Global attention with learned query
            attention_query = self.view_attention_query.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
            
        view_attn_weights = torch.bmm(attention_query, view_representations.transpose(1, 2)).squeeze(1)
        view_attn_weights = F.softmax(view_attn_weights, dim=1).unsqueeze(1)  # [batch_size, 1, 4]
        
        # Apply view-level attention weights
        package_representation = torch.bmm(view_attn_weights, view_representations).squeeze(1)  # [batch_size, hidden_dim]
        
        return package_representation
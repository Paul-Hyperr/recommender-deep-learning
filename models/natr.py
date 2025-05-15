import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class ViewLevelAttention(nn.Module):
    """View-level attention mechanism with optimized implementation"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Use weight tying (in_features = out_features)
        self.attention_project = nn.Linear(hidden_dim, hidden_dim//2)
        self.attention_weights = nn.Linear(hidden_dim//2, 1)
        
    def forward(self, views: list) -> torch.Tensor:
        """Apply attention across different views with improved efficiency"""
        # Batch concatenation is more efficient than stack+sum
        stacked_views = torch.stack(views, dim=2)  # [batch_size, seq_len, num_views, hidden_dim]
        
        # Use projection before attention to reduce computation
        projected_views = torch.relu(self.attention_project(stacked_views))  # Add non-linearity
        attention_scores = self.attention_weights(projected_views).squeeze(-1)
        
        # Stable softmax with better numerical precision
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Use bmm for more efficient batch multiplication
        attention_weights_expanded = attention_weights.unsqueeze(3)  # [batch_size, seq_len, num_views, 1]
        attended = (stacked_views * attention_weights_expanded).sum(dim=2)
        
        return attended


class PackageEncoder(nn.Module):
    """Package encoder with event awareness"""
    def __init__(self, config):
        super().__init__()
        
        # Title encoder
        self.title_encoder = nn.Sequential(
            nn.Linear(config.title_embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Coordinate encoder
        self.coordinate_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, config.hidden_dim)
        )
        
        # Category embeddings
        self.country_embedding = nn.Embedding(config.num_countries, config.embedding_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(config.num_categories, config.embedding_dim, padding_idx=0)
        self.theme_embedding = nn.Embedding(config.num_themes, config.embedding_dim, padding_idx=0)
        
        self.category_encoder = nn.Sequential(
            nn.Linear(config.embedding_dim * 3, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Event type embedding for natural learning
        self.event_embedding = nn.Embedding(5, 32, padding_idx=0)  # 5 event types
        self.event_projection = nn.Linear(32, config.hidden_dim)
        
        # View-level attention
        self.view_attention = ViewLevelAttention(config.hidden_dim)
    
    def forward(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with event awareness"""
        # Handle dimensions
        if len(batch_data['title_embeddings'].shape) == 2:
            # Single package case
            batch_size = batch_data['title_embeddings'].size(0)
            seq_len = 1
            
            title_repr = self.title_encoder(batch_data['title_embeddings']).unsqueeze(1)
            coord_repr = self.coordinate_encoder(batch_data['coordinates']).unsqueeze(1)
            
            country_emb = self.country_embedding(batch_data['country_ids']).unsqueeze(1)
            category_emb = self.category_embedding(batch_data['category_ids']).unsqueeze(1)
            theme_emb = self.theme_embedding(batch_data['theme_ids']).unsqueeze(1)
            
            # Event embeddings if available
            if 'event_types' in batch_data:
                event_emb = self.event_embedding(batch_data['event_types']).unsqueeze(1)
            else:
                event_emb = torch.zeros(batch_size, 1, 32, device=batch_data['title_embeddings'].device)
            
        else:
            # Sequence case
            batch_size, seq_len, _ = batch_data['title_embeddings'].shape
            
            title_flat = batch_data['title_embeddings'].view(-1, batch_data['title_embeddings'].size(-1))
            coord_flat = batch_data['coordinates'].view(-1, 2)
            
            title_repr = self.title_encoder(title_flat).view(batch_size, seq_len, -1)
            coord_repr = self.coordinate_encoder(coord_flat).view(batch_size, seq_len, -1)
            
            country_emb = self.country_embedding(batch_data['country_ids'])
            category_emb = self.category_embedding(batch_data['category_ids'])
            theme_emb = self.theme_embedding(batch_data['theme_ids'])
            
            # Event embeddings
            if 'event_types' in batch_data:
                event_emb = self.event_embedding(batch_data['event_types'])
            else:
                event_emb = torch.zeros(batch_size, seq_len, 32, device=batch_data['title_embeddings'].device)
        
        # Combine categorical embeddings
        cat_combined = torch.cat([country_emb, category_emb, theme_emb], dim=-1)
        cat_repr = self.category_encoder(cat_combined.view(-1, cat_combined.size(-1))).view(batch_size, seq_len, -1)
        
        # Add event information (let model learn importance)
        event_repr = self.event_projection(event_emb)
        title_repr = title_repr + 0.1 * event_repr  # Small initial contribution
        
        # Apply view-level attention
        views = [title_repr, coord_repr, cat_repr]
        unified_repr = self.view_attention(views)
        
        return unified_repr.squeeze(1) if seq_len == 1 else unified_repr


class PackageLevelAttention(nn.Module):
    """Package-level attention mechanism with optimized implementation"""
    def __init__(self, hidden_dim: int, user_embedding_dim: int):
        super().__init__()
        # Use more efficient two-step attention calculation
        self.query_projection = nn.Linear(user_embedding_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.scale_factor = hidden_dim ** 0.5  # Scaling for dot-product attention
        
    def forward(self, 
                sequence_output: torch.Tensor,
                user_embedding: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        batch_size, seq_len, hidden_dim = sequence_output.shape
        
        # Project user embedding to create query vectors
        query = self.query_projection(user_embedding).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Project sequence outputs to create key vectors
        key = self.key_projection(sequence_output)  # [batch_size, seq_len, hidden_dim]
        
        # Compute scaled dot-product attention
        attention_scores = torch.bmm(query, key.transpose(1, 2)).squeeze(1) / self.scale_factor
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Compute weighted sum using batch matrix multiplication
        attended = torch.bmm(attention_weights.unsqueeze(1), sequence_output).squeeze(1)
        
        return attended


class UserEncoder(nn.Module):
    """User encoder with Bi-LSTM and attention"""
    def __init__(self, config):
        super().__init__()
        
        self.user_embedding = nn.Embedding(config.num_users, config.user_embedding_dim, padding_idx=0)
        
        # Bi-LSTM layers
        self.short_term_lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.dropout > 0 else 0
        )
        
        self.long_term_lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.dropout > 0 else 0
        )
        
        # Package-level attention
        self.package_attention = PackageLevelAttention(config.hidden_dim, config.user_embedding_dim)
    
    def forward(self, 
                user_ids: torch.Tensor,
                short_term_repr: torch.Tensor,
                long_term_repr: torch.Tensor,
                short_term_mask: Optional[torch.Tensor] = None,
                long_term_mask: Optional[torch.Tensor] = None):
        
        # Get user embeddings
        user_emb = self.user_embedding(user_ids)
        
        # Process sequences
        short_term_output, _ = self.short_term_lstm(short_term_repr)
        short_term_pref = self.package_attention(short_term_output, user_emb, short_term_mask)
        
        long_term_output, _ = self.long_term_lstm(long_term_repr)
        long_term_pref = self.package_attention(long_term_output, user_emb, long_term_mask)
        
        return short_term_pref, long_term_pref, user_emb


class GatedFusion(nn.Module):
    """Gated fusion network"""
    def __init__(self, config):
        super().__init__()
        
        self.fusion_gate = nn.Sequential(
            nn.Linear(config.hidden_dim * 2 + config.user_embedding_dim, config.hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, 
                short_term_pref: torch.Tensor,
                long_term_pref: torch.Tensor,
                user_embedding: torch.Tensor) -> torch.Tensor:
        
        combined = torch.cat([short_term_pref, long_term_pref, user_embedding], dim=-1)
        gate = self.fusion_gate(combined)
        fused = gate * long_term_pref + (1 - gate) * short_term_pref
        
        return fused


class NATR(nn.Module):
    """
    NATR model with natural event learning capability
    """
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Encoders
        self.package_encoder = PackageEncoder(config)
        self.user_encoder = UserEncoder(config)
        self.gated_fusion = GatedFusion(config)
        
        # Final prediction layer
        self.prediction_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_packages)
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        
        # Extract user IDs
        user_ids = batch['user_id']
        
        # Encode packages with event information
        short_term_repr = self.package_encoder(batch['short_term'])
        long_term_repr = self.package_encoder(batch['long_term'])
        
        # Create masks
        short_term_mask = (batch['short_term']['package_ids'] > 0).float()
        long_term_mask = (batch['long_term']['package_ids'] > 0).float()
        
        # Encode user preferences
        short_term_pref, long_term_pref, user_emb = self.user_encoder(
            user_ids,
            short_term_repr,
            long_term_repr,
            short_term_mask,
            long_term_mask
        )
        
        # Fuse preferences
        user_final_repr = self.gated_fusion(short_term_pref, long_term_pref, user_emb)
        
        # Make predictions
        predictions = self.prediction_head(user_final_repr)
        
        # Encode purchased package
        purchased_repr = self.package_encoder(batch['purchased'])
        
        return {
            'predictions': predictions,
            'user_representation': user_final_repr,
            'purchased_representation': purchased_repr,
            'short_term_preference': short_term_pref,
            'long_term_preference': long_term_pref
        }


class NATRConfig:
    """Configuration for NATR model"""
    def __init__(self,
                 num_users: int,
                 num_packages: int,
                 num_countries: int,
                 num_categories: int,
                 num_themes: int,
                 title_embedding_dim: int = 3072,
                 hidden_dim: int = 256,
                 embedding_dim: int = 128,
                 user_embedding_dim: int = 128,
                 dropout: float = 0.2,
                 max_short_term: int = 10,
                 max_long_term: int = 20):
        
        self.num_users = num_users
        self.num_packages = num_packages
        self.num_countries = num_countries
        self.num_categories = num_categories
        self.num_themes = num_themes
        self.title_embedding_dim = title_embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.user_embedding_dim = user_embedding_dim
        self.dropout = dropout
        self.max_short_term = max_short_term
        self.max_long_term = max_long_term
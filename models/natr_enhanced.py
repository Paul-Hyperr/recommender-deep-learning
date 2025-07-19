import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from typing import Dict, Optional, Tuple

class ViewLevelAttention(nn.Module):
    """Enhanced multi-head attention-based view fusion with personalized attention using user embeddings"""
    def __init__(self, hidden_dim: int, user_embedding_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.user_embedding_dim = user_embedding_dim
        
        # Multi-head attention for view fusion - increased number of heads
        self.view_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,  # Increased for more expressive patterns
            dropout=0.1,
            batch_first=True
        )
        
        # Personalized view importance scoring - incorporates user preferences
        self.view_importance = nn.Sequential(
            nn.Linear(hidden_dim + user_embedding_dim, 128),  # Increased capacity for user info
            nn.LayerNorm(128),  # Added normalization for stable learning
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Enhanced view-wise scaling factors for 6 views
        # Initialize with different weights to encourage learning diverse patterns
        # [title, coordinates, country, category/theme, price, events]
        initial_weights = torch.tensor([1.2, 1.0, 0.8, 0.8, 1.0, 1.1])
        self.view_scalars = nn.Parameter(initial_weights)  # Different initial weights for views
        
        # Detect number of views at runtime - updated for 6 views
        self.num_expected_views = 6  # 6 views with events as separate view (no time view)
        
        # Final fusion with residual connection and layer norm
        # Linear size adapts to the actual number of views used
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * self.num_expected_views, hidden_dim * 2),  # Adaptive to num views
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, views: list, user_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Fuse views with personalized attention based on user preferences
        
        Args:
            views: List of view tensors [batch_size, seq_len, hidden_dim]
            user_embedding: User preference vector [batch_size, user_embedding_dim]
        """
        # Process all views to ensure same dimensionality
        processed_views = []
        
        # Get sequence length for processing
        seq_len = views[0].size(1) if views[0].dim() > 2 else 1
        
        for view_idx, view in enumerate(views):
            # Handle cases where the view has different dimensions than hidden_dim
            if view.size(-1) != self.hidden_dim:
                # Pad or trim the last dimension to match hidden_dim
                if view.size(-1) < self.hidden_dim:
                    # Pad with zeros
                    padding = torch.zeros_like(view).repeat(1, 1, self.hidden_dim // view.size(-1))
                    view = torch.cat([view, padding[..., :(self.hidden_dim - view.size(-1))]], dim=-1)
                else:
                    # Trim to match hidden_dim
                    view = view[..., :self.hidden_dim]
            
            # Apply view-specific scaling (learned parameter)
            # Simple linear scaling without sigmoid constraint  
            if view_idx < len(self.view_scalars):
                scale_factor = self.view_scalars[view_idx]
                view = view * scale_factor
                
            processed_views.append(view)
        
        # Calculate personalized importance scores for each view
        view_scores = []
        
        # Prepare user embedding for attention calculation
        if user_embedding is not None:
            # Expand user embedding to match sequence length
            if seq_len > 1:
                user_emb_expanded = user_embedding.unsqueeze(1).expand(-1, seq_len, -1)
            else:
                user_emb_expanded = user_embedding.unsqueeze(1)
        
        for view in processed_views:
            if user_embedding is not None:
                # Concatenate view representation with user embedding for personalized attention
                view_with_user = torch.cat([view, user_emb_expanded], dim=-1)
                score = self.view_importance(view_with_user)  # Batch x Seq x 1
            else:
                # Fallback to non-personalized attention if no user embedding provided
                # Pad with zeros to match expected input size
                padding = torch.zeros(*view.shape[:-1], self.user_embedding_dim, device=view.device)
                view_with_padding = torch.cat([view, padding], dim=-1)
                score = self.view_importance(view_with_padding)
            view_scores.append(score)
        
        # Softmax across views to get attention distribution
        view_scores_cat = torch.cat(view_scores, dim=-1)  # Batch x Seq x NumViews
        view_weights = F.softmax(view_scores_cat, dim=-1)
        
        # Apply attention weights
        weighted_views = []
        for i, view in enumerate(processed_views):
            weighted_views.append(view * view_weights[..., i:i+1])
        
        # Concatenate all views - handle potential mismatch with expected fusion layer
        combined = torch.cat(processed_views, dim=-1)
        
        # If number of views doesn't match what fusion expects, adapt the tensor
        expected_size = self.hidden_dim * self.num_expected_views
        actual_size = combined.size(-1)
        
        if actual_size < expected_size:
            # Pad with zeros to match expected size
            padding_size = expected_size - actual_size
            padding = torch.zeros(*combined.shape[:-1], padding_size, device=combined.device)
            combined = torch.cat([combined, padding], dim=-1)
        elif actual_size > expected_size:
            # Trim to match expected size
            combined = combined[..., :expected_size]
        
        # Apply fusion
        fused = self.fusion(combined)
        
        # Add multi-head self-attention for complex interactions
        fused_seq = fused.unsqueeze(1) if fused.dim() == 2 else fused
        attended_fused, _ = self.view_attention(fused_seq, fused_seq, fused_seq)
        
        # Add weighted views for stronger gradient flow (residual from each view)
        combined_weighted = torch.zeros_like(fused)
        for i, weighted_view in enumerate(weighted_views):
            # Only add if dimensions match
            if weighted_view.size(-1) == combined_weighted.size(-1):
                combined_weighted = combined_weighted + weighted_view
        
        # Residual connections from both raw fusion and attention
        final_repr = self.layer_norm(fused + attended_fused + 0.1 * combined_weighted)
        
        return final_repr


class PackageEncoder(nn.Module):
    """Enhanced package encoder with events as separate view and temporal information"""
    def __init__(self, config):
        super().__init__()
        
        # Title encoder
        self.title_encoder = nn.Sequential(
            nn.Linear(config.title_embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Enhanced coordinate encoder with more capacity for geographic info
        self.coordinate_encoder = nn.Sequential(
            nn.Linear(2, 128),  # Increased from 64 to 128
            nn.ReLU(),
            nn.Dropout(config.dropout * 0.5),  # Lower dropout for geographical features
            nn.Linear(128, 128),  # Added intermediate layer
            nn.ReLU(),
            nn.Linear(128, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim)  # Add normalization to stabilize
        )
        
        # Geographic context attention - learns regional patterns
        self.geo_attention = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Enhanced price encoder with personalized user bias
        self.user_price_bias = nn.Embedding(config.num_users, 32, padding_idx=0)
        self.price_encoder = nn.Sequential(
            nn.Linear(1 + 32, 64),  # price + user price bias
            nn.ReLU(),
            nn.Dropout(config.dropout * 0.5),  # Lower dropout for numerical features
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim)  # Stabilize price representations
        )
        
        # Category embeddings
        self.country_embedding = nn.Embedding(config.num_countries, config.embedding_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(config.num_categories, config.embedding_dim, padding_idx=0)
        self.theme_embedding = nn.Embedding(config.num_themes, config.embedding_dim, padding_idx=0)
        
        # Separate encoder for country
        self.country_encoder = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Updated category encoder that combines only category and theme
        self.category_encoder = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # ENHANCED: Separate event processing as its own view
        # Event type embedding with more dimensions for richer representation
        self.event_embedding = nn.Embedding(5, 64, padding_idx=0)  # Increased from 32 to 64
        
        # Enhanced event sequence encoder with LSTM for temporal patterns
        self.event_lstm = nn.LSTM(
            input_size=64,
            hidden_size=config.hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
            num_layers=1,
            dropout=config.dropout if config.dropout > 0 else 0
        )
        
        # Event attention layer to focus on important event patterns
        self.event_attention = nn.Sequential(
            nn.Linear(config.hidden_dim + config.user_embedding_dim, 64),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # Final event projection to hidden_dim
        self.event_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # View-level attention - updated for 6 views including separate events (no time view)
        self.view_attention = ViewLevelAttention(config.hidden_dim, config.user_embedding_dim)
    
    # Removed set_global_temporal_params as time view is no longer used
    
    def forward(self, batch_data: Dict[str, torch.Tensor], user_embedding: Optional[torch.Tensor] = None, user_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with events as separate view and temporal information
        
        Args:
            batch_data: Dictionary containing package features
            user_embedding: Optional user embeddings for personalized attention
        """
        # Handle dimensions
        if len(batch_data['title_embeddings'].shape) == 2:
            # Single package case
            batch_size = batch_data['title_embeddings'].size(0)
            seq_len = 1
            
            title_repr = self.title_encoder(batch_data['title_embeddings']).unsqueeze(1)
            
            # Enhanced coordinate processing for single package
            coord_base = self.coordinate_encoder(batch_data['coordinates'])
            
            # Apply geo-attention to emphasize geographic importance
            geo_score = torch.sigmoid(self.geo_attention(coord_base))  # 0-1 score of geographic relevance
            coord_repr = (coord_base * (1.0 + geo_score)).unsqueeze(1)  # Boost with geo score
            
            # Process price data with personalized user bias
            if 'normalized_prices' in batch_data:
                # Use pre-normalized prices if available
                price_data = batch_data['normalized_prices'].float().unsqueeze(-1)
            elif 'prices' in batch_data:
                # Normalize prices on-the-fly
                price_data = batch_data['prices'].float().unsqueeze(-1).clamp(min=0.01)
                # Apply logarithmic scaling to compress price range
                price_data = torch.log(price_data + 1.0)
            else:
                # No price data available - use zeros
                price_data = torch.zeros(batch_size, 1, device=batch_data['title_embeddings'].device)
            
            # Get user price bias if user_ids available
            if user_ids is not None:
                user_price_bias = self.user_price_bias(user_ids).unsqueeze(1)  # [batch_size, 1, 32]
                # Combine price with user bias
                price_data_expanded = price_data.unsqueeze(1)  # [batch_size, 1, 1]
                price_with_bias = torch.cat([price_data_expanded, user_price_bias], dim=-1)  # [batch_size, 1, 33]
                price_repr = self.price_encoder(price_with_bias)
            else:
                # Fallback without user bias - pad with zeros
                price_padding = torch.zeros(batch_size, 1, 32, device=batch_data['title_embeddings'].device)
                price_data_expanded = price_data.unsqueeze(1)  # [batch_size, 1, 1]
                price_with_padding = torch.cat([price_data_expanded, price_padding], dim=-1)
                price_repr = self.price_encoder(price_with_padding)
            
            country_emb = self.country_embedding(batch_data['country_ids']).unsqueeze(1)
            category_emb = self.category_embedding(batch_data['category_ids']).unsqueeze(1)
            theme_emb = self.theme_embedding(batch_data['theme_ids']).unsqueeze(1)
            
            # ENHANCED: Process events as separate view
            if 'event_types' in batch_data:
                event_emb = self.event_embedding(batch_data['event_types']).unsqueeze(1)
                # For single package, just encode directly
                event_encoded, _ = self.event_lstm(event_emb)
                event_repr = self.event_projection(event_encoded)
            else:
                event_repr = torch.zeros(batch_size, 1, self.event_projection.out_features, 
                                       device=batch_data['title_embeddings'].device)
            
            # Time information removed - handled in package attention instead
            
        else:
            # Sequence case
            batch_size, seq_len, _ = batch_data['title_embeddings'].shape
            
            title_flat = batch_data['title_embeddings'].view(-1, batch_data['title_embeddings'].size(-1))
            coord_flat = batch_data['coordinates'].view(-1, 2)
            
            title_repr = self.title_encoder(title_flat).view(batch_size, seq_len, -1)
            
            # Enhanced coordinate processing for sequence case
            coord_base = self.coordinate_encoder(coord_flat).view(batch_size, seq_len, -1)
            
            # Apply geo-attention to emphasize geographic importance (for sequences)
            geo_score = torch.sigmoid(self.geo_attention(coord_base))  # Batch x Seq x 1
            coord_repr = coord_base * (1.0 + geo_score)  # Boost with geographic relevance score
            
            # Process price data for sequences with personalized user bias
            if 'normalized_prices' in batch_data:
                # Use pre-normalized prices if available
                price_data = batch_data['normalized_prices'].float().unsqueeze(-1)
            elif 'prices' in batch_data:
                # Normalize prices on-the-fly
                price_data = batch_data['prices'].float().unsqueeze(-1).clamp(min=0.01)
                # Apply logarithmic scaling to compress price range
                price_data = torch.log(price_data + 1.0)
            else:
                # No price data available - use zeros
                price_data = torch.zeros(batch_size, seq_len, 1, device=batch_data['title_embeddings'].device)
            
            # Get user price bias if user_ids available
            if user_ids is not None:
                user_price_bias = self.user_price_bias(user_ids).unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, 32]
                # Combine price with user bias
                price_with_bias = torch.cat([price_data, user_price_bias], dim=-1)
                price_flat = price_with_bias.view(-1, 1 + 32)
                price_repr = self.price_encoder(price_flat).view(batch_size, seq_len, -1)
            else:
                # Fallback without user bias - pad with zeros
                price_padding = torch.zeros(batch_size, seq_len, 32, device=batch_data['title_embeddings'].device)
                price_with_padding = torch.cat([price_data, price_padding], dim=-1)
                price_flat = price_with_padding.view(-1, 1 + 32)
                price_repr = self.price_encoder(price_flat).view(batch_size, seq_len, -1)
            
            country_emb = self.country_embedding(batch_data['country_ids'])
            category_emb = self.category_embedding(batch_data['category_ids'])
            theme_emb = self.theme_embedding(batch_data['theme_ids'])
            
            # ENHANCED: Process events as separate sequential view
            if 'event_types' in batch_data:
                event_emb = self.event_embedding(batch_data['event_types'])
                
                # Process event sequences with LSTM to capture temporal patterns
                event_encoded, _ = self.event_lstm(event_emb)
                
                # Apply personalized attention to focus on important event patterns
                if user_embedding is not None:
                    # Expand user embedding to match sequence length
                    user_emb_for_event = user_embedding.unsqueeze(1).expand(-1, seq_len, -1)
                    
                    # Concatenate event representation with user embedding
                    event_with_user = torch.cat([event_encoded, user_emb_for_event], dim=-1)
                    event_attn_scores = self.event_attention(event_with_user)  # [batch_size, seq_len, 1]
                    event_attn_weights = F.softmax(event_attn_scores, dim=1)
                    event_weighted = event_encoded * event_attn_weights
                else:
                    # Fallback to non-personalized processing
                    event_weighted = event_encoded
                
                # Final projection to hidden_dim
                event_repr = self.event_projection(event_weighted)
            else:
                event_repr = torch.zeros(batch_size, seq_len, self.event_projection.out_features, 
                                       device=batch_data['title_embeddings'].device)
            
            # Time information removed - handled in package attention instead
        
        # Process country embeddings separately
        country_repr = self.country_encoder(country_emb.view(-1, country_emb.size(-1))).view(batch_size, seq_len, -1)
        
        # Combine only category and theme embeddings
        cat_combined = torch.cat([category_emb, theme_emb], dim=-1)
        cat_repr = self.category_encoder(cat_combined.view(-1, cat_combined.size(-1))).view(batch_size, seq_len, -1)
        
        # Apply view-level attention with 6 views including separate events
        # The order of views must match the order of view_scalars in ViewLevelAttention:
        # [title, coordinates, country, category/theme, price, events]
        views = [
            title_repr,            # Content-based similarity (view index 0 - weight 1.2)
            coord_repr,            # Geographic proximity (view index 1 - weight 1.0)
            country_repr,          # Country representation (view index 2 - weight 0.8)
            cat_repr,              # Category/theme preferences (view index 3 - weight 0.8)
            price_repr,            # Price-based recommendations (view index 4 - weight 1.0)
            event_repr             # Event sequential patterns (view index 5 - weight 1.1)
        ]
        unified_repr = self.view_attention(views, user_embedding)
        
        # Always return 3D tensor to maintain consistency
        return unified_repr


class PackageLevelAttention(nn.Module):
    """Enhanced package-level attention with multi-head attention, recency, and debuggable outputs"""
    def __init__(self, hidden_dim: int, user_embedding_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Multi-head attention for more expressive interactions
        self.num_heads = 4
        self.head_dim = hidden_dim // self.num_heads
        
        # Query, key, value projections 
        self.query_projection = nn.Linear(user_embedding_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale_factor = self.head_dim ** 0.5  # Per-head scaling
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Enhanced temporal bias module with more capacity
        self.time_bias = nn.Sequential(
            nn.Linear(1, 64),  # Increased from 32 to 64
            nn.ReLU(),
            nn.Dropout(0.1),  # Added dropout
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Save attention weights for analysis
        self.save_attention = False
        self.last_attention_weights = None
        
    def forward(self, 
                sequence_output: torch.Tensor,
                user_embedding: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        batch_size, seq_len, hidden_dim = sequence_output.shape
        
        # Project user embedding to create query vectors - now multi-head
        query = self.query_projection(user_embedding).view(
            batch_size, 1, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)  # [batch_size, num_heads, 1, head_dim]
        
        # Project sequence outputs for keys and values - multi-head
        key = self.key_projection(sequence_output).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        
        value = self.value_projection(sequence_output).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Compute scaled dot-product attention for each head
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale_factor  
        # [batch_size, num_heads, 1, seq_len]
        
        # Check for numerical issues in attention scores
        if torch.isnan(attention_scores).any() or torch.isinf(attention_scores).any():
            # Clamp values to prevent overflow
            attention_scores = torch.clamp(attention_scores, min=-100, max=100)
        
        # Apply temporal bias if timestamps are provided
        if timestamps is not None:
            # Ensure timestamps are floating point and valid
            timestamps = timestamps.float().clamp(min=1.0)  # Avoid invalid timestamps
            
            # Calculate time deltas from the most recent item (assumed to be at the end)
            # Get the maximum timestamp in each sequence (most recent)
            max_time = torch.max(timestamps, dim=1, keepdim=True)[0]
            
            # Calculate delta_t = max_time - timestamp for each item
            # This gives us how far back in time each item is from the most recent
            delta_t = max_time - timestamps + 1.0  # Add 1 to avoid log(0)
            
            # Apply -log(delta_t) for recency bias
            # Items closer in time (smaller delta_t) will have larger (less negative) values
            log_delta_t = -torch.log(delta_t).unsqueeze(-1)  # [batch_size, seq_len, 1]
            
            # Process through time bias network to get attention bias
            recency_bias = self.time_bias(log_delta_t).squeeze(-1)  # [batch_size, seq_len]
            
            # Add recency bias to attention scores - expand for multi-head
            recency_bias = recency_bias.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            attention_scores = attention_scores + recency_bias
        
        # Apply mask if provided
        if mask is not None:
            # Check for sequences that are entirely padded
            mask_sum = mask.sum(dim=1)  # Count valid positions per batch
            
            # Expand mask for multi-head attention
            expanded_mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            attention_scores = attention_scores.masked_fill(expanded_mask == 0, -1e9)
            
            # Handle edge case where entire sequence is padded
            # In this case, all scores will be -inf, leading to NaN after softmax
            for batch_idx in range(batch_size):
                if mask_sum[batch_idx] == 0:
                    # Entire sequence is padded - set uniform attention
                    # This prevents NaN by giving equal weight to all positions
                    attention_scores[batch_idx, :, :, :] = 0.0
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, num_heads, 1, seq_len]
        
        # Final safety check for NaN in attention weights
        if torch.isnan(attention_weights).any():
            # This should not happen with the fix above, but just in case
            # Replace NaN with uniform distribution
            nan_mask = torch.isnan(attention_weights)
            uniform_attn = torch.ones_like(attention_weights) / seq_len
            attention_weights = torch.where(nan_mask, uniform_attn, attention_weights)
        
        # Save attention weights if requested
        if self.save_attention:
            self.last_attention_weights = attention_weights.detach()
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, value)  # [batch_size, num_heads, 1, head_dim]
        
        # Reshape and combine heads
        attended = attended.permute(0, 2, 1, 3).contiguous()  # [batch_size, 1, num_heads, head_dim]
        attended = attended.view(batch_size, 1, hidden_dim)  # [batch_size, 1, hidden_dim]
        
        # Project to output space
        output = self.output_projection(attended)
        
        # Add residual connection and layer norm
        if seq_len > 1:  # Only apply if we have a real sequence
            # Get a weighted average of sequence outputs using the attention weights
            flattened_weights = attention_weights.mean(dim=1).squeeze(1)  # [batch_size, seq_len]
            weighted_sequence = torch.bmm(flattened_weights.unsqueeze(1), sequence_output)  # [batch_size, 1, hidden_dim]
            output = output + weighted_sequence
            
        output = self.layer_norm(output)
        
        # Return without squeezing to maintain shape consistency
        # output shape: [batch_size, 1, hidden_dim]
        return output


class UserEncoder(nn.Module):
    """User encoder with multi-layer Bi-LSTM, cross-attention, and layer normalization for stability"""
    def __init__(self, config):
        super().__init__()
        
        self.user_embedding = nn.Embedding(config.num_users, config.user_embedding_dim, padding_idx=0)
        # Initialize user embeddings with Xavier/Glorot initialization for better differentiation
        with torch.no_grad():
            # Use Xavier uniform initialization for better initial differentiation between users
            nn.init.xavier_uniform_(self.user_embedding.weight)
            # Ensure padding embedding stays at zero
            self.user_embedding.weight[0] = 0
            
            # Double-check no NaN values after initialization
            if torch.isnan(self.user_embedding.weight).any():
                print(f"WARNING: NaN found in user embedding after initialization!")
                nan_indices = torch.where(torch.isnan(self.user_embedding.weight).any(dim=1))[0]
                print(f"NaN at indices: {nan_indices.tolist()[:10]}...")  # Show first 10
                # Replace NaN with small random values
                self.user_embedding.weight[nan_indices] = torch.randn(len(nan_indices), config.user_embedding_dim) * 0.01
        
        # Increased to multi-layer Bi-LSTM with layer normalization
        self.short_term_lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
            num_layers=2,  # Increased from 1 to 2 layers
            dropout=config.dropout if config.dropout > 0 else 0
        )
        
        self.long_term_lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
            num_layers=2,  # Increased from 1 to 2 layers
            dropout=config.dropout if config.dropout > 0 else 0
        )
        
        # Layer normalization for stability
        self.short_term_norm = nn.LayerNorm(config.hidden_dim)
        self.long_term_norm = nn.LayerNorm(config.hidden_dim)
        
        # Package-level attention with recency awareness
        self.package_attention = PackageLevelAttention(config.hidden_dim, config.user_embedding_dim)
        
        # Cross-attention between short and long term representations
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim, 
            num_heads=4,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Final fusion layer after cross-attention
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
    
    def forward(self, 
                user_ids: torch.Tensor,
                short_term_repr: torch.Tensor,
                long_term_repr: torch.Tensor,
                short_term_mask: Optional[torch.Tensor] = None,
                long_term_mask: Optional[torch.Tensor] = None,
                short_term_timestamps: Optional[torch.Tensor] = None,
                long_term_timestamps: Optional[torch.Tensor] = None):
        
        # Get user embeddings
        user_emb = self.user_embedding(user_ids)
        
        # Check if embeddings are NaN right after lookup
        if torch.isnan(user_emb).any():
            print(f"ERROR: User embeddings are NaN right after lookup!")
            nan_mask = torch.isnan(user_emb).any(dim=1)
            nan_user_ids = user_ids[nan_mask]
            print(f"User IDs that produce NaN: {nan_user_ids}")
            print(f"Embedding table size: {self.user_embedding.num_embeddings}")
            print(f"Max ID with NaN: {nan_user_ids.max().item() if len(nan_user_ids) > 0 else 'N/A'}")
            
            # Check if these IDs are within bounds
            if len(nan_user_ids) > 0 and nan_user_ids.max() >= self.user_embedding.num_embeddings:
                print(f"ERROR: User ID {nan_user_ids.max().item()} >= num_embeddings {self.user_embedding.num_embeddings}")
            
            # Replace with small random values
            user_emb = torch.where(nan_mask.unsqueeze(1), torch.randn_like(user_emb) * 0.01, user_emb)
        
        # Process sequences with multi-layer LSTM
        short_term_output, _ = self.short_term_lstm(short_term_repr)
        short_term_output = self.short_term_norm(short_term_output)  # Apply layer normalization
        
        long_term_output, _ = self.long_term_lstm(long_term_repr)
        long_term_output = self.long_term_norm(long_term_output)  # Apply layer normalization
        
        # Apply cross-attention between sequences to capture interactions
        # Short-term as query, long-term as key/value
        # Handle case where long-term is entirely masked to prevent NaN
        if long_term_mask is not None:
            long_term_valid_count = long_term_mask.sum(dim=1)
            has_valid_long_term = long_term_valid_count > 0
            
            # Only apply cross-attention if there are valid long-term items
            if has_valid_long_term.all():
                # All batches have valid long-term, proceed normally
                cross_attended_short, _ = self.cross_attention(
                    short_term_output, 
                    long_term_output, 
                    long_term_output,
                    key_padding_mask=~long_term_mask.bool()
                )
                enhanced_short_term = short_term_output + cross_attended_short
            else:
                # Some batches have no valid long-term items
                # Process only batches with valid long-term separately
                enhanced_short_term = short_term_output.clone()
                
                if has_valid_long_term.any():
                    # Process batches with valid long-term
                    valid_indices = torch.where(has_valid_long_term)[0]
                    cross_attended_valid, _ = self.cross_attention(
                        short_term_output[valid_indices], 
                        long_term_output[valid_indices], 
                        long_term_output[valid_indices],
                        key_padding_mask=~long_term_mask[valid_indices].bool()
                    )
                    enhanced_short_term[valid_indices] = short_term_output[valid_indices] + cross_attended_valid
                
                # Batches with no valid long-term use original short-term (no enhancement)
        else:
            # No mask provided, apply cross-attention normally
            cross_attended_short, _ = self.cross_attention(
                short_term_output, 
                long_term_output, 
                long_term_output
            )
            enhanced_short_term = short_term_output + cross_attended_short
        
        # Apply package-level attention as before, but with enhanced representations
        short_term_pref = self.package_attention(
            enhanced_short_term, 
            user_emb, 
            short_term_mask,
            short_term_timestamps
        )
        
        long_term_pref = self.package_attention(
            long_term_output, 
            user_emb, 
            long_term_mask,
            long_term_timestamps
        )
        
        return short_term_pref, long_term_pref, user_emb


class GatedFusion(nn.Module):
    """Enhanced gated fusion network with better initialization and contextual balance"""
    def __init__(self, config):
        super().__init__()
        
        # Transformation layers for each input to enhance representation before fusion
        self.short_term_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.long_term_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.user_projection = nn.Linear(config.user_embedding_dim, config.hidden_dim)
        
        # More expressive fusion gate with deeper network
        self.fusion_gate = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),  # Normalize for stable learning
            nn.ReLU(),
            nn.Dropout(config.dropout),  # Add regularization
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.Sigmoid()  # Final activation for gate values
        )
        
        # Additional context-aware balance parameter to learn optimal fusion ratio
        # This helps the model adjust the initial balance between short vs long term
        self.balance_parameter = nn.Parameter(torch.tensor([0.5]))  # Initialize at 0.5 (equal weight)
        
        # Output projection 
        self.output_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        
    def forward(self, 
                short_term_pref: torch.Tensor,
                long_term_pref: torch.Tensor,
                user_embedding: torch.Tensor) -> torch.Tensor:
        
        # Project each component for better representation
        short_term_proj = self.short_term_projection(short_term_pref)
        long_term_proj = self.long_term_projection(long_term_pref)
        user_proj = self.user_projection(user_embedding)
        
        # Combine for gate calculation
        combined = torch.cat([short_term_proj, long_term_proj, user_proj], dim=-1)
        
        # Calculate dynamic gate values for each dimension of the hidden state
        gate = self.fusion_gate(combined)
        
        # Apply learned balance parameter to adjust initial preference
        # Sigmoid ensures balance_parameter is between 0-1
        initial_balance = torch.sigmoid(self.balance_parameter)
        adjusted_gate = initial_balance * gate + (1 - initial_balance) * (1 - gate)
        
        # Apply gated fusion with the adjusted gate
        temporal_fused = adjusted_gate * long_term_proj + (1 - adjusted_gate) * short_term_proj
        
        # Proper weighted combination: 80% temporal fusion, 20% user identity
        # This ensures weights sum to 1.0 for better normalization
        fused = 0.8 * temporal_fused + 0.2 * user_proj
        
        # Final projection and normalization
        fused = self.output_projection(fused)
        fused = self.layer_norm(fused)
        
        return fused


class NATREnhanced(nn.Module):
    """
    Enhanced NATR model with events as separate view (6 views total), natural event learning capability,
    temporal awareness through package attention, and analysis capabilities for attention patterns
    """
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Encoders
        self.package_encoder = PackageEncoder(config)
        self.user_encoder = UserEncoder(config)
        self.gated_fusion = GatedFusion(config)
        
        # Final prediction layer with larger hidden layer
        self.prediction_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),  # Increased capacity
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.hidden_dim * 2),  # Use LayerNorm instead of BatchNorm for stability
            nn.Linear(config.hidden_dim * 2, config.num_packages)
        )
        
        # Add user-specific transformation to ensure personalization
        self.user_transform = nn.Linear(config.user_embedding_dim, config.num_packages)
        # Initialize with larger scale to ensure user signals aren't overwhelmed
        with torch.no_grad():
            nn.init.xavier_uniform_(self.user_transform.weight, gain=0.5)  # Increased from 0.1 to 0.5
            nn.init.zeros_(self.user_transform.bias)
            
        # Flag to track user learning
        self.monitor_user_learning = True
        
        # Popularity modeling
        # Learnable popularity bias for each package
        self.popularity_bias = nn.Parameter(torch.zeros(config.num_packages))
        
        # Recent popularity boost - uses only last month of training data
        self.recent_popularity_weight = getattr(config, 'recent_popularity_weight', 0.2)
        self.recent_popularity_scores = None  # Will be set during training
        
        # Removed learnable popularity embeddings to avoid redundancy with recent popularity
        # Recent popularity boost provides sufficient popularity signal
        
        # Debug mode for capturing attention weights
        self.debug_mode = False
        self.attention_weights = {}
    
    def set_global_temporal_params(self, dataset_start_time: float, dataset_end_time: float):
        """
        Set global temporal parameters for all encoders
        
        Args:
            dataset_start_time: Unix timestamp of earliest event in dataset
            dataset_end_time: Unix timestamp of latest event in dataset
        """
        self.package_encoder.set_global_temporal_params(dataset_start_time, dataset_end_time)
    
    def initialize_popularity(self, item_frequencies):
        """Initialize popularity bias and embeddings based on item frequencies
        
        Args:
            item_frequencies: Tensor of shape [num_packages] with frequency counts
        """
        with torch.no_grad():
            # Normalize frequencies and apply log transform with smoothing
            # Add smoothing to avoid extreme negative values for unpurchased items
            smoothed_freq = item_frequencies + 0.1  # Add small count to all items
            normalized_freq = smoothed_freq / smoothed_freq.sum()
            log_popularity = torch.log(normalized_freq)
            
            # Center and scale to reasonable range
            log_popularity = log_popularity - log_popularity.mean()
            log_popularity = log_popularity / (log_popularity.std() + 1e-8)
            
            # Initialize bias with scaled log-popularity
            # Scale down significantly to avoid overwhelming the learned representations
            self.popularity_bias.data = log_popularity * 0.001  # Very small scaling to not dominate
            
            # Learnable popularity embeddings removed - using recent popularity instead
    
    def set_recent_popularity(self, recent_scores):
        """Set recent popularity scores from last month of training data
        
        Args:
            recent_scores: Tensor of shape [num_packages] with recent popularity scores
        """
        if recent_scores is not None:
            # Convert to same device as model
            device = next(self.parameters()).device
            self.recent_popularity_scores = recent_scores.to(device)
    
    def enable_debug(self, enable=True):
        """Enable debug mode to capture attention weights"""
        self.debug_mode = enable
        # Enable saving attention weights in attention modules
        for module in self.modules():
            if hasattr(module, 'save_attention'):
                module.save_attention = enable
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass with events as separate view, temporal information and attention tracking"""
        
        # Extract user IDs
        user_ids = batch['user_id']
        
        # Debug: Check user ID range
        max_user_id = user_ids.max().item()
        if max_user_id >= self.user_encoder.user_embedding.num_embeddings:
            print(f"ERROR: User ID {max_user_id} >= num_embeddings {self.user_encoder.user_embedding.num_embeddings}")
            print(f"User IDs sample: {user_ids[:10].tolist()}")
            # Clamp to valid range
            user_ids = torch.clamp(user_ids, 0, self.user_encoder.user_embedding.num_embeddings - 1)
        
        # Validate user IDs are within bounds
        if user_ids.max() >= self.user_encoder.user_embedding.num_embeddings:
            print(f"ERROR: User ID out of bounds! Max ID: {user_ids.max().item()}, num_embeddings: {self.user_encoder.user_embedding.num_embeddings}")
            # Clamp to valid range
            user_ids = torch.clamp(user_ids, 0, self.user_encoder.user_embedding.num_embeddings - 1)
        
        # Get user embeddings for personalized attention
        user_emb = self.user_encoder.user_embedding(user_ids)
        
        # Check for NaN values
        if torch.isnan(user_emb).any():
            print(f"WARNING: NaN detected in user embeddings! User IDs: {user_ids[:5]}")
            print(f"  Embedding shape: {user_emb.shape}, num_embeddings: {self.user_encoder.user_embedding.num_embeddings}")
            print(f"  Max user ID in batch: {user_ids.max().item()}, Min: {user_ids.min().item()}")
            # Initialize with small random values instead of zeros
            user_emb = torch.randn_like(user_emb) * 0.01
        
        # Encode packages with event information and personalized attention
        short_term_repr = self.package_encoder(batch['short_term'], user_emb, user_ids)
        long_term_repr = self.package_encoder(batch['long_term'], user_emb, user_ids)
        
        # Create masks
        short_term_mask = (batch['short_term']['package_ids'] > 0).float()
        long_term_mask = (batch['long_term']['package_ids'] > 0).float()
        
        # Get timestamps if available
        short_term_timestamps = batch['short_term'].get('timestamps', None)
        long_term_timestamps = batch['long_term'].get('timestamps', None)
        
        # Encode user preferences with temporal information
        short_term_pref, long_term_pref, user_emb = self.user_encoder(
            user_ids,
            short_term_repr,
            long_term_repr,
            short_term_mask,
            long_term_mask,
            short_term_timestamps,
            long_term_timestamps
        )
        
        # Fuse preferences
        # Squeeze out the sequence dimension (which should be 1) from attention outputs
        short_term_pref_2d = short_term_pref.squeeze(1)
        long_term_pref_2d = long_term_pref.squeeze(1)
        user_final_repr = self.gated_fusion(short_term_pref_2d, long_term_pref_2d, user_emb)
        
        # Make predictions with user-specific component
        predictions = self.prediction_head(user_final_repr)
        
        # Add user-specific bias with adaptive weighting
        user_bias = self.user_transform(user_emb)
        # Use tanh to bound the user bias and prevent extreme personalization
        user_bias = torch.tanh(user_bias / 5.0) * 5.0  # Bound between -5 and 5
        
        # Adaptive weighting: stronger personalization for users with more history
        # This helps maintain accuracy for users with clear preferences
        has_history = batch.get('has_short_term', torch.ones(user_emb.size(0), dtype=torch.bool, device=user_emb.device))
        personalization_weight = torch.where(has_history.unsqueeze(1), 0.3, 0.1)  # 0.3 for users with history, 0.1 for cold start
        predictions = predictions + personalization_weight * user_bias
        
        # Enhanced debug: Monitor user learning more frequently and thoroughly
        if self.training and self.monitor_user_learning and torch.rand(1).item() < 0.001:  # Back to less frequent monitoring
            pred_std = predictions.std(dim=0).mean().item()
            user_repr_std = user_final_repr.std(dim=0).mean().item()
            
            # Check user bias before tanh transformation
            raw_user_bias = self.user_transform(user_emb)
            raw_bias_std = raw_user_bias.std(dim=0).mean().item()
            raw_bias_range = raw_user_bias.max().item() - raw_user_bias.min().item()
            tanh_bias_std = user_bias.std(dim=0).mean().item()
            
            # Check user embedding variance
            user_emb_std = user_emb.std(dim=0).mean().item()
            user_emb_range = user_emb.max().item() - user_emb.min().item()
            
            print(f"🔍 User Learning Monitor (Enhanced NATR with 6 views):")
            print(f"  Prediction std: {pred_std:.4f}")
            print(f"  User embedding std: {user_emb_std:.4f}, range: {user_emb_range:.4f}")
            print(f"  Raw user bias std: {raw_bias_std:.4f}, range: {raw_bias_range:.4f}")
            print(f"  Tanh user bias std: {tanh_bias_std:.4f}")
            
            # Simple diversity check (much faster)
            if len(user_emb) > 1:
                # Quick check: just compare first two users if they exist
                if len(predictions) >= 2:
                    pred_diff = torch.abs(predictions[0] - predictions[1]).mean().item()
                    print(f"  Quick diversity check - pred diff: {pred_diff:.4f}")
                    if pred_diff < 0.01:
                        print("  ⚠️  Users have very similar predictions!")
                    elif pred_diff > 0.1:
                        print("  ✅ Users have diverse predictions!")
        
        # Add recent popularity boost - only uses last month of training data
        if self.recent_popularity_scores is not None and self.recent_popularity_weight > 0:
            predictions = predictions + self.recent_popularity_weight * self.recent_popularity_scores.unsqueeze(0)
        
        # Encode purchased package with personalized attention
        purchased_repr = self.package_encoder(batch['purchased'], user_emb, user_ids)
        
        # Collect attention weights in debug mode
        if self.debug_mode:
            for name, module in self.named_modules():
                if hasattr(module, 'last_attention_weights') and module.last_attention_weights is not None:
                    self.attention_weights[name] = module.last_attention_weights
        
        # Return predictions and intermediate representations
        results = {
            'predictions': predictions,
            'user_representation': user_final_repr,
            'purchased_representation': purchased_repr,
            'short_term_preference': short_term_pref_2d,
            'long_term_preference': long_term_pref_2d,
            'user_embedding': user_emb  # Include for potential regularization
        }
        
        # Add attention weights in debug mode
        if self.debug_mode:
            results['attention_weights'] = self.attention_weights
            
        return results


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
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class ViewLevelAttention(nn.Module):
    """Enhanced multi-head attention-based view fusion with geographic and price priority"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Multi-head attention for view fusion - increased number of heads
        self.view_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,  # Increased for more expressive patterns
            dropout=0.1,
            batch_first=True
        )
        
        # View importance scoring - learns which views matter most
        self.view_importance = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),  # Added normalization for stable learning
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Dynamic view-wise scaling factors with equal weights for price, coordinates, title, and country
        # Lower weights for category/theme and temporal features
        # Scaling factor for each view in this order: [title, coord_primary, country, category/theme, price, time]
        # Values close to 1.0 mean normal importance, higher values increase importance
        self.view_scalars = nn.Parameter(torch.tensor([1.2, 1.2, 1.2, 0.8, 1.2, 0.6]))  # Initialized weights
        
        # Detect number of views at runtime
        self.num_expected_views = 6  # Updated for 6 views with price
        
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
        
    def forward(self, views: list) -> torch.Tensor:
        """Fuse views with adaptive attention and importance weighting"""
        # Process all views to ensure same dimensionality
        processed_views = []
        
        # Number of views at runtime
        num_views = len(views)
        
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
            # Use view_idx if it's in bounds, otherwise use default scaling
            # The view_scalars parameters allow the model to learn the relative importance of each view
            if view_idx < len(self.view_scalars):
                scale_factor = torch.sigmoid(self.view_scalars[view_idx])  # Sigmoid to keep in [0,1]
                view = view * scale_factor
                
            processed_views.append(view)
        
        # Calculate importance scores for each view
        view_scores = []
        for view in processed_views:
            score = self.view_importance(view)  # Batch x Seq x 1
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
    """Package encoder with event awareness, temporal information, and enhanced price handling"""
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
        
        # NEW: Enhanced price encoder for better price-based recommendations
        self.price_encoder = nn.Sequential(
            nn.Linear(1, 64),
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
        
        # Enhanced event type embedding with more dimensions and attention
        # Keep original size for compatibility with existing cached data
        self.event_embedding = nn.Embedding(5, 32, padding_idx=0)  # 5 event types
        
        # Event encoder with attention to capture sequential importance
        self.event_encoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.hidden_dim)
        )
        
        # Add event attention layer - learns to focus on important event types
        self.event_attention = nn.Sequential(
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # Final event projection
        self.event_projection = nn.Linear(32, config.hidden_dim)
        
        # Time features encoder - ensure it outputs exactly hidden_dim
        self.time_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, config.hidden_dim)
        )
        
        # View-level attention - updated to include price view
        self.view_attention = ViewLevelAttention(config.hidden_dim)
    
    def forward(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with event awareness, temporal information, and price sensitivity"""
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
            
            # NEW: Process price data - check for both standard and normalized prices
            if 'normalized_prices' in batch_data:
                # Use pre-normalized prices if available
                price_data = batch_data['normalized_prices'].float().unsqueeze(-1)
                price_repr = self.price_encoder(price_data).unsqueeze(1)
            elif 'prices' in batch_data:
                # Normalize prices on-the-fly
                price_data = batch_data['prices'].float().unsqueeze(-1).clamp(min=0.01)
                # Apply logarithmic scaling to compress price range
                log_price = torch.log(price_data + 1.0)
                price_repr = self.price_encoder(log_price).unsqueeze(1)
            else:
                # No price data available
                price_repr = torch.zeros(batch_size, 1, self.price_encoder[-2].out_features, 
                                      device=batch_data['title_embeddings'].device)
            
            country_emb = self.country_embedding(batch_data['country_ids']).unsqueeze(1)
            category_emb = self.category_embedding(batch_data['category_ids']).unsqueeze(1)
            theme_emb = self.theme_embedding(batch_data['theme_ids']).unsqueeze(1)
            
            # Event embeddings if available
            if 'event_types' in batch_data:
                event_emb = self.event_embedding(batch_data['event_types']).unsqueeze(1)
            else:
                event_emb = torch.zeros(batch_size, 1, 32, device=batch_data['title_embeddings'].device)
            
            # Time information (for single item case, we don't have temporal information)
            time_repr = torch.zeros(batch_size, 1, self.time_encoder[0].out_features, 
                                   device=batch_data['title_embeddings'].device)
            
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
            
            # NEW: Process price data for sequences
            if 'normalized_prices' in batch_data:
                # Use pre-normalized prices if available
                price_data = batch_data['normalized_prices'].float().unsqueeze(-1)
                price_flat = price_data.view(-1, 1)
                price_repr = self.price_encoder(price_flat).view(batch_size, seq_len, -1)
            elif 'prices' in batch_data:
                # Normalize prices on-the-fly
                price_data = batch_data['prices'].float().unsqueeze(-1).clamp(min=0.01)
                price_flat = price_data.view(-1, 1)
                # Apply logarithmic scaling to compress price range
                log_price = torch.log(price_flat + 1.0)
                price_repr = self.price_encoder(log_price).view(batch_size, seq_len, -1)
            else:
                # No price data available
                price_repr = torch.zeros(batch_size, seq_len, self.price_encoder[-2].out_features, 
                                      device=batch_data['title_embeddings'].device)
            
            country_emb = self.country_embedding(batch_data['country_ids'])
            category_emb = self.category_embedding(batch_data['category_ids'])
            theme_emb = self.theme_embedding(batch_data['theme_ids'])
            
            # Event embeddings
            if 'event_types' in batch_data:
                event_emb = self.event_embedding(batch_data['event_types'])
            else:
                event_emb = torch.zeros(batch_size, seq_len, 32, device=batch_data['title_embeddings'].device)
            
            # Process temporal information if available
            if 'timestamps' in batch_data:
                # Ensure timestamps are floating point and valid
                timestamps = batch_data['timestamps'].float().clamp(min=1.0)  # Avoid invalid timestamps
                
                # Calculate time deltas from most recent event (within each sequence)
                # For each batch item, find its most recent timestamp
                # Create a mask for valid timestamps (non-zero values)
                timestamp_mask = (timestamps > 1.0).float()
                masked_timestamps = timestamps * timestamp_mask
                
                # Use masked max to find the most recent event in each sequence
                # Add a small value to ensure we don't have division by zero
                max_times, _ = masked_timestamps.max(dim=1, keepdim=True)
                
                # Calculate recency: how recent is each event compared to the most recent one
                # We add 1 before dividing to keep values in reasonable range
                time_deltas = torch.clamp(max_times - timestamps, min=0.0) + 1.0
                
                # Apply log scaling to compress the range of time differences
                # This converts absolute time differences to a more meaningful relative scale
                log_time_deltas = torch.log(time_deltas).unsqueeze(-1)  # [batch_size, seq_len, 1]
                
                # Process through time encoder to get representation with correct hidden_dim
                time_repr = self.time_encoder(log_time_deltas)
                
            else:
                # No temporal information available
                time_repr = torch.zeros(batch_size, seq_len, self.time_encoder[0].out_features, 
                                       device=batch_data['title_embeddings'].device)
        
        # Process country embeddings separately
        country_repr = self.country_encoder(country_emb.view(-1, country_emb.size(-1))).view(batch_size, seq_len, -1)
        
        # Combine only category and theme embeddings
        cat_combined = torch.cat([category_emb, theme_emb], dim=-1)
        cat_repr = self.category_encoder(cat_combined.view(-1, cat_combined.size(-1))).view(batch_size, seq_len, -1)
        
        # Process events with attention mechanism to weight different event types 
        # using size-aware implementation that works with cached data
        try:
            # Try to apply attention mechanism
            event_attn_scores = self.event_attention(event_emb)  # [batch_size, seq_len, 1]
            event_attn_weights = F.softmax(event_attn_scores, dim=1)
            event_weighted = event_emb * event_attn_weights  # Apply attention weights
        except Exception:
            # Fallback if dimensions don't match
            event_weighted = event_emb  # Skip attention for incompatible dimensions
        
        # Enhanced event representation with dedicated encoder
        event_encoded = self.event_encoder(event_weighted)
        event_repr = self.event_projection(event_emb)
        
        # Combine event information - now with stronger event type semantics
        # Early fusion with title representation
        title_repr = title_repr + 0.3 * event_encoded  # Increased from 0.1 to 0.3 to strengthen event signal
        
        # Apply view-level attention with all representations
        # Equal moderate weights (1.2) are given to title, coordinates, country, and price features
        # Lower weights are given to category (0.8) and temporal features (0.6)
        # The order of views must match the order of view_scalars in ViewLevelAttention:
        # [title, coord_primary, country, category/theme, price, time]
        views = [
            title_repr,            # Content-based similarity (view index 0 - weight 1.2)
            coord_repr,            # Geographic proximity (view index 1 - weight 1.2)
            country_repr,          # Country representation (view index 2 - weight 1.2)
            cat_repr,              # Category/theme preferences (view index 3 - weight 0.8)
            price_repr,            # Price-based recommendations (view index 4 - weight 1.2)
            time_repr              # Temporal information (view index 5 - weight 0.6)
        ]
        unified_repr = self.view_attention(views)
        
        return unified_repr.squeeze(1) if seq_len == 1 else unified_repr


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
        
        # Apply temporal bias if timestamps are provided
        if timestamps is not None:
            # Ensure timestamps are floating point and valid
            timestamps = timestamps.float().clamp(min=1.0)  # Avoid invalid timestamps
            
            # Create a mask for valid timestamps (non-zero values)
            timestamp_mask = (timestamps > 1.0).float()
            masked_timestamps = timestamps * timestamp_mask
            
            # Use masked max to find the most recent event in each sequence
            max_times, _ = masked_timestamps.max(dim=1, keepdim=True)
            
            # Calculate recency: how recent is each event compared to the most recent one
            # We add 1 before dividing to keep values in reasonable range
            time_deltas = torch.clamp(max_times - timestamps, min=0.0) + 1.0
            
            # Apply log scaling to compress the range of time differences
            log_time_deltas = torch.log(time_deltas).unsqueeze(-1)  # [batch_size, seq_len, 1]
            
            # Get recency bias - more recent = higher score (smaller time_delta)
            # Negative sign converts time_delta to recency (higher = more recent)
            recency_bias = self.time_bias(-log_time_deltas).squeeze(-1)  # [batch_size, seq_len]
            
            # Add recency bias to attention scores - expand for multi-head
            recency_bias = recency_bias.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            attention_scores = attention_scores + recency_bias
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multi-head attention
            expanded_mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            attention_scores = attention_scores.masked_fill(expanded_mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, num_heads, 1, seq_len]
        
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
        
        return output.squeeze(1)


class UserEncoder(nn.Module):
    """User encoder with multi-layer Bi-LSTM, cross-attention, and layer normalization for stability"""
    def __init__(self, config):
        super().__init__()
        
        self.user_embedding = nn.Embedding(config.num_users, config.user_embedding_dim, padding_idx=0)
        
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
        
        # Process sequences with multi-layer LSTM
        short_term_output, _ = self.short_term_lstm(short_term_repr)
        short_term_output = self.short_term_norm(short_term_output)  # Apply layer normalization
        
        long_term_output, _ = self.long_term_lstm(long_term_repr)
        long_term_output = self.long_term_norm(long_term_output)  # Apply layer normalization
        
        # Apply cross-attention between sequences to capture interactions
        # Short-term as query, long-term as key/value
        cross_st_mask = None
        if short_term_mask is not None and long_term_mask is not None:
            cross_st_mask = torch.bmm(
                short_term_mask.unsqueeze(-1).float(), 
                long_term_mask.unsqueeze(1).float()
            ).bool()
        
        cross_attended_short, _ = self.cross_attention(
            short_term_output, 
            long_term_output, 
            long_term_output,
            key_padding_mask=None if long_term_mask is None else ~long_term_mask.bool()
        )
        
        # Enhance short-term representation with cross-attention
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
        fused = adjusted_gate * long_term_proj + (1 - adjusted_gate) * short_term_proj
        
        # Add residual connection with the user embedding
        fused = fused + 0.1 * user_proj
        
        # Final projection and normalization
        fused = self.output_projection(fused)
        fused = self.layer_norm(fused)
        
        return fused


class NATR(nn.Module):
    """
    Enhanced NATR model with natural event learning capability, temporal awareness,
    and analysis capabilities for attention patterns
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
            nn.BatchNorm1d(config.hidden_dim * 2),  # Added batch norm for stability
            nn.Linear(config.hidden_dim * 2, config.num_packages)
        )
        
        # Debug mode for capturing attention weights
        self.debug_mode = False
        self.attention_weights = {}
    
    def enable_debug(self, enable=True):
        """Enable debug mode to capture attention weights"""
        self.debug_mode = enable
        # Enable saving attention weights in attention modules
        for module in self.modules():
            if hasattr(module, 'save_attention'):
                module.save_attention = enable
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass with temporal information and attention tracking"""
        
        # Extract user IDs
        user_ids = batch['user_id']
        
        # Encode packages with event information
        short_term_repr = self.package_encoder(batch['short_term'])
        long_term_repr = self.package_encoder(batch['long_term'])
        
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
        user_final_repr = self.gated_fusion(short_term_pref, long_term_pref, user_emb)
        
        # Make predictions
        # Handle batch size of 1 case for BatchNorm
        if user_final_repr.shape[0] == 1 and hasattr(self.prediction_head[3], 'running_mean'):
            # Skip BatchNorm layer in case of single sample
            x = self.prediction_head[0](user_final_repr)  # Linear
            x = self.prediction_head[1](x)  # ReLU
            x = self.prediction_head[2](x)  # Dropout
            predictions = self.prediction_head[4](x)  # Final Linear
        else:
            predictions = self.prediction_head(user_final_repr)
        
        # Encode purchased package
        purchased_repr = self.package_encoder(batch['purchased'])
        
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
            'short_term_preference': short_term_pref,
            'long_term_preference': long_term_pref
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
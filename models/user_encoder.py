import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

class UserEncoder(nn.Module):
    """
    User Encoder with Bi-LSTM and package-level attention
    
    This encoder learns user preferences from their short-term and long-term behaviors
    using Bi-LSTM networks and attention mechanisms.
    """
    def __init__(self, 
                 embedding_dim: int, 
                 hidden_dim: int, 
                 max_short_term: int = 5,
                 max_long_term: int = 10,
                 dropout: float = 0.2):
        """
        Initialize the User Encoder
        
        Args:
            embedding_dim: Dimension of embedding vectors
            hidden_dim: Dimension of hidden vectors
            max_short_term: Maximum length of short-term sequence
            max_long_term: Maximum length of long-term sequence
            dropout: Dropout rate for LSTM layers
        """
        super(UserEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_short_term = max_short_term
        self.max_long_term = max_long_term
        
        # Short-term preference BiLSTM
        self.short_term_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,  # Bidirectional will double this
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Long-term preference BiLSTM
        self.long_term_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,  # Bidirectional will double this
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Package-level attention projection
        self.package_attention_w = nn.Linear(hidden_dim, hidden_dim)
        self.package_attention_b = nn.Parameter(torch.zeros(hidden_dim))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) > 1:  # Only apply to matrices
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def encode_short_term(self, 
                         package_repr: torch.Tensor, 
                         user_emb: torch.Tensor,
                         mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encode short-term preferences using Bi-LSTM and attention
        
        Args:
            package_repr: Package representations [batch_size, max_short_term, embedding_dim]
            user_emb: User embedding [batch_size, embedding_dim]
            mask: Mask for valid packages [batch_size, max_short_term]
                
        Returns:
            Short-term preference vector [batch_size, hidden_dim]
        """
        batch_size = package_repr.size(0)
        
        # Apply BiLSTM to short-term sequence
        lstm_out, _ = self.short_term_lstm(package_repr)  # [batch_size, max_short_term, hidden_dim]
        
        # Package-level attention for short-term preferences
        attn = torch.tanh(self.package_attention_w(lstm_out) + self.package_attention_b)
        
        # Calculate attention scores using user embedding as query
        # Expand user_emb to match sequence dimension for broadcasting
        user_emb_exp = user_emb.unsqueeze(1).expand(-1, self.max_short_term, -1)
        
        # Dot product between attention projection and user embedding
        attn_scores = torch.sum(attn * user_emb_exp, dim=2, keepdim=True)  # [batch_size, max_short_term, 1]
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(2)  # [batch_size, max_short_term, 1]
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1)  # [batch_size, max_short_term, 1]
        
        # Weighted sum to get short-term preference
        short_term_pref = torch.sum(lstm_out * attn_weights, dim=1)  # [batch_size, hidden_dim]
        
        return short_term_pref
    
    def encode_long_term(self, 
                        package_repr: torch.Tensor, 
                        user_emb: torch.Tensor,
                        mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encode long-term preferences using Bi-LSTM and attention
        
        Args:
            package_repr: Package representations [batch_size, max_long_term, embedding_dim]
            user_emb: User embedding [batch_size, embedding_dim]
            mask: Mask for valid packages [batch_size, max_long_term]
                
        Returns:
            Long-term preference vector [batch_size, hidden_dim]
        """
        batch_size = package_repr.size(0)
        
        # Apply BiLSTM to long-term sequence
        lstm_out, _ = self.long_term_lstm(package_repr)  # [batch_size, max_long_term, hidden_dim]
        
        # Package-level attention for long-term preferences
        attn = torch.tanh(self.package_attention_w(lstm_out) + self.package_attention_b)
        
        # Calculate attention scores using user embedding as query
        # Expand user_emb to match sequence dimension for broadcasting
        user_emb_exp = user_emb.unsqueeze(1).expand(-1, self.max_long_term, -1)
        
        # Dot product between attention projection and user embedding
        attn_scores = torch.sum(attn * user_emb_exp, dim=2, keepdim=True)  # [batch_size, max_long_term, 1]
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(2)  # [batch_size, max_long_term, 1]
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1)  # [batch_size, max_long_term, 1]
        
        # Weighted sum to get long-term preference
        long_term_pref = torch.sum(lstm_out * attn_weights, dim=1)  # [batch_size, hidden_dim]
        
        return long_term_pref
    
    def forward(self, 
               short_term_repr: torch.Tensor, 
               long_term_repr: torch.Tensor, 
               user_emb: torch.Tensor,
               short_term_mask: torch.Tensor = None,
               long_term_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the User Encoder
        
        Args:
            short_term_repr: Short-term package representations 
                            [batch_size, max_short_term, embedding_dim]
            long_term_repr: Long-term package representations 
                            [batch_size, max_long_term, embedding_dim]
            user_emb: User embedding [batch_size, embedding_dim]
            short_term_mask: Mask for valid short-term packages [batch_size, max_short_term]
            long_term_mask: Mask for valid long-term packages [batch_size, max_long_term]
                
        Returns:
            Tuple of (short_term_pref, long_term_pref)
        """
        # Encode short-term preferences
        short_term_pref = self.encode_short_term(short_term_repr, user_emb, short_term_mask)
        
        # Encode long-term preferences
        long_term_pref = self.encode_long_term(long_term_repr, user_emb, long_term_mask)
        
        return short_term_pref, long_term_pref


class GatedFusion(nn.Module):
    """
    Gated Fusion Network for combining long-term and short-term preferences
    """
    def __init__(self, embedding_dim: int, hidden_dim: int):
        """
        Initialize the Gated Fusion Network
        
        Args:
            embedding_dim: Dimension of user embedding
            hidden_dim: Dimension of preference vectors
        """
        super(GatedFusion, self).__init__()
        
        # Projection layers for gate calculation
        self.user_projection = nn.Linear(embedding_dim, embedding_dim)
        self.short_term_projection = nn.Linear(hidden_dim, embedding_dim)
        self.long_term_projection = nn.Linear(hidden_dim, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(embedding_dim))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, 
               short_term_pref: torch.Tensor, 
               long_term_pref: torch.Tensor, 
               user_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Gated Fusion Network
        
        Args:
            short_term_pref: Short-term preference vector [batch_size, hidden_dim]
            long_term_pref: Long-term preference vector [batch_size, hidden_dim]
            user_emb: User embedding [batch_size, embedding_dim]
                
        Returns:
            Fused preference vector [batch_size, embedding_dim]
        """
        # Project preference vectors to embedding dimension
        short_term_proj = self.short_term_projection(short_term_pref)
        long_term_proj = self.long_term_projection(long_term_pref)
        
        # Calculate fusion gate
        gate = torch.sigmoid(
            self.user_projection(user_emb) + 
            short_term_proj + 
            long_term_proj + 
            self.bias
        )  # [batch_size, embedding_dim]
        
        # Combine preferences using gate
        fused_pref = (1 - gate) * short_term_proj + gate * long_term_proj
        
        return fused_pref


# Add a test function to validate the implementation
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define dimensions
    batch_size = 4
    embedding_dim = 128
    hidden_dim = 128
    max_short_term = 5
    max_long_term = 10
    
    # Create random input data
    user_emb = torch.randn(batch_size, embedding_dim)
    short_term_repr = torch.randn(batch_size, max_short_term, embedding_dim)
    long_term_repr = torch.randn(batch_size, max_long_term, embedding_dim)
    
    # Create masks (1 for valid, 0 for padding)
    short_term_mask = torch.ones(batch_size, max_short_term)
    # Set some positions to 0 to simulate padding
    short_term_mask[0, 3:] = 0  # First sample has 3 valid packages
    short_term_mask[1, 4:] = 0  # Second sample has 4 valid packages
    
    long_term_mask = torch.ones(batch_size, max_long_term)
    # Set some positions to 0 to simulate padding
    long_term_mask[0, 5:] = 0  # First sample has 5 valid packages
    long_term_mask[1, 7:] = 0  # Second sample has 7 valid packages
    
    # Initialize models
    print("Initializing User Encoder and Gated Fusion...")
    user_encoder = UserEncoder(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        max_short_term=max_short_term,
        max_long_term=max_long_term,
        dropout=0.2
    )
    
    gated_fusion = GatedFusion(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim
    )
    
    # Forward pass
    print("\nRunning forward pass...")
    
    # User encoder forward pass
    short_term_pref, long_term_pref = user_encoder(
        short_term_repr,
        long_term_repr,
        user_emb,
        short_term_mask,
        long_term_mask
    )
    
    # Gated fusion forward pass
    fused_pref = gated_fusion(
        short_term_pref,
        long_term_pref,
        user_emb
    )
    
    # Print output shapes
    print("\nOutput shapes:")
    print(f"Short-term preference: {short_term_pref.shape}")
    print(f"Long-term preference: {long_term_pref.shape}")
    print(f"Fused preference: {fused_pref.shape}")
    
    print("\nUser Encoder test completed successfully!")
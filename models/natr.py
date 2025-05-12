import torch
import torch.nn as nn
import torch.nn.functional as F

class NATR(nn.Module):
    """
    Neural Attentive Travel Recommendation (NATR) model
    
    An improved version that handles empty short-term packages with a special token
    and incorporates attention mechanisms for better recommendation quality.
    """
    def __init__(self, 
                num_users, 
                num_packages,
                num_countries,
                num_categories,
                num_themes,
                embedding_dim=128, 
                hidden_dim=256,
                dropout=0.2,
                empty_token=2):
        """
        Initialize the NATR model
        
        Args:
            num_users: Number of users in the dataset
            num_packages: Number of travel packages in the dataset
            num_countries: Number of countries in the dataset
            num_categories: Number of categories in the dataset
            num_themes: Number of themes in the dataset
            embedding_dim: Dimension of the embedding vectors
            hidden_dim: Dimension of the hidden layers
            dropout: Dropout rate
            empty_token: Special token for empty short-term packages
        """
        super(NATR, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.empty_token = empty_token
        
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        self.package_embedding = nn.Embedding(num_packages + 1, embedding_dim, padding_idx=0)
        self.country_embedding = nn.Embedding(num_countries + 1, embedding_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories + 1, embedding_dim, padding_idx=0)
        self.theme_embedding = nn.Embedding(num_themes + 1, embedding_dim, padding_idx=0)
        
        # Special embedding for empty short-term token
        self.empty_embedding = nn.Parameter(torch.randn(1, embedding_dim))
        
        # LSTM layers for sequence processing
        self.short_term_lstm = nn.LSTM(
            input_size=embedding_dim * 4,  # package + country + category + theme
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        
        self.long_term_lstm = nn.LSTM(
            input_size=embedding_dim * 4,  # package + country + category + theme
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention layers
        self.word_level_attention = nn.Linear(hidden_dim * 2, 1)  # For bidirectional
        self.view_level_attention = nn.Linear(hidden_dim * 2, 1)  # For bidirectional
        self.package_level_attention = nn.Linear(hidden_dim * 2, 1)  # For bidirectional
        
        # Fusion layer for combining short-term and long-term preferences
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),  # (short_term + long_term) -> hidden
            nn.Sigmoid()
        )
        
        # Final prediction layers
        self.prediction_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for the model"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def _encode_package_sequence(self, package_ids, country_ids, category_ids, theme_ids, lstm_layer, attention_layer):
        """Encode a sequence of packages using LSTM and attention"""
        # Get embeddings
        package_emb = self.package_embedding(package_ids)  # [batch_size, seq_len, embedding_dim]
        country_emb = self.country_embedding(country_ids)  # [batch_size, seq_len, embedding_dim]
        category_emb = self.category_embedding(category_ids)  # [batch_size, seq_len, embedding_dim]
        theme_emb = self.theme_embedding(theme_ids)  # [batch_size, seq_len, embedding_dim]
        
        # Concatenate embeddings
        sequence_emb = torch.cat([package_emb, country_emb, category_emb, theme_emb], dim=2)  # [batch_size, seq_len, 4*embedding_dim]
        
        # Create mask for padding (1 for real tokens, 0 for padding)
        mask = (package_ids != 0).float().unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # Process through LSTM
        lstm_out, _ = lstm_layer(sequence_emb)  # [batch_size, seq_len, 2*hidden_dim]
        
        # Apply attention
        attention_scores = attention_layer(lstm_out)  # [batch_size, seq_len, 1]
        attention_scores = attention_scores.masked_fill((1 - mask).bool(), -1e9)  # Apply mask
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len, 1]
        
        # Weight sequence by attention
        weighted_output = lstm_out * attention_weights  # [batch_size, seq_len, 2*hidden_dim]
        sequence_representation = weighted_output.sum(dim=1)  # [batch_size, 2*hidden_dim]
        
        return sequence_representation, attention_weights
    
    def forward(self, batch):
        """
        Forward pass through the NATR model
        
        Args:
            batch: Dictionary containing user and package data
                - user_id: User IDs tensor [batch_size]
                - has_short_term: Boolean tensor indicating if short-term is present [batch_size]
                - short_term: Dictionary containing short-term package data
                    - package_ids: Package IDs tensor [batch_size, max_short_term]
                    - country_ids: Country IDs tensor [batch_size, max_short_term]
                    - category_ids: Category IDs tensor [batch_size, max_short_term]
                    - theme_ids: Theme IDs tensor [batch_size, max_short_term]
                - long_term: Dictionary containing long-term package data (same structure)
                - purchased: Dictionary containing purchased package data (without sequence dimension)
                
        Returns:
            Dictionary containing model outputs
                - scores: Recommendation scores [batch_size, num_packages]
                - user_representation: User representation [batch_size, embedding_dim]
                - short_term_attention: Attention weights for short-term packages [batch_size, max_short_term, 1]
                - long_term_attention: Attention weights for long-term packages [batch_size, max_long_term, 1]
                - fusion_weights: Fusion weights for short-term and long-term [batch_size, hidden_dim * 2]
        """
        user_id = batch['user_id']
        has_short_term = batch.get('has_short_term', None)
        
        # Process short-term packages
        if has_short_term is not None:
            # If has_short_term flag is provided, handle empty short-term
            short_term_representation = torch.zeros(
                user_id.size(0), self.hidden_dim * 2, device=user_id.device
            )
            short_term_attention = None
            
            # Process users with short-term data
            if has_short_term.any():
                # Get indices of users with short-term data
                has_st_indices = has_short_term.nonzero(as_tuple=True)[0]
                
                # Extract short-term data for these users
                st_pkg_ids = batch['short_term']['package_ids'][has_st_indices]
                st_country_ids = batch['short_term']['country_ids'][has_st_indices]
                st_category_ids = batch['short_term']['category_ids'][has_st_indices]
                st_theme_ids = batch['short_term']['theme_ids'][has_st_indices]
                
                # Check for special token (empty short-term marked with token)
                empty_token_mask = (st_pkg_ids == self.empty_token)
                if empty_token_mask.any():
                    # Replace special token with a learnable embedding
                    st_representation, st_attention = self._encode_package_sequence(
                        st_pkg_ids, st_country_ids, st_category_ids, st_theme_ids,
                        self.short_term_lstm, self.package_level_attention
                    )
                    short_term_representation[has_st_indices] = st_representation
                    
                    if short_term_attention is None:
                        short_term_attention = torch.zeros(
                            user_id.size(0), st_attention.size(1), 1, device=user_id.device
                        )
                    short_term_attention[has_st_indices] = st_attention
            
            # For users without short-term, leave representation as zeros
        else:
            # Process all short-term packages if no has_short_term flag
            short_term_representation, short_term_attention = self._encode_package_sequence(
                batch['short_term']['package_ids'],
                batch['short_term']['country_ids'],
                batch['short_term']['category_ids'],
                batch['short_term']['theme_ids'],
                self.short_term_lstm,
                self.package_level_attention
            )
        
        # Process long-term packages
        long_term_representation, long_term_attention = self._encode_package_sequence(
            batch['long_term']['package_ids'],
            batch['long_term']['country_ids'],
            batch['long_term']['category_ids'],
            batch['long_term']['theme_ids'],
            self.long_term_lstm,
            self.package_level_attention
        )
        
        # Combine short-term and long-term preferences with gated fusion
        combined_input = torch.cat([short_term_representation, long_term_representation], dim=1)
        fusion_weights = self.fusion_gate(combined_input)
        
        # Apply weighted fusion
        user_representation = (1 - fusion_weights) * short_term_representation + fusion_weights * long_term_representation
        
        # Get user's final prediction vector
        user_vector = self.prediction_layer(user_representation)
        
        # Calculate scores with all packages (can be done more efficiently in the loss function)
        all_packages = self.package_embedding.weight
        scores = torch.matmul(user_vector, all_packages.t())
        
        return {
            'scores': scores,
            'user_representation': user_vector,
            'short_term_attention': short_term_attention,
            'long_term_attention': long_term_attention,
            'fusion_weights': fusion_weights
        }
    
    def calculate_loss(self, batch, output, negative_samples=5):
        """
        Calculate the BPR loss for the model
        
        Args:
            batch: Input batch data
            output: Model output
            negative_samples: Number of negative samples per positive sample
            
        Returns:
            loss: BPR loss value
        """
        # Get positive examples (purchased packages)
        pos_package_ids = batch['purchased']['package_ids']
        
        # Get embeddings for positive examples
        pos_embeddings = self.package_embedding(pos_package_ids)
        
        # Calculate positive scores
        user_vector = output['user_representation']
        pos_scores = torch.sum(user_vector * pos_embeddings, dim=1, keepdim=True)
        
        # Generate negative samples
        batch_size = user_vector.size(0)
        neg_package_ids = torch.randint(
            1, self.package_embedding.weight.size(0), 
            (batch_size, negative_samples), 
            device=user_vector.device
        )
        
        # Get embeddings for negative examples
        neg_embeddings = self.package_embedding(neg_package_ids)
        
        # Calculate negative scores
        neg_scores = torch.bmm(
            neg_embeddings,
            user_vector.unsqueeze(2)
        ).squeeze(2)
        
        # Calculate BPR loss
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        
        return loss
    
    def recommend(self, user_representation, top_k=10):
        """
        Generate recommendations for a user
        
        Args:
            user_representation: User representation vector
            top_k: Number of recommendations to generate
            
        Returns:
            top_k_indices: Indices of top-k recommended packages
            top_k_scores: Scores of top-k recommended packages
        """
        # Calculate scores with all packages
        all_packages = self.package_embedding.weight
        scores = torch.matmul(user_representation, all_packages.t())
        
        # Get top-k recommendations
        top_k_scores, top_k_indices = torch.topk(scores, k=top_k)
        
        return top_k_indices, top_k_scores
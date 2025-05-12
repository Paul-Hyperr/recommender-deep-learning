# config.py
class Config:
    # Data parameters
    session_timeout_hours = 24
    max_short_term_length = 5
    max_long_term_length = 10
    max_title_length = 20
    
    # Model parameters
    word_embedding_dim = 300
    hidden_size = 128
    embedding_dim = 256
    
    # Training parameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 5
    dropout = 0.2
    
    # Evaluation parameters
    top_k = 10  # For HR@k and MRR@k
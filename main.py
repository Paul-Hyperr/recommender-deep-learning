#!/usr/bin/env python
"""
Training script for the updated NATR model
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import argparse
import json
import time
from tqdm import tqdm

from models.updated_natr_model import UpdatedNATR
from utils.data_processor import TravelDataProcessor
from utils.package_data_loader import load_package_metadata, prepare_dataloaders
from utils.metrics import hit_rate_at_k, mrr_at_k, item_coverage_at_k

def seed_everything(seed):
    """Set seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model(model, 
               train_loader, 
               test_loader, 
               device, 
               num_packages,
               output_dir,
               num_epochs=5, 
               lr=0.001,
               weight_decay=1e-6,
               patience=3,
               eval_every=1,
               negative_samples=9,
               top_k=20):
    """
    Train the NATR model
    
    Args:
        model: NATR model
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
        device: Device to train on
        num_packages: Number of packages in the dataset
        output_dir: Directory to save model and logs
        num_epochs: Number of epochs to train
        lr: Learning rate
        weight_decay: Weight decay (L2 regularization)
        patience: Number of epochs to wait for improvement
        eval_every: Evaluate every n epochs
        negative_samples: Number of negative samples per positive sample
        top_k: K for evaluation metrics
        
    Returns:
        Trained model and training history
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [],
        f'hit_rate@{top_k}': [],
        f'mrr@{top_k}': [],
        f'item_coverage@{top_k}': [],
        'epochs': []
    }
    
    # Early stopping variables
    best_hit_rate = 0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        
        train_start_time = time.time()
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Extract batch data
            user_ids = batch['user_id'].to(device)
            
            short_term = {
                'package_ids': batch['short_term']['package_ids'].to(device),
                'country_ids': batch['short_term']['country_ids'].to(device),
                'category_ids': batch['short_term']['category_ids'].to(device),
                'theme_ids': batch['short_term']['theme_ids'].to(device),
                'price_values': batch['short_term']['price_values'].to(device)
            }
            
            long_term = {
                'package_ids': batch['long_term']['package_ids'].to(device),
                'country_ids': batch['long_term']['country_ids'].to(device),
                'category_ids': batch['long_term']['category_ids'].to(device),
                'theme_ids': batch['long_term']['theme_ids'].to(device),
                'price_values': batch['long_term']['price_values'].to(device)
            }
            
            purchased = {
                'package_ids': batch['purchased']['package_ids'].to(device),
                'country_ids': batch['purchased']['country_ids'].to(device),
                'category_ids': batch['purchased']['category_ids'].to(device),
                'theme_ids': batch['purchased']['theme_ids'].to(device),
                'price_values': batch['purchased']['price_values'].to(device)
            }
            
            batch_size = user_ids.size(0)
            
            # Generate negative samples
            # Create candidates with purchased at index 0 (positive sample) and random negatives
            candidate_package_ids = torch.cat([
                purchased['package_ids'].unsqueeze(1),
                torch.randint(1, num_packages, (batch_size, negative_samples), device=device)  # Start from 1 to avoid padding
            ], dim=1)
            
            # Create candidate dictionaries
            candidates = {
                'package_ids': candidate_package_ids,
                # Other attributes will be populated by the model
            }
            
            # Forward pass
            scores, _ = model(user_ids, short_term, long_term, candidates)
            
            # Compute loss (label 0 = purchased item)
            labels = torch.zeros(batch_size, dtype=torch.long, device=device)
            loss = criterion(scores, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Calculate training time
        train_time = time.time() - train_start_time
        
        # Print epoch stats
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Time: {train_time:.2f}s')
        
        # Save training loss
        history['train_loss'].append(avg_loss)
        history['epochs'].append(epoch + 1)
        
        # Evaluation phase (every eval_every epochs)
        if (epoch + 1) % eval_every == 0:
            model.eval()
            
            hit_rates = []
            mrrs = []
            coverages = []
            ground_truths = []
            
            eval_start_time = time.time()
            with torch.no_grad():
                for batch in tqdm(test_loader, desc='Evaluating'):
                    # Extract batch data
                    user_ids = batch['user_id'].to(device)
                    
                    short_term = {
                        'package_ids': batch['short_term']['package_ids'].to(device),
                        'country_ids': batch['short_term']['country_ids'].to(device),
                        'category_ids': batch['short_term']['category_ids'].to(device),
                        'theme_ids': batch['short_term']['theme_ids'].to(device),
                        'price_values': batch['short_term']['price_values'].to(device)
                    }
                    
                    long_term = {
                        'package_ids': batch['long_term']['package_ids'].to(device),
                        'country_ids': batch['long_term']['country_ids'].to(device),
                        'category_ids': batch['long_term']['category_ids'].to(device),
                        'theme_ids': batch['long_term']['theme_ids'].to(device),
                        'price_values': batch['long_term']['price_values'].to(device)
                    }
                    
                    purchased = {
                        'package_ids': batch['purchased']['package_ids'].to(device),
                        'country_ids': batch['purchased']['country_ids'].to(device),
                        'category_ids': batch['purchased']['category_ids'].to(device),
                        'theme_ids': batch['purchased']['theme_ids'].to(device),
                        'price_values': batch['purchased']['price_values'].to(device)
                    }
                    
                    batch_size = user_ids.size(0)
                    
                    # Generate candidates with purchased at index 0 and random negatives
                    candidate_package_ids = torch.cat([
                        purchased['package_ids'].unsqueeze(1),
                        torch.randint(1, num_packages, (batch_size, top_k-1), device=device)  # Start from 1 to avoid padding
                    ], dim=1)
                    
                    # Create candidate dictionaries
                    candidates = {
                        'package_ids': candidate_package_ids,
                        # Other attributes will be populated by the model
                    }
                    
                    # Forward pass
                    scores, _ = model(user_ids, short_term, long_term, candidates)
                    
                    # Ground truth is always at index 0
                    ground_truth = torch.zeros(batch_size, dtype=torch.long, device=device)
                    
                    # Store ground truths for item coverage calculation
                    ground_truths.extend(purchased['package_ids'].cpu().tolist())
                    
                    # Calculate metrics
                    hr = hit_rate_at_k(scores, ground_truth, k=top_k)
                    hit_rates.append(hr)
                    
                    mrr = mrr_at_k(scores, ground_truth, k=top_k)
                    mrrs.append(mrr)
                    
                    coverage = item_coverage_at_k(scores, ground_truth, k=top_k)
                    coverages.append(coverage)
            
            # Calculate evaluation time
            eval_time = time.time() - eval_start_time
            
            # Calculate average metrics
            avg_hit_rate = np.mean(hit_rates)
            avg_mrr = np.mean(mrrs)
            avg_coverage = np.mean(coverages)
            
            # Save metrics
            history[f'hit_rate@{top_k}'].append(avg_hit_rate)
            history[f'mrr@{top_k}'].append(avg_mrr)
            history[f'item_coverage@{top_k}'].append(avg_coverage)
            
            # Print evaluation results
            print(f'Evaluation - HR@{top_k}: {avg_hit_rate:.4f}, MRR@{top_k}: {avg_mrr:.4f}, Coverage@{top_k}: {avg_coverage:.4f}, Time: {eval_time:.2f}s')
            
            # Early stopping check
            if avg_hit_rate > best_hit_rate:
                best_hit_rate = avg_hit_rate
                patience_counter = 0
                
                # Save best model
                model_path = os.path.join(output_dir, 'best_model.pth')
                torch.save(model.state_dict(), model_path)
                print(f'New best model saved with HR@{top_k}: {best_hit_rate:.4f}')
                
                # Save model metadata
                metadata = {
                    'epoch': epoch + 1,
                    f'hit_rate@{top_k}': float(avg_hit_rate),
                    f'mrr@{top_k}': float(avg_mrr),
                    f'item_coverage@{top_k}': float(avg_coverage),
                    'parameters': {
                        'embedding_dim': model.embedding_dim,
                        'hidden_dim': model.hidden_dim,
                        'package_vocab_size': model.package_encoder.package_embeddings.num_embeddings,
                        'country_vocab_size': model.package_encoder.country_embedding.num_embeddings,
                        'category_vocab_size': model.package_encoder.category_embedding.num_embeddings,
                        'theme_vocab_size': model.package_encoder.theme_embedding.num_embeddings,
                        'user_vocab_size': model.user_embedding.num_embeddings
                    }
                }
                
                metadata_path = os.path.join(output_dir, 'model_metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            else:
                patience_counter += 1
                print(f'No improvement for {patience_counter} epochs (best HR@{top_k}: {best_hit_rate:.4f})')
                
                if patience_counter >= patience:
                    print(f'Early stopping after {epoch+1} epochs')
                    break
        
        # Save training history
        history_path = os.path.join(output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            # Convert numpy values to Python types for JSON serialization
            serializable_history = {}
            for key, values in history.items():
                serializable_history[key] = [float(v) if isinstance(v, (np.number, np.ndarray)) else v for v in values]
            
            json.dump(serializable_history, f, indent=2)
    
    # Load best model
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f'Loaded best model from {best_model_path}')
    
    return model, history

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train the NATR model')
    parser.add_argument('--events_data', type=str, required=True, help='Path to events data file (CSV or parquet)')
    parser.add_argument('--package_data', type=str, required=True, help='Path to package data file (parquet)')
    parser.add_argument('--output_dir', type=str, default='models/natr_output', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay')
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping')
    parser.add_argument('--eval_every', type=int, default=1, help='Evaluate every n epochs')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--negative_samples', type=int, default=9, help='Number of negative samples')
    parser.add_argument('--top_k', type=int, default=20, help='Top-k for evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--session_timeout', type=int, default=24, help='Session timeout in hours')
    parser.add_argument('--min_interactions', type=int, default=5, help='Minimum interactions per user')
    parser.add_argument('--cache_dir', type=str, default='data/cache', help='Cache directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    
    args = parser.parse_args()
    
    # Set random seed
    seed_everything(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Process events data
    print('Processing events data...')
    processor = TravelDataProcessor(
        data_path=args.events_data,
        session_timeout_hours=args.session_timeout,
        min_interactions=args.min_interactions,
        cache_dir=args.cache_dir
    )
    
    processor.load_data(use_cache=True)
    processor.create_mappings(use_cache=True)
    
    # Extract training samples
    samples = processor.prepare_training_data(use_cache=True)
    
    # Load package metadata
    package_metadata = load_package_metadata(args.package_data)
    
    # Prepare dataloaders
    train_loader, test_loader = prepare_dataloaders(
        samples,
        package_metadata,
        processor.user_to_idx,
        processor.package_to_idx,
        processor.country_to_idx,
        processor.category_to_idx,
        processor.theme_to_idx,
        batch_size=args.batch_size,
        test_size=0.2,
        random_seed=args.seed,
        num_workers=args.num_workers
    )
    
    # Initialize model
    model = UpdatedNATR(
        package_vocab_size=len(processor.package_to_idx) + 1,  # +1 for padding
        country_vocab_size=len(processor.country_to_idx) + 1,
        category_vocab_size=len(processor.category_to_idx) + 1,
        theme_vocab_size=len(processor.theme_to_idx) + 1,
        user_vocab_size=len(processor.user_to_idx) + 1,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    
    # Print model summary
    print(model)
    print(f'Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    
    # Train model
    model, history = train_model(
        model,
        train_loader,
        test_loader,
        device,
        len(processor.package_to_idx) + 1,
        args.output_dir,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        eval_every=args.eval_every,
        negative_samples=args.negative_samples,
        top_k=args.top_k
    )
    
    print('Training completed!')

if __name__ == '__main__':
    main()
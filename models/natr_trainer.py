import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any

# Import the NATR model
from natr_model import NATR

class NATR_Trainer:
    """
    Trainer for the NATR model
    
    Handles training, evaluation, and hyperparameter tuning
    """
    def __init__(self, 
                 model: NATR,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 package_to_idx: Dict[str, int],
                 idx_to_package: Dict[int, str],
                 package_metadata: Dict[str, Dict[str, Any]],
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.0001,
                 device: str = None,
                 model_dir: str = 'model_checkpoints',
                 log_dir: str = 'logs'):
        """
        Initialize the NATR trainer
        
        Args:
            model: NATR model instance
            train_loader: Training data loader
            test_loader: Testing data loader
            package_to_idx: Mapping from package ID to index
            idx_to_package: Mapping from index to package ID
            package_metadata: Dictionary mapping package ID to metadata
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            device: Device to run training on ('cuda' or 'cpu')
            model_dir: Directory to save model checkpoints
            log_dir: Directory to save training logs
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.package_to_idx = package_to_idx
        self.idx_to_package = idx_to_package
        self.package_metadata = package_metadata
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        self.model_dir = model_dir
        self.log_dir = log_dir
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Initialize logs
        self.train_losses = []
        self.train_hits = []
        self.test_losses = []
        self.test_hits = []
        self.best_hit_rate = 0.0
    
    def train_epoch(self, epoch: int, negative_samples: int = 5):
        """
        Train the model for one epoch
        
        Args:
            epoch: Current epoch number
            negative_samples: Number of negative samples per positive sample
        
        Returns:
            tuple: (average_loss, hit_rate)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()
            
            # Move batch to device
            batch = {k: {k2: v2.to(self.device) if isinstance(v2, torch.Tensor) else v2 
                         for k2, v2 in v.items()} if isinstance(v, dict) else v.to(self.device) 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(batch)
            user_pref = outputs['user_preference']
            
            # Get positive sample
            positive_repr = outputs['purchased_representation']
            
            # Create labels: 1 for positive, 0 for negative
            batch_size = user_pref.size(0)
            labels = torch.zeros(batch_size, negative_samples + 1, device=self.device)
            labels[:, 0] = 1.0  # First sample is positive
            
            # Create training pairs
            pairs = torch.zeros(batch_size, negative_samples + 1, user_pref.size(1), device=self.device)
            pairs[:, 0] = positive_repr  # First pair is positive
            
            # Generate negative samples by using other packages from the batch as negatives
            # We'll randomly sample from batch's purchased items
            purchased_packages = batch['purchased']['package_ids'].cpu().numpy()
            all_indices = np.arange(batch_size)
            
            for i in range(batch_size):
                # Get negative indices (packages from other users in batch)
                neg_indices = np.random.choice(
                    all_indices[all_indices != i], 
                    size=min(negative_samples, batch_size-1), 
                    replace=False
                )
                
                # If we don't have enough samples in the batch, sample from the whole package set
                if len(neg_indices) < negative_samples:
                    # Filter out the current positive sample
                    remaining = negative_samples - len(neg_indices)
                    all_packages = np.arange(1, self.model.num_packages + 1)  # Skip padding idx 0
                    mask = all_packages != purchased_packages[i]
                    valid_packages = all_packages[mask]
                    
                    extra_neg_indices = np.random.choice(
                        valid_packages, 
                        size=remaining, 
                        replace=False
                    )
                    
                    # Create embeddings for these packages
                    for j, neg_idx in enumerate(extra_neg_indices):
                        # Get package metadata
                        pkg_id = self.idx_to_package.get(neg_idx)
                        if pkg_id is None or pkg_id not in self.package_metadata:
                            # Use a random valid package if metadata not found
                            pkg_id = self.idx_to_package.get(np.random.choice(valid_packages))
                        
                        meta = self.package_metadata.get(pkg_id, {})
                        
                        # Create a small batch with just this negative sample
                        neg_batch = {
                            'user_id': batch['user_id'][i:i+1],
                            'purchased': {
                                'package_ids': torch.tensor([neg_idx], device=self.device),
                                'country_ids': torch.tensor([self.model.country_embedding.weight.size(0) - 1], device=self.device),
                                'category_ids': torch.tensor([self.model.category_embedding.weight.size(0) - 1], device=self.device),
                                'theme_ids': torch.tensor([self.model.theme_embedding.weight.size(0) - 1], device=self.device),
                                'price_values': torch.tensor([0.0], device=self.device)
                            }
                        }
                        
                        # Get representation
                        with torch.no_grad():
                            neg_repr = self.model.travel_package_encoder(neg_batch)['purchased']
                        
                        # Add to pairs
                        pairs[i, len(neg_indices) + j + 1] = neg_repr
                
                # Add batch negatives
                for j, neg_idx in enumerate(neg_indices):
                    pairs[i, j + 1] = outputs['purchased_representation'][neg_idx]
            
            # Calculate scores
            scores = torch.bmm(
                user_pref.unsqueeze(1),
                pairs.transpose(1, 2)
            ).squeeze(1)  # [batch_size, negative_samples + 1]
            
            # Compute loss
            loss = self.criterion(scores, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Calculate Hit@1 (accuracy)
            _, predicted = scores.max(1)
            correct += (predicted == 0).sum().item()  # 0 is the positive sample
            total += batch_size
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'hit@1': 100. * correct / total
            })
        
        # Calculate average metrics
        avg_loss = total_loss / len(self.train_loader)
        hit_rate = 100. * correct / total
        
        # Store metrics
        self.train_losses.append(avg_loss)
        self.train_hits.append(hit_rate)
        
        return avg_loss, hit_rate
    
    def evaluate(self, k_values=[1, 5, 10, 20]):
        """
        Evaluate the model on the test set
        
        Args:
            k_values: List of k values for Hit@k and MRR@k metrics
        
        Returns:
            tuple: (test_loss, hit_rates, mrr_values)
        """
        self.model.eval()
        total_loss = 0.0
        max_k = max(k_values)
        
        # Metrics
        hit_counts = {k: 0 for k in k_values}
        mrr_sum = {k: 0.0 for k in k_values}
        total = 0
        
        progress_bar = tqdm(self.test_loader, desc="Evaluation")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: {k2: v2.to(self.device) if isinstance(v2, torch.Tensor) else v2 
                            for k2, v2 in v.items()} if isinstance(v, dict) else v.to(self.device) 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch)
                user_pref = outputs['user_preference']
                positive_repr = outputs['purchased_representation']
                
                # Create positive samples
                batch_size = user_pref.size(0)
                labels = torch.zeros(batch_size, 1, device=self.device)
                labels[:, 0] = 1.0
                
                # Calculate positive scores
                positive_scores = torch.bmm(
                    user_pref.unsqueeze(1),
                    positive_repr.unsqueeze(2)
                ).squeeze()
                
                # Compute loss
                loss = self.criterion(positive_scores, labels.squeeze())
                total_loss += loss.item()
                
                # Evaluate on a larger set of candidates (top-k evaluation)
                # We'll use 1000 random packages as candidates
                candidate_indices = np.random.choice(
                    np.arange(1, self.model.num_packages + 1),  # Skip padding idx 0
                    size=min(999, self.model.num_packages - 1),  # Use all if less than 1000
                    replace=False
                )
                
                # Add the positive sample to candidates
                candidates = np.zeros(len(candidate_indices) + 1, dtype=np.int64)
                candidates[0] = batch['purchased']['package_ids'].cpu().numpy()  # Positive at index 0
                candidates[1:] = candidate_indices
                
                # Get candidate representations
                candidate_repr = []
                
                # Process in smaller batches to avoid GPU memory issues
                batch_size = 64
                num_batches = (len(candidates) + batch_size - 1) // batch_size
                
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(candidates))
                    
                    # Create candidate batch
                    candidate_batch = {
                        'user_id': batch['user_id'][:1].repeat(end_idx - start_idx),
                        'purchased': {
                            'package_ids': torch.tensor(candidates[start_idx:end_idx], device=self.device),
                            'country_ids': torch.ones(end_idx - start_idx, device=self.device).long(),  # Placeholder
                            'category_ids': torch.ones(end_idx - start_idx, device=self.device).long(),  # Placeholder
                            'theme_ids': torch.ones(end_idx - start_idx, device=self.device).long(),  # Placeholder
                            'price_values': torch.zeros(end_idx - start_idx, device=self.device)  # Placeholder
                        }
                    }
                    
                    # Replace placeholders with actual metadata
                    for j in range(start_idx, end_idx):
                        pkg_idx = candidates[j]
                        pkg_id = self.idx_to_package.get(pkg_idx)
                        
                        if pkg_id is not None and pkg_id in self.package_metadata:
                            meta = self.package_metadata[pkg_id]
                            
                            # Set country, category, theme (convert to index)
                            country = meta.get('country', '')
                            country_idx = self.model.country_embedding.weight.size(0) - 1  # Default to last index
                            
                            category = meta.get('category', '')
                            category_idx = self.model.category_embedding.weight.size(0) - 1
                            
                            theme = meta.get('theme', '')
                            theme_idx = self.model.theme_embedding.weight.size(0) - 1
                            
                            price = float(meta.get('min_price', 0))
                            
                            # Update candidate batch
                            j_batch = j - start_idx
                            candidate_batch['purchased']['country_ids'][j_batch] = country_idx
                            candidate_batch['purchased']['category_ids'][j_batch] = category_idx
                            candidate_batch['purchased']['theme_ids'][j_batch] = theme_idx
                            candidate_batch['purchased']['price_values'][j_batch] = price
                    
                    # Get representations
                    batch_repr = self.model.travel_package_encoder(candidate_batch)['purchased']
                    candidate_repr.append(batch_repr)
                
                # Concatenate all candidate representations
                candidate_repr = torch.cat(candidate_repr, dim=0)
                
                # Calculate recommendation scores
                scores = torch.matmul(user_pref, candidate_repr.t())  # [batch_size, num_candidates]
                
                # Get top-k predictions
                _, top_indices = scores.topk(max_k, dim=1)
                
                # Update metrics
                for i in range(batch_size):
                    # Check if positive sample (index 0) is in top-k
                    for k in k_values:
                        if 0 in top_indices[i, :k]:
                            hit_counts[k] += 1
                            
                            # Calculate MRR (Mean Reciprocal Rank)
                            rank = (top_indices[i, :k] == 0).nonzero(as_tuple=True)[0].item() + 1
                            mrr_sum[k] += 1.0 / rank
                
                total += batch_size
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': total_loss / (batch_idx + 1),
                    'hit@10': 100. * hit_counts[10] / total if 10 in hit_counts else 0
                })
        
        # Calculate average metrics
        avg_loss = total_loss / len(self.test_loader)
        hit_rates = {k: 100. * hit_counts[k] / total for k in k_values}
        mrr_values = {k: mrr_sum[k] / total for k in k_values}
        
        # Store metrics
        self.test_losses.append(avg_loss)
        self.test_hits.append(hit_rates[1])  # Use Hit@1 for plotting
        
        # Check if this is the best model
        if hit_rates[10] > self.best_hit_rate:
            self.best_hit_rate = hit_rates[10]
            self.save_model('best_model.pth')
        
        return avg_loss, hit_rates, mrr_values
    
    def train(self, num_epochs: int = 10, early_stopping: int = 5, k_values: List[int] = [1, 5, 10, 20]):
        """
        Train the model for multiple epochs
        
        Args:
            num_epochs: Maximum number of epochs to train
            early_stopping: Number of epochs without improvement before stopping
            k_values: List of k values for evaluation metrics
            
        Returns:
            tuple: (best_model_path, training_history)
        """
        print(f"Starting training for {num_epochs} epochs...")
        start_time = time.time()
        
        # Initialize early stopping variables
        best_epoch = 0
        epochs_without_improvement = 0
        training_history = {
            'train_loss': [],
            'train_hit': [],
            'test_loss': [],
            'test_hit': {},
            'test_mrr': {}
        }
        
        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss, train_hit = self.train_epoch(epoch)
            
            # Evaluate model
            test_loss, hit_rates, mrr_values = self.evaluate(k_values)
            
            # Print results
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Hit@1: {train_hit:.2f}%")
            print(f"  Test Loss: {test_loss:.4f}")
            for k in k_values:
                print(f"  Test Hit@{k}: {hit_rates[k]:.2f}%, MRR@{k}: {mrr_values[k]:.4f}")
            
            # Update training history
            training_history['train_loss'].append(train_loss)
            training_history['train_hit'].append(train_hit)
            training_history['test_loss'].append(test_loss)
            
            for k in k_values:
                if k not in training_history['test_hit']:
                    training_history['test_hit'][k] = []
                    training_history['test_mrr'][k] = []
                
                training_history['test_hit'][k].append(hit_rates[k])
                training_history['test_mrr'][k].append(mrr_values[k])
            
            # Save checkpoint
            self.save_model(f"checkpoint_epoch_{epoch+1}.pth")
            
            # Early stopping check
            if hit_rates[10] > self.best_hit_rate:
                self.best_hit_rate = hit_rates[10]
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                
                if epochs_without_improvement >= early_stopping:
                    print(f"No improvement for {early_stopping} epochs. Stopping training.")
                    break
        
        # Calculate training time
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Best model from epoch {best_epoch+1} with Hit@10: {self.best_hit_rate:.2f}%")
        
        # Save final model
        self.save_model("final_model.pth")
        
        # Plot training curves
        self._plot_training_curves(training_history)
        
        return os.path.join(self.model_dir, "best_model.pth"), training_history
    
    def save_model(self, filename: str):
        """
        Save model checkpoint
        
        Args:
            filename: Filename to save model to
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'train_hits': self.train_hits,
            'test_losses': self.test_losses,
            'test_hits': self.test_hits,
            'best_hit_rate': self.best_hit_rate
        }
        
        torch.save(checkpoint, os.path.join(self.model_dir, filename))
    
    def load_model(self, filename: str):
        """
        Load model checkpoint
        
        Args:
            filename: Filename to load model from
        """
        checkpoint = torch.load(os.path.join(self.model_dir, filename), map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.train_hits = checkpoint['train_hits']
        self.test_losses = checkpoint['test_losses']
        self.test_hits = checkpoint['test_hits']
        self.best_hit_rate = checkpoint['best_hit_rate']
        
        print(f"Loaded model from {filename}")
    
    def _plot_training_curves(self, history: Dict):
        """
        Plot training curves
        
        Args:
            history: Training history dictionary
        """
        # Create figure
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
        
        # Plot training and test loss
        axes[0, 0].plot(history['train_loss'], label='Train')
        axes[0, 0].plot(history['test_loss'], label='Test')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Plot training and test Hit@1
        axes[0, 1].plot(history['train_hit'], label='Train')
        axes[0, 1].plot(history['test_hit'][1], label='Test')
        axes[0, 1].set_title('Hit@1')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Hit@1 (%)')
        axes[0, 1].legend()
        
        # Plot test Hit@k
        for k in history['test_hit']:
            axes[1, 0].plot(history['test_hit'][k], label=f'Hit@{k}')
        axes[1, 0].set_title('Test Hit@k')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Hit@k (%)')
        axes[1, 0].legend()
        
        # Plot test MRR@k
        for k in history['test_mrr']:
            axes[1, 1].plot(history['test_mrr'][k], label=f'MRR@{k}')
        axes[1, 1].set_title('Test MRR@k')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MRR@k')
        axes[1, 1].legend()
        
        # Save figure
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.log_dir, f'training_curves_{timestamp}.png'))
        plt.close()
    
    def generate_recommendations(self, 
                                 user_id: int, 
                                 short_term_packages: List[int], 
                                 long_term_packages: List[int] = None,
                                 top_k: int = 10):
        """
        Generate recommendations for a user
        
        Args:
            user_id: User ID
            short_term_packages: List of recently viewed package IDs
            long_term_packages: List of previously viewed package IDs
            top_k: Number of recommendations to generate
            
        Returns:
            list: Top-k recommended package IDs with scores
        """
        self.model.eval()
        
        # Create input batch
        batch = {
            'user_id': torch.tensor([user_id], device=self.device),
            'short_term': {
                'package_ids': torch.tensor([short_term_packages + [0] * (self.model.max_short_term - len(short_term_packages))], device=self.device),
                'country_ids': torch.zeros(1, self.model.max_short_term, device=self.device).long(),
                'category_ids': torch.zeros(1, self.model.max_short_term, device=self.device).long(),
                'theme_ids': torch.zeros(1, self.model.max_short_term, device=self.device).long(),
                'price_values': torch.zeros(1, self.model.max_short_term, device=self.device)
            },
            'long_term': {
                'package_ids': torch.tensor([long_term_packages + [0] * (self.model.max_long_term - len(long_term_packages)) if long_term_packages else [0] * self.model.max_long_term], device=self.device),
                'country_ids': torch.zeros(1, self.model.max_long_term, device=self.device).long(),
                'category_ids': torch.zeros(1, self.model.max_long_term, device=self.device).long(),
                'theme_ids': torch.zeros(1, self.model.max_long_term, device=self.device).long(),
                'price_values': torch.zeros(1, self.model.max_long_term, device=self.device)
            }
        }
        
        # Fill in package metadata
        for i, pkg_idx in enumerate(short_term_packages[:self.model.max_short_term]):
            pkg_id = self.idx_to_package.get(pkg_idx)
            if pkg_id is not None and pkg_id in self.package_metadata:
                meta = self.package_metadata[pkg_id]
                
                # Set country, category, theme (need to convert to index)
                # This depends on your mapping implementation
                # Here's a simplified version assuming you have direct mappings
                country = meta.get('country', '')
                country_idx = 1  # Default index if not found
                
                category = meta.get('category', '')
                category_idx = 1
                
                theme = meta.get('theme', '')
                theme_idx = 1
                
                price = float(meta.get('min_price', 0))
                
                batch['short_term']['country_ids'][0, i] = country_idx
                batch['short_term']['category_ids'][0, i] = category_idx
                batch['short_term']['theme_ids'][0, i] = theme_idx
                batch['short_term']['price_values'][0, i] = price
        
        if long_term_packages:
            for i, pkg_idx in enumerate(long_term_packages[:self.model.max_long_term]):
                pkg_id = self.idx_to_package.get(pkg_idx)
                if pkg_id is not None and pkg_id in self.package_metadata:
                    meta = self.package_metadata[pkg_id]
                    
                    # Set metadata similar to above
                    country = meta.get('country', '')
                    country_idx = 1
                    
                    category = meta.get('category', '')
                    category_idx = 1
                    
                    theme = meta.get('theme', '')
                    theme_idx = 1
                    
                    price = float(meta.get('min_price', 0))
                    
                    batch['long_term']['country_ids'][0, i] = country_idx
                    batch['long_term']['category_ids'][0, i] = category_idx
                    batch['long_term']['theme_ids'][0, i] = theme_idx
                    batch['long_term']['price_values'][0, i] = price
        
        # Generate user preference
        with torch.no_grad():
            outputs = self.model(batch)
            user_pref = outputs['user_preference']
        
        # Get all package representations
        all_packages = list(self.package_to_idx.values())
        all_package_repr = []
        
        # Process in batches to avoid memory issues
        batch_size = 100
        num_batches = (len(all_packages) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(all_packages))
            
            pkg_batch = {
                'user_id': batch['user_id'].repeat(end_idx - start_idx),
                'purchased': {
                    'package_ids': torch.tensor(all_packages[start_idx:end_idx], device=self.device),
                    'country_ids': torch.ones(end_idx - start_idx, device=self.device).long(),
                    'category_ids': torch.ones(end_idx - start_idx, device=self.device).long(),
                    'theme_ids': torch.ones(end_idx - start_idx, device=self.device).long(),
                    'price_values': torch.zeros(end_idx - start_idx, device=self.device)
                }
            }
            
            # Fill in metadata
            for j in range(start_idx, end_idx):
                pkg_idx = all_packages[j]
                pkg_id = self.idx_to_package.get(pkg_idx)
                
                if pkg_id is not None and pkg_id in self.package_metadata:
                    meta = self.package_metadata[pkg_id]
                    
                    # Set metadata
                    country = meta.get('country', '')
                    country_idx = 1
                    
                    category = meta.get('category', '')
                    category_idx = 1
                    
                    theme = meta.get('theme', '')
                    theme_idx = 1
                    
                    price = float(meta.get('min_price', 0))
                    
                    j_batch = j - start_idx
                    pkg_batch['purchased']['country_ids'][j_batch] = country_idx
                    pkg_batch['purchased']['category_ids'][j_batch] = category_idx
                    pkg_batch['purchased']['theme_ids'][j_batch] = theme_idx
                    pkg_batch['purchased']['price_values'][j_batch] = price
            
            # Get representations
            with torch.no_grad():
                pkg_repr = self.model.travel_package_encoder(pkg_batch)['purchased']
                all_package_repr.append(pkg_repr)
        
        # Concatenate all representations
        all_package_repr = torch.cat(all_package_repr, dim=0)
        
        # Calculate scores
        with torch.no_grad():
            scores = torch.matmul(user_pref, all_package_repr.t()).squeeze()
        
        # Get top-k recommendations
        scores_np = scores.cpu().numpy()
        top_indices = np.argsort(-scores_np)[:top_k]
        
        # Convert to package IDs and scores
        recommendations = []
        for idx in top_indices:
            pkg_idx = all_packages[idx]
            pkg_id = self.idx_to_package.get(pkg_idx)
            score = float(scores_np[idx])
            
            recommendations.append({
                'package_index': pkg_idx,
                'package_id': pkg_id,
                'score': score
            })
        
        return recommendations
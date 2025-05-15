# utils/metrics.py
"""
Evaluation metrics for recommendation models
Includes Hit@k, MRR@k, Item Coverage, and NDCG@k
"""

import torch
import numpy as np
from typing import Dict, List, Optional


def hit_rate_at_k(scores: torch.Tensor, ground_truth: torch.Tensor, k: int = 10) -> float:
    """
    Calculate Hit Rate@k (also known as Recall@k)
    
    Args:
        scores: Predicted scores for all candidates [batch_size, num_candidates]
        ground_truth: Index of ground truth items [batch_size]
        k: Number of top items to consider
    
    Returns:
        Hit Rate@k (float): Proportion of cases where the correct item is in top-k
    """
    batch_size = scores.size(0)
    
    # Get top-k predictions
    _, top_indices = torch.topk(scores, k=min(k, scores.size(1)), dim=1)
    
    # Check if ground truth is in top-k
    # Expand ground truth to compare with all top-k predictions
    ground_truth_expanded = ground_truth.unsqueeze(1).expand_as(top_indices)
    hits = (top_indices == ground_truth_expanded).any(dim=1)
    
    # Calculate hit rate
    hit_rate = hits.float().mean().item()
    
    return hit_rate


def mrr_at_k(scores: torch.Tensor, ground_truth: torch.Tensor, k: int = 10) -> float:
    """
    Calculate Mean Reciprocal Rank@k
    
    Args:
        scores: Predicted scores for all candidates [batch_size, num_candidates]
        ground_truth: Index of ground truth items [batch_size]
        k: Number of top items to consider
    
    Returns:
        MRR@k (float): Mean reciprocal rank
    """
    batch_size = scores.size(0)
    
    # Get top-k predictions
    _, top_indices = torch.topk(scores, k=min(k, scores.size(1)), dim=1)
    
    # Check where ground truth appears in top-k
    ground_truth_expanded = ground_truth.unsqueeze(1).expand_as(top_indices)
    hits = (top_indices == ground_truth_expanded)
    
    # Get the position (rank) of the hit
    ranks = torch.zeros(batch_size, dtype=torch.float, device=scores.device)
    
    for i in range(batch_size):
        hit_positions = hits[i].nonzero(as_tuple=True)[0]
        if len(hit_positions) > 0:
            # +1 because ranks start from 1, not 0
            first_hit_position = hit_positions[0].item() + 1
            ranks[i] = 1.0 / first_hit_position
    
    # Calculate MRR
    mrr = ranks.mean().item()
    
    return mrr


def ndcg_at_k(scores: torch.Tensor, ground_truth: torch.Tensor, k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain@k
    
    Args:
        scores: Predicted scores for all candidates [batch_size, num_candidates]
        ground_truth: Index of ground truth items [batch_size]
        k: Number of top items to consider
    
    Returns:
        NDCG@k (float): Average NDCG score
    """
    batch_size = scores.size(0)
    
    # Get top-k predictions
    _, top_indices = torch.topk(scores, k=min(k, scores.size(1)), dim=1)
    
    # Calculate DCG
    dcg = torch.zeros(batch_size, dtype=torch.float, device=scores.device)
    
    for i in range(batch_size):
        # Check if ground truth is in top-k
        ground_truth_positions = (top_indices[i] == ground_truth[i]).nonzero(as_tuple=True)[0]
        
        if len(ground_truth_positions) > 0:
            # Position in ranking (1-indexed)
            position = ground_truth_positions[0].item() + 1
            # DCG = relevance / log2(position + 1)
            # Since we have binary relevance (1 for correct item, 0 otherwise)
            dcg[i] = 1.0 / np.log2(position + 1)
    
    # IDCG (Ideal DCG) for binary relevance is always 1/log2(2) = 1
    idcg = 1.0
    
    # Calculate NDCG
    ndcg = (dcg / idcg).mean().item()
    
    return ndcg


def item_coverage_at_k(all_predictions: List[torch.Tensor], num_items: int, k: int = 10) -> float:
    """
    Calculate Item Coverage@k - the percentage of items that appear in top-k recommendations
    
    Args:
        all_predictions: List of prediction score tensors across all batches
        num_items: Total number of items in the catalog
        k: Number of top items to consider
    
    Returns:
        Item Coverage@k (float): Proportion of items recommended at least once
    """
    recommended_items = set()
    
    for scores in all_predictions:
        # Get top-k items for each user in the batch
        _, top_indices = torch.topk(scores, k=min(k, scores.size(1)), dim=1)
        
        # Add to the set of recommended items
        recommended_items.update(top_indices.cpu().numpy().flatten().tolist())
    
    # Calculate coverage
    coverage = len(recommended_items) / num_items
    
    return coverage


def diversity_at_k(scores: torch.Tensor, item_features: Optional[torch.Tensor] = None, k: int = 10) -> float:
    """
    Calculate diversity of recommendations (if item features are available)
    
    Args:
        scores: Predicted scores for all candidates [batch_size, num_candidates]
        item_features: Feature vectors for items [num_candidates, feature_dim]
        k: Number of top items to consider
    
    Returns:
        Diversity score (float): Average pairwise distance between recommended items
    """
    if item_features is None:
        return 0.0
    
    batch_size = scores.size(0)
    _, top_indices = torch.topk(scores, k=min(k, scores.size(1)), dim=1)
    
    diversity_scores = []
    
    for i in range(batch_size):
        # Get features of top-k items
        top_k_features = item_features[top_indices[i]]
        
        # Calculate pairwise distances
        # Using cosine distance: 1 - cosine_similarity
        normalized_features = F.normalize(top_k_features, p=2, dim=1)
        similarity_matrix = torch.matmul(normalized_features, normalized_features.t())
        
        # Get upper triangular part (excluding diagonal)
        mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1).bool()
        pairwise_similarities = similarity_matrix[mask]
        
        # Convert to distance
        pairwise_distances = 1 - pairwise_similarities
        
        # Average distance
        if len(pairwise_distances) > 0:
            diversity_scores.append(pairwise_distances.mean().item())
    
    return np.mean(diversity_scores) if diversity_scores else 0.0


def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                     k_values: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
    """
    Calculate multiple metrics at once
    
    Args:
        predictions: Predicted scores [batch_size, num_items]
        targets: Ground truth indices [batch_size]
        k_values: List of k values to evaluate
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    for k in k_values:
        metrics[f'hit@{k}'] = hit_rate_at_k(predictions, targets, k)
        metrics[f'mrr@{k}'] = mrr_at_k(predictions, targets, k)
        metrics[f'ndcg@{k}'] = ndcg_at_k(predictions, targets, k)
    
    return metrics


def calculate_mrr(predictions: torch.Tensor, targets: torch.Tensor,
                  k_values: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
    """
    Calculate MRR at multiple k values
    
    Args:
        predictions: Predicted scores [batch_size, num_items]
        targets: Ground truth indices [batch_size]
        k_values: List of k values to evaluate
    
    Returns:
        Dictionary of MRR values
    """
    mrr_scores = {}
    
    for k in k_values:
        mrr_scores[f'mrr@{k}'] = mrr_at_k(predictions, targets, k)
    
    return mrr_scores


class MetricsTracker:
    """
    Track metrics during training and evaluation
    """
    def __init__(self, k_values: List[int] = [1, 5, 10, 20]):
        self.k_values = k_values
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics"""
        self.metrics = {
            'loss': [],
            'hit': {k: [] for k in self.k_values},
            'mrr': {k: [] for k in self.k_values},
            'ndcg': {k: [] for k in self.k_values}
        }
        self.num_samples = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, 
               loss: Optional[float] = None):
        """Update metrics with a new batch"""
        batch_size = predictions.size(0)
        self.num_samples += batch_size
        
        if loss is not None:
            self.metrics['loss'].append(loss)
        
        for k in self.k_values:
            self.metrics['hit'][k].append(hit_rate_at_k(predictions, targets, k))
            self.metrics['mrr'][k].append(mrr_at_k(predictions, targets, k))
            self.metrics['ndcg'][k].append(ndcg_at_k(predictions, targets, k))
    
    def compute(self) -> Dict[str, float]:
        """Compute average metrics"""
        results = {}
        
        if self.metrics['loss']:
            results['loss'] = np.mean(self.metrics['loss'])
        
        for k in self.k_values:
            results[f'hit@{k}'] = np.mean(self.metrics['hit'][k])
            results[f'mrr@{k}'] = np.mean(self.metrics['mrr'][k])
            results[f'ndcg@{k}'] = np.mean(self.metrics['ndcg'][k])
        
        return results


# Example usage for debugging
if __name__ == "__main__":
    # Test metrics calculation
    batch_size = 32
    num_items = 1000
    
    # Simulate predictions and ground truth
    predictions = torch.randn(batch_size, num_items)
    targets = torch.randint(0, num_items, (batch_size,))
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, targets)
    
    print("Test Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Test metrics tracker
    tracker = MetricsTracker()
    
    # Simulate multiple batches
    for _ in range(10):
        predictions = torch.randn(batch_size, num_items)
        targets = torch.randint(0, num_items, (batch_size,))
        loss = torch.rand(1).item()
        
        tracker.update(predictions, targets, loss)
    
    # Compute average metrics
    avg_metrics = tracker.compute()
    
    print("\nAverage Metrics:")
    for metric, value in avg_metrics.items():
        print(f"  {metric}: {value:.4f}")
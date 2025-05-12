# utils/metrics.py
import torch
import numpy as np

def hit_rate_at_k(scores, ground_truth, k=10):
    """
    Calculate Hit Rate@k
    
    Args:
        scores: Predicted scores for all candidates [batch_size, num_candidates]
        ground_truth: Index of ground truth items [batch_size]
        k: Number of top items to consider
    
    Returns:
        Hit Rate@k (float)
    """
    # Get top-k predictions
    _, top_indices = torch.topk(scores, k=k, dim=1)
    
    # Check if ground truth is in top-k
    hits = (top_indices == ground_truth.unsqueeze(1)).any(dim=1)
    
    # Calculate hit rate
    hit_rate = hits.float().mean().item()
    
    return hit_rate

def mrr_at_k(scores, ground_truth, k=10):
    """
    Calculate Mean Reciprocal Rank@k
    
    Args:
        scores: Predicted scores for all candidates [batch_size, num_candidates]
        ground_truth: Index of ground truth items [batch_size]
        k: Number of top items to consider
    
    Returns:
        MRR@k (float)
    """
    # Get top-k predictions
    _, top_indices = torch.topk(scores, k=k, dim=1)
    
    # Check if ground truth is in top-k
    hits = (top_indices == ground_truth.unsqueeze(1))
    
    # Get the position (rank) of the hit
    ranks = torch.zeros_like(ground_truth, dtype=torch.float)
    
    for i, (hit, gt) in enumerate(zip(hits, ground_truth)):
        hit_positions = hit.nonzero(as_tuple=True)[0]
        if len(hit_positions) > 0:
            # +1 because ranks start from 1
            ranks[i] = 1.0 / (hit_positions[0].item() + 1)
    
    # Calculate MRR
    mrr = ranks.mean().item()
    
    return mrr
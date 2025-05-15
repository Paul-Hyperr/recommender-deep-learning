import torch
import torch.nn as nn
import torch.nn.functional as F

class NaturalPurchaseLoss(nn.Module):
    """
    Simple loss that lets the model learn event importance naturally
    Only emphasizes that purchases are the true positive signal
    """
    def __init__(self, purchase_boost=5.0):
        super().__init__()
        self.purchase_boost = purchase_boost  # Only parameter: how much to boost purchases
        
    def forward(self, predictions, targets, is_purchase):
        """
        Standard cross-entropy with natural purchase weighting
        
        Args:
            predictions: Model predictions [batch_size, num_items]
            targets: Target item indices [batch_size]
            is_purchase: Boolean tensor indicating purchases [batch_size]
        """
        # Basic cross-entropy loss
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # Only boost purchases, let model figure out the rest
        weights = torch.ones_like(ce_loss)
        weights[is_purchase] = self.purchase_boost
        
        # Return weighted mean
        return (ce_loss * weights).mean()


class PurchaseFocusedMetrics:
    """Calculate metrics with focus on purchase prediction"""
    
    @staticmethod
    def calculate_metrics(predictions, targets, is_purchase, k_values=[5, 10, 20]):
        """
        Calculate both overall and purchase-specific metrics with vectorized operations
        
        Returns:
            dict: Metrics including recall@k, precision@k, and MRR
        """
        batch_size = predictions.size(0)
        max_k = max(k_values)
        _, top_k_indices = torch.topk(predictions, max_k, dim=1)
        
        metrics = {
            'recall@k': {k: 0.0 for k in k_values},
            'purchase_recall@k': {k: 0.0 for k in k_values},
            'purchase_mrr': 0.0,
            'overall_mrr': 0.0
        }
        
        purchase_count = is_purchase.sum().item()
        
        # Vectorized rank calculation
        # Expand targets to match shape of top_k_indices for efficient comparison
        expanded_targets = targets.unsqueeze(1).expand_as(top_k_indices)
        matches = (top_k_indices == expanded_targets)
        
        # Get the first match position for each sample
        match_positions = torch.argmax(matches.float(), dim=1)
        
        # Only consider positions where there actually was a match
        has_match = matches.any(dim=1)
        valid_positions = match_positions[has_match]
        
        # Calculate metrics efficiently
        if len(valid_positions) > 0:
            # Add 1 because ranks start from 1, not 0
            ranks = valid_positions + 1
            
            # Overall MRR for samples with matches
            metrics['overall_mrr'] = (1.0 / ranks.float()).sum().item() 
            
            # Purchase-specific MRR
            if purchase_count > 0:
                purchase_matches = has_match & is_purchase
                if purchase_matches.any():
                    purchase_ranks = match_positions[purchase_matches] + 1
                    metrics['purchase_mrr'] = (1.0 / purchase_ranks.float()).sum().item()
            
            # Calculate recall@k efficiently
            for k in k_values:
                # Convert ranks to boolean: is rank <= k?
                in_top_k = (ranks <= k)
                metrics['recall@k'][k] = in_top_k.float().sum().item()
                
                # Purchase-specific recall@k
                if purchase_count > 0:
                    purchase_in_top_k = matches[:, :k].any(dim=1) & is_purchase
                    metrics['purchase_recall@k'][k] = purchase_in_top_k.sum().item()
        
        # Normalize
        for k in k_values:
            metrics['recall@k'][k] /= batch_size
            if purchase_count > 0:
                metrics['purchase_recall@k'][k] /= purchase_count
        
        metrics['overall_mrr'] /= batch_size
        if purchase_count > 0:
            metrics['purchase_mrr'] /= purchase_count
        
        return metrics
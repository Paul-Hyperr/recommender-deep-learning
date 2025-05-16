import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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


class FocalLoss(nn.Module):
    """
    Enhanced Focal Loss for recommendation systems with severe class imbalance
    
    Focal loss addresses class imbalance by focusing on hard examples by reducing the 
    weight of well-classified examples.
    
    In recommendation scenarios, this helps the model focus more on examples that it gets wrong,
    which is especially useful for rare purchases.
    
    Paper: "Focal Loss for Dense Object Detection" - https://arxiv.org/abs/1708.02002
    """
    def __init__(self, purchase_boost=10.0, gamma=2.5, alpha=0.25, 
                 adaptive_gamma=True, online_hard_mining=True):
        super().__init__()
        self.purchase_boost = purchase_boost  # Purchase importance weight
        self.base_gamma = gamma              # Base focusing parameter (higher = more focus on hard examples)
        self.alpha = alpha                   # Class balance weight
        
        # Enable adaptive gamma for purchases vs non-purchases
        self.adaptive_gamma = adaptive_gamma
        
        # Enable online hard negative mining (focus on the hardest examples)
        self.online_hard_mining = online_hard_mining
        self.hard_mining_ratio = 0.5         # Use top 50% hardest negatives
        
        # Running stats for monitoring loss stability
        self.running_purchase_loss = 0
        self.running_non_purchase_loss = 0
        self.beta = 0.9  # Smoothing factor for running stats
        
    def forward(self, predictions, targets, is_purchase):
        """
        Enhanced focal loss with special handling for extreme recommendation imbalance
        
        Args:
            predictions: Model predictions [batch_size, num_items]
            targets: Target item indices [batch_size]
            is_purchase: Boolean tensor indicating purchases [batch_size]
        """
        # Get the softmax probs first
        probs = F.softmax(predictions, dim=1)
        
        # Get the probability of the correct class for each sample (pt in the paper)
        batch_size = predictions.size(0)
        pt = torch.zeros(batch_size, device=predictions.device)
        for i in range(batch_size):
            pt[i] = probs[i, targets[i]]
        
        # Adjust gamma based on whether it's a purchase or not (higher gamma for purchases)
        if self.adaptive_gamma:
            gamma = torch.ones_like(pt) * self.base_gamma
            # Increase gamma for purchases to focus more on hard purchase examples
            gamma[is_purchase] = self.base_gamma * 1.5
        else:
            gamma = self.base_gamma
            
        # Focal loss formula
        # -alpha * (1-pt)^gamma * log(pt)
        focal_weight = (1 - pt).pow(gamma)
        focal_loss = -torch.log(pt + 1e-10)  # Add epsilon for stability
        weighted_focal_loss = self.alpha * focal_weight * focal_loss
        
        # Add greater boosting for purchases - this creates extreme focus on purchases
        weights = torch.ones_like(weighted_focal_loss)
        weights[is_purchase] = self.purchase_boost
        
        # Apply harder weights to purchases with lower probabilities (very hard examples)
        purchase_mask = is_purchase.float()
        inverse_pt = 1.0 - pt
        hard_purchase_weight = 1.0 + inverse_pt * purchase_mask * 0.5  # Extra weight for hard purchases
        weights = weights * hard_purchase_weight
        
        # Online hard negative mining - focus training on the hardest negative examples
        if self.online_hard_mining and not is_purchase.all():
            # Only apply to non-purchase examples
            non_purchase_loss = weighted_focal_loss * (1 - purchase_mask)
            non_purchase_idx = (~is_purchase).nonzero(as_tuple=True)[0]
            
            if len(non_purchase_idx) > 0:
                # Sort non-purchase losses in descending order
                sorted_non_purchase_loss, _ = non_purchase_loss[non_purchase_idx].sort(descending=True)
                
                # Keep only the top k% hardest examples
                cutoff_idx = int(len(sorted_non_purchase_loss) * self.hard_mining_ratio)
                if cutoff_idx > 0:
                    loss_threshold = sorted_non_purchase_loss[cutoff_idx-1]
                    
                    # Create hard mining mask - only keep losses above threshold
                    hard_negative_mask = torch.ones_like(weighted_focal_loss)
                    hard_negative_mask[non_purchase_idx] = (non_purchase_loss[non_purchase_idx] >= loss_threshold).float()
                    
                    # Apply the mask to weights
                    weights = weights * (purchase_mask + (1 - purchase_mask) * hard_negative_mask)
        
        # Calculate final weighted loss
        final_loss = weighted_focal_loss * weights
        
        # Track separate loss values for purchases vs non-purchases for monitoring
        if is_purchase.any():
            purchase_loss = (final_loss * purchase_mask).sum() / purchase_mask.sum()
            self.running_purchase_loss = self.beta * self.running_purchase_loss + (1 - self.beta) * purchase_loss.item()
        
        if (~is_purchase).any():
            non_purchase_mask = 1.0 - purchase_mask
            non_purchase_loss = (final_loss * non_purchase_mask).sum() / non_purchase_mask.sum()
            self.running_non_purchase_loss = self.beta * self.running_non_purchase_loss + (1 - self.beta) * non_purchase_loss.item()
            
        # Return weighted mean - normalize by sum of weights for more stable gradients
        return final_loss.mean()


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
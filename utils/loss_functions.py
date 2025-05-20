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


class ContrastiveEventLoss(nn.Module):
    """
    Contrastive loss with temperature scaling and event-based margins
    
    This loss function lets the model organically learn the importance of different events
    by applying contrastive margins rather than explicit weights:
    - Purchase events: No margin (strongest positive signal)
    - Checkout events: Small margin (partial positive signal)
    - AddToCart events: Medium margin (weaker positive signal)
    - ViewContent events: Full margin (regular browsing signal)
    
    The temperature parameter controls how much to amplify small differences in similarity.
    Lower temperature values create sharper distinctions between predicted probabilities.
    
    Args:
        temperature (float): Scaling factor for logits. Lower values increase sensitivity to differences
        purchase_margin (float): Margin for purchase events (usually 0.0 - strongest signal)
        checkout_margin (float): Margin for checkout events (usually 0.3 - strong signal)
        add_to_cart_margin (float): Margin for add to cart events (usually 0.6 - medium signal)
        view_margin (float): Margin for view events (usually 1.0 - weakest signal)
        hard_negative_mining (bool): Whether to use hard negative mining
        hard_negative_ratio (float): Proportion of hard negatives to keep
    """
    def __init__(self, temperature=0.1, purchase_margin=0.0, checkout_margin=0.3, 
                 view_margin=1.0, hard_negative_mining=True, hard_negative_ratio=0.7,
                 add_to_cart_margin=0.6):
        super().__init__()
        self.temperature = temperature  # Controls sensitivity to similarity differences
        self.purchase_margin = purchase_margin  # No margin for purchases
        self.checkout_margin = checkout_margin  # Small margin for checkout events
        self.view_margin = view_margin  # Full margin for view events
        self.add_to_cart_margin = add_to_cart_margin  # Medium margin for add to cart events
        self.hard_negative_mining = hard_negative_mining  # Whether to use hard negative mining
        self.hard_negative_ratio = hard_negative_ratio  # Proportion of hard negatives to keep
    
    def forward(self, predictions, targets, is_purchase, has_checkout=None, has_add_to_cart=None):
        """
        Forward pass with contrastive margins for different event types
        
        Args:
            predictions: Predicted scores for each package [batch_size, num_packages]
            targets: Target package indices [batch_size]
            is_purchase: Boolean tensor indicating purchase samples [batch_size]
            has_checkout: Boolean tensor indicating checkout samples [batch_size]
            has_add_to_cart: Boolean tensor indicating add_to_cart samples [batch_size]
        
        Returns:
            Loss value
        """
        # For cross entropy calculation, we need to mask invalid indices (-1)
        valid_mask = (targets >= 0).float()
        
        # Scale logits by temperature
        # Lower temperature = more emphasis on top predictions
        logits = predictions / self.temperature
        
        # Create masks for different event types
        purchase_mask = is_purchase.float()
        
        # Initialize checkout and add_to_cart masks
        checkout_mask = torch.zeros_like(purchase_mask)
        add_to_cart_mask = torch.zeros_like(purchase_mask)
        
        # Only use checkout info if provided
        if has_checkout is not None:
            # Only count as checkout if NOT also a purchase
            checkout_mask = has_checkout.float() * (~is_purchase).float()
        
        # Only use add_to_cart info if provided
        if has_add_to_cart is not None:
            # Only count as add_to_cart if NOT also a purchase or checkout
            add_to_cart_mask = has_add_to_cart.float() * (~is_purchase).float() * (~has_checkout).float() \
                if has_checkout is not None else has_add_to_cart.float() * (~is_purchase).float()
        
        # Regular browsing is everything else
        view_mask = 1.0 - purchase_mask - checkout_mask - add_to_cart_mask
        
        # Apply different contrastive margins based on event type
        # The margin is subtracted from the logits, making it harder to predict
        # correctly for that class. Higher margin = more penalty.
        margins = torch.zeros_like(logits)
        
        batch_size = predictions.size(0)
        for i in range(batch_size):
            target_idx = targets[i].item()
            if target_idx >= 0:  # Valid target
                # Set margin for the target item based on event type
                if purchase_mask[i] > 0:
                    # Purchase event - no/minimal margin (easiest to learn)
                    margins[i, target_idx] = self.purchase_margin
                elif checkout_mask[i] > 0:
                    # Checkout event - small margin
                    margins[i, target_idx] = self.checkout_margin
                elif add_to_cart_mask[i] > 0:
                    # Add to cart event - medium margin
                    margins[i, target_idx] = self.add_to_cart_margin
                else:
                    # View event - largest margin (hardest to learn)
                    margins[i, target_idx] = self.view_margin
        
        # Apply margins to logits
        margin_logits = logits - margins
        
        # Calculate cross entropy loss with temperature scaling
        ce_loss = nn.functional.cross_entropy(margin_logits, targets, reduction='none')
        
        # Apply hard negative mining if enabled
        if self.hard_negative_mining:
            # Sort losses in descending order, but only for view samples
            view_losses = ce_loss * view_mask
            
            if view_mask.sum() > 10:  # Only if we have enough view samples
                sorted_losses, _ = torch.sort(view_losses[view_mask > 0], descending=True)
                
                # Take top N% hardest samples
                cutoff_idx = int(len(sorted_losses) * self.hard_negative_ratio)
                if cutoff_idx > 0:
                    hard_mining_threshold = sorted_losses[cutoff_idx]
                    
                    # Create hard mining mask:
                    # - Keep all purchases
                    # - Keep all checkouts
                    # - Keep all add_to_carts
                    # - Only keep hard negatives from view samples
                    mining_mask = (
                        purchase_mask +  # Keep all purchases
                        checkout_mask +  # Keep all checkouts
                        add_to_cart_mask +  # Keep all add to carts
                        (view_mask * (view_losses >= hard_mining_threshold).float())  # Hard negatives
                    )
                else:
                    mining_mask = torch.ones_like(purchase_mask)
            else:
                mining_mask = torch.ones_like(purchase_mask)
        else:
            mining_mask = torch.ones_like(purchase_mask)
        
        # Apply mining mask and valid mask
        masked_loss = ce_loss * mining_mask * valid_mask
        
        # Return mean loss
        return masked_loss.sum() / (valid_mask.sum() + 1e-6)
    
    def get_margins(self):
        """Return current margin values for reporting"""
        return {
            "purchase_margin": self.purchase_margin,
            "checkout_margin": self.checkout_margin,
            "add_to_cart_margin": self.add_to_cart_margin,
            "view_margin": self.view_margin,
            "temperature": self.temperature
        }


class CheckoutEnhancedLoss(nn.Module):
    """
    Enhanced loss with special handling for both purchase and checkout events
    
    This loss handles four distinct types of samples:
    1. Purchase events (strongest signal)
    2. Checkout events (strong intent signal, but not as strong as purchase)
    3. AddToCart events (medium intent signal)
    4. Regular browsing samples (weakest signal)
    
    Each type gets appropriate weighting to prioritize the right signals
    """
    def __init__(self, purchase_boost=15.0, checkout_boost=7.5, add_to_cart_boost=3.0):
        super().__init__()
        self.purchase_boost = purchase_boost  # Highest weight for purchases
        self.checkout_boost = checkout_boost  # Medium-high weight for checkouts
        self.add_to_cart_boost = add_to_cart_boost  # Medium weight for add-to-cart
        
    def forward(self, predictions, targets, is_purchase, has_checkout=None, has_add_to_cart=None):
        """
        Forward pass with hierarchical weighting for different event types
        
        Args:
            predictions: Predicted scores for each package
            targets: Target package indices
            is_purchase: Boolean tensor indicating purchase samples
            has_checkout: Boolean tensor indicating checkout samples
            has_add_to_cart: Boolean tensor indicating add_to_cart samples
        
        Returns:
            Loss value
        """
        # Basic cross-entropy loss
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # Initialize weights with base value
        weights = torch.ones_like(ce_loss)
        
        # Apply purchase boost (highest priority)
        weights[is_purchase] = self.purchase_boost
        
        # Apply checkout boost (if provided and not a purchase)
        if has_checkout is not None:
            # Only apply checkout boost if it's not already a purchase
            checkout_only = has_checkout & (~is_purchase)
            weights[checkout_only] = self.checkout_boost
        
        # Apply add_to_cart boost (if provided and not already boosted)
        if has_add_to_cart is not None:
            # Only apply if not already a purchase or checkout
            if has_checkout is not None:
                add_to_cart_only = has_add_to_cart & (~is_purchase) & (~has_checkout)
            else:
                add_to_cart_only = has_add_to_cart & (~is_purchase)
            
            weights[add_to_cart_only] = self.add_to_cart_boost
        
        # Return weighted mean loss
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


class EnhancedEventMetrics:
    """Calculate comprehensive metrics with different event types"""
    
    @staticmethod
    def calculate_metrics(predictions, targets, is_purchase, 
                          has_checkout=None, has_add_to_cart=None, 
                          k_values=[5, 10, 20]):
        """
        Calculate metrics for different event types with vectorized operations
        
        Args:
            predictions: Predicted scores for each package
            targets: Target package indices
            is_purchase: Boolean tensor indicating purchase samples
            has_checkout: Boolean tensor indicating checkout samples (optional)
            has_add_to_cart: Boolean tensor indicating add_to_cart samples (optional)
            k_values: List of k values for recall@k calculation
            
        Returns:
            dict: Comprehensive metrics including event-type specific recalls and MRR
        """
        batch_size = predictions.size(0)
        max_k = max(k_values)
        _, top_k_indices = torch.topk(predictions, max_k, dim=1)
        
        # Initialize metrics dictionary
        metrics = {
            'recall@k': {k: 0.0 for k in k_values},
            'purchase_recall@k': {k: 0.0 for k in k_values},
            'overall_mrr': 0.0,
            'purchase_mrr': 0.0
        }
        
        # Add checkout metrics if checkout info provided
        if has_checkout is not None:
            metrics['checkout_recall@k'] = {k: 0.0 for k in k_values}
            metrics['checkout_mrr'] = 0.0
            metrics['intent_recall@k'] = {k: 0.0 for k in k_values}  # Purchase + Checkout
            metrics['intent_mrr'] = 0.0
            
            # Create combined intent mask
            intent_mask = is_purchase | has_checkout
            
            checkout_count = has_checkout.sum().item()
            intent_count = intent_mask.sum().item()
            metrics['checkout_count'] = checkout_count
            metrics['intent_count'] = intent_count
        
        # Add add_to_cart metrics if provided
        if has_add_to_cart is not None:
            metrics['add_to_cart_recall@k'] = {k: 0.0 for k in k_values}
            metrics['add_to_cart_mrr'] = 0.0
            
            add_to_cart_count = has_add_to_cart.sum().item()
            metrics['add_to_cart_count'] = add_to_cart_count
        
        purchase_count = is_purchase.sum().item()
        metrics['purchase_count'] = purchase_count
        metrics['total_count'] = batch_size
        
        # Vectorized rank calculation
        expanded_targets = targets.unsqueeze(1).expand_as(top_k_indices)
        matches = (top_k_indices == expanded_targets)
        
        # Get position of first match for each sample 
        has_match = matches.any(dim=1)
        
        # Calculate ranks and reciprocal ranks
        ranks = torch.full((batch_size,), float('inf'), device=predictions.device)
        for i in range(batch_size):
            if has_match[i]:
                match_idx = matches[i].nonzero(as_tuple=True)[0][0]
                ranks[i] = match_idx + 1  # Add 1 because ranks start from 1
        
        # Calculate MRR for matches
        valid_ranks = ranks < float('inf')
        if valid_ranks.any():
            reciprocal_ranks = torch.zeros_like(ranks)
            reciprocal_ranks[valid_ranks] = 1.0 / ranks[valid_ranks]
            
            # Overall MRR
            metrics['overall_mrr'] = reciprocal_ranks.mean().item()
            
            # Purchase MRR
            if purchase_count > 0:
                purchase_mask = is_purchase & valid_ranks
                if purchase_mask.any():
                    metrics['purchase_mrr'] = reciprocal_ranks[purchase_mask].mean().item()
            
            # Checkout MRR (if provided)
            if has_checkout is not None and checkout_count > 0:
                checkout_mask = has_checkout & valid_ranks
                if checkout_mask.any():
                    metrics['checkout_mrr'] = reciprocal_ranks[checkout_mask].mean().item()
                
                # Intent MRR (Purchase + Checkout)
                intent_mask = (is_purchase | has_checkout) & valid_ranks
                if intent_mask.any():
                    metrics['intent_mrr'] = reciprocal_ranks[intent_mask].mean().item()
            
            # AddToCart MRR (if provided)
            if has_add_to_cart is not None and add_to_cart_count > 0:
                add_to_cart_mask = has_add_to_cart & valid_ranks
                if add_to_cart_mask.any():
                    metrics['add_to_cart_mrr'] = reciprocal_ranks[add_to_cart_mask].mean().item()
        
        # Calculate recall@k for all event types
        for k in k_values:
            # Overall recall
            in_top_k = matches[:, :k].any(dim=1)
            metrics['recall@k'][k] = in_top_k.float().mean().item()
            
            # Purchase recall
            if purchase_count > 0:
                purchase_in_top_k = in_top_k & is_purchase
                metrics['purchase_recall@k'][k] = purchase_in_top_k.sum().item() / purchase_count
            
            # Checkout recall (if provided)
            if has_checkout is not None and checkout_count > 0:
                checkout_in_top_k = in_top_k & has_checkout
                metrics['checkout_recall@k'][k] = checkout_in_top_k.sum().item() / checkout_count
                
                # Intent recall (Purchase + Checkout)
                intent_in_top_k = in_top_k & intent_mask
                metrics['intent_recall@k'][k] = intent_in_top_k.sum().item() / intent_count
            
            # AddToCart recall (if provided)
            if has_add_to_cart is not None and add_to_cart_count > 0:
                add_to_cart_in_top_k = in_top_k & has_add_to_cart
                metrics['add_to_cart_recall@k'][k] = add_to_cart_in_top_k.sum().item() / add_to_cart_count
        
        return metrics
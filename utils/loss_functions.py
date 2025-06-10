import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class NaturalPurchaseLoss(nn.Module):
    """
    Hierarchical loss that weights events by intent level:
    - View: 0 (no signal)
    - Add-to-cart: 0.1 (weak signal)
    - Checkout: 0.5 (moderate signal)
    - Purchase: 5.0 (strong signal)
    """
    def __init__(self, purchase_boost=5.0):
        super().__init__()
        self.purchase_weight = purchase_boost
        self.checkout_weight = 0.5
        self.cart_weight = 0.1
        self.view_weight = 0.05  # Small positive weight to ensure views contribute to learning
        
    def forward(self, predictions, targets, is_purchase, has_checkout=None, has_add_to_cart=None):
        """
        Cross-entropy with hierarchical event weighting
        
        Args:
            predictions: Model predictions [batch_size, num_items]
            targets: Target item indices [batch_size]
            is_purchase: Boolean tensor indicating purchases [batch_size]
            has_checkout: Boolean tensor indicating checkout events [batch_size]
            has_add_to_cart: Boolean tensor indicating add-to-cart events [batch_size]
        """
        # Ensure targets has the right shape for cross_entropy (1D)
        if len(targets.shape) > 1:
            targets = targets.squeeze()
            
        # Handle zero-sized tensors
        if targets.numel() == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
            
        # Basic cross-entropy loss
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # Initialize weights with view weight (0.0)
        weights = torch.ones_like(ce_loss) * self.view_weight
        
        # Apply hierarchical weights - order matters! Higher intent overrides lower
        if has_add_to_cart is not None:
            # Cart but not checkout/purchase
            cart_only = has_add_to_cart & ~has_checkout & ~is_purchase
            weights[cart_only] = self.cart_weight
            
        if has_checkout is not None:
            # Checkout but not purchase
            checkout_only = has_checkout & ~is_purchase
            weights[checkout_only] = self.checkout_weight
            
        # Purchase gets highest weight (overrides all)
        if is_purchase.any():
            weights[is_purchase] = self.purchase_weight
        
        # Filter out zero-weight samples for efficiency
        non_zero_mask = weights > 0
        if non_zero_mask.sum() == 0:
            # No samples with positive weight
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        # Calculate weighted mean only for non-zero weights
        weighted_loss = (ce_loss[non_zero_mask] * weights[non_zero_mask]).sum()
        total_weight = weights[non_zero_mask].sum()
        loss_val = weighted_loss / total_weight if total_weight > 0 else weighted_loss
        
        # Check for NaN or Inf values
        if torch.isnan(loss_val) or torch.isinf(loss_val):
            print(f"Warning: NaN or Inf loss detected in NaturalPurchaseLoss: {loss_val.item()}")
            return torch.tensor(0.1, device=predictions.device, requires_grad=True)
            
        return loss_val


class HierarchicalPurchaseLoss(nn.Module):
    """
    Hierarchical loss that treats different user actions with appropriate weights:
    - Purchase: Strong positive signal (1.0)
    - Checkout: Weak positive signal (0.1) - shows intent
    - Add-to-cart only: Negative signal (-0.1) - abandoned cart
    - View only: Very weak positive (0.01) - minimal interest
    """
    def __init__(self):
        super().__init__()
        self.purchase_weight = 1.0
        self.checkout_weight = 0.1  
        self.cart_abandon_weight = -0.1
        self.view_weight = 0.05  # Small positive weight to ensure views contribute to learning1
        
    def forward(self, predictions, targets, is_purchase, has_checkout=None, has_add_to_cart=None):
        """
        Hierarchical loss with event-specific weights
        
        Args:
            predictions: Model predictions [batch_size, num_items]
            targets: Target item indices [batch_size]
            is_purchase: Boolean tensor indicating purchases [batch_size]
            has_checkout: Boolean tensor indicating checkout events [batch_size]
            has_add_to_cart: Boolean tensor indicating add-to-cart events [batch_size]
        """
        # Ensure targets has the right shape for cross_entropy (1D)
        if len(targets.shape) > 1:
            targets = targets.squeeze()
            
        # Handle zero-sized tensors
        if targets.numel() == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
            
        # Basic cross-entropy loss
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # Initialize weights with view weight (default)
        weights = torch.ones_like(ce_loss) * self.view_weight
        
        # Apply hierarchical weights based on event types
        if has_checkout is not None and has_add_to_cart is not None:
            # Cart abandonment: add-to-cart but no checkout
            cart_abandon_mask = has_add_to_cart & ~has_checkout & ~is_purchase
            weights[cart_abandon_mask] = self.cart_abandon_weight
            
            # Checkout (but not purchase)
            checkout_only_mask = has_checkout & ~is_purchase
            weights[checkout_only_mask] = self.checkout_weight
        
        # Purchases get highest weight (overrides all others)
        if is_purchase.any():
            weights[is_purchase] = self.purchase_weight
        
        # For negative weights, we want to maximize the loss (push predictions down)
        # So we use absolute value for weighting but keep the sign for direction
        weighted_loss = ce_loss * weights.abs()
        
        # Apply sign to encourage/discourage predictions
        # Negative weights mean we want higher loss (discourage prediction)
        final_loss = weighted_loss * weights.sign()
        
        # Calculate mean
        loss_val = final_loss.mean()
        
        # Check for NaN or Inf values
        if torch.isnan(loss_val) or torch.isinf(loss_val):
            print(f"Warning: NaN or Inf loss detected in HierarchicalPurchaseLoss: {loss_val.item()}")
            return torch.tensor(0.1, device=predictions.device, requires_grad=True)
            
        return loss_val


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
        # Ensure targets has the right shape
        if len(targets.shape) > 1:
            targets = targets.squeeze()
            
        # Handle empty tensor case
        if targets.numel() == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
            
        # Get the softmax probs first
        probs = F.softmax(predictions, dim=1)
        
        # Get the probability of the correct class for each sample (pt in the paper)
        batch_size = predictions.size(0)
        pt = torch.zeros(batch_size, device=predictions.device)
        for i in range(batch_size):
            if i < len(targets):  # Ensure index is in range
                target_idx = targets[i].item()
                if 0 <= target_idx < predictions.size(1):  # Check target is valid
                    pt[i] = probs[i, target_idx]
        
        # Adjust gamma based on whether it's a purchase or not (higher gamma for purchases)
        if self.adaptive_gamma:
            gamma = torch.ones_like(pt) * self.base_gamma
            # Increase gamma for purchases to focus more on hard purchase examples
            if is_purchase.any():
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
        if is_purchase.any():
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
            
        # Calculate weighted mean - normalize by sum of weights for more stable gradients
        loss_val = final_loss.mean()
        
        # Check for NaN or Inf values
        if torch.isnan(loss_val) or torch.isinf(loss_val):
            print(f"Warning: NaN or Inf loss detected in FocalLoss: {loss_val.item()}")
            # Return a small but non-zero loss value that can be backpropagated
            return torch.tensor(0.1, device=predictions.device, requires_grad=True)
            
        return loss_val


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
                 add_to_cart_margin=0.6, purchase_weight=1.0):
        super().__init__()
        self.temperature = temperature  # Controls sensitivity to similarity differences
        self.purchase_margin = purchase_margin  # No margin for purchases
        self.checkout_margin = checkout_margin  # Small margin for checkout events
        self.view_margin = view_margin  # Full margin for view events
        self.add_to_cart_margin = add_to_cart_margin  # Medium margin for add to cart events
        self.hard_negative_mining = hard_negative_mining  # Whether to use hard negative mining
        self.hard_negative_ratio = hard_negative_ratio  # Proportion of hard negatives to keep
        self.purchase_weight = purchase_weight  # Weight multiplier for purchase events
    
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
        # Ensure targets has the right shape
        if len(targets.shape) > 1:
            targets = targets.squeeze()
            
        # Handle empty tensor case
        if targets.numel() == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
            
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
            # Handle multi-dimensional checkout tensor (squeeze to 1D if needed)
            if len(has_checkout.shape) > 1:
                # If has_checkout is 2D or more, take the first element or squeeze
                if has_checkout.shape[1] == 1:
                    has_checkout = has_checkout.squeeze(1)
                else:
                    # Take the first column as the indicator
                    has_checkout = has_checkout[:, 0]
                
            # Only count as checkout if NOT also a purchase
            checkout_mask = has_checkout.float() * (~is_purchase).float()
        
        # Only use add_to_cart info if provided
        if has_add_to_cart is not None:
            # Handle multi-dimensional add_to_cart tensor (squeeze to 1D if needed)
            if len(has_add_to_cart.shape) > 1:
                if has_add_to_cart.shape[1] == 1:
                    has_add_to_cart = has_add_to_cart.squeeze(1)
                else:
                    # Take the first column as the indicator
                    has_add_to_cart = has_add_to_cart[:, 0]
                    
            # Only count as add_to_cart if NOT also a purchase or checkout
            if has_checkout is not None:
                add_to_cart_mask = has_add_to_cart.float() * (~is_purchase).float() * (~has_checkout).float()
            else:
                add_to_cart_mask = has_add_to_cart.float() * (~is_purchase).float()
        
        # Regular browsing is everything else
        view_mask = 1.0 - purchase_mask - checkout_mask - add_to_cart_mask
        
        # Apply different contrastive margins based on event type
        # The margin is subtracted from the logits, making it harder to predict
        # correctly for that class. Higher margin = more penalty.
        margins = torch.zeros_like(logits)
        
        batch_size = predictions.size(0)
        for i in range(batch_size):
            if i < len(targets):  # Make sure index is valid
                target_idx = targets[i].item()
                if target_idx >= 0 and target_idx < predictions.size(1):  # Valid target
                    # Set margin for the target item based on event type
                    # Handle potential tensor shape issues - make sure we're getting a scalar value
                    is_purchase_val = purchase_mask[i].item() if isinstance(purchase_mask[i], torch.Tensor) and purchase_mask[i].numel() == 1 else purchase_mask[i]
                    
                    if is_purchase_val > 0:
                        # Purchase event - no/minimal margin (easiest to learn)
                        margins[i, target_idx] = self.purchase_margin
                    else:
                        # Check checkout - ensure it's a scalar
                        is_checkout_val = False
                        if i < len(checkout_mask):
                            if isinstance(checkout_mask[i], torch.Tensor) and checkout_mask[i].numel() == 1:
                                is_checkout_val = checkout_mask[i].item() > 0
                            else:
                                is_checkout_val = checkout_mask[i] > 0
                                
                        # Check add_to_cart - ensure it's a scalar
                        is_add_to_cart_val = False
                        if i < len(add_to_cart_mask):
                            if isinstance(add_to_cart_mask[i], torch.Tensor) and add_to_cart_mask[i].numel() == 1:
                                is_add_to_cart_val = add_to_cart_mask[i].item() > 0
                            else:
                                is_add_to_cart_val = add_to_cart_mask[i] > 0
                        
                        # Apply the appropriate margin based on the event type
                        if is_checkout_val:
                            # Checkout event - small margin
                            margins[i, target_idx] = self.checkout_margin
                        elif is_add_to_cart_val:
                            # Add to cart event - medium margin
                            margins[i, target_idx] = self.add_to_cart_margin
                        else:
                            # View event - largest margin (hardest to learn)
                            margins[i, target_idx] = self.view_margin
        
        # Apply margins to logits
        margin_logits = logits - margins
        
        # Calculate cross entropy loss with temperature scaling
        try:
            ce_loss = nn.functional.cross_entropy(margin_logits, targets, reduction='none')
        except Exception as e:
            print(f"Error in cross_entropy: {e}")
            print(f"margin_logits shape: {margin_logits.shape}, targets shape: {targets.shape}")
            print(f"targets min: {targets.min().item() if targets.numel() > 0 else 'empty'}, max: {targets.max().item() if targets.numel() > 0 else 'empty'}")
            # Return a zero loss as fallback
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        # Apply hard negative mining if enabled
        if self.hard_negative_mining:
            # Sort losses in descending order, but only for view samples
            view_losses = ce_loss * view_mask
            
            # Create a mask for samples where view_mask > 0
            view_mask_bool = view_mask > 0
            view_samples_count = view_mask_bool.sum().item()
            
            if view_samples_count > 10:  # Only if we have enough view samples
                # Get view losses for samples with view_mask > 0
                view_losses_filtered = view_losses[view_mask_bool]
                
                # Sort filtered losses
                sorted_losses, _ = torch.sort(view_losses_filtered, descending=True)
                
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
        
        # Apply purchase weighting to nudge training toward purchase focus
        purchase_weights = torch.ones_like(ce_loss)
        if is_purchase.any():
            purchase_weights[is_purchase] = self.purchase_weight
        
        # Apply mining mask, valid mask, and purchase weights
        masked_loss = ce_loss * mining_mask * valid_mask * purchase_weights
        
        # Return weighted mean loss - with safety checks for NaN/Inf values
        if valid_mask.sum() > 0:
            # Calculate weighted loss
            total_weights = (mining_mask * valid_mask * purchase_weights).sum()
            loss_val = masked_loss.sum() / (total_weights + 1e-6)
            
            # Check for NaN or Inf values
            if torch.isnan(loss_val) or torch.isinf(loss_val):
                print(f"Warning: NaN or Inf loss detected: {loss_val.item()}")
                # Return a small but non-zero loss value that can be backpropagated
                return torch.tensor(0.1, device=predictions.device, requires_grad=True)
            
            return loss_val
        else:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
    
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
        # Ensure targets has the right shape for cross_entropy (1D)
        if len(targets.shape) > 1:
            targets = targets.squeeze()
            
        # Handle zero-sized tensors - ensure we have at least one valid target
        if targets.numel() == 0:
            # Return a zero loss if no targets
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        # Handle multi-dimensional checkout tensor (squeeze to 1D if needed)
        if has_checkout is not None and len(has_checkout.shape) > 1:
            if has_checkout.shape[1] == 1:
                has_checkout = has_checkout.squeeze(1)
            else:
                # Take the first column as the indicator
                has_checkout = has_checkout[:, 0]
                
        # Handle multi-dimensional add_to_cart tensor (squeeze to 1D if needed)
        if has_add_to_cart is not None and len(has_add_to_cart.shape) > 1:
            if has_add_to_cart.shape[1] == 1:
                has_add_to_cart = has_add_to_cart.squeeze(1)
            else:
                # Take the first column as the indicator
                has_add_to_cart = has_add_to_cart[:, 0]
            
        # Basic cross-entropy loss
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # Initialize weights with base value
        weights = torch.ones_like(ce_loss)
        
        # Apply purchase boost (highest priority)
        if is_purchase.any():
            weights[is_purchase] = self.purchase_boost
        
        # Apply checkout boost (if provided and not a purchase)
        if has_checkout is not None and has_checkout.any():
            # Only apply checkout boost if it's not already a purchase
            checkout_only = has_checkout & (~is_purchase)
            if checkout_only.any():
                weights[checkout_only] = self.checkout_boost
        
        # Apply add_to_cart boost (if provided and not already boosted)
        if has_add_to_cart is not None and has_add_to_cart.any():
            # Only apply if not already a purchase or checkout
            if has_checkout is not None:
                add_to_cart_only = has_add_to_cart & (~is_purchase) & (~has_checkout)
            else:
                add_to_cart_only = has_add_to_cart & (~is_purchase)
            
            if add_to_cart_only.any():
                weights[add_to_cart_only] = self.add_to_cart_boost
        
        # Calculate weighted mean loss
        loss_val = (ce_loss * weights).mean()
        
        # Check for NaN or Inf values
        if torch.isnan(loss_val) or torch.isinf(loss_val):
            print(f"Warning: NaN or Inf loss detected in CheckoutEnhancedLoss: {loss_val.item()}")
            # Return a small but non-zero loss value that can be backpropagated
            return torch.tensor(0.1, device=predictions.device, requires_grad=True)
            
        return loss_val


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
            'overall_mrr': 0.0,
            'purchase_count': is_purchase.sum().item()
        }
        
        purchase_count = metrics['purchase_count']
        
        # Check for any unusual tensor shapes and fix if needed
        if len(targets.shape) > 1:
            # If targets has shape [batch_size, 1] or similar, flatten it
            targets = targets.view(-1)
        
        # Vectorized rank calculation
        # Ensure targets has the right shape before expanding
        targets_flat = targets.view(-1)
        expanded_targets = targets_flat.unsqueeze(1).expand(-1, top_k_indices.size(1))
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
        
        # Check for any unusual tensor shapes and fix if needed
        if len(targets.shape) > 1:
            # If targets has shape [batch_size, 1] or similar, flatten it
            targets = targets.view(-1)
        
        # Vectorized rank calculation
        # Ensure targets has the right shape before expanding
        targets_flat = targets.view(-1)
        expanded_targets = targets_flat.unsqueeze(1).expand(-1, top_k_indices.size(1))
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


class LogUniformSampledSoftmaxLoss(nn.Module):
    """
    Sampled softmax loss with log-uniform negative sampling.
    
    This implements the approach from the NATR paper and other large-scale recommenders
    where we sample a subset of negative items rather than computing scores for all items.
    
    Popular items (with lower IDs, assuming items are sorted by frequency) are sampled
    more often as negatives, which helps the model learn to distinguish from popular items.
    """
    def __init__(self, 
                 num_items: int, 
                 num_sampled: int = 100,
                 purchase_boost: float = 5.0,
                 unique_negatives: bool = True,
                 item_frequencies: Optional[torch.Tensor] = None,
                 temperature: float = 0.75,
                 dynamic_hard_negatives: bool = True,
                 hard_negative_start_epoch: int = 10):
        """
        Args:
            num_items: Total number of items in the catalog
            num_sampled: Number of negative items to sample per positive
            purchase_boost: Weight boost for purchase events
            unique_negatives: If True, ensure sampled negatives don't include the positive item
            item_frequencies: Optional tensor of item frequencies for proper log-uniform sampling
            temperature: Power law temperature for popularity-aware sampling (0-1, lower = more uniform)
            dynamic_hard_negatives: Enable dynamic hard negative sampling
            hard_negative_start_epoch: Epoch to start introducing hard negatives
        """
        super().__init__()
        self.num_items = num_items
        self.num_sampled = num_sampled
        self.purchase_boost = purchase_boost
        self.unique_negatives = unique_negatives
        self.temperature = temperature
        self.dynamic_hard_negatives = dynamic_hard_negatives
        self.hard_negative_start_epoch = hard_negative_start_epoch
        
        if item_frequencies is not None:
            # Use actual frequencies to compute log-uniform sampling probabilities
            # Add small epsilon to avoid zero frequencies
            item_frequencies = item_frequencies + 1e-6
            
            # Sort items by frequency and create mapping
            sorted_freqs, sorted_indices = torch.sort(item_frequencies, descending=True)
            
            # Create inverse mapping (item_id -> rank)
            self.item_to_rank = torch.zeros(num_items, dtype=torch.long)
            for rank, item_id in enumerate(sorted_indices):
                self.item_to_rank[item_id] = rank
            
            # Compute log-uniform probabilities based on rank
            # Popular items (low rank) have higher probability
            rank_probs = 1.0 / torch.log(torch.arange(2, num_items + 2, dtype=torch.float))
            
            # Create sampling probabilities for each item based on its rank
            item_probs = torch.zeros(num_items)
            for item_id in range(num_items):
                rank = self.item_to_rank[item_id]
                item_probs[item_id] = rank_probs[rank]
            
            # Store both original and temperature-adjusted probabilities
            self.register_buffer('sampling_probs', item_probs / item_probs.sum())
            
            # Apply temperature to adjust sampling distribution
            adjusted_freqs = torch.pow(item_frequencies, self.temperature)
            adjusted_probs = adjusted_freqs / adjusted_freqs.sum()
            self.register_buffer('adjusted_sampling_probs', adjusted_probs)
            
            # Print statistics
            popular_items = sorted_indices[:10]
            unpopular_items = sorted_indices[-10:]
            print(f"Top 10 popular items (frequencies): {sorted_freqs[:10].tolist()}")
            print(f"Bottom 10 items (frequencies): {sorted_freqs[-10:].tolist()}")
            
            # Find actual most/least popular items with non-zero frequency
            non_zero_mask = sorted_freqs > 0
            if non_zero_mask.any():
                most_popular_idx = sorted_indices[0].item()
                # Find least popular item with non-zero frequency
                non_zero_indices = sorted_indices[non_zero_mask]
                least_popular_idx = non_zero_indices[-1].item() if len(non_zero_indices) > 0 else sorted_indices[-1].item()
                
                print(f"\nMost popular item (ID={most_popular_idx}) sample prob: {self.sampling_probs[most_popular_idx]:.6f}")
                print(f"Least popular non-zero item (ID={least_popular_idx}) sample prob: {self.sampling_probs[least_popular_idx]:.6f}")
                print(f"Ratio: {self.sampling_probs[most_popular_idx] / self.sampling_probs[least_popular_idx]:.1f}x more likely to sample popular items")
        else:
            # Fallback to position-based (assumes items sorted by popularity)
            item_probs = 1.0 / torch.log(torch.arange(2, num_items + 2, dtype=torch.float))
            self.register_buffer('sampling_probs', item_probs / item_probs.sum())
            
            # Apply temperature to position-based probs
            adjusted_probs = torch.pow(item_probs, self.temperature)
            self.register_buffer('adjusted_sampling_probs', adjusted_probs / adjusted_probs.sum())
            print("Warning: Using position-based sampling (assumes items pre-sorted by popularity)")
        
        print(f"\nInitialized LogUniformSampledSoftmax with {num_items} items, sampling {num_sampled} negatives")
        print(f"Temperature: {self.temperature}, Dynamic hard negatives: {self.dynamic_hard_negatives}")
    
    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor,
                is_purchase: torch.Tensor,
                has_checkout: Optional[torch.Tensor] = None,
                has_add_to_cart: Optional[torch.Tensor] = None,
                epoch: int = 0,
                max_epochs: int = 50) -> torch.Tensor:
        """
        Compute sampled softmax loss.
        
        Args:
            predictions: Full predictions for all items [batch_size, num_items]
            targets: Target item indices [batch_size]
            is_purchase: Boolean tensor indicating purchases [batch_size]
            has_checkout: Optional (not used, for compatibility)
            has_add_to_cart: Optional (not used, for compatibility)
            
        Returns:
            Scalar loss value
        """
        batch_size = predictions.size(0)
        device = predictions.device
        
        # Ensure targets has the right shape
        if len(targets.shape) > 1:
            targets = targets.squeeze()
        
        # Handle zero-sized tensors
        if targets.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Dynamic hard negative sampling
        device = targets.device
        if self.dynamic_hard_negatives and epoch >= self.hard_negative_start_epoch:
            # Calculate ratio of hard negatives based on training progress
            progress = (epoch - self.hard_negative_start_epoch) / max(1, (max_epochs - self.hard_negative_start_epoch))
            hard_negative_ratio = min(progress * 0.5, 0.5)  # Cap at 50% hard negatives
            num_hard = int(self.num_sampled * hard_negative_ratio)
            num_regular = self.num_sampled - num_hard
            
            if num_hard > 0:
                # Get hard negatives: items with high predicted scores
                with torch.no_grad():
                    # Mask out the positive items
                    masked_predictions = predictions.clone()
                    masked_predictions[torch.arange(batch_size), targets] = -float('inf')
                    
                    # Get top-k predictions (these are the hardest negatives)
                    top_k_hard = min(500, predictions.size(1) // 2)  # Don't sample from more than half the catalog
                    _, hard_negative_candidates = torch.topk(masked_predictions, top_k_hard, dim=1)
                    
                    # Sample from the hard negatives
                    hard_neg_indices = torch.randint(0, top_k_hard, (batch_size, num_hard), device=device)
                    hard_neg_samples = torch.gather(hard_negative_candidates, 1, hard_neg_indices)
                
                # Sample remaining negatives using adjusted probability distribution
                if num_regular > 0:
                    regular_neg_samples = torch.multinomial(
                        self.adjusted_sampling_probs.to(device), 
                        batch_size * num_regular, 
                        replacement=True
                    ).view(batch_size, num_regular)
                    
                    # Combine hard and regular negatives
                    neg_samples = torch.cat([hard_neg_samples, regular_neg_samples], dim=1)
                else:
                    neg_samples = hard_neg_samples
            else:
                # Use regular sampling
                neg_samples = torch.multinomial(
                    self.adjusted_sampling_probs.to(device), 
                    batch_size * self.num_sampled, 
                    replacement=True
                ).view(batch_size, self.num_sampled)
        else:
            # Use regular adjusted probability sampling
            neg_samples = torch.multinomial(
                self.adjusted_sampling_probs.to(device), 
                batch_size * self.num_sampled, 
                replacement=True
            ).view(batch_size, self.num_sampled)
        
        # If unique_negatives, ensure we don't sample the positive item as negative
        if self.unique_negatives:
            # Create mask for samples that match the target
            target_expanded = targets.unsqueeze(1).expand(-1, self.num_sampled)
            matches = neg_samples == target_expanded
            
            # Resample matching items
            while matches.any():
                new_samples = torch.multinomial(
                    self.sampling_probs.to(device), 
                    matches.sum().item(), 
                    replacement=True
                )
                neg_samples[matches] = new_samples
                matches = neg_samples == target_expanded
        
        # Ensure targets are int64 for gather operation
        targets = targets.long()
        
        # Get positive scores (scores for the correct items)
        # Shape: [batch_size, 1]
        pos_scores = torch.gather(predictions, 1, targets.unsqueeze(1))
        
        # Get negative scores (scores for sampled negative items)
        # Shape: [batch_size, num_sampled]
        neg_scores = torch.gather(predictions, 1, neg_samples)
        
        # CRITICAL: Apply importance sampling correction for negative scores
        # Adjust negative scores based on their sampling probabilities
        neg_probs = torch.gather(self.sampling_probs.to(device).unsqueeze(0).expand(batch_size, -1), 1, neg_samples)
        # Correct for sampling bias: subtract log(q(neg)) where q is sampling probability
        corrected_neg_scores = neg_scores - torch.log(neg_probs * self.num_items + 1e-10)
        
        # Concatenate positive and corrected negative scores
        # Positive score is always at index 0
        # Shape: [batch_size, 1 + num_sampled]
        all_scores = torch.cat([pos_scores, corrected_neg_scores], dim=1)
        
        # Labels are always 0 (first position) since positive is at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Compute cross-entropy loss on the corrected sampled subset
        ce_loss = F.cross_entropy(all_scores, labels, reduction='none')
        
        # Apply hierarchical event weighting (like NaturalPurchaseLoss)
        weights = torch.ones_like(ce_loss) * 0.05  # Base weight for views
        
        # Apply hierarchical weights - order matters! Higher intent overrides lower
        if has_add_to_cart is not None:
            # Cart but not checkout/purchase
            cart_only = has_add_to_cart & ~has_checkout & ~is_purchase
            weights[cart_only] = 0.1
            
        if has_checkout is not None:
            # Checkout but not purchase
            checkout_only = has_checkout & ~is_purchase
            weights[checkout_only] = 0.5
            
        # Purchase gets highest weight (overrides all)
        if is_purchase.any():
            weights[is_purchase] = self.purchase_boost
        
        # Filter out zero-weight samples for efficiency
        non_zero_mask = weights > 0
        if non_zero_mask.sum() == 0:
            # No samples with positive weight
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Calculate weighted mean only for non-zero weights
        weighted_loss = (ce_loss[non_zero_mask] * weights[non_zero_mask]).sum()
        total_weight = weights[non_zero_mask].sum()
        loss = weighted_loss / total_weight if total_weight > 0 else weighted_loss
        
        # Add logging for debugging
        with torch.no_grad():
            # Calculate accuracy on sampled subset
            sampled_accuracy = (all_scores.argmax(dim=1) == 0).float().mean()
            
            # Calculate what would be the rank of positive item among sampled negatives
            ranks = (neg_scores > pos_scores).sum(dim=1).float() + 1
            mrr = (1.0 / ranks).mean()
            
            # Log every 100 batches
            if hasattr(self, '_batch_count'):
                self._batch_count += 1
            else:
                self._batch_count = 1
                
            if self._batch_count % 100 == 0:
                # Log basic metrics
                log_msg = f"Sampled softmax - Loss: {loss.item():.4f}, "
                log_msg += f"Sampled Acc: {sampled_accuracy.item():.3f}, "
                log_msg += f"Sampled MRR: {mrr.item():.3f}"
                
                # Add hard negative info if applicable
                if self.dynamic_hard_negatives and epoch >= self.hard_negative_start_epoch:
                    progress = (epoch - self.hard_negative_start_epoch) / max(1, (max_epochs - self.hard_negative_start_epoch))
                    hard_ratio = min(progress * 0.5, 0.5)
                    log_msg += f", Hard neg ratio: {hard_ratio:.2%}"
                
                print(log_msg)
        
        # Check for NaN or Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN or Inf loss detected: {loss.item()}")
            return torch.tensor(0.1, device=device, requires_grad=True)
        
        return loss


class SampledSoftmaxEvaluator:
    """
    Helper class to properly evaluate a model trained with sampled softmax.
    During evaluation, we still need to compute full predictions for accurate metrics.
    """
    @staticmethod
    def evaluate_batch(predictions: torch.Tensor, 
                      targets: torch.Tensor,
                      k_values: list = [10, 20]) -> dict:
        """
        Evaluate predictions using full softmax (all items).
        
        Args:
            predictions: Full predictions [batch_size, num_items]
            targets: Target indices [batch_size]
            k_values: List of k values for recall@k
            
        Returns:
            Dictionary of metrics
        """
        batch_size = predictions.size(0)
        
        # Get top-k predictions
        metrics = {}
        for k in k_values:
            _, top_k = torch.topk(predictions, k, dim=1)
            
            # Check if target is in top-k
            target_expanded = targets.unsqueeze(1).expand(-1, k)
            in_top_k = (top_k == target_expanded).any(dim=1).float()
            
            metrics[f'recall@{k}'] = in_top_k.mean().item()
        
        # Calculate MRR
        # Get ranks of all items
        _, indices = torch.sort(predictions, dim=1, descending=True)
        ranks = torch.zeros_like(targets, dtype=torch.float)
        
        for i in range(batch_size):
            rank_list = (indices[i] == targets[i]).nonzero(as_tuple=True)[0]
            if len(rank_list) > 0:
                ranks[i] = rank_list[0] + 1  # 1-indexed
            else:
                ranks[i] = predictions.size(1) + 1  # Beyond last position
        
        mrr = (1.0 / ranks).mean().item()
        metrics['mrr'] = mrr
        
        return metrics
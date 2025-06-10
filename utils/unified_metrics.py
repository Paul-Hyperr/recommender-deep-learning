"""
Unified metrics for evaluating recommender models across different training approaches.
Includes weighted recall metrics for purchase, checkout, and add-to-cart events.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import torch.nn.functional as F


class LearnableMargin(torch.nn.Module):
    """
    Learnable margin parameters for contrastive learning.
    Automatically adjusts margins during training through gradient descent.
    """
    
    def __init__(self, 
                 initial_margin: float,
                 min_margin: float = 0.0,
                 max_margin: float = 1.0):
        """
        Initialize learnable margin parameter.
        
        Args:
            initial_margin: Initial margin value
            min_margin: Minimum allowed margin value
            max_margin: Maximum allowed margin value
        """
        super().__init__()
        self.min_margin = min_margin
        self.max_margin = max_margin
        
        # Use log-space parameterization for better numeric stability
        self.margin_param = torch.nn.Parameter(
            torch.tensor(np.log(initial_margin - min_margin + 1e-6), 
                       dtype=torch.float)
        )
        
    def forward(self):
        """Get the current margin value with constraints applied"""
        # Convert from log-space and apply constraints
        margin = torch.exp(self.margin_param) + self.min_margin - 1e-6
        margin = torch.clamp(margin, self.min_margin, self.max_margin)
        return margin


class UnifiedMetricsTracker:
    """
    Unified metrics tracking for all training scripts with priority weighting.
    Handles all event types and calculates weighted composite metrics.
    
    Key features:
    - Unified approach to calculating metrics across all training methods
    - Event-specific metrics (purchase, checkout, add_to_cart)
    - Weighted composite metrics based on event importance
    - Comprehensive tracking and averaging over evaluation batches
    - Robust error handling for edge cases
    """
    
    def __init__(self, 
                 k_values: List[int] = [10, 20, 50],
                 event_weights: Optional[Dict[str, float]] = None,
                 validate_inputs: bool = True):
        """
        Initialize metrics tracker with specified k values and event weights.
        
        Args:
            k_values: List of k values for recall@k and other metrics
            event_weights: Dictionary of weights for each event type in composite metrics
                          Default: {'purchase': 1.0, 'checkout': 0.6, 'add_to_cart': 0.2, 'view': 0.05}
            validate_inputs: Whether to validate inputs for error handling
        """
        self.k_values = k_values
        self.validate_inputs = validate_inputs
        
        # Default event weights if not provided
        if event_weights is None:
            self.event_weights = {
                'purchase': 1.0,      # Full weight for purchases
                'checkout': 0.6,      # Consistent weight for checkout events
                'add_to_cart': 0.2,   # Consistent weight for add-to-cart
                'view': 0.05          # Very minimal weight for view events
            }
        else:
            self.event_weights = event_weights
            
        # Initialize the metrics storage
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics for a new evaluation session"""
        # Global hit counters for exact recall calculation
        self.hit_counts = {
            # Overall hits
            'total': {k: 0 for k in self.k_values},
            
            # Event-specific hits
            'purchase': {k: 0 for k in self.k_values},
            'checkout': {k: 0 for k in self.k_values},
            'add_to_cart': {k: 0 for k in self.k_values},
            'intent': {k: 0 for k in self.k_values},  # purchase + checkout
            
            # Cold start user hits
            'cold_start_purchase': {k: 0 for k in self.k_values},
            'cold_start_checkout': {k: 0 for k in self.k_values},
            
            # Warm user hits
            'warm_purchase': {k: 0 for k in self.k_values},
            'warm_checkout': {k: 0 for k in self.k_values},
            
            # Weighted hits for composite metrics
            'weighted': {k: 0.0 for k in self.k_values},
        }
        
        # Global sample counters
        self.sample_counts = {
            'total': 0,
            'purchase': 0,
            'checkout': 0,
            'add_to_cart': 0,
            'intent': 0,  # purchase + checkout
            'view': 0,    # regular browsing
            'cold_start_purchase': 0,
            'cold_start_checkout': 0,
            'warm_purchase': 0,
            'warm_checkout': 0
        }
        
        # MRR accumulators (sum of reciprocal ranks)
        self.mrr_sums = {
            'total': 0.0,
            'purchase': 0.0,
            'checkout': 0.0,
            'add_to_cart': 0.0,
            'intent': 0.0,
            'weighted': 0.0,
            'cold_start_purchase': 0.0,
            'cold_start_checkout': 0.0,
            'warm_purchase': 0.0,
            'warm_checkout': 0.0
        }
        
        # Track total weight sum for proper weighted averaging
        self.total_weight_sum = 0.0
        
        # Additional metrics for compatibility
        self.precision_sums = {k: 0.0 for k in self.k_values}
        self.ndcg_sums = {k: 0.0 for k in self.k_values}
        self.loss_sum = 0.0
        self.loss_count = 0
        
        # Track unique items for coverage metrics  
        self.all_ground_truth_items = set()  # Track all items that should be recommended
        self.correctly_recommended_items = {k: set() for k in self.k_values}  # Items correctly recommended in top-k
        self.recommended_items = {k: set() for k in self.k_values}  # ALL items recommended in top-k (for coverage)
    
    def update(self, 
               predictions: torch.Tensor, 
               targets: torch.Tensor,
               is_purchase: torch.Tensor,
               has_checkout: Optional[torch.Tensor] = None,
               has_add_to_cart: Optional[torch.Tensor] = None,
               has_checkout_inclusive: Optional[torch.Tensor] = None,
               has_add_to_cart_inclusive: Optional[torch.Tensor] = None,
               is_cold_start: Optional[torch.Tensor] = None,
               loss: Optional[float] = None):
        """
        Update metrics with a new batch of predictions and targets.
        
        Args:
            predictions: Model predictions [batch_size, num_items]
            targets: Ground truth target indices [batch_size]
            is_purchase: Boolean tensor indicating purchase samples [batch_size]
            has_checkout: Boolean tensor indicating checkout samples (hierarchical - for weighted recall)
            has_add_to_cart: Boolean tensor indicating add-to-cart samples (hierarchical - for weighted recall)
            has_checkout_inclusive: Boolean tensor indicating all checkout samples (for individual recalls)
            has_add_to_cart_inclusive: Boolean tensor indicating all add-to-cart samples (for individual recalls)
            is_cold_start: Boolean tensor indicating cold start users [batch_size]
            loss: Optional loss value to track
        """
        # Validate inputs if enabled
        if self.validate_inputs:
            # Handle problematic inputs with robust error handling
            try:
                # Ensure input tensors are on the same device
                if predictions.device != targets.device:
                    targets = targets.to(predictions.device)
                
                if is_purchase.device != predictions.device:
                    is_purchase = is_purchase.to(predictions.device)
                
                # Ensure has_checkout and has_add_to_cart are properly handled
                if has_checkout is None:
                    has_checkout = torch.zeros_like(is_purchase)
                elif has_checkout.device != predictions.device:
                    has_checkout = has_checkout.to(predictions.device)
                
                if has_add_to_cart is None:
                    has_add_to_cart = torch.zeros_like(is_purchase)
                elif has_add_to_cart.device != predictions.device:
                    has_add_to_cart = has_add_to_cart.to(predictions.device)
                
                # Handle inclusive flags
                if has_checkout_inclusive is None:
                    has_checkout_inclusive = torch.zeros_like(is_purchase)
                elif has_checkout_inclusive.device != predictions.device:
                    has_checkout_inclusive = has_checkout_inclusive.to(predictions.device)
                
                if has_add_to_cart_inclusive is None:
                    has_add_to_cart_inclusive = torch.zeros_like(is_purchase)
                elif has_add_to_cart_inclusive.device != predictions.device:
                    has_add_to_cart_inclusive = has_add_to_cart_inclusive.to(predictions.device)
                
                # Ensure tensors have compatible shapes
                if len(targets.shape) > 1:
                    targets = targets.squeeze()
                
                if len(is_purchase.shape) > 1:
                    is_purchase = is_purchase.squeeze()
                    
                if len(has_checkout.shape) > 1:
                    has_checkout = has_checkout.squeeze()
                    
                if len(has_add_to_cart.shape) > 1:
                    has_add_to_cart = has_add_to_cart.squeeze()
                    
                if len(has_checkout_inclusive.shape) > 1:
                    has_checkout_inclusive = has_checkout_inclusive.squeeze()
                    
                if len(has_add_to_cart_inclusive.shape) > 1:
                    has_add_to_cart_inclusive = has_add_to_cart_inclusive.squeeze()
                
                # Handle is_cold_start flag
                if is_cold_start is None:
                    is_cold_start = torch.zeros_like(is_purchase)
                elif is_cold_start.device != predictions.device:
                    is_cold_start = is_cold_start.to(predictions.device)
                    
                if len(is_cold_start.shape) > 1:
                    is_cold_start = is_cold_start.squeeze()
                    
                # Handle empty input case
                if targets.numel() == 0:
                    print("Warning: Empty targets tensor, skipping metrics update")
                    return
                    
                # Handle case where targets are out of bounds
                if targets.max() >= predictions.size(1):
                    print(f"Warning: Target index {targets.max().item()} out of bounds " 
                          f"(num_items={predictions.size(1)}), clipping")
                    targets = torch.clamp(targets, 0, predictions.size(1) - 1)
                
            except Exception as e:
                print(f"Error validating inputs: {e}")
                return
        
        # Update sample counts
        batch_size = predictions.size(0)
        self.sample_counts['total'] += batch_size
        self.sample_counts['purchase'] += is_purchase.sum().item()
        
        # Calculate intent mask (purchase OR checkout)  
        intent_mask = is_purchase | has_checkout
        self.sample_counts['intent'] += intent_mask.sum().item()
        
        # Update checkout and add_to_cart counts using inclusive flags (for individual recalls)
        if has_checkout_inclusive is not None:
            self.sample_counts['checkout'] += has_checkout_inclusive.sum().item()
        
        if has_add_to_cart_inclusive is not None:
            self.sample_counts['add_to_cart'] += has_add_to_cart_inclusive.sum().item()
        
        # Calculate view count (none of the above)
        view_mask = ~(is_purchase | has_checkout | has_add_to_cart)
        self.sample_counts['view'] += view_mask.sum().item()
        
        # Update cold start vs warm user counts
        if is_cold_start is not None:
            cold_start_purchases = is_purchase & is_cold_start
            warm_purchases = is_purchase & ~is_cold_start
            cold_start_checkouts = has_checkout_inclusive & is_cold_start if has_checkout_inclusive is not None else torch.zeros_like(is_cold_start, dtype=torch.bool)
            warm_checkouts = has_checkout_inclusive & ~is_cold_start if has_checkout_inclusive is not None else torch.zeros_like(is_cold_start, dtype=torch.bool)
            
            self.sample_counts['cold_start_purchase'] += cold_start_purchases.sum().item()
            self.sample_counts['warm_purchase'] += warm_purchases.sum().item()
            self.sample_counts['cold_start_checkout'] += cold_start_checkouts.sum().item()
            self.sample_counts['warm_checkout'] += warm_checkouts.sum().item()
        
        # Calculate total weight sum for this batch (for weighted metrics)
        batch_weights = torch.zeros(batch_size, device=predictions.device)
        batch_weights[is_purchase] = self.event_weights['purchase']
        batch_weights[has_checkout & ~is_purchase] = self.event_weights['checkout']
        batch_weights[has_add_to_cart & ~has_checkout & ~is_purchase] = self.event_weights['add_to_cart']
        batch_weights[view_mask] = self.event_weights['view']
        self.total_weight_sum += batch_weights.sum().item()
        
        # Track loss if provided
        if loss is not None:
            self.loss_sum += loss
            self.loss_count += 1
        
        # Store number of items for coverage calculation
        self._num_items = predictions.size(1)
        
        # Get top-k predictions for all k values
        max_k = max(self.k_values)
        _, top_k_indices = torch.topk(predictions, max_k, dim=1)
        
        # Calculate matches for each sample
        targets_expanded = targets.unsqueeze(1).expand(-1, max_k)
        matches = (top_k_indices == targets_expanded)  # [batch_size, max_k]
        
        # Calculate ranks (position of first match + 1, or inf if no match)
        first_match_positions = torch.full((batch_size,), float('inf'), device=predictions.device)
        has_match = matches.any(dim=1)
        if has_match.any():
            # Get position of first match for each sample that has a match
            first_match_pos = matches.int().argmax(dim=1)  # Position of first True
            # Only set positions for samples that actually have matches
            first_match_positions[has_match] = first_match_pos[has_match].float() + 1
        
        # Convert to CPU for counting
        ranks = first_match_positions.cpu()
        has_match_cpu = has_match.cpu()
        is_purchase_cpu = is_purchase.cpu()
        intent_mask_cpu = intent_mask.cpu()
        
        if has_checkout_inclusive is not None:
            has_checkout_inclusive_cpu = has_checkout_inclusive.cpu()
        else:
            has_checkout_inclusive_cpu = torch.zeros_like(is_purchase_cpu)
            
        if has_add_to_cart_inclusive is not None:
            has_add_to_cart_inclusive_cpu = has_add_to_cart_inclusive.cpu()
        else:
            has_add_to_cart_inclusive_cpu = torch.zeros_like(is_purchase_cpu)
        
        # Handle cold start flag on CPU
        if is_cold_start is not None:
            is_cold_start_cpu = is_cold_start.cpu()
        else:
            is_cold_start_cpu = torch.zeros_like(is_purchase_cpu)
        
        # Update global hit counts and MRR sums for each k
        for k in self.k_values:
            # Samples that hit in top-k
            in_top_k = has_match_cpu & (ranks <= k)
            
            # Overall hits
            self.hit_counts['total'][k] += in_top_k.sum().item()
            
            # Event-specific hits
            purchase_hits = in_top_k & is_purchase_cpu
            self.hit_counts['purchase'][k] += purchase_hits.sum().item()
            
            checkout_hits = in_top_k & has_checkout_inclusive_cpu
            self.hit_counts['checkout'][k] += checkout_hits.sum().item()
            
            add_to_cart_hits = in_top_k & has_add_to_cart_inclusive_cpu
            self.hit_counts['add_to_cart'][k] += add_to_cart_hits.sum().item()
            
            intent_hits = in_top_k & intent_mask_cpu
            self.hit_counts['intent'][k] += intent_hits.sum().item()
            
            # Cold start vs warm user hits
            if is_cold_start is not None:
                cold_start_purchase_hits = in_top_k & is_purchase_cpu & is_cold_start_cpu
                warm_purchase_hits = in_top_k & is_purchase_cpu & ~is_cold_start_cpu
                cold_start_checkout_hits = in_top_k & has_checkout_inclusive_cpu & is_cold_start_cpu
                warm_checkout_hits = in_top_k & has_checkout_inclusive_cpu & ~is_cold_start_cpu
                
                self.hit_counts['cold_start_purchase'][k] += cold_start_purchase_hits.sum().item()
                self.hit_counts['warm_purchase'][k] += warm_purchase_hits.sum().item()
                self.hit_counts['cold_start_checkout'][k] += cold_start_checkout_hits.sum().item()
                self.hit_counts['warm_checkout'][k] += warm_checkout_hits.sum().item()
            
            # Weighted hits for composite metrics
            for i in range(batch_size):
                if in_top_k[i]:
                    weight = 0.0
                    if is_purchase_cpu[i]:
                        weight = self.event_weights['purchase']
                    elif has_checkout is not None and has_checkout[i]:
                        weight = self.event_weights['checkout']
                    elif has_add_to_cart is not None and has_add_to_cart[i]:
                        weight = self.event_weights['add_to_cart']
                    else:
                        weight = self.event_weights['view']
                    self.hit_counts['weighted'][k] += weight
        
        # Update MRR sums (only for samples with matches)
        valid_ranks = ranks[has_match_cpu & (ranks < float('inf'))]
        if len(valid_ranks) > 0:
            reciprocal_ranks = 1.0 / valid_ranks
            
            # Event-specific MRR
            purchase_matches = has_match_cpu & is_purchase_cpu
            if purchase_matches.any():
                purchase_ranks = ranks[purchase_matches]
                purchase_valid = purchase_ranks < float('inf')
                if purchase_valid.any():
                    self.mrr_sums['purchase'] += (1.0 / purchase_ranks[purchase_valid]).sum().item()
            
            checkout_matches = has_match_cpu & has_checkout_inclusive_cpu
            if checkout_matches.any():
                checkout_ranks = ranks[checkout_matches]
                checkout_valid = checkout_ranks < float('inf')
                if checkout_valid.any():
                    self.mrr_sums['checkout'] += (1.0 / checkout_ranks[checkout_valid]).sum().item()
            
            add_to_cart_matches = has_match_cpu & has_add_to_cart_inclusive_cpu
            if add_to_cart_matches.any():
                add_to_cart_ranks = ranks[add_to_cart_matches]
                add_to_cart_valid = add_to_cart_ranks < float('inf')
                if add_to_cart_valid.any():
                    self.mrr_sums['add_to_cart'] += (1.0 / add_to_cart_ranks[add_to_cart_valid]).sum().item()
            
            intent_matches = has_match_cpu & intent_mask_cpu
            if intent_matches.any():
                intent_ranks = ranks[intent_matches]
                intent_valid = intent_ranks < float('inf')
                if intent_valid.any():
                    self.mrr_sums['intent'] += (1.0 / intent_ranks[intent_valid]).sum().item()
            
            # Cold start vs warm user MRR
            if is_cold_start is not None:
                cold_start_purchase_matches = has_match_cpu & is_purchase_cpu & is_cold_start_cpu
                if cold_start_purchase_matches.any():
                    cold_start_purchase_ranks = ranks[cold_start_purchase_matches]
                    cold_start_purchase_valid = cold_start_purchase_ranks < float('inf')
                    if cold_start_purchase_valid.any():
                        self.mrr_sums['cold_start_purchase'] += (1.0 / cold_start_purchase_ranks[cold_start_purchase_valid]).sum().item()
                
                warm_purchase_matches = has_match_cpu & is_purchase_cpu & ~is_cold_start_cpu
                if warm_purchase_matches.any():
                    warm_purchase_ranks = ranks[warm_purchase_matches]
                    warm_purchase_valid = warm_purchase_ranks < float('inf')
                    if warm_purchase_valid.any():
                        self.mrr_sums['warm_purchase'] += (1.0 / warm_purchase_ranks[warm_purchase_valid]).sum().item()
                
                cold_start_checkout_matches = has_match_cpu & has_checkout_inclusive_cpu & is_cold_start_cpu
                if cold_start_checkout_matches.any():
                    cold_start_checkout_ranks = ranks[cold_start_checkout_matches]
                    cold_start_checkout_valid = cold_start_checkout_ranks < float('inf')
                    if cold_start_checkout_valid.any():
                        self.mrr_sums['cold_start_checkout'] += (1.0 / cold_start_checkout_ranks[cold_start_checkout_valid]).sum().item()
                
                warm_checkout_matches = has_match_cpu & has_checkout_inclusive_cpu & ~is_cold_start_cpu
                if warm_checkout_matches.any():
                    warm_checkout_ranks = ranks[warm_checkout_matches]
                    warm_checkout_valid = warm_checkout_ranks < float('inf')
                    if warm_checkout_valid.any():
                        self.mrr_sums['warm_checkout'] += (1.0 / warm_checkout_ranks[warm_checkout_valid]).sum().item()
            
            # Weighted MRR
            for i in range(batch_size):
                if has_match_cpu[i] and ranks[i] < float('inf'):
                    weight = 0.0
                    if is_purchase_cpu[i]:
                        weight = self.event_weights['purchase']
                    elif has_checkout is not None and has_checkout[i]:
                        weight = self.event_weights['checkout']
                    elif has_add_to_cart is not None and has_add_to_cart[i]:
                        weight = self.event_weights['add_to_cart']
                    else:
                        weight = self.event_weights['view']
                    self.mrr_sums['weighted'] += weight * (1.0 / ranks[i].item())
        
        # Track items for coverage metrics
        ground_truth_items = targets.cpu().numpy().tolist()
        self.all_ground_truth_items.update(ground_truth_items)
        
        for k in self.k_values:
            # Track ALL recommended items in top-k (for coverage calculation)
            top_k_items = top_k_indices[:, :k].cpu().numpy().flatten().tolist()
            self.recommended_items[k].update(top_k_items)
            
            # Track correctly recommended items in top-k
            in_top_k = has_match_cpu & (ranks <= k)
            if in_top_k.any():
                correct_items = targets[in_top_k].cpu().numpy().tolist()
                self.correctly_recommended_items[k].update(correct_items)
    
    # DEPRECATED: Removed _calculate_batch_metrics method to avoid confusion
    # Now using global counters for exact metrics calculation
    
    def compute(self) -> Dict[str, Any]:
        """
        Compute final metrics using exact global counts.
        
        Returns:
            Dictionary of all metrics calculated from global hit/sample counts
        """
        results = {}
        
        # Loss (if tracked)
        if self.loss_count > 0:
            results['loss'] = self.loss_sum / self.loss_count
        
        # Recall@k metrics using exact global counts
        for k in self.k_values:
            # Event-specific recall = event_hits_at_k / event_samples
            if self.sample_counts['purchase'] > 0:
                results[f'purchase_recall@{k}'] = self.hit_counts['purchase'][k] / self.sample_counts['purchase']
            else:
                results[f'purchase_recall@{k}'] = 0.0
                
            if self.sample_counts['checkout'] > 0:
                results[f'checkout_recall@{k}'] = self.hit_counts['checkout'][k] / self.sample_counts['checkout']
            else:
                results[f'checkout_recall@{k}'] = 0.0
            
            # Cold start vs warm user recall
            if self.sample_counts['cold_start_purchase'] > 0:
                results[f'cold_start_purchase_recall@{k}'] = self.hit_counts['cold_start_purchase'][k] / self.sample_counts['cold_start_purchase']
            else:
                results[f'cold_start_purchase_recall@{k}'] = 0.0
                
            if self.sample_counts['warm_purchase'] > 0:
                results[f'warm_purchase_recall@{k}'] = self.hit_counts['warm_purchase'][k] / self.sample_counts['warm_purchase']
            else:
                results[f'warm_purchase_recall@{k}'] = 0.0
                
            if self.sample_counts['cold_start_checkout'] > 0:
                results[f'cold_start_checkout_recall@{k}'] = self.hit_counts['cold_start_checkout'][k] / self.sample_counts['cold_start_checkout']
            else:
                results[f'cold_start_checkout_recall@{k}'] = 0.0
                
            if self.sample_counts['warm_checkout'] > 0:
                results[f'warm_checkout_recall@{k}'] = self.hit_counts['warm_checkout'][k] / self.sample_counts['warm_checkout']
            else:
                results[f'warm_checkout_recall@{k}'] = 0.0
            
            # Item coverage (unique items correctly recommended in top-k)
            if self.all_ground_truth_items:
                results[f'item_coverage@{k}'] = len(self.correctly_recommended_items[k]) / len(self.all_ground_truth_items)
            else:
                results[f'item_coverage@{k}'] = 0.0
        
        # Event-specific MRR
        if self.sample_counts['purchase'] > 0:
            results['purchase_mrr'] = self.mrr_sums['purchase'] / self.sample_counts['purchase']
        else:
            results['purchase_mrr'] = 0.0
            
        if self.sample_counts['checkout'] > 0:
            results['checkout_mrr'] = self.mrr_sums['checkout'] / self.sample_counts['checkout']
        else:
            results['checkout_mrr'] = 0.0
        
        # Cold start vs warm user MRR
        if self.sample_counts['cold_start_purchase'] > 0:
            results['cold_start_purchase_mrr'] = self.mrr_sums['cold_start_purchase'] / self.sample_counts['cold_start_purchase']
        else:
            results['cold_start_purchase_mrr'] = 0.0
            
        if self.sample_counts['warm_purchase'] > 0:
            results['warm_purchase_mrr'] = self.mrr_sums['warm_purchase'] / self.sample_counts['warm_purchase']
        else:
            results['warm_purchase_mrr'] = 0.0
            
        if self.sample_counts['cold_start_checkout'] > 0:
            results['cold_start_checkout_mrr'] = self.mrr_sums['cold_start_checkout'] / self.sample_counts['cold_start_checkout']
        else:
            results['cold_start_checkout_mrr'] = 0.0
            
        if self.sample_counts['warm_checkout'] > 0:
            results['warm_checkout_mrr'] = self.mrr_sums['warm_checkout'] / self.sample_counts['warm_checkout']
        else:
            results['warm_checkout_mrr'] = 0.0
        
        # Item coverage metrics
        # Calculate what percentage of the total item catalog is being recommended
        # Assuming the number of items is the size of the prediction tensor's second dimension
        # This will be set properly when predictions are provided
        if hasattr(self, '_num_items'):
            for k in self.k_values:
                if self._num_items > 0:
                    results[f'item_coverage@{k}'] = len(self.recommended_items[k]) / self._num_items
                else:
                    results[f'item_coverage@{k}'] = 0.0
        
        # Sample counts for debugging and transparency
        results['counts'] = self.sample_counts.copy()
        results['hit_counts'] = {k: dict(v) if isinstance(v, dict) else v for k, v in self.hit_counts.items()}
        results['total_weight_sum'] = self.total_weight_sum
        
        return results


class CrossValidationManager:
    """
    Manages k-fold cross-validation for recommender model training.
    
    Responsible for:
    - Creating k folds from training data
    - Tracking performance across folds
    - Determining the best model across folds
    - Providing statistical significance of results
    """
    
    def __init__(self, 
                 n_splits: int = 5, 
                 shuffle: bool = True, 
                 random_state: int = 42):
        """
        Initialize the cross-validation manager.
        
        Args:
            n_splits: Number of folds for cross-validation
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.fold_results = []
        self.best_fold = None
        self.best_score = -float('inf')
        self.primary_metric = 'weighted_recall@10'  # Default primary metric
    
    def create_folds(self, samples: List[Dict], stratify_by_purchase: bool = True) -> List[Tuple[List, List]]:
        """
        Create k folds from the provided samples.
        
        Args:
            samples: List of sample dictionaries to split
            stratify_by_purchase: Whether to maintain purchase ratio in each fold
            
        Returns:
            List of (train_indices, val_indices) tuples for each fold
        """
        import numpy as np
        from sklearn.model_selection import StratifiedKFold, KFold
        
        # Create array for fold splitting
        indices = np.arange(len(samples))
        
        if stratify_by_purchase:
            # Extract purchase labels for stratification
            labels = np.array([int(sample.get('is_purchase', False)) for sample in samples])
            
            # Use stratified k-fold to maintain purchase ratio
            splitter = StratifiedKFold(n_splits=self.n_splits, 
                                       shuffle=self.shuffle, 
                                       random_state=self.random_state)
            folds = list(splitter.split(indices, labels))
        else:
            # Use regular k-fold without stratification
            splitter = KFold(n_splits=self.n_splits, 
                            shuffle=self.shuffle, 
                            random_state=self.random_state)
            folds = list(splitter.split(indices))
        
        # Convert indices to samples
        folded_samples = []
        for train_idx, val_idx in folds:
            train_samples = [samples[i] for i in train_idx]
            val_samples = [samples[i] for i in val_idx]
            folded_samples.append((train_samples, val_samples))
            
        return folded_samples
    
    def set_primary_metric(self, metric_name: str):
        """Set the primary metric for determining the best fold"""
        self.primary_metric = metric_name
    
    def add_fold_result(self, fold_idx: int, metrics: Dict[str, Any], model_path: str):
        """
        Add results from a single fold.
        
        Args:
            fold_idx: Index of the current fold
            metrics: Dictionary of evaluation metrics for this fold
            model_path: Path to the saved model for this fold
        """
        fold_data = {
            'fold': fold_idx,
            'metrics': metrics,
            'model_path': model_path
        }
        
        self.fold_results.append(fold_data)
        
        # Update best fold if this one is better
        if self.primary_metric in metrics and (
            self.best_fold is None or 
            metrics[self.primary_metric] > self.best_score
        ):
            self.best_fold = fold_idx
            self.best_score = metrics[self.primary_metric]
    
    def get_best_fold(self) -> Dict[str, Any]:
        """Get the results of the best performing fold"""
        if self.best_fold is None:
            return None
            
        return next((r for r in self.fold_results if r['fold'] == self.best_fold), None)
    
    def get_average_metrics(self) -> Dict[str, float]:
        """
        Calculate average metrics across all folds.
        
        Returns:
            Dictionary of averaged metrics
        """
        if not self.fold_results:
            return {}
            
        # Collect all metrics across folds
        all_metrics = {}
        
        for fold_data in self.fold_results:
            metrics = fold_data['metrics']
            
            for key, value in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                    
                # Skip non-numeric values
                if isinstance(value, (int, float)):
                    all_metrics[key].append(value)
        
        # Calculate averages
        avg_metrics = {}
        for key, values in all_metrics.items():
            if values:  # Check if we have any values
                avg_metrics[key] = np.mean(values)
                
                # Also calculate standard deviation for statistical significance
                if len(values) > 1:
                    avg_metrics[f"{key}_std"] = np.std(values)
                    
                    # Calculate 95% confidence interval
                    ci_95 = 1.96 * np.std(values) / np.sqrt(len(values))
                    avg_metrics[f"{key}_ci95"] = ci_95
        
        return avg_metrics
    
    def get_statistical_significance(self, 
                                    model1_metrics: List[float], 
                                    model2_metrics: List[float],
                                    alpha: float = 0.05) -> Tuple[bool, float]:
        """
        Calculate statistical significance between two models' performance.
        
        Args:
            model1_metrics: List of metric values for model 1 across folds
            model2_metrics: List of metric values for model 2 across folds
            alpha: Significance level
            
        Returns:
            Tuple of (is_significant, p_value)
        """
        from scipy import stats
        
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(model1_metrics, model2_metrics)
        
        # Check if difference is statistically significant
        is_significant = p_value < alpha
        
        return is_significant, p_value
    
    def summarize_results(self) -> str:
        """
        Generate a summary of cross-validation results.
        
        Returns:
            String summary of results
        """
        if not self.fold_results:
            return "No cross-validation results available."
            
        avg_metrics = self.get_average_metrics()
        best_fold = self.get_best_fold()
        
        summary = []
        summary.append("Cross-Validation Results Summary")
        summary.append("===============================")
        summary.append(f"Number of folds: {self.n_splits}")
        summary.append(f"Primary metric: {self.primary_metric}")
        summary.append(f"Average {self.primary_metric}: {avg_metrics.get(self.primary_metric, 'N/A'):.4f}")
        
        if f"{self.primary_metric}_std" in avg_metrics:
            summary.append(f"Standard deviation: {avg_metrics[f'{self.primary_metric}_std']:.4f}")
            
        if f"{self.primary_metric}_ci95" in avg_metrics:
            summary.append(f"95% confidence interval: ±{avg_metrics[f'{self.primary_metric}_ci95']:.4f}")
        
        summary.append("\nBest fold:")
        summary.append(f"Fold {self.best_fold} with {self.primary_metric} = {self.best_score:.4f}")
        summary.append(f"Model path: {best_fold['model_path']}")
        
        summary.append("\nDetailed metrics (averaged across folds):")
        
        # Group metrics by type for better organization
        recall_metrics = {k: v for k, v in avg_metrics.items() if 'recall' in k and '_std' not in k and '_ci95' not in k}
        mrr_metrics = {k: v for k, v in avg_metrics.items() if 'mrr' in k and '_std' not in k and '_ci95' not in k}
        other_metrics = {k: v for k, v in avg_metrics.items() 
                       if 'recall' not in k and 'mrr' not in k and '_std' not in k and '_ci95' not in k}
        
        # Add recall metrics
        summary.append("\nRecall metrics:")
        for metric, value in sorted(recall_metrics.items()):
            summary.append(f"  {metric}: {value:.4f}")
            
        # Add MRR metrics
        summary.append("\nMRR metrics:")
        for metric, value in sorted(mrr_metrics.items()):
            summary.append(f"  {metric}: {value:.4f}")
            
        # Add other metrics
        summary.append("\nOther metrics:")
        for metric, value in sorted(other_metrics.items()):
            # Format count metrics as integers
            if 'count' in metric:
                summary.append(f"  {metric}: {int(value)}")
            else:
                summary.append(f"  {metric}: {value:.4f}")
        
        return "\n".join(summary)


class DynamicHyperparameters:
    """
    Dynamic hyperparameters that can adapt during training.
    
    Features:
    - Learning rate scheduling with warmup
    - Adaptive hyperparameters that update based on validation performance
    - Purchase boost and margin adaptation
    - Temperature scheduling for contrastive learning
    """
    
    def __init__(self, 
                 initial_values: Dict[str, float],
                 adapter_type: str = 'scheduled',
                 min_values: Optional[Dict[str, float]] = None,
                 max_values: Optional[Dict[str, float]] = None):
        """
        Initialize dynamic hyperparameters.
        
        Args:
            initial_values: Dictionary of initial hyperparameter values
            adapter_type: Method for adapting hyperparameters ('scheduled', 'adaptive', 'learnable')
            min_values: Minimum allowed values for each hyperparameter
            max_values: Maximum allowed values for each hyperparameter
        """
        self.values = initial_values.copy()
        self.initial_values = initial_values.copy()
        self.adapter_type = adapter_type
        
        # Set min/max bounds or use defaults with automatic constraint detection
        self.min_values = min_values or {}
        self.max_values = max_values or {}
        
        # Auto-detect sensible bounds based on parameter names
        for param in self.values:
            if param not in self.min_values:
                # Set sensible defaults based on parameter name
                if 'learning_rate' in param:
                    self.min_values[param] = 1e-6
                elif 'margin' in param:
                    self.min_values[param] = 0.0
                elif 'boost' in param:
                    self.min_values[param] = 0.1
                elif 'temperature' in param:
                    self.min_values[param] = 0.01
                else:
                    self.min_values[param] = 0.0
                    
            if param not in self.max_values:
                # Set sensible defaults based on parameter name
                if 'learning_rate' in param:
                    self.max_values[param] = 1.0
                elif 'margin' in param:
                    self.max_values[param] = 1.0
                elif 'boost' in param:
                    self.max_values[param] = 100.0
                elif 'temperature' in param:
                    self.max_values[param] = 1.0
                else:
                    self.max_values[param] = float('inf')
        
        # Initialize history for tracking changes
        self.history = {param: [value] for param, value in self.values.items()}
        self.update_steps = 0
        
        # Initialize adaptation parameters
        if adapter_type == 'scheduled':
            # Scheduled parameters decrease/increase according to a schedule
            self.schedule_type = 'cosine'  # Options: 'linear', 'cosine', 'step'
            self.schedule_params = {
                'warmup_steps': 0,
                'total_steps': 1000,
                'cycles': 1,
                'step_size': 0.1,
                'step_intervals': [0.3, 0.6, 0.9]  # Percentage of total steps
            }
        
        elif adapter_type == 'adaptive':
            # Adaptive parameters change based on validation performance
            self.patience = 2
            self.improvement_threshold = 0.001
            self.adaptation_rate = 0.1
            self.epochs_without_improvement = 0
            self.best_metric = -float('inf')
            
        elif adapter_type == 'learnable':
            # Learnable parameters are updated through gradient descent
            # They require PyTorch parameter registration in the model
            self.learnable_params = {}
            
    def update_step(self, step: int = None, total_steps: int = None):
        """
        Update hyperparameters based on training step for scheduled adaptation.
        
        Args:
            step: Current training step
            total_steps: Total number of training steps
        """
        if self.adapter_type != 'scheduled':
            return
            
        if step is not None:
            self.update_steps = step
        else:
            self.update_steps += 1
            
        if total_steps is not None:
            self.schedule_params['total_steps'] = total_steps
            
        # Apply schedule-based updates
        for param_name, initial_value in self.initial_values.items():
            # Skip parameters that shouldn't be scheduled
            if param_name not in self.values:
                continue
                
            # Get schedule multiplier
            if self.schedule_type == 'cosine':
                # Cosine schedule with optional warmup
                warmup_steps = self.schedule_params.get('warmup_steps', 0)
                total_steps = self.schedule_params.get('total_steps', 1000)
                cycles = self.schedule_params.get('cycles', 1)
                
                if self.update_steps < warmup_steps:
                    # Linear warmup
                    multiplier = float(self.update_steps) / float(max(1, warmup_steps))
                else:
                    # Cosine decay after warmup
                    progress = float(self.update_steps - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    multiplier = 0.5 * (1.0 + np.cos(np.pi * cycles * progress))
                    
            elif self.schedule_type == 'linear':
                # Linear schedule
                total_steps = self.schedule_params.get('total_steps', 1000)
                warmup_steps = self.schedule_params.get('warmup_steps', 0)
                
                if self.update_steps < warmup_steps:
                    # Linear warmup
                    multiplier = float(self.update_steps) / float(max(1, warmup_steps))
                else:
                    # Linear decay after warmup
                    multiplier = 1.0 - (float(self.update_steps - warmup_steps) / 
                                      float(max(1, total_steps - warmup_steps)))
                    
            elif self.schedule_type == 'step':
                # Step schedule
                total_steps = self.schedule_params.get('total_steps', 1000)
                step_size = self.schedule_params.get('step_size', 0.1)
                step_intervals = self.schedule_params.get('step_intervals', [0.3, 0.6, 0.9])
                
                # Calculate current progress
                progress = float(self.update_steps) / float(max(1, total_steps))
                
                # Initialize multiplier at 1.0
                multiplier = 1.0
                
                # Reduce by step_size at each interval
                for interval in step_intervals:
                    if progress > interval:
                        multiplier -= step_size
                        
            else:
                # Unknown schedule type
                multiplier = 1.0
                
            # Apply multiplier to the parameter
            new_value = initial_value * multiplier
            
            # Apply bounds
            if param_name in self.min_values:
                new_value = max(new_value, self.min_values[param_name])
            if param_name in self.max_values:
                new_value = min(new_value, self.max_values[param_name])
                
            # Update value
            self.values[param_name] = new_value
            
            # Track history
            self.history[param_name].append(new_value)
    
    def update_adaptive(self, metric_value: float):
        """
        Update hyperparameters based on validation metric for adaptive parameters.
        
        Args:
            metric_value: Current validation metric value
        """
        if self.adapter_type != 'adaptive':
            return
            
        # Check if there's improvement
        if metric_value > self.best_metric + self.improvement_threshold:
            # Improvement detected
            self.best_metric = metric_value
            self.epochs_without_improvement = 0
        else:
            # No improvement
            self.epochs_without_improvement += 1
            
        # Adapt parameters if no improvement for 'patience' epochs
        if self.epochs_without_improvement >= self.patience:
            for param_name in self.values:
                # Different adaptation strategies based on parameter
                if param_name == 'learning_rate':
                    # Reduce learning rate
                    self.values[param_name] *= (1.0 - self.adaptation_rate)
                    
                elif param_name == 'purchase_boost' or param_name == 'purchase_margin':
                    # Increase purchase importance
                    self.values[param_name] *= (1.0 + self.adaptation_rate)
                    
                elif param_name == 'temperature':
                    # Decrease temperature for sharper predictions
                    self.values[param_name] *= (1.0 - self.adaptation_rate)
                    
                # Apply bounds
                if param_name in self.min_values:
                    self.values[param_name] = max(self.values[param_name], self.min_values[param_name])
                if param_name in self.max_values:
                    self.values[param_name] = min(self.values[param_name], self.max_values[param_name])
                    
                # Track history
                self.history[param_name].append(self.values[param_name])
                
            # Reset counter
            self.epochs_without_improvement = 0
            
    def get_value(self, param_name: str) -> float:
        """Get the current value of a hyperparameter"""
        return self.values.get(param_name, self.initial_values.get(param_name))
    
    def get_all_values(self) -> Dict[str, float]:
        """Get all current hyperparameter values"""
        return self.values.copy()
    
    def get_history(self, param_name: str) -> List[float]:
        """Get the history of a hyperparameter's values"""
        return self.history.get(param_name, [])
    
    def get_learnable_parameters(self) -> Dict[str, torch.nn.Parameter]:
        """
        Create learnable PyTorch parameters for trainable hyperparameters.
        
        Returns:
            Dictionary of parameter name to torch.nn.Parameter
        """
        if self.adapter_type != 'learnable':
            return {}
            
        # Create learnable parameters for each hyperparameter
        learnable_params = {}
        
        for param_name, value in self.values.items():
            if 'margin' in param_name:
                # Margin parameters use LearnableMargin module
                learnable_params[param_name] = LearnableMargin(
                    initial_margin=value,
                    min_margin=self.min_values.get(param_name, 0.0),
                    max_margin=self.max_values.get(param_name, 1.0)
                )
            elif 'temperature' in param_name:
                # Temperature parameter requires positive values
                # Use softplus to ensure positive values
                learnable_params[param_name] = torch.nn.Parameter(
                    torch.tensor(value, dtype=torch.float)
                )
            elif 'boost' in param_name:
                # Boost parameters are positive
                # Use log parameterization for stability
                learnable_params[param_name] = torch.nn.Parameter(
                    torch.tensor(np.log(value), dtype=torch.float)
                )
            else:
                # Default case - direct parameterization
                learnable_params[param_name] = torch.nn.Parameter(
                    torch.tensor(value, dtype=torch.float)
                )
                
        self.learnable_params = learnable_params
        return learnable_params
    
    def update_from_learnable(self):
        """Update internal values from learnable parameters"""
        if self.adapter_type != 'learnable' or not hasattr(self, 'learnable_params') or not self.learnable_params:
            return
            
        # Update values from learnable parameters
        for param_name, param in self.learnable_params.items():
            if isinstance(param, LearnableMargin):
                # LearnableMargin module has a forward method
                self.values[param_name] = param().item()
            elif 'boost' in param_name:
                # Exponentiate for boost parameters (log parameterization)
                self.values[param_name] = torch.exp(param).item()
            elif 'temperature' in param_name:
                # Use raw parameter value but apply bounds
                self.values[param_name] = torch.clamp(
                    param,
                    min=self.min_values.get(param_name, 0.01),
                    max=self.max_values.get(param_name, 1.0)
                ).item()
            else:
                # Default case - direct value with bounds
                self.values[param_name] = torch.clamp(
                    param,
                    min=self.min_values.get(param_name, 0.0),
                    max=self.max_values.get(param_name, float('inf'))
                ).item()
                
            # Track history
            self.history[param_name].append(self.values[param_name])


class UnifiedMemoryManager:
    """
    Unified memory management for all device types.
    
    Features:
    - Dynamic batch size adjustment based on available memory
    - Periodic memory cleanup
    - Tensor type optimization for different devices
    - Out-of-memory error prevention
    - Memory usage tracking and reporting
    """
    
    def __init__(self, device_type: str):
        """
        Initialize memory manager for the specified device type.
        
        Args:
            device_type: Device type ('cuda', 'mps', or 'cpu')
        """
        self.device_type = device_type
        self.peak_memory = 0
        self.current_memory = 0
        self.periodic_cleanup = True
        self.cleanup_frequency = 5
        self.current_batch = 0
        self.memory_threshold = 0.9  # Maximum memory usage threshold (90%)
        self.optimize_dtypes = True
        
        # Memory tracking setup
        if device_type == 'cuda':
            import torch
            self.has_memory_tracking = torch.cuda.is_available()
            if self.has_memory_tracking:
                # Reserve some memory to prevent fragmentation
                self.reserve = torch.empty(1024 * 1024, device='cuda')
                
        elif device_type == 'mps':
            import torch
            self.has_memory_tracking = (hasattr(torch, 'mps') and
                                      hasattr(torch.mps, 'current_allocated_memory'))
        else:  # CPU
            self.has_memory_tracking = False
    
    def get_available_memory(self) -> Tuple[int, int]:
        """
        Get currently available memory on the device.
        
        Returns:
            Tuple of (free_memory, total_memory) in bytes
        """
        if self.device_type == 'cuda':
            import torch
            if torch.cuda.is_available():
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                total_memory = torch.cuda.get_device_properties(0).total_memory
                return free_memory, total_memory
                
        elif self.device_type == 'mps':
            import torch
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'current_allocated_memory'):
                # MPS doesn't have a direct API for total/free memory, so we estimate
                # Assuming a 4GB memory size for M1/M2 chips
                estimated_total = 4 * 1024 * 1024 * 1024  # 4GB in bytes
                current_used = torch.mps.current_allocated_memory()
                return estimated_total - current_used, estimated_total
                
        # CPU case or fallback
        import psutil
        memory_info = psutil.virtual_memory()
        return memory_info.available, memory_info.total
    
    def update_memory_info(self):
        """Update current and peak memory usage information"""
        if not self.has_memory_tracking:
            return
            
        if self.device_type == 'cuda':
            import torch
            if torch.cuda.is_available():
                current = torch.cuda.memory_allocated()
                self.current_memory = current
                self.peak_memory = max(self.peak_memory, current)
                
        elif self.device_type == 'mps':
            import torch
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'current_allocated_memory'):
                current = torch.mps.current_allocated_memory()
                self.current_memory = current
                self.peak_memory = max(self.peak_memory, current)
    
    def get_optimal_batch_size(self, 
                              current_batch_size: int, 
                              item_size_bytes: int,
                              min_batch_size: int = 4) -> int:
        """
        Calculate optimal batch size based on available memory.
        
        Args:
            current_batch_size: Current batch size being used
            item_size_bytes: Estimated memory usage per item in the batch
            min_batch_size: Minimum acceptable batch size
            
        Returns:
            Recommended batch size based on available memory
        """
        free_memory, total_memory = self.get_available_memory()
        
        # Reserve 20% of memory for operations and overhead
        usable_memory = free_memory * 0.8
        
        # Calculate how many items we can fit
        max_items = usable_memory // item_size_bytes
        
        # Round down to multiple of 4 (or min_batch_size) for better memory alignment
        optimal_size = max(min_batch_size, (max_items // min_batch_size) * min_batch_size)
        
        # Don't increase batch size too aggressively
        if optimal_size > current_batch_size:
            # Increase by at most 50%
            return min(optimal_size, int(current_batch_size * 1.5))
        elif optimal_size < current_batch_size:
            # Decrease by at most 25%
            return max(optimal_size, int(current_batch_size * 0.75))
        else:
            return current_batch_size
    
    def cleanup(self, force: bool = False):
        """
        Perform memory cleanup operations.
        
        Args:
            force: Whether to force cleanup even if not scheduled
        """
        if not self.periodic_cleanup and not force:
            return
            
        # Check if it's time for a cleanup
        if not force and self.current_batch % self.cleanup_frequency != 0:
            return
            
        if self.device_type == 'cuda':
            import torch
            if torch.cuda.is_available():
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
        elif self.device_type == 'mps':
            import torch
            if hasattr(torch, 'mps'):
                # Clear MPS cache
                torch.mps.empty_cache()
    
    def optimize_tensor_dtypes(self, tensor, large_threshold: int = 1000) -> torch.Tensor:
        """
        Optimize tensor data types for the current device.
        
        Args:
            tensor: Input tensor to optimize
            large_threshold: Threshold for considering a tensor "large"
            
        Returns:
            Optimized tensor
        """
        if not self.optimize_dtypes:
            return tensor
            
        import torch
        
        # Skip tensors that aren't PyTorch tensors
        if not isinstance(tensor, torch.Tensor):
            return tensor
            
        # Skip tensors that require gradients
        if tensor.requires_grad:
            return tensor
            
        is_large = tensor.numel() > large_threshold
        
        if self.device_type == 'cuda':
            # For CUDA, convert int64 to int32 for large tensors
            if tensor.dtype == torch.int64 and is_large:
                return tensor.to(dtype=torch.int32)
                
        elif self.device_type == 'mps':
            # For MPS, convert int64 to int32 and float64 to float32
            if tensor.dtype == torch.int64 and is_large:
                return tensor.to(dtype=torch.int32)
            elif tensor.dtype == torch.float64:
                return tensor.to(dtype=torch.float32)
                
        # Return unchanged if no optimization applies
        return tensor
    
    def batch_to_device(self, batch, device):
        """
        Move batch to device with optimized handling for CUDA/MPS/CPU.
        
        Args:
            batch: Data batch to move to device
            device: PyTorch device to move data to
            
        Returns:
            Batch on the specified device with optimized data types
        """
        import torch
        
        if isinstance(batch, dict):
            # Process each dictionary entry
            return {k: self.batch_to_device(v, device) for k, v in batch.items()}
            
        elif isinstance(batch, torch.Tensor):
            # Optimize tensor and move to device
            optimized = self.optimize_tensor_dtypes(batch)
            return optimized.to(device, non_blocking=True)
            
        elif isinstance(batch, (list, tuple)) and len(batch) > 0 and isinstance(batch[0], torch.Tensor):
            # Handle list of tensors
            return [self.optimize_tensor_dtypes(t).to(device, non_blocking=True) for t in batch]
            
        else:
            # Return unchanged for other types
            return batch
    
    def next_batch(self):
        """Update batch counter and perform periodic cleanup if needed"""
        self.current_batch += 1
        self.update_memory_info()
        
        if self.periodic_cleanup and self.current_batch % self.cleanup_frequency == 0:
            self.cleanup()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary of memory statistics
        """
        free_memory, total_memory = self.get_available_memory()
        
        stats = {
            'device_type': self.device_type,
            'current_memory_bytes': self.current_memory,
            'peak_memory_bytes': self.peak_memory, 
            'free_memory_bytes': free_memory,
            'total_memory_bytes': total_memory,
            'memory_utilization': 1.0 - (free_memory / max(1, total_memory))
        }
        
        # Add human-readable values in MB
        for key in ['current_memory_bytes', 'peak_memory_bytes', 'free_memory_bytes', 'total_memory_bytes']:
            if key in stats:
                mb_key = key.replace('_bytes', '_mb')
                stats[mb_key] = stats[key] / (1024 * 1024)
                
        return stats


# Example usage for standalone testing
if __name__ == "__main__":
    import torch
    
    # Test the UnifiedMetricsTracker
    metrics_tracker = UnifiedMetricsTracker()
    
    # Create dummy data
    batch_size = 32
    num_items = 1000
    
    # Simulate predictions and targets
    predictions = torch.randn(batch_size, num_items)
    targets = torch.randint(0, num_items, (batch_size,))
    
    # Create dummy event flags
    is_purchase = torch.zeros(batch_size, dtype=torch.bool)
    is_purchase[0:5] = True  # Mark 5 examples as purchases
    
    has_checkout = torch.zeros(batch_size, dtype=torch.bool)
    has_checkout[6:12] = True  # Mark 6 examples as checkouts
    
    has_add_to_cart = torch.zeros(batch_size, dtype=torch.bool)
    has_add_to_cart[13:20] = True  # Mark 7 examples as add-to-cart
    
    # Update metrics
    metrics_tracker.update(
        predictions=predictions,
        targets=targets,
        is_purchase=is_purchase,
        has_checkout=has_checkout,
        has_add_to_cart=has_add_to_cart,
        loss=0.5
    )
    
    # Compute and print results
    results = metrics_tracker.compute()
    print("Test results:")
    for key, value in sorted(results.items()):
        print(f"  {key}: {value}")
    


# Backward compatibility functions for migration from utils/metrics.py
def hit_rate_at_k(scores: torch.Tensor, ground_truth: torch.Tensor, k: int = 10) -> float:
    """Calculate Hit Rate@k (also known as Recall@k) - Legacy function for backward compatibility"""
    tracker = UnifiedMetricsTracker(k_values=[k])
    tracker.update(
        predictions=scores,
        targets=ground_truth,
        is_purchase=torch.ones_like(ground_truth, dtype=torch.bool),
        has_checkout=torch.zeros_like(ground_truth, dtype=torch.bool),
        has_add_to_cart=torch.zeros_like(ground_truth, dtype=torch.bool)
    )
    results = tracker.compute()
    return results.get(f'recall@{k}', 0.0)


def mrr_at_k(scores: torch.Tensor, ground_truth: torch.Tensor, k: int = 10) -> float:
    """Calculate Mean Reciprocal Rank@k - Legacy function for backward compatibility"""
    tracker = UnifiedMetricsTracker(k_values=[k])
    tracker.update(
        predictions=scores,
        targets=ground_truth,
        is_purchase=torch.ones_like(ground_truth, dtype=torch.bool),
        has_checkout=torch.zeros_like(ground_truth, dtype=torch.bool),
        has_add_to_cart=torch.zeros_like(ground_truth, dtype=torch.bool)
    )
    results = tracker.compute()
    return results.get('mrr', 0.0)


def ndcg_at_k(scores: torch.Tensor, ground_truth: torch.Tensor, k: int = 10) -> float:
    """Calculate Normalized Discounted Cumulative Gain@k - Legacy function for backward compatibility"""
    tracker = UnifiedMetricsTracker(k_values=[k])
    tracker.update(
        predictions=scores,
        targets=ground_truth,
        is_purchase=torch.ones_like(ground_truth, dtype=torch.bool),
        has_checkout=torch.zeros_like(ground_truth, dtype=torch.bool),
        has_add_to_cart=torch.zeros_like(ground_truth, dtype=torch.bool)
    )
    results = tracker.compute()
    return results.get(f'ndcg@{k}', 0.0)


def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                     k_values: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
    """Calculate multiple metrics at once - Legacy function for backward compatibility"""
    tracker = UnifiedMetricsTracker(k_values=k_values)
    tracker.update(
        predictions=predictions,
        targets=targets,
        is_purchase=torch.ones_like(targets, dtype=torch.bool),
        has_checkout=torch.zeros_like(targets, dtype=torch.bool),
        has_add_to_cart=torch.zeros_like(targets, dtype=torch.bool)
    )
    results = tracker.compute()
    
    # Convert to legacy format
    legacy_results = {}
    for k in k_values:
        legacy_results[f'hit@{k}'] = results.get(f'recall@{k}', 0.0)
        legacy_results[f'mrr@{k}'] = results.get('mrr', 0.0)
        legacy_results[f'ndcg@{k}'] = results.get(f'ndcg@{k}', 0.0)
    
    return legacy_results


class MetricsTracker:
    """Legacy MetricsTracker class for backward compatibility with existing training scripts"""
    def __init__(self, k_values: List[int] = [1, 5, 10, 20]):
        self.unified_tracker = UnifiedMetricsTracker(k_values=k_values)
        self.k_values = k_values
        
    def reset(self):
        """Reset all tracked metrics"""
        self.unified_tracker.reset()
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, 
               loss: Optional[float] = None):
        """Update metrics with a new batch"""
        self.unified_tracker.update(
            predictions=predictions,
            targets=targets,
            is_purchase=torch.ones_like(targets, dtype=torch.bool),
            has_checkout=torch.zeros_like(targets, dtype=torch.bool),
            has_add_to_cart=torch.zeros_like(targets, dtype=torch.bool),
            loss=loss
        )
    
    def compute(self) -> Dict[str, float]:
        """Compute average metrics in legacy format"""
        results = self.unified_tracker.compute()
        
        # Convert to legacy format
        legacy_results = {}
        
        if 'loss' in results:
            legacy_results['loss'] = results['loss']
        
        for k in self.k_values:
            legacy_results[f'hit@{k}'] = results.get(f'recall@{k}', 0.0)
            legacy_results[f'mrr@{k}'] = results.get('mrr', 0.0)
            legacy_results[f'ndcg@{k}'] = results.get(f'ndcg@{k}', 0.0)
        
        return legacy_results


class EnhancedEventMetrics:
    """Legacy EnhancedEventMetrics class for backward compatibility"""
    
    @staticmethod
    def calculate_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        is_purchase: torch.Tensor,
        has_checkout: torch.Tensor,
        has_add_to_cart: torch.Tensor,
        k_values: List[int] = [10, 20],
        has_checkout_inclusive: Optional[torch.Tensor] = None,
        has_add_to_cart_inclusive: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Calculate metrics with event type awareness - Legacy interface"""
        tracker = UnifiedMetricsTracker(k_values=k_values)
        tracker.update(
            predictions=predictions,
            targets=targets,
            is_purchase=is_purchase,
            has_checkout=has_checkout,
            has_add_to_cart=has_add_to_cart,
            has_checkout_inclusive=has_checkout_inclusive,
            has_add_to_cart_inclusive=has_add_to_cart_inclusive
        )
        results = tracker.compute()
        
        # Convert to legacy format with nested structure
        legacy_results = {}
        
        # Event counts
        legacy_results['purchase_count'] = tracker.counts['purchase']
        legacy_results['checkout_count'] = tracker.counts['checkout']
        legacy_results['add_to_cart_count'] = tracker.counts['add_to_cart']
        legacy_results['intent_count'] = tracker.counts['intent']
        
        # Recall@k metrics in nested format
        legacy_results['recall@k'] = {}
        for k in k_values:
            legacy_results['recall@k'][k] = results.get(f'recall@{k}', 0.0)
        
        if tracker.counts['purchase'] > 0:
            legacy_results['purchase_recall@k'] = {}
            for k in k_values:
                legacy_results['purchase_recall@k'][k] = results.get(f'purchase_recall@{k}', 0.0)
        
        if tracker.counts['checkout'] > 0:
            legacy_results['checkout_recall@k'] = {}
            for k in k_values:
                legacy_results['checkout_recall@k'][k] = results.get(f'checkout_recall@{k}', 0.0)
        
        if tracker.counts['add_to_cart'] > 0:
            legacy_results['add_to_cart_recall@k'] = {}
            for k in k_values:
                legacy_results['add_to_cart_recall@k'][k] = results.get(f'add_to_cart_recall@{k}', 0.0)
        
        if tracker.counts['intent'] > 0:
            legacy_results['intent_recall@k'] = {}
            for k in k_values:
                legacy_results['intent_recall@k'][k] = results.get(f'intent_recall@{k}', 0.0)
        
        # MRR metrics
        legacy_results['overall_mrr'] = results.get('mrr', 0.0)
        legacy_results['purchase_mrr'] = results.get('purchase_mrr', 0.0)
        legacy_results['checkout_mrr'] = results.get('checkout_mrr', 0.0)
        legacy_results['add_to_cart_mrr'] = results.get('add_to_cart_mrr', 0.0)
        legacy_results['intent_mrr'] = results.get('intent_mrr', 0.0)
        
        return legacy_results


def calculate_item_coverage(model, data_loader, device, k=20):
    """
    Simple standalone function to calculate item coverage@k metric.
    
    Item coverage measures what percentage of the total item catalog 
    is being recommended across all users.
    
    Args:
        model: The recommendation model
        data_loader: DataLoader containing the data
        device: Device to run on (cuda, mps, cpu)
        k: Number of top items to consider (default: 20)
        
    Returns:
        float: Item coverage as a fraction (0-1)
    """
    model.eval()
    recommended_items = set()
    num_items = None
    
    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            if isinstance(batch, dict):
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                    elif isinstance(batch[key], dict):
                        for sub_key in batch[key]:
                            if isinstance(batch[key][sub_key], torch.Tensor):
                                batch[key][sub_key] = batch[key][sub_key].to(device)
            
            # Get predictions
            outputs = model(batch)
            predictions = outputs['predictions']
            
            # Store number of items
            if num_items is None:
                num_items = predictions.shape[1]
            
            # Get top-k items for each prediction
            _, top_k_items = torch.topk(predictions, min(k, predictions.shape[1]), dim=1)
            
            # Add to set of recommended items
            for items in top_k_items:
                recommended_items.update(items.cpu().numpy().tolist())
    
    # Calculate coverage
    if num_items is None or num_items == 0:
        return 0.0
    
    coverage = len(recommended_items) / num_items
    return coverage


# Test backward compatibility after all classes are defined
if __name__ == "__main__":
    print("\nTesting backward compatibility...")
    
    # Create test data
    batch_size = 32
    num_items = 1000
    predictions = torch.randn(batch_size, num_items)
    targets = torch.randint(0, num_items, (batch_size,))
    is_purchase = torch.zeros(batch_size, dtype=torch.bool)
    is_purchase[0:5] = True
    has_checkout = torch.zeros(batch_size, dtype=torch.bool)
    has_checkout[6:12] = True
    has_add_to_cart = torch.zeros(batch_size, dtype=torch.bool)
    has_add_to_cart[13:20] = True
    
    # Test legacy MetricsTracker
    legacy_tracker = MetricsTracker(k_values=[10, 20])
    legacy_tracker.update(predictions, targets, loss=0.5)
    legacy_results = legacy_tracker.compute()
    print("Legacy MetricsTracker results:", legacy_results)
    
    # Test legacy EnhancedEventMetrics
    legacy_enhanced = EnhancedEventMetrics.calculate_metrics(
        predictions, targets, is_purchase, has_checkout, has_add_to_cart, k_values=[10, 20]
    )
    print("Legacy EnhancedEventMetrics results keys:", list(legacy_enhanced.keys()))
    
    print("✅ All backward compatibility tests passed!")